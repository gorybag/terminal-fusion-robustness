#!/usr/bin/env bash
# run_pipeline.sh - apply the downloaded public data to the paper-1 experiments (container-terminal profile)
# 1) python locate_data.py <data folder>   (writes paths.env)   2) source .venv/bin/activate   3) bash run_pipeline.sh
# Paths come from paths.env when present; otherwise edit the defaults below. Skip steps with SKIP="3 5" bash run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ -f paths.env ]; then                                 # values given on the command line take precedence
  while IFS= read -r line; do
    case "$line" in export\ *=*) k=${line#export }; k=${k%%=*}; if [ -z "${!k:-}" ]; then eval "$line"; fi;; esac
  done < paths.env
  echo "paths from paths.env (command-line values kept)"
fi

# ---- paths (edit) -------------------------------------------------------------------------------------------
PORT_VL="${PORT_VL:-$HOME/data/AIhub_Validation_data/VL.zip}"       # AI Hub port labels (zip or unzipped folder)
PORT_VS_YT="${PORT_VS_YT:-$HOME/data/AIhub_Validation_data/VS_YT}"      # YT images:  unzip VS.zip "*SAF_CRANE_YT*" -d VS_YT
YOLO_CFG="${YOLO_CFG:-$HOME/data/aihub_model/yolov4.cfg}"             # AI Hub model (Darknet); missing files -> detection skipped
YOLO_WEIGHTS="${YOLO_WEIGHTS:-$HOME/data/aihub_model/yolov4.weights}"
DETECTOR="${DETECTOR:-opencv}"                           # opencv (AI Hub YOLOv4 cfg/weights) | ultralytics (e.g. yolo11x.pt)
KEEP_CLASSES="${KEEP_CLASSES:-}"                         # detector classes to keep, e.g. COCO "2,5,7" (car,bus,truck); empty = all
MATCH="${MATCH:-}"                                       # eval matching: cover (default, generic detectors) | iou (site-trained)
DEVICE="${DEVICE:-}"                                     # ultralytics device: mps | cpu (empty = auto)
YT_CLASS="${YT_CLASS:-}"                                              # YT class id(s) printed by step 2 (e.g. "7"); empty = all objects
NUSC_ROOT="${NUSC_ROOT:-$HOME/data/nuscenes}"                          # folder that CONTAINS v1.0-mini/
ADRIVE_DIR="${ADRIVE_DIR:-$HOME/data/AIhub_adrive_lableDB}"             # folder with the 11 VL zip files
MMFI_E01="${MMFI_E01:-$HOME/data/MMFi/E01}"                           # contains S01..S10
PAISS_ROOT="${PAISS_ROOT:-$HOME/data/MTMC_Tracking_2026}"               # contains train/ val/
SEEDS="7 11 23 31 47"
OUT="${OUT:-runs/$(date +%Y%m%d_%H%M)}"                  # set OUT=runs/<folder> to resume an earlier run
SKIP="${SKIP:-}"
# --------------------------------------------------------------------------------------------------------------
mkdir -p "$OUT"
run() { case " $SKIP " in *" $1 "*) echo "---- skip step $1";; *) echo; echo "==== step $1: $2"; return 0;; esac; return 1; }

env PORT_VL="$PORT_VL" PORT_VS_YT="$PORT_VS_YT" NUSC_ROOT="$NUSC_ROOT" ADRIVE_DIR="$ADRIVE_DIR" MMFI_E01="$MMFI_E01" \
    PAISS_ROOT="$PAISS_ROOT" YOLO_WEIGHTS="$YOLO_WEIGHTS" DETECTOR="$DETECTOR" python - <<PY
import os
for k in ["PORT_VL","PORT_VS_YT","NUSC_ROOT","ADRIVE_DIR","MMFI_E01","PAISS_ROOT","YOLO_WEIGHTS"]:
    v = os.environ.get(k) or ""
    ok = v and os.path.exists(v)
    note = "auto-download" if k == "YOLO_WEIGHTS" and not ok and os.environ.get("DETECTOR") == "ultralytics" else ("OK " if ok else "MISSING")
    print(f"  {k:13s} {note} {v}")
PY
if run 1 "nuScenes-mini calibration (LiDAR/camera daytime curves, asynchrony)"; then
  test -d "$NUSC_ROOT/v1.0-mini" || { echo "v1.0-mini not found under $NUSC_ROOT"; exit 1; }
  python p1_bench/nf_calib_nuscenes.py --dataroot "$NUSC_ROOT" --version v1.0-mini --out "$OUT/calib_nusc.json"
fi
CALIB="${CALIB_INIT:-}"                                   # optional existing calibration file
if [ -z "$CALIB" ]; then                                  # resume: take the most complete calibration already in OUT
  for f in adrive3d_lidar.json calib_port.json calib_nusc.json; do
    if [ -f "$OUT/$f" ]; then CALIB="$OUT/$f"; break; fi
  done
fi
echo "calibration so far: ${CALIB:-none}"

if run 2 "AI Hub port images: labels -> detection -> condition recall -> camera calibration"; then
  python aihub_port_prep.py stats --labels "$PORT_VL" --out "$OUT/port_labels.csv"
  if [ "$DETECTOR" = "ultralytics" ] || [ -f "$YOLO_WEIGHTS" ]; then
    if [ "$DETECTOR" = "ultralytics" ]; then
      python aihub_port_prep.py detect --labels "$OUT/port_labels.csv" --images "$PORT_VS_YT" --backend ultralytics \
             --weights "$YOLO_WEIGHTS" ${KEEP_CLASSES:+--keep-classes "$KEEP_CLASSES"} ${DEVICE:+--device "$DEVICE"} --out "$OUT/port_dets.csv"
    else
      python aihub_port_prep.py detect --labels "$OUT/port_labels.csv" --images "$PORT_VS_YT" --backend opencv \
             --cfg "$YOLO_CFG" --weights "$YOLO_WEIGHTS" ${KEEP_CLASSES:+--keep-classes "$KEEP_CLASSES"} --out "$OUT/port_dets.csv"
    fi
    echo "detector: $DETECTOR $YOLO_WEIGHTS classes=${KEEP_CLASSES:-all} device=${DEVICE:-auto} yt_class=${YT_CLASS:-all}" > "$OUT/detector.txt"
    python aihub_port_prep.py eval --labels "$OUT/port_labels.csv" --dets "$OUT/port_dets.csv" \
           ${YT_CLASS:+--gt-class "$YT_CLASS"} ${MATCH:+--match "$MATCH"} --out "$OUT/port_eval.json"
    python aihub_port_prep.py convert --eval "$OUT/port_eval.json" ${CALIB:+--merge "$CALIB"} --out "$OUT/calib_port.json"
    CALIB="$OUT/calib_port.json"
  else
    echo "   detector weights not found -> detection skipped (camera keeps nuScenes/assumed values)"
  fi
fi

if run 3 "AI Hub 3D labels: LiDAR point density by time/weather (cross-check, optional)"; then
  python adrive3d_prep.py --zips "$ADRIVE_DIR" --k 10 ${CALIB:+--merge "$CALIB"} --out-prefix "$OUT/adrive3d"
  CALIB="$OUT/adrive3d_lidar.json"
fi

if run 4 "PAISS 2026: kinematics, person-vehicle proximity, visibility (report only)"; then
  python paiss_stats.py --root "$PAISS_ROOT" --splits train val --stride 3 --out-prefix "$OUT/paiss"
fi

if run 5 "MM-Fi E01: real-data missing-modality check (supplement)"; then
  python mmfi_dropout.py --root "$MMFI_E01" --train-subjects S01-S07 --test-subjects S08-S10 --stride 2 --out-prefix "$OUT/mmfi"
fi

if run 6 "controlled experiments E1-E3 (port profile, calibrated probabilities)"; then
  python p1_sim/p1_nearfield_fusion_sim.py --profile port_terminal ${CALIB:+--calib "$CALIB"} --out "$OUT/sim_port"
fi

if run 7 "scene experiments A, C and P (collision-oriented), seeds: $SEEDS"; then
  for s in $SEEDS; do
    python p1_bench/nf_experiments.py --profile port_terminal ${CALIB:+--calib "$CALIB"} --seed "$s" --parts ACP --p-nodes N1,N2 --out "$OUT/bench_port_s$s"
  done
fi

if run 8 "comparison with v1 and journal figures"; then
  python compare_calibration.py --base p1_bench/results_bench --calib "$OUT/bench_port_s7" --out "$OUT/comparison_v1_vs_port.md"
  python make_figures.py --kit . --bench-dir "$OUT/bench_port_s7" --sim-dir "$OUT/sim_port" --profile port_terminal --calib "$CALIB" --out "$OUT/figures"
  python summarize_p.py --level "this run=$OUT/bench_port_s*" --out "$OUT/p_summary"
fi
echo; echo "done -> $OUT   (calibration used: $CALIB)"
