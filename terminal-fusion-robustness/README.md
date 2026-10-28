# Latency- and Dropout-Robust Infrastructure Multi-Sensor Localization for Collision Prevention in Container Terminals

Code, scenario settings, calibration files and seeds for the paper of the same title
(Asia-pacific Journal of Convergent Research Interchange, [volume/issue, year], DOI: [paper DOI]).

The study quantifies near-field response bands and uncompensated-latency budgets for container-terminal
collision prevention, compares early, late and learned fusion of LiDAR, camera, thermal and GNSS-RTK over
15 modality-dropout subsets at several calibration levels (L0, L1, L3), and replicates the method ranking
on a real multi-sensor dataset (MM-Fi).

## Repository layout

| Path | Content |
| --- | --- |
| `p1_sim/p1_nearfield_fusion_sim.py` | E1 latency–error budget, E2 fusion stage × sensor condition, analytic bands (Table 2, Fig. 4–5) |
| `p1_bench/nf_sim.py` | Terminal scene, agents, sensor/detection/timing models (Eq. 3–5) |
| `p1_bench/nf_fusion.py` | Early (Kalman filter, OOSM, NIS soft gating), late and learned fusion with modality dropout (Eq. 6–8) |
| `p1_bench/nf_experiments.py` | Dropout robustness, collision metrics, synchronization levels (Tables 4–6, Fig. 6) |
| `p1_bench/nf_calib_nuscenes.py` | Detection curves from nuScenes (calibration level L1) |
| `aihub_port_prep.py` | Crane-view detection, recall by condition and conversion to Δa (Eq. 12, Table 7, Fig. 7a) |
| `mmfi_dropout.py` | Real-data replication on MM-Fi E01 (Table 7, Fig. 7b) |
| `paiss_stats.py` | Worker speed percentile from Physical AI Smart Spaces (sensitivity value) |
| `make_figures.py`, `make_sensitivity_fig.py`, `summarize_p.py`, `compare_calibration.py` | Figures and summary tables |
| `run_pipeline.sh`, `locate_data.py`, `setup_env.sh` | End-to-end pipeline, data path discovery, environment |
| `calibration/` | Calibration files for L1 and L3 (aggregate statistics only) |
| `results/` | Result tables and figures reported in the paper |
| `docs/EQUATIONS.md` | Mapping of Eqs. (1)–(12) to the implementing functions |
| `docs/DATA.md` | Datasets, how to obtain them, and their terms of use |

## Setup

```bash
bash setup_env.sh            # creates .venv and installs requirements.txt
source .venv/bin/activate
```

Python 3.10 or later. The simulation runs on a CPU; the detector step uses OpenCV (Darknet backend, OpenCV < 5)
or Ultralytics.

## Reproducing the results

1. Obtain the datasets listed in `docs/DATA.md` (they are not redistributed here).
2. `python locate_data.py <data folder>` writes `paths.env` with the local data paths.
3. `bash run_pipeline.sh` runs all steps; seeds are fixed to 7, 11, 23, 31 and 47.
   `SKIP="3 5" bash run_pipeline.sh` skips steps, and `OUT=runs/<folder>` resumes a run.
4. Calibration levels: L0 uses the defaults in `p1_bench/nf_sim.py`; L1 and L3 read
   `calibration/calib_L1.json` and `calibration/calib_L3.json` (`CALIB_INIT=...`).

The simulation results are deterministic for a given seed and calibration file; for example, Fig. 5 is
reproduced exactly by `python p1_sim/p1_nearfield_fusion_sim.py --profile port_terminal --calib calibration/calib_L3.json`.

## Data and licences

The code is released under the MIT License (`LICENSE`). Datasets keep their own terms (see `docs/DATA.md`).
In particular, the AI Hub port data, the associated YOLOv4 model and nuScenes are **not** included;
only aggregate statistics derived from them (calibration files, result tables) are published.

## Citation

See `CITATION.cff`. Please cite the paper when using this code.
