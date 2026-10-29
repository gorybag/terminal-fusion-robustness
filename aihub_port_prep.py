#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aihub_port_prep.py - AI Hub "커넥티드 항만을 위한 물류 인프라 데이터" (dataSetSn=523) -> camera model calibration

Only the Safety / Crane / YT part is needed (yard tractors seen from quay-crane cameras).
  stats    parse label JSONs (folder or .zip, TL/VL) -> one row per object + summary by weather x time band
  detect   run a detector on the images listed in stats -> detections CSV
           backends: opencv  (Darknet cfg/weights, e.g. the YOLOv4 model offered on the dataset page)
                     ultralytics (any YOLO .pt; COCO 'truck' etc. for a zero-shot baseline)
  eval     match detections to labels (IoU >= 0.5) -> recall by condition and by box height + logistic fits
  convert  turn the fits into simulator calibration: daytime camera curve p(r) and condition offsets da,
           written as {"camera_port": ...}; can be merged into a nuScenes calibration file

Label fields used (dataset manual): Images.filename, addinfo.weather (0 clear, 1 rain, 2 cloudy, 3 thunder),
addinfo.daynight (0 05-08h, 1 08-17h, 2 listed as 07-20h, likely 17-20h, 3 20-05h), addinfo.area (A BPT, B HJNC,
C BNCT, D SNCT), addinfo.location (QC/Gate), addinfo.camid, addinfo.time, addinfo.dataregion (0 Container, 1 Safety),
annotations[].bbox.classid, annotations[].bbox.points (4 points, clockwise from top-left).

Typical run (after merging the split archives and unzipping VL.zip / VS):
  python aihub_port_prep.py stats   --labels <VL folder or VL.zip> --out port_labels.csv
  python aihub_port_prep.py detect  --labels port_labels.csv --images <VS folder> --backend opencv \
         --cfg yolov4.cfg --weights yolov4.weights --out port_dets.csv
  python aihub_port_prep.py eval    --labels port_labels.csv --dets port_dets.csv --gt-class <YT class id> --out port_eval.json
  python aihub_port_prep.py convert --eval port_eval.json --merge nuscenes_calib.json --out calib_port.json
  python p1_bench/nf_experiments.py --profile port_terminal --calib calib_port.json --out p1_bench/results_port
"""
import argparse
import json
import math
import os
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

WEATHER = {"0": "clear", "1": "rain", "2": "cloudy", "3": "thunder"}
DAYNIGHT = {"0": "dawn", "1": "day", "2": "evening", "3": "night"}
AREA = {"A": "BPT", "B": "HJNC", "C": "BNCT", "D": "SNCT"}
# simulator condition <- port-image condition (night = 20-05h, fog/rain = rain or thunder)
# each group isolates one factor: night = night & dry, fog = day & rain/thunder, base = day & dry
SIM_COND = {"night": lambda r: (r.daynight == "night") & r.weather.isin(["clear", "cloudy"]),
            "fog": lambda r: (r.daynight == "day") & r.weather.isin(["rain", "thunder"])}
BASE = lambda r: (r.daynight == "day") & r.weather.isin(["clear", "cloudy"])


def _get(d, *keys, default=None):
    """Case-insensitive nested lookup."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        m = {str(kk).lower(): vv for kk, vv in cur.items()}
        cur = m.get(k.lower(), default)
        if cur is default:
            return default
    return cur


def _iter_json(path):
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.lower().endswith(".json"):
                    yield n, z.read(n)
    else:
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".json"):
                    p = os.path.join(root, f)
                    with open(p, "rb") as fh:
                        yield p, fh.read()


def _box(points):
    pts = np.asarray(points, float).reshape(-1, 2)
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


# ------------------------------------------------------------------ stats
def cmd_stats(a):
    rows, bad = [], 0
    for name, raw in _iter_json(a.labels):
        if a.only_yt and "YT" not in name.upper().replace("\\", "/").split("/")[-1] and "/YT/" not in name.upper().replace("\\", "/"):
            continue
        try:
            d = json.loads(raw.decode("utf-8-sig"))
        except Exception:
            bad += 1
            continue
        info = _get(d, "addinfo") or {}
        if a.only_safety and str(_get(info, "dataregion", default="1")) not in ("1", "Safety"):
            continue
        img = _get(d, "Images") or {}
        fn = _get(img, "filename") or os.path.basename(name).replace(".json", ".jpg")
        W, H = _get(img, "width", default=1920), _get(img, "height", default=1080)
        for k, ann in enumerate(_get(d, "annotations", default=[]) or []):
            bb = _get(ann, "bbox") or ann
            pts = _get(bb, "points")
            if not pts:
                continue
            x1, y1, x2, y2 = _box(pts)
            rows.append(dict(
                file=fn, obj=k, classid=str(_get(bb, "classid")),
                weather=WEATHER.get(str(_get(info, "weather")), str(_get(info, "weather"))),
                daynight=DAYNIGHT.get(str(_get(info, "daynight")), str(_get(info, "daynight"))),
                terminal=AREA.get(str(_get(info, "area")), str(_get(info, "area"))),
                location=_get(info, "location"), camid=_get(info, "camid"), time=_get(info, "time"),
                anglex=_get(info, "anglex"), angley=_get(info, "angley"),
                x1=x1, y1=y1, x2=x2, y2=y2, w=x2 - x1, h=y2 - y1,
                truncated=bool(x1 <= 1 or y1 <= 1 or x2 >= float(W) - 2 or y2 >= float(H) - 2)))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no objects parsed - check --labels and the Safety/YT filters")
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    piv = pd.crosstab(df.weather, df.daynight, margins=True)
    q = df.groupby(["daynight", "weather"]).h.describe(percentiles=[0.1, 0.5, 0.9])[["count", "10%", "50%", "90%"]]
    md = [f"objects: {len(df)}, images: {df.file.nunique()}, unreadable json: {bad}", "",
          "class ids: " + ", ".join(f"{k} ({v})" for k, v in df.classid.value_counts().items()), "",
          "objects by weather x time band", piv.to_markdown(), "",
          "box height (px) by condition", q.round(1).to_markdown(), "",
          "terminals: " + ", ".join(f"{k} ({v})" for k, v in df.terminal.value_counts().items()),
          f"cameras: {df.camid.nunique()}, truncated boxes: {df.truncated.mean():.1%}"]
    out_md = os.path.splitext(a.out)[0] + "_summary.md"
    open(out_md, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------------ detect
def _index_images(root):
    idx = {}
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                idx[f] = os.path.join(r, f)
    return idx


def _progress(i, n, t0, every=200):
    import time
    if (i + 1) % every == 0 or i + 1 == n:
        el = time.time() - t0
        print(f"  {i + 1}/{n} images, {el / (i + 1):.2f} s/img, remaining ~{el / (i + 1) * (n - i - 1) / 60:.0f} min", flush=True)


def cmd_detect(a):
    import time
    import cv2
    files = pd.read_csv(a.labels).file.unique().tolist()
    idx = _index_images(a.images) if a.images else {}
    if not any(f in idx for f in files):
        raise SystemExit(f"none of the {len(files)} labelled images were found under --images '{a.images}'. "
                         "Extract them (unzip VS.zip \"*SAF_CRANE_YT*\" -d VS_YT) and set PORT_VS_YT.")
    keep = set(int(c) for c in a.keep_classes.split(",")) if a.keep_classes else None
    rows = []
    if a.backend == "opencv":
        if not hasattr(cv2.dnn, "readNetFromDarknet"):
            raise SystemExit(f"OpenCV {cv2.__version__} has no Darknet importer (removed in OpenCV 5). "
                             'Install OpenCV 4:  uv pip install "opencv-python>=4.8,<5"')
        net = cv2.dnn.readNetFromDarknet(a.cfg, a.weights)
        outs = net.getUnconnectedOutLayersNames()
        print(f"opencv-dnn (CPU), {len(files)} images", flush=True)
        t0 = time.time()
        for i, f in enumerate(files):
            _progress(i, len(files), t0)
            p = idx.get(f)
            if p is None:
                continue
            im = cv2.imread(p)
            Hh, Ww = im.shape[:2]
            net.setInput(cv2.dnn.blobFromImage(im, 1 / 255.0, (a.size, a.size), swapRB=True, crop=False))
            boxes, scores, cls = [], [], []
            for o in net.forward(outs):
                for det in o:
                    sc = det[5:]
                    c = int(np.argmax(sc))
                    s = float(sc[c] * det[4])
                    if s < a.conf or (keep and c not in keep):
                        continue
                    cx, cy, bw, bh = det[0] * Ww, det[1] * Hh, det[2] * Ww, det[3] * Hh
                    boxes.append([cx - bw / 2, cy - bh / 2, bw, bh]); scores.append(s); cls.append(c)
            for i in np.array(cv2.dnn.NMSBoxes(boxes, scores, a.conf, 0.45)).reshape(-1):
                x, y, bw, bh = boxes[i]
                rows.append(dict(file=f, x1=x, y1=y, x2=x + bw, y2=y + bh, score=scores[i], cls=cls[i]))
    else:
        from ultralytics import YOLO
        dev = a.device
        if not dev:
            try:
                import torch
                dev = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
            except Exception:
                dev = "cpu"
        model = YOLO(a.weights)
        print(f"ultralytics on device {dev}, {len(files)} images", flush=True)
        t0 = time.time()
        for i, f in enumerate(files):
            _progress(i, len(files), t0)
            p = idx.get(f)
            if p is None:
                continue
            for r in model.predict(p, conf=a.conf, imgsz=a.size, device=dev, verbose=False):
                for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                    if keep and int(c) not in keep:
                        continue
                    rows.append(dict(file=f, x1=x1, y1=y1, x2=x2, y2=y2, score=s, cls=int(c)))
    pd.DataFrame(rows, columns=["file", "x1", "y1", "x2", "y2", "score", "cls"]).to_csv(a.out, index=False)
    if not rows:
        raise SystemExit(f"0 detections on {len(files)} images. Check with the 200-image sample: try --device cpu "
                         f"(Apple GPU issue) and/or drop --keep-classes (top-down crane views may get other COCO classes).")
    print(f"{len(rows)} detections on {len(files)} images -> {a.out} (missing images: {sum(f not in idx for f in files)})")


# ------------------------------------------------------------------ eval
def _iou(b, B):
    ix = np.maximum(0, np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]))
    iy = np.maximum(0, np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]))
    inter = ix * iy
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter + 1e-9)


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _logit_fit(x, y, iters=50, ridge=1e-3):
    """Logistic regression y ~ sigmoid(alpha + beta * x) by Newton steps."""
    X = np.c_[np.ones_like(x), x]
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (y - p) - ridge * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(Hm, g)
        w += step
        if np.abs(step).max() < 1e-6:
            break
    return float(w[0]), float(w[1])


def cmd_eval(a):
    L = pd.read_csv(a.labels, dtype={"classid": str})
    if a.gt_class:
        L = L[L.classid.isin(a.gt_class.split(","))]
    L = L[~L.truncated] if a.drop_truncated else L
    try:
        D = pd.read_csv(a.dets)
    except pd.errors.EmptyDataError:
        raise SystemExit(f"{a.dets} has no detections - rerun the detect step (see its message)")
    det_by = {f: g.sort_values("score", ascending=False) for f, g in D.groupby("file")}
    hit = np.zeros(len(L), bool)
    best_iou, best_cov = np.zeros(len(L)), np.zeros(len(L))
    for f, g in L.groupby("file"):
        dd = det_by.get(f)
        if dd is None or dd.empty:
            continue
        G = g[["x1", "y1", "x2", "y2"]].to_numpy()
        ga = (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1])
        idx = L.index.get_indexer(g.index)
        used = np.zeros(len(G), bool)
        for _, r in dd.iterrows():
            b = np.array([r.x1, r.y1, r.x2, r.y2])
            ix = np.maximum(0, np.minimum(b[2], G[:, 2]) - np.maximum(b[0], G[:, 0]))
            iy = np.maximum(0, np.minimum(b[3], G[:, 3]) - np.maximum(b[1], G[:, 1]))
            inter, da_ = ix * iy, (b[2] - b[0]) * (b[3] - b[1])
            iou = inter / (da_ + ga - inter + 1e-9)
            cov = inter / (ga + 1e-9)                              # share of the labelled box inside the detection
            best_iou[idx] = np.maximum(best_iou[idx], iou); best_cov[idx] = np.maximum(best_cov[idx], cov)
            if a.match == "iou":
                sc = iou.copy()
                ok = sc >= a.iou
            else:                                                  # detection must contain most of the box and not be huge
                sc = cov.copy()
                ok = (sc >= a.cover) & (da_ <= a.max_area_ratio * ga)
            sc[used | ~ok] = -1
            j = int(np.argmax(sc))
            if sc[j] >= 0:
                used[j] = True
        hit[idx] = used
    diag = dict(match=a.match, share_any_overlap=round(float((best_cov > 0).mean()), 4),
                share_iou_ge_0p5=round(float((best_iou >= 0.5).mean()), 4),
                share_cover_ge_0p5=round(float((best_cov >= 0.5).mean()), 4),
                best_iou_quantiles=[round(float(q), 3) for q in np.quantile(best_iou, [0.25, 0.5, 0.75, 0.9])],
                n_detections=int(len(D)), detection_classes=D.cls.value_counts().head(8).to_dict() if "cls" in D else {})
    print("matching diagnostics:", json.dumps(diag))
    L = L.assign(detected=hit, logh=np.log(np.maximum(L.h, 1.0)))
    out = {"n_objects": int(len(L)), "iou": a.iou, "cover": a.cover, "diagnostics": diag, "conditions": {}}
    groups = {"base_day_clear": BASE(L)}
    groups.update({k: f(L) for k, f in SIM_COND.items()})
    for name, m in groups.items():
        g = L[m]
        if len(g) < 30:
            out["conditions"][name] = dict(n=int(len(g)), note="too few objects")
            continue
        al, be = _logit_fit(g.logh.to_numpy(), g.detected.to_numpy().astype(float))
        lo, hi = _wilson(int(g.detected.sum()), len(g))
        out["conditions"][name] = dict(n=int(len(g)), recall=round(float(g.detected.mean()), 4), ci95=[round(lo, 4), round(hi, 4)],
                                       alpha=round(al, 4), beta=round(be, 4), median_h_px=round(float(g.h.median()), 1),
                                       h_quantiles_px=[round(float(v), 1) for v in g.h.quantile([0.1, 0.3, 0.5, 0.7, 0.9])])
    rb = out["conditions"].get("base_day_clear", {}).get("recall")
    out["degenerate"] = bool(rb is None or rb < 0.01)
    out["saturated"] = bool(rb is not None and rb > 0.97)
    if rb is None:
        print("WARNING: too few daytime/dry objects for a baseline (fine for a small sample, not for the full set)")
    elif out["degenerate"]:
        print(f"WARNING: daytime recall = {rb} - no usable calibration (detector boxes do not match the labelled boxes; "
              "try --match cover or another detector)")
    elif out["saturated"]:
        print(f"note: daytime recall = {rb} (near 1) - convert will use recall differences instead of logistic fits")
    # bootstrap CI of the condition offset da (logit difference at the daytime sizes), as used by convert
    b0 = L[groups["base_day_clear"]]
    rng = np.random.default_rng(0)
    if len(b0) >= 30:
        hq = np.log(b0.h.quantile([0.1, 0.3, 0.5, 0.7, 0.9]).to_numpy())
        for name in SIM_COND:
            g = L[groups[name]]
            if len(g) < 30:
                continue
            das = []
            for _ in range(a.boot):
                bb = b0.sample(len(b0), replace=True, random_state=int(rng.integers(1e9)))
                gg = g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
                ab, bb_ = _logit_fit(bb.logh.to_numpy(), bb.detected.to_numpy().astype(float))
                ag, bg = _logit_fit(gg.logh.to_numpy(), gg.detected.to_numpy().astype(float))
                das.append(np.mean((ag + bg * hq) - (ab + bb_ * hq)))
            out["conditions"][name]["da_ci95"] = [round(float(np.percentile(das, 2.5)), 3), round(float(np.percentile(das, 97.5)), 3)]
    bins = pd.qcut(L.h, q=min(6, max(2, len(L) // 200)), duplicates="drop")
    tab = L.groupby([bins, L.daynight], observed=True).detected.agg(["mean", "size"]).round(3)
    tab.to_csv(os.path.splitext(a.out)[0] + "_recall_by_height.csv")
    _plot_recall(L, groups, out, os.path.splitext(a.out)[0] + "_recall_by_height")
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


def _plot_recall(L, groups, out, stem):
    """Journal-style figure: recall vs box height per condition (points = binned rates, lines = logistic fits)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 600, "pdf.fonttype": 42})
    sty = {"base_day_clear": ("-", "o", "#000000", "Day, dry"), "night": ("--", "s", "#555555", "Night, dry"),
           "fog": (":", "^", "#8c8c8c", "Day, rain")}
    fig, ax = plt.subplots(figsize=(8.4 / 2.54, 6.0 / 2.54))
    hgrid = np.exp(np.linspace(np.log(max(L.h.min(), 5)), np.log(L.h.max()), 100))
    for name, (ls, mk, c, lab) in sty.items():
        g, f = L[groups[name]], out["conditions"].get(name, {})
        if "alpha" not in f:
            continue
        bins = np.exp(np.linspace(np.log(g.h.min()), np.log(g.h.max()), 7))
        idx = np.digitize(g.h, bins[1:-1])
        hb = [np.exp(g.logh[idx == i].mean()) for i in range(len(bins) - 1) if (idx == i).sum() >= 10]
        rb = [g.detected[idx == i].mean() for i in range(len(bins) - 1) if (idx == i).sum() >= 10]
        ax.plot(hb, rb, mk, color=c, ms=3.5, mfc="white" if name != "base_day_clear" else c, ls="none")
        ax.plot(hgrid, 1 / (1 + np.exp(-(f["alpha"] + f["beta"] * np.log(hgrid)))), ls, color=c,
                label=f"{lab} (n={f['n']}, recall {f['recall']:.2f})")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Bounding-box height (px)"); ax.set_ylabel("Detection recall (IoU ≥ 0.5)")
    ax.legend(frameon=False, fontsize=6.2, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ------------------------------------------------------------------ convert
def cmd_convert(a):
    EV = json.load(open(a.eval, encoding="utf-8"))
    if EV.get("degenerate") and not a.force:
        raise SystemExit("evaluation is degenerate (no daytime baseline or recall ~0) - no calibration written")
    E = EV["conditions"]
    base = E.get("base_day_clear", {})
    if "alpha" not in base:
        raise SystemExit("daytime/clear fit missing (too few objects)")
    hq = np.asarray(base["h_quantiles_px"], float)               # observed apparent sizes (daytime, dry)
    r_obs = a.focal_px * a.extent_m / hq                         # pinhole: range that gives each size
    lo_r, hi_r = max(a.rmin, r_obs.min()), min(a.rmax, r_obs.max())
    r = np.linspace(lo_r, hi_r, 200) if hi_r > lo_r else np.linspace(a.rmin, a.rmax, 200)
    hpx = lambda rr: a.focal_px * a.extent_m / rr

    def logit_h(c, h):
        return c["alpha"] + c["beta"] * np.log(h)

    b, a0 = np.polyfit(r, logit_h(base, hpx(r)), 1)             # logit p = a + b r (simulator form), observed range only
    spread = float(hq[-1] / hq[0])
    day = dict(a=round(float(a0), 4), b=round(float(b), 5), n=base["n"], recall=base["recall"],
               fitted_range_m=[round(r.min(), 1), round(r.max(), 1)], size_spread_p90_p10=round(spread, 2))
    if a.mode == "offsets":                                       # level/slope stay with nuScenes; images give condition effects
        day = dict(recall=base["recall"], n=base["n"], size_spread_p90_p10=round(spread, 2), applied=False)
    cam = {"mode": a.mode, "day": day, "conditions": {}}
    lg = lambda p: float(np.log(p / (1 - p)))
    for k in SIM_COND:
        c = E.get(k, {})
        if "recall" not in c:
            continue
        if EV.get("saturated") or c["recall"] > 0.97:
            # near-perfect detector: logistic fits are unstable, so the absolute recall drop is applied at the
            # simulator's reference detection probability p_ref
            dp = c["recall"] - base["recall"]
            p1 = float(np.clip(a.p_ref + dp, 0.01, 0.99))
            cam["conditions"][k] = dict(da=round(lg(p1) - lg(a.p_ref), 4), n=c["n"], recall=c["recall"],
                                        recall_drop=round(dp, 4), method=f"recall difference at p_ref={a.p_ref}")
        elif "alpha" in c:                                       # offset averaged over the observed daytime sizes
            cam["conditions"][k] = dict(da=round(float(np.mean(logit_h(c, hq) - logit_h(base, hq))), 4), n=c["n"],
                                        recall=c["recall"], da_ci95=c.get("da_ci95"), method="logistic fit")
    cam["assumptions"] = dict(focal_px=a.focal_px, extent_m=a.extent_m, range_m=[a.rmin, a.rmax],
                              note="pinhole size-to-range mapping; replace with the crane camera intrinsics when known; "
                                   "rain/thunder images map to the simulator's fog/rain condition")
    out = json.load(open(a.merge, encoding="utf-8")) if a.merge else {}
    out["camera_port"] = cam
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(cam, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("stats"); s.add_argument("--labels", required=True); s.add_argument("--out", default="port_labels.csv")
    s.add_argument("--all-regions", dest="only_safety", action="store_false"); s.add_argument("--all-folders", dest="only_yt", action="store_false")
    d = sp.add_parser("detect"); d.add_argument("--labels", required=True); d.add_argument("--images", required=True)
    d.add_argument("--backend", choices=["opencv", "ultralytics"], default="opencv"); d.add_argument("--cfg"); d.add_argument("--weights", required=True)
    d.add_argument("--size", type=int, default=608); d.add_argument("--conf", type=float, default=0.25)
    d.add_argument("--keep-classes", default=""); d.add_argument("--out", default="port_dets.csv")
    d.add_argument("--device", default="", help="ultralytics device: mps | cpu | 0 (default: auto, Apple GPU if available)")
    e = sp.add_parser("eval"); e.add_argument("--labels", required=True); e.add_argument("--dets", required=True)
    e.add_argument("--gt-class", default=""); e.add_argument("--iou", type=float, default=0.5)
    e.add_argument("--match", choices=["cover", "iou"], default="cover",
                   help="cover: detection contains >= --cover of the labelled box (robust when the label is the tractor head "
                        "and a generic detector boxes the whole truck); iou: classic IoU")
    e.add_argument("--cover", type=float, default=0.5); e.add_argument("--max-area-ratio", type=float, default=25.0)
    e.add_argument("--keep-truncated", dest="drop_truncated", action="store_false"); e.add_argument("--out", default="port_eval.json")
    e.add_argument("--boot", type=int, default=200, help="bootstrap resamples for the condition offsets")
    c = sp.add_parser("convert"); c.add_argument("--eval", required=True); c.add_argument("--merge", default="")
    c.add_argument("--focal-px", type=float, default=1400.0); c.add_argument("--extent-m", type=float, default=5.0)
    c.add_argument("--rmin", type=float, default=5.0); c.add_argument("--rmax", type=float, default=70.0)
    c.add_argument("--out", default="calib_port.json")
    c.add_argument("--mode", choices=["offsets", "full"], default="offsets",
                   help="offsets: only night/rain offsets from the port images (daytime curve stays from nuScenes); "
                        "full: also replace the daytime curve (needs a wide spread of apparent sizes)")
    c.add_argument("--force", action="store_true")
    c.add_argument("--p-ref", type=float, default=0.9, help="reference detection probability for saturated detectors")
    a = ap.parse_args()
    dict(stats=cmd_stats, detect=cmd_detect, eval=cmd_eval, convert=cmd_convert)[a.cmd](a)


if __name__ == "__main__":
    main()
