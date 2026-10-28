#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adrive3d_prep.py - AI Hub "특수환경 자율주행 3D 데이터 고도화" (dataSetSn=71287) -> LiDAR point density vs range

Reads the label zip files directly (no unzip, PCD and JPG are not needed). Each object carries its labelled points
(3D_points), the shortest distance to the ego origin (distance) and a class; time band and weather come from the file
name  <region>_<road code>_<time>_<weather>_<scene>.json  (e.g. seoul_0102_sunset_clear_0001).

Segmentation labels only exist for objects that received points, so the curve estimated here is
P(n_points >= k | range, condition) - a detectability proxy, not a miss probability.

Outputs (prefix --out-prefix)
  _objects.csv   one row per dynamic object (class, group, distance, n_points, time, weather, road code)
  _summary.md    tokens found in the file names, counts by time x weather, classes
  _lidar.json    logistic fits logit P = a + b*r per condition, offsets da vs the base condition -> {"lidar_kr": ...}
  _lidar.png/pdf figure for the paper / supplement

Usage
  python adrive3d_prep.py --zips <folder with the VL zip files> --k 10 --out-prefix adrive3d
  python adrive3d_prep.py --zips <folder> --k 10 --merge calib_port.json --out-prefix adrive3d   # adds lidar_kr to a calibration file
Options: --max-files N (sample for a quick look), --night-times night,dark  --wet-weathers rain,fog,snow  --base-time day
"""
import argparse
import glob
import json
import os
import zipfile

import numpy as np
import pandas as pd

try:
    import orjson as _fastjson          # optional, faster
    _loads = _fastjson.loads
except Exception:                       # pragma: no cover
    _loads = json.loads

GROUP = {"sedan": "vehicle", "suv": "vehicle", "truck": "vehicle", "bus": "vehicle", "car": "vehicle", "van": "vehicle",
         "pedestrian": "pedestrian", "person": "pedestrian", "bicycle": "two_wheeler", "motorcycle": "two_wheeler",
         "승용차": "vehicle", "세단": "vehicle", "트럭": "vehicle", "화물차": "vehicle", "버스": "vehicle", "승합차": "vehicle",
         "보행자": "pedestrian", "사람": "pedestrian", "자전거": "two_wheeler", "이륜차": "two_wheeler", "오토바이": "two_wheeler"}
CLASS_KEYS = ("class", "category", "label", "class_name", "name", "type")
POINT_KEYS = ("3D_points", "3d_points", "points", "point")


def _first(d, keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return None


def parse_name(stem):
    """<region>_<road code 4 digits>_<time>_<weather>_<scene>[_<frame>] - located from the road code, so an extra
    trailing frame number (as in the released files) does not shift the fields."""
    t = stem.split("_")
    i = next((k for k, x in enumerate(t) if k > 0 and len(x) == 4 and x.isdigit()), None)
    if i is None or i + 2 >= len(t):
        return dict(region=stem, road="", time="", weather="", scene="")
    return dict(region="_".join(t[:i]), road=t[i], time=t[i + 1].lower(), weather=t[i + 2].lower(),
                scene=t[i + 3] if i + 3 < len(t) else "")


def iter_labels(folder, max_files=0):
    zips = sorted(glob.glob(os.path.join(folder, "**", "*.zip"), recursive=True))
    if not zips:
        raise SystemExit(f"no zip files under {folder}")
    n = 0
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if not name.lower().endswith(".json"):
                    continue
                yield zp, name, z.read(name)
                n += 1
                if max_files and n >= max_files:
                    return


def fit_logistic(x, y, iters=60, ridge=1e-2, scale=50.0):
    """logit P = a + b*x, fitted on x/scale for conditioning; returns (a, b) in the original units."""
    X = np.c_[np.ones_like(x), x / scale]
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * w)
        w += step
        if np.abs(step).max() < 1e-7:
            break
    return float(w[0]), float(w[1] / scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", required=True, help="folder that contains the label zip files (searched recursively)")
    ap.add_argument("--k", type=int, default=10, help="points needed to call an object detectable")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--rmax", type=float, default=80.0)
    ap.add_argument("--base-time", default="", help="time token used as baseline (default: most frequent)")
    ap.add_argument("--base-weathers", default="clear,cloudy,sunny")
    ap.add_argument("--night-times", default="night,dark,evening")
    ap.add_argument("--wet-weathers", default="rain,rainy,fog,foggy,snow,snowy,haze")
    ap.add_argument("--merge", default="", help="calibration JSON to extend with the lidar_kr block")
    ap.add_argument("--min-n", type=int, default=200, help="objects needed per condition before an offset is reported")
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--out-prefix", default="adrive3d")
    a = ap.parse_args()

    from collections import Counter
    rows, bad, seen_cls, top_keys, ann_keys = [], 0, Counter(), Counter(), Counter()
    for zp, name, raw in iter_labels(a.zips, a.max_files):
        try:
            d = _loads(raw)
        except Exception:
            bad += 1
            continue
        meta = parse_name(os.path.splitext(os.path.basename(name))[0])
        if isinstance(d, dict):
            top_keys.update(list(d)[:15])
        anns = (_first(d, ("annotations", "annotation", "objects", "labels")) or []) if isinstance(d, dict) else []
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            ann_keys.update(list(ann)[:15])
            raw_cls = str(_first(ann, CLASS_KEYS) or "").strip()
            seen_cls[raw_cls] += 1
            cls = raw_cls.lower()
            if cls not in GROUP:
                continue                                   # static classes (median, guardrail) and unknown
            dist = ann.get("distance")
            pts = _first(ann, POINT_KEYS) or []
            if dist is None:
                c = _first(ann, ("box_center", "center", "location"))
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    dist = float(np.hypot(float(c[0]), float(c[1])))
            if dist is None:
                continue
            rows.append(dict(file=os.path.basename(name), cls=cls, group=GROUP[cls], distance=float(dist),
                             n_points=len(pts), **meta))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no dynamic objects parsed.\n"
                         f"  JSON files read: {sum(seen_cls.values()) and 'yes' or 'check zip contents'}, unreadable: {bad}\n"
                         f"  top-level keys seen: {dict(top_keys.most_common(10))}\n"
                         f"  annotation keys seen: {dict(ann_keys.most_common(12))}\n"
                         f"  class values seen: {dict(seen_cls.most_common(15))}\n"
                         "  -> if the zips are the 3D label files, send these lines so the class/key mapping can be added")
    df = df[df.distance <= a.rmax]
    df.to_csv(a.out_prefix + "_objects.csv", index=False)

    base_time = a.base_time or ("day" if (df.time == "day").any() else df.time.mode().iat[0])
    base_w, night_t, wet_w = (set(s.split(",")) for s in (a.base_weathers, a.night_times, a.wet_weathers))
    groups = {
        "base": (df.time == base_time) & df.weather.isin(base_w),
        "night": df.time.isin(night_t) & df.weather.isin(base_w),
        "fog": (df.time == base_time) & df.weather.isin(wet_w),     # simulator 'fog' = fog/rain condition
    }
    det = (df.n_points >= a.k).astype(float).to_numpy()
    out = {"k": a.k, "base_time": base_time, "fits": {}, "conditions": {}}
    for name, m in groups.items():
        m = m.to_numpy()
        for grp in ("vehicle", "pedestrian"):
            mm = m & (df.group == grp).to_numpy()
            if mm.sum() < 30:
                continue
            aa, bb = fit_logistic(df.distance.to_numpy()[mm], det[mm])
            out["fits"][f"{name}/{grp}"] = dict(a=round(aa, 4), b=round(bb, 5), n=int(mm.sum()),
                                                rate=round(float(det[mm].mean()), 4))
    r = np.linspace(5, 60, 56)
    rng = np.random.default_rng(0)
    # labels exist only for objects that received points, so P(n >= k) can saturate at 1 -> logistic fit meaningless
    base_rate = out["fits"].get("base/vehicle", {}).get("rate")
    out["degenerate"] = bool(base_rate is not None and (base_rate > 0.99 or base_rate < 0.01))
    # point-density analysis (always): log(n_points) ~ c + d*r per condition, ratio to the base at 20 m and 40 m
    dens = {}
    for name, m in groups.items():
        mm = (m & (df.group == "vehicle")).to_numpy()
        if mm.sum() >= 30:
            d_, c_ = np.polyfit(df.distance.to_numpy()[mm], np.log(df.n_points.to_numpy()[mm]), 1)
            dens[name] = dict(c=round(float(c_), 4), d=round(float(d_), 5), n=int(mm.sum()))
    if "base" in dens:
        for name in ("night", "fog"):
            if name in dens:
                b0, b1 = dens["base"], dens[name]
                dens[name]["ratio_20m"] = round(float(np.exp((b1["c"] + 20 * b1["d"]) - (b0["c"] + 20 * b0["d"]))), 3)
                dens[name]["ratio_40m"] = round(float(np.exp((b1["c"] + 40 * b1["d"]) - (b0["c"] + 40 * b0["d"]))), 3)
    out["point_density"] = dens
    m0 = (groups["base"] & (df.group == "vehicle")).to_numpy()
    for cond in ("night", "fog"):
        m1 = (groups[cond] & (df.group == "vehicle")).to_numpy()
        if out["degenerate"]:
            out["conditions"][cond] = dict(n=int(m1.sum()), note=f"P(n_points >= {a.k}) saturated at the baseline - "
                                                                 "offset not estimated; see point_density")
            continue
        if m0.sum() < a.min_n or m1.sum() < a.min_n:
            out["conditions"][cond] = dict(n=int(m1.sum()), note=f"fewer than {a.min_n} objects - offset not estimated")
            continue
        x0, y0, x1, y1 = df.distance.to_numpy()[m0], det[m0], df.distance.to_numpy()[m1], det[m1]

        def offset(xa, ya, xb, yb):
            a0, b0 = fit_logistic(xa, ya); a1, b1 = fit_logistic(xb, yb)
            return float(np.clip(np.mean((a1 + b1 * r) - (a0 + b0 * r)), -6, 6))
        da = offset(x0, y0, x1, y1)
        bs = []
        for _ in range(a.boot):
            i0 = rng.integers(0, len(x0), len(x0)); i1 = rng.integers(0, len(x1), len(x1))
            bs.append(offset(x0[i0], y0[i0], x1[i1], y1[i1]))
        out["conditions"][cond] = dict(da=round(da, 4), da_ci95=[round(float(np.percentile(bs, 2.5)), 3),
                                                                 round(float(np.percentile(bs, 97.5)), 3)], n=int(m1.sum()))
    tok = ["time tokens: " + ", ".join(f"{k} ({v})" for k, v in df.time.value_counts().items()),
           "weather tokens: " + ", ".join(f"{k} ({v})" for k, v in df.weather.value_counts().items()),
           "road codes: " + ", ".join(f"{k} ({v})" for k, v in df.road.value_counts().head(12).items())]
    md = [f"objects: {len(df)} (files read: {df.file.nunique()}, unreadable: {bad}), k = {a.k}, base time = {base_time}", *tok, "",
          "objects by time x weather", pd.crosstab(df.time, df.weather, margins=True).to_markdown(), "",
          "classes", df.cls.value_counts().to_markdown(), "",
          "fits (logit P(n_points >= k) = a + b*r)", json.dumps(out["fits"], indent=1), "",
          "condition offsets vs base (vehicles): " + json.dumps(out["conditions"]),
          "point density, log(n_points) = c + d*r, ratio to baseline: " + json.dumps(out["point_density"])]
    open(a.out_prefix + "_summary.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))

    cal = json.load(open(a.merge, encoding="utf-8")) if a.merge else {}
    cal["lidar_kr"] = out
    json.dump(cal, open(a.out_prefix + "_lidar.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 600, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.4 / 2.54, 6.0 / 2.54))
    sty = {"base": ("-", "o", "#000000", f"{base_time}, dry"), "night": ("--", "s", "#555555", "night, dry"),
           "fog": (":", "^", "#8c8c8c", "rain/fog")}
    bins = np.arange(0, a.rmax + 5, 5)
    for name, (ls, mk, c, lab) in sty.items():
        f = out["fits"].get(f"{name}/vehicle")
        if not f:
            continue
        g = df[groups[name] & (df.group == "vehicle")]
        idx = np.digitize(g.distance, bins)
        xb = [g.distance[idx == i].mean() for i in np.unique(idx) if (idx == i).sum() >= 20]
        yb = [(g.n_points[idx == i] >= a.k).mean() for i in np.unique(idx) if (idx == i).sum() >= 20]
        ax.plot(xb, yb, mk, color=c, ms=3.2, mfc="white" if name != "base" else c, ls="none")
        rr = np.linspace(0, a.rmax, 100)
        ax.plot(rr, 1 / (1 + np.exp(-(f["a"] + f["b"] * rr))), ls, color=c, label=f"{lab} (n={f['n']})")
    ax.set_xlabel("Range (m)"); ax.set_ylabel(f"P(points ≥ {a.k})"); ax.set_ylim(0, 1.02)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=6.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out_prefix}_lidar.{ext}", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
