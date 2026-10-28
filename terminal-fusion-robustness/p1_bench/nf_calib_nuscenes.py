#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nf_calib_nuscenes.py - estimate nf_sim sensor-model parameters from nuScenes metadata (no devkit needed)

Why: real airport data are not available, so the synthetic generator is grounded in public data.
nuScenes keyframe annotations carry num_lidar_pts, num_radar_pts and a camera visibility level, plus
per-sensor timestamps. From them we estimate, inside the near-field band (<= 60 m):
  lidar / radar / camera_vis40   logistic detection curves p(r) = 1/(1+exp(-(a+b r)))
                                  detected := num_lidar_pts>=1 / num_radar_pts>=1 / visibility>=40 %
                                  per class group (pedestrian, vehicle, two_wheeler) and condition
                                  (day, night, rain, all; from the scene description)
  timing                          per-channel median rate (Hz) and keyframe asynchrony vs LIDAR_TOP
Output JSON is read by nf_sim.apply_calib (via --calib) to replace the default detection curves.

Tables read from <dataroot>/<version>/: sample, sample_data, sample_annotation, instance, category,
ego_pose, calibrated_sensor, sensor, scene, visibility.
nuScenes is licensed for non-commercial use: use the estimates for research, not in deployed systems.

Usage
  python nf_calib_nuscenes.py --dataroot /data/nuscenes --version v1.0-mini --out nuscenes_calib.json
  python nf_calib_nuscenes.py --selftest
"""
import argparse
import json
import math
import os
import tempfile
from collections import defaultdict

import numpy as np

TABLES = ["sample", "sample_data", "sample_annotation", "instance", "category", "ego_pose",
          "calibrated_sensor", "sensor", "scene", "visibility"]
GROUPS = ("pedestrian", "vehicle", "two_wheeler")
CONDS = ("day", "night", "rain", "all")


def load_tables(root, version):
    d = os.path.join(root, version)
    return {n: json.load(open(os.path.join(d, n + ".json"), encoding="utf-8")) for n in TABLES}


def group_of(name):
    if name.startswith("human.pedestrian"):
        return "pedestrian"
    if name.startswith(("vehicle.bicycle", "vehicle.motorcycle")):
        return "two_wheeler"
    if name.startswith(("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.construction",
                        "vehicle.trailer", "vehicle.emergency")):
        return "vehicle"
    return None


def condition_of(desc):
    d = (desc or "").lower()
    return "night" if "night" in d else ("rain" if "rain" in d else "day")


def fit_logistic(r, y, ridge=1e-3, iters=50):
    """Newton-Raphson logistic regression of y on [1, r] (r in metres)."""
    X = np.c_[np.ones_like(r), r / 10.0]
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - ridge * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(Hm, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0]), float(w[1] / 10.0)


def calibrate(t, rmax=60.0, min_n=20):
    by = {n: {r["token"]: r for r in t[n]} for n in ("sample", "ego_pose", "calibrated_sensor", "sensor",
                                                     "scene", "instance", "category", "visibility")}
    chan = {sd["token"]: by["sensor"][by["calibrated_sensor"][sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
            for sd in t["sample_data"]}
    ego_kf, kf_ts, scene_ts = {}, defaultdict(dict), defaultdict(lambda: defaultdict(list))
    for sd in t["sample_data"]:
        ch = chan[sd["token"]]
        scn = by["sample"][sd["sample_token"]]["scene_token"]
        scene_ts[scn][ch].append(sd["timestamp"])
        if sd.get("is_key_frame"):
            kf_ts[sd["sample_token"]][ch] = sd["timestamp"]
            if ch == "LIDAR_TOP":
                ego_kf[sd["sample_token"]] = by["ego_pose"][sd["ego_pose_token"]]["translation"]
    rows = []
    for ann in t["sample_annotation"]:
        st = ann["sample_token"]
        if st not in ego_kf:
            continue
        g = group_of(by["category"][by["instance"][ann["instance_token"]]["category_token"]]["name"])
        if g is None:
            continue
        e = ego_kf[st]
        r = math.hypot(ann["translation"][0] - e[0], ann["translation"][1] - e[1])
        if r > rmax:
            continue
        cond = condition_of(by["scene"][by["sample"][st]["scene_token"]].get("description", ""))
        lvl = by["visibility"].get(ann.get("visibility_token"), {}).get("level", "")
        rows.append((g, cond, r, ann.get("num_lidar_pts", 0) >= 1, ann.get("num_radar_pts", 0) >= 1,
                     lvl in ("v40-60", "v60-80", "v80-100")))
    out = {"n_annotations": len(rows), "rmax_m": rmax}
    for key, col in (("lidar", 3), ("radar", 4), ("camera_vis40", 5)):
        out[key] = {}
        for g in GROUPS:
            out[key][g] = {}
            for c in CONDS:
                sel = [x for x in rows if x[0] == g and (c == "all" or x[1] == c)]
                if len(sel) < min_n:
                    continue
                r = np.array([x[2] for x in sel])
                y = np.array([x[col] for x in sel], float)
                a, b = fit_logistic(r, y)
                p = lambda d: 1 / (1 + math.exp(-(a + b * d)))
                out[key][g][c] = dict(a=round(a, 4), b=round(b, 5), n=len(sel), empirical_rate=round(float(y.mean()), 3),
                                      p10=round(p(10), 3), p30=round(p(30), 3), p50=round(p(50), 3))
    rates = defaultdict(list)
    for scn, chs in scene_ts.items():
        for ch, ts in chs.items():
            ts = np.sort(np.array(ts, float))
            if ts.size > 2:
                rates[ch].append(1e6 / np.median(np.diff(ts)))
    asyn = defaultdict(list)
    for st, chs in kf_ts.items():
        if "LIDAR_TOP" in chs:
            for ch, ts in chs.items():
                if ch != "LIDAR_TOP":
                    asyn[ch].append((ts - chs["LIDAR_TOP"]) / 1000.0)
    out["timing"] = dict(
        rate_hz={ch: round(float(np.median(v)), 2) for ch, v in rates.items()},
        keyframe_async_ms={ch: dict(median=round(float(np.median(v)), 1), p95_abs=round(float(np.percentile(np.abs(v), 95)), 1),
                                    n=len(v)) for ch, v in asyn.items()})
    return out


# ------------------------------------------------------------------ self-test with fake tables
def _fake_tables(seed=0):
    """Minimal nuScenes-shaped tables with known behaviour: detection falls with range; camera worse at night."""
    rng = np.random.default_rng(seed)
    T = {n: [] for n in TABLES}
    for lvl, tok in (("v0-40", "1"), ("v40-60", "2"), ("v60-80", "3"), ("v80-100", "4")):
        T["visibility"].append(dict(token=tok, level=lvl, description=lvl))
    for ch, mod in (("LIDAR_TOP", "lidar"), ("CAM_FRONT", "camera"), ("RADAR_FRONT", "radar")):
        T["sensor"].append(dict(token="s_" + ch, channel=ch, modality=mod))
        T["calibrated_sensor"].append(dict(token="c_" + ch, sensor_token="s_" + ch, translation=[0, 0, 0],
                                           rotation=[1, 0, 0, 0], camera_intrinsic=[]))
    cats = [("human.pedestrian.adult", "pedestrian"), ("vehicle.car", "vehicle"), ("vehicle.bicycle", "two_wheeler")]
    for i, (name, _) in enumerate(cats):
        T["category"].append(dict(token=f"cat{i}", name=name, description=name))
        T["instance"].append(dict(token=f"inst{i}", category_token=f"cat{i}", nbr_annotations=0))
    tsd = 0
    for si, desc in enumerate(("Day, peds crossing", "Night, rain, parking lot")):
        night = "night" in desc.lower()
        T["scene"].append(dict(token=f"scene{si}", name=f"scene-{si}", description=desc, log_token="log"))
        t0 = 1_600_000_000_000_000 + si * 10 ** 9
        for k in range(40):
            stok, ts = f"smp{si}_{k}", t0 + k * 500_000
            T["sample"].append(dict(token=stok, timestamp=ts, scene_token=f"scene{si}", prev="", next=""))
            ego = [10.0 * k * 0.5, 0.0, 0.0]
            for ch, period_us, jit in (("LIDAR_TOP", 50_000, 0), ("CAM_FRONT", 83_333, 30_000), ("RADAR_FRONT", 76_923, 40_000)):
                n_sweep = int(500_000 // period_us)
                for j in range(n_sweep):
                    tsd += 1
                    t_s = ts + j * period_us + (int(rng.integers(0, jit)) if (j == 0 and jit) else 0)
                    T["ego_pose"].append(dict(token=f"ego{tsd}", timestamp=t_s, translation=ego, rotation=[1, 0, 0, 0]))
                    T["sample_data"].append(dict(token=f"sd{tsd}", sample_token=stok, ego_pose_token=f"ego{tsd}",
                                                 calibrated_sensor_token="c_" + ch, timestamp=t_s, is_key_frame=(j == 0),
                                                 fileformat="", filename="", prev="", next=""))
            for a in range(30):
                ci = int(rng.integers(0, 3))
                r = float(rng.uniform(2, 60))
                ang = rng.uniform(0, 2 * math.pi)
                lam = 4000.0 / (r * r) * (3.0 if ci == 1 else 1.0)
                vis = 4 if rng.random() < (0.9 - r / 120 - (0.45 if night else 0.0)) else 1
                T["sample_annotation"].append(dict(
                    token=f"ann{si}_{k}_{a}", sample_token=stok, instance_token=f"inst{ci}", visibility_token=str(vis),
                    attribute_tokens=[], translation=[ego[0] + r * math.cos(ang), ego[1] + r * math.sin(ang), 0.0],
                    size=[1, 1, 1], rotation=[1, 0, 0, 0], prev="", next="",
                    num_lidar_pts=int(rng.poisson(lam)), num_radar_pts=int(rng.random() < max(0.05, 0.9 - r / 60))))
    return T


def selftest():
    T = _fake_tables()
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "v1.0-fake"))
        for n, rows in T.items():
            json.dump(rows, open(os.path.join(d, "v1.0-fake", n + ".json"), "w"))
        out = calibrate(load_tables(d, "v1.0-fake"))
    lv = out["lidar"]["vehicle"]["all"]
    cd, cn = out["camera_vis40"]["pedestrian"]["day"], out["camera_vis40"]["pedestrian"]["night"]
    checks = {"lidar detection falls with range (b < 0)": lv["b"] < 0,
              "camera visibility lower at night": cn["p30"] < cd["p30"],
              "lidar rate ~20 Hz": abs(out["timing"]["rate_hz"]["LIDAR_TOP"] - 20.0) < 0.5,
              "camera async measured": out["timing"]["keyframe_async_ms"]["CAM_FRONT"]["n"] > 0}
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    print("  lidar/vehicle/all:", lv, "| camera/ped day p30", cd["p30"], "night p30", cn["p30"])
    if not all(checks.values()):
        raise SystemExit("selftest failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataroot")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default="nuscenes_calib.json")
    ap.add_argument("--rmax", type=float, default=60.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.dataroot:
        raise SystemExit("--dataroot required (or --selftest)")
    out = calibrate(load_tables(a.dataroot, a.version), a.rmax)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"{out['n_annotations']} annotations -> {a.out}")


if __name__ == "__main__":
    main()
