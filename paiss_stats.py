#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paiss_stats.py - NVIDIA Physical AI Smart Spaces (MTMC_Tracking_2026) ground truth -> statistics for the paper

Uses only ground_truth.json and calibration.json of each scene (videos and depth maps are not needed).
  kinematics   speed / acceleration percentiles per object class (forklift, pallet truck, transporter, AMR, person)
  proximity    distance from each person to the nearest vehicle-class object, per sampled frame
  visibility   number of cameras that see each object (0 / 1 / >=2) -> occlusion statistics for multi-node layouts
  cameras      camera mounting height from the extrinsic matrix (camera centre C = -R^T t)

Scenes 001/003/005/007/021/024 are Cosmos-Transfer re-renderings with the same ground truth as their sources and
are skipped by default so trajectories are not counted twice.

Usage
  python paiss_stats.py --root <.../MTMC_Tracking_2026> --splits train val --stride 3 --out-prefix paiss
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

VEHICLES = {"forklift", "pallettruck", "transporter", "novacarter"}
DUPLICATES = {"Warehouse_001", "Warehouse_003", "Warehouse_005", "Warehouse_007", "Warehouse_021", "Warehouse_024"}
FPS = 30.0


def g(o, key, default=None):
    """ground-truth keys appear as 'object_type' or 'object type' depending on the release"""
    for k in (key, key.replace("_", " ")):
        if k in o:
            return o[k]
    return default


def cam_heights(calib_path):
    out = []
    try:
        cal = json.load(open(calib_path, encoding="utf-8"))
    except Exception:
        return out
    for s in cal.get("sensors", []):
        E = g(s, "extrinsicMatrix") or s.get("extrinsic matrix")
        if not E:
            continue
        E = np.asarray(E, float)
        R, t = E[:, :3], E[:, 3]
        C = -R.T @ t
        out.append(dict(camera=s.get("id"), x=C[0], y=C[1], z=C[2]))
    return out


def scene_stats(scene_dir, stride, max_frames):
    gt = json.load(open(os.path.join(scene_dir, "ground_truth.json"), encoding="utf-8"))
    frames = sorted(gt, key=lambda k: int(k))[::stride]
    if max_frames:
        frames = frames[:max_frames]
    dt = stride / FPS
    track = defaultdict(list)                       # (class, id) -> [(frame, x, y)]
    vis, prox = [], []
    for f in frames:
        objs = gt[f]
        persons, vehicles = [], []
        for o in objs:
            cls = str(g(o, "object_type", "")).lower().replace("_", "").replace(" ", "")
            loc = g(o, "3d_location")
            if loc is None:
                continue
            x, y = loc[0], loc[1]
            track[(cls, g(o, "object_id"))].append((int(f), x, y))
            vis.append(dict(cls=cls, n_cams=len(g(o, "2d_bounding_box_visible") or {})))
            (persons if cls == "person" else vehicles if cls in VEHICLES else []).append((x, y))
        if persons and vehicles:
            P, V = np.asarray(persons), np.asarray(vehicles)
            d = np.sqrt(((P[:, None, :] - V[None, :, :]) ** 2).sum(-1)).min(1)
            prox.extend(d.tolist())
    kin = []
    for (cls, oid), tr in track.items():
        tr = np.asarray(sorted(tr))
        if len(tr) < 3:
            continue
        gap = np.diff(tr[:, 0]) / FPS
        v = np.sqrt((np.diff(tr[:, 1:], axis=0) ** 2).sum(1)) / np.maximum(gap, 1e-6)
        ok = (gap <= 2 * dt + 1e-6) & (v < 20)        # drop re-entries and teleports
        v = v[ok]
        if len(v) < 2:
            continue
        acc = np.abs(np.diff(v)) / dt
        kin.append(dict(cls=cls, obj=oid, v_p50=np.percentile(v, 50), v_p95=np.percentile(v, 95), v_max=v.max(),
                        a_p95=np.percentile(acc, 95) if len(acc) else np.nan, moving_frac=float((v > 0.2).mean())))
    return pd.DataFrame(kin), pd.DataFrame(vis), np.asarray(prox)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--stride", type=int, default=3, help="frame step (30 fps / 3 = 10 Hz)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--out-prefix", default="paiss")
    a = ap.parse_args()
    K, Vis, Prox, Cams, used = [], [], [], [], []
    for sp in a.splits:
        for sd in sorted(glob.glob(os.path.join(a.root, sp, "*"))):
            name = os.path.basename(sd)
            if not os.path.isfile(os.path.join(sd, "ground_truth.json")):
                continue
            if name in DUPLICATES and not a.keep_duplicates:
                continue
            k, v, p = scene_stats(sd, a.stride, a.max_frames)
            K.append(k.assign(scene=name)); Vis.append(v.assign(scene=name)); Prox.append(p)
            Cams += [dict(scene=name, **c) for c in cam_heights(os.path.join(sd, "calibration.json"))]
            used.append(name)
            print(f"{name}: {len(k)} tracks")
    if not used:
        raise SystemExit("no scene with ground_truth.json found under --root/<split>/")
    K, Vis, Cams = pd.concat(K), pd.concat(Vis), pd.DataFrame(Cams)
    prox = np.concatenate(Prox) if Prox else np.array([])
    kin = K.groupby("cls").agg(n_tracks=("obj", "size"), speed_p50=("v_p50", "median"), speed_p95=("v_p95", "median"),
                               speed_p99_all=("v_max", lambda s: np.percentile(s, 99)), acc_p95=("a_p95", "median"),
                               moving_frac=("moving_frac", "median")).round(2)
    vis = Vis.assign(b=pd.cut(Vis.n_cams, [-1, 0, 1, 99], labels=["0", "1", ">=2"])).groupby("cls").b.value_counts(normalize=True)
    vis = vis.unstack().round(3)
    kin.to_csv(a.out_prefix + "_kinematics.csv"); vis.to_csv(a.out_prefix + "_visibility.csv")
    Cams.to_csv(a.out_prefix + "_cameras.csv", index=False)
    pq = {f"p{q}": round(float(np.percentile(prox, q)), 2) for q in (5, 25, 50)} if len(prox) else {}
    summ = dict(scenes=used, stride=a.stride, person_to_vehicle_m=pq,
                frac_person_within_2m=round(float((prox < 2).mean()), 4) if len(prox) else None,
                frac_person_within_5m=round(float((prox < 5).mean()), 4) if len(prox) else None,
                camera_height_m=dict(median=round(float(Cams.z.median()), 2), p10=round(float(Cams.z.quantile(.1)), 2),
                                     p90=round(float(Cams.z.quantile(.9)), 2)) if len(Cams) else None)
    json.dump(summ, open(a.out_prefix + "_summary.json", "w"), indent=1)
    print("\nkinematics (m/s, m/s^2)\n", kin.to_string(), "\n\nvisibility (share of object-frames)\n", vis.to_string(),
          "\n\n", json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
