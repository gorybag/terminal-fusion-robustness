#!/usr/bin/env python3
"""
compare_calibration.py - before/after table for the public-data calibration (assumed vs calibrated runs).

    python p1_bench/nf_experiments.py --out results_bench                       # assumed parameters (existing)
    python p1_bench/nf_experiments.py --calib nuscenes_calib.json --out results_calib
    python compare_calibration.py --base p1_bench/results_bench --calib p1_bench/results_calib --out calib_comparison.md

Reports, per fusion method: macro P(e<=0.5 m) over the 15 modality subsets, the k=1 mean, the worst subset and the
GNSS-RTK dependency D = P(all) - P(all minus GNSS); then whether the method ranking survives (Kendall tau), plus the
clock-sync and coverage rows. The calibration currently overrides only the LiDAR/camera daytime detection curves.
"""
import argparse, json, os
import pandas as pd
from scipy.stats import kendalltau

METHODS = ["LEARN_ModDrop", "EARLY", "LATE", "LEARN_noDrop"]
FULL = {"lidar", "camera", "thermal", "gnss"}


def summarize(d):
    rows = {}
    for m in METHODS:
        g = d[d.method == m]
        by = {frozenset(p.split("+")): v for p, v in zip(g.pattern, g.p_le)}
        rows[m] = dict(macro=g.p_le.mean(), k1=g[g.k == 1].p_le.mean(), worst=g.p_le.min(),
                       d_gnss=by[frozenset(FULL)] - by[frozenset(FULL - {"gnss"})])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", default="calib_comparison.md")
    a = ap.parse_args()
    A0 = summarize(pd.read_csv(os.path.join(a.base, "A_missing_modality.csv")))
    A1 = summarize(pd.read_csv(os.path.join(a.calib, "A_missing_modality.csv")))
    lines = ["| Method | Metric | Assumed | Calibrated | Change |", "| --- | --- | --- | --- | --- |"]
    for m in METHODS:
        for k, lab in (("macro", "macro P (15 subsets)"), ("k1", "mean P, one modality"),
                       ("worst", "worst subset"), ("d_gnss", "GNSS-RTK dependency D")):
            lines.append(f"| {m} | {lab} | {A0[m][k]:.3f} | {A1[m][k]:.3f} | {A1[m][k]-A0[m][k]:+.3f} |")
    tau, p = kendalltau([A0[m]["macro"] for m in METHODS], [A1[m]["macro"] for m in METHODS])
    r0 = sorted(METHODS, key=lambda m: -A0[m]["macro"]); r1 = sorted(METHODS, key=lambda m: -A1[m]["macro"])
    lines += ["", f"Ranking (assumed):    {' > '.join(r0)}", f"Ranking (calibrated): {' > '.join(r1)}",
              f"Kendall tau = {tau:.3f} (p = {p:.3f}); ranking {'preserved' if r0 == r1 else 'CHANGED'}"]
    s0 = json.load(open(os.path.join(a.base, "summary.json"))); s1 = json.load(open(os.path.join(a.calib, "summary.json")))
    if "C_sync" in s0 and "C_sync" in s1:
        lines += ["", "| Clock sync | P (assumed) | P (calibrated) | p95 m (assumed) | p95 m (calibrated) |", "| --- | --- | --- | --- | --- |"]
        for x, y in zip(s0["C_sync"], s1["C_sync"]):
            lines.append(f"| {x['clock']} | {x['p_le']:.3f} | {y['p_le']:.3f} | {x['p95']:.3f} | {y['p95']:.3f} |")
    lines += ["", f"calibration file: {s1.get('calib')}"]
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
