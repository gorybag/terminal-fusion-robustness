#!/usr/bin/env python3
"""
make_sensitivity_fig.py - calibration-level sensitivity figure/table for the missing-modality experiment (A)

Each --level is NAME=GLOB of result folders written by nf_experiments.py (one folder per seed); seeds are averaged.
    python make_sensitivity_fig.py --level "L0 assumed=p1_bench/results_bench" \
        --level "L1 public=runs/<run>/bench_port_s*" --level "L2 lower=runs/<runA>/bench_port_s*" \
        --level "L3 upper=runs/<runB>/bench_port_s*" --out sensitivity
Outputs: <out>.csv (mean and sd over seeds), <out>.png/.pdf (macro P and GNSS-RTK dependency per method and level).
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["LEARN_ModDrop", "EARLY", "LATE", "LEARN_noDrop"]
LABEL = {"LEARN_ModDrop": "Learned +\nmodality dropout", "EARLY": "Early", "LATE": "Late", "LEARN_noDrop": "Learned\n(no dropout)"}
FULL = {"lidar", "camera", "thermal", "gnss"}


def metrics(csv):
    d = pd.read_csv(csv)
    out = {}
    for m in METHODS:
        g = d[d.method == m]
        by = {frozenset(p.split("+")): v for p, v in zip(g.pattern, g.p_le)}
        out[m] = dict(macro=g.p_le.mean(), k1=g[g.k == 1].p_le.mean(), worst=g.p_le.min(),
                      d_gnss=by[frozenset(FULL)] - by[frozenset(FULL - {"gnss"})])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", action="append", required=True)
    ap.add_argument("--out", default="sensitivity")
    a = ap.parse_args()
    rows = []
    for spec in a.level:
        name, pat = spec.split("=", 1)
        dirs = sorted(glob.glob(pat))
        for dd in dirs:
            f = os.path.join(dd, "A_missing_modality.csv")
            if os.path.isfile(f):
                for m, v in metrics(f).items():
                    rows.append(dict(level=name, seed=os.path.basename(dd), method=m, **v))
    df = pd.DataFrame(rows)
    num = df.drop(columns=["seed"])
    tab = num.groupby(["level", "method"]).agg(["mean", "std"]).round(3)
    tab[("n_seeds", "")] = df.groupby(["level", "method"]).seed.nunique()
    tab.to_csv(a.out + ".csv"); print(tab.to_string())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 600, "pdf.fonttype": 42})
    levels = list(dict.fromkeys(df.level))
    mk = ["o", "s", "^", "D"]; col = ["#000000", "#555555", "#8c8c8c", "#b5b5b5"]
    fig, axs = plt.subplots(1, 2, figsize=(17 / 2.54, 6.0 / 2.54))
    for ax, key, ttl in zip(axs, ("macro", "d_gnss"), ("(a) Macro P(error ≤ 0.5 m), 15 subsets", "(b) GNSS-RTK dependency D")):
        for j, lv in enumerate(levels):
            g = df[df.level == lv].groupby("method")[key]
            mu, sd = g.mean().reindex(METHODS), g.std().reindex(METHODS).fillna(0)
            x = np.arange(len(METHODS)) + (j - (len(levels) - 1) / 2) * 0.16
            ax.errorbar(x, mu, yerr=sd, fmt=mk[j % 4], color=col[j % 4], mfc="white" if j % 2 else col[j % 4],
                        capsize=1.5, elinewidth=0.6, ls="none", label=lv)
        ax.set_xticks(range(len(METHODS)), [LABEL[m] for m in METHODS], fontsize=6.5)
        ax.set_title(ttl, loc="left", fontsize=7)
        ax.set_ylim(0, 1.0)
    axs[0].legend(frameon=False, fontsize=6.3, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
