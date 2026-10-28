#!/usr/bin/env python3
"""
summarize_p.py - collision-oriented results (experiment P of nf_experiments.py) averaged over seeds, per level.

    python summarize_p.py --level "L1 public=runs/<run>/bench_port_s*" [--level "L2 lower=runs/<runA>/bench_port_s*"] --out p_summary
Tables (mean over seeds, sd in brackets):
  class   macro P(abs error <= 0.5 m) per class (YT=tractor, RS=forklift, XT=amr, worker) and method, incl. coverage
  pairs   close pairs (<= 10 m): P(|relative-distance error| <= 0.5 m), over-estimation rate while closing
  latency closing pairs, full modalities: over-estimation rate without vs with timestamp alignment
"""
import argparse, glob, os
import pandas as pd

NAMES = {"tractor": "YT", "forklift": "RS", "amr": "XT", "worker": "worker"}


def load(pattern, fn):
    out = []
    for d in sorted(glob.glob(pattern)):
        f = os.path.join(d, fn)
        if os.path.isfile(f):
            out.append(pd.read_csv(f).assign(seed=os.path.basename(d)))
    return pd.concat(out) if out else pd.DataFrame()


def fmt(g, col):
    m, s = g[col].mean(), g[col].std()
    return f"{m:.3f}" + (f" ({s:.3f})" if pd.notna(s) and len(g) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", action="append", required=True)
    ap.add_argument("--out", default="p_summary")
    a = ap.parse_args()
    md = []
    for spec in a.level:
        name, pat = spec.split("=", 1)
        C, P, L = load(pat, "P_by_class.csv"), load(pat, "P_pairs.csv"), load(pat, "P_latency_pairs.csv")
        if C.empty:
            md.append(f"## {name}: no P results under {pat}"); continue
        md.append(f"## {name} ({C.seed.nunique()} seeds)")
        c = C.groupby(["nodes", "cls", "method", "seed"]).p_le.mean().reset_index()
        t = c.groupby(["nodes", "cls", "method"]).apply(lambda g: fmt(g, "p_le"), include_groups=False).unstack("method")
        t.index = [f"{n} {NAMES.get(k, k)}" for n, k in t.index]
        md += ["### class: macro P(abs error <= 0.5 m) over 15 subsets", t.to_markdown(), ""]
        p = P.groupby(["nodes", "pair_type", "method", "seed"])[["p_rel_le", "overest_closing"]].mean().reset_index()
        for col, lab in (("p_rel_le", "P(|relative error| <= 0.5 m), close pairs"), ("overest_closing", "over-estimation rate while closing (> 0.5 m)")):
            t = p.groupby(["nodes", "pair_type", "method"]).apply(lambda g: fmt(g, col), include_groups=False).unstack("method")
            md += [f"### pairs: {lab}", t.to_markdown(), ""]
        if not L.empty:
            t = L.groupby(["nodes", "pair_type", "fusion"]).apply(lambda g: fmt(g, "overest_rate"), include_groups=False).unstack("fusion")
            md += ["### latency: over-estimation rate on closing pairs (full modalities)", t.to_markdown(), ""]
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
