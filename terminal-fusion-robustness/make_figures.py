#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py - journal-style figures for the near-field fusion paper (KCI two-column layout).

Reads the CSV/JSON results already produced by p1_sim/ and p1_bench/ and writes PNG (600 dpi) + PDF (vector)
figures with English labels, grayscale-safe styling (markers + line styles, gray maps with printed values),
and widths for one column (8.4 cm) or the full page (17.0 cm). Check the journal template and change COL/PAGE.

Usage:  python make_figures.py --kit . --out nearfield_figures
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import ListedColormap, BoundaryNorm

CM = 1 / 2.54
COL, PAGE = 8.4 * CM, 17.0 * CM          # one column / full width (inches)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "lines.markersize": 3.6, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
})
GRAYS = ["#000000", "#404040", "#7a7a7a", "#a8a8a8"]
STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]
BENCH, SIM = "p1_bench/results_bench", "p1_sim/results"


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig. 2 latency-error budget
NAMES = {"airport_yard": {"yard_tractor": "Yard tractor", "forklift": "Forklift", "amr": "Cargo AMR", "pedestrian": "Pedestrian"},
         "port_terminal": {"yard_tractor": "Yard tractor (YT)", "forklift": "Reach stacker (RS)",
                           "amr": "External truck (XT)", "pedestrian": "Worker"}}
PROFILE = "airport_yard"


def fig_latency(kit, out):
    d = pd.read_csv(os.path.join(SIM, "e1_latency_error.csv"))
    order = ["yard_tractor", "forklift", "amr", "pedestrian"]
    vmax = d.groupby("agent").v_max.first().to_dict()
    fig, axs = plt.subplots(1, 2, figsize=(PAGE, 6.2 * CM), sharey=True)
    for i, k in enumerate(order):
        g = d[d.agent == k].sort_values("tau_ms")
        lab = f"{NAMES[PROFILE][k]} ({vmax[k]:.1f} m/s)"
        axs[0].plot(g.tau_ms, g.naive_mean_err_m, STYLES[i], marker=MARKERS[i], color=GRAYS[i], label=lab)
        axs[0].plot(g.tau_ms, g.analytic_bias_m, ":", color=GRAYS[i], lw=0.6)
        axs[1].plot(g.tau_ms, g.ts_aware_mean_err_m, STYLES[i], marker=MARKERS[i], color=GRAYS[i], label=lab)
    for ax, ttl in zip(axs, ("(a) Fusion without timestamp compensation", "(b) Timestamp-aware fusion")):
        ax.axhline(0.5, color="k", lw=0.6, ls=(0, (4, 2)))
        ax.text(203, 0.5, "0.5 m budget", va="bottom", ha="right", fontsize=6.5)
        ax.set_xlabel("Uncompensated latency τ (ms)")
        ax.set_title(ttl, loc="left")
        ax.set_xlim(-5, 205)
    axs[0].set_ylabel("Mean position error (m)")
    axs[0].set_ylim(0, 1.05)
    fast = sorted({round(v, 2) for v in vmax.values()}, reverse=True)[:2]
    def short(k):
        import re
        m = re.search(r"\(([A-Za-z]+)\)", NAMES[PROFILE][k])
        return m.group(1) if m else NAMES[PROFILE][k]
    for j, v in enumerate(fast):          # labels placed in empty regions: left of the first line, right of the second
        tau = 500.0 / v
        axs[0].axvline(tau, color="#7a7a7a", lw=0.5)
        who = "/".join(short(k) for k in order if round(vmax[k], 2) == v)
        x, ha, y = (tau - 3, "right", 0.62) if j == 0 else (tau + 3, "left", 0.30)
        axs[0].text(x, y, f"τmax = {tau:.0f} ms\n({who})", fontsize=6.0, color="#404040", va="bottom", ha=ha)
    axs[0].text(4, 0.97, "dotted: analytic bias v·τ", fontsize=6.3, color="#404040", va="top")
    axs[0].set_xlim(-5, 205)
    axs[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.2)
    save(fig, out, "fig02_latency_error_budget")


# ---------------------------------------------------------------- Fig. 4 strategy x condition
def fig_strategy(kit, out):
    d = pd.read_csv(os.path.join(SIM, "e2_strategy_condition.csv"))
    conds = ["C0_nominal", "C1_night", "C2_fog_rain", "C3_multipath", "C4_silent_fault", "C5_compound"]
    clab = ["Nominal", "Night", "Fog/\nrain", "Multi-\npath", "Silent\nfault", "Com-\npound"]
    strats = ["S0_lidar_only", "S1_naive", "S2_ts_aware", "S3_ts_qt_robust", "S4_ts_qt_hm"]
    slab = ["LiDAR only", "Naive fusion", "Timestamp-aware", "+ quality-weighted gating", "+ consistency monitoring"]
    M = np.array([[d[(d.condition == c) & (d.strategy == s)].p_le.values[0] for c in conds] for s in strats])
    fig, ax = plt.subplots(figsize=(12.0 * CM, 5.6 * CM))
    im = ax.imshow(M, cmap="Greys_r", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6.6,
                    color="white" if M[i, j] < 0.75 else "black")
    ax.set_xticks(range(len(conds)), clab)
    ax.set_yticks(range(len(strats)), slab)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("P(position error ≤ 0.5 m)")
    cb.outline.set_linewidth(0.5)
    ax.set_xticks(np.arange(-0.5, len(conds), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(strats), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    save(fig, out, "fig04_strategy_by_condition")


# ---------------------------------------------------------------- Fig. 5 missing modality
def fig_missing(kit, out):
    d = pd.read_csv(os.path.join(BENCH, "A_missing_modality.csv"))
    spec = [("LEARN_ModDrop", "Learned fusion + modality dropout", "-", "D", "#000000", True),
            ("EARLY", "Early fusion", "--", "o", "#404040", True),
            ("LATE", "Late fusion", "-.", "s", "#7a7a7a", False),
            ("LEARN_noDrop", "Learned fusion (no dropout)", ":", "^", "#7a7a7a", False)]
    fig, ax = plt.subplots(figsize=(COL, 7.2 * CM))
    k = np.array(sorted(d.k.unique()))
    for i, (m, lab, ls, mk, c, filled) in enumerate(spec):
        g = d[d.method == m].groupby("k").p_le
        mean, lo, hi = g.mean().values, g.min().values, g.max().values
        x = k + (i - 1.5) * 0.09
        ax.vlines(x, lo, hi, color=c, lw=0.5, alpha=0.8)
        ax.plot(x, mean, ls, marker=mk, color=c, mfc=c if filled else "white", mec=c, label=lab)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Number of available modalities")
    ax.set_ylabel("P(position error ≤ 0.5 m)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0)
    fig.tight_layout()
    save(fig, out, "fig05_missing_modality")


def fig_missing_matrix(kit, out):
    d = pd.read_csv(os.path.join(BENCH, "A_missing_modality.csv"))
    meth = ["EARLY", "LATE", "LEARN_noDrop", "LEARN_ModDrop"]
    lab = ["Early", "Late", "Learned", "Learned + MD"]
    nice = {"camera": "Camera", "thermal": "Thermal", "lidar": "LiDAR", "gnss": "GNSS-RTK"}
    pats = d.drop_duplicates("pattern").sort_values(["k", "pattern"]).pattern.tolist()
    M = np.array([[d[(d.pattern == p) & (d.method == m)].p_le.values[0] for m in meth] for p in pats])
    fig, ax = plt.subplots(figsize=(COL, 11.5 * CM))
    im = ax.imshow(M, cmap="Greys_r", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] < 0.5 else "black")
    ax.set_xticks(range(len(meth)), lab, fontsize=6.3, rotation=35, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(pats)), [" + ".join(nice[x] for x in p.split("+")) for p in pats], fontsize=6.3)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(meth), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pats), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03, aspect=30)
    cb.set_label("P(position error ≤ 0.5 m)")
    cb.outline.set_linewidth(0.5)
    fig.tight_layout()
    save(fig, out, "figS1_missing_modality_matrix")


# ---------------------------------------------------------------- Fig. 6 coverage
def fig_coverage(kit, out, seed):
    sys.path.insert(0, os.path.join(kit, "p1_bench"))
    import nf_sim as S
    import nf_experiments as E
    layout = S.yard_layout(seed)
    res = 1.0
    xs = np.arange(-50.0, 50.0 + res / 2, res)
    fig, axs = plt.subplots(1, 3, figsize=(PAGE, 6.4 * CM))
    cmap = ListedColormap(["#ececec", "#9a9a9a", "#303030"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 9.5], cmap.N)
    titles = {"N1": "(a) 1 node", "N2": "(b) 2 nodes", "N3": "(c) 3 nodes"}
    for ax, name in zip(axs, ("N1", "N2", "N3")):
        st, (Q, cnt) = E.coverage_stats(layout, S.NODE_SETS[name])
        img = np.full((len(xs), len(xs)), np.nan)
        ix = np.round((Q[:, 0] + 50) / res).astype(int)
        iy = np.round((Q[:, 1] + 50) / res).astype(int)
        img[ix, iy] = cnt
        ax.imshow(img.T, origin="lower", extent=[-50.5, 50.5, -50.5, 50.5], cmap=cmap, norm=norm,
                  interpolation="nearest")
        for b in layout["boxes"]:
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fc="white", ec="black", lw=0.5))
        ax.add_patch(Circle((0, 0), 50, fill=False, lw=0.5, ls="--", ec="black"))
        for s in S.NODE_SETS[name]:
            ax.plot(*s, marker="^", ms=5.5, mfc="white", mec="black", mew=0.9, ls="none")
        ax.set_title(f"{titles[name]}  (≥1: {st['ge1']:.0%}, ≥2: {st['ge2']:.0%})", fontsize=6.8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.plot([-46, -26], [-47, -47], color="black", lw=1.2)
        ax.text(-36, -45.5, "20 m", ha="center", va="bottom", fontsize=6)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(fc="#ececec", ec="0.6", lw=0.4, label="no line of sight"),
               Patch(fc="#9a9a9a", label="1 node"), Patch(fc="#303030", label="≥2 nodes"),
               Patch(fc="white", ec="black", lw=0.5, label="container/ULD stack"),
               Line2D([], [], marker="^", ls="none", mfc="white", mec="black", label="edge node"),
               Line2D([], [], ls="--", color="black", lw=0.5, label="50 m near-field band")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=6.3,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, "fig06_line_of_sight_coverage")


# ---------------------------------------------------------------- scene illustration
def fig_scene(kit, out, seed):
    sys.path.insert(0, os.path.join(kit, "p1_bench"))
    import nf_sim as S
    T = 60.0
    if PROFILE == "port_terminal":
        S.apply_profile("port_terminal")
        T = 50.0
    lab = {k: v[0].upper() + v[1:] for k, v in S.PROFILES[PROFILE]["labels"].items()}
    layout = S.yard_layout(seed)
    rng = np.random.default_rng(seed)
    agents = S.make_agents(rng, layout, T)
    style = {"tractor": ("-", "#000000", lab["tractor"]), "forklift": ("--", "#404040", lab["forklift"]),
             "amr": ("-.", "#7a7a7a", lab["amr"]), "worker": (":", "#000000", lab["worker"])}
    fig, ax = plt.subplots(figsize=(COL, COL))
    for b in layout["boxes"]:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fc="#d9d9d9", ec="black", lw=0.4))
    seen = set()
    for a in agents:
        ls, c, lab = style[a["cls"]]
        p = a["pos"]
        ax.plot(p[:, 0], p[:, 1], ls, color=c, lw=0.9, label=None if a["cls"] in seen else lab)
        ax.plot(*p[-1], marker="o", ms=2.6, color=c)
        seen.add(a["cls"])
    for s in S.NODE_SETS["N2"]:
        ax.plot(*s, marker="^", ms=6, mfc="white", mec="black", mew=0.9, ls="none")
    ax.plot([], [], marker="^", ls="none", mfc="white", mec="black", label="Edge node (two-node layout)")
    ax.plot([], [], marker="o", ls="none", color="black", ms=2.6, label=f"Position at t = {T:.0f} s")
    ax.add_patch(Circle((0, 0), 50, fill=False, lw=0.5, ls="--", ec="black"))
    ax.set_xlim(-52, 52)
    ax.set_ylim(-52, 52)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=6.2, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout()
    save(fig, out, "fig03b_synthetic_yard_scene")


# ---------------------------------------------------------------- sensor models (Fig. 4(b))
def fig_sensor_models(kit, out, calib_path=None, seed=7):
    sys.path.insert(0, os.path.join(kit, "p1_bench"))
    import nf_sim as S
    import nf_experiments as E
    calib = json.load(open(calib_path)) if calib_path else None
    sens = S.apply_calib(S.SENSORS, calib, "vehicle")
    camc = (calib or {}).get("camera_port", {}).get("conditions", {})
    r = np.linspace(0, 70, 281)
    mods = [("lidar", "LiDAR", "#000000"), ("camera", "Camera", "#555555"), ("thermal", "Thermal", "#9a9a9a")]
    conds = [("nominal", "-", "nominal"), ("night", "--", "night"), ("fog", ":", "rain/fog")]
    fig, axs = plt.subplots(1, 3, figsize=(PAGE, 5.8 * CM), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    for m, lab, c in mods:
        p = sens[m]
        for cname, ls, clab in conds:
            mod = dict(S.CONDITIONS[cname].get(m, {}))
            if m == "camera" and cname in camc and "da" in camc[cname]:
                mod["da"] = camc[cname]["da"]
            if cname != "nominal" and not mod:
                continue
            a = p["a"] + mod.get("da", 0.0)
            rmax = mod.get("rmax", p["rmax"])
            pr = 1 / (1 + np.exp(-(a + p["b"] * r)))
            pr[r > rmax] = 0
            sg = (p["s0"] + p["s1"] * r) * mod.get("smul", 1.0)
            axs[0].plot(r, pr, ls, color=c, label=f"{lab}, {clab}")
            axs[1].plot(r, sg, ls, color=c)
    for ax in axs[:2]:
        ax.set_xlabel("Range r (m)")
        ax.set_xlim(0, 70)
    axs[0].set_ylabel("Detection probability p(r)")
    axs[0].set_ylim(0, 1.02)
    axs[0].set_title("(b1) Detection probability, vehicles", loc="left", fontsize=6.6)
    axs[1].set_ylabel("Position noise 1σ (m)")
    axs[1].set_title("(b2) σ(r) = s0 + s1·r (× condition factor)", loc="left", fontsize=6.6)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", bbox_to_anchor=(0.36, 0.02), ncol=4, fontsize=5.8)
    axs[1].text(1, axs[1].get_ylim()[1] * 0.97, "GNSS-RTK: σ = 0.03 m when fitted\n(not on external trucks or workers)",
                va="top", fontsize=5.8, color="0.3")
    layout = S.yard_layout(seed)
    st, (Q, cnt) = E.coverage_stats(layout, S.NODE_SETS["N1"])
    img = np.full((101, 101), np.nan)
    img[np.round(Q[:, 0] + 50).astype(int), np.round(Q[:, 1] + 50).astype(int)] = np.minimum(cnt, 1)
    ax = axs[2]
    ax.imshow(img.T, origin="lower", extent=[-50.5, 50.5, -50.5, 50.5], cmap=ListedColormap(["#e6e6e6", "#8c8c8c"]),
              norm=BoundaryNorm([-.5, .5, 1.5], 2), interpolation="nearest")
    for b in layout["boxes"]:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fc="white", ec="black", lw=0.5))
    ax.add_patch(Circle((0, 0), 50, fill=False, ls="--", lw=0.5))
    ax.plot(*S.NODE_SETS["N1"][0], marker="^", ms=5.5, mfc="white", mec="black")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"(b3) 2-D line of sight, one node ({st['ge1']:.0%} visible)", loc="left", fontsize=6.6)
    ax.plot([-46, -26], [-47, -47], color="k", lw=1.2)
    ax.text(-36, -45.5, "20 m", ha="center", fontsize=6)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig03c_sensor_models")


# ---------------------------------------------------------------- near-miss false alarms
def fig_nearmiss(kit, out):
    d = pd.read_csv(os.path.join(SIM, "e3_nearmiss.csv"))
    strats = ["S0_lidar_only", "S1_naive", "S2_ts_aware", "S3_ts_qt_robust", "S4_ts_qt_hm"]
    slab = ["LiDAR only", "Naive fusion", "Timestamp-aware", "+ quality gating", "+ consistency monitor"]
    fig, ax = plt.subplots(figsize=(COL, 5.6 * CM))
    for j, (cond, lab, mk) in enumerate((("C3_multipath", "Multipath", "o"), ("C5_compound", "Compound", "s"))):
        g = d[d.condition == cond].set_index("strategy").loc[strats]
        y = np.arange(len(strats)) + (j - 0.5) * 0.22
        ax.errorbar(g.false_alarm_rate, y, xerr=[g.false_alarm_rate - g.fa_ci_lo, g.fa_ci_hi - g.false_alarm_rate],
                    fmt=mk, color=GRAYS[j * 2], mfc="white" if j else GRAYS[0], capsize=1.8, elinewidth=0.6,
                    label=lab, ls="none")
    ax.set_yticks(range(len(strats)), slab)
    ax.invert_yaxis()
    ax.set_xlabel("False-alarm rate (95% Wilson CI)")
    ax.set_xlim(0, 0.7)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, out, "figS2_nearmiss_false_alarm")


# ---------------------------------------------------------------- Table 4 configuration summary
def table_config(kit, out):
    s = json.load(open(os.path.join(BENCH, "summary.json")))
    rows = []
    for n in ("N1", "N2", "N3"):
        c = s["C_coverage"][n]
        rows.append(dict(item="Edge nodes", setting=n, metric="LoS coverage >=1 node / >=2 nodes",
                         value=f"{c['ge1']:.0%} / {c['ge2']:.0%}"))
    for r in s["C_sync"]:
        rows.append(dict(item="Clock sync", setting=f"{r['clock']} (±{r['offset_range_ms']} ms)",
                         metric="P(e<=0.5 m) / p95 error (m)", value=f"{r['p_le']:.3f} / {r['p95']:.3f}"))
    pd.DataFrame(rows).to_csv(os.path.join(out, "table4_configuration_summary.csv"), index=False)
    print("wrote table4_configuration_summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=".")
    ap.add_argument("--out", default="nearfield_figures")
    ap.add_argument("--seed", type=int, default=7, help="layout seed used by nf_experiments (summary.json)")
    ap.add_argument("--bench-dir", default=None, help="results of nf_experiments.py (default p1_bench/results_bench)")
    ap.add_argument("--sim-dir", default=None, help="results of p1_nearfield_fusion_sim.py (default p1_sim/results)")
    ap.add_argument("--profile", default="airport_yard", choices=["airport_yard", "port_terminal"], help="class names in legends")
    ap.add_argument("--calib", default=None, help="calibration JSON used by the run (draws the calibrated sensor curves)")
    a = ap.parse_args()
    global BENCH, SIM, PROFILE
    PROFILE = a.profile
    BENCH = a.bench_dir or os.path.join(a.kit, "p1_bench/results_bench")
    SIM = a.sim_dir or os.path.join(a.kit, "p1_sim/results")
    os.makedirs(a.out, exist_ok=True)
    fig_latency(a.kit, a.out)
    fig_strategy(a.kit, a.out)
    fig_missing(a.kit, a.out)
    fig_missing_matrix(a.kit, a.out)
    fig_coverage(a.kit, a.out, a.seed)
    fig_scene(a.kit, a.out, a.seed)
    fig_sensor_models(a.kit, a.out, a.calib, a.seed)
    fig_nearmiss(a.kit, a.out)
    table_config(a.kit, a.out)


if __name__ == "__main__":
    main()
