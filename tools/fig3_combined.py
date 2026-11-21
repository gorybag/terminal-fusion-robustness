#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 3 as one row: (a) evaluation scene, (b1) detection probability, (b2) position noise, (b3) line of sight.
Same data and drawing rules as fig_scene() and fig_sensor_models() in make_figures.py; only the arrangement differs.
Usage: python fig3_combined.py --kit . --calib calibration/calib_L3.json --out figures"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import ListedColormap, BoundaryNorm

CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.6, "axes.labelsize": 6.6, "axes.titlesize": 6.4,
    "legend.fontsize": 5.6, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "axes.linewidth": 0.6,
    "lines.linewidth": 0.9, "lines.markersize": 3.2, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=".")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.kit, "p1_bench"))
    import nf_sim as S
    import nf_experiments as E
    S.apply_profile("port_terminal")
    T = 50.0
    fig = plt.figure(figsize=(16.0 * CM, 5.3 * CM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.12, 1, 1, 0.95], left=0.06, right=0.99, top=0.925, bottom=0.37, wspace=0.44)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    # ---- (a) scene
    lab = {k: v[0].upper() + v[1:] for k, v in S.PROFILES["port_terminal"]["labels"].items()}
    layout = S.yard_layout(a.seed)
    rng = np.random.default_rng(a.seed)
    agents = S.make_agents(rng, layout, T)
    style = {"tractor": ("-", "#000000", lab["tractor"]), "forklift": ("--", "#404040", lab["forklift"]),
             "amr": ("-.", "#7a7a7a", lab["amr"]), "worker": (":", "#000000", lab["worker"])}
    ax = axs[0]
    for b in layout["boxes"]:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fc="#d9d9d9", ec="black", lw=0.3))
    seen = set()
    for ag in agents:
        ls, c, lb = style[ag["cls"]]
        p = ag["pos"]
        ax.plot(p[:, 0], p[:, 1], ls, color=c, lw=0.8, label=None if ag["cls"] in seen else lb)
        ax.plot(*p[-1], marker="o", ms=2.2, color=c)
        seen.add(ag["cls"])
    for s in S.NODE_SETS["N2"]:
        ax.plot(*s, marker="^", ms=5, mfc="white", mec="black", mew=0.8, ls="none")
    ax.plot([], [], marker="^", ls="none", mfc="white", mec="black", label="Edge node (two-node layout)")
    ax.plot([], [], marker="o", ls="none", color="black", ms=2.2, label=f"Position at t = {T:.0f} s")
    ax.add_patch(Circle((0, 0), 50, fill=False, lw=0.5, ls="--", ec="black"))
    ax.set_xlim(-52, 52); ax.set_ylim(-52, 52); ax.set_aspect("equal")
    ax.set_xticks([-40, -20, 0, 20, 40]); ax.set_yticks([-40, -20, 0, 20, 40])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)", labelpad=1)
    ax.set_title("(a) Evaluation scene", loc="left")
    ha, la = ax.get_legend_handles_labels()
    # ---- (b1), (b2) sensor models
    calib = json.load(open(a.calib)) if a.calib else None
    sens = S.apply_calib(S.SENSORS, calib, "vehicle")
    camc = (calib or {}).get("camera_port", {}).get("conditions", {})
    r = np.linspace(0, 70, 281)
    mods = [("lidar", "LiDAR", "#000000"), ("camera", "Camera", "#555555"), ("thermal", "Thermal", "#9a9a9a")]
    conds = [("nominal", "-", "nominal"), ("night", "--", "night"), ("fog", ":", "rain/fog")]
    for m, lb, c in mods:
        p = sens[m]
        for cname, ls, clab in conds:
            mod = dict(S.CONDITIONS[cname].get(m, {}))
            if m == "camera" and cname in camc and "da" in camc[cname]:
                mod["da"] = camc[cname]["da"]
            if cname != "nominal" and not mod:
                continue
            aa = p["a"] + mod.get("da", 0.0)
            rmax = mod.get("rmax", p["rmax"])
            pr = 1 / (1 + np.exp(-(aa + p["b"] * r)))
            pr[r > rmax] = 0
            sg = (p["s0"] + p["s1"] * r) * mod.get("smul", 1.0)
            axs[1].plot(r, pr, ls, color=c, label=f"{lb}, {clab}")
            axs[2].plot(r, sg, ls, color=c)
    for ax in axs[1:3]:
        ax.set_xlabel("Range r (m)"); ax.set_xlim(0, 70); ax.set_xticks([0, 20, 40, 60])
    axs[1].set_ylabel("Detection probability p(r)", labelpad=1); axs[1].set_ylim(0, 1.02)
    axs[1].set_title("(b1) Detection, vehicles", loc="left")
    axs[2].set_ylabel("Position noise 1σ (m)", labelpad=1)
    axs[2].set_title("(b2) Noise σ = s0 + s1·r", loc="left")
    axs[2].text(1, axs[2].get_ylim()[1] * 0.97, "GNSS-RTK: σ = 0.03 m when fitted\n(not on external trucks or workers)",
                va="top", fontsize=5.0, color="0.3")
    hb, lb_ = axs[1].get_legend_handles_labels()
    # ---- (b3) line of sight
    st, (Q, cnt) = E.coverage_stats(layout, S.NODE_SETS["N1"])
    img = np.full((101, 101), np.nan)
    img[np.round(Q[:, 0] + 50).astype(int), np.round(Q[:, 1] + 50).astype(int)] = np.minimum(cnt, 1)
    ax = axs[3]
    ax.imshow(img.T, origin="lower", extent=[-50.5, 50.5, -50.5, 50.5], cmap=ListedColormap(["#e6e6e6", "#8c8c8c"]),
              norm=BoundaryNorm([-.5, .5, 1.5], 2), interpolation="nearest")
    for b in layout["boxes"]:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fc="white", ec="black", lw=0.4))
    ax.add_patch(Circle((0, 0), 50, fill=False, ls="--", lw=0.5))
    ax.plot(*S.NODE_SETS["N1"][0], marker="^", ms=4.8, mfc="white", mec="black")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"(b3) Line of sight ({st['ge1']:.0%})", loc="left")
    ax.plot([-46, -26], [-47, -47], color="k", lw=1.1)
    ax.text(-36, -45.5, "20 m", ha="center", fontsize=5.4)
    # ---- legends below the panels
    fig.legend(ha, la, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.2), ncol=2, columnspacing=0.8,
               handlelength=1.8, handletextpad=0.5, labelspacing=0.3)
    fig.legend(hb, lb_, frameon=False, loc="upper left", bbox_to_anchor=(0.405, 0.2), ncol=3, columnspacing=0.9,
               handlelength=1.8, handletextpad=0.5, labelspacing=0.3)
    os.makedirs(a.out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"fig03_evaluation_setting_combined.{ext}"))
    print("wrote fig03_evaluation_setting_combined", fig.get_size_inches() * 2.54)


if __name__ == "__main__":
    main()
