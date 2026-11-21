#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision2_experiments.py - analyses requested in the second review of the APJCRI manuscript (IR2782).

Parts (run all with --part all, or one at a time):
  seeds      per-seed results of the six methods: is the ranking the same in every seed (each seed has its own
             stack layout and trained models)? Episode-level means with bootstrap CIs; computational cost of each
             method on the same CPU (training time, time per estimate).                              [R2-8, R2-5]
  tuning     baselines tuned on training episodes (Kalman process-noise scale, gate threshold, track timeout of
             late fusion and covariance intersection), then compared on the test episodes with learned fusion,
             whose hyperparameters were fixed in advance.                                             [R2-5]
  stress     simultaneous degradation and correlated failures on the test episodes (models trained under the
             standard conditions): all episodes in rain/fog; node outage (all infrastructure sensors drop together);
             compound (rain/fog + node outage + latency x2 + GNSS-RTK float 0.75 + NTP clock).        [R2-6]
  variation  sensor placement (three node positions) and occlusion (sparse / standard / dense stack gaps);
             does the ranking of the four fusion structures hold?                                     [R2-6]
  calib      (a) bootstrap intervals of the nuScenes detection-curve coefficients (needs --nusc-root);
             (b) ranking under perturbed calibrated coefficients (intercept +-0.5 logit, slope x0.8 / x1.2).  [R2-4]

Usage (repository root):
  python p1_bench/revision2_experiments.py --part all --calib calibration/calib_L3.json --out results/revision2 \
         --nusc-root ~/data/nuscenes
  python p1_bench/revision2_experiments.py --part all --quick --calib calibration/calib_L3.json --out /tmp/rev2_quick
"""
import argparse
import contextlib
import copy
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nf_experiments as E          # noqa: E402
import nf_fusion as F               # noqa: E402
import nf_sim as S                  # noqa: E402
import revision_experiments as R    # noqa: E402  (hard-gate filter, covariance intersection, statistics helpers)

PATTERNS, FULL, NOGNSS = R.PATTERNS, R.FULL, R.NOGNSS
E_BUDGET = 0.5
STRUCT = ["EARLY", "LATE", "LEARN_noDrop", "LEARN_ModDrop"]          # the four structures of the paper
ALL6 = ["EARLY", "EARLY_HARD", "LATE", "CI", "LEARN_noDrop", "LEARN_ModDrop"]
INFRA = ("lidar", "camera", "thermal")


# ------------------------------------------------------------------ scene / series builders
def layout_variant(seed, gap=(3.0, 6.0), half=50.0):
    """Same construction as nf_sim.yard_layout (identical for gap=(3, 6)); gap sets the spacing between stacks."""
    rng = np.random.default_rng(seed)
    boxes = []
    for row_y in (-30.0, -16.0, 16.0, 30.0):
        x = -44.0 + rng.uniform(0, 3)
        while x < 40.0:
            L, W = rng.uniform(6.0, 12.5), 2.6
            boxes.append((x, row_y - W / 2, min(x + L, 46.0), row_y + W / 2))
            x += L + rng.uniform(*gap)
    for cx, cy in ((-22.0, 3.0), (24.0, -4.0)):
        boxes.append((cx - 2.0, cy - 1.6, cx + 2.0, cy + 1.6))
    return dict(name="yard", half=half, boxes=np.array(boxes, float))


def make_series(seed, ep, layout, T, node, cond=None, clock="perfect", node_outage=0.0):
    """Vehicle series of one episode. With the defaults this reproduces nf_experiments.vehicle_series exactly
    (same random streams). cond forces one condition; node_outage removes ALL infrastructure measurements
    (LiDAR, camera, thermal) of every object in common intervals (correlated failure of the edge node)."""
    rng = np.random.default_rng([seed, 11, ep])
    c0 = str(rng.choice(["nominal", "night", "fog"], p=[0.6, 0.2, 0.2]))
    c = cond or c0
    agents = S.make_agents(rng, layout, T)
    ev = np.arange(3.0, T - 0.5, 0.1)
    outs = S._intervals(np.random.default_rng([seed, 17, ep]), T, node_outage, 3.0) if node_outage > 0 else []
    out = []
    for a in agents:
        if a["cls"] not in ("tractor", "forklift"):
            continue
        meas = S.sense_agent(rng, a, node, layout, T, c, clock, 0.0, E.CALIB)
        if outs:
            for m in INFRA:
                d = meas.get(m)
                if d is None:
                    continue
                keep = ~S._inside(d["t_cap"], outs)
                meas[m] = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim == 1 else v) for k, v in d.items()}
        q = S.CLASSES[a["cls"]]["q_sig"] ** 2
        out.append(dict(ep=ep, cls=a["cls"], cond=c, meas=meas, q=q, ev=ev, node=np.array(node, float),
                        tracks=F.modality_tracks(meas, ev, q),
                        truth=a["pos"][np.round(ev / a["dt"]).astype(int)]))
    return out


def build(seed, eps, layout, T, node, **kw):
    return [s for ep in eps for s in make_series(seed, ep, layout, T, node, **kw)]


def train_models(train, seed):
    """Same order and random stream as revision_experiments.run_core -> identical models."""
    rng = np.random.default_rng([seed, 99])
    t0 = time.perf_counter()
    Xn, Yn = F.build_training(train, False, rng)
    m_nd = F.train_mlp(Xn, Yn, seed)
    t1 = time.perf_counter()
    Xd, Yd = F.build_training(train, True, rng)
    m_md = F.train_mlp(Xd, Yd, seed)
    t2 = time.perf_counter()
    return dict(LEARN_noDrop=m_nd, LEARN_ModDrop=m_md), dict(noDrop_s=t1 - t0, ModDrop_s=t2 - t1,
                                                              noDrop_n=int(len(Xn)), ModDrop_n=int(len(Xd)))


@contextlib.contextmanager
def gate(chi2):
    old = F.CHI2_99
    F.CHI2_99 = chi2
    try:
        yield
    finally:
        F.CHI2_99 = old


def _err(e, tr):
    d = np.hypot(e[:, 0] - tr[:, 0], e[:, 1] - tr[:, 1])
    return np.where(np.isfinite(d), d, np.inf)


def evaluate(test, models, methods, tuned=None):
    """Hits and counts per method, sensor subset and episode. tuned: {method: dict(q=, chi2=, age=)} for the
    model-based methods (default: q scale 1, chi2 9.21, timeout 1 s). The evaluated time steps are those with a
    valid default late-fusion estimate, as in the main experiment."""
    tuned = tuned or {}
    eps = sorted({s["ep"] for s in test})
    ei = {e: i for i, e in enumerate(eps)}
    H = {m: np.zeros((len(PATTERNS), len(eps))) for m in methods}
    N = {m: np.zeros((len(PATTERNS), len(eps))) for m in methods}
    for s in test:
        alt = {}

        def tracks_for(cfg):
            key = (cfg.get("q", 1.0), cfg.get("chi2", R.F.CHI2_99))
            if key == (1.0, F.CHI2_99):
                return s["tracks"]
            if key not in alt:
                with gate(key[1]):
                    alt[key] = F.modality_tracks(s["meas"], s["ev"], s["q"] * key[0])
            return alt[key]
        for pi, p in enumerate(PATTERNS):
            mask = set(p)
            late0, good = F.late_fuse(s["tracks"], mask)
            if not good.any():
                continue
            tr = s["truth"][good]
            for m in methods:
                cfg = tuned.get(m, {})
                if m in ("EARLY", "EARLY_HARD"):
                    with gate(cfg.get("chi2", F.CHI2_99)):
                        e = F.run_filter(F.merge(s["meas"], p), R.HARD_CFG if m == "EARLY_HARD" else F.EARLY_CFG,
                                         s["ev"], s["q"] * cfg.get("q", 1.0))[good, :2]
                elif m == "LATE":
                    e = (late0 if not cfg else F.late_fuse(tracks_for(cfg), mask, cfg.get("age", 1.0))[0])[good, :2]
                elif m == "CI":
                    e = R.ci_fuse(tracks_for(cfg), mask, cfg.get("age", 1.0))[0][good]
                else:
                    e = F.learned_fuse(models[m], s["tracks"], mask, s["node"])[0][good]
                d = _err(np.asarray(e), tr)
                H[m][pi, ei[s["ep"]]] += float(np.sum(d <= E_BUDGET))
                N[m][pi, ei[s["ep"]]] += d.size
    out = {}
    for m in methods:
        with np.errstate(invalid="ignore", divide="ignore"):
            P = H[m].sum(1) / N[m].sum(1)
            Pe = np.nanmean(H[m] / N[m], axis=0)
        fi, ni = PATTERNS.index(FULL), PATTERNS.index(NOGNSS)
        out[m] = dict(macro=float(np.nanmean(P)), per_episode=Pe, gnss_dep=float(P[fi] - P[ni]),
                      one_sensor=float(np.nanmean([P[i] for i, p in enumerate(PATTERNS) if len(p) == 1])))
    return out


def _rank(vals):
    """Rank order of methods by value (1 = best)."""
    order = sorted(vals, key=vals.get, reverse=True)
    return {m: i + 1 for i, m in enumerate(order)}


# ------------------------------------------------------------------ seeds (+ computational cost)
def part_seeds(seeds, a, out):
    rows, pe, cost_rows = [], {m: [] for m in ALL6}, []
    for seed in seeds:
        t0 = time.time()
        layout, node = S.yard_layout(seed), S.NODE_SETS["N1"][0]
        train = build(seed, range(a.n_train), layout, a.T, node)
        test = build(seed, range(1000, 1000 + a.n_test), layout, a.T, node)
        models, tt = train_models(train, seed)
        res = evaluate(test, models, ALL6)
        rk = _rank({m: res[m]["macro"] for m in ALL6})
        for m in ALL6:
            rows.append(dict(seed=seed, method=m, macro=round(res[m]["macro"], 4), rank=rk[m],
                             episode_mean=round(float(np.nanmean(res[m]["per_episode"])), 4)))
            pe[m].append(res[m]["per_episode"])
        print(f"[seeds] seed {seed}: " + "  ".join(f"{m} {res[m]['macro']:.3f}" for m in ALL6) + f"  ({time.time() - t0:.0f}s)", flush=True)
        if seed == seeds[0]:                       # computational cost on the same CPU (first seed)
            sub = test[:min(len(test), 24)]
            n_est = sum(len(s["ev"]) for s in sub) * len(PATTERNS)
            t = {}
            c0 = time.perf_counter()
            for s in sub:
                F.modality_tracks(s["meas"], s["ev"], s["q"])
            t_tracks = time.perf_counter() - c0
            for m in ALL6:
                c0 = time.perf_counter()
                for s in sub:
                    for p in PATTERNS:
                        mask = set(p)
                        if m in ("EARLY", "EARLY_HARD"):
                            F.run_filter(F.merge(s["meas"], p), R.HARD_CFG if m == "EARLY_HARD" else F.EARLY_CFG, s["ev"], s["q"])
                        elif m == "LATE":
                            F.late_fuse(s["tracks"], mask)
                        elif m == "CI":
                            R.ci_fuse(s["tracks"], mask)
                        else:
                            F.learned_fuse(models[m], s["tracks"], mask, s["node"])
                t[m] = time.perf_counter() - c0 + (0.0 if m.startswith("EARLY") else t_tracks)
            for m in ALL6:
                cost_rows.append(dict(method=m, ms_per_series_all_subsets=round(1000 * t[m] / len(sub), 1),
                                      us_per_estimate=round(1e6 * t[m] / n_est, 1),
                                      train_s=round(tt["noDrop_s"], 1) if m == "LEARN_noDrop" else round(tt["ModDrop_s"], 1) if m == "LEARN_ModDrop" else 0.0,
                                      train_samples=tt["noDrop_n"] if m == "LEARN_noDrop" else tt["ModDrop_n"] if m == "LEARN_ModDrop" else 0,
                                      tuned_parameters=("MLP weights (hyperparameters fixed in advance)" if m.startswith("LEARN") else "none (defaults)")))
            R._write_csv(os.path.join(out, "cost_methods.csv"), cost_rows)
    R._write_csv(os.path.join(out, "core_by_seed.csv"), rows)
    order0 = None
    same_order, top_all = True, True
    for seed in seeds:
        o = [r["method"] for r in sorted((x for x in rows if x["seed"] == seed), key=lambda x: x["rank"])]
        order0 = order0 or o
        same_order &= (o == order0)
        top_all &= (o[0] == "LEARN_ModDrop")
        s4 = [m for m in o if m in STRUCT]
        print(f"[seeds] seed {seed} order: {' > '.join(o)}", flush=True)
    s4_same = len({tuple(m for m in [r["method"] for r in sorted((x for x in rows if x["seed"] == sd), key=lambda x: x["rank"])] if m in STRUCT) for sd in seeds}) == 1
    summ = dict(seeds=seeds, six_method_order_identical_in_all_seeds=bool(same_order),
                four_structure_order_identical_in_all_seeds=bool(s4_same),
                ModDrop_first_in_all_seeds=bool(top_all), methods={})
    for m in ALL6:
        v = np.concatenate(pe[m])
        mac = [r["macro"] for r in rows if r["method"] == m]
        summ["methods"][m] = dict(seed_mean=round(float(np.mean(mac)), 4), seed_sd=round(float(np.std(mac)), 4),
                                  episode_mean=round(float(np.nanmean(v)), 4), episode_ci95=[round(x, 4) for x in R._boot(v)],
                                  n_episodes=int(np.isfinite(v).sum()))
    json.dump(summ, open(os.path.join(out, "core_by_seed_summary.json"), "w"), indent=2)
    print("[seeds] summary", json.dumps({k: v for k, v in summ.items() if k != "methods"}), flush=True)


# ------------------------------------------------------------------ tuning of the baselines
def part_tuning(seeds, a, out):
    QS = (0.5, 1.0, 2.0) if a.quick else (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
    CH = (9.2103,) if a.quick else (5.9915, 9.2103, 13.8155, 18.4207)   # chi-square (2 dof) at 95, 99, 99.9, 99.99 %
    AG = (0.5, 1.0, 2.0) if a.quick else (0.5, 1.0, 2.0, 4.0)
    rows, pe_all = [], {}
    for seed in seeds:
        t0 = time.time()
        layout, node = S.yard_layout(seed), S.NODE_SETS["N1"][0]
        train = build(seed, range(a.n_train), layout, a.T, node)
        test = build(seed, range(1000, 1000 + a.n_test), layout, a.T, node)
        models, _ = train_models(train, seed)
        tune = [s for s in train if s["ep"] < a.n_tune]                 # tuning uses training episodes only
        goods = [[F.late_fuse(s["tracks"], set(p))[1] for p in PATTERNS] for s in tune]

        def score(fn):
            h, n = np.zeros(len(PATTERNS)), np.zeros(len(PATTERNS))
            for si, s in enumerate(tune):
                for pi, p in enumerate(PATTERNS):
                    g = goods[si][pi]
                    if not g.any():
                        continue
                    d = _err(np.asarray(fn(si, s, p))[g], s["truth"][g])
                    h[pi] += np.sum(d <= E_BUDGET); n[pi] += d.size
            with np.errstate(invalid="ignore", divide="ignore"):
                return float(np.nanmean(h / n))
        sc = {"EARLY": {}, "LATE": {}, "CI": {}}
        for qs in QS:
            for c2 in CH:
                with gate(c2):
                    sc["EARLY"][(qs, c2, None)] = score(lambda si, s, p: F.run_filter(F.merge(s["meas"], p), F.EARLY_CFG, s["ev"], s["q"] * qs)[:, :2])
                    trs = [F.modality_tracks(s["meas"], s["ev"], s["q"] * qs) for s in tune]
                for ag in AG:
                    sc["LATE"][(qs, c2, ag)] = score(lambda si, s, p: F.late_fuse(trs[si], set(p), ag)[0][:, :2])
                    sc["CI"][(qs, c2, ag)] = score(lambda si, s, p: R.ci_fuse(trs[si], set(p), ag)[0])
        best = {m: max(v, key=v.get) for m, v in sc.items()}
        tuned = {m: dict(q=b[0], chi2=b[1], **({} if b[2] is None else dict(age=b[2]))) for m, b in best.items()}
        base = evaluate(test, models, ["EARLY", "LATE", "CI", "LEARN_noDrop", "LEARN_ModDrop"])
        tun = evaluate(test, models, ["EARLY", "LATE", "CI"], tuned)
        for m in ("EARLY", "LATE", "CI"):
            rows.append(dict(seed=seed, method=m, default=round(base[m]["macro"], 4), tuned=round(tun[m]["macro"], 4),
                             q_scale=best[m][0], chi2=round(best[m][1], 2), timeout_s=best[m][2],
                             tuning_score_default=round(sc[m][(1.0, 9.2103, None if m == "EARLY" else 1.0)], 4),
                             tuning_score_best=round(sc[m][best[m]], 4),
                             at_grid_edge="/".join(n for n, v, g in (("q", best[m][0], QS), ("chi2", best[m][1], CH), ("timeout", best[m][2], AG))
                                                   if v is not None and v in (g[0], g[-1])) or "no"))
            pe_all.setdefault(m + "_tuned", []).append(tun[m]["per_episode"])
            pe_all.setdefault(m, []).append(base[m]["per_episode"])
        for m in ("LEARN_noDrop", "LEARN_ModDrop"):
            rows.append(dict(seed=seed, method=m, default=round(base[m]["macro"], 4), tuned=None))
            pe_all.setdefault(m, []).append(base[m]["per_episode"])
        print(f"[tuning] seed {seed}: " + "  ".join(f"{m} {base[m]['macro']:.3f}->{tun[m]['macro']:.3f} {best[m]}" for m in ("EARLY", "LATE", "CI"))
              + f"  ModDrop {base['LEARN_ModDrop']['macro']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        R._write_csv(os.path.join(out, "tuning_baselines.csv"), rows)
    ref = np.concatenate(pe_all["LEARN_ModDrop"])
    pv = {k: R._wilcoxon(ref, np.concatenate(v)) for k, v in pe_all.items() if k.endswith("_tuned")}
    summ = {}
    for m in ("EARLY", "LATE", "CI"):
        d = [r["default"] for r in rows if r["method"] == m]
        t = [r["tuned"] for r in rows if r["method"] == m]
        diff = ref - np.concatenate(pe_all[m + "_tuned"])
        summ[m] = dict(default_mean=round(float(np.mean(d)), 4), tuned_mean=round(float(np.mean(t)), 4),
                       tuned_sd=round(float(np.std(t)), 4), gain=round(float(np.mean(t) - np.mean(d)), 4),
                       ModDrop_minus_tuned_episode_mean=round(float(np.nanmean(diff)), 4),
                       ModDrop_minus_tuned_ci95=[round(x, 4) for x in R._boot(diff)],
                       ModDrop_better_in_every_seed=bool(all(rm["default"] > rt["tuned"] for rm, rt in zip(
                           [r for r in rows if r["method"] == "LEARN_ModDrop"], [r for r in rows if r["method"] == m]))))
    summ["LEARN_ModDrop"] = dict(default_mean=round(float(np.mean([r["default"] for r in rows if r["method"] == "LEARN_ModDrop"])), 4))
    summ["holm_p_ModDrop_vs_tuned"] = R._holm(pv)
    summ["grid"] = dict(q_scale=QS, chi2=CH, timeout_s=AG, tuning_episodes=a.n_tune,
                        note="baselines tuned on training episodes; learned-fusion hyperparameters fixed in advance (not tuned)")
    json.dump(summ, open(os.path.join(out, "tuning_summary.json"), "w"), indent=2)
    print("[tuning] summary", json.dumps(summ), flush=True)


# ------------------------------------------------------------------ stress: simultaneous degradation, correlated failure
SCEN = [("standard", dict(), dict()),
        ("all rain/fog", dict(cond="fog"), dict()),
        ("node outage 20%", dict(node_outage=0.2), dict()),
        ("compound", dict(cond="fog", node_outage=0.2, clock="ntp"), dict(latency=2.0, gnss_float=0.75))]


def part_stress(seeds, a, out):
    meth = ["EARLY", "LATE", "CI", "LEARN_noDrop", "LEARN_ModDrop"]
    acc = {name: {m: dict(macro=[], pe=[], dep=[], one=[]) for m in meth} for name, _, _ in SCEN}
    for seed in seeds:
        t0 = time.time()
        layout, node = S.yard_layout(seed), S.NODE_SETS["N1"][0]
        train = build(seed, range(a.n_train), layout, a.T, node)
        models, _ = train_models(train, seed)                            # trained under the standard conditions
        for name, kw, pert in SCEN:
            with R.perturbed(**dict(pert)):
                test = build(seed, range(1000, 1000 + a.n_test), layout, a.T, node, **kw)
            res = evaluate(test, models, meth)
            for m in meth:
                acc[name][m]["macro"].append(res[m]["macro"]); acc[name][m]["pe"].append(res[m]["per_episode"])
                acc[name][m]["dep"].append(res[m]["gnss_dep"]); acc[name][m]["one"].append(res[m]["one_sensor"])
        print(f"[stress] seed {seed}: " + " | ".join(f"{name}: " + " ".join(f"{acc[name][m]['macro'][-1]:.3f}" for m in meth) for name, _, _ in SCEN)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    b4 = [float(np.mean(acc["standard"][m]["macro"])) for m in STRUCT]
    for name, _, _ in SCEN:
        mm = {m: float(np.mean(acc[name][m]["macro"])) for m in meth}
        others = [m for m in meth if m != "LEARN_ModDrop"]
        runner = max(others, key=mm.get)
        diff = np.concatenate(acc[name]["LEARN_ModDrop"]["pe"]) - np.concatenate(acc[name][runner]["pe"])
        ci = R._boot(diff)
        row = dict(scenario=name, **{m: round(mm[m], 3) for m in meth},
                   gnss_dep_ModDrop=round(float(np.mean(acc[name]["LEARN_ModDrop"]["dep"])), 3),
                   one_sensor_ModDrop=round(float(np.mean(acc[name]["LEARN_ModDrop"]["one"])), 3),
                   top=max(mm, key=mm.get), runner_up=runner,
                   ModDrop_minus_runner_up=round(float(np.nanmean(diff)), 3), diff_ci95=f"{ci[0]:.3f}-{ci[1]:.3f}",
                   wilcoxon_p=R._wilcoxon(np.concatenate(acc[name]["LEARN_ModDrop"]["pe"]), np.concatenate(acc[name][runner]["pe"])),
                   ModDrop_first_in_every_seed=bool(all(acc[name]["LEARN_ModDrop"]["macro"][i] > max(acc[name][m]["macro"][i] for m in others)
                                                        for i in range(len(seeds)))),
                   kendall_tau_4_structures=round(R._kendall(b4, [mm[m] for m in STRUCT]), 3))
        rows.append(row)
        print(f"[stress] {name}: {row}", flush=True)
    R._write_csv(os.path.join(out, "stress_scenarios.csv"), rows)


# ------------------------------------------------------------------ variation: sensor placement, occlusion
VARI = [("standard (node at centre; gaps 3-6 m)", (0.0, 0.0), (3.0, 6.0)),
        ("node between stack rows (-30; 23)", (-30.0, 23.0), (3.0, 6.0)),
        ("node at lane side (30; 8)", (30.0, 8.0), (3.0, 6.0)),
        ("sparse stacks (gaps 8-12 m)", (0.0, 0.0), (8.0, 12.0)),
        ("dense stacks (gaps 1-2 m)", (0.0, 0.0), (1.0, 2.0))]


def part_variation(seeds, a, out):
    rows, base = [], None
    for name, node, gap in VARI:
        t0 = time.time()
        mm, per_seed_top, los = {m: [] for m in STRUCT}, [], []
        for seed in seeds:
            layout = layout_variant(seed, gap)
            train = build(seed, range(a.n_train_s), layout, a.T, node)
            test = build(seed, range(1000, 1000 + a.n_test_s), layout, a.T, node)
            models, _ = train_models(train, seed)
            res = evaluate(test, models, STRUCT)
            for m in STRUCT:
                mm[m].append(res[m]["macro"])
            per_seed_top.append(max(STRUCT, key=lambda m: res[m]["macro"]))
            g = np.arange(-48, 49, 2.0)
            P = np.array([(x, y) for x in g for y in g])
            free = ~np.array([S._inside_any(p, layout["boxes"], 0.5) for p in P])
            los.append(float(np.mean(S.los_many(np.asarray(node, float), P[free], layout["boxes"]))))
        avg = {m: float(np.mean(v)) for m, v in mm.items()}
        base = base or avg
        rows.append(dict(setting=name, **{m: round(avg[m], 3) for m in STRUCT}, line_of_sight_share=round(float(np.mean(los)), 2),
                         kendall_tau=round(R._kendall([base[m] for m in STRUCT], [avg[m] for m in STRUCT]), 3),
                         top=max(avg, key=avg.get), ModDrop_first_in_every_seed=bool(all(t == "LEARN_ModDrop" for t in per_seed_top))))
        print(f"[variation] {name}: {rows[-1]}  ({time.time() - t0:.0f}s)", flush=True)
        R._write_csv(os.path.join(out, "variation_layout.csv"), rows)


# ------------------------------------------------------------------ calibration uncertainty
def _nusc_rows(root, version, rmax=60.0):
    import nf_calib_nuscenes as C
    t = C.load_tables(os.path.expanduser(root), version)
    by = {n: {r["token"]: r for r in t[n]} for n in ("sample", "ego_pose", "calibrated_sensor", "sensor", "scene",
                                                     "instance", "category", "visibility")}
    chan = {sd["token"]: by["sensor"][by["calibrated_sensor"][sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
            for sd in t["sample_data"]}
    ego = {}
    for sd in t["sample_data"]:
        if sd.get("is_key_frame") and chan[sd["token"]] == "LIDAR_TOP":
            ego[sd["sample_token"]] = by["ego_pose"][sd["ego_pose_token"]]["translation"]
    rows = []
    for ann in t["sample_annotation"]:
        st = ann["sample_token"]
        if st not in ego:
            continue
        g = C.group_of(by["category"][by["instance"][ann["instance_token"]]["category_token"]]["name"])
        if g is None:
            continue
        r = math.hypot(ann["translation"][0] - ego[st][0], ann["translation"][1] - ego[st][1])
        if r > rmax:
            continue
        scn = by["sample"][st]["scene_token"]
        lvl = by["visibility"].get(ann.get("visibility_token"), {}).get("level", "")
        rows.append((g, C.condition_of(by["scene"][scn].get("description", "")), r, ann.get("num_lidar_pts", 0) >= 1,
                     lvl in ("v40-60", "v60-80", "v80-100"), scn))
    return rows, C


def part_calib(seeds, a, out, only_ci=False):
    # (a) bootstrap intervals of the fitted coefficients
    if a.nusc_root and os.path.isdir(os.path.join(os.path.expanduser(a.nusc_root), a.nusc_version)):
        rows, C = _nusc_rows(a.nusc_root, a.nusc_version)
        rng = np.random.default_rng(0)
        B = 40 if a.quick else 400
        tab = []
        for key, col in (("lidar", 3), ("camera_vis40", 4)):
            for g in ("vehicle", "pedestrian"):
                sel = [x for x in rows if x[0] == g and x[1] == "day"]
                if len(sel) < 50:
                    continue
                r = np.array([x[2] for x in sel]); y = np.array([x[col] for x in sel], float)
                scn = np.array([x[5] for x in sel]); us = sorted(set(scn))
                a0, b0 = C.fit_logistic(r, y)
                p30 = lambda aa, bb: 1 / (1 + math.exp(-(aa + bb * 30.0)))
                ann, sce = [], []
                for _ in range(B):
                    i = rng.integers(0, len(r), len(r))
                    ann.append(C.fit_logistic(r[i], y[i]))
                    pick = rng.choice(len(us), len(us))
                    idx = np.concatenate([np.where(scn == us[k])[0] for k in pick])
                    sce.append(C.fit_logistic(r[idx], y[idx]))
                ann, sce = np.array(ann), np.array(sce)
                q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
                tab.append(dict(sensor=key, group=g, n=len(sel), scenes=len(us), a=round(a0, 4), b=round(b0, 5),
                                p30=round(p30(a0, b0), 3),
                                a_ci_annotation=q(ann[:, 0]), b_ci_annotation=q(ann[:, 1]),
                                p30_ci_annotation=q([p30(x, z) for x, z in ann]),
                                a_ci_scene=q(sce[:, 0]), b_ci_scene=q(sce[:, 1]), p30_ci_scene=q([p30(x, z) for x, z in sce])))
                print(f"[calib] {key}/{g}: n {len(sel)}, scenes {len(us)}, a {a0:.3f} {tab[-1]['a_ci_scene']}, "
                      f"b {b0:.4f} {tab[-1]['b_ci_scene']}, p30 {tab[-1]['p30']} {tab[-1]['p30_ci_scene']}", flush=True)
        json.dump(tab, open(os.path.join(out, "calib_uncertainty.json"), "w"), indent=2)
    else:
        print("[calib] nuScenes tables not found (--nusc-root); coefficient intervals skipped, perturbation analysis only", flush=True)
    if only_ci:
        return
    # (b) ranking under perturbed calibrated coefficients
    if E.CALIB is None:
        print("[calib] no --calib file: perturbation analysis skipped", flush=True)
        return
    base_cal = copy.deepcopy(E.CALIB)

    def cal(da=0.0, bmul=1.0):
        c = copy.deepcopy(base_cal)
        for key in ("lidar", "camera_vis40"):
            for g in c.get(key, {}):
                d = c[key][g].get("day")
                if d and "a" in d:
                    d["a"] += da; d["b"] *= bmul
        return c
    sets = [("calibrated (L3)", cal()), ("intercept -0.5 logit", cal(da=-0.5)), ("intercept +0.5 logit", cal(da=0.5)),
            ("intercept -1.0 logit", cal(da=-1.0)), ("intercept +1.0 logit", cal(da=1.0)),
            ("slope x1.2 (faster decay)", cal(bmul=1.2)), ("slope x0.8 (slower decay)", cal(bmul=0.8)),
            ("slope x1.4 (faster decay)", cal(bmul=1.4)), ("slope x0.7 (slower decay)", cal(bmul=0.7))]
    if a.quick:
        sets = sets[:3]
    rows, base = [], None
    try:
        for name, c in sets:
            t0 = time.time()
            E.CALIB = c
            mm = {m: [] for m in STRUCT}
            for seed in seeds:
                layout, node = S.yard_layout(seed), S.NODE_SETS["N1"][0]
                train = build(seed, range(a.n_train_s), layout, a.T, node)
                test = build(seed, range(1000, 1000 + a.n_test_s), layout, a.T, node)
                models, _ = train_models(train, seed)
                res = evaluate(test, models, STRUCT)
                for m in STRUCT:
                    mm[m].append(res[m]["macro"])
            avg = {m: float(np.mean(v)) for m, v in mm.items()}
            base = base or avg
            rows.append(dict(setting=name, **{m: round(avg[m], 3) for m in STRUCT},
                             kendall_tau=round(R._kendall([base[m] for m in STRUCT], [avg[m] for m in STRUCT]), 3),
                             top=max(avg, key=avg.get)))
            print(f"[calib] {name}: {rows[-1]}  ({time.time() - t0:.0f}s)", flush=True)
            R._write_csv(os.path.join(out, "calib_perturbation.csv"), rows)
    finally:
        E.CALIB = base_cal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all", help="seeds,tuning,stress,variation,calib or all (calibci: coefficient intervals only)")
    ap.add_argument("--out", default="results/revision2")
    ap.add_argument("--seeds", default="7,11,23,31,47")
    ap.add_argument("--sens-seeds", default="7,11,23", help="seeds for the variation and calibration-perturbation runs")
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--n-train-s", type=int, default=20)
    ap.add_argument("--n-test-s", type=int, default=15)
    ap.add_argument("--n-tune", type=int, default=30, help="training episodes used to tune the baselines (default: all)")
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--profile", default="port_terminal")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--nusc-root", default=os.environ.get("NUSC_ROOT", "~/data/nuscenes"), help="folder that contains v1.0-mini/")
    ap.add_argument("--nusc-version", default="v1.0-mini")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    S.apply_profile(a.profile)
    if a.calib:
        E.CALIB = json.load(open(a.calib, encoding="utf-8"))
    seeds = [int(x) for x in a.seeds.split(",")]
    sseeds = [int(x) for x in a.sens_seeds.split(",")]
    if a.quick:
        seeds, sseeds = seeds[:1], sseeds[:1]
        a.n_train, a.n_test, a.n_train_s, a.n_test_s, a.n_tune, a.T = 3, 2, 3, 2, 2, 30.0
    os.makedirs(a.out, exist_ok=True)
    parts = ["seeds", "tuning", "stress", "variation", "calib"] if a.part == "all" else a.part.split(",")
    json.dump(dict(vars(a), started=time.strftime("%Y-%m-%d %H:%M")), open(os.path.join(a.out, f"run_info_{'_'.join(parts)}.json"), "w"), indent=2)
    for p in parts:
        t0 = time.time()
        {"seeds": lambda: part_seeds(seeds, a, a.out), "tuning": lambda: part_tuning(seeds, a, a.out),
         "stress": lambda: part_stress(seeds, a, a.out), "variation": lambda: part_variation(sseeds, a, a.out),
         "calib": lambda: part_calib(sseeds, a, a.out),
         "calibci": lambda: part_calib(sseeds, a, a.out, only_ci=True)}[p]()
        print(f"== {p} finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
