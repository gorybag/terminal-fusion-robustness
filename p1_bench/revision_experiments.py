#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision_experiments.py - additional analyses requested in the review of the APJCRI manuscript.

Parts (run all with --part all, or one at a time):
  analytic     response band / latency budget for T_h in 5-15 s, e_budget 0.25-1.0 m, speed limits 20-30 km/h,
               and the size of second-order terms (acceleration, turning) in the latency error      [R1-5, R1-6, R1-7]
  core         vehicles, 15 modality subsets, methods EARLY, EARLY_HARD (hard chi-square gate), LATE, CI (fast
               covariance intersection), LEARN_noDrop, LEARN_ModDrop; P(<=0.5 m), MAE, RMSE, median, p95, max;
               macro averages, GNSS-RTK dependency, episode-bootstrap 95 % CI, paired Wilcoxon + Holm   [R1-12, R1-14, R1-15]
  sensitivity  one-at-a-time perturbation of the assumed parameters (thermal noise and fog effect, LiDAR fog effect,
               GNSS-RTK float/jump rates, sensor latency, clock level); does the method ranking hold?  [R1-8, R2-5, R2-6]
  dropout      modality-dropout keep probability x number of augmented subsets                      [R1-11]
  pairs        all classes, 1 and 2 nodes: class accuracy, vehicle-worker risk-underestimation rate, signed relative-
               distance error, effective latency of naive vs timestamp-aware fusion, with CIs         [R1-6, R1-14, R1-15]
  sync         PTP / NTP / unsynchronized: P(<=0.5 m) and p95 error with CIs                          [R1-15]
  scale_nodes  (batch 2) scene density x1 / x2 and node layouts N1 / N2 / N3: XT, worker, YT accuracy and
               vehicle-worker risk underestimation (all sensors), with CIs                          [R1-13, R1-18]
  examples     (batch 2) supplement figure: XT error through an occlusion gap, and closing-pair distance error
               with and without timestamp alignment                                               [R1-19]

Usage (repository root):
  python p1_bench/revision_experiments.py --part all --calib calibration/calib_L3.json --out results/revision
  python p1_bench/revision_experiments.py --part all --quick --out /tmp/rev_quick         # smoke test (minutes)
"""
import argparse
import contextlib
import copy
import itertools
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nf_experiments as E  # noqa: E402
import nf_fusion as F       # noqa: E402
import nf_sim as S          # noqa: E402

PATTERNS = E.PATTERNS
FULL = tuple(S.MODALITIES)
NOGNSS = tuple(m for m in S.MODALITIES if m != "gnss")
METHODS = ["EARLY", "EARLY_HARD", "LATE", "CI", "LEARN_noDrop", "LEARN_ModDrop"]
E_BUDGET = 0.5

# ------------------------------------------------------------------ baselines
HARD_CFG = dict(F.EARLY_CFG, hard=True)
_orig_apply = F._apply


def _apply_ext(kf, t, m, cfg):
    """Hard chi-square gate (reject outliers) when cfg['hard']; otherwise the original soft gate."""
    if cfg.get("hard") and kf.init:
        zx, zy, srep, snom = m
        R = (srep if cfg["quality"] else snom) ** 2
        kf.predict(t)
        dx, dy = zx - kf.px, zy - kf.py
        if (dx * dx + dy * dy) / (kf.P00 + R) > F.CHI2_99:
            return
        kf.update(dx, dy, R)
        return
    _orig_apply(kf, t, m, cfg)


F._apply = _apply_ext


def ci_fuse(tracks, mask, max_age=1.0):
    """Fast covariance intersection of the per-modality tracks (omega_m proportional to 1 / tr P_m)."""
    n = len(next(v for v in tracks.values() if v is not None))
    num, den = np.zeros((n, 2)), np.zeros(n)
    for m in mask:
        o = tracks.get(m)
        if o is None:
            continue
        ok = np.isfinite(o[:, 0]) & (o[:, 5] <= max_age)
        P = np.maximum(np.nan_to_num(o[:, 4], nan=1.0), 1e-6)
        w = np.where(ok, 1.0 / P, 0.0)                    # omega (unnormalised; cancels below)
        num += (w / P)[:, None] * np.nan_to_num(o[:, 0:2])
        den += w / P
    out = np.full((n, 2), np.nan)
    good = den > 0
    out[good] = num[good] / den[good, None]
    return out, good


# ------------------------------------------------------------------ helpers
def _stats(d):
    d = d[np.isfinite(d)]
    if not d.size:
        return dict(mae=np.nan, rmse=np.nan, median=np.nan, p95=np.nan, max=np.nan)
    return dict(mae=float(d.mean()), rmse=float(np.sqrt(np.mean(d ** 2))), median=float(np.median(d)),
                p95=float(np.percentile(d, 95)), max=float(d.max()))


def _boot(v, B=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return [float("nan"), float("nan")]
    bm = np.random.default_rng(seed).choice(v, (B, v.size), replace=True).mean(1)
    return [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def _holm(pvals):
    keys = sorted(pvals, key=pvals.get)
    out, m, run = {}, len(keys), 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


def _wilcoxon(a, b):
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.all(a[ok] == b[ok]):
        return 1.0
    return float(wilcoxon(a[ok], b[ok]).pvalue)


def _kendall(r1, r2):
    from scipy.stats import kendalltau
    return float(kendalltau(r1, r2).statistic)


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")


@contextlib.contextmanager
def perturbed(**kw):
    """Temporarily change assumed simulator parameters."""
    saved = (copy.deepcopy(S.SENSORS), copy.deepcopy(S.CONDITIONS), S.GNSS_FLOAT_P, S.GNSS_JUMP_P)
    try:
        if "thermal_noise" in kw:
            S.SENSORS["thermal"]["s0"] *= kw["thermal_noise"]; S.SENSORS["thermal"]["s1"] *= kw["thermal_noise"]
        if "thermal_fog_da" in kw:
            S.CONDITIONS["fog"]["thermal"]["da"] = kw["thermal_fog_da"]
        if "lidar_fog_da" in kw:
            S.CONDITIONS["fog"]["lidar"]["da"] = kw["lidar_fog_da"]
        if "lidar_fog_rmax" in kw:
            S.CONDITIONS["fog"]["lidar"]["rmax"] = kw["lidar_fog_rmax"]
        if "latency" in kw:
            for m in S.SENSORS:
                S.SENSORS[m]["lat_mu"] *= kw["latency"]; S.SENSORS[m]["lat_sd"] *= kw["latency"]
        if "gnss_float" in kw:
            S.GNSS_FLOAT_P = kw["gnss_float"]
        if "gnss_jump" in kw:
            S.GNSS_JUMP_P = kw["gnss_jump"]
        yield
    finally:
        S.SENSORS.clear(); S.SENSORS.update(saved[0])
        S.CONDITIONS.clear(); S.CONDITIONS.update(saved[1])
        S.GNSS_FLOAT_P, S.GNSS_JUMP_P = saved[2], saved[3]


# ------------------------------------------------------------------ analytic
def part_analytic(out):
    rows = []
    vw = (1.2, 2.0)
    for kmh in (20, 25, 30):
        v_yt = kmh / 3.6
        for T_h in (5.0, 7.5, 10.0, 12.5, 15.0):
            for e in (0.25, 0.5, 1.0):
                for w in vw:
                    rows.append(dict(speed_limit_kmh=kmh, T_h_s=T_h, e_budget_m=e, worker_mps=w,
                                     band_object_m=round(v_yt * T_h, 1), band_pair_m=round((v_yt + w) * T_h, 1),
                                     tau_max_object_ms=round(1000 * e / v_yt, 1),
                                     tau_max_pair_ms=round(1000 * e / (v_yt + w), 1)))
    _write_csv(os.path.join(out, "analytic_sensitivity.csv"), rows)
    v, tau = 6.9, 0.072
    second = dict(first_order_m=v * tau, accel_term_m_at_1mps2=0.5 * 1.0 * tau ** 2,
                  accel_term_m_at_3mps2=0.5 * 3.0 * tau ** 2, turn_term_m_yaw_0p6=0.5 * v * 0.6 * tau ** 2,
                  stopping_distance_m_25kmh=round(6.9 * 1.5 + 6.9 ** 2 / (2 * 1.5), 1),
                  note="v=6.9 m/s, tau=72 ms; turn term = 0.5*v*omega*tau^2; stopping: 1.5 s reaction, 1.5 m/s^2")
    json.dump(second, open(os.path.join(out, "analytic_second_order.json"), "w"), indent=2)
    print("[analytic] done", json.dumps(second))


# ------------------------------------------------------------------ core (vehicles, 15 subsets)
def run_core(seed, n_train, n_test, T, p_keep=0.6, n_aug=4, clock="perfect", methods=METHODS, keep_errors=True):
    layout, node = S.yard_layout(seed), S.NODE_SETS["N1"][0]
    train = [s for ep in range(n_train) for s in E.vehicle_series(seed, ep, layout, T, node, clock=clock)]
    test = [s for ep in range(1000, 1000 + n_test) for s in E.vehicle_series(seed, ep, layout, T, node, clock=clock)]
    rng = np.random.default_rng([seed, 99])
    models = {}
    if "LEARN_noDrop" in methods:
        X, Y = F.build_training(train, False, rng); models["LEARN_noDrop"] = F.train_mlp(X, Y, seed)
    if "LEARN_ModDrop" in methods:
        X, Y = F.build_training(train, True, rng, n_aug=n_aug, p_keep=p_keep); models["LEARN_ModDrop"] = F.train_mlp(X, Y, seed)
    eps = sorted({s["ep"] for s in test})
    ei = {e: i for i, e in enumerate(eps)}
    H = {m: np.zeros((len(PATTERNS), len(eps))) for m in methods}
    N = {m: np.zeros((len(PATTERNS), len(eps))) for m in methods}
    errs = {(m, pi): [] for m in methods for pi in range(len(PATTERNS))}
    for s in test:
        for pi, p in enumerate(PATTERNS):
            mask = set(p)
            late, good = F.late_fuse(s["tracks"], mask)
            if not good.any():
                continue
            tr = s["truth"][good]
            est = {}
            for m in methods:
                if m == "EARLY":
                    e = F.run_filter(F.merge(s["meas"], p), F.EARLY_CFG, s["ev"], s["q"])[good, :2]
                elif m == "EARLY_HARD":
                    e = F.run_filter(F.merge(s["meas"], p), HARD_CFG, s["ev"], s["q"])[good, :2]
                elif m == "LATE":
                    e = late[good, :2]
                elif m == "CI":
                    e = ci_fuse(s["tracks"], mask)[0][good]
                else:
                    e = F.learned_fuse(models[m], s["tracks"], mask, s["node"])[0][good]
                d = np.hypot(e[:, 0] - tr[:, 0], e[:, 1] - tr[:, 1])
                d = np.where(np.isfinite(d), d, np.inf)
                H[m][pi, ei[s["ep"]]] += float(np.sum(d <= E_BUDGET))
                N[m][pi, ei[s["ep"]]] += d.size
                if keep_errors:
                    errs[(m, pi)].append(d)
    res = dict(seed=seed, eps=eps, methods={})
    for m in methods:
        with np.errstate(invalid="ignore", divide="ignore"):
            P = H[m].sum(1) / N[m].sum(1)                       # per pattern (pooled)
            Pe = np.nanmean(H[m] / N[m], axis=0)                # per episode macro
        full_i, nog_i = PATTERNS.index(FULL), PATTERNS.index(NOGNSS)
        r = dict(macro=float(np.nanmean(P)), per_pattern=P.tolist(), per_episode_macro=Pe.tolist(),
                 gnss_dep=float(P[full_i] - P[nog_i]),
                 one_sensor=float(np.nanmean([P[i] for i, p in enumerate(PATTERNS) if len(p) == 1])))
        if keep_errors:
            pooled = [np.concatenate(errs[(m, i)]) for i in range(len(PATTERNS)) if errs[(m, i)]]
            r["macro_at"] = {str(t): float(np.mean([np.mean(d <= t) for d in pooled])) for t in (0.25, 0.5, 1.0)}
            r["err_full"] = _stats(np.concatenate(errs[(m, full_i)]) if errs[(m, full_i)] else np.array([]))
            st = [_stats(np.concatenate(errs[(m, i)])) for i in range(len(PATTERNS)) if errs[(m, i)]]
            r["err_macro"] = {k: float(np.nanmean([x[k] for x in st])) for k in ("mae", "rmse", "median", "p95", "max")}
        res["methods"][m] = r
    return res


def part_core(seeds, a, out):
    allr = []
    for seed in seeds:
        t0 = time.time()
        r = run_core(seed, a.n_train, a.n_test, a.T)
        allr.append(r)
        print(f"[core] seed {seed}: " + "  ".join(f"{m} {r['methods'][m]['macro']:.3f}" for m in METHODS)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    rows, summ = [], {}
    for m in METHODS:
        mac = [r["methods"][m]["macro"] for r in allr]
        pe = np.concatenate([r["methods"][m]["per_episode_macro"] for r in allr])
        summ[m] = dict(macro_mean=float(np.mean(mac)), macro_sd=float(np.std(mac)), macro_ci95=_boot(pe),
                       gnss_dep=float(np.mean([r["methods"][m]["gnss_dep"] for r in allr])),
                       one_sensor=float(np.mean([r["methods"][m]["one_sensor"] for r in allr])),
                       macro_at={t: float(np.mean([r["methods"][m]["macro_at"][t] for r in allr])) for t in ("0.25", "0.5", "1.0")},
                       err_full={k: float(np.nanmean([r["methods"][m]["err_full"][k] for r in allr]))
                                 for k in ("mae", "rmse", "median", "p95", "max")},
                       err_macro={k: float(np.nanmean([r["methods"][m]["err_macro"][k] for r in allr]))
                                  for k in ("mae", "rmse", "median", "p95", "max")})
        rows.append(dict(method=m, macro=round(summ[m]["macro_mean"], 3), sd=round(summ[m]["macro_sd"], 3),
                         ci_lo=round(summ[m]["macro_ci95"][0], 3), ci_hi=round(summ[m]["macro_ci95"][1], 3),
                         gnss_dep=round(summ[m]["gnss_dep"], 3), one_sensor=round(summ[m]["one_sensor"], 3),
                         **{f"macro_at_{t}m": round(v, 3) for t, v in summ[m]["macro_at"].items()},
                         **{f"full_{k}": round(v, 3) for k, v in summ[m]["err_full"].items()},
                         **{f"macro_{k}": round(v, 3) for k, v in summ[m]["err_macro"].items()}))
    ref = "LEARN_ModDrop"
    pv = {m: _wilcoxon(np.concatenate([r["methods"][ref]["per_episode_macro"] for r in allr]),
                       np.concatenate([r["methods"][m]["per_episode_macro"] for r in allr])) for m in METHODS if m != ref}
    tests = {f"{ref} vs {m}": dict(p=pv[m], p_holm=ph) for m, ph in _holm(pv).items()}
    _write_csv(os.path.join(out, "core_methods.csv"), rows)
    json.dump(dict(summary=summ, tests=tests, seeds=seeds), open(os.path.join(out, "core_summary.json"), "w"), indent=2)
    print("[core] tests", json.dumps(tests))
    return summ


# ------------------------------------------------------------------ sensitivity
PERTURB = [("thermal noise x0.5", dict(thermal_noise=0.5)), ("thermal noise x2", dict(thermal_noise=2.0)),
           ("thermal rain/fog da -0.35", dict(thermal_fog_da=-0.35)), ("thermal rain/fog da -1.4", dict(thermal_fog_da=-1.4)),
           ("LiDAR rain/fog da -1.0", dict(lidar_fog_da=-1.0)), ("LiDAR rain/fog da -3.0", dict(lidar_fog_da=-3.0)),
           ("LiDAR rain/fog range 25 m", dict(lidar_fog_rmax=25.0)), ("LiDAR rain/fog range 45 m", dict(lidar_fog_rmax=45.0)),
           ("GNSS float 0.25", dict(gnss_float=0.25)), ("GNSS float 0.75", dict(gnss_float=0.75)),
           ("GNSS jump 0.02", dict(gnss_jump=0.02)), ("GNSS jump 0.10", dict(gnss_jump=0.10)),
           ("latency x0.5", dict(latency=0.5)), ("latency x2", dict(latency=2.0)),
           ("clock NTP", dict(clock="ntp")), ("clock unsynchronized", dict(clock="none"))]


def part_sensitivity(seeds, a, out):
    meth = ["EARLY", "LATE", "LEARN_noDrop", "LEARN_ModDrop"]          # the four structures of the paper
    rows = []
    base = {}
    for seed in seeds:
        base[seed] = run_core(seed, a.n_train_s, a.n_test_s, a.T, methods=meth, keep_errors=False)
    bm = {m: float(np.mean([base[s]["methods"][m]["macro"] for s in seeds])) for m in meth}
    rows.append(dict(setting="baseline", **{m: round(bm[m], 3) for m in meth},
                     gnss_dep_ModDrop=round(float(np.mean([base[s]["methods"]["LEARN_ModDrop"]["gnss_dep"] for s in seeds])), 3),
                     kendall_tau=1.0, top=max(bm, key=bm.get)))
    for name, kw in PERTURB:
        t0 = time.time()
        clock = kw.pop("clock", "perfect")
        with perturbed(**kw):
            rr = [run_core(seed, a.n_train_s, a.n_test_s, a.T, clock=clock, methods=meth, keep_errors=False) for seed in seeds]
        mm = {m: float(np.mean([r["methods"][m]["macro"] for r in rr])) for m in meth}
        tau = _kendall([bm[m] for m in meth], [mm[m] for m in meth])
        rows.append(dict(setting=name, **{m: round(mm[m], 3) for m in meth},
                         gnss_dep_ModDrop=round(float(np.mean([r["methods"]["LEARN_ModDrop"]["gnss_dep"] for r in rr])), 3),
                         kendall_tau=round(tau, 3), top=max(mm, key=mm.get)))
        print(f"[sens] {name:<28} " + " ".join(f"{m} {mm[m]:.3f}" for m in meth) + f"  tau {tau:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        _write_csv(os.path.join(out, "sensitivity_parameters.csv"), rows)
    return rows


# ------------------------------------------------------------------ dropout sweep
def part_dropout(seeds, a, out):
    rows = []
    for p_keep in (0.4, 0.6, 0.8, 0.9):
        for n_aug in (1, 2, 4, 8):
            t0 = time.time()
            rr = [run_core(seed, a.n_train_s, a.n_test_s, a.T, p_keep=p_keep, n_aug=n_aug,
                           methods=["LEARN_ModDrop"], keep_errors=False) for seed in seeds]
            mac = [r["methods"]["LEARN_ModDrop"]["macro"] for r in rr]
            rows.append(dict(dropout_prob=round(1 - p_keep, 2), p_keep=p_keep, n_aug=n_aug,
                             macro=round(float(np.mean(mac)), 3), sd=round(float(np.std(mac)), 3),
                             one_sensor=round(float(np.mean([r["methods"]["LEARN_ModDrop"]["one_sensor"] for r in rr])), 3),
                             gnss_dep=round(float(np.mean([r["methods"]["LEARN_ModDrop"]["gnss_dep"] for r in rr])), 3)))
            print(f"[dropout] p_drop {1 - p_keep:.1f} n_aug {n_aug}: macro {np.mean(mac):.3f} ({time.time() - t0:.0f}s)", flush=True)
            _write_csv(os.path.join(out, "dropout_sweep.csv"), rows)
    return rows


# ------------------------------------------------------------------ pairs (all classes, collision metrics)
def part_pairs(seeds, a, out):
    meth = ["EARLY", "LATE", "LEARN_noDrop", "LEARN_ModDrop"]
    rows, lat_rows = [], []
    for nodes_name in ("N1", "N2"):
        nodes = S.NODE_SETS[nodes_name]
        acc = {m: dict(cls={c: [] for c in E.ALL_CLASSES}, rho=[], prel=[], signed=[]) for m in meth}
        lat = {tag: dict(e=[], vc=[], ep_tau=[]) for tag in ("naive", "ts_aware")}
        for seed in seeds:
            t0 = time.time()
            layout = S.yard_layout(seed)
            train = [s for ep in range(a.n_train) for s in E.agent_series(seed, ep, layout, a.T, nodes)]
            test = [s for ep in range(3000, 3000 + a.n_test) for s in E.agent_series(seed, ep, layout, a.T, nodes)]
            rng = np.random.default_rng([seed, 98])
            Xn, Yn = F.build_training(train, False, rng); Xd, Yd = F.build_training(train, True, rng)
            models = dict(LEARN_noDrop=F.train_mlp(Xn, Yn, seed), LEARN_ModDrop=F.train_mlp(Xd, Yd, seed))
            est = {}
            for s in test:
                n = len(s["ev"])
                for p in PATTERNS:
                    mask = set(p)
                    late, good = F.late_fuse(s["tracks"], mask)
                    for m in meth:
                        arr = np.full((n, 2), np.nan)
                        if good.any():
                            if m == "EARLY":
                                e = F.run_filter(F.merge(s["meas"], p), F.EARLY_CFG, s["ev"], s["q"])[:, :2]
                            elif m == "LATE":
                                e = late[:, :2]
                            else:
                                e = F.learned_fuse(models[m], s["tracks"], mask, s["node"])[0]
                            arr[good] = np.asarray(e)[good]
                        est[(s["ep"], s["aid"], p, m)] = arr
                for tag, cfg in (("naive", F.NAIVE_CFG), ("ts_aware", F.EARLY_CFG)):
                    est[(s["ep"], s["aid"], tag)] = F.run_filter(F.merge(s["meas"], FULL), cfg, s["ev"], s["q"])[:, :2]
            by_ep = {}
            for s in test:
                by_ep.setdefault(s["ep"], []).append(s)
            for ep, ser in by_ep.items():
                for m in meth:                                         # per-episode macro over patterns
                    for c in E.ALL_CLASSES:
                        vals = []
                        for p in PATTERNS:
                            d = np.concatenate([np.hypot(*(est[(ep, s["aid"], p, m)] - s["truth"]).T) for s in ser if s["cls"] == c] or [np.array([])])
                            if d.size:
                                vals.append(np.mean(np.where(np.isfinite(d), d, np.inf) <= E_BUDGET))
                        if vals:
                            acc[m]["cls"][c].append(float(np.mean(vals)))
                    rho_p, prel_p, sg = [], [], []
                    for p in PATTERNS:
                        over = ncl = ok = seen = 0
                        for i in range(len(ser)):
                            for j in range(i + 1, len(ser)):
                                A_, B_ = ser[i], ser[j]
                                if (A_["cls"] == "worker") + (B_["cls"] == "worker") != 1:
                                    continue
                                dp, dv = B_["truth"] - A_["truth"], B_["vtruth"] - A_["vtruth"]
                                dt_ = np.hypot(dp[:, 0], dp[:, 1])
                                vc = -np.sum(dp * dv, 1) / np.maximum(dt_, 1e-6)
                                de = np.hypot(*(est[(ep, B_["aid"], p, m)] - est[(ep, A_["aid"], p, m)]).T)
                                both, close = np.isfinite(de), dt_ <= E.CLOSE_M
                                e = de - dt_
                                sel = both & close & (vc > 0.2)
                                ncl += sel.sum(); over += (sel & (e > E_BUDGET)).sum()
                                seen += (both & close).sum(); ok += (both & close & (np.abs(e) <= E_BUDGET)).sum()
                                if p == FULL:
                                    sg.append(e[sel])
                        if ncl:
                            rho_p.append(over / ncl)
                        if seen:
                            prel_p.append(ok / seen)
                    if rho_p:
                        acc[m]["rho"].append(float(np.mean(rho_p)))
                    if prel_p:
                        acc[m]["prel"].append(float(np.mean(prel_p)))
                    if sg:
                        acc[m]["signed"].append(np.concatenate(sg))
                for tag in ("naive", "ts_aware"):                     # effective latency on closing pairs
                    ee, vv = [], []
                    for i in range(len(ser)):
                        for j in range(i + 1, len(ser)):
                            A_, B_ = ser[i], ser[j]
                            dp, dv = B_["truth"] - A_["truth"], B_["vtruth"] - A_["vtruth"]
                            dt_ = np.hypot(dp[:, 0], dp[:, 1])
                            vc = -np.sum(dp * dv, 1) / np.maximum(dt_, 1e-6)
                            de = np.hypot(*(est[(ep, B_["aid"], tag)] - est[(ep, A_["aid"], tag)]).T)
                            sel = np.isfinite(de) & (dt_ <= E.CLOSE_M) & (vc > 0.2)
                            ee.append((de - dt_)[sel]); vv.append(vc[sel])
                    ee, vv = np.concatenate(ee), np.concatenate(vv)
                    lat[tag]["e"].append(ee); lat[tag]["vc"].append(vv)
                    if vv.size > 20:
                        lat[tag]["ep_tau"].append(1000 * float(np.sum(ee * vv) / np.sum(vv * vv)))
            print(f"[pairs] {nodes_name} seed {seed} ({time.time() - t0:.0f}s)", flush=True)
        for m in meth:
            sg = np.concatenate(acc[m]["signed"]) if acc[m]["signed"] else np.array([])
            row = dict(nodes=nodes_name, method=m)
            for c in E.ALL_CLASSES:
                v = acc[m]["cls"][c]
                row[f"p_le_{c}"] = round(float(np.mean(v)), 3); ci = _boot(v); row[f"p_le_{c}_ci"] = f"{ci[0]:.3f}-{ci[1]:.3f}"
            ci = _boot(acc[m]["rho"])
            row.update(rho_over=round(float(np.mean(acc[m]["rho"])), 3), rho_over_ci=f"{ci[0]:.3f}-{ci[1]:.3f}",
                       p_rel_le=round(float(np.mean(acc[m]["prel"])), 3),
                       signed_mean_m=round(float(np.mean(sg)), 3) if sg.size else None,
                       signed_median_m=round(float(np.median(sg)), 3) if sg.size else None,
                       signed_p95_m=round(float(np.percentile(sg, 95)), 3) if sg.size else None)
            rows.append(row)
        ref = acc["LEARN_noDrop"]["rho"]
        for m in ("EARLY", "LATE", "LEARN_ModDrop"):
            rows[-4 + meth.index(m)][f"rho_vs_noDrop_wilcoxon_p"] = round(_wilcoxon(ref, acc[m]["rho"]), 4)
        for tag in ("naive", "ts_aware"):
            e, vc = np.concatenate(lat[tag]["e"]), np.concatenate(lat[tag]["vc"])
            ci = _boot(lat[tag]["ep_tau"])
            lat_rows.append(dict(nodes=nodes_name, fusion=tag, n=int(e.size),
                                 tau_eff_ms=round(1000 * float(np.sum(e * vc) / np.sum(vc * vc)), 1),
                                 tau_eff_ci=f"{ci[0]:.1f}-{ci[1]:.1f}", overest_rate=round(float(np.mean(e > E_BUDGET)), 3),
                                 signed_mean_m=round(float(np.mean(e)), 3)))
        _write_csv(os.path.join(out, "pairs_collision.csv"), rows)
        _write_csv(os.path.join(out, "pairs_latency.csv"), lat_rows)
    return rows, lat_rows


# ------------------------------------------------------------------ sync
def part_sync(seeds, a, out):
    rows = []
    for clock in ("ptp", "ntp", "none"):
        p_le, p95 = [], []
        for seed in seeds:
            layout, node = S.yard_layout(seed), S.NODE_SETS["N1"][0]
            for ep in range(2000, 2000 + a.n_test):
                for s in E.vehicle_series(seed, ep, layout, a.T, node, clock=clock):
                    e = F.run_filter(F.merge(s["meas"], FULL), F.EARLY_CFG, s["ev"], s["q"])[:, :2]
                    d = np.hypot(*(e - s["truth"]).T)
                    d = d[np.isfinite(d)]
                    if d.size:
                        p_le.append(float(np.mean(d <= E_BUDGET))); p95.append(float(np.percentile(d, 95)))
        rows.append(dict(clock=clock, p_le=round(float(np.mean(p_le)), 3), p_le_ci="%.3f-%.3f" % tuple(_boot(p_le)),
                         p95_m=round(float(np.mean(p95)), 3), p95_ci="%.3f-%.3f" % tuple(_boot(p95)), n_series=len(p_le)))
        print(f"[sync] {clock}: {rows[-1]}", flush=True)
    _write_csv(os.path.join(out, "sync_levels.csv"), rows)
    return rows


# ------------------------------------------------------------------ batch 2: scene density x node layout  [R1-13, R1-18]
def part_scale_nodes(seeds, a, out):
    base_mix = list(S.DEFAULT_MIX)
    rows = []
    try:
        for dens in (1, 2):
            S.DEFAULT_MIX = [(c, k * dens) for c, k in base_mix]
            for nodes_name in ("N1", "N2", "N3"):
                nodes = S.NODE_SETS[nodes_name]
                acc = {m: dict(amr=[], worker=[], tractor=[], rho=[]) for m in ("EARLY", "LEARN_ModDrop")}
                t0 = time.time()
                for seed in seeds:
                    layout = S.yard_layout(seed)
                    train = [s for ep in range(a.n_train_s) for s in E.agent_series(seed, ep, layout, a.T, nodes)]
                    test = [s for ep in range(3000, 3000 + a.n_test_s) for s in E.agent_series(seed, ep, layout, a.T, nodes)]
                    X, Y = F.build_training(train, True, np.random.default_rng([seed, 98]))
                    model = F.train_mlp(X, Y, seed)
                    est = {}
                    for s in test:
                        late, good = F.late_fuse(s["tracks"], set(FULL))
                        for m in acc:
                            arr = np.full((len(s["ev"]), 2), np.nan)
                            if good.any():
                                e = (F.run_filter(F.merge(s["meas"], FULL), F.EARLY_CFG, s["ev"], s["q"])[:, :2] if m == "EARLY"
                                     else F.learned_fuse(model, s["tracks"], set(FULL), s["node"])[0])
                                arr[good] = np.asarray(e)[good]
                            est[(s["ep"], s["aid"], m)] = arr
                    by_ep = {}
                    for s in test:
                        by_ep.setdefault(s["ep"], []).append(s)
                    for ep, ser in by_ep.items():
                        for m in acc:
                            for c in ("amr", "worker", "tractor"):
                                d = [np.hypot(*(est[(ep, s["aid"], m)] - s["truth"]).T) for s in ser if s["cls"] == c]
                                if d:
                                    d = np.concatenate(d)
                                    acc[m][c].append(float(np.mean(np.where(np.isfinite(d), d, np.inf) <= E_BUDGET)))
                            over = ncl = 0
                            for i in range(len(ser)):
                                for j in range(i + 1, len(ser)):
                                    A_, B_ = ser[i], ser[j]
                                    if (A_["cls"] == "worker") + (B_["cls"] == "worker") != 1:
                                        continue
                                    dp, dv = B_["truth"] - A_["truth"], B_["vtruth"] - A_["vtruth"]
                                    dt_ = np.hypot(dp[:, 0], dp[:, 1])
                                    vc = -np.sum(dp * dv, 1) / np.maximum(dt_, 1e-6)
                                    de = np.hypot(*(est[(ep, B_["aid"], m)] - est[(ep, A_["aid"], m)]).T)
                                    sel = np.isfinite(de) & (dt_ <= E.CLOSE_M) & (vc > 0.2)
                                    ncl += sel.sum(); over += (sel & (de - dt_ > E_BUDGET)).sum()
                            if ncl:
                                acc[m]["rho"].append(over / ncl)
                for m in acc:
                    row = dict(density=dens, nodes=nodes_name, method=m)
                    for k in ("amr", "worker", "tractor", "rho"):
                        v = acc[m][k]; ci = _boot(v)
                        row[k] = round(float(np.mean(v)), 3) if v else None
                        row[k + "_ci"] = f"{ci[0]:.3f}-{ci[1]:.3f}"
                    rows.append(row)
                print(f"[scale] density x{dens} {nodes_name}: XT EARLY {rows[-2]['amr']} ModDrop {rows[-1]['amr']}  ({time.time() - t0:.0f}s)", flush=True)
                _write_csv(os.path.join(out, "scale_nodes.csv"), rows)
    finally:
        S.DEFAULT_MIX = base_mix
    return rows


# ------------------------------------------------------------------ batch 2: failure examples (supplement figure)  [R1-19]
def part_examples(seeds, a, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seed = seeds[0]
    layout, nodes = S.yard_layout(seed), S.NODE_SETS["N1"]
    train = [s for ep in range(a.n_train_s) for s in E.agent_series(seed, ep, layout, a.T, nodes)]
    X, Y = F.build_training(train, True, np.random.default_rng([seed, 98]))
    model = F.train_mlp(X, Y, seed)
    best = None
    for ep in range(3000, 3000 + a.n_test_s):
        ser = E.agent_series(seed, ep, layout, a.T, nodes)
        xt = [s for s in ser if s["cls"] == "amr"]
        if not xt:
            continue
        s = xt[0]
        e = F.run_filter(F.merge(s["meas"], FULL), F.EARLY_CFG, s["ev"], s["q"])[:, :2]
        err = np.hypot(*(e - s["truth"]).T)
        gap = float(np.mean(~np.isfinite(err) | (err > 1.0)))
        if best is None or gap > best[0]:
            best = (gap, ep, ser, s)
    _, ep, ser, s = best
    ev = s["ev"]
    obs = np.zeros(len(ev), bool)
    for m in ("lidar", "camera", "thermal"):
        if s["meas"].get(m) is not None and len(s["meas"][m]["t_cap"]):
            tc = np.sort(s["meas"][m]["t_cap"]); j = np.searchsorted(tc, ev, side="right") - 1
            obs |= (j >= 0) & (ev - tc[np.maximum(j, 0)] <= 0.5)
    e_early = np.hypot(*(F.run_filter(F.merge(s["meas"], FULL), F.EARLY_CFG, ev, s["q"])[:, :2] - s["truth"]).T)
    e_md = np.hypot(*(F.learned_fuse(model, s["tracks"], set(FULL), s["node"])[0] - s["truth"]).T)
    pair, bestc = None, -1
    for A_ in ser:
        for B_ in ser:
            if A_["cls"] in ("tractor", "amr") and B_["cls"] == "worker":
                dp, dv = B_["truth"] - A_["truth"], B_["vtruth"] - A_["vtruth"]
                dt_ = np.hypot(dp[:, 0], dp[:, 1]); vc = -np.sum(dp * dv, 1) / np.maximum(dt_, 1e-6)
                c = int(np.sum((dt_ <= E.CLOSE_M) & (vc > 0.2)))
                if c > bestc:
                    bestc, pair = c, (A_, B_, dt_, vc)
    A_, B_, dt_, vc = pair
    rel = {}
    for tag, cfg in (("timestamps ignored", F.NAIVE_CFG), ("timestamp-aware", F.EARLY_CFG)):
        ea = F.run_filter(F.merge(A_["meas"], FULL), cfg, ev, A_["q"])[:, :2]
        eb = F.run_filter(F.merge(B_["meas"], FULL), cfg, ev, B_["q"])[:, :2]
        rel[tag] = np.hypot(*(eb - ea).T) - dt_
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ax[0].plot(ev, e_early, lw=1, label="early fusion"); ax[0].plot(ev, e_md, lw=1, label="learned + modality dropout")
    ax[0].fill_between(ev, 0, 1, where=~obs, transform=ax[0].get_xaxis_transform(), color="0.85", label="no infrastructure observation")
    ax[0].axhline(E_BUDGET, ls="--", c="k", lw=0.8); ax[0].set_ylim(0, 3); ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("XT position error (m)")
    ax[0].set_title(f"(a) External truck, seed {seed}, episode {ep}", fontsize=9); ax[0].legend(fontsize=7)
    close = (dt_ <= E.CLOSE_M) & (vc > 0.2)
    for tag, v in rel.items():
        ax[1].plot(ev, v, lw=1, label=tag)
    ax[1].fill_between(ev, 0, 1, where=close, transform=ax[1].get_xaxis_transform(), color="0.9", label="closing within 10 m")
    ax[1].axhline(E_BUDGET, ls="--", c="k", lw=0.8); ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("estimated − true distance (m)")
    ax[1].set_title(f"(b) {A_['cls']}–worker pair: relative-distance error", fontsize=9); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "failure_examples.png"), dpi=300); fig.savefig(os.path.join(out, "failure_examples.pdf"))
    print(f"[examples] seed {seed} episode {ep}: unobserved share {np.mean(~obs):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all", help="analytic,core,sensitivity,dropout,pairs,sync,scale_nodes,examples or all (batch 1)")
    ap.add_argument("--out", default="results/revision")
    ap.add_argument("--seeds", default="7,11,23,31,47")
    ap.add_argument("--sens-seeds", default="7,11,23", help="seeds for sensitivity and dropout sweeps")
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--n-train-s", type=int, default=20, help="episodes for the sweeps")
    ap.add_argument("--n-test-s", type=int, default=15)
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--profile", default="port_terminal")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    S.apply_profile(a.profile)
    if a.calib:
        E.CALIB = json.load(open(a.calib, encoding="utf-8"))
    seeds = [int(x) for x in a.seeds.split(",")]
    sseeds = [int(x) for x in a.sens_seeds.split(",")]
    if a.quick:
        seeds, sseeds = seeds[:1], sseeds[:1]
        a.n_train, a.n_test, a.n_train_s, a.n_test_s, a.T = 3, 2, 3, 2, 30.0
    os.makedirs(a.out, exist_ok=True)
    parts = ["analytic", "core", "sensitivity", "dropout", "pairs", "sync"] if a.part == "all" else a.part.split(",")
    json.dump(dict(vars(a), started=time.strftime("%Y-%m-%d %H:%M")), open(os.path.join(a.out, "run_info.json"), "w"), indent=2)
    for p in parts:
        t0 = time.time()
        {"analytic": lambda: part_analytic(a.out), "core": lambda: part_core(seeds, a, a.out),
         "sensitivity": lambda: part_sensitivity(sseeds, a, a.out), "dropout": lambda: part_dropout(sseeds, a, a.out),
         "pairs": lambda: part_pairs(seeds, a, a.out), "sync": lambda: part_sync(seeds, a, a.out),
         "scale_nodes": lambda: part_scale_nodes(sseeds, a, a.out), "examples": lambda: part_examples(seeds, a, a.out)}[p]()
        print(f"== {p} finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
