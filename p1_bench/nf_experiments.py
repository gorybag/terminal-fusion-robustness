#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nf_experiments.py - Paper-1 benchmark experiments on NF-Synth

  A  missing modality x integration structure x learning: yard vehicles, all 15 subsets of
     {LiDAR, camera, thermal, GNSS-RTK}; EARLY / LATE / LEARN_noDrop / LEARN_ModDrop;
     episode bootstrap CI and paired Wilcoxon tests
  B  spatial-perception output: LiDAR occupancy grid (1 vs 2 edge nodes, clear vs fog)
  C  PoC deployment analytics: (i) line-of-sight coverage of the 50 m band vs edge nodes,
     (ii) clock synchronisation (none / NTP / PTP / perfect) vs registration error,
     (iii) indoor UWB anchor geometry (LOS anchors, GDOP) in a terminal hall with pillars
  D  indoor fire: time to first raw candidate per modality (thermal camera, smoke-detector proxy,
     ceiling heat sensor) vs detector spacing and t-squared growth class (Alpert ceiling jet)

Usage
  python nf_experiments.py --out results_bench            # full run (CPU, a few minutes)
  python nf_experiments.py --out results_quick --quick    # smoke test
All numbers are from a conceptual benchmark with assumed parameters (see nf_sim.py).
"""
import argparse
import itertools
import json
import math
import os
import time

import numpy as np

import nf_fusion as F
import nf_sim as S

CALIB = None   # optional detection-curve calibration (nf_calib_nuscenes.py output), set by --calib

PATTERNS = [tuple(m for m, b in zip(S.MODALITIES, bits) if b)
            for bits in itertools.product([0, 1], repeat=len(S.MODALITIES)) if any(bits)]
METHODS = ["EARLY", "LATE", "LEARN_noDrop", "LEARN_ModDrop"]
E_BUDGET = 0.5


def _boot_ci(vals, B=2000, seed=0):
    v = np.asarray(vals, float)
    if v.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bm = rng.choice(v, (B, v.size), replace=True).mean(1)
    return float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


# ------------------------------------------------------------------ A
def vehicle_series(seed, ep, layout, T, node, cond=None, clock="perfect"):
    rng = np.random.default_rng([seed, 11, ep])
    c = cond or str(rng.choice(["nominal", "night", "fog"], p=[0.6, 0.2, 0.2]))
    agents = S.make_agents(rng, layout, T)
    ev = np.arange(3.0, T - 0.5, 0.1)
    out = []
    for a in agents:
        if a["cls"] not in ("tractor", "forklift"):
            continue
        meas = S.sense_agent(rng, a, node, layout, T, c, clock, 0.0, CALIB)
        q = S.CLASSES[a["cls"]]["q_sig"] ** 2
        out.append(dict(ep=ep, cls=a["cls"], cond=c, meas=meas, q=q, ev=ev, node=np.array(node, float),
                        tracks=F.modality_tracks(meas, ev, q),
                        truth=a["pos"][np.round(ev / a["dt"]).astype(int)]))
    return out


def exp_a(seed, n_train, n_test, T, out_dir):
    from scipy.stats import wilcoxon
    layout, node = S.yard_layout(seed), S.NODE_SETS["N1"][0]
    tA = time.time()
    train = [s for ep in range(n_train) for s in vehicle_series(seed, ep, layout, T, node)]
    test = [s for ep in range(1000, 1000 + n_test) for s in vehicle_series(seed, ep, layout, T, node)]
    print(f"  [A] series generated+tracked: {time.time() - tA:.0f}s", flush=True)
    rng = np.random.default_rng([seed, 99])
    Xn, Yn = F.build_training(train, False, rng)
    Xd, Yd = F.build_training(train, True, rng)
    t0 = time.time()
    m_nd, m_md = F.train_mlp(Xn, Yn, seed), F.train_mlp(Xd, Yd, seed)
    train_s = time.time() - t0
    print(f"  [A] MLP training: {train_s:.0f}s ({len(Xn)} / {len(Xd)} samples)", flush=True)
    eps = sorted({s["ep"] for s in test})
    ei = {e: i for i, e in enumerate(eps)}
    Hh = np.zeros((len(METHODS), len(PATTERNS), len(eps)))
    Nn = np.zeros_like(Hh)
    sq = {(p, m): [] for p in PATTERNS for m in METHODS}
    avail = {p: [] for p in PATTERNS}
    for s in test:
        for pi, p in enumerate(PATTERNS):
            mask = set(p)
            late, good = F.late_fuse(s["tracks"], mask)
            avail[p].append(float(good.mean()))
            if not good.any():
                continue
            tr = s["truth"][good]
            early = F.run_filter(F.merge(s["meas"], p), F.EARLY_CFG, s["ev"], s["q"])[good, :2]
            lnd = F.learned_fuse(m_nd, s["tracks"], mask, s["node"])[0][good]
            lmd = F.learned_fuse(m_md, s["tracks"], mask, s["node"])[0][good]
            for mi, e in enumerate((early, late[good, :2], lnd, lmd)):
                d = np.hypot(e[:, 0] - tr[:, 0], e[:, 1] - tr[:, 1])
                d = np.where(np.isfinite(d), d, np.inf)
                Hh[mi, pi, ei[s["ep"]]] += float(np.sum(d <= E_BUDGET))
                Nn[mi, pi, ei[s["ep"]]] += d.size
                sq[(p, METHODS[mi])].append(d[np.isfinite(d)])
    with np.errstate(invalid="ignore", divide="ignore"):
        Pp = Hh.sum(2) / Nn.sum(2)                                   # methods x patterns (pooled ticks)
        bidx = np.random.default_rng([seed, 5]).integers(0, len(eps), (2000, len(eps)))
        Pb = np.nanmean(Hh[:, :, bidx].sum(3) / Nn[:, :, bidx].sum(3), axis=1)   # methods x B
        Pe = np.nanmean(Hh / Nn, axis=1)                             # methods x episodes (paired tests)
    rows = []
    for pi, p in enumerate(PATTERNS):
        for mi, m in enumerate(METHODS):
            fin = np.concatenate(sq[(p, m)]) if sq[(p, m)] else np.array([])
            rows.append(dict(pattern="+".join(p), k=len(p), method=m, n=int(Nn[mi, pi].sum()),
                             p_le=float(Pp[mi, pi]),
                             rmse=float(np.sqrt(np.mean(fin ** 2))) if fin.size else float("nan"),
                             availability=float(np.mean(avail[p]))))
    summ = {}
    for mi, m in enumerate(METHODS):
        r = [x for x in rows if x["method"] == m]
        worst = min(r, key=lambda x: x["p_le"])
        summ[m] = dict(macro_p_le=float(np.nanmean(Pp[mi])),
                       macro_ci=[float(np.percentile(Pb[mi], 2.5)), float(np.percentile(Pb[mi], 97.5))],
                       worst_pattern=worst["pattern"], worst_p_le=worst["p_le"],
                       full_p_le=next(x["p_le"] for x in r if x["k"] == 4),
                       by_k={k: float(np.mean([x["p_le"] for x in r if x["k"] == k])) for k in (1, 2, 3, 4)})
    tests = {}
    for a_, b_ in (("LEARN_ModDrop", "EARLY"), ("LEARN_ModDrop", "LEARN_noDrop"), ("LATE", "EARLY")):
        va, vb = Pe[METHODS.index(a_)], Pe[METHODS.index(b_)]
        dlt = va - vb
        pv = float(wilcoxon(va, vb).pvalue) if np.any(dlt != 0) else 1.0
        tests[f"{a_} vs {b_}"] = dict(mean_diff=float(np.mean(dlt)), wilcoxon_p=pv, n_episodes=len(eps))
    _write_csv(os.path.join(out_dir, "A_missing_modality.csv"), rows)
    res = dict(summary=summ, paired_tests=tests, n_train_samples=dict(noDrop=int(len(Xn)), ModDrop=int(len(Xd))),
               train_seconds=round(train_s, 1), n_test_series=len(test), n_train_series=len(train))
    _fig_a(rows, summ, out_dir)
    return res


# ------------------------------------------------------------------ P  (collision-oriented: all classes, pairs)
ALL_CLASSES = ("tractor", "forklift", "amr", "worker")
CLOSE_M, ALERT_M, T_H = 10.0, 3.0, 10.0


def sense_nodes(rng, a, nodes, layout, T, cond, clock):
    """Measurements of one agent from several edge nodes: per modality the node streams are concatenated
    (fusion sorts them by arrival); GNSS-RTK is on board and sensed once."""
    out = {}
    for m in S.MODALITIES:
        if m == "gnss":
            out[m] = S.sense(rng, a, m, nodes[0], layout, T, cond, clock, 0.0, CALIB)
            continue
        parts = [S.sense(rng, a, m, nd, layout, T, cond, clock, 0.0, CALIB) for nd in nodes]
        parts = [p for p in parts if p is not None]
        if not parts:
            out[m] = None
            continue
        keys = [k for k, v in parts[0].items() if isinstance(v, np.ndarray) and v.ndim == 1]
        cat = {k: np.concatenate([p[k] for p in parts]) for k in keys}
        for k, v in parts[0].items():
            cat.setdefault(k, v)
        out[m] = cat
    return out


def agent_series(seed, ep, layout, T, nodes, classes=ALL_CLASSES, cond=None, clock="perfect"):
    rng = np.random.default_rng([seed, 13, ep])
    c = cond or str(rng.choice(["nominal", "night", "fog"], p=[0.6, 0.2, 0.2]))
    agents = S.make_agents(rng, layout, T)
    ev = np.arange(3.0, T - 0.5, 0.1)
    out = []
    for ai, a in enumerate(agents):
        if a["cls"] not in classes:
            continue
        meas = sense_nodes(rng, a, nodes, layout, T, c, clock)
        q = S.CLASSES[a["cls"]]["q_sig"] ** 2
        k = np.round(ev / a["dt"]).astype(int)
        out.append(dict(ep=ep, aid=ai, cls=a["cls"], cond=c, meas=meas, q=q, ev=ev, node=np.array(nodes[0], float),
                        tracks=F.modality_tracks(meas, ev, q), truth=a["pos"][k], vtruth=a["vel"][k]))
    return out


def _pair_type(c1, c2):
    w = (c1 == "worker") + (c2 == "worker")
    return "vehicle-worker" if w == 1 else ("vehicle-vehicle" if w == 0 else None)


def exp_p(seed, n_train, n_test, T, out_dir, node_sets=("N1", "N2")):
    res = {}
    rows = {"P_by_class.csv": [], "P_pairs.csv": [], "P_latency_pairs.csv": []}
    for ns in node_sets:
        r, tabs = _exp_p_nodes(seed, n_train, n_test, T, S.NODE_SETS[ns])
        res[ns] = r
        for fn, tab in tabs.items():
            rows.setdefault(fn, []).extend([dict(nodes=ns, **x) for x in tab])
    for fn, tab in rows.items():
        _write_csv(os.path.join(out_dir, fn), tab)
    return res


def _exp_p_nodes(seed, n_train, n_test, T, nodes):
    """Positions of ALL classes (YT, RS, XT, workers) -> class-wise accuracy and pairwise (relative) metrics:
    relative-distance error, over-estimation while closing, unseen/missed close pairs, and the latency check
    (naive vs timestamp-aware) on closing pairs."""
    layout = S.yard_layout(seed)
    tP = time.time()
    train = [s for ep in range(n_train) for s in agent_series(seed, ep, layout, T, nodes)]
    test = [s for ep in range(3000, 3000 + n_test) for s in agent_series(seed, ep, layout, T, nodes)]
    rng = np.random.default_rng([seed, 98])
    Xn, Yn = F.build_training(train, False, rng)
    Xd, Yd = F.build_training(train, True, rng)
    m_nd, m_md = F.train_mlp(Xn, Yn, seed), F.train_mlp(Xd, Yd, seed)
    print(f"  [P] data+training: {time.time() - tP:.0f}s ({len(test)} test series)", flush=True)
    est = {}
    for s in test:
        n = len(s["ev"])
        for p in PATTERNS:
            mask = set(p)
            late, good = F.late_fuse(s["tracks"], mask)
            arrs = {m: np.full((n, 2), np.nan) for m in METHODS}
            if good.any():
                early = F.run_filter(F.merge(s["meas"], p), F.EARLY_CFG, s["ev"], s["q"])[:, :2]
                lnd = F.learned_fuse(m_nd, s["tracks"], mask, s["node"])[0]
                lmd = F.learned_fuse(m_md, s["tracks"], mask, s["node"])[0]
                for m, e in zip(METHODS, (early, late[:, :2], lnd, lmd)):
                    arrs[m][good] = np.asarray(e)[good]
            for m in METHODS:
                est[(s["ep"], s["aid"], p, m)] = arrs[m]
        full = tuple(S.MODALITIES)
        for tag, cfg in (("naive", F.NAIVE_CFG), ("ts_aware", F.EARLY_CFG)):
            est[(s["ep"], s["aid"], tag, "EARLY")] = F.run_filter(F.merge(s["meas"], full), cfg, s["ev"], s["q"])[:, :2]
    # class-wise absolute accuracy
    crow = []
    for cls in ALL_CLASSES:
        ser = [s for s in test if s["cls"] == cls]
        for p in PATTERNS:
            for m in METHODS:
                d = np.concatenate([np.hypot(*(est[(s["ep"], s["aid"], p, m)] - s["truth"]).T) for s in ser]) if ser else np.array([])
                seen = np.isfinite(d)
                crow.append(dict(cls=cls, pattern="+".join(p), k=len(p), method=m, n=int(d.size),
                                 coverage=float(seen.mean()) if d.size else float("nan"),
                                 p_le=float(np.mean(np.where(seen, d, np.inf) <= E_BUDGET)) if d.size else float("nan")))
    # pairwise metrics
    acc = {}
    lat = {}
    by_ep = {}
    for s in test:
        by_ep.setdefault(s["ep"], []).append(s)
    for ep, ser in by_ep.items():
        for i in range(len(ser)):
            for j in range(i + 1, len(ser)):
                a, b = ser[i], ser[j]
                pt = _pair_type(a["cls"], b["cls"])
                if pt is None:
                    continue
                dp, dv = b["truth"] - a["truth"], b["vtruth"] - a["vtruth"]
                dt_ = np.hypot(dp[:, 0], dp[:, 1])
                vc = -np.sum(dp * dv, 1) / np.maximum(dt_, 1e-6)          # closing speed (>0 when approaching)
                close, closing, alert = dt_ <= CLOSE_M, vc > 0.2, dt_ <= ALERT_M
                keys = [(p, m) for p in PATTERNS for m in METHODS] + [("naive", "EARLY"), ("ts_aware", "EARLY")]
                for p, m in keys:
                    ea, eb = est[(ep, a["aid"], p, m)], est[(ep, b["aid"], p, m)]
                    de = np.hypot(*(eb - ea).T)
                    both = np.isfinite(de)
                    e = de - dt_
                    if p in ("naive", "ts_aware"):
                        sel = both & close & closing
                        L = lat.setdefault((pt, p), dict(e=[], vc=[]))
                        L["e"].append(e[sel]); L["vc"].append(vc[sel])
                        continue
                    A = acc.setdefault((pt, p, m), np.zeros(9))
                    A += [close.sum(), (both & close).sum(), (both & close & (np.abs(e) <= E_BUDGET)).sum(),
                          (both & close & closing).sum(), np.nansum(np.where(both & close & closing, e, 0.0)),
                          (both & close & closing & (e > E_BUDGET)).sum(), alert.sum(),
                          (alert & (~both | (de > ALERT_M))).sum(), (both & close & (e > E_BUDGET)).sum()]
    prow = []
    for (pt, p, m), A in acc.items():
        n_close, seen, ok, ncl, sum_e, over, n_al, miss, over_all = A
        prow.append(dict(pair_type=pt, pattern="+".join(p), k=len(p), method=m, n_close=int(n_close),
                         coverage_close=seen / n_close if n_close else np.nan,
                         p_rel_le=ok / seen if seen else np.nan,
                         mean_err_closing=sum_e / ncl if ncl else np.nan,
                         overest_closing=over / ncl if ncl else np.nan,
                         n_alert=int(n_al), miss_alert=miss / n_al if n_al else np.nan))
    lrow = []
    for (pt, tag), L in lat.items():
        e, vc = np.concatenate(L["e"]), np.concatenate(L["vc"])
        slope = float(np.sum(e * vc) / np.sum(vc * vc)) if vc.size else np.nan     # e ~ tau_eff * closing speed
        lrow.append(dict(pair_type=pt, fusion=tag, n=int(e.size), mean_err=float(np.mean(e)) if e.size else np.nan,
                         overest_rate=float(np.mean(e > E_BUDGET)) if e.size else np.nan, tau_eff_ms=1000 * slope))
    vmax = {c: S.CLASSES[c]["v_max"] for c in ALL_CLASSES}
    brow = []
    for c1, c2 in (("tractor", "worker"), ("forklift", "worker"), ("amr", "worker"), ("tractor", "forklift"), ("tractor", "tractor")):
        vs = vmax[c1] + vmax[c2]
        brow.append(dict(pair=f"{c1}-{c2}", closing_speed=vs, band_R_m=vs * T_H, tau_max_ms=1000 * E_BUDGET / vs))
    summ = {}
    for pt in ("vehicle-worker", "vehicle-vehicle"):
        for m in METHODS:
            r = [x for x in prow if x["pair_type"] == pt and x["method"] == m]
            if not r:
                continue
            full = next((x for x in r if x["k"] == 4), {})
            summ[f"{pt}/{m}"] = dict(macro_p_rel_le=float(np.nanmean([x["p_rel_le"] for x in r])),
                                     macro_overest_closing=float(np.nanmean([x["overest_closing"] for x in r])),
                                     macro_miss_alert=float(np.nanmean([x["miss_alert"] for x in r])),
                                     full_p_rel_le=full.get("p_rel_le"), full_miss_alert=full.get("miss_alert"))
    cls_summ = {f"{c}/{m}": float(np.nanmean([x["p_le"] for x in crow if x["cls"] == c and x["method"] == m]))
                for c in ALL_CLASSES for m in METHODS}
    out = dict(pairs=summ, class_macro_p_le=cls_summ, latency=lrow, pair_budget=brow,
               n_test_series=len(test), close_m=CLOSE_M, alert_m=ALERT_M)
    return out, {"P_by_class.csv": crow, "P_pairs.csv": prow, "P_latency_pairs.csv": lrow, "P_pair_budget.csv": brow}


# ------------------------------------------------------------------ B
def exp_b(seed, T, out_dir):
    layout = S.yard_layout(seed)
    boxes = layout["boxes"]
    rows, grids = [], {}
    for nodes in ("N1", "N2"):
        for fog in (False, True):
            rng = np.random.default_rng([seed, 21, int(nodes == "N2"), int(fog)])
            agents = S.make_agents(rng, layout, T)
            g = F.OccGrid()
            for t in np.arange(0.0, T, 0.5):
                k = int(round(t / 0.05))
                axy = np.array([[a["pos"][k, 0], a["pos"][k, 1], S.CLASSES[a["cls"]]["radius"]] for a in agents])
                for s in S.NODE_SETS[nodes]:
                    th, rr = S.lidar_scan(rng, np.array(s, float), axy, boxes, fog=fog)
                    g.integrate(s, th, rr)
            r = g.evaluate(boxes)
            r.update(nodes=nodes, condition="fog" if fog else "clear")
            rows.append(r)
            grids[(nodes, fog)] = g.L.copy()
    _write_csv(os.path.join(out_dir, "B_occupancy.csv"), rows)
    _fig_b(grids, layout, out_dir)
    return rows


# ------------------------------------------------------------------ C
def coverage_stats(layout, nodes, rmax=50.0, res=1.0):
    xs = np.arange(-50.0, 50.0 + res / 2, res)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    P = np.c_[X.ravel(), Y.ravel()]
    boxes = layout["boxes"]
    inside = np.zeros(len(P), bool)
    for b in boxes:
        inside |= (P[:, 0] >= b[0]) & (P[:, 0] <= b[2]) & (P[:, 1] >= b[1]) & (P[:, 1] <= b[3])
    Q = P[(np.hypot(P[:, 0], P[:, 1]) <= 50.0) & ~inside]
    cnt = np.zeros(len(Q), int)
    for s in nodes:
        s = np.array(s, float)
        cnt += (S.los_many(s, Q, boxes) & (np.hypot(*(Q - s).T) <= rmax)).astype(int)
    return dict(ge1=float((cnt >= 1).mean()), ge2=float((cnt >= 2).mean()), n_points=int(len(Q))), (Q, cnt)


def exp_sync(seed, T, n_ep, out_dir):
    layout, node = S.yard_layout(seed), S.NODE_SETS["N1"][0]
    rows = []
    for clk in ("perfect", "ptp", "ntp", "none"):
        d_all = []
        for ep in range(n_ep):
            for s in vehicle_series(seed, 2000 + ep, layout, T, node, cond="nominal", clock=clk):
                o = F.run_filter(F.merge(s["meas"], S.MODALITIES), F.EARLY_CFG, s["ev"], s["q"])
                d = np.hypot(o[:, 0] - s["truth"][:, 0], o[:, 1] - s["truth"][:, 1])
                d_all.append(np.where(np.isfinite(d), d, np.inf))
        d = np.concatenate(d_all)
        fin = d[np.isfinite(d)]
        rows.append(dict(clock=clk, offset_range_ms=S.CLOCKS[clk][0] * 1000, drift_ppm=S.CLOCKS[clk][1],
                         p_le=float(np.mean(d <= E_BUDGET)), rmse=float(np.sqrt(np.mean(fin ** 2))),
                         p95=float(np.percentile(fin, 95))))
    _write_csv(os.path.join(out_dir, "C_clock_sync.csv"), rows)
    return rows


def terminal_hall():
    pillars = [(x - 0.5, y - 0.5, x + 0.5, y + 0.5) for x in (-20, -10, 0, 10, 20) for y in (-10, 0, 10)]
    return dict(W=60.0, H=40.0, boxes=np.array(pillars, float))


UWB_ANCHORS = {
    4: [(-29.5, -19.5), (29.5, -19.5), (29.5, 19.5), (-29.5, 19.5)],
    6: [(-29.5, -19.5), (29.5, -19.5), (29.5, 19.5), (-29.5, 19.5), (0.0, -19.5), (0.0, 19.5)],
    8: [(-29.5, -19.5), (29.5, -19.5), (29.5, 19.5), (-29.5, 19.5), (0.0, -19.5), (0.0, 19.5),
        (-29.5, 0.0), (29.5, 0.0)],
}


def exp_uwb(out_dir, res=0.5):
    hall = terminal_hall()
    boxes = hall["boxes"]
    xs = np.arange(-hall["W"] / 2 + res / 2, hall["W"] / 2, res)
    ys = np.arange(-hall["H"] / 2 + res / 2, hall["H"] / 2, res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.c_[X.ravel(), Y.ravel()]
    inside = np.zeros(len(P), bool)
    for b in boxes:
        inside |= (P[:, 0] >= b[0]) & (P[:, 0] <= b[2]) & (P[:, 1] >= b[1]) & (P[:, 1] <= b[3])
    P = P[~inside]
    rows = []
    for n, anchors in UWB_ANCHORS.items():
        G = np.zeros((len(P), 2, 2))
        nlos = np.zeros(len(P), int)
        for a in anchors:
            a = np.array(a, float)
            vis = S.los_many(a, P, boxes)
            u = (P - a) / np.maximum(np.hypot(*(P - a).T), 1e-6)[:, None]
            G += vis[:, None, None] * np.einsum("ni,nj->nij", u, u)
            nlos += vis.astype(int)
        ok = nlos >= 3
        det = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] ** 2
        gdop = np.full(len(P), np.inf)
        good = ok & (det > 1e-9)
        gdop[good] = np.sqrt((G[good, 0, 0] + G[good, 1, 1]) / det[good])
        rows.append(dict(anchors=n, frac_ge3_los=float(ok.mean()), frac_gdop_le2=float((gdop <= 2.0).mean()),
                         median_gdop=float(np.median(gdop[good])) if good.any() else float("nan"),
                         frac_any_nlos=float((nlos < n).mean())))
    _write_csv(os.path.join(out_dir, "C_uwb_geometry.csv"), rows)
    return rows


# ------------------------------------------------------------------ D
GROWTH = dict(slow=0.00293, medium=0.01172, fast=0.0469, ultrafast=0.1876)   # kW/s^2 (t-squared fires)


def _q_for_ceiling_dT(dT, r, H):
    """Invert Alpert's ceiling-jet correlation: HRR (kW) giving gas temperature rise dT at radius r."""
    if r / H <= 0.18:
        return (dT * H ** (5 / 3) / 16.9) ** 1.5
    return r * (dT * H / 5.38) ** 1.5


def exp_fire(out_dir, heights=(4.0, 10.0), spacings=(5.0, 10.0, 15.0), cam_dist=30.0, ifov=1.36e-3,
             heat_dT=30.0, smoke_dT=13.0, q_area=500.0, q_min=10.0):
    rows = []
    for H, (g, alpha) in itertools.product(heights, GROWTH.items()):
        q_th = max(q_min, math.pi * q_area * (2 * cam_dist * ifov) ** 2 / 4)   # >= 2x2 px hotspot
        t_th = math.sqrt(q_th / alpha)
        for sp in spacings:
            r = sp / math.sqrt(2)                                           # worst case: centre of a grid cell
            t_sm = math.sqrt(_q_for_ceiling_dT(smoke_dT, r, H) / alpha)
            t_ht = math.sqrt(_q_for_ceiling_dT(heat_dT, r, H) / alpha)
            ts = sorted([t_th, t_sm, t_ht])
            rows.append(dict(growth=g, alpha=alpha, spacing_m=sp, ceiling_m=H, t_thermal_s=round(t_th, 1),
                             t_smoke_proxy_s=round(t_sm, 1), t_heat_s=round(t_ht, 1),
                             t_cross_check_s=round(ts[1], 1)))
    _write_csv(os.path.join(out_dir, "D_fire_time_to_candidate.csv"), rows)
    return rows


# ------------------------------------------------------------------ io and figures
def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r[k]) for k in keys) + "\n")


def _fmt(v):
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.6g}"
    return str(v)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _fig_a(rows, summ, out_dir):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in METHODS:
        ax.plot([1, 2, 3, 4], [summ[m]["by_k"][k] for k in (1, 2, 3, 4)], "-o", ms=3, label=m)
    ax.set_xlabel("number of available modalities")
    ax.set_ylabel("P(error <= 0.5 m), mean over subsets")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_A_degradation.png"), dpi=200)
    plt.close(fig)
    M = np.array([[next(r["p_le"] for r in rows if r["pattern"] == "+".join(p) and r["method"] == m)
                   for m in METHODS] for p in PATTERNS])
    fig, ax = plt.subplots(figsize=(5.6, 6.0))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(PATTERNS)), ["+".join(p) for p in PATTERNS], fontsize=6)
    ax.set_xticks(range(len(METHODS)), METHODS, fontsize=6, rotation=20)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if M[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=ax, shrink=0.6, label="P(error <= 0.5 m)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_A_missing_matrix.png"), dpi=200)
    plt.close(fig)


def _draw_boxes(ax, boxes):
    from matplotlib.patches import Rectangle
    for b in boxes:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=0.4, ec="k"))


def _fig_b(grids, layout, out_dir):
    plt = _plt()
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 7.0))
    for ax, ((nodes, fog), L) in zip(axs.ravel(), grids.items()):
        ax.imshow(1 / (1 + np.exp(-L.T)), origin="lower", extent=[-50, 50, -50, 50], cmap="gray_r", vmin=0, vmax=1)
        _draw_boxes(ax, layout["boxes"])
        for s in S.NODE_SETS[nodes]:
            ax.plot(*s, "r^", ms=5)
        ax.set_title(f"{nodes}, {'fog' if fog else 'clear'}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_B_occupancy.png"), dpi=200)
    plt.close(fig)


def _fig_cov(cov, layout, out_dir):
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, (name, (st, (Q, cnt))) in zip(axs, cov.items()):
        sc = ax.scatter(Q[:, 0], Q[:, 1], c=np.minimum(cnt, 3), s=1.5, cmap="viridis", vmin=0, vmax=3)
        _draw_boxes(ax, layout["boxes"])
        for s in S.NODE_SETS[name]:
            ax.plot(*s, "r^", ms=6)
        ax.set_title(f"{name}: >=1 node {st['ge1']:.0%}, >=2 nodes {st['ge2']:.0%}", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(sc, ax=axs, shrink=0.7, label="edge nodes with line of sight")
    fig.savefig(os.path.join(out_dir, "fig_C_coverage.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results_bench")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="ABCD", help="subset of experiments to run, e.g. A or BCD; P = all-class and pairwise (collision-oriented) metrics")
    ap.add_argument("--p-nodes", default="N1,N2", help="node layouts evaluated in part P")
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--calib", help="JSON from nf_calib_nuscenes.py and/or aihub_port_prep.py convert")
    ap.add_argument("--profile", default="airport_yard", choices=["airport_yard", "port_terminal"],
                    help="scenario profile: class speeds, GNSS availability, agent mix")
    a = ap.parse_args()
    global CALIB
    S.apply_profile(a.profile)
    if a.calib:
        CALIB = json.load(open(a.calib, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    n_train, n_test, T, n_sync = (6, 4, 40.0, 2) if a.quick else (a.n_train, a.n_test, a.T, 8)
    t0 = time.time()
    sp = os.path.join(a.out, "summary.json")
    res = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
    res.update(seed=a.seed, quick=a.quick, calib=a.calib, profile=a.profile)
    if "A" in a.parts:
        res["A"] = exp_a(a.seed, n_train, n_test, T, a.out)
        res["A"]["config"] = dict(n_train_episodes=n_train, n_test_episodes=n_test, T=T)
        print(f"[A] {time.time() - t0:.0f}s", flush=True)
    if "B" in a.parts:
        res["B"] = exp_b(a.seed, 30.0, a.out)
        print(f"[B] {time.time() - t0:.0f}s", flush=True)
    if "C" in a.parts:
        layout = S.yard_layout(a.seed)
        cov = {n: coverage_stats(layout, S.NODE_SETS[n]) for n in S.NODE_SETS}
        res["C_coverage"] = {n: v[0] for n, v in cov.items()}
        _fig_cov(cov, layout, a.out)
        res["C_sync"] = exp_sync(a.seed, T, n_sync, a.out)
        res["C_uwb"] = exp_uwb(a.out)
        print(f"[C] {time.time() - t0:.0f}s", flush=True)
    if "D" in a.parts:
        res["D_fire"] = exp_fire(a.out)
    if "P" in a.parts:
        res["P"] = exp_p(a.seed, n_train, n_test, T, a.out, tuple(a.p_nodes.split(",")))
        print(f"[P] {time.time() - t0:.0f}s", flush=True)
    res["runtime_s_last"] = round(time.time() - t0, 1)
    res["note"] = "Conceptual benchmark; parameters are assumptions (see nf_sim.py)."
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=float)
    print(f"done {res['runtime_s_last']}s -> {a.out}")


if __name__ == "__main__":
    main()
