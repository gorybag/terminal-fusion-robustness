#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p1_nearfield_fusion_sim.py
==========================
Near-field response band: multimodal fusion simulation (E1-E3)
(Paper 1 experiment for the conceptual study; cargo-yard event sub-scenario)

Experiments
  E1  Latency-error budget: un-modelled latency tau -> position bias ~ v * tau
  E2  Fusion strategy x sensor condition -> P(e<=0.5 m), p95 error,
      robustness ratio (KPI 11 proxy = performance under condition / nominal)
  E3  Forklift-pedestrian near-miss alert lead time and false alarms
      (proxy of the L3 input quality; L0 does NOT make the final judgement)

Strategies (all share the same measurement realisations = paired comparison)
  S0_lidar_only   LiDAR only, capture-timestamp aware
  S1_naive        all sensors, measurement time := arrival time, spec noise
  S2_ts_aware     all sensors, capture timestamps + out-of-sequence rollback (OOSM)
  S3_ts_qt_robust S2 + per-measurement quality tags (reported sigma) + robust soft gating
                  (R inflated by NIS/chi2_99 when NIS exceeds the 99% gate; no hard rejection,
                  which avoids the well-known 'gating divergence' of hard gates)
  S4_ts_qt_hm     S3 + innovation-based sensor health monitor (NIS EWMA -> R inflation)

IMPORTANT
  Every numeric parameter is an ASSUMPTION for a conceptual study.
  Replace them with measured values (sensor datasheets, Jetson Orin profiling,
  field logs) before making final claims. Data association is assumed solved
  (track IDs known) to isolate temporal/quality effects.

Usage
  python p1_nearfield_fusion_sim.py --out results --episodes 20 --nm-episodes 80 --seed 2026
  (quick check: --episodes 3 --nm-episodes 10)
"""
import argparse
import bisect
import json
import math
import os
import time

import numpy as np

# ------------------------------------------------------------------ constants
T_HORIZON = 10.0        # s   ; near-field band R = v_max * T_horizon
E_BUDGET = 0.5          # m   ; KPI 3 proxy (registration error)
LAT_BUDGET = 0.050      # s   ; KPI 10 (heterogeneous registration/inference latency)
CHI2_99_2DOF = 9.2103   # chi-square 99% quantile, 2 dof
DT_TRUTH = 0.01         # s   ; ground-truth integration step
EVAL_HZ = 10.0          # Hz  ; fused output rate evaluated
WARMUP = 2.0            # s
D_ALERT = 2.0           # m   ; predicted closest-approach alert threshold (E3)
T_ALERT = 5.0           # s   ; time-to-closest-approach window used for alerting (E3)
ALERT_PERSIST = 3       # consecutive output ticks needed for an alert (E3)
CONFLICT_DMIN = 1.5     # m   ; E3 label: conflict if true d_min < this
SAFE_DMIN = 4.0         # m   ; E3 label: safe pass if true d_min > this

AGENT_TYPES = {
    # v_max [m/s]; a_sig: motion generator accel scale; q_sig: filter process noise [m/s^2]
    "yard_tractor": dict(v_max=5.0, a_sig=0.8, q_sig=1.5, gnss=True),
    "forklift":     dict(v_max=4.2, a_sig=1.0, q_sig=1.5, gnss=True),
    "amr":          dict(v_max=1.5, a_sig=0.4, q_sig=0.6, gnss=False),
    "pedestrian":   dict(v_max=1.2, a_sig=0.6, q_sig=0.8, gnss=False),
}

SENSOR_SPECS = {
    # rate [Hz]; latency mean/sd [s]; sigma = s0 + s1 * range [m]; detection probability
    "lidar":   dict(rate=10.0, lat_mu=0.060, lat_sd=0.015, s0=0.05, s1=0.002, pd=0.98),
    "camera":  dict(rate=15.0, lat_mu=0.045, lat_sd=0.010, s0=0.10, s1=0.010, pd=0.95),
    "thermal": dict(rate=9.0,  lat_mu=0.080, lat_sd=0.015, s0=0.15, s1=0.012, pd=0.90),
    "gnss":    dict(rate=10.0, lat_mu=0.120, lat_sd=0.030, s0=0.03, s1=0.000, pd=0.99),
}
# port profile: yard_tractor = YT, forklift = reach stacker, amr = external truck (no terminal RTK), pedestrian = worker
PORT_PROFILE = {"yard_tractor": dict(v_max=6.9), "forklift": dict(v_max=4.2),
                "amr": dict(v_max=6.9, a_sig=0.8, q_sig=1.5, gnss=False)}


def apply_profile(name):
    if name == "port_terminal":
        for k, upd in PORT_PROFILE.items():
            AGENT_TYPES[k].update(upd)


def apply_calib(path):
    """Camera detection probabilities from port images (aihub_port_prep.py convert) and LiDAR wet-weather offset
    (adrive3d_prep.py); this simulator uses range-independent probabilities, so recalls are used directly."""
    cal = json.load(open(path, encoding="utf-8"))
    cam = cal.get("camera_port", {})
    lg = lambda p: np.log(p / (1 - p))
    sg = lambda x: float(1 / (1 + np.exp(-x)))
    day = cam.get("day") or {}
    if cam.get("mode") == "full" and day.get("recall") is not None:
        SENSOR_SPECS["camera"]["pd"] = day["recall"]
    base = SENSOR_SPECS["camera"]["pd"]
    cond = cam.get("conditions", {})
    if "da" in cond.get("night", {}):
        CONDITIONS["C1_night"]["camera"]["pd"] = round(sg(lg(base) + cond["night"]["da"]), 4)
        CONDITIONS["C5_compound"]["camera"]["pd"] = CONDITIONS["C1_night"]["camera"]["pd"]
    if "da" in cond.get("fog", {}):
        CONDITIONS["C2_fog_rain"]["camera"]["pd"] = round(sg(lg(base) + cond["fog"]["da"]), 4)
    lk = cal.get("lidar_kr", {}).get("conditions", {}).get("fog")
    if lk and "da" in lk:                      # offset is absent when the wet-weather sample was too small
        base = SENSOR_SPECS["lidar"]["pd"]
        pd_w = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + lk["da"])))
        CONDITIONS["C2_fog_rain"]["lidar"]["pd"] = round(float(pd_w), 4)
        CONDITIONS["C5_compound"]["lidar"]["pd"] = round(float(pd_w), 4)
    return dict(camera_pd=SENSOR_SPECS["camera"]["pd"], night_pd=CONDITIONS["C1_night"]["camera"]["pd"],
                fog_cam_pd=CONDITIONS["C2_fog_rain"]["camera"]["pd"], fog_lidar_pd=CONDITIONS["C2_fog_rain"]["lidar"]["pd"])


SENSOR_LIST = list(SENSOR_SPECS)
SID = {k: i for i, k in enumerate(SENSOR_LIST)}

# sensor-condition modifiers ("tagged": the sensor itself reports the degradation)
CONDITIONS = {
    "C0_nominal": {},
    "C1_night": {
        "camera": dict(pd=0.30, sigma_mult=2.0, tagged=True)},
    "C2_fog_rain": {
        "lidar": dict(pd=0.60, sigma_mult=3.0, outlier_p=0.05, outlier_mag=(1.0, 3.0), tagged=False),
        "camera": dict(pd=0.85, sigma_mult=1.5, tagged=True)},
    "C3_multipath": {
        "gnss": dict(float_frac=0.30, float_sigma=0.50, jump_p=0.05, jump_mag=(2.0, 5.0))},
    "C4_silent_fault": {
        "camera": dict(bias=(0.8, -0.6)),          # 1.0 m extrinsic drift, NOT reported
        "lidar": dict(outage_frac=0.20)},
    "C5_compound": {
        "camera": dict(pd=0.30, sigma_mult=2.0, tagged=True, outage_frac=0.25),
        "lidar": dict(pd=0.60, sigma_mult=3.0, outlier_p=0.05, outlier_mag=(1.0, 3.0),
                      tagged=False, outage_frac=0.25),
        "thermal": dict(outage_frac=0.25),
        "gnss": dict(float_frac=0.30, float_sigma=0.50, jump_p=0.05, jump_mag=(2.0, 5.0),
                     outage_frac=0.25)},
}

STRATEGIES = {
    "S0_lidar_only": dict(sensors=("lidar",), ts_aware=True,  quality=False, gating=False, health=False),
    "S1_naive":      dict(sensors=None,       ts_aware=False, quality=False, gating=False, health=False),
    "S2_ts_aware":   dict(sensors=None,       ts_aware=True,  quality=False, gating=False, health=False),
    "S3_ts_qt_robust": dict(sensors=None,     ts_aware=True,  quality=True,  gating=True,  health=False),
    "S4_ts_qt_hm":     dict(sensors=None,     ts_aware=True,  quality=True,  gating=True,  health=True),
}


# ------------------------------------------------------------------ filter
class KF:
    """2D constant-velocity Kalman filter with isotropic position measurements.
    With isotropic R and Q the x/y axes decouple and share one 2x2 covariance."""
    __slots__ = ("q", "t", "px", "py", "vx", "vy", "P00", "P01", "P11", "init", "la")

    def __init__(self, q):
        self.q = q
        self.reset()

    def reset(self):
        self.t = 0.0
        self.px = self.py = self.vx = self.vy = 0.0
        self.P00 = self.P01 = self.P11 = 0.0
        self.init = False
        self.la = -1e9                      # time of last accepted update

    def initialize(self, t, zx, zy, R):
        self.t, self.px, self.py, self.vx, self.vy = t, zx, zy, 0.0, 0.0
        self.P00, self.P01, self.P11 = R, 0.0, 9.0   # (3 m/s)^2 velocity prior
        self.init, self.la = True, t

    def get(self):
        return (self.t, self.px, self.py, self.vx, self.vy,
                self.P00, self.P01, self.P11, self.init, self.la)

    def set(self, s):
        (self.t, self.px, self.py, self.vx, self.vy,
         self.P00, self.P01, self.P11, self.init, self.la) = s

    def predict(self, t):
        dt = t - self.t
        if dt <= 0.0:
            return
        q, dt2 = self.q, dt * dt
        P01, P11 = self.P01, self.P11
        self.P00 = self.P00 + 2.0 * dt * P01 + dt2 * P11 + q * dt2 * dt2 * 0.25
        self.P01 = P01 + dt * P11 + q * dt2 * dt * 0.5
        self.P11 = P11 + q * dt2
        self.px += self.vx * dt
        self.py += self.vy * dt
        self.t = t

    def update(self, dx, dy, R):
        S = self.P00 + R
        K0, K1 = self.P00 / S, self.P01 / S
        self.px += K0 * dx
        self.py += K0 * dy
        self.vx += K1 * dx
        self.vy += K1 * dy
        P00, P01 = self.P00, self.P01
        self.P00 = P00 - K0 * P00
        self.P01 = P01 - K0 * P01
        self.P11 = self.P11 - K1 * P01

    def extrap(self, t):
        dt = t - self.t
        return (self.px + self.vx * dt, self.py + self.vy * dt, self.vx, self.vy)


def _apply(kf, t, m, cfg, health, first):
    """One measurement update. m = (zx, zy, sig_rep, sig_nom, sid)."""
    zx, zy, srep, snom, sid = m
    sig = srep if cfg["quality"] else snom
    R0 = sig * sig
    lam = min(25.0, max(1.0, health[sid] * 0.5)) if cfg["health"] else 1.0
    R = R0 * lam
    if not kf.init:
        kf.initialize(t, zx, zy, R)
        return
    kf.predict(t)
    dx, dy = zx - kf.px, zy - kf.py
    d2 = dx * dx + dy * dy
    if cfg["health"] and first:
        # NIS against the un-inflated (tagged) R: E[NIS] = 2 when the tag is consistent
        health[sid] = 0.95 * health[sid] + 0.05 * min(d2 / (kf.P00 + R0), 50.0)
    if cfg["gating"]:
        nis = d2 / (kf.P00 + R)
        if nis > CHI2_99_2DOF:          # robust soft gate: down-weight instead of rejecting
            R *= nis / CHI2_99_2DOF
    kf.update(dx, dy, R)
    kf.la = t


def run_filter(M, cfg, eval_t, q):
    """Process measurements in ARRIVAL order (real-time causality).
    ts-aware strategies insert each measurement at its CAPTURE time and re-filter
    (out-of-sequence measurement rollback). Output = latest state extrapolated to eval time.
    Returns (out[n_eval,4] = x, y, vx, vy ; NaN when no track) and CPU microseconds/measurement."""
    n_eval = len(eval_t)
    out = np.full((n_eval, 4), np.nan)
    ts_aware = cfg["ts_aware"]
    health = [2.0] * len(SENSOR_LIST)
    kf = KF(q)
    H_t, H_m, H_s = [], [], []
    base = None
    ei = 0
    tc_l, ta_l = M["t_cap"], M["t_arr"]
    rows = list(zip(M["zx"], M["zy"], M["sig_rep"], M["sig_nom"], M["sid"]))
    ev = eval_t.tolist()
    n = len(ta_l)
    t0 = time.perf_counter()
    for j in range(n):
        ta = ta_l[j]
        while ei < n_eval and ev[ei] < ta:
            if kf.init:
                out[ei] = kf.extrap(ev[ei])
            ei += 1
        m = rows[j]
        if not ts_aware:
            _apply(kf, ta, m, cfg, health, True)
            continue
        tc = tc_l[j]
        if base is not None and tc < base[0]:
            continue                                  # older than the kept history -> drop
        k = bisect.bisect_right(H_t, tc)
        H_t.insert(k, tc)
        H_m.insert(k, m)
        H_s.insert(k, None)
        if k == 0:
            if base is None:
                kf.reset()
            else:
                kf.set(base)
        else:
            kf.set(H_s[k - 1])
        for i in range(k, len(H_t)):
            _apply(kf, H_t[i], H_m[i], cfg, health, i == k)
            H_s[i] = kf.get()
        if len(H_t) > 64:                              # prune history older than 0.6 s
            cut = bisect.bisect_left(H_t, ta - 0.6)
            if cut > 0:
                base = H_s[cut - 1]
                del H_t[:cut], H_m[:cut], H_s[:cut]
    cpu_us = (time.perf_counter() - t0) / max(n, 1) * 1e6
    while ei < n_eval:
        if kf.init:
            out[ei] = kf.extrap(ev[ei])
        ei += 1
    return out, cpu_us


# ------------------------------------------------------------------ data generation
def gen_trajectory(rng, a, T, area_r=45.0):
    """Smooth random motion inside the near-field area (edge node at the origin)."""
    n = int(round(T / DT_TRUTH)) + 1
    pos = np.empty((n, 2))
    nz = rng.standard_normal(n)
    uu = rng.random(n)
    ut = rng.uniform(0.3, 1.0, n)
    r0 = area_r * math.sqrt(rng.uniform(0.05, 0.9))
    th = rng.uniform(0, 2 * math.pi)
    x, y = r0 * math.cos(th), r0 * math.sin(th)
    hd = rng.uniform(0, 2 * math.pi)
    vmax, asig = a["v_max"], a["a_sig"]
    v = rng.uniform(0.4, 1.0) * vmax
    tgt, w = v, 0.0
    sq = math.sqrt(DT_TRUTH)
    pos[0] = (x, y)
    for k in range(1, n):
        w += -0.5 * w * DT_TRUTH + 0.25 * sq * nz[k]
        if math.hypot(x, y) > area_r:
            des = math.atan2(-y, -x)
            err = (des - hd + math.pi) % (2 * math.pi) - math.pi
            w = 0.8 * err
        w = max(-0.6, min(0.6, w))
        hd += w * DT_TRUTH
        if uu[k] < DT_TRUTH / 8.0:
            tgt = ut[k] * vmax
        acc = max(-2 * asig, min(2 * asig, 0.5 * (tgt - v)))
        v = max(0.0, min(vmax, v + acc * DT_TRUTH))
        x += v * math.cos(hd) * DT_TRUTH
        y += v * math.sin(hd) * DT_TRUTH
        pos[k] = (x, y)
    return pos


def _intervals(rng, T, frac, mean_dur):
    if frac <= 0:
        return []
    mean_on = mean_dur * (1.0 - frac) / frac
    ints, t = [], rng.exponential(mean_on)
    while t < T:
        d = rng.exponential(mean_dur)
        ints.append((t, t + d))
        t += d + rng.exponential(mean_on)
    return ints


def _inside(t, ints):
    m = np.zeros(t.shape, bool)
    for a, b in ints:
        m |= (t >= a) & (t < b)
    return m


def gen_meas(rng, pos, T, sensor, mod, has_gnss, fixed_latency=None):
    spec = SENSOR_SPECS[sensor]
    if sensor == "gnss" and not has_gnss:
        return None
    period = 1.0 / spec["rate"]
    tc = np.arange(rng.uniform(0, period), T - 0.35, period)
    idx = np.minimum(np.round(tc / DT_TRUTH).astype(int), len(pos) - 1)
    p = pos[idx]
    r_true = np.hypot(p[:, 0], p[:, 1])
    sig_spec = spec["s0"] + spec["s1"] * r_true
    sm = mod.get("sigma_mult", 1.0)
    tagged = mod.get("tagged", True)
    z = p + rng.standard_normal(p.shape) * (sig_spec * sm)[:, None]
    sig_rep = sig_spec * (sm if tagged else 1.0)
    if "bias" in mod:
        z = z + np.asarray(mod["bias"], float)[None, :]
    if mod.get("outlier_p", 0) > 0:                      # silent outliers
        o = rng.random(len(tc)) < mod["outlier_p"]
        mag = rng.uniform(*mod["outlier_mag"], len(tc))
        ang = rng.uniform(0, 2 * np.pi, len(tc))
        z[o, 0] += (mag * np.cos(ang))[o]
        z[o, 1] += (mag * np.sin(ang))[o]
    if sensor == "gnss":
        fl = np.zeros(len(tc), bool)
        if mod.get("float_frac", 0) > 0:                 # RTK float episodes (fix type reported)
            fl = _inside(tc, _intervals(rng, T, mod["float_frac"], 3.0))
            z[fl] = p[fl] + rng.standard_normal((int(fl.sum()), 2)) * mod["float_sigma"]
            sig_rep = np.where(fl, mod["float_sigma"], sig_rep)
        if mod.get("jump_p", 0) > 0:                     # multipath jumps while "fixed" (silent)
            jm = (rng.random(len(tc)) < mod["jump_p"]) & ~fl
            mag = rng.uniform(*mod["jump_mag"], len(tc))
            ang = rng.uniform(0, 2 * np.pi, len(tc))
            z[jm, 0] += (mag * np.cos(ang))[jm]
            z[jm, 1] += (mag * np.sin(ang))[jm]
    sig_rep = sig_rep * np.exp(rng.normal(0.0, 0.2, len(tc)))    # imperfect self-assessment
    r_meas = np.hypot(z[:, 0], z[:, 1])
    sig_nom = spec["s0"] + spec["s1"] * r_meas                    # datasheet model only
    keep = rng.random(len(tc)) < mod.get("pd", spec["pd"])
    keep &= ~_inside(tc, _intervals(rng, T, mod.get("outage_frac", 0.0), 2.0))
    if fixed_latency is None:
        lat = np.maximum(0.005, rng.normal(spec["lat_mu"], spec["lat_sd"], len(tc)))
    else:
        lat = np.full(len(tc), fixed_latency)
    return dict(t_cap=tc[keep], t_arr=(tc + lat)[keep], zx=z[keep, 0], zy=z[keep, 1],
                sig_rep=sig_rep[keep], sig_nom=sig_nom[keep],
                sid=np.full(int(keep.sum()), SID[sensor]))


def merge(meas, sensors):
    parts = [meas[s] for s in sensors if meas.get(s) is not None]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    o = np.argsort(cat["t_arr"], kind="stable")
    return {k: v[o].tolist() for k, v in cat.items()}


def error_metrics(err):
    """err: NaN where no track. NaN counts as failure for P(e<=0.5)."""
    ok = np.isfinite(err)
    le = np.where(ok, err <= E_BUDGET, False)
    nwin = len(err) // 10
    w = le[: nwin * 10].reshape(nwin, 10).all(axis=1) if nwin else np.array([False])
    e = err[ok]
    return dict(avail=float(ok.mean()), p_le=float(le.mean()), p_1s=float(w.mean()),
                rmse=float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan"),
                p95=float(np.percentile(e, 95)) if e.size else float("nan"),
                median=float(np.median(e)) if e.size else float("nan"))


# ------------------------------------------------------------------ experiments
def run_e1(seed, episodes, out_dir):
    taus = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    cfg_naive = dict(STRATEGIES["S1_naive"])
    cfg_ts = dict(STRATEGIES["S2_ts_aware"])
    rows = []
    T = 30.0
    for ai, (name, a) in enumerate(AGENT_TYPES.items()):
        v = a["v_max"]
        for ti, tau in enumerate(taus):
            err_n, err_t = [], []
            for ep in range(episodes):
                rng = np.random.default_rng([seed, 1, ai, ti, ep])
                t = np.arange(0.0, T + DT_TRUTH / 2, DT_TRUTH)
                th = rng.uniform(0, 2 * np.pi)
                p0 = rng.uniform(-20, 20, 2)
                pos = p0[None, :] + v * t[:, None] * np.array([np.cos(th), np.sin(th)])[None, :]
                meas = {"lidar": gen_meas(rng, pos, T, "lidar", {}, False, fixed_latency=tau)}
                M = merge(meas, ["lidar"])
                ev = np.arange(WARMUP + 3.0, T - 0.5, 1.0 / EVAL_HZ)
                truth = pos[np.round(ev / DT_TRUTH).astype(int)]
                for cfg, store in ((cfg_naive, err_n), (cfg_ts, err_t)):
                    out, _ = run_filter(M, cfg, ev, 0.3 ** 2)
                    store.append(np.hypot(out[:, 0] - truth[:, 0], out[:, 1] - truth[:, 1]))
            en, et = np.concatenate(err_n), np.concatenate(err_t)
            rows.append(dict(agent=name, v_max=v, tau_ms=tau * 1000,
                             analytic_bias_m=v * tau,
                             naive_mean_err_m=float(np.nanmean(en)),
                             ts_aware_mean_err_m=float(np.nanmean(et))))
    _write_csv(os.path.join(out_dir, "e1_latency_error.csv"), rows)
    return rows


def run_e2(seed, episodes, out_dir, T=60.0):
    rows, per_agent = [], []
    cpu = {s: [] for s in STRATEGIES}
    err_c5 = {s: [] for s in STRATEGIES}
    ep_ple = {}
    for ci, (cname, cmod) in enumerate(CONDITIONS.items()):
        errs = {s: [] for s in STRATEGIES}
        errs_agent = {(s, a): [] for s in STRATEGIES for a in AGENT_TYPES}
        for ep in range(episodes):
            ep_err = {s: [] for s in STRATEGIES}
            for ai, (aname, a) in enumerate(AGENT_TYPES.items()):
                rng = np.random.default_rng([seed, 2, ci, ep, ai])
                pos = gen_trajectory(rng, a, T)
                meas = {s: gen_meas(rng, pos, T, s, cmod.get(s, {}), a["gnss"]) for s in SENSOR_LIST}
                ev = np.arange(WARMUP, T - 0.5, 1.0 / EVAL_HZ)
                truth = pos[np.round(ev / DT_TRUTH).astype(int)]
                for sname, scfg in STRATEGIES.items():
                    M = merge(meas, scfg["sensors"] or SENSOR_LIST)
                    out, cu = run_filter(M, scfg, ev, a["q_sig"] ** 2)
                    e = np.hypot(out[:, 0] - truth[:, 0], out[:, 1] - truth[:, 1])
                    errs[sname].append(e)
                    errs_agent[(sname, aname)].append(e)
                    ep_err[sname].append(e)
                    cpu[sname].append(cu)
            for sname in STRATEGIES:
                ee = np.concatenate(ep_err[sname])
                ep_ple.setdefault((cname, sname), []).append(
                    float(np.mean(np.where(np.isfinite(ee), ee <= E_BUDGET, False))))
        for sname in STRATEGIES:
            e = np.concatenate(errs[sname])
            rows.append(dict(condition=cname, strategy=sname, **error_metrics(e)))
            if cname == "C5_compound":
                err_c5[sname] = e
            for aname in AGENT_TYPES:
                ea = np.concatenate(errs_agent[(sname, aname)])
                per_agent.append(dict(condition=cname, strategy=sname, agent=aname,
                                      **error_metrics(ea)))
    base = {r["strategy"]: r["p_le"] for r in rows if r["condition"] == "C0_nominal"}
    brng = np.random.default_rng([seed, 99])
    for r in rows:
        r["robustness_ratio"] = r["p_le"] / base[r["strategy"]] if base[r["strategy"]] > 0 else float("nan")
        x = np.array(ep_ple[(r["condition"], r["strategy"])])
        x0 = np.array(ep_ple[("C0_nominal", r["strategy"])])
        bm = np.array([brng.choice(x, x.size).mean() for _ in range(2000)])
        b0 = np.array([brng.choice(x0, x0.size).mean() for _ in range(2000)])
        r["p_le_ci_lo"], r["p_le_ci_hi"] = [float(v) for v in np.percentile(bm, [2.5, 97.5])]
        rr = bm / np.where(b0 > 0, b0, np.nan)
        r["rr_ci_lo"], r["rr_ci_hi"] = [float(v) for v in np.nanpercentile(rr, [2.5, 97.5])]
    cpu_rows = [dict(strategy=s, cpu_us_per_meas_mean=float(np.mean(v)),
                     cpu_us_per_meas_p95=float(np.percentile(v, 95))) for s, v in cpu.items()]
    _write_csv(os.path.join(out_dir, "e2_strategy_condition.csv"), rows)
    _write_csv(os.path.join(out_dir, "e2_per_agent.csv"), per_agent)
    _write_csv(os.path.join(out_dir, "e2_cpu_cost.csv"), cpu_rows)
    return rows, per_agent, cpu_rows, err_c5


def gen_crossing(rng, T=30.0):
    """Forklift and pedestrian on crossing paths; label conflict (d_min < CONFLICT_DMIN) or
    safe pass (d_min > SAFE_DMIN); ambiguous configurations are re-sampled."""
    t = np.arange(0.0, T + DT_TRUTH / 2, DT_TRUTH)
    while True:
        cp = rng.uniform(-15, 15, 2)
        vf, vp = rng.uniform(3.0, 4.2), rng.uniform(0.9, 1.2)
        thf = rng.uniform(0, 2 * np.pi)
        thp = thf + rng.choice([-1.0, 1.0]) * rng.uniform(np.pi / 3, 2 * np.pi / 3)
        tcx = rng.uniform(14.0, 20.0)
        conflict = rng.random() < 0.5
        off = rng.uniform(-0.3, 0.3) if conflict else rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 6.0)
        uf = np.array([np.cos(thf), np.sin(thf)])
        up = np.array([np.cos(thp), np.sin(thp)])
        npd = np.array([-up[1], up[0]])
        lat = np.zeros(len(t))                       # pedestrian lateral OU wiggle (sigma ~0.15 m)
        e = rng.standard_normal(len(t))
        for k in range(1, len(t)):
            lat[k] = lat[k - 1] - lat[k - 1] * DT_TRUTH / 2.0 + 0.15 * math.sqrt(DT_TRUTH) * e[k]
        pf = cp[None, :] + vf * (t - tcx)[:, None] * uf[None, :]
        pp = cp[None, :] + vp * (t - tcx - off)[:, None] * up[None, :] + lat[:, None] * npd[None, :]
        d = np.hypot(*(pf - pp).T)
        k = int(np.argmin(d))
        if (conflict and d[k] < CONFLICT_DMIN) or ((not conflict) and d[k] > SAFE_DMIN):
            return pf, pp, conflict, t[k], float(d[k])


def _alert_time(of, op, ev):
    """First persistent alert time from fused tracks (constant-velocity CPA within T_ALERT)."""
    run = 0
    for i in range(len(ev)):
        if not (np.isfinite(of[i, 0]) and np.isfinite(op[i, 0])):
            run = 0
            continue
        rx, ry = op[i, 0] - of[i, 0], op[i, 1] - of[i, 1]
        wx, wy = op[i, 2] - of[i, 2], op[i, 3] - of[i, 3]
        ww = wx * wx + wy * wy
        tc = -(rx * wx + ry * wy) / ww if ww > 1e-9 else 0.0
        if 0.0 < tc <= T_ALERT:
            dc = math.hypot(rx + wx * tc, ry + wy * tc)
        else:
            dc = math.hypot(rx, ry)      # receding or beyond the window: current separation
        hit = dc < D_ALERT
        run = run + 1 if hit else 0
        if run >= ALERT_PERSIST:
            return ev[i]
    return None


def run_e3(seed, episodes, out_dir, conds=("C0_nominal", "C1_night", "C3_multipath", "C5_compound")):
    T = 30.0
    fk, pd_ = AGENT_TYPES["forklift"], AGENT_TYPES["pedestrian"]
    rows, raw = [], []
    for ci, cname in enumerate(conds):
        cmod = CONDITIONS[cname]
        acc = {s: dict(lead=[], det=0, ncf=0, fa=0, nsafe=0) for s in STRATEGIES}
        for ep in range(episodes):
            rng = np.random.default_rng([seed, 3, ci, ep])
            pf, pp, conflict, t_min, dmin = gen_crossing(rng, T)
            mf = {s: gen_meas(rng, pf, T, s, cmod.get(s, {}), True) for s in SENSOR_LIST}
            mp = {s: gen_meas(rng, pp, T, s, cmod.get(s, {}), False) for s in SENSOR_LIST}
            ev = np.arange(WARMUP, T - 0.5, 1.0 / EVAL_HZ)
            for sname, scfg in STRATEGIES.items():
                sens = scfg["sensors"] or SENSOR_LIST
                of, _ = run_filter(merge(mf, sens), scfg, ev, fk["q_sig"] ** 2)
                op, _ = run_filter(merge(mp, sens), scfg, ev, pd_["q_sig"] ** 2)
                ta = _alert_time(of, op, ev)
                a = acc[sname]
                if conflict:
                    a["ncf"] += 1
                    if ta is not None and ta <= t_min:
                        a["det"] += 1
                        a["lead"].append(t_min - ta)
                else:
                    a["nsafe"] += 1
                    if ta is not None:
                        a["fa"] += 1
                raw.append(dict(condition=cname, strategy=sname, episode=ep, conflict=conflict,
                                d_min=dmin, t_min=t_min, t_alert=ta if ta is not None else float("nan")))
        for sname, a in acc.items():
            L = np.array(a["lead"]) if a["lead"] else np.array([np.nan])
            dlo, dhi = wilson(a["det"], a["ncf"])
            flo, fhi = wilson(a["fa"], a["nsafe"])
            rows.append(dict(condition=cname, strategy=sname,
                             n_conflict=a["ncf"], n_safe=a["nsafe"],
                             detection_rate=a["det"] / max(a["ncf"], 1),
                             det_ci_lo=dlo, det_ci_hi=dhi,
                             false_alarm_rate=a["fa"] / max(a["nsafe"], 1),
                             fa_ci_lo=flo, fa_ci_hi=fhi,
                             lead_median_s=float(np.nanmedian(L)),
                             lead_p10_s=float(np.nanpercentile(L, 10)),
                             p_lead_ge_2s=float(np.sum(np.array(a["lead"]) >= 2.0)) / max(a["ncf"], 1)
                             if a["lead"] else 0.0))
    _write_csv(os.path.join(out_dir, "e3_nearmiss.csv"), rows)
    _write_csv(os.path.join(out_dir, "e3_nearmiss_raw.csv"), raw)
    return rows, raw


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ------------------------------------------------------------------ output helpers
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
        return "" if not np.isfinite(v) else f"{v:.6g}"
    return str(v)


def make_figures(out_dir, e1, e2, e3, raw3, err_c5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # E1
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    colors = dict(yard_tractor="#1f4e79", forklift="#c55a11", amr="#548235", pedestrian="#7f6000")
    for name in AGENT_TYPES:
        r = [x for x in e1 if x["agent"] == name]
        tau = [x["tau_ms"] for x in r]
        ax.plot(tau, [x["naive_mean_err_m"] for x in r], "-o", ms=3, color=colors[name],
                label=f"{name} (v={AGENT_TYPES[name]['v_max']} m/s) naive")
        ax.plot(tau, [x["ts_aware_mean_err_m"] for x in r], "--", color=colors[name], lw=1)
        ax.plot(tau, [x["analytic_bias_m"] for x in r], ":", color=colors[name], lw=1)
    ax.axhline(E_BUDGET, color="k", lw=0.8)
    ax.axvline(LAT_BUDGET * 1000, color="grey", lw=0.8, ls="-.")
    ax.text(LAT_BUDGET * 1000 + 3, E_BUDGET + 0.035, "KPI3 0.5 m (horizontal) / KPI10 50 ms (vertical)", fontsize=7)
    ax.set_xlabel("un-modelled latency tau [ms]")
    ax.set_ylabel("mean position error [m]")
    ax.set_title("E1  Latency-error budget: naive (solid), timestamp-aware (dashed), v*tau (dotted)",
                 fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_e1_latency_error.png"), dpi=200)
    plt.close(fig)

    # E2 heatmaps
    conds, strs = list(CONDITIONS), list(STRATEGIES)
    P = np.array([[next(r["p_le"] for r in e2 if r["condition"] == c and r["strategy"] == s)
                   for s in strs] for c in conds])
    RR = np.array([[next(r["robustness_ratio"] for r in e2 if r["condition"] == c and r["strategy"] == s)
                    for s in strs] for c in conds])
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.6), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, M, title in ((axs[0], P, "P(error <= 0.5 m)"), (axs[1], RR, "robustness ratio vs C0 (KPI11 proxy)")):
        im = ax.imshow(np.clip(M, 0, 1), vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(strs)), [s.split("_", 1)[0] for s in strs])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if M[i, j] < 0.8 else "k")
        ax.set_title(title, fontsize=9)
    axs[0].set_yticks(range(len(conds)), conds)
    fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02)
    fig.savefig(os.path.join(out_dir, "fig_e2_strategy_condition.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # E2 CDF under compound condition
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for s in strs:
        e = err_c5[s]
        e = np.sort(e[np.isfinite(e)])
        if e.size:
            ax.plot(e, np.arange(1, e.size + 1) / len(err_c5[s]), label=s, lw=1.2)
    ax.axvline(E_BUDGET, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e2)
    ax.set_xlabel("position error [m] (log)")
    ax.set_ylabel("cumulative fraction (no-track counted as failure)")
    ax.set_title("E2  Error CDF, C5 compound degradation", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_e2_cdf_compound.png"), dpi=200)
    plt.close(fig)

    # E3: lead time (median, p10-p90) and false alarm rate with Wilson 95% CI
    c3 = list(dict.fromkeys(r["condition"] for r in e3))
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.4))
    w = 0.15
    for j, s in enumerate(strs):
        x = np.arange(len(c3)) + (j - 2) * w
        med, lo, hi, fa, flo, fhi = [], [], [], [], [], []
        for c in c3:
            L = np.array([r["t_min"] - r["t_alert"] for r in raw3
                          if r["condition"] == c and r["strategy"] == s and r["conflict"]
                          and np.isfinite(r["t_alert"]) and r["t_alert"] <= r["t_min"]])
            med.append(np.median(L) if L.size else np.nan)
            lo.append(np.percentile(L, 10) if L.size else np.nan)
            hi.append(np.percentile(L, 90) if L.size else np.nan)
            row = next(r for r in e3 if r["condition"] == c and r["strategy"] == s)
            fa.append(row["false_alarm_rate"])
            flo.append(row["fa_ci_lo"])
            fhi.append(row["fa_ci_hi"])
        med, lo, hi = np.array(med), np.array(lo), np.array(hi)
        axs[0].errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", ms=3, capsize=2, label=s)
        fa, flo, fhi = np.array(fa), np.array(flo), np.array(fhi)
        axs[1].bar(x, fa, w, yerr=[fa - flo, fhi - fa], capsize=2, label=s)
    axs[0].set_ylabel("alert lead time before closest approach [s]")
    axs[0].set_title("E3  lead time: median and p10-p90 (conflicts)", fontsize=9)
    axs[0].axhline(T_ALERT, color="grey", lw=0.8, ls=":")
    axs[1].set_ylabel("false alarm rate")
    axs[1].set_title("E3  false alarms on safe passes (Wilson 95% CI)", fontsize=9)
    axs[1].set_ylim(0, 1.0)
    for ax in axs:
        ax.set_xticks(range(len(c3)), c3, fontsize=7)
    axs[1].legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_e3_nearmiss.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--e1-episodes", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=20, help="E2 episodes per condition (x4 agents)")
    ap.add_argument("--nm-episodes", type=int, default=80, help="E3 episodes per condition")
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--profile", default="airport_yard", choices=["airport_yard", "port_terminal"])
    ap.add_argument("--calib", default="", help="calibration JSON (camera_port / lidar_kr blocks)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    apply_profile(args.profile)
    applied = apply_calib(args.calib) if args.calib else None
    print("profile:", args.profile, "| calibration:", applied)
    t0 = time.time()
    e1 = run_e1(args.seed, args.e1_episodes, args.out)
    print(f"[E1] done {time.time() - t0:.1f}s")
    e2, per_agent, cpu_rows, err_c5 = run_e2(args.seed, args.episodes, args.out)
    print(f"[E2] done {time.time() - t0:.1f}s")
    e3, raw3 = run_e3(args.seed, args.nm_episodes, args.out)
    print(f"[E3] done {time.time() - t0:.1f}s")
    budget = [dict(agent=k, v_max=a["v_max"], band_R_m=a["v_max"] * T_HORIZON,
                   tau_max_ms_for_0p5m=1000 * E_BUDGET / a["v_max"],
                   bias_at_50ms_m=a["v_max"] * LAT_BUDGET) for k, a in AGENT_TYPES.items()]
    _write_csv(os.path.join(args.out, "analytic_budget.csv"), budget)
    summary = dict(seed=args.seed, profile=args.profile, calib=args.calib or None, calib_applied=applied,
                   e1_episodes=args.e1_episodes, e2_episodes=args.episodes,
                   e3_episodes=args.nm_episodes, analytic_budget=budget, e1=e1, e2=e2,
                   e2_cpu=cpu_rows, e3=e3, runtime_s=time.time() - t0,
                   note="Conceptual simulation; parameters are assumptions, not measurements.")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    if not args.no_fig:
        make_figures(args.out, e1, e2, e3, raw3, err_c5)
    print(f"done in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
