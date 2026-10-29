#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nf_fusion.py - integration structures for near-field fusion (NF-Synth)

  EARLY  measurement-level fusion: one timestamp-aware constant-velocity KF over all modalities
         (out-of-sequence rollback, quality tags, robust soft gating)
  LATE   track-level fusion: one KF per modality, inverse-variance combination at output time
  LEARN  learned track-level fusion: MLP on per-modality tracks + availability masks, trained with
         or without modality dropout (ModDrop); predicts a residual on top of the LATE reference
  OGM    LiDAR log-odds occupancy grid (static-map output handed to L1/L2)
These are well-known techniques (OOSM, soft gating, track fusion, ModDrop, log-odds mapping);
they are used here as transparent baselines, not as the project's patent claims.
"""
import bisect
import math

import numpy as np

from nf_sim import MODALITIES

CHI2_99 = 9.2103
EARLY_CFG = dict(ts_aware=True, quality=True, gating=True)
NAIVE_CFG = dict(ts_aware=False, quality=False, gating=False)


class KF:
    """2-D constant-velocity KF with isotropic position measurements (x/y share one 2x2 covariance)."""
    __slots__ = ("q", "t", "px", "py", "vx", "vy", "P00", "P01", "P11", "init")

    def __init__(self, q):
        self.q = q
        self.reset()

    def reset(self):
        self.t = self.px = self.py = self.vx = self.vy = 0.0
        self.P00 = self.P01 = self.P11 = 0.0
        self.init = False

    def initialize(self, t, zx, zy, R):
        self.t, self.px, self.py, self.vx, self.vy = t, zx, zy, 0.0, 0.0
        self.P00, self.P01, self.P11, self.init = R, 0.0, 9.0, True

    def get(self):
        return (self.t, self.px, self.py, self.vx, self.vy, self.P00, self.P01, self.P11, self.init)

    def set(self, s):
        self.t, self.px, self.py, self.vx, self.vy, self.P00, self.P01, self.P11, self.init = s

    def predict(self, t):
        dt = t - self.t
        if dt <= 0.0:
            return
        q, dt2 = self.q, dt * dt
        P01, P11 = self.P01, self.P11
        self.P00 += 2.0 * dt * P01 + dt2 * P11 + q * dt2 * dt2 * 0.25
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
        self.P11 -= K1 * P01

    def extrap(self, t):
        dt = t - self.t
        P = self.P00 + 2 * dt * self.P01 + dt * dt * self.P11 + self.q * dt ** 4 / 4
        return (self.px + self.vx * dt, self.py + self.vy * dt, self.vx, self.vy, P, dt)


def _apply(kf, t, m, cfg):
    zx, zy, srep, snom = m
    R = (srep if cfg["quality"] else snom) ** 2
    if not kf.init:
        kf.initialize(t, zx, zy, R)
        return
    kf.predict(t)
    dx, dy = zx - kf.px, zy - kf.py
    if cfg["gating"]:
        nis = (dx * dx + dy * dy) / (kf.P00 + R)
        if nis > CHI2_99:                       # robust soft gate: down-weight, never reject
            R *= nis / CHI2_99
    kf.update(dx, dy, R)


def merge(meas, mods):
    """Concatenate the selected modalities' measurements, sorted by arrival time."""
    parts = [meas[m] for m in mods if meas.get(m) is not None and len(meas[m]["t_cap"])]
    if not parts:
        return None
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("t_cap", "t_stamp", "t_arr", "zx", "zy", "sig_rep", "sig_nom")}
    o = np.argsort(cat["t_arr"], kind="stable")
    return {k: v[o] for k, v in cat.items()}


def run_filter(M, cfg, ev, q, use_stamp=True):
    """Arrival-ordered processing (real-time causal). Timestamp-aware configs insert each measurement at
    its (sensor-clock) capture stamp and re-filter. Returns out (n,6): x, y, vx, vy, P_pos, age."""
    out = np.full((len(ev), 6), np.nan)
    if M is None:
        return out
    kf = KF(q)
    tcap = (M["t_stamp"] if use_stamp else M["t_cap"]).tolist()
    tarr = M["t_arr"].tolist()
    rows = list(zip(M["zx"].tolist(), M["zy"].tolist(), M["sig_rep"].tolist(), M["sig_nom"].tolist()))
    evl = ev.tolist()
    H_t, H_m, H_s, base, ei = [], [], [], None, 0
    for j in range(len(tarr)):
        ta = tarr[j]
        while ei < len(evl) and evl[ei] < ta:
            if kf.init:
                out[ei] = kf.extrap(evl[ei])
            ei += 1
        if not cfg["ts_aware"]:
            _apply(kf, ta, rows[j], cfg)
            continue
        tc = tcap[j]
        if base is not None and tc < base[0]:
            continue
        k = bisect.bisect_right(H_t, tc)
        H_t.insert(k, tc)
        H_m.insert(k, rows[j])
        H_s.insert(k, None)
        if k == 0:
            kf.reset() if base is None else kf.set(base)
        else:
            kf.set(H_s[k - 1])
        for i in range(k, len(H_t)):
            _apply(kf, H_t[i], H_m[i], cfg)
            H_s[i] = kf.get()
        if len(H_t) > 64:
            cut = bisect.bisect_left(H_t, ta - 0.6)
            if cut > 0:
                base = H_s[cut - 1]
                del H_t[:cut], H_m[:cut], H_s[:cut]
    while ei < len(evl):
        if kf.init:
            out[ei] = kf.extrap(evl[ei])
        ei += 1
    return out


def modality_tracks(meas, ev, q, cfg=EARLY_CFG):
    return {m: (run_filter(merge(meas, [m]), cfg, ev, q) if meas.get(m) is not None else None) for m in MODALITIES}


def late_fuse(tracks, mask, max_age=1.0):
    """Inverse-variance combination of the per-modality tracks in `mask` that were updated within max_age."""
    n = len(next(v for v in tracks.values() if v is not None))
    num, vnum, den = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros(n)
    for m in mask:
        o = tracks.get(m)
        if o is None:
            continue
        ok = np.isfinite(o[:, 0]) & (o[:, 5] <= max_age)
        w = np.where(ok, 1.0 / np.maximum(np.nan_to_num(o[:, 4], nan=1.0), 1e-6), 0.0)
        num += w[:, None] * np.nan_to_num(o[:, 0:2])
        vnum += w[:, None] * np.nan_to_num(o[:, 2:4])
        den += w
    fused = np.full((n, 4), np.nan)
    good = den > 0
    fused[good, :2] = num[good] / den[good, None]
    fused[good, 2:] = vnum[good] / den[good, None]
    return fused, good


def learn_features(tracks, mask, node, max_age=1.0):
    """Per tick: for each modality [available, dx, dy, log var, age] relative to the LATE reference, plus
    geometry [rx, ry, range, vx, vy, speed] relative to the edge node. Returns X, reference, valid mask."""
    ref, good = late_fuse(tracks, mask, max_age)
    n = len(ref)
    blocks = []
    for m in MODALITIES:
        f = np.zeros((n, 5))
        o = tracks.get(m)
        if o is not None and m in mask:
            ok = np.isfinite(o[:, 0]) & (o[:, 5] <= max_age) & good
            f[ok, 0] = 1.0
            f[ok, 1] = o[ok, 0] - ref[ok, 0]
            f[ok, 2] = o[ok, 1] - ref[ok, 1]
            f[ok, 3] = np.log(np.maximum(o[ok, 4], 1e-6))
            f[ok, 4] = o[ok, 5]
        blocks.append(f)
    g = np.zeros((n, 6))
    g[good, 0] = ref[good, 0] - node[0]
    g[good, 1] = ref[good, 1] - node[1]
    g[good, 2] = np.hypot(g[good, 0], g[good, 1])
    g[good, 3:5] = ref[good, 2:4]
    g[good, 5] = np.hypot(ref[good, 2], ref[good, 3])
    return np.hstack(blocks + [g]), ref, good


def build_training(series, moddrop, rng, n_aug=4, p_keep=0.6):
    """series: list of dict(tracks, truth, node). ModDrop adds random modality subsets per series."""
    X, Y = [], []
    for s in series:
        avail = [m for m in MODALITIES if s["tracks"].get(m) is not None]
        pats = [set(avail)]
        if moddrop:
            for _ in range(n_aug):
                keep = {m for m in avail if rng.random() < p_keep}
                pats.append(keep or {avail[int(rng.integers(len(avail)))]})
        for pat in pats:
            Xi, ref, good = learn_features(s["tracks"], pat, s["node"])
            if good.any():
                X.append(Xi[good])
                Y.append(np.clip(s["truth"][good] - ref[good, :2], -3.0, 3.0))
    return np.vstack(X), np.vstack(Y)


def train_mlp(X, Y, seed=0):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-4, learning_rate_init=1e-3,
                                       batch_size=256, max_iter=100, early_stopping=True, n_iter_no_change=8,
                                       random_state=seed))
    model.fit(X, Y)
    return model


def learned_fuse(model, tracks, mask, node):
    X, ref, good = learn_features(tracks, mask, node)
    out = np.full((len(ref), 2), np.nan)
    if good.any():
        out[good] = ref[good, :2] + model.predict(X[good])
    return out, good


# ------------------------------------------------------------------ occupancy grid (L1/L2 input)
class OccGrid:
    def __init__(self, half=50.0, res=0.5):
        self.half, self.res = half, res
        self.n = int(round(2 * half / res))
        self.L = np.zeros((self.n, self.n))

    def _idx(self, P):
        ij = np.floor((P + self.half) / self.res).astype(int)
        ok = (ij >= 0).all(1) & (ij < self.n).all(1)
        return ij[ok, 0] * self.n + ij[ok, 1]

    def integrate(self, s, th, rr, rmax=60.0, l_free=-0.4, l_occ=0.85):
        s = np.asarray(s, float)
        valid = np.isfinite(rr)
        if not valid.any():
            return
        d = np.c_[np.cos(th[valid]), np.sin(th[valid])]
        r = rr[valid]
        steps = np.arange(0.0, rmax, self.res * 0.5)
        mask = steps[None, :] < (r[:, None] - self.res)
        pts = s[None, None, :] + steps[None, :, None] * d[:, None, :]
        free = np.unique(self._idx(pts[mask]))
        occ = np.unique(self._idx(s[None, :] + r[:, None] * d))
        Lf = self.L.reshape(-1)
        Lf[free] += l_free
        Lf[occ] += l_occ
        np.clip(Lf, -5.0, 5.0, out=Lf)

    def evaluate(self, boxes, band_r=50.0, thr=1.0):
        c = (np.arange(self.n) + 0.5) * self.res - self.half
        X, Y = np.meshgrid(c, c, indexing="ij")
        inside = np.zeros_like(X, bool)
        for b in boxes:
            inside |= (X >= b[0]) & (X <= b[2]) & (Y >= b[1]) & (Y <= b[3])
        band = np.hypot(X, Y) <= band_r
        pad = np.pad(inside, 1, constant_values=False)
        nb_free = ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
        surface = inside & nb_free
        free_gt = ~inside & band
        occ, fre = self.L > thr, self.L < -thr
        return dict(free_coverage=float((fre & free_gt).sum() / max(free_gt.sum(), 1)),
                    surface_recall=float((occ & surface & band).sum() / max((surface & band).sum(), 1)),
                    false_occupied=float((occ & free_gt).sum() / max(free_gt.sum(), 1)))
