#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mmfi_dropout.py - real-data check of the missing-modality findings on MM-Fi (one environment, e.g. E01)

Task: per frame, estimate the person's 3-D position (mean of the 17 ground-truth joints) from each modality, then fuse
under every subset of modalities and compare the fusion structures used in the simulation:
  LATE           inverse-variance weighting of the per-modality estimates
  EARLY          constant-velocity Kalman filter over the frames, updated by each available per-modality estimate
  LEARN_noDrop   MLP residual over LATE with an availability mask, trained with the modalities as recorded
  LEARN_ModDrop  same MLP trained with random modality dropout (keep probability 0.6, at least one kept)
Per-modality estimators are ridge regressions fitted on separate training subjects, because the sensors are not
co-registered with the ground-truth frame:
  lidar   median xyz + log point count   (binary float64, N x 3)
  mmwave  median xyz + log point count   (binary float64, N x 5)
  rgb     mean / spread of the 2-D keypoints (.npy, 17 x 2)
  wifi    PCA of the CSI amplitude (.mat, key CSIamp, 3 x 114)
Frames where a modality returns nothing (e.g. empty radar frames) are treated as naturally missing.

Subjects: the per-modality estimators are cross-fitted over --train-subjects (leave one subject out), so the fusion
variances and MLPs are trained on honest out-of-subject estimates; --test-subjects are only evaluated.

Usage
  python mmfi_dropout.py --root <.../E01> --train-subjects S01-S07 --test-subjects S08-S10 \
         --mods lidar,mmwave,rgb,wifi --stride 2 --out-prefix mmfi
"""
import argparse
import glob
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FOLDER = {"lidar": "lidar", "mmwave": "mmwave", "rgb": "rgb", "wifi": "wifi-csi"}
SIM_RANKING = ["LEARN_ModDrop", "EARLY", "LATE", "LEARN_noDrop"]     # macro ranking in the simulation (v1 full run)
FPS = 10.0


def subjects(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += [f"S{i:02d}" for i in range(int(a[1:]), int(b[1:]) + 1)]
        elif part:
            out.append(part)
    return out


def _num(p):
    m = re.findall(r"\d+", os.path.basename(p))
    return int(m[-1]) if m else 0


def _points(path, cols):
    a = np.fromfile(path, dtype=np.float64)
    if a.size == 0 or a.size % cols:
        return None
    p = a.reshape(-1, cols)[:, :3]
    p = p[np.isfinite(p).all(1)]
    return p if len(p) >= 3 else None


def feat(mod, path):
    try:
        if mod in ("lidar", "mmwave"):
            p = _points(path, 3 if mod == "lidar" else 5)
            return None if p is None else np.r_[np.median(p, 0), np.log1p(len(p))]
        if mod == "rgb":
            k = np.asarray(np.load(path), float).reshape(-1, 2)
            if not np.isfinite(k).all() or np.abs(k).sum() == 0:
                return None
            return np.r_[k.mean(0), k.std(0), np.ptp(k[:, 1])]
        if mod == "wifi":
            from scipy.io import loadmat
            m = loadmat(path)
            key = "CSIamp" if "CSIamp" in m else [k for k in m if not k.startswith("__")][0]
            x = np.asarray(m[key], float).ravel()
            x[~np.isfinite(x)] = np.nan
            if np.isnan(x).all():
                return None
            return np.where(np.isnan(x), np.nanmean(x), x)
    except Exception:
        return None
    return None


def load(root, subs, mods, stride):
    rows = []
    for s in subs:
        for act in sorted(glob.glob(os.path.join(root, s, "A*"))):
            gtp = os.path.join(act, "ground_truth.npy")
            if not os.path.isfile(gtp):
                c = glob.glob(os.path.join(act, "*ground*truth*.npy"))
                if not c:
                    continue
                gtp = c[0]
            gt = np.load(gtp)
            files = {m: sorted(glob.glob(os.path.join(act, FOLDER[m], "*")), key=_num) for m in mods}
            T = min([len(gt)] + [len(f) for f in files.values() if f])
            for i in range(0, T, stride):
                r = dict(subject=s, seq=f"{s}/{os.path.basename(act)}", frame=i, target=gt[i].reshape(-1, 3).mean(0))
                for m in mods:
                    r[m] = feat(m, files[m][i]) if i < len(files[m]) else None
                rows.append(r)
    if not rows:
        raise SystemExit(f"no frames found under {root} for {subs}")
    return rows


def fit_estimators(rows, mods):
    est = {}
    for m in mods:
        X = [r[m] for r in rows if r[m] is not None]
        Y = [r["target"] for r in rows if r[m] is not None]
        if len(X) < 20:
            continue
        X = np.vstack(X)
        steps = [StandardScaler()] + ([PCA(n_components=min(20, X.shape[1], len(X) - 1))] if X.shape[1] > 20 else [])
        est[m] = make_pipeline(*steps, Ridge(alpha=1.0)).fit(X, np.vstack(Y))
    return est


def predict(rows, est):
    for r in rows:
        r["est"] = {m: (e.predict(r[m][None])[0] if r[m] is not None else None) for m, e in est.items()}


def late(r, S, var, prior):
    ws, xs = [], []
    for m in S:
        if r["est"].get(m) is not None:
            ws.append(1 / var[m]); xs.append(r["est"][m])
    return (np.average(np.vstack(xs), axis=0, weights=ws), True) if ws else (prior, False)


def early(seq_rows, S, var, prior, q=1.0):
    dt = None
    x = np.r_[prior, 0, 0, 0]
    P = np.diag([1, 1, 1, 1, 1, 1.0])
    out, last_f = [], None
    for r in seq_rows:
        dt = 0.1 if last_f is None else max((r["frame"] - last_f) / FPS, 1e-3)
        last_f = r["frame"]
        F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
        G = np.r_[0.5 * dt * dt * np.ones(3), dt * np.ones(3)]
        Q = q * np.diag(G ** 2)
        x = F @ x; P = F @ P @ F.T + Q
        H = np.c_[np.eye(3), np.zeros((3, 3))]
        for m in S:
            z = r["est"].get(m)
            if z is None:
                continue
            Sm = H @ P @ H.T + var[m] * np.eye(3)
            K = P @ H.T @ np.linalg.inv(Sm)
            x = x + K @ (z - H @ x); P = (np.eye(6) - K @ H) @ P
        out.append(x[:3].copy())
    return out


def mlp_features(r, S, var, prior, mods):
    p0, _ = late(r, S, var, prior)
    f = []
    for m in mods:
        e = r["est"].get(m) if m in S else None
        f += [1.0, *(e - p0), np.log(var[m])] if e is not None else [0.0, 0.0, 0.0, 0.0, 0.0]
    return np.r_[f, p0], p0


def train_mlp(rows, var, prior, mods, dropout, rng, n_aug=4, p_keep=0.6):
    X, Y = [], []
    for r in rows:
        avail = [m for m in mods if r["est"].get(m) is not None]
        if not avail:
            continue
        masks = [avail]
        if dropout:
            for _ in range(n_aug):
                keep = [m for m in avail if rng.random() < p_keep] or [avail[rng.integers(len(avail))]]
                masks.append(keep)
        for S in masks:
            f, p0 = mlp_features(r, S, var, prior, mods)
            X.append(f); Y.append(r["target"] - p0)
    net = MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=500, early_stopping=True, random_state=0)
    return make_pipeline(StandardScaler(), net).fit(np.vstack(X), np.clip(np.vstack(Y), -1.0, 1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="environment folder, e.g. .../E01 (contains S01..S10)")
    ap.add_argument("--train-subjects", default="S01-S07")
    ap.add_argument("--test-subjects", default="S08-S10")
    ap.add_argument("--mods", default="lidar,mmwave,rgb,wifi")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--eps", default="0.05,0.1,0.2")
    ap.add_argument("--main-eps", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="mmfi")
    a = ap.parse_args()
    mods = a.mods.split(",")
    rng = np.random.default_rng(a.seed)
    tr_subs = subjects(a.train_subjects)
    R_fuse = load(a.root, tr_subs, mods, a.stride)
    R_test = load(a.root, subjects(a.test_subjects), mods, a.stride)
    for sub in tr_subs:                                    # leave-one-subject-out estimates for the training set
        held = [r for r in R_fuse if r["subject"] == sub]
        if held:
            predict(held, fit_estimators([r for r in R_fuse if r["subject"] != sub], mods))
    est = fit_estimators(R_fuse, mods)
    mods = [m for m in mods if m in est]
    predict(R_test, est)
    for r in R_fuse:
        r["est"] = {m: r["est"].get(m) for m in mods}
    prior = np.vstack([r["target"] for r in R_fuse]).mean(0)
    var = {}
    for m in mods:
        e = [r["est"][m] - r["target"] for r in R_fuse if r["est"].get(m) is not None]
        var[m] = float(np.mean(np.sum(np.vstack(e) ** 2, 1)) / 3) if e else 1.0
    avail = {m: float(np.mean([r["est"].get(m) is not None for r in R_test])) for m in mods}
    print("per-modality RMS error (m):", {m: round(np.sqrt(3 * v), 3) for m, v in var.items()})
    print("natural availability in test frames:", {m: round(v, 3) for m, v in avail.items()})
    nets = {"LEARN_noDrop": train_mlp(R_fuse, var, prior, mods, False, rng),
            "LEARN_ModDrop": train_mlp(R_fuse, var, prior, mods, True, rng)}
    seqs = {}
    for r in R_test:
        seqs.setdefault(r["seq"], []).append(r)
    eps = [float(e) for e in a.eps.split(",")]
    rows = []
    for k in range(1, len(mods) + 1):
        for S in itertools.combinations(mods, k):
            err = {mth: [] for mth in ["EARLY", "LATE", "LEARN_noDrop", "LEARN_ModDrop", "PRIOR"]}
            for sq in seqs.values():
                pe = early(sq, S, var, prior)
                lat = [late(r, S, var, prior) for r in sq]
                F = np.vstack([mlp_features(r, S, var, prior, mods)[0] for r in sq])
                tgt = np.vstack([r["target"] for r in sq])
                P0 = np.vstack([p for p, _ in lat]); OK = np.array([ok for _, ok in lat])
                err["EARLY"] += list(np.linalg.norm(np.vstack(pe) - tgt, axis=1))
                err["LATE"] += list(np.linalg.norm(P0 - tgt, axis=1))
                err["PRIOR"] += list(np.linalg.norm(prior - tgt, axis=1))
                for nm, net in nets.items():
                    P = P0 + np.where(OK[:, None], net.predict(F), 0.0)
                    err[nm] += list(np.linalg.norm(P - tgt, axis=1))
            for mth, e in err.items():
                e = np.asarray(e)
                rows.append(dict(subset="+".join(S), k=k, method=mth, rmse=float(np.sqrt((e ** 2).mean())),
                                 **{f"p_le_{x}": float((e <= x).mean()) for x in eps}))
    df = pd.DataFrame(rows)
    df.to_csv(a.out_prefix + "_subsets.csv", index=False)
    key = f"p_le_{a.main_eps}"
    macro = df.groupby("method")[key].mean().sort_values(ascending=False)
    full = "+".join(mods)
    dep = {}
    for mth in SIM_RANKING:
        g = df[df.method == mth].set_index("subset")[key]
        dep[mth] = {m: round(float(g[full] - g["+".join(x for x in mods if x != m)]), 3) for m in mods} if len(mods) > 1 else {}
    order = [m for m in macro.index if m in SIM_RANKING]
    tau, p = kendalltau([SIM_RANKING.index(m) for m in order], range(len(order)))
    summ = dict(eps_main=a.main_eps, per_modality_rms_m={m: round(np.sqrt(3 * v), 3) for m, v in var.items()},
                availability=avail, macro=macro.round(4).to_dict(), ranking_mmfi=order, ranking_simulation=SIM_RANKING,
                kendall_tau_vs_simulation=round(float(tau), 3), dependency=dep,
                by_k=df[df.method.isin(SIM_RANKING)].groupby(["method", "k"])[key].mean().round(4).unstack().to_dict(orient="index"))
    json.dump(summ, open(a.out_prefix + "_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 600, "pdf.fonttype": 42})
    spec = [("LEARN_ModDrop", "Learned + modality dropout", "-", "D", "#000000", True), ("EARLY", "Early (KF)", "--", "o", "#404040", True),
            ("LATE", "Late (inverse variance)", "-.", "s", "#7a7a7a", False), ("LEARN_noDrop", "Learned (no dropout)", ":", "^", "#7a7a7a", False)]
    fig, ax = plt.subplots(figsize=(8.4 / 2.54, 7.0 / 2.54))
    ks = sorted(df.k.unique())
    for i, (m, lab, ls, mk, c, filled) in enumerate(spec):
        g = df[df.method == m].groupby("k")[key]
        x = np.asarray(ks) + (i - 1.5) * 0.08
        ax.vlines(x, g.min().values, g.max().values, color=c, lw=0.5)
        ax.plot(x, g.mean().values, ls, marker=mk, color=c, mfc=c if filled else "white", label=lab)
    ax.set_xticks(ks); ax.set_xlabel("Number of available modalities"); ax.set_ylabel(f"P(error ≤ {a.main_eps} m)")
    ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out_prefix}_degradation.{ext}", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
