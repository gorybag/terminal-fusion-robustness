#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nf_sim.py - NF-Synth: near-field multimodal synthetic data generator (infrastructure edge view)

What it models (2-D ground plane, edge sensors on poles)
  scene      cargo yard: container/ULD stack rows with lanes (occlusion + RTK multipath sources)
  agents     yard tractors, forklifts (GNSS-RTK), workers, AMRs; waypoint motion in free space
  sensors    LiDAR / RGB camera / thermal on edge nodes, GNSS-RTK on vehicles
             - range-dependent detection probability p(r) = 1/(1+exp(-(a+b r)))  (calibratable,
               see nf_calib_nuscenes.py), 2-D line-of-sight occlusion by stacks
             - systematic errors: LiDAR/camera near-side bias toward the sensor, thermal hotspot
               offset (engine), RTK float/multipath jumps near stacks (environment-driven)
             - latency (capture -> arrival), sensor clock offset/drift, outages
             - quality tag: reported sigma with imperfect self-assessment
  scans      2-D LiDAR range scans for occupancy mapping (fog: shorter range, ghost returns)
  output     per-sensor streams + ground truth, written as CSV (see write_episode)

All parameters are ASSUMPTIONS for a conceptual benchmark; calibrate detection curves with public
data (nf_calib_nuscenes.py) and latency/noise with the actual hardware before final claims.
2-D line of sight ignores pole height, i.e. it is conservative for low obstacles.
"""
import json
import math
import os

import numpy as np

CLASSES = {
    "tractor":  dict(v_max=5.0, acc=0.8, turn=0.6, length=5.0, width=2.2, radius=2.0, gnss=True, warm=True, q_sig=1.5),
    "forklift": dict(v_max=4.2, acc=1.0, turn=0.8, length=3.0, width=1.2, radius=1.2, gnss=True, warm=True, q_sig=1.5),
    "worker":   dict(v_max=1.2, acc=0.6, turn=2.0, length=0.5, width=0.5, radius=0.3, gnss=False, warm=True, q_sig=0.8),
    "amr":      dict(v_max=1.5, acc=0.4, turn=1.0, length=1.2, width=0.8, radius=0.6, gnss=False, warm=False, q_sig=0.6),
}
# Scenario profiles. Class keys stay the same; in the port profile they stand for
#   tractor = yard tractor (YT, RTK tag), forklift = reach stacker (RS, RTK tag),
#   amr = external truck (XT, no terminal RTK), worker = lashing/checker staff.
# 25 km/h (6.9 m/s) is a typical posted terminal speed limit (e.g. PSA terminal rules); check the target terminal.
PROFILES = {
    "airport_yard": dict(classes={}, mix=(("tractor", 2), ("forklift", 2), ("worker", 3), ("amr", 1)),
                         labels=dict(tractor="yard tractor", forklift="forklift", worker="worker", amr="cargo AMR")),
    "port_terminal": dict(classes=dict(tractor=dict(v_max=6.9, acc=1.0), forklift=dict(v_max=4.2, acc=0.8, length=6.0, width=3.0),
                                       amr=dict(v_max=6.9, acc=1.0, turn=0.6, length=5.0, width=2.5, gnss=False, warm=True, q_sig=1.5)),
                          mix=(("tractor", 3), ("forklift", 1), ("amr", 1), ("worker", 3)),
                          labels=dict(tractor="yard tractor (YT)", forklift="reach stacker (RS)", worker="worker",
                                      amr="external truck (XT)")),
}
PROFILE = "airport_yard"
DEFAULT_MIX = PROFILES[PROFILE]["mix"]


def apply_profile(name):
    """Switch the scenario profile in place (class kinematics, GNSS availability, agent mix)."""
    global PROFILE, DEFAULT_MIX
    for cls, upd in PROFILES[name]["classes"].items():
        CLASSES[cls].update(upd)
    PROFILE, DEFAULT_MIX = name, PROFILES[name]["mix"]
    return PROFILES[name]


MODALITIES = ["lidar", "camera", "thermal", "gnss"]
MID = {m: i for i, m in enumerate(MODALITIES)}

SENSORS = {
    # rate Hz; latency mean/sd s; noise sigma = s0 + s1*r (m); logistic detection a + b*r; max range m
    "lidar":   dict(rate=10.0, lat_mu=0.060, lat_sd=0.015, s0=0.05, s1=0.002, a=5.0, b=-0.06, rmax=60.0, bias=0.08),
    "camera":  dict(rate=15.0, lat_mu=0.045, lat_sd=0.010, s0=0.10, s1=0.010, a=4.0, b=-0.05, rmax=70.0, bias=0.25),
    "thermal": dict(rate=9.0,  lat_mu=0.080, lat_sd=0.015, s0=0.15, s1=0.012, a=3.5, b=-0.05, rmax=60.0, bias=0.30),
    "gnss":    dict(rate=10.0, lat_mu=0.120, lat_sd=0.030, s0=0.03, s1=0.000, a=6.0, b=0.0,   rmax=1e9,  bias=0.0),
}
CONDITIONS = {
    "nominal": {},
    "night":   dict(camera=dict(da=-3.0, smul=1.6)),
    "fog":     dict(lidar=dict(da=-2.0, smul=2.0, rmax=35.0, ghost=0.05), camera=dict(da=-1.5, smul=1.4),
                    thermal=dict(da=-0.7, smul=1.2)),
}
CLOCKS = {  # per-sensor constant offset range (s) and drift range (ppm)
    "perfect": (0.0, 0.0), "ptp": (1e-4, 1.0), "ntp": (5e-3, 5.0), "none": (8e-2, 50.0),
}


# ------------------------------------------------------------------ scene
def yard_layout(seed=0, half=50.0):
    """Stack rows at y = -30, -16, 16, 30 (lanes between them) plus two ULD clusters near the centre."""
    rng = np.random.default_rng(seed)
    boxes = []
    for row_y in (-30.0, -16.0, 16.0, 30.0):
        x = -44.0 + rng.uniform(0, 3)
        while x < 40.0:
            L, W = rng.uniform(6.0, 12.5), 2.6
            boxes.append((x, row_y - W / 2, min(x + L, 46.0), row_y + W / 2))
            x += L + rng.uniform(3.0, 6.0)
    for cx, cy in ((-22.0, 3.0), (24.0, -4.0)):
        boxes.append((cx - 2.0, cy - 1.6, cx + 2.0, cy + 1.6))
    return dict(name="yard", half=half, boxes=np.array(boxes, float))


NODE_SETS = {  # edge-node (pole) positions in the yard
    "N1": [(0.0, 0.0)],
    "N2": [(-12.0, 9.0), (12.0, -9.0)],
    "N3": [(0.0, 0.0), (-30.0, 23.0), (30.0, -23.0)],
}


def _inside_any(p, boxes, margin):
    x, y = p
    return bool(((x > boxes[:, 0] - margin) & (x < boxes[:, 2] + margin)
                 & (y > boxes[:, 1] - margin) & (y < boxes[:, 3] + margin)).any())


def seg_hits(p0, p1, boxes, margin=0.0):
    """Slab test of one segment against all boxes -> bool per box."""
    d = np.where(np.abs(p1 - p0) < 1e-12, 1e-12, p1 - p0)
    t1 = (boxes[:, :2] - margin - p0) / d
    t2 = (boxes[:, 2:] + margin - p0) / d
    tmin = np.minimum(t1, t2).max(axis=1)
    tmax = np.maximum(t1, t2).min(axis=1)
    return (tmax >= np.maximum(tmin, 0.0)) & (tmin <= 1.0)


def los_many(s, P, boxes):
    """Line of sight from a fixed sensor s (2,) to many points P (K,2) -> bool (K,)."""
    if len(boxes) == 0:
        return np.ones(len(P), bool)
    d = P - s[None, :]
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    lo = boxes[None, :, :2] - s[None, None, :]
    hi = boxes[None, :, 2:] - s[None, None, :]
    t1 = lo / d[:, None, :]
    t2 = hi / d[:, None, :]
    tmin = np.minimum(t1, t2).max(axis=2)
    tmax = np.maximum(t1, t2).min(axis=2)
    return ~(((tmax >= np.maximum(tmin, 0.0)) & (tmin <= 1.0)).any(axis=1))


# ------------------------------------------------------------------ agents
def gen_agent(rng, cls, layout, T, dt=0.05):
    """Waypoint motion in free space: accelerate to a cruise speed, turn toward the waypoint,
    brake before it, sometimes pause. Returns dict(t, pos, vel, heading, cls)."""
    c, boxes, half = CLASSES[cls], layout["boxes"], layout["half"] - 3.0
    margin = c["radius"] + 0.6
    inflated = boxes + np.array([-margin, -margin, margin, margin])

    def rand_free(k=256):
        P = rng.uniform(-half, half, (k, 2))
        ok = ~(((P[:, None, 0] > boxes[None, :, 0] - margin) & (P[:, None, 0] < boxes[None, :, 2] + margin)
                & (P[:, None, 1] > boxes[None, :, 1] - margin) & (P[:, None, 1] < boxes[None, :, 3] + margin)).any(1))
        if not ok.any():
            raise RuntimeError("no free space")
        return P[ok]
    n = int(round(T / dt)) + 1
    t = np.arange(n) * dt
    pos, vel, hd = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros(n)
    p = rand_free()[0]
    heading, v, wp, cruise, pause = rng.uniform(0, 2 * math.pi), 0.0, None, 0.0, 0.0
    for k in range(n):
        if wp is None and pause <= 0:
            cands = rand_free()
            cands = cands[np.hypot(*(cands - p).T) > 6.0]
            if len(cands):
                ok = los_many(p, cands, inflated)             # straight path clear of inflated stacks
                if ok.any():
                    wp, cruise = cands[np.argmax(ok)], rng.uniform(0.5, 1.0) * c["v_max"]
            if wp is None:
                pause = 1.0                      # no reachable waypoint now: wait instead of re-searching every step
        if pause > 0:
            pause -= dt
            v = max(0.0, v - c["acc"] * dt)
        elif wp is not None:
            dvec = wp - p
            dist = math.hypot(*dvec)
            want = math.atan2(dvec[1], dvec[0])
            err = (want - heading + math.pi) % (2 * math.pi) - math.pi
            heading += max(-c["turn"] * dt, min(c["turn"] * dt, err))
            target = cruise if dist > v * v / (2 * c["acc"]) + 0.5 else 0.3
            if abs(err) > 1.0:
                target = min(target, 0.4 * c["v_max"])
            v = max(0.0, min(c["v_max"], v + max(-c["acc"] * dt, min(c["acc"] * dt, target - v))))
            if dist < 0.8:
                wp = None
                pause = rng.exponential(2.0) if rng.random() < 0.4 else 0.0
        step = np.array([math.cos(heading), math.sin(heading)]) * v * dt
        if not _inside_any(p + step, boxes, c["radius"]):
            p = p + step
        else:
            v, wp, pause = 0.0, None, 0.5
        pos[k], vel[k], hd[k] = p, (math.cos(heading) * v, math.sin(heading) * v), heading
    return dict(t=t, pos=pos, vel=vel, heading=hd, cls=cls, dt=dt)


def make_agents(rng, layout, T, mix=None):
    mix = mix or DEFAULT_MIX
    agents = []
    for cls, k in mix:
        for _ in range(k):
            a = gen_agent(rng, cls, layout, T)
            a["id"] = len(agents)
            agents.append(a)
    return agents


# ------------------------------------------------------------------ sensors
def _logistic(a, b, r):
    return 1.0 / (1.0 + np.exp(-(a + b * r)))


def _intervals(rng, T, frac, mean_dur):
    if frac <= 0:
        return []
    mean_on = mean_dur * (1 - frac) / frac
    out, t = [], rng.exponential(mean_on)
    while t < T:
        d = rng.exponential(mean_dur)
        out.append((t, t + d))
        t += d + rng.exponential(mean_on)
    return out


def _inside(t, ints):
    m = np.zeros(t.shape, bool)
    for a, b in ints:
        m |= (t >= a) & (t < b)
    return m


def apply_calib(sensors, calib, cls_group):
    """Override logistic detection curves from nf_calib_nuscenes.py output (day condition)."""
    s = json.loads(json.dumps(sensors))
    if not calib:
        return s
    for mod, key in (("lidar", "lidar"), ("camera", "camera_vis40")):
        c = calib.get(key, {}).get(cls_group, {}).get("day")
        if c and c.get("n", 0) >= 50:
            s[mod]["a"], s[mod]["b"] = c["a"], c["b"]
    cam = calib.get("camera_port", {}).get("day")            # aihub_port_prep.py convert --mode full (vehicles only)
    if cam and "a" in cam and "b" in cam and cls_group == "vehicle":
        s["camera"]["a"], s["camera"]["b"] = cam["a"], cam["b"]
    return s


def sense(rng, agent, mod, node, layout, T, cond="nominal", clock="perfect", outage=0.0, calib=None):
    """Measurements of one agent by one modality (node = sensor position, None for on-board GNSS).
    Returns dict of arrays: t_cap, t_stamp, t_arr, zx, zy, sig_rep, sig_nom, sid."""
    c = CLASSES[agent["cls"]]
    if mod == "gnss" and not c["gnss"]:
        return None
    grp = "pedestrian" if agent["cls"] == "worker" else "vehicle"
    S = apply_calib(SENSORS, calib, grp)[mod]
    md = CONDITIONS[cond].get(mod, {})
    if calib and mod == "camera" and grp == "vehicle":          # port-image condition offsets (night, rain->fog)
        cc = calib.get("camera_port", {}).get("conditions", {}).get(cond)
        if cc and "da" in cc:
            md = dict(md, da=cc["da"])
    if calib and mod == "lidar":                                 # Korean-road LiDAR offsets (adrive3d_prep.py)
        lc = calib.get("lidar_kr", {}).get("conditions", {}).get(cond)
        if lc and "da" in lc:
            md = dict(md, da=lc["da"])
    period = 1.0 / S["rate"]
    tc = np.arange(rng.uniform(0, period), T - 0.4, period)
    k = np.minimum(np.round(tc / agent["dt"]).astype(int), len(agent["t"]) - 1)
    p, hd = agent["pos"][k], agent["heading"][k]
    if mod == "gnss":
        r = np.zeros(len(tc))
        keep = np.ones(len(tc), bool)
    else:
        s = np.asarray(node, float)
        rel = s[None, :] - p
        r = np.hypot(rel[:, 0], rel[:, 1])
        u = rel / np.maximum(r, 1e-6)[:, None]
        pdet = _logistic(S["a"] + md.get("da", 0.0), S["b"], r)
        if mod == "thermal" and not c["warm"]:
            pdet *= 0.2
        keep = (rng.random(len(tc)) < pdet) & (r <= md.get("rmax", S["rmax"])) & los_many(s, p, layout["boxes"])
    smul = md.get("smul", 1.0)
    sig_nom = S["s0"] + S["s1"] * r
    sig = sig_nom * smul
    z = p + rng.standard_normal(p.shape) * sig[:, None]
    hv = np.c_[np.cos(hd), np.sin(hd)]
    if mod in ("lidar", "camera"):   # near-side bias toward the sensor, proportional to projected extent
        ext = np.abs((u * hv).sum(1)) * c["length"] + np.abs(u[:, 0] * hv[:, 1] - u[:, 1] * hv[:, 0]) * c["width"]
        z += (S["bias"] * ext)[:, None] * u
    if mod == "thermal" and agent["cls"] in ("tractor", "forklift"):
        z -= (S["bias"] * c["length"]) * hv           # engine hotspot behind the vehicle centre
    sig_rep = sig.copy()
    if mod == "lidar" and md.get("ghost", 0) > 0:      # fog ghosts, not self-reported
        g = rng.random(len(tc)) < md["ghost"]
        z[g] += rng.normal(0, 1.5, (int(g.sum()), 2))
        sig_rep = sig_nom.copy()
    if mod == "gnss":                                  # RTK float / multipath near steel stacks
        boxes = layout["boxes"]
        dx = np.maximum(np.maximum(boxes[None, :, 0] - p[:, None, 0], 0), p[:, None, 0] - boxes[None, :, 2])
        dy = np.maximum(np.maximum(boxes[None, :, 1] - p[:, None, 1], 0), p[:, None, 1] - boxes[None, :, 3])
        near = np.hypot(dx, dy).min(axis=1) < 6.0
        fl = near & (rng.random(len(tc)) < 0.5)
        z[fl] = p[fl] + rng.standard_normal((int(fl.sum()), 2)) * 0.5
        sig_rep = np.where(fl, 0.5, sig_rep)
        jm = near & ~fl & (rng.random(len(tc)) < 0.05)
        ang = rng.uniform(0, 2 * math.pi, int(jm.sum()))
        z[jm] += np.c_[np.cos(ang), np.sin(ang)] * rng.uniform(1.5, 4.0, int(jm.sum()))[:, None]
    sig_rep = sig_rep * np.exp(rng.normal(0, 0.2, len(tc)))
    keep &= ~_inside(tc, _intervals(rng, T, outage, 2.0))
    off_rng, drift_rng = CLOCKS[clock]
    off, drift = rng.uniform(-off_rng, off_rng), rng.uniform(-drift_rng, drift_rng) * 1e-6
    lat = np.maximum(0.005, rng.normal(S["lat_mu"], S["lat_sd"], len(tc)))
    return dict(t_cap=tc[keep], t_stamp=(tc + off + drift * tc)[keep], t_arr=(tc + lat)[keep],
                zx=z[keep, 0], zy=z[keep, 1], sig_rep=sig_rep[keep], sig_nom=sig_nom[keep],
                sid=np.full(int(keep.sum()), MID[mod]))


def sense_agent(rng, agent, node, layout, T, cond="nominal", clock="perfect", outage=0.0, calib=None):
    return {m: sense(rng, agent, m, node, layout, T, cond, clock, outage, calib) for m in MODALITIES}


# ------------------------------------------------------------------ LiDAR range scans (occupancy output)
def lidar_scan(rng, s, agents_xy_r, boxes, n_rays=360, rmax=60.0, fog=False):
    """2-D range scan from s. agents_xy_r: (K,3) [x, y, radius]. Returns angles, ranges (inf = no return)."""
    th = np.linspace(0, 2 * math.pi, n_rays, endpoint=False)
    d = np.c_[np.cos(th), np.sin(th)]
    dd = np.where(np.abs(d) < 1e-12, 1e-12, d)
    t1 = (boxes[None, :, :2] - s[None, None, :]) / dd[:, None, :]
    t2 = (boxes[None, :, 2:] - s[None, None, :]) / dd[:, None, :]
    tmin = np.minimum(t1, t2).max(axis=2)
    tmax = np.maximum(t1, t2).min(axis=2)
    hit = (tmax >= np.maximum(tmin, 0.0))
    rng_box = np.where(hit & (tmin > 0), tmin, np.inf).min(axis=1)
    rr = rng_box
    if len(agents_xy_r):
        c = agents_xy_r[:, :2] - s[None, :]
        proj = d @ c.T                                    # rays x agents
        perp2 = (c ** 2).sum(1)[None, :] - proj ** 2
        rad2 = agents_xy_r[:, 2][None, :] ** 2
        ok = (proj > 0) & (perp2 <= rad2)
        ra = np.where(ok, proj - np.sqrt(np.maximum(rad2 - perp2, 0)), np.inf).min(axis=1)
        rr = np.minimum(rr, ra)
    lim = 35.0 if fog else rmax
    rr = np.where(rr <= lim, rr + rng.normal(0, 0.03, n_rays), np.inf)
    rr = np.where(rng.random(n_rays) < (0.10 if fog else 0.02), np.inf, rr)       # dropouts
    if fog:
        g = rng.random(n_rays) < 0.05
        rr = np.where(g, rng.uniform(2.0, 20.0, n_rays), rr)                         # ghost returns
    return th, rr


# ------------------------------------------------------------------ dataset writer
def write_episode(out_dir, ep, layout, agents, streams, meta):
    """Per-sensor CSV streams + ground truth (positions at 20 Hz) + manifest."""
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    gt = []
    for a in agents:
        gt.append(pd.DataFrame(dict(t=a["t"], agent=a["id"], cls=a["cls"], x=a["pos"][:, 0], y=a["pos"][:, 1],
                                    vx=a["vel"][:, 0], vy=a["vel"][:, 1], heading=a["heading"])))
    pd.concat(gt).round(4).to_csv(os.path.join(out_dir, f"ep{ep:03d}_gt.csv.gz"), index=False)
    for mod in MODALITIES:
        rows = []
        for aid, st in streams.items():
            s = st.get(mod)
            if s is None or len(s["t_cap"]) == 0:
                continue
            df = pd.DataFrame({k: s[k] for k in ("t_cap", "t_stamp", "t_arr", "zx", "zy", "sig_rep")})
            df["agent"] = aid
            rows.append(df)
        if rows:
            pd.concat(rows).sort_values("t_arr").round(5).to_csv(
                os.path.join(out_dir, f"ep{ep:03d}_{mod}.csv.gz"), index=False)
    with open(os.path.join(out_dir, f"ep{ep:03d}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta, boxes=layout["boxes"].round(3).tolist(), modalities=MODALITIES,
                       sensors=SENSORS, note="t_cap: true capture time (hidden from fusion in real use); "
                       "t_stamp: sensor clock; t_arr: arrival at the edge"), f, indent=1)


def generate(out_dir, n_episodes=5, T=60.0, cond="nominal", clock="perfect", nodes="N1", seed=0, calib=None):
    layout = yard_layout(seed)
    for ep in range(n_episodes):
        rng = np.random.default_rng([seed, ep])
        agents = make_agents(rng, layout, T)
        node = NODE_SETS[nodes][0]
        streams = {a["id"]: sense_agent(rng, a, node, layout, T, cond, clock, 0.0, calib) for a in agents}
        write_episode(out_dir, ep, layout, agents, streams,
                      dict(episode=ep, T=T, condition=cond, clock=clock, node=node, seed=seed))
    return out_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="nf_synth")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument("--cond", choices=list(CONDITIONS), default="nominal")
    ap.add_argument("--clock", choices=list(CLOCKS), default="perfect")
    ap.add_argument("--nodes", choices=list(NODE_SETS), default="N1")
    ap.add_argument("--calib", help="JSON from nf_calib_nuscenes.py")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cal = json.load(open(a.calib, encoding="utf-8")) if a.calib else None
    print(generate(a.out, a.episodes, a.T, a.cond, a.clock, a.nodes, a.seed, cal))
