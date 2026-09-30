#!/usr/bin/env python3
"""Ground-truth-free calibration of the loop-closure noise model from a short recording (about a minute).

Input: the graph dump (graph_s0.json) of a short mapping run on the robot's own data (any environment, normal motion
including turns), e.g. `scripts/map_and_reloc_rgbd.py --map <seq> --query <seq> --map-end 600 --skip-reloc --dump-graph`,
and optionally a trace with the pass-internal poses of a multi-view estimator (not needed for PnP).
No ground truth is used.  Three self-consistency statistics are exploited:

  * reference pairs registered in one feed-forward pass whose map relation is a single odometry edge (consecutive
    keyframes, ~0.2 m apart): the pass-relative pose and the odometry edge measure the same quantity, so their
    residual is (pair noise) + (one short odometry step); this gives sigma_pair(d) and, since a pair relation and a
    visual edge are the same kind of prediction, sigma_visual(d);
  * odometry-vs-visual innovations over short spans (visual edges to keyframes a few steps back): the residual of the
    measured relative pose against the odometry chain has covariance Sigma_visual(d) + Sigma_chain(k_t, k_r); with
    sigma_visual known, k_t and k_r are chosen so that the translation and rotation parts of the normalised
    innovation have unit variance (innovation-based adaptive estimation);
  * the map consistency model is set from the visual model (a map keyframe pair is related through verified edges).

Output: a YAML with the NoiseModelConfig fields (see cross/core/config.py), and a comparison with the ground-truth
fit when ground truth is available in the graph (validation only).

usage: python scripts/lc/calibrate_noise.py --graph graph_s0.json [--trace trace.json] --out configs/noise/<robot>.yaml
       [--frames 600] [--validate] [--snr 10] [--k-default auto|k|k_t,k_r]
       [--short-max-edges 10] [--short-max-theta 0.15] [--short-max-len 10] [--straight-max-theta 0.05]
       [--pair-max-theta 0.15] [--scale-max-edges 5] [--scale-min-len 1.0] [--long-min-len 1.0] [--turn-min-theta 0.3]
       [--bins 0,0.5,1,2,4,8,16,32,64] [--min-short 20]

The span-selection thresholds are metric (driving / indoor scale by default); for slow, short-range motion (e.g. a
platform at 0.2-0.5 m/s) lower --scale-min-len / --long-min-len (~0.3 m) and use finer --bins
(e.g. 0,0.1,0.2,0.4,0.8,1.6,3.2,6.4).  --k-default auto uses 1/(snr*sqrt(3)) for the odometry constants of the
chain-variance subtraction when --snr is given (the simulated-odometry noise of the recording), else 0.06 / 0.07.
The flag values are stored under 'params' in the sidecar JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import nnls
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cross.core.lc_verify import to_gtsam, logmap, residual, transport  # noqa: E402

P90_3 = math.sqrt(chi2.ppf(0.90, 3))

# span-selection thresholds of the calibration (CLI flags of the same names; the values are the historical constants)
DEFAULT_BINS = (0, 0.5, 1, 2, 4, 8, 16, 32, 64)
DEFAULT_K = (0.06, 0.07)          # odometry constants (k_t, k_r) assumed in the chain-variance subtraction = SNR-10 odometry
CAL_DEFAULTS = {
    "short_max_edges": 10,        # 'short' spans: odometry chain of at most this many edges ...
    "short_max_theta": 0.15,      # ... with a chain rotation below this (rad) ...
    "short_max_len": 10.0,        # ... and a chain length of at most this (m): visual translation model
    "straight_max_theta": 0.05,   # 2-3-edge spans with less rotation than this (rad): translation / rotation floors
    "pair_max_theta": 0.15,       # reference pairs of one pass: odometry chain (<= 3 edges) rotation below this (rad)
    "scale_max_edges": 5,         # metric scale: spans of at most this many edges ...
    "scale_min_len": 1.0,         # ... and a chain length of at least this (m)
    "long_min_len": 1.0,          # odometry k_t: spans of at least this chain length (m)
    "turn_min_theta": 0.3,        # odometry k_r: spans with at least this chain rotation (rad)
    "bins": DEFAULT_BINS,         # distance-bin edges (m) of the pair and visual fits
    "k_default": DEFAULT_K,       # (k_t, k_r) of the chain-variance subtraction
    "min_short": 20,              # hard stop: fewer short spans than this -> no calibration
}


def resolve_params(params: dict | None = None) -> dict:
    """Complete a partial parameter dict with the historical defaults (see CAL_DEFAULTS)."""
    out = dict(CAL_DEFAULTS)
    out.update({k: v for k, v in (params or {}).items() if v is not None})
    out["bins"] = tuple(float(b) for b in out["bins"])
    out["k_default"] = tuple(float(k) for k in out["k_default"])
    return out


def parse_k_default(spec: str, snr: float | None) -> tuple:
    """--k-default: 'auto' (1/(snr*sqrt(3)) for both constants when --snr is given, else 0.06 / 0.07), one number
    (used for both) or 'k_t,k_r'."""
    spec = str(spec).strip().lower()
    if spec == "auto":
        if snr is not None and snr > 0:
            k = 1.0 / (float(snr) * math.sqrt(3.0))
            return (k, k)
        return DEFAULT_K
    parts = [float(x) for x in spec.split(",") if x.strip()]
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) == 2:
        return (parts[0], parts[1])
    raise ValueError(f"--k-default: expected 'auto', a number or 'k_t,k_r', got {spec!r}")


def tail_sigma(norms):
    return float(np.percentile(norms, 90) / P90_3) if len(norms) else float("nan")


def fit_linear(d, norms, bins=DEFAULT_BINS, min_n=15):
    xs, ys, ws = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() >= min_n:
            xs.append(float(np.median(d[m]))); ys.append(tail_sigma(norms[m])); ws.append(math.sqrt(m.sum()))
    xs, ys, ws = np.asarray(xs), np.asarray(ys), np.asarray(ws)
    if len(xs) == 0:
        return None
    if len(xs) == 1:
        return {"a": float(ys[0]), "b": 0.0, "bins": [(float(xs[0]), float(ys[0]))]}
    A = np.stack([np.ones_like(xs), xs], 1) * ws[:, None]
    coef, _ = nnls(A, ys * ws)
    return {"a": float(coef[0]), "b": float(coef[1]), "bins": [(float(x), float(y)) for x, y in zip(xs, ys)]}


def load_json(p):
    t = Path(p).read_text()
    if str(p).endswith(".js"):
        t = t[t.index("]=") + 2:].rstrip().rstrip(";")
    return json.loads(t)


def calibrate(graph, trace, frames=None, params=None):
    prm = resolve_params(params)
    bins = prm["bins"]; k_def = prm["k_default"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    step = {n["id"]: n.get("step_created") for n in graph["nodes"]}
    odom = {(e["a"], e["b"]): e for e in graph["odom"]}
    nxt = {a: b for (a, b) in odom}
    if frames is not None:
        keep = {i for i, s in step.items() if s is None or s <= frames}
    else:
        keep = set(nodes)
    out = {}

    # ---- 1. pair statistic: consecutive keyframes registered in one pass vs their odometry edge
    d_pair, nt, nr = [], [], []
    for st in trace["steps"]:
        f = st.get("ffp")
        if not f or (frames is not None and st["i"] > frames):
            continue
        ids = f["ids"]; c2w = [to_gtsam(np.asarray(p)) for p in f["c2w"]]
        for i in range(len(ids)):
            for j in range(len(ids)):
                a, b = ids[i], ids[j]
                if (a, b) not in odom or not f["valid"][i] or not f["valid"][j] or a not in keep or b not in keep:
                    continue
                P = c2w[1 + i].between(c2w[1 + j])
                O = to_gtsam(np.asarray(odom[(a, b)]["mean"]))
                r = residual(P, O)
                d_pair.append(float(np.linalg.norm(P.translation()))); nt.append(np.linalg.norm(r[3:])); nr.append(np.linalg.norm(r[:3]))
    d_pair, nt, nr = np.asarray(d_pair), np.asarray(nt), np.asarray(nr)
    out["n_pairs"] = int(len(d_pair))
    # ---- reference pairs of one pass related by a short odometry chain (<= 3 edges, no turn inside): the raw pair
    # residual (including the short chain's noise) is a conservative estimate of the pair noise
    pairs = []
    for st in trace["steps"]:
        f = st.get("ffp")
        if not f or (frames is not None and st["i"] > frames):
            continue
        ids = f["ids"]; c2w = [to_gtsam(np.asarray(p)) for p in f["c2w"]]
        for i in range(len(ids)):
            for j in range(len(ids)):
                a, b = ids[i], ids[j]
                if a == b or not f["valid"][i] or not f["valid"][j] or a not in keep or b not in keep:
                    continue
                cur = a; chain = []
                for _ in range(3):
                    if cur == b:
                        break
                    if cur not in nxt:
                        chain = None; break
                    chain.append((cur, nxt[cur])); cur = nxt[cur]
                if chain is None or cur != b or not chain:
                    continue
                O = to_gtsam(np.eye(4))
                for key in chain:
                    O = O.compose(to_gtsam(np.asarray(odom[key]["mean"])))
                if np.linalg.norm(logmap(O)[:3]) > prm["pair_max_theta"]:
                    continue
                P = c2w[1 + i].between(c2w[1 + j])
                r = residual(P, O)
                pairs.append((float(np.linalg.norm(P.translation())), np.linalg.norm(r[3:]), np.linalg.norm(r[:3])))
    pairs = np.asarray(pairs) if pairs else np.zeros((0, 3))
    out["n_pairs_chain"] = int(len(pairs))
    fpt = fit_linear(pairs[:, 0], pairs[:, 1], bins=bins) if len(pairs) else None
    fpr = fit_linear(pairs[:, 0], pairs[:, 2], bins=bins) if len(pairs) else None

    def chain_pred(chain, k_t, k_r):
        Ts = [to_gtsam(np.asarray(odom[key]["mean"])) for key in chain]
        ns = [max((step.get(key[1]) or 0) - (step.get(key[0]) or 0), 1) for key in chain]
        cov = np.zeros((6, 6)); tail = to_gtsam(np.eye(4))
        for k in range(len(chain) - 1, -1, -1):
            L = float(np.linalg.norm(Ts[k].translation())); th = float(np.linalg.norm(logmap(Ts[k])[:3]))
            sg = np.array([k_r * th / math.sqrt(ns[k]) + 1e-3] * 3 + [k_t * L / math.sqrt(ns[k]) + 2e-3] * 3)
            cov = cov + transport(np.diag(sg ** 2), tail)
            tail = Ts[k].compose(tail)
        return tail, cov

    # ---- visual edges vs the odometry chain over short spans (<= 3 edges, <= 1.5 m, no turn): the innovation is an
    # upper bound of the visual measurement noise (it includes the short chain's odometry noise) -> visual model
    vis = [e for e in graph["visual"] if e["a"] in keep and e["b"] in keep and e.get("session", 0) == graph["meta"].get("session", 0)]
    spans = []
    for e in vis:
        a, b = (e["a"], e["b"]) if e["a"] < e["b"] else (e["b"], e["a"])
        chain, cur = [], a
        for _ in range(40):
            if cur == b:
                break
            if cur not in nxt:
                chain = None; break
            chain.append((cur, nxt[cur])); cur = nxt[cur]
        if chain is None or cur != b or not chain:
            continue
        T_meas = to_gtsam(np.asarray(e["mean"]))
        if e["a"] > e["b"]:
            T_meas = T_meas.inverse()
        O = to_gtsam(np.eye(4))
        for key in chain:
            O = O.compose(to_gtsam(np.asarray(odom[key]["mean"])))
        L, th = float(np.linalg.norm(O.translation())), float(np.linalg.norm(logmap(O)[:3]))
        r = residual(T_meas, O)
        spans.append({"chain": chain, "T": T_meas, "L": L, "theta": th, "d": float(np.linalg.norm(T_meas.translation())),
                      "et": float(np.linalg.norm(r[3:])), "er": float(np.linalg.norm(r[:3]))})
    out["n_spans"] = len(spans)
    short = [sp for sp in spans if len(sp["chain"]) <= prm["short_max_edges"] and sp["theta"] < prm["short_max_theta"] and sp["L"] <= prm["short_max_len"]]
    # the immediately preceding keyframe is registered far better than any other reference (near-identical view):
    # the rotation floor is taken from references 2-3 keyframes back
    straight1 = [sp for sp in spans if 2 <= len(sp["chain"]) <= 3 and sp["theta"] < prm["straight_max_theta"]]
    if len(short) < prm["min_short"]:
        raise SystemExit(f"not enough short spans for the calibration ({len(short)} < {prm['min_short']}; record a longer sequence "
                         "or relax --short-max-edges / --short-max-theta / --short-max-len / --min-short)")
    # subtract the chain's own translation variance (default odometry constants) from the tail-calibrated innovation
    # per distance bin: sigma_vis^2 = sigma_innov^2 - var_chain (floored at half the innovation)
    def fit_linear_minus_chain(sel):
        xs, ys, ws = [], [], []
        d = np.array([sp["d"] for sp in sel]); et = np.array([sp["et"] for sp in sel])
        cv = np.array([float(np.trace(chain_pred(sp["chain"], k_def[0], k_def[1])[1][3:, 3:])) / 3.0 for sp in sel])
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (d >= lo) & (d < hi)
            if m.sum() >= 15:
                s2 = tail_sigma(et[m]) ** 2 - float(np.median(cv[m]))
                xs.append(float(np.median(d[m]))); ys.append(math.sqrt(max(s2, 0.25 * tail_sigma(et[m]) ** 2))); ws.append(math.sqrt(m.sum()))
        xs, ys, ws = np.asarray(xs), np.asarray(ys), np.asarray(ws)
        if len(xs) == 0:
            return None
        if len(xs) == 1:
            return {"a": float(ys[0]), "b": 0.0, "bins": [(float(xs[0]), float(ys[0]))]}
        A = np.stack([np.ones_like(xs), xs], 1) * ws[:, None]
        coef, _ = nnls(A, ys * ws)
        return {"a": float(coef[0]), "b": float(coef[1]), "bins": [(float(x), float(y)) for x, y in zip(xs, ys)]}
    ft = fit_linear_minus_chain(short)
    # the linear fit through the distance bins may extrapolate to a zero intercept, which no estimator has: the
    # intercept is at least the (unsubtracted, conservative) innovation of the shortest straight spans
    t_floor = tail_sigma(np.array([sp["et"] for sp in (straight1 if len(straight1) >= 20 else short)]))
    ft["a_fit"] = ft["a"]; ft["a"] = max(ft["a"], float(t_floor))
    r_floor = tail_sigma(np.array([sp["er"] for sp in (straight1 if len(straight1) >= 20 else short)]))
    fr = {"a": r_floor, "b": 0.0, "bins": []}
    out["pair_fit"] = {"t": fpt or ft, "r": fpr or fr}
    out["visual"] = {"t_a": ft["a"], "t_b": ft["b"], "r_a": fr["a"], "r_b": fr["b"]}
    # metric scale of the estimator: median ratio of the measured translation to the odometry-chain translation over
    # short straight spans of at least 1 m (the odometry is metric; the feed-forward scale comes from the stereo
    # anchors and is biased where the anchors are far, e.g. 0.82 on KITTI-06)
    sc = [sp["d"] / sp["L"] for sp in spans if len(sp["chain"]) <= prm["scale_max_edges"] and sp["L"] >= prm["scale_min_len"]]
    out["visual"]["scale"] = float(np.median(sc)) if len(sc) >= 20 else 1.0
    out["visual"]["scale_n"] = len(sc)

    # ---- odometry: rotation noise from spans that contain a turn (>= 0.3 rad): the odometry's turn noise dominates
    # the rotation innovation; translation noise is only upper-bounded on short spans (the visual noise dominates)
    def norm_innov(sel, k_t, k_r):
        zt, zr = [], []
        for sp in sel:
            T_pred, cov = chain_pred(sp["chain"], k_t, k_r)
            d = sp["d"]
            sv = np.array([fr["a"] + fr["b"] * d] * 3 + [ft["a"] + ft["b"] * d] * 3)
            S = cov + np.diag(sv ** 2)
            r = residual(sp["T"], T_pred)
            zr.append(r[:3] @ np.linalg.solve(S[:3, :3], r[:3])); zt.append(r[3:] @ np.linalg.solve(S[3:, 3:], r[3:]))
        return (np.median(zt) / chi2.ppf(0.5, 3) if zt else np.nan), (np.median(zr) / chi2.ppf(0.5, 3) if zr else np.nan)

    grid = np.logspace(-3, 0, 61)
    def identify(sel, part, k_other):
        if len(sel) < 10:
            return None, "no spans"
        z = np.array([norm_innov(sel, (g if part == "t" else k_other), (k_other if part == "t" else g))[0 if part == "t" else 1] for g in grid])
        if not np.isfinite(z).any():
            return None, "no spans"
        if z[0] >= 1.0:
            return float(grid[int(np.argmin(np.abs(np.log(np.maximum(z, 1e-9)))))]), "identified"
        below = np.where(z <= 0.5)[0]
        return (float(grid[int(below[0])]) if len(below) else float(grid[-1])), "upper bound (visual noise dominates the spans)"

    K_DEFAULT = k_def
    turn_spans = [sp for sp in spans if sp["theta"] >= prm["turn_min_theta"]]
    long_spans = [sp for sp in spans if sp["L"] >= prm["long_min_len"]]
    out["n_turn_spans"], out["n_long_spans"] = len(turn_spans), len(long_spans)
    kr, st_r = identify(turn_spans, "r", K_DEFAULT[0])
    k_r = kr if (kr is not None and st_r == "identified") else K_DEFAULT[1]
    if kr is None:
        st_r = f"default (no turn spans)"
    kt, st_t = identify(long_spans, "t", k_r)
    # an upper bound below the default means the odometry is at least as good as the visual measurements on these
    # spans: keep the default (a looser gate is the safe side); an identified value is used as it is
    k_t = kt if (kt is not None and st_t == "identified") else max(kt or 0.0, K_DEFAULT[0])
    if kt is None:
        st_t = "default (no long spans)"
    zt, zr = norm_innov(spans, k_t, k_r)
    out["odom"] = {"k_t": k_t, "k_r": k_r, "floor_t": 0.002, "floor_r": 0.001, "status_t": st_t, "status_r": st_r,
                   "check_norm_innov_all_spans": [float(zt), float(zr)]}
    # ---- 3. map consistency: a stored map is as consistent as its verified edges (visual model with a 2x margin)
    # default map model = the visual model; a saved map carries its own consistency model (System.save_map)
    out["map"] = {"t_a": ft["a"], "t_b": ft["b"], "r_a": fr["a"], "r_b": fr["b"]}
    out["pair_fit"]["r"]["bins"] = out["pair_fit"]["r"].get("bins", [])
    out["params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in prm.items()}
    return out


def to_config(cal: dict) -> dict:
    v, o, p, m = cal["visual"], cal["odom"], cal["pair_fit"], cal["map"]
    return {
        "visual_t_a": round(v["t_a"], 4), "visual_t_b": round(v["t_b"], 4), "visual_r_a": round(v["r_a"], 5), "visual_r_b": round(v["r_b"], 5),
        "visual_scale": round(v.get("scale", 1.0), 4),
        "odom_k_t": round(o["k_t"], 4), "odom_k_r": round(o["k_r"], 4), "odom_floor_t": o["floor_t"], "odom_floor_r": o["floor_r"],
        "pair_t_a": round(p["t"]["a"], 4), "pair_t_b": round(p["t"]["b"], 4), "pair_r_a": round(p["r"]["a"], 5), "pair_r_b": round(p["r"]["b"], 5),
        "map_t_a": round(m["t_a"], 4), "map_t_b": round(m["t_b"], 4), "map_r_a": round(m["r_a"], 5), "map_r_b": round(m["r_b"], 5),
        "gate_inflation": 2.0,
    }


def gt_fit(graph, frames=None):
    """Ground-truth fit of the same quantities (validation only)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from offline_graph_study import GraphStudy
    except Exception:
        return None
    st = GraphStudy.__new__(GraphStudy)
    import tempfile
    g = json.loads(json.dumps(graph))
    if frames is not None:
        keep = {n["id"] for n in g["nodes"] if n.get("step_created") is None or n["step_created"] <= frames}
        g["nodes"] = [n for n in g["nodes"] if n["id"] in keep]
        g["odom"] = [e for e in g["odom"] if e["a"] in keep and e["b"] in keep]
        g["visual"] = [e for e in g["visual"] if e["a"] in keep and e["b"] in keep]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(g, f); path = f.name
    st = GraphStudy(path, snr=g["meta"].get("snr"))
    c = st.calibrate()
    # metric-scale ratio against ground truth over the same kind of spans as the calibration (<= 5 keyframes apart, >= 1 m)
    gtp = {n["id"]: np.asarray(n["gt"], dtype=np.float64).reshape(4, 4) for n in g["nodes"] if n.get("gt")}
    ratios = []
    for e in g["visual"]:
        a, b = e["a"], e["b"]
        if abs(a - b) > 5 or a not in gtp or b not in gtp:
            continue
        tg = (np.linalg.inv(gtp[a]) @ gtp[b])[:3, 3]
        if np.linalg.norm(tg) >= 1.0:
            ratios.append(float(np.linalg.norm(np.asarray(e["mean"], dtype=np.float64)[:3])) / float(np.linalg.norm(tg)))
    out = {"visual": c["visual_fit"], "odom": c.get("odom", {}).get("fit")}
    out["visual"]["scale"] = float(np.median(ratios)) if len(ratios) >= 20 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--trace", default=None, help="trace with pass-internal poses (multi-view estimators only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=None, help="use only the first N frames (e.g. 600 = one minute at 10 Hz)")
    ap.add_argument("--validate", action="store_true", help="compare with the ground-truth fit (graph must contain gt)")
    ap.add_argument("--snr", type=float, default=None, help="odometry SNR of the recording (--snr of record_trace.py); with --k-default auto "
                    "the chain-variance subtraction uses k = 1/(snr*sqrt(3)) instead of 0.06 / 0.07")
    ap.add_argument("--k-default", default="auto", help="odometry constants (k_t, k_r) assumed in the chain-variance subtraction: 'auto' "
                    "(1/(snr*sqrt(3)) when --snr is given, else 0.06 / 0.07), a number or 'k_t,k_r'")
    g = ap.add_argument_group("span selection (metric thresholds; the defaults are the historical constants)")
    g.add_argument("--short-max-edges", type=int, default=CAL_DEFAULTS["short_max_edges"], help="short spans: at most this many odometry edges")
    g.add_argument("--short-max-theta", type=float, default=CAL_DEFAULTS["short_max_theta"], help="short spans: chain rotation below this (rad)")
    g.add_argument("--short-max-len", type=float, default=CAL_DEFAULTS["short_max_len"], help="short spans: chain length at most this (m)")
    g.add_argument("--straight-max-theta", type=float, default=CAL_DEFAULTS["straight_max_theta"], help="floor spans (2-3 edges): rotation below this (rad)")
    g.add_argument("--pair-max-theta", type=float, default=CAL_DEFAULTS["pair_max_theta"], help="chained reference pairs: chain rotation below this (rad)")
    g.add_argument("--scale-max-edges", type=int, default=CAL_DEFAULTS["scale_max_edges"], help="metric scale: spans of at most this many edges")
    g.add_argument("--scale-min-len", type=float, default=CAL_DEFAULTS["scale_min_len"], help="metric scale: chain length at least this (m)")
    g.add_argument("--long-min-len", type=float, default=CAL_DEFAULTS["long_min_len"], help="odometry k_t: spans of at least this chain length (m)")
    g.add_argument("--turn-min-theta", type=float, default=CAL_DEFAULTS["turn_min_theta"], help="odometry k_r: spans with at least this rotation (rad)")
    g.add_argument("--bins", default=",".join(str(b) for b in CAL_DEFAULTS["bins"]), help="distance-bin edges (m) of the pair and visual fits, comma-separated")
    g.add_argument("--min-short", type=int, default=CAL_DEFAULTS["min_short"], help="hard stop: at least this many short spans are required")
    args = ap.parse_args()
    bins = tuple(float(b) for b in str(args.bins).replace(";", ",").split(",") if b.strip())
    if len(bins) < 2 or any(b2 <= b1 for b1, b2 in zip(bins[:-1], bins[1:])):
        raise SystemExit(f"--bins must be at least two increasing edges, got {args.bins!r}")
    params = {"short_max_edges": args.short_max_edges, "short_max_theta": args.short_max_theta, "short_max_len": args.short_max_len,
              "straight_max_theta": args.straight_max_theta, "pair_max_theta": args.pair_max_theta,
              "scale_max_edges": args.scale_max_edges, "scale_min_len": args.scale_min_len, "long_min_len": args.long_min_len,
              "turn_min_theta": args.turn_min_theta, "bins": bins, "k_default": parse_k_default(args.k_default, args.snr),
              "min_short": args.min_short}
    graph = load_json(args.graph)
    trace = load_json(args.trace) if args.trace else {"steps": []}
    cal = calibrate(graph, trace, frames=args.frames, params=params)
    cal["params"].update({"snr": args.snr, "k_default_spec": args.k_default, "frames": args.frames})
    cfg = to_config(cal)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"== calibration from {cal['n_pairs']} consecutive pairs, {cal['n_pairs_chain']} chained pairs, {cal['n_spans']} odometry spans"
          + (f" (first {args.frames} frames)" if args.frames else ""))
    print(f"   visual: sigma_t = {cfg['visual_t_a']:.4f} + {cfg['visual_t_b']:.4f} d m, sigma_r = {math.degrees(cfg['visual_r_a']):.3f} + {math.degrees(cfg['visual_r_b']):.4f} d deg   bins {[(round(x, 1), round(y, 3)) for x, y in cal['pair_fit']['t']['bins']]}")
    print(f"   metric scale (measured / odometry): {cfg['visual_scale']:.3f} ({cal['visual'].get('scale_n', 0)} spans)")
    print(f"   odom:   k_t = {cfg['odom_k_t']:.4f} ({cal['odom']['status_t']}, {cal['n_long_spans']} spans), k_r = {cfg['odom_k_r']:.4f} ({cal['odom']['status_r']}, {cal['n_turn_spans']} spans); innovation check all spans {cal['odom']['check_norm_innov_all_spans']}")
    print(f"   -> {args.out}")
    if args.validate:
        gt = gt_fit(graph, frames=args.frames)
        if gt:
            v, o = gt["visual"], gt["odom"]
            print(f"   GT fit: visual sigma_t = {v['t_a']:.4f} + {v['t_b']:.4f} d, sigma_r = {math.degrees(v['r_a']):.3f} + {math.degrees(v['r_b']):.4f} d deg; odom k_t = {o['k_t']:.4f}, k_r = {o['k_r']:.4f}")
            cal["gt_fit"] = gt
    Path(args.out).with_suffix(".json").write_text(json.dumps(cal, indent=1, default=float))


if __name__ == "__main__":
    main()
