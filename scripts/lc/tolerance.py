#!/usr/bin/env python3
"""Tolerance of the verified loop closure to its single parameter (chi-square confidence c), offline on dumped graphs.

For every confidence level the prior test (odometry chain / session anchor, deployed noise model from a calibration
YAML, gate inflation 2, translation marginal) and the posterior test (map-fixed calibrated Huber optimisation for
relocalization graphs) are scored against the ground-truth labels (false = translation error > --false-thr).

usage: python scripts/lc/tolerance.py --noise configs/noise/lonemonk_full.yaml graph_s0.json graph_s1.json ... [--out tol.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import gtsam
import numpy as np
import yaml
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_graph_study import GraphStudy, pose7, residual  # noqa: E402

LEVELS = [0.9, 0.99, 0.999, 0.9999, 0.999999]


def deployed_models(st: GraphStudy, noise: dict):
    """Install the deployed (calibrated YAML) noise model into a GraphStudy instead of the GT fit."""
    st.fit = {"visual": {"t_a": noise["visual_t_a"], "t_b": noise["visual_t_b"], "r_a": noise["visual_r_a"], "r_b": noise["visual_r_b"], "conf_ref": None},
              "odom": {"k_t": noise["odom_k_t"], "k_r": noise["odom_k_r"], "floor_t": noise["odom_floor_t"], "floor_r": noise["odom_floor_r"]},
              "map": {"t_a": noise["map_t_a"], "t_b": noise["map_t_b"], "r_a": noise["map_r_a"], "r_b": noise["map_r_b"]}}
    from offline_graph_study import NoiseModels
    st.models = NoiseModels(st.fit, st.snr)


def adaptive_scales(st: GraphStudy, window=150, scale_max=5.0):
    """Replay of the online noise scale: the normalised prior-test residual of every measurement (loop, cross and
    local edges, in step order) feeds a sliding window; each edge gets the scale in force at its step
    (median / 1.538, bounded to [1, scale_max]), like the online verifier."""
    import collections
    steps = {}
    for e in st.visual:
        steps.setdefault(e.get("step") or 0, []).append(e)
    dq = {"cross": collections.deque(maxlen=window), "sess": collections.deque(maxlen=window)}
    for stp in sorted(steps):
        sc = {k: (min(max(1.0, float(np.median(np.asarray(q))) / 1.5382), scale_max) if len(q) >= 20 else 1.0) for k, q in dq.items()}
        for e in steps[stp]:
            e["scale"] = sc["cross" if e["kind"] == "cross" else "sess"]
        for e in steps[stp]:
            if "r_gt" not in e:
                continue
            if e["kind"] == "cross":
                T_pred, cov = st.anchor_prediction(e, "fitted*2", "fitted"); T_meas = e["T"]
            else:
                a, b = (e["a"], e["b"]) if e["a"] < e["b"] else (e["b"], e["a"])
                T_pred, cov = st.chain(a, b, "fitted*2"); T_meas = e["T"] if e["a"] < e["b"] else e["T"].inverse()
            if T_pred is None:
                continue
            r = residual(T_meas, T_pred)
            S0 = cov + np.diag(st.visual_sigma(e, "fitted") ** 2)
            dq["cross" if e["kind"] == "cross" else "sess"].append(float(np.linalg.norm(r[3:])) / math.sqrt(max(float(np.trace(S0[3:, 3:])) / 3.0, 1e-12)))


def prior_chi2_translation(st: GraphStudy, e, inflate=2.0):
    if e["kind"] == "loop":
        a, b = (e["a"], e["b"]) if e["a"] < e["b"] else (e["b"], e["a"])
        T_pred, cov = st.chain(a, b, f"fitted*{inflate}")
        T_meas = e["T"] if e["a"] < e["b"] else e["T"].inverse()
    else:
        T_pred, cov = st.anchor_prediction(e, f"fitted*{inflate}", "fitted")
        T_meas = e["T"]
    if T_pred is None:
        return None
    s = st.visual_sigma(e, "fitted") * e.get("scale", 1.0)
    r = residual(T_meas, T_pred)
    S = cov + np.diag(s ** 2)
    return float(r[3:] @ np.linalg.solve(S[3:, 3:], r[3:]))


def posterior_chi2_reloc(st: GraphStudy):
    """Map-fixed calibrated Huber optimisation; translation chi2 of every cross edge (3 dof)."""
    graph = gtsam.NonlinearFactorGraph(); init = gtsam.Values()
    ids = sorted(st.nodes.keys())
    for i in ids:
        init.insert(i, pose7(st.nodes[i]["pose"]))
    for i in ids:
        if st.session.get(i, 0) == 0:
            graph.add(gtsam.PriorFactorPose3(i, init.atPose3(i), gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))))
    for e in st.odom:
        if st.session.get(e["a"], 0) == 0 and st.session.get(e["b"], 0) == 0:
            continue
        graph.add(gtsam.BetweenFactorPose3(e["a"], e["b"], e["T"], gtsam.noiseModel.Diagonal.Sigmas(st.odom_sigma(e, "fitted"))))
    cross = [e for e in st.visual if e["kind"] == "cross" and "r_gt" in e]
    for e in [x for x in st.visual if x["kind"] != "cross" and st.session.get(x["a"], 0) != 0 and st.session.get(x["b"], 0) != 0] + cross:
        s_ = st.visual_sigma(e, "fitted") * e.get("scale", 1.0)
        if e["kind"] == "cross":
            s_ = np.sqrt(s_ ** 2 + st.models.map_sigma(e["d"]) ** 2)
        base = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.0), gtsam.noiseModel.Diagonal.Sigmas(s_))
        graph.add(gtsam.BetweenFactorPose3(e["a"], e["b"], e["T"], base))
    lp = gtsam.LevenbergMarquardtParams(); lp.setMaxIterations(100)
    res = gtsam.LevenbergMarquardtOptimizer(graph, init, lp).optimize()
    out = []
    for e in cross:
        s_ = st.visual_sigma(e, "fitted") * e.get("scale", 1.0); s_ = np.sqrt(s_ ** 2 + st.models.map_sigma(e["d"]) ** 2)
        r = residual(e["T"], res.atPose3(e["a"]).between(res.atPose3(e["b"])))
        out.append((float(r[3:] @ (r[3:] / (s_[3:] ** 2))), e["gt_false"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="+")
    ap.add_argument("--noise", required=True)
    ap.add_argument("--false-thr", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--adaptive", action="store_true", help="replay the online innovation-based noise scale")
    args = ap.parse_args()
    noise = yaml.safe_load(open(args.noise))
    results = {}
    for g in args.graphs:
        st = GraphStudy(g, false_t=args.false_thr)
        st.calibrate()
        deployed_models(st, noise)
        if args.adaptive:
            adaptive_scales(st)
        cands = [e for e in st.visual if e["kind"] in ("loop", "cross") and "r_gt" in e]
        if args.adaptive and cands:
            sc = [e.get("scale", 1.0) for e in cands]
            print(f"   adaptive scale over candidates: median {np.median(sc):.2f}, p90 {np.percentile(sc, 90):.2f}")
        pri = [(prior_chi2_translation(st, e), e["gt_false"]) for e in cands]
        pri = [(c, f) for c, f in pri if c is not None]
        post = posterior_chi2_reloc(st) if st.meta.get("session", 0) != 0 else []
        rec = {"n_candidates": len(cands), "n_false": sum(e["gt_false"] for e in cands), "levels": {}}
        for c in LEVELS:
            thr = chi2.ppf(c, 3)
            def score(pairs):
                t = [x for x in pairs if not x[1]]; f_ = [x for x in pairs if x[1]]
                return {"TPR": (np.mean([x[0] <= thr for x in t]) if t else None), "FPR": (np.mean([x[0] <= thr for x in f_]) if f_ else None),
                        "n_true": len(t), "n_false": len(f_)}
            rec["levels"][str(c)] = {"prior": score(pri), "posterior": score(post) if post else None}
        results[g] = rec
        print(f"== {g}: {len(cands)} candidates, {rec['n_false']} false")
        for c, v in rec["levels"].items():
            p, q = v["prior"], v["posterior"]
            print(f"   c={c:9s} prior TPR {p['TPR'] if p['TPR'] is None else round(p['TPR'], 3)} FPR {p['FPR'] if p['FPR'] is None else round(p['FPR'], 3)}"
                  + (f" | posterior TPR {round(q['TPR'], 3) if q['TPR'] is not None else None} FPR {round(q['FPR'], 3) if q['FPR'] is not None else None}" if q else ""))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
