#!/usr/bin/env python3
"""Replay the online hypothesis-0 pose-graph optimisation of a recorded run offline, on the saved map (map.pkl:
the real factor objects with their metadata) and the graph dump (ground truth), through the system's own PoseGraph
code path.  Reports the map ATE of the online result, of the exact replay, and of controlled variants, and compares
the stored `informative` flags with the information criterion evaluated offline.

usage: python scripts/lc/replay_pgo.py outputs/lcstudy/<scene>/<run> [--noise configs/noise/x.yaml]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch
import pypose as pp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross.core.hypothesis import HypothesisManager  # noqa: E402
from cross.core.config import HypothesisConfig, LoopClosureConfig, PGOConfig  # noqa: E402
from cross.core.lc_verify import LoopClosureVerifier, to_gtsam, informative  # noqa: E402
from cross.core.pgo import PoseGraph  # noqa: E402
from cross.core.types import EdgeType, Keyframe  # noqa: E402
from lc.offline_graph_study import umeyama_ate  # noqa: E402
import gtsam  # noqa: E402


def pose3(p7):
    p = np.asarray(p7, dtype=np.float64)
    return gtsam.Pose3(gtsam.Rot3(p[6], p[3], p[4], p[5]), p[:3])


def load_run(run: Path, noise_file: str | None, session: int = 0, device: str = "cpu"):
    from cross.db.store import read_map
    d = read_map(run / "map.pkl")          # format v2 or the old single pickle
    g = json.load(open(run / f"graph_s{session}.json"))
    gt = {n["id"]: gtsam.Pose3(np.asarray(n["gt"], dtype=np.float64).reshape(4, 4)) for n in g["nodes"] if n.get("gt")}
    online = {n["id"]: pose3(n["pose"]) for n in g["nodes"]}
    kfs = {}
    for n in g["nodes"]:
        dt = torch.float64 if device == "cpu" else torch.float32
        kf = Keyframe(pp.SE3(torch.tensor(n["pose"], dtype=dt).view(1, 7)).to(device),
                      pp.se3(torch.tensor(n["std"], dtype=dt).view(1, 6)).to(device), torch.ones(1), None, None)
        kf.id = int(n["id"]); kf.temporary = bool(n.get("temporary", False)); kf.step_created = n.get("step_created")
        kfs[kf.id] = kf
    sys_ = types.SimpleNamespace(topo_map=None, _session_start_kf_id=int(g["meta"].get("session_start_kf_id", 0)),
                                 config=types.SimpleNamespace(pgo=PGOConfig()))
    hm = HypothesisManager(sys_, n_components=1, config=HypothesisConfig())
    sys_.hypothesis_manager = hm
    hm.load_state(d["hypo_data"], types.SimpleNamespace(get_atlas=lambda i: None), device, device, kfs)
    hm.device = device
    lc = LoopClosureConfig()
    if noise_file:
        for k, v in (yaml.safe_load(open(noise_file)) or {}).items():
            setattr(lc.noise, k, v)
    mc = d.get("map_consistency")
    v = LoopClosureVerifier(sys_, lc)
    sys_._lc_verifier = v
    return hm, sys_, v, gt, online, mc


def chain_init(hm):
    """Node poses by integrating the odometry edges from the first node (odometry initialisation)."""
    nxt = {a: (b, e) for (a, b), e in hm.odom_edges.items()}
    cur = min(hm.nodes); T = gtsam.Pose3(); poses = {cur: T}
    while cur in nxt:
        b, e = nxt[cur]; T = T.compose(to_gtsam(e)); poses[b] = T; cur = b
    return poses


def set_poses(hm, poses):
    for i, T in poses.items():
        if i in hm.nodes:
            q = T.rotation().toQuaternion(); t = T.translation()
            hm.nodes[i].pose_mu = pp.SE3(torch.tensor([t[0], t[1], t[2], q.x(), q.y(), q.z(), q.w()], dtype=torch.float64).view(1, 7))


def informative_offline(hm, v):
    """Information criterion evaluated on the stored graph (chain a->b vs measurement)."""
    n_tot = n_inf = n_stored = n_agree = 0
    for (a, b), fs in hm.hypotheses[0].visual_edges.items():
        for f in fs:
            n_tot += 1
            T, cov = v.chain.predict(a, b, inflate=v.cfg.noise.gate_inflation)
            dd = float(np.linalg.norm(f.mean_np[:3])); sv = v.noise.visual(dd, float(getattr(f, "noise_scale", 1.0) or 1.0))
            inf = True if T is None else informative(cov / v.cfg.noise.gate_inflation ** 2, sv, rotation=False)
            st = getattr(f, "informative", None)
            n_inf += inf; n_stored += (st is True); n_agree += (st == inf)
            f._informative_offline = inf
            f._informative_rot = True if T is None else informative(cov / v.cfg.noise.gate_inflation ** 2, sv, rotation=True)
    return {"edges": n_tot, "informative_offline": n_inf, "informative_stored": n_stored, "agree": n_agree}


def solve(hm, sys_, label, init=None, multiplicity=True, skip="stored", robust=True, fixed="first"):
    if init is not None:
        set_poses(hm, init)
    sys_.config.pgo.visual_robust_enabled = robust
    for fs in hm.hypotheses[0].visual_edges.values():
        for f in fs:
            if skip == "offline":
                f.informative = f._informative_offline
            elif skip == "rot":
                f.informative = f._informative_rot
            elif skip == "none":
                f.informative = True
            else:
                f.informative = f._informative_stored
    pg = PoseGraph(hm, depth=1000, k_hop=2, device="cpu", noise_fn=hm.pgo_noise_fn(), skip_fn=hm.pgo_skip_fn())
    pg.scale_by_multiplicity = multiplicity
    target = max(hm.nodes)
    pg.construct_for_loop_closure(target_node_id=target, other_hypothesis_id=0)
    ids = [vv.id for vv in pg.vertices if vv.id in hm.nodes]
    fixed_ids = {min(ids)} if fixed == "first" else {max(ids)}
    pg.solve(optim_node_ids=set(ids) - fixed_ids, fixed_node_ids=fixed_ids)
    est = {}
    for vv in pg.vertices:
        if vv.id in hm.nodes:
            p = pg.optimized_poses.get(vv.id)
            est[vv.id] = pose3(p.tensor().reshape(-1).numpy()) if p is not None else pose3(vv.pose.tensor().reshape(-1).numpy())
    return est, pg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--noise", default=None)
    ap.add_argument("--session", type=int, default=0)
    ap.add_argument("--scale", default=None, help="metric-scale correction of the visual measurements: 'auto' (median ratio of the "
                    "measured to the odometry-chain translation over short spans, no ground truth) or a number; translations are divided by it")
    args = ap.parse_args()
    run = Path(args.run)
    noise = args.noise
    if noise is None:
        a = json.load(open(run / "args.json")) if (run / "args.json").exists() else {}
        noise = a.get("noise_config")
    hm, sys_, v, gt, online, mc = load_run(run, noise, args.session)
    for fs in hm.hypotheses[0].visual_edges.values():
        for f in fs:
            f._informative_stored = getattr(f, "informative", None)
    print(f"run {run}  noise {noise}  nodes {len(hm.nodes)}  odom {len(hm.odom_edges)}  visual {sum(len(x) for x in hm.hypotheses[0].visual_edges.values())}")
    if args.scale:
        ratios = []
        for (a, b), fs in hm.hypotheses[0].visual_edges.items():
            T, _ = v.chain.predict(a, b)
            if T is None or abs(a - b) > 5:
                continue
            L = float(np.linalg.norm(T.translation()))
            if L < 1.0:
                continue
            for f in fs:
                ratios.append(float(np.linalg.norm(f.mean_np[:3])) / L)
        s_auto = float(np.median(ratios)) if ratios else 1.0
        scale = s_auto if args.scale == "auto" else float(args.scale)
        print(f"scale ratio measured/chain over {len(ratios)} short spans: median {s_auto:.3f} (p10 {np.percentile(ratios, 10):.3f}, p90 {np.percentile(ratios, 90):.3f}); correcting by 1/{scale:.3f}")
        for fs in hm.hypotheses[0].visual_edges.values():
            for f in fs:
                m = f.mean_np.copy(); m[:3] /= scale
                f.mean = pp.SE3(torch.tensor(m, dtype=torch.float64)); f._mean_np = None
    print("informative:", informative_offline(hm, v))
    print(f"online map ATE            {umeyama_ate(online, gt):.3f}")
    ate = lambda est: umeyama_ate(est, gt)
    chain = chain_init(hm)
    print(f"odometry only             {ate(chain):.3f}")
    est, pg = solve(hm, sys_, "replay", init=online)
    print(f"replay (online init)      {ate(est):.3f}  cost {pg.optimization_cost:.1f} (initial {pg.initial_cost:.1f})  factors {pg.n_factors}")
    # the factors that dominate the cost of the online state
    worst = []
    for (a, b), fs in hm.hypotheses[0].visual_edges.items():
        for f in fs:
            if a in online and b in online:
                r = to_gtsam(f).between(online[a].between(online[b]))
                s = v.noise.visual_from_factor(f)
                worst.append((float(np.linalg.norm(gtsam.Pose3.Logmap(r)[3:]) / s[3]), a, b, round(float(np.linalg.norm(f.mean_np[:3])), 3), round(float(s[3]), 4)))
    worst.sort(reverse=True)
    print("worst visual factors of the online state (|r_t|/sigma, a, b, d, sigma_t):", worst[:6])
    wo = []
    for (a, b), e in hm.odom_edges.items():
        if a in online and b in online:
            r = gtsam.Pose3.Logmap(to_gtsam(e).between(online[a].between(online[b]))); s = v.noise.odom_from_factor(e)
            wo.append((round(float(np.linalg.norm(r[3:]) / s[3]), 1), round(float(np.linalg.norm(r[:3]) / s[0]), 1), a, b, round(float(np.linalg.norm(e.mean_np[:3])), 3), getattr(e, "n_frames", None)))
    wo.sort(reverse=True)
    print("worst odometry factors of the online state (|r_t|/sigma, |r_r|/sigma, a, b, L, n_frames):", wo[:6])
    print("visual edges with d < 0.02 m:", sum(1 for w in worst if w[3] < 0.02), "of", len(worst))
    est, pg = solve(hm, sys_, "replay", init=chain)
    print(f"replay (odometry init)    {ate(est):.3f}  cost {pg.optimization_cost:.1f} (initial {pg.initial_cost:.1f})")
    est, pg = solve(hm, sys_, "replay", init=online, multiplicity=False)
    print(f"no multiplicity scaling   {ate(est):.3f}  cost {pg.optimization_cost:.1f}")
    est, pg = solve(hm, sys_, "replay", init=online, skip="offline")
    print(f"offline informative flags {ate(est):.3f}  cost {pg.optimization_cost:.1f}  factors {pg.n_factors}")
    est, pg = solve(hm, sys_, "replay", init=chain, skip="offline")
    print(f"  ... odometry init       {ate(est):.3f}")
    est, pg = solve(hm, sys_, "replay", init=online, skip="rot")
    print(f"translation-or-rotation   {ate(est):.3f}  cost {pg.optimization_cost:.1f}  factors {pg.n_factors}")
    est, pg = solve(hm, sys_, "replay", init=chain, skip="rot")
    print(f"  ... odometry init       {ate(est):.3f}")
    est, pg = solve(hm, sys_, "replay", init=online, skip="none")
    print(f"all edges                 {ate(est):.3f}  cost {pg.optimization_cost:.1f}")
    est, pg = solve(hm, sys_, "replay", init=online, robust=False)
    print(f"gaussian (no robust)      {ate(est):.3f}")
    a0 = v.noise.cfg.visual_t_a
    for fl in (0.01, 0.03):
        v.noise.cfg.visual_t_a = max(a0, fl)
        est, pg = solve(hm, sys_, "replay", init=online)
        print(f"visual intercept >= {fl}   {ate(est):.3f}  cost {pg.optimization_cost:.1f}")
    v.noise.cfg.visual_t_a = a0
    est, pg = solve(hm, sys_, "replay", init=online, fixed="last")
    print(f"fix last node             {ate(est):.3f}")


if __name__ == "__main__":
    main()
