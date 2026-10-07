"""Pose-graph optimisation at scale (cross/core/pgo.py): the per-factor caches and lazy vertices must not change any
result, and the windowed optimisation must leave the keyframes before its window untouched (CPU only)."""
import math
import types

import gtsam
import numpy as np
import pypose as pp
import torch

from cross.core import pgo as pgo_mod
from cross.core.config import HypothesisConfig, NoiseModelConfig, SystemConfig
from cross.core.hypothesis import HypothesisManager
from cross.core.lc_verify import NoiseModel
from cross.core.types import Edge, EdgeType, Keyframe, VisualEdge


def _lie(T: gtsam.Pose3) -> pp.LieTensor:
    q = T.rotation().toQuaternion().coeffs()      # x y z w
    t = T.translation()
    return pp.SE3(torch.tensor([t[0], t[1], t[2], q[0], q[1], q[2], q[3]], dtype=torch.float32))


def make_manager(n=240, loops=((10, 200), (40, 230), (120, 160)), window_min_nodes=0, seed=0):
    """Hypothesis-0 graph of a noisy square trajectory: odometry chain, visual edges to the previous keyframes
    (non-informative, skipped) and loop edges between revisits, all with the verified loop closure's noise model."""
    rng = np.random.default_rng(seed)
    cfg = SystemConfig()
    cfg.mapping.loop_closure.pgo_window_min_nodes = window_min_nodes
    verifier = types.SimpleNamespace(noise=NoiseModel(NoiseModelConfig()))
    sys_ = types.SimpleNamespace(_lc_verifier=verifier, topo_map=None, _session_start_kf_id=0, last_added_kf_id=None,
                                 config=cfg)
    hm = HypothesisManager(sys_, n_components=3, config=HypothesisConfig())
    sys_.hypothesis_manager = hm
    gt = [gtsam.Pose3()]
    for i in range(1, n):
        turn = math.pi / 2 if i % 60 == 0 else 0.0
        gt.append(gt[-1].compose(gtsam.Pose3(gtsam.Rot3.Ypr(turn, 0, 0), gtsam.Point3(0.25, 0, 0))))
    est = [gt[0]]
    for i in range(n):
        if i > 0:
            d = gt[i - 1].between(gt[i]).compose(gtsam.Pose3.Expmap(rng.normal(0, [2e-3] * 3 + [5e-3] * 3)))
            est.append(est[-1].compose(d))
            e = Edge(_lie(d), pp.se3(torch.full((6,), 0.05)), EdgeType.ODOMETRY)
            e.n_frames = 3
            hm.odom_edges[(i - 1, i)] = e
        mu = torch.stack([_lie(est[i]).tensor()] * 3)
        kf = Keyframe(pp.SE3(mu), pp.se3(torch.full((3, 6), 0.05)), torch.tensor([1.0, 0.0, 0.0]), None, None)
        kf.id = i
        hm.nodes[i] = kf

    def visual(a, b, informative):
        m = gt[a].between(gt[b]).compose(gtsam.Pose3.Expmap(rng.normal(0, [1e-3] * 3 + [2e-2] * 3)))
        f = VisualEdge(_lie(m), pp.se3(torch.full((6,), 0.05)), EdgeType.VISUAL, 0, 0)
        f.noise_scale, f.noise_scale_along, f.noise_scale_rot = 1.0, (1.5 if informative else None), None
        f.informative = informative
        hm.hypotheses[0].visual_edges.setdefault((a, b), []).append(f)
        hm.hypotheses[0].visual_adjacency.setdefault(a, set()).add(b)
        hm.hypotheses[0].visual_adjacency.setdefault(b, set()).add(a)

    for i in range(2, n):
        visual(i - 2, i, False)
    for a, b in loops:
        visual(a, b, True)
    return hm


def _poses(res):
    return {k: v.tensor().detach().numpy().copy() for k, v in res["optimized_poses"].items()}


def test_cached_factors_give_the_uncached_solution():
    hm = make_manager()
    pgo_mod._measurement_cache.clear(); pgo_mod._noise_cache.clear()
    cold = _poses(hm.handle_loop_closure(0, apply=False))
    warm = _poses(hm.handle_loop_closure(0, apply=False))         # every factor from the caches
    fn = hm.pgo_noise_fn

    def uncached():                                                # a noise function without cache_key: no caching
        f = fn()
        return lambda factor: f(factor)
    hm.pgo_noise_fn = uncached
    plain = _poses(hm.handle_loop_closure(0, apply=False))
    assert cold.keys() == warm.keys() == plain.keys()
    for k in cold:
        assert np.array_equal(cold[k], warm[k]) and np.array_equal(cold[k], plain[k])


def test_noise_cache_follows_the_calibrated_model():
    hm = make_manager()
    hm.handle_loop_closure(0, apply=False)                         # fills the caches
    hm.system._lc_verifier.noise.cfg.visual_t_a = 0.2              # recalibration changes every visual factor
    after = _poses(hm.handle_loop_closure(0, apply=False))
    pgo_mod._measurement_cache.clear(); pgo_mod._noise_cache.clear()
    fresh = _poses(hm.handle_loop_closure(0, apply=False))
    for k in fresh:
        assert np.array_equal(after[k], fresh[k])


def test_lazy_vertex_is_a_view_of_the_keyframe_row():
    hm = make_manager(n=30, loops=())
    pg = pgo_mod.PoseGraph(hm, depth=100, device="cpu")
    v = pg._vertex(5, 0)
    hm.nodes[5].pose_mu[0] = pp.SE3(torch.tensor([1.0, 2.0, 3.0, 0, 0, 0, 1.0]))   # in place, as a write-back
    assert torch.equal(v.pose_row(), hm.nodes[5].pose_mu.tensor()[0])
    assert torch.equal(v.pose.tensor(), hm.nodes[5].pose_mu.tensor()[0]) and isinstance(v.pose, pp.LieTensor)


def test_window_keeps_the_keyframes_before_it():
    hm = make_manager(window_min_nodes=1)
    before = {k: kf.pose_mu.tensor()[0].clone() for k, kf in hm.nodes.items()}
    res = hm.handle_loop_closure(0, window_ref=160)               # loops (10, 200), (40, 230) reach before it
    assert res["success"] and res["window"] == 160 - SystemConfig().mapping.loop_closure.pgo_window_margin
    w = res["window"]
    assert min(res["optim_nodes_ids"]) == w
    for k, kf in hm.nodes.items():
        moved = not torch.equal(kf.pose_mu.tensor()[0], before[k])
        assert moved == (k >= w and k in res["optimized_poses"])
    assert all(k < w for k in res["fixed_nodes_ids"])
    assert {w - 1, 10, 40} <= set(res["fixed_nodes_ids"])           # chain into the window and the old loop ends


def test_window_off_or_reaching_the_start_is_the_full_optimisation():
    full = _poses(make_manager().handle_loop_closure(0, apply=False, window_ref=10))      # option off
    start = _poses(make_manager(window_min_nodes=1).handle_loop_closure(0, apply=False, window_ref=10))
    assert full.keys() == start.keys()
    for k in full:
        assert np.array_equal(full[k], start[k])
