"""Gravity prior in the pose-graph optimisation (mapping.loop_closure.gravity_prior_deg): loop edges that carry a pitch
error tilt the chain without it; with it the keyframes stay level (their up direction stays the map's) while the loop
still closes in heading and position (CPU only)."""
import math
import types

import gtsam
import numpy as np
import pypose as pp
import torch

from cross.core.config import HypothesisConfig, NoiseModelConfig, SystemConfig
from cross.core.hypothesis import HypothesisManager
from cross.core.lc_verify import NoiseModel
from cross.core.types import Edge, EdgeType, Keyframe, VisualEdge


def _lie(T: gtsam.Pose3) -> pp.LieTensor:
    q = T.rotation().toQuaternion().coeffs()
    t = T.translation()
    return pp.SE3(torch.tensor([t[0], t[1], t[2], q[0], q[1], q[2], q[3]], dtype=torch.float32))


def make(gravity_deg, n=200, loop_pitch_deg=6.0, seed=0):
    rng = np.random.default_rng(seed)
    cfg = SystemConfig()
    cfg.mapping.loop_closure.gravity_prior_deg = gravity_deg
    verifier = types.SimpleNamespace(noise=NoiseModel(NoiseModelConfig()))
    up = np.array([0.0, 0.0, 1.0])                   # the map's up direction (z, yaw-only ground-truth turns)
    sys_ = types.SimpleNamespace(_lc_verifier=verifier, topo_map=None, _session_start_kf_id=0, last_added_kf_id=None,
                                 config=cfg, state_device="cpu", _gravity_up_map=up)
    hm = HypothesisManager(sys_, n_components=3, config=HypothesisConfig())
    sys_.hypothesis_manager = hm
    gt = [gtsam.Pose3()]
    for i in range(1, n):
        turn = math.pi / 2 if i % 50 == 0 else 0.0
        gt.append(gt[-1].compose(gtsam.Pose3(gtsam.Rot3.Ypr(turn, 0, 0), gtsam.Point3(0.25, 0, 0))))
    est = [gt[0]]
    for i in range(n):
        if i > 0:
            d = gt[i - 1].between(gt[i]).compose(gtsam.Pose3.Expmap(rng.normal(0, [1e-4] * 3 + [5e-3] * 3)))
            est.append(est[-1].compose(d))
            e = Edge(_lie(d), pp.se3(torch.full((6,), 0.05)), EdgeType.ODOMETRY)
            e.n_frames = 3
            hm.odom_edges[(i - 1, i)] = e
        mu = torch.stack([_lie(est[i]).tensor()] * 3)
        kf = Keyframe(pp.SE3(mu), pp.se3(torch.full((3, 6), 0.05)), torch.tensor([1.0, 0.0, 0.0]), None, None)
        kf.id = i
        kf.gravity = est[i].rotation().matrix().T @ up   # the odometry's up direction in the camera frame
        hm.nodes[i] = kf

    def visual(a, b, pitch):
        m = gt[a].between(gt[b]).compose(gtsam.Pose3(gtsam.Rot3.Ypr(0, math.radians(pitch), 0), gtsam.Point3(0, 0, 0)))
        f = VisualEdge(_lie(m), pp.se3(torch.full((6,), 0.01)), EdgeType.VISUAL, 0, 0)
        f.noise_scale, f.noise_scale_along, f.noise_scale_rot = 1.0, 1.5, None
        f.informative = True
        hm.hypotheses[0].visual_edges.setdefault((a, b), []).append(f)
        hm.hypotheses[0].visual_adjacency.setdefault(a, set()).add(b)
        hm.hypotheses[0].visual_adjacency.setdefault(b, set()).add(a)

    for a, b in ((10, 190), (20, 180), (60, 140)):
        visual(a, b, loop_pitch_deg)
    return hm, up


def max_tilt_deg(hm, up, res):
    worst = 0.0
    for k, p in res["optimized_poses"].items():
        R = p.matrix().detach().cpu().numpy().astype(np.float64)[:3, :3]
        u = R @ hm.nodes[k].gravity
        worst = max(worst, math.degrees(math.acos(np.clip(u @ up / np.linalg.norm(u), -1, 1))))
    return worst


def test_gravity_prior_keeps_keyframes_level():
    hm0, up = make(0.0)
    tilted = max_tilt_deg(hm0, up, hm0.handle_loop_closure(0, apply=False))
    hm1, up = make(0.5)
    res = hm1.handle_loop_closure(0, apply=False)
    level = max_tilt_deg(hm1, up, res)
    assert tilted > 1.0, tilted                      # the pitch error of the loop edges tilts the chain
    assert level < 0.5 * tilted and level < 1.0, (level, tilted)


def test_gravity_prior_off_without_map_up():
    hm, up = make(0.5)
    hm.system._gravity_up_map = None
    hm0, _ = make(0.0)
    a = hm.handle_loop_closure(0, apply=False)["optimized_poses"]
    b = hm0.handle_loop_closure(0, apply=False)["optimized_poses"]
    for k in a:
        assert torch.allclose(a[k].tensor(), b[k].tensor())
