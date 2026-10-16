"""Navigation on CROSS's own feed-forward pass: extra-keyframe coverage heuristic and the planner on given maps."""
import math
import threading
import types

import numpy as np
import pytest
import torch

from cross.core.system import System


def _quat_yaw(theta):
    """Camera-to-world quaternion [x y z w] of a level OpenCV camera turned by theta about the up axis (-y): the
    optical axis then points to (sin(-theta)... ) as camera_delta's rotation."""
    # rotation about the camera's y axis by -theta (up = -y): R = Ry(-theta)
    h = -theta / 2
    return np.array([0., math.sin(h), 0., math.cos(h)])


class _KF:
    def __init__(self, i, pos, yaw):
        self.id = i
        self._row = torch.tensor(np.r_[pos, _quat_yaw(yaw)], dtype=torch.float32)

    def plain_row(self, name, index):
        return self._row

    def has_image(self, name):
        return True


def _stub(kfs, cur_pos=(0., 0., 0.), cur_yaw=0.):
    cfg = types.SimpleNamespace(pose_est=types.SimpleNamespace(ff=types.SimpleNamespace(
        nav_extra_radius_m=6., nav_extra_range_m=5., nav_extra_cell_m=.5)))
    hm = types.SimpleNamespace(nodes={k.id: k for k in kfs}, graph_lock=threading.Lock())
    cur = torch.tensor(np.r_[cur_pos, _quat_yaw(cur_yaw)], dtype=torch.float32)
    return types.SimpleNamespace(config=cfg, hypothesis_manager=hm, get_current_pose=lambda: cur,
                                 pose_est=types.SimpleNamespace(_gravity_cam=None),
                                 camera=types.SimpleNamespace(frame_width=512, fx=256.))   # 90 deg field of view


def test_coverage_prefers_unseen_directions():
    same = _KF(0, (0., 0., -0.3), 0.)              # just behind, looking the same way: adds little
    back = _KF(1, (0., 0., 0.), math.pi)          # same place, looking backward
    left = _KF(2, (0., 0., 0.), math.pi / 2)      # same place, looking to one side
    far = _KF(3, (0., 0., 20.), math.pi)          # beyond the candidate radius
    sys_ = _stub([same, back, left, far])
    picks = System._nav_coverage_keyframes(sys_, [], 4)
    ids = [k.id for k in picks]
    assert ids[0] == 1                              # the backward view covers the most new ground
    assert 3 not in ids                             # too far
    assert set(ids[:2]) == {1, 2}


def test_coverage_excludes_references_and_counts_them_covered():
    back = _KF(1, (0., 0., 0.), math.pi)
    back2 = _KF(4, (0., 0., 0.2), math.pi)         # nearly the same view as `back`
    sys_ = _stub([back, back2])
    picks = System._nav_coverage_keyframes(sys_, [back], 2)
    assert [k.id for k in picks] in ([], [4])     # `back` is a reference: never picked; back2 adds little or nothing
    assert System._nav_coverage_keyframes(sys_, [], 0) == []


def test_camera_delta_matches_odometry_delta():
    from cross_nav.policy.sim.cross_nav_service import camera_delta, odom_delta
    prev, cur = np.array([1., 2., .3]), np.array([1.5, 2.4, .5])
    fwd, left, th = odom_delta(prev, cur)
    c, s = math.cos(prev[2]), math.sin(prev[2])
    assert np.allclose([prev[0] + fwd * c - left * s, prev[1] + fwd * s + left * c], cur[:2])
    T = camera_delta(fwd, left, th)
    assert np.allclose(T[:3, 3], [-left, 0., fwd])          # camera x = -left, z = forward
    assert np.isclose(math.atan2(-T[0, 2], T[2, 2]), th)     # turn about the up axis


def test_plan_maps_skips_fusion_without_fresh_map():
    from cross_nav.policy.bev_policy import BEVPlannerPolicy, Episode
    if not torch.cuda.is_available():
        pytest.skip('BEVPlannerPolicy sets a CUDA device')
    pred = types.SimpleNamespace(size=100, res=.2)
    cfg = {'map': {}, 'history_rule': None}
    pol = BEVPlannerPolicy(cfg, predictor=pred)
    st = Episode()
    N = 100
    cat = {'prob': np.zeros((1, N, N), np.float32), 'obs': np.ones((1, N, N), np.float32),
           'err': np.zeros((1, N, N), np.float32), 'log_scale': np.zeros(1, np.float32), 'odometry': np.array([True]),
           'up_head_deg': np.zeros(1, np.float32), 'n_views': np.ones(1), 'fresh': np.array([False])}
    paths, _ = pol.plan_maps(cat, np.array([[3., 0.]], np.float32), [st], [{'ep': 'e', 'odom': np.zeros(3)}], [False])
    assert st.world is None and paths.shape == (1, 16, 2)
    cat['fresh'] = np.array([True])
    pol.plan_maps(cat, np.array([[3., 0.]], np.float32), [st], [{'ep': 'e', 'odom': np.zeros(3)}], [False])
    assert st.world is not None


def test_visibility_head_inputs():
    """cross_nav.train_visibility.head_inputs: 'head' passes the estimates through; 'gtmix' gives the GT scale / gravity
    (no noise, probabilities 0) and the estimates with probability 1."""
    from cross_nav.train_visibility import head_inputs
    B, S, H, W = 2, 3, 32, 32       # gt_log_scale needs >= 500 valid pixels
    depth = torch.ones(B, S, H, W)
    bb = {"log_scale_head": torch.full((B,), 0.5), "depth": depth}
    batch = {"depths": depth * 2.0, "masks": torch.ones(B, S, H, W, dtype=torch.bool),
             "up_cam": torch.tensor([[0., -1., 0.]] * B)}
    up_pred = torch.tensor([[0., 0., 1.]] * B)
    ls, up = head_inputs(bb, batch, up_pred, {}, "head")
    assert torch.equal(ls, bb["log_scale_head"]) and torch.equal(up, up_pred)
    exact = {"scale_noise": 0.0, "p_scale_head": 0.0, "gravity_noise_deg": 0.0, "p_gravity_pred": 0.0}
    ls, up = head_inputs(bb, batch, up_pred, exact, "gtmix")
    assert torch.allclose(ls, torch.full((B,), math.log(2.0))) and torch.allclose(up, batch["up_cam"])
    ls, up = head_inputs(bb, batch, up_pred, {**exact, "p_scale_head": 1.0, "p_gravity_pred": 1.0}, "gtmix")
    assert torch.allclose(ls, bb["log_scale_head"]) and torch.allclose(up, up_pred)
