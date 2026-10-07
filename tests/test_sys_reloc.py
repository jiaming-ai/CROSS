import numpy as np
import torch

from cross.cv.stereo_scale import ScaleAnchor, estimate_scale


def _c2w(centres):
    c2w = np.repeat(np.eye(4)[None], len(centres), 0)
    c2w[:, :3, 3] = np.asarray(centres, dtype=np.float64)
    return c2w


def _anchor(c2w_true, a, b, kind="map"):
    T = np.linalg.inv(c2w_true[a]) @ c2w_true[b]
    return ScaleAnchor(a, b, T, kind=kind)


def test_view_jackknife_floors_anchors_through_one_view():
    # predicted geometry: views 1, 2, 4 placed right (scale 2), view 3 placed at a wrong distance
    true = _c2w([[0, 0, 0], [2, 0, 0], [0, 2, 0], [6, 6, 0], [-2, 0, 0]])
    pred = _c2w([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0], [-1, 0, 0]])
    through_3 = [_anchor(true, 1, 3), _anchor(true, 2, 3), _anchor(true, 3, 4)]
    plain = estimate_scale(pred, through_3, max_rot_err_deg=180, min_dir_cos=-1)
    jk = estimate_scale(pred, through_3, max_rot_err_deg=180, min_dir_cos=-1, view_logstd_floor=0.5)
    assert plain.log_std < 0.2               # the shared view makes the anchors look consistent
    assert jk.log_std >= 0.5                 # one view carries every anchor
    assert abs(jk.scale - plain.scale) < 1e-9


def test_view_jackknife_keeps_odometry_only_and_consistent_anchors():
    true = _c2w([[0, 0, 0], [2, 0, 0], [0, 2, 0], [-2, 0, 0], [0, 0, 0.6]])
    pred = true.copy()
    pred[:, :3, 3] /= 2.0
    odom = [_anchor(true, 4, 0, kind="odom")]
    assert estimate_scale(pred, odom, view_logstd_floor=0.5).log_std == 0.0
    maps = [_anchor(true, 1, 2), _anchor(true, 2, 3), _anchor(true, 1, 3)] + odom
    est = estimate_scale(pred, maps, max_rot_err_deg=180, min_dir_cos=-1, view_logstd_floor=0.5)
    assert est.log_std < 1e-6 and abs(est.scale - 2.0) < 1e-6


def test_strong_pass_effective_frames():
    from cross.core.hypothesis import HypothesisManager
    hm = HypothesisManager.__new__(HypothesisManager)
    hm.strong_pass_frames, hm.strong_pass_min_refs = 3, 2
    hm.hist_valid = torch.tensor([[True, False, False], [True, True, False]])
    hm.strong_hist = torch.tensor([[True, False, False], [False, False, False]])
    n_valid = hm.hist_valid.sum(dim=1)
    assert hm._effective_frames(n_valid).tolist() == [3, 2]
    assert hm._effective_frames(n_valid[1:], torch.tensor([1])).tolist() == [2]
    assert hm._is_strong({"strong_refs": 2}) and not hm._is_strong({"strong_refs": 1})
    hm.strong_pass_frames = 0
    assert hm._effective_frames(n_valid).tolist() == [1, 2] and not hm._is_strong({"strong_refs": 5})
