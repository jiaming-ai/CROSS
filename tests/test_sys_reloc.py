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
