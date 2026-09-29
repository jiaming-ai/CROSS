"""Feed-forward retrieval geometry helpers (no neural model)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cross.mono.ff_retrieval import covisibility, depth_scale


def views(offset=0.):
    K = np.array([[100., 0, 31.5], [0, 100., 23.5], [0, 0, 1]])
    c2w = np.stack([np.eye(4), np.eye(4)])
    c2w[1, 0, 3] = offset
    depth = torch.full((2, 48, 64), 2.)
    return c2w, np.stack([K, K]), depth


def test_identical_views_are_fully_covisible_and_disjoint_views_are_not():
    c2w, K, depth = views()
    assert covisibility(c2w, K, depth, None, 0, 1) == pytest.approx(1.)
    c2w, K, depth = views(offset=50.)
    assert covisibility(c2w, K, depth, None, 0, 1) == 0.


def test_depth_scale_recovers_ratio_and_rejects_missing_depth():
    predicted = np.full((30, 40), 2., dtype=np.float32)
    log_scale, mad = depth_scale(np.full((60, 80), 3.), predicted)
    assert log_scale == pytest.approx(np.log(1.5)) and mad == pytest.approx(0.)
    assert depth_scale(np.full((60, 80), np.nan), predicted) == (None, None)
