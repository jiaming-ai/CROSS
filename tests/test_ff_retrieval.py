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


def test_fallback_keeps_primary_poses_and_fills_rejected_references_in_order():
    pp = pytest.importorskip("pypose")
    from types import SimpleNamespace
    from cross.mono.ff_retrieval import FallbackFeedForwardRelativePose

    def poses(xs):
        p = pp.identity_SE3(len(xs))
        for i, x in enumerate(xs):
            p[i, 0] = x
        return p

    class Primary:
        def estimate_pose(self, ref, ref_depth, cur, cur_depth, **kw):
            self.last_pair_audit = [dict(reason="accepted"), dict(reason="two_view_rejected"), dict(reason="few_matches")]
            self.last_stds = torch.full((1, 6), .1)
            return poses([1.]), np.array([True, False, False]), torch.tensor([.5])

    class FeedForward:
        device = "cpu"

        def estimate_pose(self, ref, ref_depth, cur, cur_depth, **kw):
            assert len(ref) == 2           # only the two rejected references
            self.last_pair_audit = [dict(reason="accepted", accepted=True), dict(reason="low_covisibility", accepted=False)]
            self.last_stds = torch.full((1, 6), .2)
            return poses([2.]), np.array([True, False]), torch.tensor([.4])

    estimator = FallbackFeedForwardRelativePose(Primary(), FeedForward())
    ref = torch.zeros(3, 3, 8, 8)
    out, valid, conf = estimator.estimate_pose(ref, torch.ones(3, 1, 8, 8), torch.zeros(3, 8, 8), None)
    assert valid.tolist() == [True, True, False]
    assert out.tensor()[:, 0].tolist() == [1., 2.]
    assert conf.tolist() == pytest.approx([.5, .4])
    assert estimator.last_stds[:, 0].tolist() == pytest.approx([.1, .2])
    assert estimator.last_pair_audit[1]["reason"] == "accepted"
    assert estimator.last_pair_audit[2]["reason"] == "few_matches"
