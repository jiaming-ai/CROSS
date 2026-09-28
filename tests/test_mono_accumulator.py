"""Requires the mapping extra; small CPU-only covariance regression tests."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pypose")
from cross.core.odom_accum import OdomAccumulator


def test_shared_scale_uncertainty_is_not_clipped_or_counted_independently():
    accumulator = OdomAccumulator(device="cpu", min_std_translation=0, min_std_rotation=0)
    accumulator.register_item("keyframe")
    motion = np.eye(4)
    motion[0, 3] = 1
    covariance = np.diag([0.25, 0., 0., 0., 0., 0.])
    for _ in range(3):
        accumulator.update_odom(motion, covariance=covariance)
    delta, std = accumulator.get_since_last_reading("keyframe")
    assert delta.tensor()[0].item() == pytest.approx(3)
    assert std.tensor()[0].item() == pytest.approx(1.5)
    _, reset_std = accumulator.get_since_last_reading("keyframe")
    assert reset_std.tensor()[0].item() == 0


def test_skipped_retrieval_cannot_reduce_motion_prior_uncertainty():
    from types import SimpleNamespace
    import pypose as pp
    from cross.core.hypothesis import HypothesisManager

    manager = HypothesisManager(SimpleNamespace(device="cpu"), 1)
    prior = pp.identity_SE3(1)
    prior[0, 0] = 0.5
    prior_std = pp.se3(torch.full((1, 6), 0.25))
    manager.dist = (prior.clone(), prior_std.clone(), torch.ones(1))
    # Repeated correlated retrievals are permitted to affect place evidence,
    # but a gated pose update must not manufacture motion information.
    for _ in range(5):
        manager.gmm_filtering(pp.identity_SE3(1), pp.se3(torch.full((1, 6), 0.01)),
                              torch.ones(1), torch.ones(1), pose_update_mask=torch.tensor([False]))
        torch.testing.assert_close(manager.dist[0].tensor(), prior.tensor())
        torch.testing.assert_close(manager.dist[1].tensor(), prior_std.tensor())


def test_background_patch_selection_excludes_people_and_reports_no_support():
    from cross.mono.background_patches import background_indices
    points = torch.tensor([[10., 10.], [20., 20.], [30., 30.], [40., 40.]])
    boxes = torch.tensor([[0., 0., 25., 25.]])
    indices, supported = background_indices(points, boxes, 3)
    assert supported == 2
    assert indices.tolist() == [2, 3, 2]
    _, supported = background_indices(points, torch.tensor([[0., 0., 50., 50.]]), 3)
    assert supported == 0
