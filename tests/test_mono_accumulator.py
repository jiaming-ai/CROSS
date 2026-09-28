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
