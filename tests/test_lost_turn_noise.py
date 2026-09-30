"""Lost-turn regime of the odometry noise model (NoiseModelConfig.odom_lost_turn_rad / odom_k_lost)."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pp = pytest.importorskip("pypose")
pytest.importorskip("gtsam")

from cross.core.config import NoiseModelConfig
from cross.core.lc_verify import NoiseModel
from cross.core.odom_accum import OdomAccumulator
from cross.core.types import Edge, EdgeType


def _yaw(angle, x=0.0):
    T = np.eye(4)
    T[:3, :3] = [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    T[0, 3] = x
    return T


def test_default_is_unchanged():
    noise = NoiseModel(NoiseModelConfig())
    assert np.allclose(noise.odom(1.0, 1.5, 4), noise.odom(1.0, 1.5, 4, max_step_rot=3.0))


def test_lost_turn_inflates_rotation_and_translation():
    noise = NoiseModel(NoiseModelConfig(odom_lost_turn_rad=math.radians(30), odom_k_lost=1.0))
    smooth = noise.odom(1.0, 1.5, 10, max_step_rot=math.radians(15))
    lost = noise.odom(1.0, 1.5, 10, max_step_rot=math.radians(90))
    assert np.allclose(smooth, NoiseModel(NoiseModelConfig()).odom(1.0, 1.5, 10))
    assert np.allclose(lost[:3] - smooth[:3], 1.5) and np.allclose(lost[3:] - smooth[3:], 1.0)


def test_factor_carries_max_step_rotation():
    noise = NoiseModel(NoiseModelConfig(odom_lost_turn_rad=math.radians(30), odom_k_lost=1.0))
    edge = Edge(pp.from_matrix(torch.tensor(_yaw(1.2, 0.5), dtype=torch.float32), pp.SE3_type),
                pp.se3(torch.full((6,), 0.1)), EdgeType.ODOMETRY)
    edge.n_frames = 4
    base = noise.odom_from_factor(edge)
    edge.max_step_rot = 1.0
    assert np.all(noise.odom_from_factor(edge) > base)


def test_accumulator_tracks_largest_step_until_reset():
    accumulator = OdomAccumulator(device="cpu")
    accumulator.register_item("since_last_add_kf")
    for angle in (0.05, 0.9, 0.1):
        accumulator.update_odom(_yaw(angle, 0.1))
    assert accumulator.max_step_rotation("since_last_add_kf") == pytest.approx(0.9, abs=1e-5)
    accumulator.reset_item("since_last_add_kf")
    assert accumulator.max_step_rotation("since_last_add_kf") == 0.0
    accumulator.update_odom(_yaw(0.2))
    assert accumulator.max_step_rotation("since_last_add_kf") == pytest.approx(0.2, abs=1e-5)
