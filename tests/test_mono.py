import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cross.mono.config import MonoConfig, ScaleConfig
from cross.mono.evaluate import evaluate
from cross.mono.frontend import MonoFrontend
from cross.mono.geometry import relative_pose, scale_translation_covariance
from cross.mono.scale import LogScaleFilter, ScaleObservation, observe_scale


def test_robust_metric_scale_and_spatial_uncertainty():
    rng = np.random.default_rng(4)
    source = rng.uniform(0.5, 5, (80, 96))
    target = source * 2.7 * np.exp(rng.normal(0, 0.015, source.shape))
    target[:15] *= 15
    target[20:25] = np.nan
    observation = observe_scale(target, source)
    assert observation.accepted
    assert np.exp(observation.log_scale) == pytest.approx(2.7, rel=0.01)
    assert observation.variance >= 0.12**2
    observation2 = observe_scale(target, source * 4)
    assert observation2.log_scale == pytest.approx(observation.log_scale - np.log(4))


def test_scale_filter_rejects_outlier_and_retains_systematic_floor():
    filter_ = LogScaleFilter()
    for _ in range(100):
        filter_.update(ScaleObservation(log_scale=np.log(2), variance=0.12**2, accepted=True))
    assert filter_.scale == pytest.approx(2)
    assert filter_.variance >= 0.08**2
    assert not filter_.update(ScaleObservation(log_scale=np.log(20), variance=0.12**2, accepted=True))
    assert filter_.scale == pytest.approx(2)
    assert not observe_scale(np.zeros((80, 80)), np.ones((80, 80))).accepted


def test_camera_convention_and_shared_scale_covariance():
    ref, current = np.eye(4), np.eye(4)
    current[0, 3] = -0.3  # world-to-camera, camera moved +x
    assert relative_pose(ref, current, 2)[0, 3] == pytest.approx(0.6)
    covariance = scale_translation_covariance(np.array([1., 2., 3.]), 0.2)
    assert np.linalg.matrix_rank(covariance) == 1
    assert covariance[0, 1] == pytest.approx(0.4)


class SyntheticGeometry:
    def __init__(self):
        self.calls = 0

    def prepare(self, rgb):
        return int(rgb[0, 0, 0])

    def predict(self, images):
        from types import SimpleNamespace
        self.calls += 1
        gauge = 0.5 + self.calls % 4
        matrices = np.broadcast_to(np.eye(4), (len(images), 4, 4)).copy()
        matrices[:, 0, 3] = -np.array(images) * 0.1 * gauge
        depth = np.ones((len(images), 64, 64)) * 2 * gauge
        return SimpleNamespace(extrinsics=matrices, depth=depth, confidence=np.ones_like(depth))


class SyntheticMetric:
    def predict_metric(self, rgb, K, output_shape):
        return np.full(output_shape, 2.0)


def test_changing_model_gauge_does_not_change_metric_trajectory():
    config = MonoConfig(anchor_interval=3, scale=ScaleConfig(interval=3))
    tracker = MonoFrontend(np.diag([500., 500., 1.]), config, geometry_model=SyntheticGeometry(), metric_model=SyntheticMetric())
    for i in range(12):
        result = tracker.step(np.full((64, 64, 3), i, dtype=np.uint8), i * 0.1)
        assert result.diagnostics["valid"]
        assert result.pose[0, 3] == pytest.approx(i * 0.1, abs=1e-6)
        assert np.median(result.depth) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="timestamps"):
        tracker.step(np.zeros((64, 64, 3), dtype=np.uint8), 0.0)


def test_evaluator_exposes_metric_error_hidden_by_similarity(tmp_path):
    times = np.arange(21) * 0.1
    points = np.c_[np.sin(times), np.cos(times), times * 0.1]
    quaternions = Rotation.from_rotvec(np.c_[times * 0, times * 0, times * 0.1]).as_quat()
    truth = tmp_path / "truth.txt"
    estimate = tmp_path / "estimate.txt"
    np.savetxt(truth, np.c_[times, points, quaternions])
    np.savetxt(estimate, np.c_[times, points * 2, quaternions])
    metrics = evaluate(estimate, truth, max_gap=0.11)
    assert metrics["ate_sim3_rmse_m"] < 1e-8
    assert metrics["ate_se3_rmse_m"] > 0.1
    assert metrics["sim3_alignment_scale"] == pytest.approx(0.5)
    assert metrics["metric_scale_error_percent"] == pytest.approx(100.)
    assert metrics["rpe_translation_rmse_m"] > 0.1
