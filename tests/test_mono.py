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


def test_scale_recovery_requires_repeated_consistent_evidence():
    filter_ = LogScaleFilter()
    def observation(scale):
        return ScaleObservation(log_scale=np.log(scale), variance=0.12**2, accepted=True)
    filter_.update(observation(4))
    assert not filter_.update(observation(1))
    assert filter_.uncertainty_variance > filter_.variance
    assert not filter_.update(observation(8))  # contradicts pending alternative
    assert not filter_.update(observation(1))
    assert not filter_.update(observation(1.05))
    assert filter_.scale == pytest.approx(4)
    assert filter_.update(observation(0.95))
    assert filter_.scale == pytest.approx(1)
    assert filter_.reinitializations == 1
    assert filter_.uncertainty_variance == filter_.variance


def test_sparse_metric_scale_requires_spatially_distributed_patches():
    from cross.mono.scale import observe_sparse_scale
    rng = np.random.default_rng(12)
    pixels = rng.uniform([0, 0], [640, 480], (96, 2))
    source = rng.uniform(1, 4, 96)
    observation = observe_sparse_scale(source * 1.8, source, pixels, (480, 640))
    assert observation.accepted
    assert np.exp(observation.log_scale) == pytest.approx(1.8)
    collapsed = observe_sparse_scale(source * 1.8, source, pixels * 0.1, (480, 640))
    assert not collapsed.accepted


def test_camera_convention_and_shared_scale_covariance():
    ref, current = np.eye(4), np.eye(4)
    current[0, 3] = -0.3  # world-to-camera, camera moved +x
    assert relative_pose(ref, current, 2)[0, 3] == pytest.approx(0.6)
    covariance = scale_translation_covariance(np.array([1., 2., 3.]), 0.2)
    assert np.linalg.matrix_rank(covariance) == 1
    assert covariance[0, 1] == pytest.approx(0.4)


def test_retrieved_geometry_rejects_wrong_camera_direction():
    from cross.mono.verification import reprojection_inliers
    K = np.array([[100., 0, 50.], [0, 100., 50.], [0, 0, 1.]])
    pixels = np.array([[40., 40.], [50., 50.], [60., 60.]])
    depth = np.full((100, 100), 2.)
    T_ref_current = np.eye(4)
    T_ref_current[0, 3] = 0.2
    current = pixels - [10., 0.]
    assert reprojection_inliers(pixels, current, depth, K, K, T_ref_current).all()
    T_ref_current[0, 3] = -0.2
    assert not reprojection_inliers(pixels, current, depth, K, K, T_ref_current).any()


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


@pytest.mark.parametrize("metric_shape", [False, True])
def test_changing_model_gauge_does_not_change_metric_trajectory(metric_shape):
    config = MonoConfig(anchor_interval=3, scale=ScaleConfig(interval=3), metric_shape=metric_shape)
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


def test_scale_replay_is_causal_and_preserves_source_geometry():
    from dataclasses import asdict
    from cross.mono.replay_scale import replay

    # Unit increments of one metre; a changed teacher prior changes only new
    # increments. Startup-only must recover a straight constant-speed track.
    original_scales = np.array([2., 2., 3., 3.])
    positions = np.cumsum(original_scales)
    rows = np.zeros((4, 8))
    rows[:, 0], rows[:, 1], rows[:, 7] = np.arange(4), positions, 1
    diagnostics = [{"scale": s} for s in original_scales]
    for i in (0, 2):
        diagnostics[i]["scale_observation"] = asdict(ScaleObservation(
            log_scale=np.log(original_scales[i]), variance=0.12**2, accepted=True, reason="accepted"))
    direct, _ = replay(rows, diagnostics, ScaleConfig(mode="direct"))
    initial, _ = replay(rows, diagnostics, ScaleConfig(mode="initial"))
    prefix, _ = replay(rows[:2], diagnostics[:2], ScaleConfig(mode="initial"))
    np.testing.assert_allclose(direct, rows)
    np.testing.assert_allclose(initial[:, 1], [2., 4., 6., 8.])
    np.testing.assert_array_equal(initial[:2], prefix)
    np.testing.assert_array_equal(initial[:, 4:], rows[:, 4:])
