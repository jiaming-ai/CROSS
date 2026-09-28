"""Geometry conventions, gauge response, and integration of the experimental fallback."""
import hashlib
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cross.mono.config import MonoConfig
from cross.mono.two_view_geometry import estimate_metric_two_view


def scene():
    K = np.array([[400., 0, 256], [0, 400., 192], [0, 0, 1.]])
    X = np.random.default_rng(4).uniform([-2, -1, 4], [2, 1, 9], (160, 3))
    R = cv2.Rodrigues(np.array([.01, .05, -.02]))[0]
    t = np.array([.6, .02, .01])
    Y = X @ R.T + t
    p, q = X @ K.T, Y @ K.T
    return K, X, Y, R, t, p[:, :2]/p[:, 2:], q[:, :2]/q[:, 2:]


def test_calibrated_geometry_metric_gauge_and_rotation_degeneracy():
    K, X, Y, R, t, p, q = scene()
    pose, audit = estimate_metric_two_view(p, q, X[:, 2], Y[:, 2], K, (384, 512))
    assert audit['accepted']
    np.testing.assert_allclose(pose[:3, :3], R.T, atol=1e-4)
    np.testing.assert_allclose(pose[:3, 3], -R.T @ t, atol=1e-4)
    # Reference prior supplies magnitude once; the current prior is a screen.
    scaled, audit = estimate_metric_two_view(p, q, X[:, 2]*1.25, Y[:, 2], K, (384, 512))
    assert audit['accepted']
    np.testing.assert_allclose(scaled[:3, :3], pose[:3, :3], atol=1e-10)
    np.testing.assert_allclose(scaled[:3, 3], pose[:3, 3]*1.25, atol=1e-10)
    bad, audit = estimate_metric_two_view(p, q, X[:, 2], Y[:, 2]*4, K, (384, 512))
    assert bad is None and audit['reason'] == 'inconsistent_metric_scale'
    rotated = X @ R.T
    pixels = rotated @ K.T
    bad, audit = estimate_metric_two_view(p, pixels[:, :2]/pixels[:, 2:], X[:, 2], rotated[:, 2], K, (384, 512))
    assert bad is None


def test_two_view_configuration_preserves_default_and_requires_explicit_backend():
    assert MonoConfig().retrieval_pose == 'da3'
    with pytest.raises(ValueError, match='superpoint_lightglue'):
        MonoConfig(retrieval_pose='metric_two_view')
    config = MonoConfig(frontend='streaming_pnp', retrieval_pose='metric_two_view',
                        retrieval_matcher='superpoint_lightglue', conditional_sources=True,
                        session_recovery=True, chart_aware=True)
    assert config.trace_metric_sources


def test_fallback_keeps_one_component_and_deduplicates_source_factor(monkeypatch):
    torch = pytest.importorskip('torch')
    pp = pytest.importorskip('pypose')
    from cross.mono.two_view_retrieval import MetricTwoViewRelativePose
    from cross.mono import two_view_retrieval as module
    from cross.mono.metric_sources import pnp_scale_response

    estimator = MetricTwoViewRelativePose.__new__(MetricTwoViewRelativePose)
    estimator.device, estimator.conditional_sources, estimator.source_log_std = 'cpu', True, .12
    estimator.factor_policy = 'test-two-view'
    features = dict(keypoints=np.zeros((30, 2)), shape=(8, 8))
    estimator.features = lambda rgb: features
    estimator.refiner = SimpleNamespace(K=np.eye(3), match=lambda *_: (np.zeros((30, 2)), np.zeros((30, 2))))
    def failed(*_):
        estimator.refiner.last_match_audit = dict(reason='pnp_consensus')
        return None
    estimator.refiner.estimate = failed
    T = np.eye(4); T[0, 3] = 2.
    calls = []
    def fallback(*args):
        calls.append(1)
        return T.copy(), dict(accepted=True, metric_inliers=30, median_reprojection_px=.5)
    monkeypatch.setattr(module, 'estimate_metric_two_view', fallback)
    images, depths = torch.zeros(1, 3, 8, 8), torch.ones(1, 8, 8)
    kwargs = dict(ref_metric_sources=[dict(source_id='reference')], curr_metric_source=dict(source_id='current'))
    poses, valid, confidence = estimator.estimate_pose(images, depths, images[0], depths[0], **kwargs)
    assert valid.tolist() == [True] and poses.shape == (1, 7) and len(confidence) == 1
    assert len(estimator.last_conditional_poses) == 1
    factor = estimator.last_conditional_poses[0].factor
    np.testing.assert_allclose(factor.jacobian[:, 0], pnp_scale_response(T))
    assert factor.keys == ('image:reference',)
    identity = hashlib.sha256(b'test-two-viewreferencecurrent').hexdigest()
    assert factor.factor_id == identity
    assert estimator.last_pair_audit[0]['pnp_rejection'] == 'forward_pnp'
    poses, valid, _ = estimator.estimate_pose(images, depths, images[0], depths[0], excluded_factor_ids=[identity], **kwargs)
    assert not valid.any() and len(poses) == 0 and len(calls) == 1
    assert estimator.last_pair_audit[0]['reason'] == 'reused_geometric_factor'
