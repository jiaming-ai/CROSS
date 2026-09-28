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
    assert not MonoConfig().two_view_rotation_check
    with pytest.raises(ValueError, match='metric_two_view'):
        MonoConfig(two_view_rotation_check=True)
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
    estimator.fallback_features = estimator.features
    estimator.refiner = SimpleNamespace(K=np.eye(3), match=lambda *_: (np.zeros((30, 2)), np.zeros((30, 2))))
    estimator.fallback_refiner = estimator.refiner
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


def test_learned_rotation_screen_rejects_without_new_evidence(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('pypose')
    from collections import OrderedDict
    from cross.mono.two_view_retrieval import MetricTwoViewRelativePose
    from cross.mono import two_view_retrieval as module

    estimator = MetricTwoViewRelativePose.__new__(MetricTwoViewRelativePose)
    estimator.device, estimator.conditional_sources, estimator.source_log_std = 'cpu', True, .12
    estimator.factor_policy = 'same-physical-pair'
    features = dict(keypoints=np.zeros((30,2)),shape=(8,8))
    estimator.features = estimator.fallback_features = lambda rgb: features
    estimator.refiner = SimpleNamespace(K=np.eye(3),match=lambda *_: (np.zeros((30,2)),np.zeros((30,2))))
    estimator.fallback_refiner = estimator.refiner
    def failed(*_):
        estimator.refiner.last_match_audit = dict(reason='pnp_consensus')
        return None
    estimator.refiner.estimate = failed
    pose = np.eye(4); pose[0,3] = .5
    monkeypatch.setattr(module,'estimate_metric_two_view',lambda *_: (pose.copy(),dict(metric_inliers=30,median_reprojection_px=.2)))
    calls = []
    extrinsics = np.array([np.eye(4),np.eye(4)])
    extrinsics[1,:3,:3] = cv2.Rodrigues(np.array([0.,.2,0.]))[0]
    def predict(images):
        calls.append(len(images))
        return SimpleNamespace(extrinsics=extrinsics)
    estimator.rotation_geometry = SimpleNamespace(prepare=lambda x:x,predict=predict,model_id='test',resolution=336)
    estimator.rotation_cache = OrderedDict()
    rgb, depth = torch.zeros(1,3,8,8),torch.ones(1,8,8)
    kwargs = dict(ref_metric_sources=[dict(source_id='reference')],curr_metric_source=dict(source_id='current'))
    _, valid, _ = estimator.estimate_pose(rgb,depth,rgb[0],depth[0],**kwargs)
    assert not valid.any() and not estimator.last_conditional_poses
    assert estimator.last_pair_audit[0]['reason'] == 'learned_rotation_rejected'
    assert np.isclose(estimator.last_pair_audit[0]['rotation_check']['rotation_difference_rad'],.2)
    extrinsics[1] = np.eye(4)
    poses, valid, confidence = estimator.estimate_pose(rgb,depth,rgb[0],depth[0],**kwargs)
    assert valid.tolist() == [True] and len(poses) == len(confidence) == 1
    np.testing.assert_allclose(poses[0].matrix().numpy(),pose,atol=1e-7)
    factor = estimator.last_conditional_poses[0].factor
    assert factor.factor_id == hashlib.sha256(b'same-physical-pairreferencecurrent').hexdigest()
    assert factor.keys == ('image:reference',)
    _, valid, _ = estimator.estimate_pose(rgb,depth,rgb[0],depth[0],excluded_factor_ids=[factor.factor_id],**kwargs)
    assert not valid.any() and calls == [2,2]


def test_rotation_screen_uses_reference_current_convention_and_rejects_nonfinite():
    pytest.importorskip('torch')
    from collections import OrderedDict
    from cross.mono.two_view_retrieval import MetricTwoViewRelativePose
    from cross.mono.geometry import inverse
    estimator = MetricTwoViewRelativePose.__new__(MetricTwoViewRelativePose)
    pose = np.eye(4); pose[:3,:3] = cv2.Rodrigues(np.array([.2,-.3,.1]))[0]
    extrinsics = np.array([np.eye(4),inverse(pose)])
    estimator.rotation_geometry = SimpleNamespace(prepare=lambda x:x,model_id='test',resolution=336,
        predict=lambda _:SimpleNamespace(extrinsics=extrinsics))
    estimator.rotation_cache = OrderedDict()
    image = np.zeros((8,8,3),np.uint8)
    audit = estimator._check_rotation(image,image,pose)
    assert audit['accepted'] and audit['rotation_difference_rad'] < 1e-10
    extrinsics[1,0,0] = np.nan
    audit = estimator._check_rotation(image,image,pose)
    assert not audit['accepted'] and audit['reason'] == 'nonfinite_prediction'


def test_verified_primary_pose_never_calls_fallback():
    torch = pytest.importorskip('torch')
    pytest.importorskip('pypose')
    from cross.mono.two_view_retrieval import MetricTwoViewRelativePose
    estimator = MetricTwoViewRelativePose.__new__(MetricTwoViewRelativePose)
    estimator.device, estimator.conditional_sources = 'cpu', False
    estimator.features = lambda rgb: rgb
    estimator.refiner = SimpleNamespace()
    T = np.eye(4); T[0, 3] = .4
    values = iter([(T, 70, .2), (np.linalg.inv(T), 60, .3)])
    def primary(*_):
        estimator.refiner.last_match_audit = dict(reason='accepted')
        return next(values)
    estimator.refiner.estimate = primary
    def forbidden(*_):
        raise AssertionError('A verified primary pose must not request fallback features')
    estimator.fallback_features = forbidden
    estimator.rotation_geometry = SimpleNamespace(predict=forbidden)
    rgb, depth = torch.zeros(1, 3, 8, 8), torch.ones(1, 8, 8)
    poses, valid, _ = estimator.estimate_pose(rgb, depth, rgb[0], depth[0])
    assert valid.tolist() == [True]
    np.testing.assert_allclose(poses[0].matrix().numpy(), T, atol=1e-7)
    assert estimator.last_pair_audit[0]['proposal_method'] == 'metric_pnp'
