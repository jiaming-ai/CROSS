from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cross.mono.config import MonoConfig
from cross.mono.motion_uncertainty import correlated_interval_bound, diagonal_envelope


def adjoint(T):
    R = T[:3, :3]
    x, y, z = T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = A[3:, 3:] = R
    A[:3, 3:] = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) @ R
    return A


def test_bound_dominates_correlated_joint_errors_without_independence():
    rng = np.random.default_rng(29)
    # Each increment depends on the SAME latent vector. Off-diagonal blocks
    # arise from this independent construction, rather than the bound formula.
    for n in (1, 2, 7, 20):
        for _ in range(12):
            responses = rng.normal(size=(n, 6, 11))
            exact = responses.sum(axis=0) @ responses.sum(axis=0).T
            moments = np.einsum('ijk,ilk->jl', responses, responses)
            bound = correlated_interval_bound(np.zeros((6, 6)), moments, np.eye(6), n)
            assert np.linalg.eigvalsh(bound - exact).min() >= -1e-10
    # Fully correlated equal diagonal errors saturate Cauchy--Schwarz.
    Q = np.diag(np.arange(1., 7.))
    np.testing.assert_allclose(correlated_interval_bound(np.zeros((6, 6)), 7*Q, np.eye(6), 7), 49*Q)


def test_diagonal_interface_does_not_discard_covariance_cross_terms():
    v = np.array([1., -2., 3., -.1, .2, -.3])
    Q = np.outer(v, v)
    assert np.linalg.eigvalsh(np.diag(Q.diagonal()) - Q).min() < -1
    assert np.linalg.eigvalsh(diagonal_envelope(Q) - Q).min() >= -1e-12


def test_motion_bound_is_invariant_to_world_chart_and_retains_skipped_frames():
    pytest.importorskip('torch')
    from cross.mono.streaming import snapshot_motion
    rng = np.random.default_rng(42)
    poses = np.repeat(np.eye(4)[None], 23, axis=0)
    poses[:, :3, :3] = Rotation.from_rotvec(rng.normal(size=(23, 3))*.15).as_matrix()
    poses[:, :3, 3] = rng.normal(size=(23, 3))
    factors = rng.normal(size=(23, 6, 6))*.01
    Q = factors @ factors.transpose(0, 2, 1)
    gauge = np.eye(4)
    gauge[:3, :3] = Rotation.from_rotvec([.4, -.7, .8]).as_matrix()
    gauge[:3, 3] = [30., -15., 11.]
    results = []
    for G in (np.eye(4), gauge):
        moved = G @ poses
        prefix = np.cumsum([adjoint(T) @ q @ adjoint(T).T for T, q in zip(moved, Q)], axis=0)
        first, last = [SimpleNamespace(index=i, pose=moved[i], world_geometry_covariance_prefix=prefix[i])
                       for i in (3, 22)]
        delta, bound = snapshot_motion(first, last, conditional=True, covariance_bound='matrix')
        direct = sum(adjoint(np.linalg.inv(moved[22]) @ moved[i]) @ Q[i]
                     @ adjoint(np.linalg.inv(moved[22]) @ moved[i]).T for i in range(4, 23))
        np.testing.assert_allclose(bound, diagonal_envelope(19*direct), rtol=1e-9, atol=1e-10)
        results.append((delta, bound))
    for i in (0, 1):
        np.testing.assert_allclose(results[0][i], results[1][i], rtol=1e-9, atol=1e-10)


def test_invalid_motion_prefixes_and_configurations_fail_explicitly():
    Z = np.zeros((6, 6))
    for steps in (0, -1, 1.5):
        with pytest.raises(ValueError, match='step'):
            correlated_interval_bound(Z, Z, np.eye(6), steps)
    with pytest.raises(ValueError, match='finite'):
        correlated_interval_bound(Z, np.full((6, 6), np.nan), np.eye(6), 1)
    with pytest.raises(ValueError, match='semidefinite'):
        diagonal_envelope(-np.eye(6))
    with pytest.raises(ValueError, match='conditional'):
        MonoConfig(motion_covariance_bound='matrix')
    with pytest.raises(ValueError, match='Unknown'):
        MonoConfig(motion_covariance_bound='unknown')
    config = MonoConfig(frontend='streaming_pnp', conditional_sources=True, chart_aware=True,
                        session_recovery=True, retrieval_pose='metric_pnp', motion_covariance_bound='matrix')
    assert config.motion_covariance_bound == 'matrix'


def test_streaming_prefix_excludes_metric_bias_and_snapshots_own_their_matrices():
    pytest.importorskip('torch')
    from test_metric_source_streaming import make_frontend
    frontend = make_frontend(True)
    frontend.world_geometry_covariance_prefix = np.zeros((6, 6))
    expected = np.zeros((6, 6))
    snapshots = []
    for i in range(6):
        result = frontend.step(np.full((4, 4, 3), i, np.uint8), i*.05)
        Q = np.diag([frontend.config.translation_std_floor**2]*3 + [frontend.config.rotation_std_floor**2]*3)
        if not result.diagnostics['valid']:
            Q += np.eye(6)
        A = adjoint(result.pose)
        expected += A @ Q @ A.T
        np.testing.assert_allclose(frontend.world_geometry_covariance_prefix, expected, atol=1e-12)
        for (snapshot, _), _ in frontend.take_depths():
            snapshots.append((snapshot, snapshot.world_geometry_covariance_prefix.copy()))
    assert snapshots
    for snapshot, original in snapshots:
        np.testing.assert_array_equal(snapshot.world_geometry_covariance_prefix, original)
        assert not np.shares_memory(snapshot.world_geometry_covariance_prefix, frontend.world_geometry_covariance_prefix)
