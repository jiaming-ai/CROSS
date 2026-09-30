import numpy as np
from scipy.spatial.transform import Rotation

from cross.cv.stereo_scale import ScaleAnchor, estimate_scale, invert_poses, scale_camera_centers


def _rig(baseline=0.5):
    T = np.eye(4)
    T[0, 3] = baseline
    return T


def _random_traj(n, rng):
    poses = []
    for i in range(n):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.1, 3)).as_matrix()
        T[:3, 3] = rng.normal(0, 3.0, 3)
        poses.append(T)
    return np.stack(poses)


def test_scale_recovery_exact():
    rng = np.random.default_rng(0)
    B = _rig(0.5)
    left = _random_traj(4, rng)                       # metric left poses
    right = left @ B                                  # metric right poses
    s_true = 1 / 7.3                                  # model gauge = metric / 7.3
    c2w = np.concatenate([left, right])
    c2w = scale_camera_centers(c2w, s_true, origin_index=0)
    anchors = [ScaleAnchor(i, 4 + i, B) for i in range(4)]
    for method in ["adaptive", "huber_log", "median", "mean", "norm_ls"]:
        est = estimate_scale(c2w, anchors, method=method)
        assert est.valid
        assert abs(est.scale * s_true - 1.0) < 1e-6, (method, est.scale)
    metric = scale_camera_centers(c2w, est.scale, origin_index=0)
    rel = invert_poses(metric[0]) @ metric[1]
    rel_true = invert_poses(left[0]) @ left[1]
    assert np.allclose(rel, rel_true, atol=1e-6)


def test_outlier_anchor_rejected():
    rng = np.random.default_rng(1)
    B = _rig(0.5)
    left = _random_traj(3, rng)
    right = left @ B
    c2w = scale_camera_centers(np.concatenate([left, right]), 0.2, origin_index=0)
    # corrupt one right pose: wrong rotation -> must be rejected by the rotation gate
    c2w[5, :3, :3] = Rotation.from_rotvec([0, 1.0, 0]).as_matrix() @ c2w[5, :3, :3]
    c2w[5, :3, 3] += 1.0
    anchors = [ScaleAnchor(i, 3 + i, B) for i in range(3)]
    est = estimate_scale(c2w, anchors, method="huber_log")
    assert est.n_used == 2 and est.valid
    assert abs(est.scale * 0.2 - 1.0) < 1e-6


def test_no_anchor_invalid():
    est = estimate_scale(np.repeat(np.eye(4)[None], 2, 0), [], method="adaptive")
    assert not est.valid


def test_odom_anchor_direction_not_required():
    B = _rig(0.5)
    rng = np.random.default_rng(2)
    left = _random_traj(3, rng)
    T_pc = invert_poses(left[0]) @ left[1]
    c2w = scale_camera_centers(left, 0.5, origin_index=0)
    # odometry anchor with a slightly wrong direction but right length still counts
    est = estimate_scale(c2w, [ScaleAnchor(0, 1, T_pc, kind="odom", weight=0.5)], method="adaptive")
    assert est.valid and abs(est.scale * 0.5 - 1.0) < 1e-6
