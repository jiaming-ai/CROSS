"""The inertial scale filter recovers the metric scale of an up-to-scale visual trajectory from a simulated IMU."""

import numpy as np
from scipy.spatial.transform import Rotation

from cross.imu import ImuConfig, InertialScaleFilter, preintegrate
from cross.imu.simulate import simulate_imu


def robot_path(seconds=60.0, fps=10.0, seed=0, speed=0.5):
    """Camera poses (forward-looking: z forward, y down) of a ground robot that drives, turns and stops."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    t = np.arange(n) / fps
    v = speed * (1 + 0.3 * np.sin(2 * np.pi * t / 17.0)) * np.clip(t / 2.0, 0, 1)
    v[(t > 30) & (t < 33)] = 0.0                                     # a stop
    yaw_rate = 0.4 * np.sin(2 * np.pi * t / 23.0 + rng.uniform(0, 6))
    yaw = np.cumsum(yaw_rate) / fps
    pos = np.cumsum(np.stack([v * np.cos(yaw), v * np.sin(yaw), np.zeros(n)], 1), 0) / fps
    # body x forward, y left, z up (world z up); camera = body rotated to z forward, y down
    R_bc = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    poses = np.tile(np.eye(4), (n, 1, 1))
    for k in range(n):
        poses[k, :3, :3] = Rotation.from_euler("z", yaw[k]).as_matrix() @ R_bc
        poses[k, :3, 3] = pos[k] + np.array([0, 0, 0.5])
    return poses


def run_filter(poses, scale, fps=10.0, seed=1, drift_per_s=0.0, noise=0.001):
    rng = np.random.default_rng(seed)
    t, w, a = simulate_imu(poses, fps, seed)
    samples = np.concatenate([t[:, None], w, a], 1)
    # visual odometry: an arbitrary world frame, positions in units of `scale` metres (with drift and noise)
    R0 = Rotation.from_rotvec(rng.normal(0, 1, 3)).as_matrix()
    n = len(poses)
    unit = np.zeros((n, 3))
    for k in range(1, n):
        s_k = scale * (1 + drift_per_s * k / fps)
        unit[k] = unit[k - 1] + R0 @ (poses[k, :3, 3] - poses[k - 1, :3, 3]) / s_k + rng.normal(0, noise / scale, 3)
    R_vo = np.einsum("ij,njk->nik", R0, poses[:, :3, :3])
    f = InertialScaleFilter(ImuConfig(enabled=True), np.eye(4), 1.1e-3, 1.2e-2)
    history = []
    for k in range(1, n):
        t0, t1 = (k - 1) / fps, k / fps
        lo = max(np.searchsorted(t, t0, "right") - 1, 0)
        hi = min(np.searchsorted(t, t1, "left") + 1, len(t))
        pre = preintegrate(samples[lo:hi], t0, t1, 1.1e-3, 1.2e-2)
        if not f.started:
            f.start(R_vo[k - 1], pre.accel_mean)
        f.step(pre, R_vo[k - 1], R_vo[k], unit[k] - unit[k - 1], depth_units=2.0 / scale)
        s_true = scale * (1 + drift_per_s * k / fps)
        history.append((f.scale / s_true - 1, f.log_std, f.initialized))
    return np.asarray(history)


def test_scale_converges():
    poses = robot_path()
    h = run_filter(poses, scale=3.7)
    # the scale is known (log std < 0.1) within 20 s, and the error stays within the filter's 3-sigma from then on
    assert h[200:, 2].all()
    err = np.log1p(h[200:, 0])
    assert (np.abs(err) < 3 * h[200:, 1]).all(), np.abs(err / h[200:, 1]).max()
    assert abs(err[-1]) < 0.1, err[-1]


def test_scale_tracks_drift():
    poses = robot_path(seed=2)
    h = run_filter(poses, scale=0.4, drift_per_s=0.003)        # 18 % scale drift over the minute (a slow robot with
    # a gravity drift allowance: the scale lags the drift by up to ~25 % between turns)
    err = np.log1p(h[-200:, 0])
    assert (np.abs(err) < 4 * h[-200:, 1]).all() and np.abs(err).max() < 0.3, np.abs(err).max()
