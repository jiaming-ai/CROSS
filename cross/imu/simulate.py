"""A simulated IMU along a camera trajectory (SimChange, tests)."""

from __future__ import annotations

import numpy as np

# BMI055 (the IMU of the D435i and the T265): Kalibr values of ROVER's calib_d435i.yaml
BMI055 = dict(gyro_noise_density=0.0010662953470339657, accel_noise_density=0.012204288735250874,
              gyro_random_walk=5.776283991470186e-05, accel_random_walk=0.0005354903138724063)
GRAVITY = 9.81


def simulate_imu(poses: np.ndarray, fps: float, seed: int, rate: float = 200.0, noise: dict = BMI055,
                 gyro_bias_std: float = 0.002, accel_bias_std: float = 0.05, down=None):
    """IMU (frame = the camera frame) along camera-to-world poses sampled at `fps`, interpolated by C2 splines.

    down: world direction of gravity (default: the camera's y axis averaged over the run, i.e. a ground robot with a
    forward-looking camera).  Returns t (from 0 = the first pose), gyro (rad/s), specific force (m/s^2)."""
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, RotationSpline
    n = len(poses)
    tf = np.arange(n) / fps
    pos = CubicSpline(tf, poses[:, :3, 3], bc_type="natural")
    rot = RotationSpline(tf, Rotation.from_matrix(poses[:, :3, :3]))
    t = np.arange(0.0, tf[-1] + 1e-9, 1.0 / rate)
    if down is None:
        down = poses[:, :3, 1].mean(0)
    g = GRAVITY * np.asarray(down, dtype=np.float64) / np.linalg.norm(down)
    R = rot(t).as_matrix()
    h = 1e-3
    ta, tb = np.clip(t - h, 0, tf[-1]), np.clip(t + h, 0, tf[-1])
    rel = rot(ta).inv() * rot(tb)
    w = rel.as_rotvec() / (tb - ta)[:, None]
    f = np.einsum("nji,nj->ni", R, pos(t, 2) - g)          # specific force in the IMU frame: R^T (a - g)
    rng = np.random.default_rng(seed)
    dt = 1.0 / rate
    m = len(t)
    b_g = rng.normal(0, gyro_bias_std, 3) + np.cumsum(rng.normal(0, noise["gyro_random_walk"] * np.sqrt(dt), (m, 3)), 0)
    b_a = rng.normal(0, accel_bias_std, 3) + np.cumsum(rng.normal(0, noise["accel_random_walk"] * np.sqrt(dt), (m, 3)), 0)
    w_m = w + b_g + rng.normal(0, noise["gyro_noise_density"] / np.sqrt(dt), (m, 3))
    f_m = f + b_a + rng.normal(0, noise["accel_noise_density"] / np.sqrt(dt), (m, 3))
    return t, w_m, f_m
