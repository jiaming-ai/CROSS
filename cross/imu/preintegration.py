"""IMU preintegration between two image times (Forster et al., RSS 2015), in the body frame at the first time.

The accelerometer bias enters the preintegrated velocity and position exactly linearly (the rotation does not depend
on it): dv(b_a) = dv - J_v b_a, dp(b_a) = dp - J_p b_a.  The gyroscope bias enters to first order: dR(b_g + d) =
dR Exp(J_Rg d), dv(b_g + d) = dv + J_vg d, dp(b_g + d) = dp + J_pg d (d: an increment of the bias subtracted from the
gyro's rates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def right_jacobian(phi):
    """Right Jacobian of SO(3): Exp(phi + d) = Exp(phi) Exp(J_r(phi) d) to first order."""
    angle = float(np.linalg.norm(phi))
    K = skew(phi)
    if angle < 1e-6:
        return np.eye(3) - 0.5 * K
    return np.eye(3) - (1.0 - np.cos(angle)) / angle ** 2 * K + (angle - np.sin(angle)) / angle ** 3 * (K @ K)


def so3_exp(phi):
    angle = float(np.linalg.norm(phi))
    if angle < 1e-10:
        return np.eye(3) + skew(phi)
    k = skew(phi / angle)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


@dataclass
class Preintegrated:
    dt: float
    dR: np.ndarray          # rotation of the body at t1 in the body frame at t0
    dv: np.ndarray          # velocity change (body frame at t0, gravity excluded)
    dp: np.ndarray          # position change (body frame at t0, gravity and initial velocity excluded)
    J_v: np.ndarray         # d dv / d b_a
    J_p: np.ndarray         # d dp / d b_a
    cov: np.ndarray         # 9x9 covariance of (rotation, velocity, position) from the sensor noise
    n_samples: int
    accel_mean: np.ndarray  # mean specific force over the interval (body frame at t0), for gravity initialization
    J_Rg: np.ndarray = None  # d log(dR) / d b_g (right perturbation)
    J_vg: np.ndarray = None  # d dv / d b_g
    J_pg: np.ndarray = None  # d dp / d b_g


def preintegrate(samples: np.ndarray, t0: float, t1: float, gyro_noise: float, accel_noise: float,
                 full: bool = True) -> Preintegrated:
    """samples: (N, 7) rows t wx wy wz ax ay az covering [t0, t1] (a sample at or before t0 and one at or after t1;
    values are interpolated linearly in time at the interval ends).  gyro_noise / accel_noise: noise densities.
    full=False: only dR, dv, dp (no covariance, no bias Jacobians)."""
    out = Preintegrated(max(t1 - t0, 0.0), np.eye(3), np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)),
                        np.zeros((9, 9)), 0, np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)))
    if t1 <= t0 or len(samples) == 0:
        return out
    ts = samples[:, 0]
    inner = ts[(ts > t0) & (ts < t1)]
    grid = np.concatenate([[t0], inner, [t1]])
    vals = np.stack([np.interp(grid, ts, samples[:, k]) for k in range(1, 7)], axis=1)
    dR, dv, dp = np.eye(3), np.zeros(3), np.zeros(3)
    J_v, J_p = np.zeros((3, 3)), np.zeros((3, 3))
    J_Rg, J_vg, J_pg = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
    cov = np.zeros((9, 9))
    acc_sum = np.zeros(3)
    I3 = np.eye(3)
    for m in range(len(grid) - 1):
        h = grid[m + 1] - grid[m]
        if h <= 0:
            continue
        w = 0.5 * (vals[m, :3] + vals[m + 1, :3])
        a = 0.5 * (vals[m, 3:] + vals[m + 1, 3:])
        dR_inc = so3_exp(w * h)
        Ra = dR @ a
        if not full:
            dp = dp + dv * h + 0.5 * Ra * h * h
            dv = dv + Ra * h
            dR = dR @ dR_inc
            continue
        # gyro-bias Jacobians (Forster et al., T-RO 2017, appendix B), from the values before this step
        dRa_x = dR @ skew(a)
        J_pg = J_pg + J_vg * h - 0.5 * dRa_x @ J_Rg * h * h
        J_vg = J_vg - dRa_x @ J_Rg * h
        J_Rg = dR_inc.T @ J_Rg - right_jacobian(w * h) * h
        # noise propagation (rotation error in the frame of the current body, first order)
        A = np.eye(9)
        A[0:3, 0:3] = dR_inc.T
        A[3:6, 0:3] = -dR @ skew(a) * h
        A[6:9, 0:3] = -0.5 * dR @ skew(a) * h * h
        A[6:9, 3:6] = I3 * h
        Bg = np.zeros((9, 3))
        Bg[0:3] = I3 * h
        Ba = np.zeros((9, 3))
        Ba[3:6] = dR * h
        Ba[6:9] = 0.5 * dR * h * h
        cov = A @ cov @ A.T + (gyro_noise ** 2 / h) * (Bg @ Bg.T) + (accel_noise ** 2 / h) * (Ba @ Ba.T)
        dp = dp + dv * h + 0.5 * Ra * h * h
        J_p = J_p + J_v * h - 0.5 * dR * h * h
        dv = dv + Ra * h
        J_v = J_v - dR * h
        acc_sum += Ra * h
        dR = dR @ dR_inc
    out.dR, out.dv, out.dp, out.J_v, out.J_p, out.cov = dR, dv, dp, J_v, J_p, cov
    out.J_Rg, out.J_vg, out.J_pg = J_Rg, J_vg, J_pg
    out.n_samples = len(inner) + 2
    out.accel_mean = acc_sum / out.dt
    return out
