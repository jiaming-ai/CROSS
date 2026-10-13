"""Gravity from the ground plane (vision only): the up direction in the camera frame as the normal of the floor seen in a
depth image.  For odometry without an IMU (no gravity-aligned world), the floor's normal gives roll and pitch; the
odometry's rotation carries it between fits (GroundPlaneGravity).

Floor candidates are the points below the camera (along the expected down direction) within a range; RANSAC fits a
plane whose normal is within max_tilt_deg of the expected up direction; a fit is accepted with enough inliers.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def floor_normal(depth: np.ndarray, K: np.ndarray, up_hint=(0.0, -1.0, 0.0), max_range: float = 8.0,
                 min_height: float = 0.5, max_tilt_deg: float = 35.0, iters: int = 200, tol: float = 0.05,
                 min_inliers: int = 400, stride: int = 4, seed: int = 0) -> Tuple[Optional[np.ndarray], float]:
    """(up direction in the camera frame, inlier fraction of the floor candidates) or (None, 0).  up_hint: the expected
    up direction in the camera frame (OpenCV camera: y points down, so up ~ -y for a level camera)."""
    d = depth[::stride, ::stride].astype(np.float64)
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride
    v = v * stride
    ok = np.isfinite(d) & (d > 0.2) & (d < max_range)
    if ok.sum() < min_inliers:
        return None, 0.0
    z = d[ok]
    x = (u[ok] - K[0, 2]) * z / K[0, 0]
    y = (v[ok] - K[1, 2]) * z / K[1, 1]
    P = np.stack([x, y, z], 1)
    up = np.asarray(up_hint, float)
    up /= np.linalg.norm(up)
    below = P @ up < -min_height                 # points at least min_height below the camera
    P = P[below]
    if len(P) < min_inliers:
        return None, 0.0
    rng = np.random.default_rng(seed)
    best_n, best_d, best_cnt = None, 0.0, 0
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n @ up < 0:
            n = -n
        if n @ up < cos_tilt:
            continue
        dist = P @ n - a @ n
        cnt = int((np.abs(dist) < tol).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, float(a @ n), cnt
    if best_n is None or best_cnt < min_inliers:
        return None, 0.0
    inl = P[np.abs(P @ best_n - best_d) < tol]          # refine: least-squares normal of the inliers
    n = np.linalg.svd(inl - inl.mean(0), full_matrices=False)[2][-1]
    best_n = n if n @ up > 0 else -n
    return best_n, best_cnt / max(len(P), 1)


class GroundPlaneGravity:
    """Per-frame up direction from floor fits every `every` frames, carried by the odometry's rotation in between (and
    blended with each new fit)."""

    def __init__(self, every: int = 20, blend: float = 0.5, **fit_kw):
        self.every, self.blend, self.fit_kw = int(every), float(blend), fit_kw
        self.up_c: Optional[np.ndarray] = None       # current up direction in the current camera frame
        self.n = 0

    def step(self, R_delta: Optional[np.ndarray], depth_fn, K) -> Optional[np.ndarray]:
        """R_delta: rotation of the odometry increment (T_cur = T_prev @ delta); depth_fn(): the frame's depth (called
        only on fit frames).  Returns the up direction in the current camera frame (None before the first fit)."""
        if self.up_c is not None and R_delta is not None:
            self.up_c = np.asarray(R_delta, float).T @ self.up_c
        if self.n % self.every == 0:
            hint = self.up_c if self.up_c is not None else np.array([0.0, -1.0, 0.0])
            n, frac = floor_normal(depth_fn(), K, up_hint=hint, **self.fit_kw)
            if n is not None:
                if self.up_c is None:
                    self.up_c = n
                else:
                    m = (1 - self.blend) * self.up_c + self.blend * n
                    self.up_c = m / np.linalg.norm(m)
        self.n += 1
        return None if self.up_c is None else self.up_c.copy()
