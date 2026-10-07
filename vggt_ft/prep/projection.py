"""Depth maps from 3D points (a static scene cloud or tracked points) projected into a camera, for datasets without
depth images (HOI4D mirror, Dynamic Replica tracking subset)."""
from __future__ import annotations

import cv2
import numpy as np
import torch


def resize_for_cache(rgb: np.ndarray, K: np.ndarray, max_side: int = 768):
    """The image and intrinsics at the cache's stored resolution (SceneWriter's resize), so that sparse depth is
    rendered at that resolution instead of being resized by nearest neighbour (which drops most points)."""
    H, W = rgb.shape[:2]
    K = np.asarray(K, np.float64).copy()
    s = min(1.0, max_side / max(H, W))
    if s < 1.0:
        nw, nh = int(round(W * s)), int(round(H * s))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        sx, sy = nw / W, nh / H
        K[0, 0], K[1, 1] = K[0, 0] * sx, K[1, 1] * sy
        K[0, 2], K[1, 2] = (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5
    return rgb, K


def zbuffer(points_world: np.ndarray, E: np.ndarray, K: np.ndarray, hw, near: float = 0.05) -> np.ndarray:
    """Nearest z per pixel of points (N,3) seen by camera E (4x4 camera-from-world, OpenCV), 0 where empty."""
    H, W = hw
    P = torch.from_numpy(np.asarray(points_world, np.float32))
    R = torch.from_numpy(np.asarray(E[:3, :3], np.float32))
    t = torch.from_numpy(np.asarray(E[:3, 3], np.float32))
    X = P @ R.T + t
    z = X[:, 2]
    keep = z > near
    X, z = X[keep], z[keep]
    u = torch.round(X[:, 0] / z * float(K[0, 0]) + float(K[0, 2])).long()
    v = torch.round(X[:, 1] / z * float(K[1, 1]) + float(K[1, 2])).long()
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = v[ok] * W + u[ok]
    d = torch.full((H * W,), float("inf"))
    d.scatter_reduce_(0, idx, z[ok], reduce="amin")
    d[torch.isinf(d)] = 0
    return d.view(H, W).numpy()


def drop_see_through(depth: np.ndarray, win: int = 7, rel: float = 0.08) -> np.ndarray:
    """Invalidate points that show through gaps of a nearer surface: deeper than the window's nearest valid depth by
    more than `rel` (also thins the far side of true depth edges by up to win/2 px)."""
    d = depth.copy()
    big = np.where(d > 0, d, np.float32(1e6)).astype(np.float32)
    near = cv2.erode(big, np.ones((win, win), np.uint8))
    d[(d > 0) & (d > near * (1 + rel))] = 0
    return d


def edge_alignment(rgb: np.ndarray, depth: np.ndarray, tol_px: int = 2, rel_jump: float = 0.1) -> float:
    """Share of depth-discontinuity pixels within tol_px of an RGB (Canny) edge: high when the projection lines up with
    the image, low for a wrong pose / frame index (a check, not a filter on its own)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 50, 150) > 0
    d = depth.astype(np.float32)
    v = d > 0
    big = np.where(v, d, 0)
    dx = np.zeros_like(d, bool)
    dy = np.zeros_like(d, bool)
    both = v[:, 1:] & v[:, :-1]
    dx[:, 1:] = both & (np.abs(big[:, 1:] - big[:, :-1]) > rel_jump * np.minimum(big[:, 1:], big[:, :-1]))
    both = v[1:] & v[:-1]
    dy[1:] = both & (np.abs(big[1:] - big[:-1]) > rel_jump * np.minimum(big[1:], big[:-1]))
    de = dx | dy
    if de.sum() < 50:
        return float("nan")
    near_e = cv2.dilate(e.astype(np.uint8), np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8)) > 0
    return float(near_e[de].mean())
