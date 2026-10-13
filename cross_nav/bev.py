"""The BEV frame of a camera and the crop of a world BEV into it.

BEV frame of a camera (the current frame of a VGGT pass, frame 0): origin at the camera centre, z = up (gravity),
y = forward (the optical axis projected onto the horizontal plane), x = right (y x z), so it depends only on the
camera's gravity direction ("up" in camera coordinates), which an IMU measures or the gravity head predicts.  Heights
in the BEV frame are relative to the camera.  The grid has N x N cells of `res` metres with the camera at its centre;
row 0 is the far forward edge, column 0 the left edge:

    x = (col + 0.5 - N / 2) * res,   y = (N / 2 - row - 0.5) * res

Cell classes: 0 = unknown (no label), 1 = free (flat ground the robot can stand on), 2 = occupied (anything between
`h_lo` and `h_hi` above the local ground, whatever its height, or ground that is not flat).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

UNKNOWN, FREE, OCC = 0, 1, 2


@dataclass(frozen=True)
class BEVSpec:
    size: int = 400          # cells per side
    res: float = 0.05        # metres per cell

    @property
    def extent(self) -> float:
        return self.size * self.res

    def cell_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) of every cell centre in the BEV frame, each (N, N)."""
        i = np.arange(self.size, dtype=np.float64) + 0.5 - self.size / 2
        x = np.broadcast_to(i[None, :] * self.res, (self.size, self.size))
        y = np.broadcast_to(-i[:, None] * self.res, (self.size, self.size))
        return x, y


def bev_axes_np(up_cam: np.ndarray) -> np.ndarray:
    """R (3,3): BEV-from-camera rotation (rows = right, forward, up in camera coordinates) for a unit up vector in
    camera coordinates (OpenCV axes: x right, y down, z forward)."""
    up = up_cam / np.linalg.norm(up_cam)
    f = np.array([0.0, 0.0, 1.0]) - up[2] * up
    if np.linalg.norm(f) < 0.2:                 # optical axis nearly vertical: the image top points forward
        f = np.array([0.0, -1.0, 0.0]) - (-up[1]) * up
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    return np.stack([r, f, up])


def bev_axes(up_cam: torch.Tensor) -> torch.Tensor:
    """Batched bev_axes_np: up_cam (..., 3) -> (..., 3, 3)."""
    up = torch.nn.functional.normalize(up_cam, dim=-1)
    ez = torch.zeros_like(up)
    ez[..., 2] = 1
    ey = torch.zeros_like(up)
    ey[..., 1] = -1
    f1 = ez - up[..., 2:3] * up
    f2 = ey + up[..., 1:2] * up
    f = torch.where(f1.norm(dim=-1, keepdim=True) < 0.2, f2, f1)
    f = torch.nn.functional.normalize(f, dim=-1)
    r = torch.cross(f, up, dim=-1)
    return torch.stack([r, f, up], -2)


def camera_bev_pose(E: np.ndarray, R_g: np.ndarray):
    """Camera (camera-from-world E) in a gravity-aligned world frame (x_g = R_g x_world, z up).
    Returns centre p (3,), right (3,), forward (3,) in that frame and up_cam (3,) = gravity up in camera coordinates."""
    R_cw = E[:3, :3]
    c_w = -R_cw.T @ E[:3, 3]
    p = R_g @ c_w
    up_cam = R_cw @ R_g.T @ np.array([0.0, 0.0, 1.0])
    A = bev_axes_np(up_cam)                     # rows in camera coordinates
    M = R_g @ R_cw.T                            # camera -> gravity frame
    right, fwd = M @ A[0], M @ A[1]
    return p, right, fwd, up_cam


def crop_world(world: np.ndarray, origin: np.ndarray, res_w: float, p: np.ndarray, right: np.ndarray,
               fwd: np.ndarray, spec: BEVSpec, fill: int = UNKNOWN) -> np.ndarray:
    """Resample a world BEV (H_w, W_w), cell [iy, ix] centred at origin + ((ix, iy) + 0.5) * res_w in the gravity frame,
    into the BEV frame of a camera at p with horizontal axes right / fwd: (N, N), nearest neighbour."""
    x, y = spec.cell_xy()
    r2, f2 = right[:2] / max(np.linalg.norm(right[:2]), 1e-9), fwd[:2] / max(np.linalg.norm(fwd[:2]), 1e-9)
    wx = p[0] + x * r2[0] + y * f2[0]
    wy = p[1] + x * r2[1] + y * f2[1]
    u = ((wx - origin[0]) / res_w - 0.5).astype(np.float32)
    v = ((wy - origin[1]) / res_w - 0.5).astype(np.float32)
    return cv2.remap(world, u, v, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def points_to_cells(pts_bev: np.ndarray, spec: BEVSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BEV-frame points (M, 3) -> (row, col, inside mask)."""
    col = np.floor(pts_bev[:, 0] / spec.res + spec.size / 2).astype(np.int64)
    row = np.floor(spec.size / 2 - pts_bev[:, 1] / spec.res).astype(np.int64)
    ok = (col >= 0) & (col < spec.size) & (row >= 0) & (row < spec.size)
    return row, col, ok
