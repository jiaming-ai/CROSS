"""Metric depth of each view: what initialises the Gaussians and supervises their geometry.

Sources, by availability:
  sensor   the RGB-D keyframe's depth (stored or from the source folder), clipped to the sensor's reliable range
  sgbm     semi-global matching of the rectified stereo pair, with a left-right consistency check (stereo maps)
  none     no depth (monocular maps without a learned-depth model): the Gaussians start from the other views' depth
           or from random points, and only the photometric loss trains them

`consistent_mask` keeps the depth pixels that at least one neighbouring view confirms (projection into the neighbour
lands within a relative depth tolerance of its own depth), which removes most stereo mismatches and sensor flying
pixels before they become Gaussians.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch

from cross_world.map_views import MapViews, View


class StereoDepth:
    """SGBM depth of a rectified pair with a left-right check (the 'accurate' profile of the CROSS stereo loader)."""

    def __init__(self, width: int):
        num_disp = max(96, min(256, (width // 3 // 16) * 16))
        bs = 5
        kw = dict(minDisparity=0, numDisparities=num_disp, blockSize=bs, P1=8 * bs * bs, P2=32 * bs * bs,
                  disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, preFilterCap=31,
                  mode=cv2.STEREO_SGBM_MODE_HH)
        self.left = cv2.StereoSGBM_create(**kw)
        self.num_disp = num_disp

    def __call__(self, left_rgb: np.ndarray, right_rgb: np.ndarray, fx: float, baseline: float) -> np.ndarray:
        l = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
        r = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
        dl = self.left.compute(l, r).astype(np.float32) / 16.0
        # right disparity from the mirrored pair, for the left-right consistency check
        dr = self.left.compute(np.ascontiguousarray(r[:, ::-1]), np.ascontiguousarray(l[:, ::-1])).astype(np.float32)[:, ::-1] / 16.0
        h, w = dl.shape
        xs = np.arange(w)[None, :].repeat(h, 0)
        xr = np.clip(np.round(xs - dl).astype(np.int64), 0, w - 1)
        back = np.take_along_axis(dr, xr, axis=1)
        ok = (dl > 0.5) & (np.abs(dl - back) <= 1.0) & (xs >= self.num_disp)   # left border: no full search range
        depth = np.zeros_like(dl)
        depth[ok] = fx * baseline / dl[ok]
        depth[~np.isfinite(depth)] = 0.0
        return depth


def view_depth(view: View, mv: MapViews, source: str = "auto", max_depth: Optional[float] = None,
               stereo: Optional[StereoDepth] = None) -> Optional[np.ndarray]:
    """Depth (H, W) float32 in metres of one view (0 = unknown), or None when the view has no depth source."""
    d = None
    if source in ("auto", "sensor") and view.has_depth:
        d = view.depth().astype(np.float32)
    elif source in ("auto", "sgbm") and view.has_right and mv.T_right_in_left is not None:
        baseline = float(np.linalg.norm(mv.T_right_in_left[:3, 3]))
        st = stereo or StereoDepth(view.width)
        d = st(view.image(), view.right(), float(view.K[0, 0]), baseline)
    if d is None:
        return None
    d[~np.isfinite(d)] = 0
    if max_depth:
        d[d > max_depth] = 0
    return d


@torch.no_grad()
def consistent_mask(depths: List[torch.Tensor], Ks: torch.Tensor, T_wc: torch.Tensor, neighbours: List[List[int]],
                    rel_tol: float = 0.03, min_agree: int = 1) -> List[torch.Tensor]:
    """For each view i: pixels whose depth at least `min_agree` neighbouring views confirm.  A neighbour confirms a
    pixel when the pixel's 3-D point projects inside it with a depth within `rel_tol` (relative) of the neighbour's
    depth there.  A neighbour votes only when it sees a surface there at or in front of the point's depth: one that
    sees nothing (zero depth, outside the image) or a nearer surface (the point is occluded there) does not.  Pixels
    with no voting neighbour at all are kept (nothing contradicts them)."""
    out = []
    dev = depths[0].device
    for i, d in enumerate(depths):
        h, w = d.shape
        valid = d > 0
        if not neighbours[i]:
            out.append(valid)
            continue
        ys, xs = torch.nonzero(valid, as_tuple=True)
        z = d[ys, xs]
        Kinv = torch.linalg.inv(Ks[i])
        pix = torch.stack([xs.float() + 0.5, ys.float() + 0.5, torch.ones_like(z)], 0)
        pc = (Kinv @ pix) * z[None]
        pw = T_wc[i, :3, :3] @ pc + T_wc[i, :3, 3:4]
        agree = torch.zeros_like(z, dtype=torch.int32)
        votes = torch.zeros_like(z, dtype=torch.int32)
        for j in neighbours[i]:
            T_cw = torch.linalg.inv(T_wc[j])
            q = T_cw[:3, :3] @ pw + T_cw[:3, 3:4]
            zj = q[2]
            uv = (Ks[j] @ q)[:2] / zj.clamp(min=1e-6)
            hj, wj = depths[j].shape
            u, v = uv[0].floor().long(), uv[1].floor().long()
            inside = (zj > 1e-3) & (u >= 0) & (u < wj) & (v >= 0) & (v < hj)
            dj = torch.zeros_like(z)
            dj[inside] = depths[j][v[inside], u[inside]]
            voted = inside & (dj > 0) & (zj <= (1 + rel_tol) * dj)     # occluded in j: no vote
            votes += voted.int()
            agree += (voted & ((zj - dj).abs() <= rel_tol * dj)).int()
        keep = (agree >= min_agree) | (votes == 0)
        m = torch.zeros_like(valid)
        m[ys[keep], xs[keep]] = True
        out.append(m)
    return out
