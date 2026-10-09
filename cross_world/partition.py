"""Chunks of a large map, trained independently (in parallel, with bounded memory) and merged.

The keyframe camera centres are projected on the ground plane (the map's vertical from the cameras) and the region
they cover (their bounding rectangle grown by `margin`) is split recursively at the median of the longer side until
every cell holds at most `max_views` keyframes: balanced cells that tile the region.  A chunk trains on

  * the keyframes inside its cell grown by `margin` (context across the seam), and
  * any other keyframe that sees enough of the cell (share `min_visible` of its depth points falls in it), so that
    a surface is trained with every view that sees it, not only by the cameras that happen to stand in its cell
    (VastGaussian's visibility-based selection),

and keeps the Gaussians inside its own cell ("near" layer) plus those beyond the whole region ("far" layer: sky and
distant scenery that only this chunk's cameras explain).  A viewer draws every near layer and the far layer of the
chunk whose cell holds the camera.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Chunk:
    index: int
    lo: np.ndarray                    # cell lower corner, ground-plane coordinates (2,)
    hi: np.ndarray                    # cell upper corner
    core_ids: List[int]               # keyframes whose camera centre is in the cell
    train_ids: List[int] = field(default_factory=list)


@dataclass
class Partition:
    origin: np.ndarray                # a point of the map (3,)
    axes: np.ndarray                  # (2, 3): ground-plane axes; ground coordinates = axes @ (x - origin)
    up: np.ndarray                    # (3,)
    region_lo: np.ndarray             # the covered region (2,)
    region_hi: np.ndarray
    chunks: List[Chunk]
    margin: float

    def ground(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts) - self.origin) @ self.axes.T

    def cell_of(self, pts: np.ndarray) -> np.ndarray:
        """Index of the chunk whose cell holds each point; -1 outside the region."""
        g = self.ground(np.atleast_2d(pts))
        out = np.full(len(g), -1, np.int64)
        for c in self.chunks:
            inside = np.all((g >= c.lo) & (g < c.hi), axis=1)
            out[inside & (out < 0)] = c.index
        return out

    def nearest_cell(self, pts: np.ndarray) -> np.ndarray:
        """Chunk of each point: its cell, or the nearest cell for points outside the region (a camera there)."""
        g = self.ground(np.atleast_2d(pts))
        d = np.stack([np.linalg.norm(np.maximum(np.maximum(c.lo - g, g - c.hi), 0), axis=1) for c in self.chunks], 1)
        return np.argmin(d, axis=1)

    def in_region(self, pts: np.ndarray) -> np.ndarray:
        g = self.ground(np.atleast_2d(pts))
        return np.all((g >= self.region_lo) & (g < self.region_hi), axis=1)

    def to_dict(self) -> dict:
        return {"origin": self.origin.tolist(), "axes": self.axes.tolist(), "up": self.up.tolist(),
                "region_lo": self.region_lo.tolist(), "region_hi": self.region_hi.tolist(), "margin": self.margin,
                "chunks": [{"index": c.index, "lo": c.lo.tolist(), "hi": c.hi.tolist(), "core_ids": c.core_ids,
                            "train_ids": c.train_ids} for c in self.chunks]}

    @staticmethod
    def from_dict(d: dict) -> "Partition":
        return Partition(np.asarray(d["origin"]), np.asarray(d["axes"]), np.asarray(d["up"]), np.asarray(d["region_lo"]),
                         np.asarray(d["region_hi"]), [Chunk(c["index"], np.asarray(c["lo"]), np.asarray(c["hi"]),
                                                            list(c["core_ids"]), list(c["train_ids"])) for c in d["chunks"]],
                         float(d["margin"]))


def ground_axes(centers: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Two orthonormal axes perpendicular to `up`, the first along the principal direction of the camera centres."""
    c = centers - centers.mean(0)
    c = c - np.outer(c @ up, up)
    if len(c) >= 2 and np.linalg.norm(c) > 1e-9:
        e1 = np.linalg.svd(c, full_matrices=False)[2][0]
    else:
        e1 = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    e1 = e1 - (e1 @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return np.stack([e1, e2])


def split_cells(g: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_views: int, idx: Optional[np.ndarray] = None):
    """Recursive median split of the ground points `g` inside [lo, hi) until a cell holds <= max_views points."""
    idx = np.arange(len(g)) if idx is None else idx
    if len(idx) <= max_views:
        return [(lo, hi, idx)]
    ax = int(np.argmax(hi - lo))
    v = g[idx, ax]
    m = float(np.median(v))
    if not (v.min() < m <= v.max()):            # degenerate (repeated values): split the extent instead
        m = float((lo[ax] + hi[ax]) / 2)
    left = idx[v < m]
    right = idx[v >= m]
    if len(left) == 0 or len(right) == 0:
        return [(lo, hi, idx)]
    hi_l, lo_r = hi.copy(), lo.copy()
    hi_l[ax] = m
    lo_r[ax] = m
    return split_cells(g, lo, hi_l, max_views, left) + split_cells(g, lo_r, hi, max_views, right)


def make_partition(ids: List[int], centers: np.ndarray, up: np.ndarray, max_views: int, margin: float) -> Partition:
    axes = ground_axes(centers, up)
    origin = centers.mean(0)
    g = (centers - origin) @ axes.T
    lo, hi = g.min(0) - margin, g.max(0) + margin
    cells = split_cells(g, lo, hi, max_views)
    chunks = [Chunk(i, c_lo, c_hi, [int(ids[j]) for j in idx]) for i, (c_lo, c_hi, idx) in enumerate(cells)]
    return Partition(origin, axes, up, lo, hi, chunks, margin)


def assign_views(part: Partition, ids: List[int], centers: np.ndarray, samples: Dict[int, np.ndarray],
                 min_visible: float = 0.1, min_points: int = 200) -> None:
    """Training views of each chunk (see the module doc).  `samples[id]`: a subsample of the view's 3-D depth points
    (world frame) to measure what the view sees."""
    g_c = part.ground(centers)
    for c in part.chunks:
        lo, hi = c.lo - part.margin, c.hi + part.margin
        near = np.all((g_c >= lo) & (g_c < hi), axis=1)
        sel = {int(ids[k]) for k in np.nonzero(near)[0]}
        for k, vid in enumerate(ids):
            if vid in sel or vid not in samples or len(samples[vid]) == 0:
                continue
            gp = part.ground(samples[vid])
            inside = np.all((gp >= c.lo) & (gp < c.hi), axis=1)
            if inside.mean() >= min_visible and inside.sum() >= min_points:
                sel.add(int(vid))
        c.train_ids = sorted(sel | set(c.core_ids))
