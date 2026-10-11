"""Planning on BEV grids and its evaluation against the ground truth.

A planner gets a binary map (occupied / free) from a method (thresholded head prediction, geometric BEV), inflates
the obstacles by the robot radius and plans with A* (8-connected) from the start to a goal.  The plan is judged on
the ground-truth label:

    success   a path was found, no cell of it lies within `check_radius` of a GT-occupied cell, and it is at most
              `max_detour` x the GT shortest path (planned on the GT free space with the same inflation)
    no_path   the method's map has no path (start or goal blocked, or disconnected)
    collision the path passes within check_radius of a GT obstacle
    detour    a collision-free path more than max_detour x longer than the GT one
    SPL       success x gt_length / max(gt_length, path_length)

Goals: random GT-free cells at `dist_range` metres from the start that the GT free space (inflated) connects to it.
"""
from __future__ import annotations

import heapq
import math

import cv2
import numpy as np

from .bev import FREE, OCC

try:
    from numba import njit
except ImportError:                      # pure-Python fallback (slow)
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

SQ2 = math.sqrt(2.0)
NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]


def disk(r: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def inflate(occ: np.ndarray, r: int) -> np.ndarray:
    return cv2.dilate(occ.astype(np.uint8), disk(r)) > 0 if r > 0 else occ.astype(bool)


def risk_cost(prob, res, robot_radius=0.20, weight=4.0, uncertainty=None, uncertainty_weight=0.0):
    """Soft footprint risk: 1 + weight * -log(1 - max_footprint P(occ)).

    Max pooling is a conservative local score, not a calibrated probability of
    footprint/path collision (neighbouring cells are strongly correlated).
    Optional predicted decision error adds an independent exploration penalty.
    All cells remain traversable; use a separate obstacle map for hard constraints.
    """
    p = np.asarray(prob, np.float32)
    if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("prob must be a finite 2D probability map")
    if res <= 0 or robot_radius < 0 or weight < 0 or uncertainty_weight < 0:
        raise ValueError("invalid resolution, radius or risk weight")
    r = int(round(robot_radius / res))
    pooled = cv2.dilate(p, disk(r)) if r else p
    cost = 1.0 + weight * -np.log1p(-np.minimum(pooled, 1 - 1e-6))
    if uncertainty is not None:
        u = np.asarray(uncertainty, np.float32)
        if u.shape != p.shape or not np.isfinite(u).all() or (u < 0).any() or (u > 1).any():
            raise ValueError("uncertainty must be a matching probability map")
        cost += uncertainty_weight * (cv2.dilate(u, disk(r)) if r else u)
    return cost


def astar(ok: np.ndarray, start, goal):
    """Shortest 8-connected path over cells where ok is True (list of (row, col)) or None."""
    H, W = ok.shape
    sr, sc = start
    gr, gc = goal
    if not (ok[sr, sc] and ok[gr, gc]):
        return None
    g = np.full((H, W), np.inf)
    g[sr, sc] = 0.0
    parent = -np.ones((H, W), np.int64)
    closed = np.zeros((H, W), bool)
    h0 = math.hypot(gr - sr, gc - sc)
    heap = [(h0, 0.0, sr, sc)]
    while heap:
        f, gc_, r, c = heapq.heappop(heap)
        if closed[r, c]:
            continue
        closed[r, c] = True
        if r == gr and c == gc:
            path = [(r, c)]
            k = parent[r, c]
            while k >= 0:
                r, c = divmod(int(k), W)
                path.append((r, c))
                k = parent[r, c]
            return path[::-1]
        for dr, dc, w in NB:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and ok[nr, nc] and not closed[nr, nc]:
                ng = gc_ + w
                if ng < g[nr, nc]:
                    g[nr, nc] = ng
                    parent[nr, nc] = r * W + c
                    heapq.heappush(heap, (ng + math.hypot(gr - nr, gc - nc), ng, nr, nc))
    return None


@njit(cache=True)
def _dijkstra(ok, sr, sc, cost):
    """8-connected shortest-path field from (sr, sc) over ok cells: distance (cells) and parent (flat index, -1)."""
    H, W = ok.shape
    dist = np.full((H, W), np.inf)
    parent = -np.ones((H, W), np.int64)
    dist[sr, sc] = 0.0
    heap = [(0.0, sr * W + sc)]
    while len(heap) > 0:
        d, k = heapq.heappop(heap)
        r, c = k // W, k % W
        if d > dist[r, c]:
            continue
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W or not ok[nr, nc]:
                    continue
                nd = d + (1.4142135623730951 if dr != 0 and dc != 0 else 1.0) * (cost[r, c] + cost[nr, nc]) * 0.5
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    parent[nr, nc] = k
                    heapq.heappush(heap, (nd, nr * W + nc))
    return dist, parent


class Field:
    """Shortest paths from one start to any goal (one Dijkstra run)."""

    def __init__(self, ok, start, cost=None):
        if cost is None:
            cost = np.ones(ok.shape, np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        if cost.shape != ok.shape or not np.isfinite(cost).all() or (cost < 1).any():
            raise ValueError("cost must match map, be finite, and >= 1")
        self.ok = ok
        self.W = ok.shape[1]
        self.valid = bool(ok[start])
        if self.valid:
            self.dist, self.parent = _dijkstra(ok, int(start[0]), int(start[1]), cost)

    def path(self, goal):
        if not self.valid or not np.isfinite(self.dist[goal]):
            return None
        out = [goal]
        k = self.parent[goal]
        while k >= 0:
            out.append((int(k // self.W), int(k % self.W)))
            k = self.parent[out[-1]]
        return out[::-1]


def path_length(path, res):
    if not path or len(path) < 2:
        return 0.0
    p = np.asarray(path, np.float64)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() * res)


def sample_goals(gt_ok: np.ndarray, start, res, n, dist_range, rng):
    """n goal cells: GT-traversable, connected to the start, at dist_range metres (straight line)."""
    num, lab = cv2.connectedComponents(gt_ok.astype(np.uint8), connectivity=8)
    if lab[start] == 0:
        return []
    comp = lab == lab[start]
    rr, cc = np.nonzero(comp)
    d = np.hypot(rr - start[0], cc - start[1]) * res
    sel = np.flatnonzero((d >= dist_range[0]) & (d <= dist_range[1]))
    if len(sel) == 0:
        return []
    pick = rng.choice(sel, min(n, len(sel)), replace=False)
    return [(int(rr[k]), int(cc[k])) for k in pick]


def snap(ok: np.ndarray, cell, max_cells):
    """Nearest cell of ok within max_cells (or None)."""
    r0, c0 = cell
    H, W = ok.shape
    best, bd = None, 1e9
    for r in range(max(0, r0 - max_cells), min(H, r0 + max_cells + 1)):
        for c in range(max(0, c0 - max_cells), min(W, c0 + max_cells + 1)):
            if ok[r, c]:
                d = (r - r0) ** 2 + (c - c0) ** 2
                if d < bd:
                    best, bd = (r, c), d
    return best


def evaluate_planning(gt: np.ndarray, maps: dict, res: float, start=None, n_goals=5, dist_range=(2.0, 8.0),
                      robot_radius=0.20, check_radius=0.10, max_detour=3.0, seed=0, starts=None, costs=None):
    """gt: label (0 unknown, 1 free, 2 occupied); maps: {method: occupied bool (H, W)} (unknown cells of a method's
    map must already be resolved to occupied or free).  Returns one record per goal: start, goal, gt_path, gt_len and
    per method status / spl / len / path. costs optionally maps method names to finite
    traversal costs >= 1 (already footprint-aware; no second cost inflation)."""
    costs = costs or {}
    if set(costs) - set(maps):
        raise ValueError("every cost field needs a corresponding obstacle map")
    rng = np.random.default_rng(seed)
    r_cells = max(0, int(round(robot_radius / res)))
    gt_occ_infl = inflate(gt == OCC, r_cells)
    gt_ok = (gt == FREE) & ~gt_occ_infl
    near_obstacle = inflate(gt == OCC, max(0, int(round(check_radius / res))))
    starts = starts if starts is not None else [start]
    jobs = []
    for s in starts:
        s2 = snap(gt_ok, s, int(round(0.5 / res)))
        if s2 is None:
            continue
        for gl in sample_goals(gt_ok, s2, res, n_goals, dist_range, rng):
            jobs.append((s2, gl))
    records = []
    fields = {}

    def field(key, ok, s_):
        if (key, s_) not in fields:
            fields[(key, s_)] = Field(ok, s_, costs.get(key))
        return fields[(key, s_)]

    method_ok = {}
    for m, occ in maps.items():
        method_ok[m] = ~inflate(occ, r_cells)
    for s_, gl in jobs:
        gp = field("__gt__", gt_ok, s_).path(gl)
        if gp is None:
            continue
        gl_len = path_length(gp, res)
        rec = {"start": s_, "goal": gl, "gt_path": gp, "gt_len": round(gl_len, 2), "methods": {}}
        for m, occ in maps.items():
            ok = method_ok[m]
            if not ok[s_] and not occ[s_]:        # the robot's own cell is blocked only if the map marks it occupied
                ok = ok.copy()
                ok[s_] = True
                p = Field(ok, s_, costs.get(m)).path(gl) if ok[gl] else None
            else:
                p = field(m, ok, s_).path(gl) if ok[gl] else None
            if p is None:
                rec["methods"][m] = {"status": "no_path", "spl": 0.0, "path": None}
                continue
            pl = path_length(p, res)
            pr = np.asarray(p)
            if near_obstacle[pr[:, 0], pr[:, 1]].any():
                st = "collision"
            elif pl > max_detour * max(gl_len, res):
                st = "detour"
            else:
                st = "success"
            rec["methods"][m] = {"status": st, "spl": (gl_len / max(gl_len, pl)) if st == "success" else 0.0,
                                 "len": round(pl, 2), "path": p,
                                 "unknown_fraction": float((gt[pr[:, 0], pr[:, 1]] == 0).mean())}
        records.append(rec)
    return records


def summarize(records: list[dict], method: str) -> dict:
    results = [r["methods"][method] for r in records if method in r["methods"]]
    n = len(results)
    if not n:
        return {"n": 0}
    c = {k: sum(r["status"] == k for r in results) / n for k in ("success", "no_path", "collision", "detour")}
    c["spl"] = float(np.mean([r["spl"] for r in results]))
    # Preserve the historical collision metric, but expose paths whose safety
    # cannot be checked because they leave the labelled region.
    c["unknown_path_rate"] = sum(r.get("unknown_fraction", 0) > 0 for r in results) / n
    c["known_success"] = sum(r["status"] == "success" and r.get("unknown_fraction", 0) == 0
                             for r in results) / n
    c["n"] = n
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}
