"""Closed-loop point-goal policy from the selected BEV candidate: occupancy + observability heads + adaptive planner.

    policy = BEVPlannerPolicy(json.load(open('bev_planner.json')), backbone=server_backbone)
    paths, debug = policy(imgs, goal, hist, want_debug=[True, False, ...])

Inputs (one causal window per sample, batched):

* imgs (B, S, 3, H, W) float CUDA in [0, 1], view 0 = current frame, views at `policy.history_m` metres of past
  travel (repeated frames at the start of an episode are allowed: exact duplicates are masked out of the BEV heads,
  see `dedupe_views`). Images are centre-cropped to the training aspect and resized to `image_hw` (default: the
  training rule `target_shape(H / W, area=512^2)`, 448 x 592 for 4:3 input), as in the BEV training data.
* goal (B, 2) metres in the current robot frame (forward, left).
* hist (B, S) metric camera-centre distances from the current camera to each view (odometry). The metric scale is the
  median ratio of these to VGGT's predicted baselines over views with hist > `min_baseline_m` (0.2 m); without such a
  view, the scale head's estimate (as in the frozen evaluation).

Outputs: `paths` numpy float32 (B, 16, 2) in the robot frame (forward, left) metres, points at 0.5 m arc length along
the planned route, padded with the last point; `debug` a list with a dict per sample (None where not requested):

    occ, obs, err  (n, n) float16  P(occupied), P(observed by the inputs), occupancy error score; mean-pooled to
                   n = debug_cells (100) cells of map_res metres. Robot frame: the map centre is the robot, row 0 is
                   the far forward edge, column 0 the far left edge; cell (i, j) centre:
                   forward = (n / 2 - i - 0.5) * map_res, left = (n / 2 - j - 0.5) * map_res
    map_res        float32   metres per debug cell (0.2)
    map_half       float32   half extent in metres (10): the maps cover forward, left in [-map_half, map_half]
    path           (16, 2)   float32 the returned path (forward, left)
    target         (2,)      float32 end of the planned route (goal or local waypoint), NaN without a route
    goal_used      (2,)      float32 the goal the planner used: the input goal, or its projection onto the BEV
                             (inset by goal_margin_m) when the goal lies outside the map
    status         int8      0 route to the goal, 1 local waypoint (frozen re-observation stop), 2 no route
    vis_threshold  float32   predicted-observability threshold of the route (NaN without a route)
    log_scale      float32   log(metres per VGGT unit) used;  odometry_scale  bool, True when from odometry
    up_head_deg    float32   angle between the gravity head's estimate and the gravity used (0 with gravity=head)
    n_views        int8      distinct views given to the BEV heads
    blind_zone     bool      the route treats the camera's floor blind zone ahead as observed (see below)

Frames: the BEV frame (cross_nav/bev.py) of camera 0 has x = right, y = forward (optical axis projected on the
horizontal plane), origin at the camera centre; cell (row, col) centre x = (col + .5 - N/2) res,
y = (N/2 - row - .5) res. A level camera above the robot gives robot (forward, left) = (y, -x), with the robot at the
camera's ground projection = the BEV origin, the corner shared by cells N/2-1 and N/2 (the planner's start cell
is (N/2, N/2), whose centre is 2.5 cm off the origin; returned paths are expressed relative to that start cell).

Planner (frozen, outputs/2026-10-11_reserved_candidate/frozen_policy.json): AdaptiveVisibilityPlanner with
thresholds (.75, .625, .5, .375, .25), occupancy risk weight 128, visibility weight 16, radius 0.2 m, local waypoints
with >= 1 m travel and >= 0.5 m goal progress. It is evaluated lazily (stops at the first threshold with a route),
which gives exactly AdaptiveVisibilityPlanner's result. To plan on occupancy alone (ignore observability), set
planner.thresholds = [0.0] and planner.visibility_weight = 0.

Goals outside the BEV (|forward| or |left| > 10 m - goal_margin_m) are moved along their ray onto the map border
(inset by the margin); progress toward that point is progress along the goal direction. The frozen planner then
finds a route to it when observed space allows, else its local waypoint toward it.

Blind zone (blind_zone): a level camera 0.8 m high with a 90 deg x 74 deg FoV never sees the floor closer than
0.8 / tan(36.9 deg) = 1.07 m. The frozen rule (only the robot's own footprint counts as observed) then finds no route
whenever the past views have not covered that floor: at every episode start (all views are the same frame) and after
turning in place; smoke tests on HSSD windows confirm it. With 'fallback' (default), only when the frozen planner
finds no route, the floor ahead within blind_zone_m (1.3 m) inside the horizontal FoV (blind_zone_hfov_deg, at least
the robot's width) is treated as observed (P(observed) = 1) and the frozen planner runs again; the occupancy risk
there still comes from the occupancy head, which predicts unobserved cells. 'always' applies it before the first
attempt; 'off' is the frozen behaviour. Obstacles lower than about 0.8 - 0.75 d m at distance d inside the zone are
invisible to the camera, so this is an assumption, reported per window in debug['blind_zone'].

No route (no_path): 'rotate' (default) returns 16 copies of a point 0.3 m to the left (90 deg), which the waypoint
controller executes as a turn in place (30 deg per step, no translation): the robot scans until a route appears.
The direction is fixed (not toward the goal side) because a stateless goal-side rule oscillates between two headings
when the goal is near the field-of-view edge and neither heading has a route. 'rotate_goal' turns toward the goal
side instead; 'stop' returns zeros (the robot then stays put, the same window repeats, so it never recovers).

Closed-loop rules (all off by default = the frozen behaviour; cross_nav/configs/policy/bev_planner_best.json enables the
selected set; outputs/2026-10-12_sim_nav/bev_closed_loop.md has the validation). Rules marked * keep per-episode
state (Episode), keyed by the caller's episode id (`episodes=[{'ep', 'odom', 'window_odom'}]`, odometry pose
(x, y, yaw) REP-103) and dropped by end_episode:

    history_rule     keyframe windows (sim.geom.window_ids): views of new headings while turning in place count
    occ_block        cells with P(occupied) > it are impassable (footprint-inflated; the robot's own footprint never)
    near_goal_m      a goal closer than this is driven to straight when that segment is predicted free
    map *            WorldMap: per-window maps fused in the odometry frame, A* to the goal (optimistic: unseen
                     cells passable at a cost); falls back to the per-window planner when it finds no route
    memory_steps *   without a route, keep following the last route (odometry frame) while it stays free
    no_path 'rotate_hold' *  turn toward the goal side and keep that side for the whole no-route streak
    relax_after *    after that many no-route windows, the relaxed planner (occupancy only, short waypoints)
    bump *           commanded forward motion not executed (odometry) -> an obstacle remembered ahead
    scale_memory *   the episode's last odometry scale when a window has no baseline
    scale_pairs      odometry scale from all pairwise view baselines (window odometry)
    unstick_after *  every that many consecutive no-route windows, forget the remembered bumps (and the map's
                     evidence within unstick_clear_m of the robot): bumps left from sliding along walls can enclose
                     the robot, which then turns in place for the rest of the episode
"""
from __future__ import annotations

import functools
import heapq
import math
import os
import time
import types

import numpy as np
import torch
from torch.nn import functional as F

from .sim.geom import control, resample

HISTORY_M = (0., .5, 1., 2., 4., 8., 16., 32.)
FROZEN_PLANNER = {'thresholds': [.75, .625, .5, .375, .25], 'risk_weight': 128., 'visibility_weight': 16.,
                  'radius': .2, 'min_travel': 1., 'min_progress': .5}
DEFAULTS = {
    'backbone': None, 'occupancy': None, 'visibility': None, 'visibility_gravity': None,
    'history_m': list(HISTORY_M),
    'image_hw': None,            # (H, W); None = target_shape(input aspect, area), the BEV training rule
    'area': 512 * 512,
    'gravity': 'imu',            # 'imu': level camera, up = -y of camera 0; 'head': the gravity heads' estimates
    'scale': 'odometry',         # 'odometry' (scale head fallback) | 'mono' (scale head only)
    'min_baseline_m': .2,
    'dedupe_views': True,
    'planner': FROZEN_PLANNER,
    'goal_margin_m': .25,
    'blind_zone': 'fallback',    # 'fallback' | 'always' | 'off' (frozen)
    'blind_zone_m': 1.3,         # floor hidden below the lower FoV edge: 0.8 m / tan(36.9 deg) = 1.07 m, + margin
    'blind_zone_hfov_deg': 90.,
    'no_path': 'rotate',         # 'rotate' | 'rotate_goal' | 'rotate_hold' | 'stop'
    'turn_point_m': .3,
    'horizon': 16, 'spacing': .5,
    'debug_cells': 100,
    'max_chunk': 16,             # windows per backbone pass
    # closed-loop rules (the defaults are the frozen behaviour; see "Closed-loop rules" in the module docstring)
    'history_rule': None,        # None: frames at history_m metres of travel; {'type': 'keyframe', 'trans', 'rot_deg', 'lags'}
    'occ_block': None,           # P(occupied) above which a cell is impassable (footprint-inflated); None: cost only
    'near_goal_m': 0.,           # > 0: a goal closer than this is approached straight when the segment is predicted free
    'near_goal_occ': .5,         # max P(occupied) within near_goal_clear_m of that segment
    'near_goal_clear_m': .1,
    'memory_steps': 0,           # > 0: without a route, follow the last route (odometry frame) for up to this many windows
    'relax_after': -1,           # >= 0: after this many consecutive no-route windows, plan with relax_planner
    'relax_planner': {'thresholds': [0.], 'min_travel': .3, 'min_progress': .1},
    'bump': False,               # commanded forward motion not executed (odometry) -> remembered obstacle ahead
    'bump_ratio': .5, 'bump_ahead_m': .3, 'bump_radius_m': .15, 'bump_max': 64,
    'control': {'lookahead': .5, 'step': .25},   # the executing controller (sim.geom.control), for the bump rule
    'scale_memory': False,       # keep the episode's last odometry scale when the window has no baseline
    'scale_pairs': False,        # odometry scale from all pairwise view baselines (needs window odometry)
    'unstick_after': -1,         # > 0: every that many consecutive no-route windows, forget bumps (+ nearby map evidence)
    'unstick_clear_m': 0.,       # > 0: also clear the episode map's evidence within this radius of the robot
    'map': None,                 # None: plan per window (frozen); a dict (MAP_DEFAULTS keys): episode map, see WorldMap
}
MAP_DEFAULTS = {
    'res': .1, 'size': 800,      # odometry-frame grid (metres per cell, cells per side), centred on the first pose
    'range_m': 6.,               # window cells farther than this from the robot are not fused
    'w_floor': .05,              # fusion weight = max(P(observed), w_floor): unobserved completions count a little
    'decay': 1.,                 # evidence kept per window (1 = no forgetting)
    'unknown_p': .25,            # P(occupied) planned for cells no window has covered (optimistic, passable)
    'block_p': .6, 'block_w': .5,  # impassable: fused P(occupied) > block_p with summed P(observed) >= block_w
    'risk_weight': 4.,           # cell cost 1 + risk_weight * footprint-max P(occupied)
    'radius': .2,
    'margin_m': 4.,              # the search box: robot and goal + this margin
    'retry': False,              # no route in the box: search the whole grid, then block only P > retry_block_p
    'retry_block_p': .85,
    'retry_radii': [],           # after the retries: search again with these smaller inflation radii (narrow passages)
}
STATUS_GOAL, STATUS_WAYPOINT, STATUS_NONE, STATUS_APPROACH, STATUS_MEMORY, STATUS_RELAXED, STATUS_MAP = 0, 1, 2, 3, 4, 5, 6


# ------------------------------------------------------------------------------------------------ geometry (numpy)
def crop_box(H, W, out_hw):
    """The largest crop of an (H, W) image centred on the image centre with the aspect of out_hw, as
    vggt_ft.dataio.transforms.centred_crop_resize (zoom 1, principal point at the centre): (y0, y1, x0, x1)."""
    oh, ow = out_hw
    cx, cy = (W - 1) / 2, (H - 1) / 2
    hw, hh = min(cx + .5, W - .5 - cx), min(cy + .5, H - .5 - cy)
    a = ow / oh
    if hw / hh > a:
        hw = hh * a
    else:
        hh = hw / a
    x0 = int(np.clip(round(cx + .5 - hw), 0, W - 1))
    y0 = int(np.clip(round(cy + .5 - hh), 0, H - 1))
    x1 = int(np.clip(x0 + max(1, round(2 * hw)), 1, W))
    y1 = int(np.clip(y0 + max(1, round(2 * hh)), 1, H))
    return y0, y1, x0, x1


def robot_to_cell(fl, size, res):
    """(..., 2) robot-frame (forward, left) metres -> integer (row, col) of the BEV cell containing the point."""
    fl = np.asarray(fl, np.float64)
    row = np.floor(size / 2 - fl[..., 0] / res).astype(np.int64)
    col = np.floor(size / 2 - fl[..., 1] / res).astype(np.int64)   # x = right = -left
    return np.stack((row, col), -1)


def cell_to_robot(rc, size, res):
    """(..., 2) (row, col) -> robot-frame (forward, left) metres of the cell centres."""
    rc = np.asarray(rc, np.float64)
    return np.stack(((size / 2 - rc[..., 0] - .5) * res, (size / 2 - rc[..., 1] - .5) * res), -1)


def clip_goal(goal, half, margin):
    """Move a goal outside the square [-half + margin, half - margin]^2 along its ray onto that square's border."""
    g = np.asarray(goal, np.float64)
    m, lim = float(np.abs(g).max()), half - margin
    return g * (lim / m) if m > lim else g


def cells_to_path(cells, size, res, horizon=16, spacing=.5):
    """Planner cells (start first) -> (horizon, 2) points at `spacing` arc length, relative to the start cell."""
    p = cell_to_robot(np.asarray(cells), size, res)
    p = p - p[:1]
    return resample(p, horizon, spacing).astype(np.float32)


def turn_path(goal, mode, horizon=16, dist=.3, side=None):
    """Path for 'no route': a turn in place (a point 90 deg aside, beyond the controller's turn-only angle) or stop.
    side (+1 left / -1 right) overrides the mode's direction ('rotate_hold' keeps the side of a no-route streak)."""
    if mode == 'stop':
        return np.zeros((horizon, 2), np.float32)
    if mode not in ('rotate', 'rotate_goal', 'rotate_hold'):
        raise ValueError(f'unknown no_path mode {mode!r}')
    if side is None:
        side = 1. if mode == 'rotate' or float(goal[1]) >= 0 else -1.
    return np.tile(np.array([0., side * dist], np.float32), (horizon, 1))


@functools.lru_cache(maxsize=4)
def blind_zone_mask(size, res, depth, hfov_deg, radius):
    """(size, size) bool: floor ahead of the robot up to `depth` metres, within the horizontal FoV wedge and at least
    the robot's width (radius + one cell) on each side; it starts at -radius so it joins the robot's own footprint."""
    i = np.arange(size)
    fl = cell_to_robot(np.stack(np.meshgrid(i, i, indexing='ij'), -1), size, res)
    f, l = fl[..., 0], fl[..., 1]
    half = np.maximum(np.maximum(f, 0) * math.tan(math.radians(hfov_deg) / 2), radius + res)
    return (f >= -radius) & (f <= depth) & (np.abs(l) <= half)


def adaptive_plan(prob, obs, goal_cell, res, planner, blocked=None):
    """AdaptiveVisibilityPlanner(...).plan(goal_cell), constructing a threshold's planner only when the higher
    thresholds found no route (identical result, fewer shortest-path fields). blocked: optional hard obstacles."""
    from cross_nav.local_plan import LocalWaypointPlanner
    start = (prob.shape[0] // 2, prob.shape[1] // 2)
    kw = {k: planner[k] for k in ('radius', 'risk_weight', 'visibility_weight', 'min_travel', 'min_progress')}
    th = tuple(planner['thresholds'])
    if not th or any(a <= b for a, b in zip(th, th[1:])):
        raise ValueError('thresholds must be strictly decreasing')
    for t in th:
        r = LocalWaypointPlanner(prob, obs, start, res, threshold=t, blocked=blocked, **kw).plan(tuple(int(x) for x in goal_cell))
        if r['path'] is not None:
            break
    return {**r, 'visibility_threshold': t}


def max_filter(prob, radius_m, res):
    """Max of prob within radius_m of each cell."""
    import cv2
    from cross_nav.plan import disk
    r = int(math.ceil(radius_m / res - 1e-9))
    return cv2.dilate(np.asarray(prob, np.float32), disk(r)) if r > 0 else np.asarray(prob, np.float32)


def polyline_max(pmax, pts, res, skip_m=0.):
    """Max of the map pmax over the cells along a robot-frame polyline starting at the robot, ignoring points within
    skip_m of the robot; 1 when the line leaves the map."""
    size = pmax.shape[0]
    p = np.vstack(([0., 0.], np.asarray(pts, np.float64)))
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / (res / 2))))
        out.append(a + np.linspace(0, 1, n + 1)[1:, None] * (b - a))
    q = np.concatenate(out)
    q = q[np.linalg.norm(q, axis=1) >= skip_m]
    if not len(q):
        return 0.
    rc = robot_to_cell(q, size, res)
    if (rc < 0).any() or (rc >= size).any():
        return 1.
    return float(pmax[rc[:, 0], rc[:, 1]].max())


def to_robot_2d(points, pose):
    """Odometry-frame (x, y) points -> robot frame (forward, left) of pose (x, y, yaw), yaw counter-clockwise."""
    x, y, yaw = (float(v) for v in pose)
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(points, np.float64) - (x, y)
    return np.stack((d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c), -1)


def to_odom_2d(points, pose):
    x, y, yaw = (float(v) for v in pose)
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(points, np.float64)
    return np.stack((x + p[..., 0] * c - p[..., 1] * s, y + p[..., 0] * s + p[..., 1] * c), -1)


class Episode:
    """Per-episode planner state (keyed by the caller's episode id, dropped at the episode's end). Rules needing
    motion use the odometry pose (x, y, yaw) of each window's current frame; without it they are inactive."""
    def __init__(self):
        self.pose = None          # odometry pose of the previous window
        self.expect = 0.          # forward motion the previous path commands (metres)
        self.bumps = []           # odometry-frame points where commanded motion was not executed
        self.route = None         # last planned route, odometry frame (N, 2)
        self.route_age = 0        # windows since it was planned
        self.no_route = 0         # consecutive windows without a route (turn-in-place)
        self.side = 0.            # turn direction of the current no-route streak
        self.log_scale = math.nan  # last odometry scale
        self.n_bumps = 0
        self.world = None         # WorldMap (cfg['map'])
        self.n_unstick = 0        # times the unstick rule fired

    def observe(self, pose, cfg):
        """New window at pose: detect blocked motion of the previous command (bump rule)."""
        if pose is None:
            return
        pose = np.asarray(pose, np.float64)
        if cfg['bump'] and self.pose is not None and self.expect > .05:
            moved = float(np.hypot(*(pose[:2] - self.pose[:2])))
            if moved < cfg['bump_ratio'] * self.expect:
                ahead = cfg['bump_ahead_m']
                self.bumps.append(pose[:2] + ahead * np.array([math.cos(pose[2]), math.sin(pose[2])]))
                self.bumps = self.bumps[-cfg['bump_max']:]
                self.n_bumps += 1
        self.pose = pose

    def unstick(self, cfg):
        """After a no-route streak of unstick_after windows (and each multiple): forget bumps and nearby map evidence."""
        n = cfg['unstick_after']
        if n <= 0 or self.no_route < n or self.no_route % n or self.pose is None:
            return False
        self.bumps = []
        if self.world is not None and cfg['unstick_clear_m'] > 0:
            self.world.clear(self.pose[:2], cfg['unstick_clear_m'])
        self.n_unstick += 1
        return True

    def add_bumps(self, prob, res, cfg):
        """prob with remembered obstacles (P = 1 within bump_radius_m of each bump point)."""
        if not self.bumps or self.pose is None:
            return prob
        size = prob.shape[0]
        fl = to_robot_2d(np.asarray(self.bumps), self.pose)
        half = size * res / 2
        fl = fl[(np.abs(fl) < half).all(1)]
        if not len(fl):
            return prob
        prob = prob.copy()
        r = cfg['bump_radius_m']
        k = int(math.ceil(r / res))
        for c in robot_to_cell(fl, size, res):
            r0, r1, c0, c1 = max(0, c[0] - k), min(size, c[0] + k + 1), max(0, c[1] - k), min(size, c[1] + k + 1)
            rr, cc = np.mgrid[r0:r1, c0:c1]
            m = np.hypot(rr - c[0], cc - c[1]) * res <= r
            prob[rr[m], cc[m]] = 1.
        return prob

    def remembered(self, pmax, res, cfg):
        """The unfinished part of the last route in the current robot frame, or None (too old, done or blocked)."""
        if self.route is None or self.pose is None or self.route_age >= cfg['memory_steps']:
            return None
        fl = to_robot_2d(self.route, self.pose)
        k = int(np.argmin(np.linalg.norm(fl, axis=1)))
        rest = fl[k:] if np.linalg.norm(fl[k]) > .05 else fl[k + 1:]
        if len(rest) == 0 or np.linalg.norm(rest[-1]) < .25:
            return None
        if cfg['occ_block'] is not None and polyline_max(pmax, rest, res, skip_m=cfg['planner']['radius']) > cfg['occ_block']:
            return None
        return resample(np.vstack(([0., 0.], rest)), cfg['horizon'], cfg['spacing']).astype(np.float32)


def _astar_impl(ok, cost, sr, sc, gr, gc):
    """8-connected A* over ok cells (no corner cutting) with edge cost length * mean cell cost (cost >= 1, so the
    octile distance is admissible): the path as an (n, 2) array of (row, col), empty when unreachable."""
    H, W = ok.shape
    g = np.full((H, W), np.inf)
    parent = -np.ones((H, W), np.int64)
    closed = np.zeros((H, W), np.bool_)
    g[sr, sc] = 0.
    heap = [(0., sr * W + sc)]
    found = False
    while len(heap) > 0:
        f, k = heapq.heappop(heap)
        r, c = k // W, k % W
        if closed[r, c]:
            continue
        closed[r, c] = True
        if r == gr and c == gc:
            found = True
            break
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W or not ok[nr, nc] or closed[nr, nc]:
                    continue
                diag = dr != 0 and dc != 0
                if diag and (not ok[r, nc] or not ok[nr, c]):
                    continue
                nd = g[r, c] + (1.4142135623730951 if diag else 1.) * .5 * (cost[r, c] + cost[nr, nc])
                if nd < g[nr, nc]:
                    g[nr, nc] = nd
                    parent[nr, nc] = k
                    ar, ac = abs(nr - gr), abs(nc - gc)
                    h = max(ar, ac) + .41421356237309515 * min(ar, ac)
                    heapq.heappush(heap, (nd + h, nr * W + nc))
    if not found:
        return np.empty((0, 2), np.int64)
    n = 0
    k = gr * W + gc
    while k >= 0:
        n += 1
        k = parent[k // W, k % W]
    out = np.empty((n, 2), np.int64)
    k = gr * W + gc
    for i in range(n - 1, -1, -1):
        out[i, 0], out[i, 1] = k // W, k % W
        k = parent[k // W, k % W]
    return out


@functools.lru_cache(maxsize=1)
def _astar():
    from cross_nav.plan import njit       # numba when installed, else plain Python
    return njit(cache=True)(_astar_impl)


def astar(ok, cost, start, goal):
    return _astar()(np.ascontiguousarray(ok, np.bool_), np.ascontiguousarray(cost, np.float64), int(start[0]),
                    int(start[1]), int(goal[0]), int(goal[1]))


class WorldMap:
    """Episode occupancy map in the odometry frame: the per-window BEV maps fused by a weighted average with weight
    max(P(observed), w_floor) within range_m of the robot (odometry pose of each window); blocking needs observed
    evidence (summed P(observed) >= block_w), so unobserved completions only add cost. Cell (v, u) centre:
    x = ox + (u + .5) res, y = oy + (v + .5) res. Planning (plan): A* from the robot to the goal with cost
    1 + risk_weight * (footprint-max P(occupied)); cells no window has covered get unknown_p (optimistic: the
    route may enter unseen space and is re-planned as it is observed); cells with fused P > block_p and weight
    >= block_w, and remembered bumps, are impassable (inflated by the radius; the robot's and the goal's own
    footprints stay free)."""
    def __init__(self, origin, cfg):
        self.c = {**MAP_DEFAULTS, **cfg}
        n, r = self.c['size'], self.c['res']
        self.o = np.asarray(origin[:2], np.float64) - n * r / 2
        self.num = np.zeros((n, n), np.float32)
        self.den = np.zeros((n, n), np.float32)
        self.seen = np.zeros((n, n), np.float32)    # summed P(observed): evidence that may block

    def clear(self, xy, radius_m):
        """Forget the evidence within radius_m of the odometry-frame point xy (the cells become unseen)."""
        R = self.c['res']
        v, u = self.cell(xy)
        k = int(math.ceil(radius_m / R))
        r0, r1, c0, c1 = max(0, v - k), min(self.c['size'], v + k + 1), max(0, u - k), min(self.c['size'], u + k + 1)
        if r0 >= r1 or c0 >= c1:
            return
        rr, cc = np.mgrid[r0:r1, c0:c1]
        m = np.hypot(rr - v, cc - u) * R <= radius_m
        for a in (self.num, self.den, self.seen):
            a[rr[m], cc[m]] = 0.

    def cell(self, xy):
        return np.floor((np.asarray(xy, np.float64) - self.o) / self.c['res']).astype(np.int64)[..., ::-1]   # (v, u)

    def fuse(self, prob, obs, pose, res_b):
        import cv2
        c, n, R = self.c, self.c['size'], self.c['res']
        N = prob.shape[0]
        x0, y0, th = (float(v) for v in pose)
        ct, st = math.cos(th), math.sin(th)
        # world grid pixel (u, v) -> BEV pixel (col, row); BEV cell centre: forward = (N/2 - row - .5) res_b, left = ...
        bx, by = self.o[0] + .5 * R - x0, self.o[1] + .5 * R - y0
        M = np.array([[R * st / res_b, -R * ct / res_b, N / 2 - .5 - (-bx * st + by * ct) / res_b],
                      [-R * ct / res_b, -R * st / res_b, N / 2 - .5 - (bx * ct + by * st) / res_b]], np.float64)
        i = (np.arange(N) - N / 2 + .5) * res_b
        rng = np.hypot(i[:, None], i[None]) <= c['range_m']
        w = np.where(rng, np.maximum(obs, c['w_floor']), 0.).astype(np.float32)
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        warp = lambda a: cv2.warpAffine(a, M, (n, n), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if c['decay'] < 1:
            self.num *= c['decay']; self.den *= c['decay']; self.seen *= c['decay']
        self.num += warp(w * prob.astype(np.float32))
        self.den += warp(w)
        self.seen += warp(np.where(rng, obs, 0.).astype(np.float32))

    def plan(self, pose, goal_xy, bumps=(), bump_radius=.15):
        """Odometry-frame route (n, 2) from the pose to the goal, or None. With retry, a failed search is repeated
        over the whole grid, then with only P(occupied) > retry_block_p blocking (bumps always block), then with each
        inflation radius of retry_radii (a robot in a passage narrower than the inflated predicted walls)."""
        route = self._plan(pose, goal_xy, bumps, bump_radius, self.c['margin_m'], self.c['block_p'])
        if route is None and self.c['retry']:
            full = self.c['size'] * self.c['res']
            route = self._plan(pose, goal_xy, bumps, bump_radius, full, self.c['block_p'])
            if route is None:
                route = self._plan(pose, goal_xy, bumps, bump_radius, full, self.c['retry_block_p'])
            for r in self.c['retry_radii']:
                if route is not None:
                    break
                route = self._plan(pose, goal_xy, bumps, bump_radius, full, self.c['retry_block_p'], r)
        return route

    def _plan(self, pose, goal_xy, bumps, bump_radius, margin_m, block_p, radius=None):
        import cv2
        from cross_nav.plan import disk
        c, n, R = self.c, self.c['size'], self.c['res']
        s, g = self.cell(pose[:2]), self.cell(goal_xy)
        g = np.clip(g, 0, n - 1)
        lo = np.maximum(np.minimum(s, g) - int(margin_m / R), 0)
        hi = np.minimum(np.maximum(s, g) + int(margin_m / R) + 1, n)
        sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))
        den, num = self.den[sl], self.num[sl]
        p = np.where(den > 1e-3, num / np.maximum(den, 1e-6), c['unknown_p']).astype(np.float32)
        blocked = (p > block_p) & (self.seen[sl] >= c['block_w'])
        for b in bumps:
            v, u = self.cell(b) - lo
            k = int(math.ceil(bump_radius / R))
            r0, r1, c0, c1 = max(0, v - k), min(p.shape[0], v + k + 1), max(0, u - k), min(p.shape[1], u + k + 1)
            if r0 < r1 and c0 < c1:
                rr, cc = np.mgrid[r0:r1, c0:c1]
                m = np.hypot(rr - v, cc - u) * R <= bump_radius
                blocked[rr[m], cc[m]] = True
                p[rr[m], cc[m]] = 1.
        k = int(math.ceil((c['radius'] if radius is None else radius) / R - 1e-9))
        own = np.zeros(p.shape, np.uint8)
        for q in (s - lo, g - lo):
            own[tuple(q)] = 1
        own = cv2.dilate(own, disk(k)) > 0
        blocked &= ~own
        ok = ~(cv2.dilate(blocked.astype(np.uint8), disk(k)) > 0) if k else ~blocked
        ok[tuple(s - lo)] = ok[tuple(g - lo)] = True
        pooled = cv2.dilate(p, disk(k)) if k else p
        cost = 1. + c['risk_weight'] * np.where(own, p, pooled)
        path = astar(ok, cost, s - lo, g - lo)
        if not len(path):
            return None
        vu = path + lo
        return self.o + (vu[:, ::-1] + .5) * R


def plan_window(prob, obs, goal, res, cfg, state=None):
    """One window: (16, 2) path and a small record (status, target, goal used, threshold). state: the episode's
    Episode (None: stateless, the per-episode rules are inactive); remembered bumps must already be in prob
    (Episode.add_bumps)."""
    size = prob.shape[0]
    pc = cfg['planner']
    goal = np.asarray(goal, np.float64)
    g = clip_goal(goal, size * res / 2, cfg['goal_margin_m'])
    gc = robot_to_cell(g, size, res)
    mode = cfg['blind_zone']
    if mode not in ('fallback', 'always', 'off'):
        raise ValueError(f'unknown blind_zone mode {mode!r}')
    zone = None
    if mode != 'off':
        zone = blind_zone_mask(size, res, cfg['blind_zone_m'], cfg['blind_zone_hfov_deg'], pc['radius'])
    blocked = prob > cfg['occ_block'] if cfg['occ_block'] is not None else None
    start = np.array([size // 2, size // 2])
    rec = {'target': np.full(2, np.nan, np.float32), 'goal_used': g.astype(np.float32),
           'vis_threshold': np.float32(np.nan), 'blind_zone': False}
    pmax = max_filter(prob, pc['radius'], res) if (state is not None and cfg['memory_steps'] > 0) else None

    def done(path, status, **kw):
        path = np.asarray(path, np.float32)
        if state is not None:
            if status == STATUS_NONE:
                state.no_route += 1
            else:
                state.no_route = 0
            if status in (STATUS_GOAL, STATUS_WAYPOINT, STATUS_APPROACH, STATUS_RELAXED, STATUS_MAP) and state.pose is not None:
                state.route, state.route_age = to_odom_2d(path, state.pose), 0
            elif state.route is not None:
                state.route_age += 1
        return path, {**rec, **kw, 'status': status}

    # near-goal approach: the straight segment to a close goal, when predicted free (no observability needed)
    if cfg['near_goal_m'] > 0 and float(np.linalg.norm(goal)) < cfg['near_goal_m']:
        pm = max_filter(prob, cfg['near_goal_clear_m'], res)
        if polyline_max(pm, goal[None], res, skip_m=pc['radius']) < cfg['near_goal_occ']:
            path = resample(np.vstack(([0., 0.], goal)), cfg['horizon'], cfg['spacing'])
            return done(path, STATUS_APPROACH, target=goal.astype(np.float32))
    # episode map: A* on the fused odometry-frame map to the goal
    if cfg['map'] and state is not None and state.world is not None and state.pose is not None:
        route = state.world.plan(state.pose, to_odom_2d(goal, state.pose), state.bumps if cfg['bump'] else (),
                                 cfg['bump_radius_m'])
        if route is not None:
            fl = to_robot_2d(route, state.pose)
            fl[0] = 0.
            path = resample(fl, cfg['horizon'], cfg['spacing'])
            return done(path, STATUS_MAP, target=fl[-1].astype(np.float32))
    used = mode == 'always'
    r = adaptive_plan(prob, np.maximum(obs, zone) if used else obs, gc, res, pc, blocked)
    if r['path'] is None and mode == 'fallback':
        used = True
        r = adaptive_plan(prob, np.maximum(obs, zone).astype(np.float32), gc, res, pc, blocked)
    rec['blind_zone'] = used
    status = STATUS_WAYPOINT if r['stop_for_reobserve'] else STATUS_GOAL
    if r['path'] is None and state is not None and cfg['memory_steps'] > 0:
        path = state.remembered(pmax, res, cfg)
        if path is not None:
            state.route_age += 1
            state.no_route = 0
            return path, {**rec, 'status': STATUS_MEMORY}
    if r['path'] is None and cfg['relax_after'] >= 0 and (state.no_route if state is not None else 0) >= cfg['relax_after']:
        rp = {**pc, **cfg['relax_planner']}
        r = adaptive_plan(prob, np.maximum(obs, zone).astype(np.float32) if zone is not None else obs, gc, res, rp, blocked)
        status = STATUS_RELAXED
    if r['path'] is None:
        side = None
        if state is not None and cfg['no_path'] == 'rotate_hold':
            if state.no_route == 0 or state.side == 0:
                state.side = 1. if float(goal[1]) >= 0 else -1.
            side = state.side
        path = turn_path(goal, cfg['no_path'], cfg['horizon'], cfg['turn_point_m'], side)
        return done(path, STATUS_NONE)
    path = cells_to_path(r['path'], size, res, cfg['horizon'], cfg['spacing'])
    target = cell_to_robot(np.asarray(r['target']), size, res) - cell_to_robot(start, size, res)
    return done(path, status, target=target.astype(np.float32), vis_threshold=np.float32(r['visibility_threshold']))


def resolve(cfg):
    """Defaults + config; environment variables in path strings are expanded (${CROSS_NAV_ROOT}/...)."""
    out = {**DEFAULTS, **cfg, 'planner': {**FROZEN_PLANNER, **cfg.get('planner', {})},
           'relax_planner': {**DEFAULTS['relax_planner'], **cfg.get('relax_planner', {})},
           'control': {**DEFAULTS['control'], **cfg.get('control', {})},
           'map': {**MAP_DEFAULTS, **cfg['map']} if cfg.get('map') is not None else None}
    if out['no_path'] not in ('rotate', 'rotate_goal', 'rotate_hold', 'stop'):
        raise ValueError(f"unknown no_path mode {out['no_path']!r}")
    for k in ('backbone', 'occupancy', 'visibility', 'visibility_gravity'):
        if out[k]:
            out[k] = os.path.expandvars(os.path.expanduser(out[k]))
            if '$' in out[k]:
                raise ValueError(f'unset environment variable in {k}: {out[k]}')
    return out


# ------------------------------------------------------------------------------------------------ policy (GPU)
class BEVPlannerPolicy:
    def __init__(self, cfg: dict, backbone=None, predictor=None):
        """predictor: a loaded cross_nav.infer.BEVPredictor whose maps the caller computes elsewhere (CROSS's own
        feed-forward pass, pose_est.ff.nav_bev_head): no backbone is loaded and only plan_maps() is used."""
        from cross_nav.infer import BEVPredictor
        self.cfg = cfg = resolve(cfg)
        self.dev = torch.device('cuda')
        self.model = None
        if predictor is not None:
            self.pred = predictor
        else:
            if backbone is not None:          # the server's FrozenMetricTokens: same VGGTOmegaFT + scale head weights
                self.model = backbone.model
            else:
                from .backbone import FrozenMetricTokens
                self.model = FrozenMetricTokens(cfg['backbone']).model.to(self.dev)
            self.model.eval()
            if (cfg['visibility'] is None) != (cfg['visibility_gravity'] is None) or cfg['visibility'] is None:
                raise ValueError('the selected candidate needs occupancy, visibility and visibility_gravity checkpoints')
            self.pred = BEVPredictor(cfg['occupancy'], 'cuda', visibility_path=cfg['visibility'],
                                     visibility_gravity_path=cfg['visibility_gravity'])
        self.size, self.res = self.pred.size, self.pred.res
        self.history_m = tuple(float(h) for h in cfg['history_m'])
        if cfg['gravity'] not in ('imu', 'head') or cfg['scale'] not in ('odometry', 'mono'):
            raise ValueError('gravity must be imu|head, scale odometry|mono')
        # the window rule the caller should use (sim workers / NavigationAgent); None = distances history_m
        self.history_rule = dict(cfg['history_rule']) if cfg['history_rule'] else None
        self.episodes = {}
        self.timing = {}

    def end_episode(self, ep):
        self.episodes.pop(ep, None)

    # images -> the network's input resolution
    def preprocess(self, imgs):
        B, S, _, H, W = imgs.shape
        if self.cfg['image_hw']:
            out_hw = tuple(self.cfg['image_hw'])
        else:
            from vggt_ft.dataio.transforms import target_shape
            out_hw = target_shape(H / W, self.cfg['area'])
        y0, y1, x0, x1 = crop_box(H, W, out_hw)
        x = imgs[..., y0:y1, x0:x1].reshape(B * S, 3, y1 - y0, x1 - x0).float()
        if (y1 - y0, x1 - x0) != out_hw:
            down = out_hw[0] < y1 - y0
            # cv2.INTER_LINEAR (upsampling) / INTER_AREA-like antialiasing (downsampling), as in training
            x = F.interpolate(x, size=out_hw, mode='bilinear', align_corners=False, antialias=down)
        return x.clamp_(0, 1).reshape(B, S, 3, *out_hw)

    @staticmethod
    def duplicate_views(imgs):
        """(B, S) bool: views identical to an earlier view of the same window."""
        B, S = imgs.shape[:2]
        dup = torch.zeros(B, S, dtype=torch.bool, device=imgs.device)
        flat = imgs.reshape(B, S, -1)
        for s in range(1, S):
            for t in range(s):
                dup[:, s] |= (flat[:, s] == flat[:, t]).all(1)
        return dup

    @torch.no_grad()
    def maps(self, imgs, hist, up_cam=None, prior_log_scale=None, pair_dist=None):
        """Per-window BEV maps (B, N, N) on the GPU and the scale / gravity used."""
        from cross_nav.model import NavModel
        from cross_nav.uncertainty import uncertainty_score
        from .server import odometry_or_mono
        cfg, P = self.cfg, self.pred
        B, S = imgs.shape[:2]
        valid = ~self.duplicate_views(imgs) if cfg['dedupe_views'] else torch.ones(B, S, dtype=torch.bool, device=imgs.device)
        x = self.preprocess(imgs)
        bb = NavModel.backbone(types.SimpleNamespace(backbone_model=self.model), x)   # exactly the BEV training pass
        tl, pts, phw = bb['tokens_list'], bb['pts'], bb['patch_hw']
        E = bb['extrinsics'].float()
        centres = -torch.einsum('bsji,bsj->bsi', E[..., :3, :3], E[..., :3, 3])
        chord = (centres - centres[:, :1]).norm(dim=-1)
        mono = bb['log_scale_head'].float().reshape(B)
        if cfg['scale'] == 'odometry':
            # odometry_or_mono uses baselines > 0.2 m; a larger min_baseline_m and duplicate views drop more
            h = torch.where(valid & (hist.float() > cfg['min_baseline_m']), hist.float(), torch.zeros_like(hist.float()))
            ls = odometry_or_mono(chord, h, mono)
            odo = ((h > .2) & (chord > 1e-3)).any(1) & torch.isfinite(ls)
            if pair_dist is not None:
                # all pairwise baselines of the window (metric from odometry, NaN unknown) vs VGGT's camera centres
                pd = pair_dist.to(x.device).float()
                pp = (centres[:, :, None] - centres[:, None]).norm(dim=-1)
                upper = torch.ones(S, S, dtype=torch.bool, device=x.device).triu(1)
                ok = (valid[:, :, None] & valid[:, None] & upper & torch.isfinite(pd) & (pd > max(cfg['min_baseline_m'], .2))
                      & (pp > 1e-3))
                ratio = torch.where(ok, pd / pp.clamp_min(1e-3), torch.nan).reshape(B, -1)
                lp = ratio.nanmedian(dim=1).values.log()
                okp = ok.reshape(B, -1).any(1) & torch.isfinite(lp)
                ls, odo = torch.where(okp, lp, ls), odo | okp
            if prior_log_scale is not None:
                pr = prior_log_scale.to(x.device).float()
                ls = torch.where(~odo & torch.isfinite(pr), pr, ls)
        else:
            ls = mono
            odo = torch.zeros(B, dtype=torch.bool, device=x.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            up_head = P.grav(tl, pts).float()
        imu = torch.tensor([0., -1., 0.], device=x.device).expand(B, 3)
        up = imu if cfg['gravity'] == 'imu' else up_head
        if up_cam is not None:
            up = F.normalize(up_cam.to(x.device).float().reshape(B, 3), dim=-1)
        d, c = bb['depth'].float(), bb['depth_conf'].float()
        intr = bb['intrinsics'].float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = P.bev(tl, pts, phw, bb['dino'], d, c, E, intr, ls, up, valid, valid, return_aux=True)
            prob = pred['logits'].float().sigmoid()
            vis_up = P.visibility_grav(tl, pts).float() if cfg['gravity'] == 'head' and up_cam is None else up
            obs = P.visibility(tl, pts, phw, bb['dino'], d, c, E, intr, ls, vis_up, valid, valid).float().sigmoid()
        err = (uncertainty_score(pred['error_logits'], P.bev.uncertainty_mode).float()
               if pred.get('error_logits') is not None else torch.zeros_like(prob))
        cos = (F.normalize(up_head, dim=-1) * F.normalize(up, dim=-1)).sum(-1).clamp(-1, 1)
        return {'prob': prob, 'obs': obs, 'err': err, 'log_scale': ls, 'odometry': odo,
                'up_head_deg': torch.rad2deg(torch.acos(cos)), 'n_views': valid.sum(1)}

    def __call__(self, imgs, goal, hist, want_debug=None, up_cam=None, episodes=None):
        """See the module docstring. up_cam: optional (B, 3) gravity up in camera 0 (OpenCV axes), e.g. from a tilted
        IMU; it overrides the configured gravity source. episodes: optional list (B) of dicts {'ep': id, 'odom':
        (x, y, yaw) of the current frame, 'window_odom': (S, 3) poses of the window's frames, optional 'expect': the
        forward motion (metres) the robot commanded since the previous window, for the bump rule; without it the rule
        assumes the simulator's step controller (cfg['control'])}, which enable the per-episode rules (state is kept
        per id until end_episode)."""
        cfg = self.cfg
        B = imgs.shape[0]
        want_debug = list(want_debug) if want_debug is not None else [False] * B
        goal_np = goal.detach().float().cpu().numpy() if torch.is_tensor(goal) else np.asarray(goal, np.float32)
        hist = hist if torch.is_tensor(hist) else torch.as_tensor(hist)
        hist = hist.to(imgs.device).float()
        episodes = list(episodes) if episodes is not None else [None] * B
        states = [None if e is None or e.get('ep') is None else self.episodes.setdefault(e['ep'], Episode()) for e in episodes]
        prior = None
        if cfg['scale_memory']:
            prior = torch.tensor([s.log_scale if s is not None else math.nan for s in states], dtype=torch.float32)
        pairs = None
        if cfg['scale_pairs'] and any(e is not None and e.get('window_odom') is not None for e in episodes):
            S = imgs.shape[1]
            pairs = torch.full((B, S, S), math.nan)
            for j, e in enumerate(episodes):
                if e is not None and e.get('window_odom') is not None:
                    w = torch.as_tensor(np.asarray(e['window_odom'], np.float64)[:, :2], dtype=torch.float32)
                    pairs[j] = (w[:, None] - w[None]).norm(dim=-1)
        t0 = time.perf_counter()
        parts = []
        for i in range(0, B, cfg['max_chunk']):
            sl = slice(i, i + cfg['max_chunk'])
            m = self.maps(imgs[sl], hist[sl], None if up_cam is None else up_cam[sl],
                          None if prior is None else prior[sl], None if pairs is None else pairs[sl])
            parts.append({'prob': m['prob'].cpu().numpy(), 'obs': m['obs'].cpu().numpy(), 'err': m['err'].cpu().numpy(),
                          **{n: m[n].cpu().numpy() for n in ('log_scale', 'odometry', 'up_head_deg', 'n_views')}})
        cat = {n: np.concatenate([p[n] for p in parts]) for n in parts[0]}
        t1 = time.perf_counter()
        paths, debug = self.plan_maps(cat, goal_np, states, episodes, want_debug)
        self.timing = {'maps_s': t1 - t0, 'plan_s': time.perf_counter() - t1}
        return paths, debug

    def plan_maps(self, cat, goal_np, states, episodes, want_debug):
        """Paths from per-window maps: cat = numpy arrays with a leading batch axis, 'prob' / 'obs' / 'err' (B, N, N)
        and 'log_scale', 'odometry', 'up_head_deg', 'n_views' (B,); optional 'fresh' (B,) bool, False = no new map
        for that window (planned on the episode map / last route only: prob and obs are not fused). states: per sample
        Episode or None; episodes: the caller's dicts (see __call__)."""
        cfg = self.cfg
        B = len(goal_np)
        fresh = cat.get('fresh', np.ones(B, dtype=bool))
        paths = np.zeros((B, cfg['horizon'], 2), np.float32)
        debug = [None] * B
        k = cfg['debug_cells']
        f = self.size // k

        def pool(a):
            return a.reshape(k, f, k, f).mean((1, 3)).astype(np.float16)
        for j in range(B):
            st = states[j]
            if st is not None:
                if episodes[j].get('expect') is not None:    # forward motion the robot commanded since the last window
                    st.expect = float(episodes[j]['expect'])
                st.observe(episodes[j].get('odom'), cfg)
                st.unstick(cfg)
                if cat['odometry'][j]:
                    st.log_scale = float(cat['log_scale'][j])
            prob = np.clip(cat['prob'][j], 0, 1)    # the planner validates probability maps strictly
            if st is not None and cfg['map'] and st.pose is not None and fresh[j]:
                if st.world is None:
                    st.world = WorldMap(st.pose, cfg['map'])
                st.world.fuse(prob, np.clip(cat['obs'][j], 0, 1), st.pose, self.res)
            if st is not None:
                prob = st.add_bumps(prob, self.res, cfg)
            paths[j], rec = plan_window(prob, np.clip(cat['obs'][j], 0, 1), goal_np[j], self.res, cfg,
                                        st)
            if st is not None:
                c = cfg['control']
                st.expect = float(control(paths[j], c['lookahead'], c['step'])[1])
            if want_debug[j]:
                debug[j] = {'occ': pool(prob), 'obs': pool(cat['obs'][j]), 'err': pool(cat['err'][j]),
                            'map_res': np.float32(self.size * self.res / k), 'map_half': np.float32(self.size * self.res / 2),
                            'path': paths[j].copy(), 'target': rec['target'], 'goal_used': rec['goal_used'],
                            'status': np.int8(rec['status']), 'vis_threshold': np.float32(rec['vis_threshold']),
                            'log_scale': np.float32(cat['log_scale'][j]), 'odometry_scale': np.bool_(cat['odometry'][j]),
                            'up_head_deg': np.float32(cat['up_head_deg'][j]), 'n_views': np.int8(cat['n_views'][j]),
                            'blind_zone': np.bool_(rec['blind_zone']),
                            'bumps': np.int16(st.n_bumps if st is not None else 0)}
                debug[j] = {n: np.asarray(v) for n, v in debug[j].items()}   # 0-d arrays survive the wire format
        return paths, debug

