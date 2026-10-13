"""Hybrid navigation: the BEV planner and the direct trajectory head, switched per step from odometry and BEV state.

The two systems fail on different episodes (HSSD test: either succeeds on 99.4% of episodes, each alone 94-96%), so
one of them (`primary`) drives and the other takes over for `block` planning steps when the primary is in trouble:

* stall      the robot moved less than `stall_disp` m from where it was `stall_window` steps ago (pushing, spinning)
* no_route   the BEV planner had no route for `no_route` consecutive windows (turning in place)
* progress   the straight-line goal distance shrank by less than `progress_m` over `progress_window` steps (wandering)

A trigger set to 0 is off. After the block, control returns to the primary with empty trigger histories. The BEV
planner runs on every step either way, so its episode map and bump memory stay current; the trajectory head keeps
its own stuck recovery (recovery.py), reset at each hand-over.

Windows count planning steps (simulator: one step = up to 0.25 m or 45 deg, about 0.5 s at 0.5 m/s). On a robot,
give time stamps and the `*_s` keys (block_s, stall_s, progress_s, in seconds): the windows then span that time,
whatever the planning rate (robot.HybridPlanner).

    switch = Switch(cfg)                                  # one per episode
    ctrl = switch(pose, goal_rel, bev_no_route)           # 'bev' or 'traj' for this step
    ctrl = switch(pose, goal_rel, bev_no_route, stamp)    # time windows (cfg block_s / stall_s / progress_s)
"""
from __future__ import annotations

from collections import deque

import numpy as np

DEFAULTS = {
    'primary': 'bev',           # 'bev' | 'traj'
    'block': 40,                # planning steps under the secondary system per hand-over
    'stall_window': 30, 'stall_disp': .5,
    'no_route': 0,
    'progress_window': 0, 'progress_m': 1.,
    'block_s': 0., 'stall_s': 0., 'progress_s': 0.,     # time windows (seconds) used when stamps are given
}


class Switch:
    def __init__(self, cfg=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        if self.cfg['primary'] not in ('bev', 'traj'):
            raise ValueError("primary must be 'bev' or 'traj'")
        self.secondary = 'traj' if self.cfg['primary'] == 'bev' else 'bev'
        self.left, self.n_switch, self.fresh, self.until = 0, 0, False, None
        self.poses = deque(maxlen=max(1, self.cfg['stall_window']))
        self.dist = deque(maxlen=max(1, self.cfg['progress_window']) + 1)
        self.timed = deque()        # (stamp, xy, goal distance) for the time windows

    def clear(self):
        self.poses.clear(); self.dist.clear(); self.timed.clear()

    def __call__(self, pose, goal_rel, no_route=0, stamp=None):
        """pose (x, y, yaw) odometry, goal_rel (2,) goal in the robot frame, no_route the BEV planner's current no-route
        streak, stamp time in seconds (time windows when cfg block_s > 0). Returns the controller for this step;
        self.fresh is True on the first step of a hand-over."""
        c = self.cfg
        self.fresh = False
        if stamp is not None and c['block_s'] > 0:
            return self._timed(pose, goal_rel, no_route, float(stamp))
        if self.left > 0:
            self.left -= 1
            if self.left == 0:
                self.clear()
            return self.secondary
        xy = np.asarray(pose[:2], np.float64)
        self.poses.append(xy); self.dist.append(float(np.hypot(*goal_rel)))
        trouble = False
        if c['stall_window'] and len(self.poses) == self.poses.maxlen:
            trouble |= float(np.max(np.linalg.norm(np.stack(self.poses) - xy, axis=-1))) < c['stall_disp']
        if c['no_route'] and no_route >= c['no_route']:
            trouble = True
        if c['progress_window'] and len(self.dist) == self.dist.maxlen:
            trouble |= self.dist[0] - self.dist[-1] < c['progress_m']
        if not trouble:
            return c['primary']
        self.n_switch += 1
        self.fresh = True
        self.left = c['block'] - 1
        self.clear()
        return self.secondary

    def _timed(self, pose, goal_rel, no_route, t):
        c = self.cfg
        if self.until is not None:
            if t < self.until:
                return self.secondary
            self.until = None
            self.clear()
        xy, d = np.asarray(pose[:2], np.float64), float(np.hypot(*goal_rel))
        self.timed.append((t, xy, d))
        span = max(c['stall_s'], c['progress_s'])
        while len(self.timed) > 1 and self.timed[1][0] <= t - span:    # keep one entry at or before the window start
            self.timed.popleft()
        t0 = self.timed[0][0]
        trouble = False
        if c['stall_s'] and t - t0 >= c['stall_s']:
            pts = np.stack([q for s, q, _ in self.timed if s >= t - c['stall_s']])
            trouble |= float(np.max(np.linalg.norm(pts - xy, axis=-1))) < c['stall_disp']
        if c['no_route'] and no_route >= c['no_route']:
            trouble = True
        if c['progress_s'] and t - t0 >= c['progress_s']:
            past = [g for s, _, g in self.timed if s <= t - c['progress_s']][-1]
            trouble |= past - d < c['progress_m']
        if not trouble:
            return c['primary']
        self.n_switch += 1
        self.fresh = True
        self.until = t + c['block_s']
        self.clear()
        return self.secondary
