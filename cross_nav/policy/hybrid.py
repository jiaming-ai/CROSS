"""Hybrid navigation: the BEV planner and the direct trajectory head, switched per step from odometry and BEV state.

The two systems fail on different episodes (HSSD test: either succeeds on 99.4% of episodes, each alone 94-96%), so
one of them (`primary`) drives and the other takes over for `block` planning steps when the primary is in trouble:

* stall      the robot moved less than `stall_disp` m from where it was `stall_window` steps ago (pushing, spinning)
* no_route   the BEV planner had no route for `no_route` consecutive windows (turning in place)
* progress   the straight-line goal distance shrank by less than `progress_m` over `progress_window` steps (wandering)

A trigger set to 0 is off. After the block, control returns to the primary with empty trigger histories. The BEV
planner runs on every step either way, so its episode map and bump memory stay current; the trajectory head keeps
its own stuck recovery (recovery.py), reset at each hand-over.

    switch = Switch(cfg)                                  # one per episode
    ctrl = switch(pose, goal_rel, bev_no_route)           # 'bev' or 'traj' for this step
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
}


class Switch:
    def __init__(self, cfg=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        if self.cfg['primary'] not in ('bev', 'traj'):
            raise ValueError("primary must be 'bev' or 'traj'")
        self.secondary = 'traj' if self.cfg['primary'] == 'bev' else 'bev'
        self.left, self.n_switch, self.fresh = 0, 0, False
        self.poses = deque(maxlen=max(1, self.cfg['stall_window']))
        self.dist = deque(maxlen=max(1, self.cfg['progress_window']) + 1)

    def clear(self):
        self.poses.clear(); self.dist.clear()

    def __call__(self, pose, goal_rel, no_route=0):
        """pose (x, y, yaw) odometry, goal_rel (2,) goal in the robot frame, no_route the BEV planner's current no-route
        streak. Returns the controller for this step; self.fresh is True on the first step of a hand-over."""
        c = self.cfg
        self.fresh = False
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
