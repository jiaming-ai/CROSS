"""Stuck recovery for the direct trajectory policy, from odometry alone (same code in simulation and on a robot).

A policy that keeps predicting a path into a wall deadlocks: the robot slides or stops, the views barely change, and
the next prediction is the same. When the robot has moved less than `min_disp` metres from where it was `window`
planning steps ago, the recovery replaces the policy's path with an in-place turn of `turn_deg`, alternating left and
right on successive recoveries, then hands control back to the policy (with fresh views of a new heading).
`window` counts planning steps: in simulation one step moves up to 0.25 m (20 steps = up to 5 m of commanded motion).
On a robot planning at camera rate, give `window_s` (seconds; ~10 s = 5 m at 0.5 m/s) and pass time stamps: the
window then spans that time, whatever the planning rate (a step count would also fire during an initial turn).
Validated on HSSD val240 (trajectory head r1 trail, window 20): 92.1% vs 90.0% success, wall deadlocks 2 vs 8.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


class StuckRecovery:
    def __init__(self, window=20, min_disp=.3, turn_deg=90., max_turn_steps=20, horizon=16, spacing=.5, window_s=None):
        self.window, self.min_disp, self.turn, self.max_turn_steps = window, min_disp, math.radians(turn_deg), max_turn_steps
        self.window_s = window_s
        self.horizon, self.spacing = horizon, spacing
        self.reset()

    def reset(self):
        self.hist, self.times, self.t0, self.turning, self.count = deque(maxlen=self.window), deque(), 0., None, 0

    def turn_path(self, sign):
        """A path straight to the robot's side: a waypoint follower turns in place toward it."""
        s = np.arange(1, self.horizon + 1) * self.spacing
        return np.stack((np.zeros_like(s), sign * s), -1)

    def __call__(self, pose, path, stamp=None):
        """pose (x, y, yaw) odometry of the current frame; path (16, 2) the policy's path; stamp: time in seconds (used
        with window_s). Returns the path to follow."""
        if self.turning is not None:
            start, sign, k = self.turning
            if abs((pose[2] - start + math.pi) % (2 * math.pi) - math.pi) < self.turn - math.radians(5) and k < self.max_turn_steps:
                self.turning = (start, sign, k + 1)
                return self.turn_path(sign)
            self.turning = None
            self.hist.clear(); self.times.clear()
        xy = np.asarray(pose[:2], np.float64)
        if self.window_s is not None and stamp is not None:      # time window: positions of the last window_s seconds
            if not self.times:
                self.t0 = float(stamp)
            self.times.append((float(stamp), xy))
            while self.times[0][0] < stamp - self.window_s:
                self.times.popleft()
            full, pts = stamp - self.t0 >= self.window_s, np.stack([q for _, q in self.times])
        else:                                                     # step window: the last `window` planning steps
            self.hist.append(xy)
            full, pts = len(self.hist) == self.window, np.stack(self.hist)
        if full and float(np.max(np.linalg.norm(pts - xy, axis=-1))) < self.min_disp:
            self.count += 1
            sign = 1. if self.count % 2 else -1.
            self.turning = (pose[2], sign, 1)
            self.hist.clear(); self.times.clear()
            return self.turn_path(sign)
        return path
