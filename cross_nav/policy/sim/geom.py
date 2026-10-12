"""Planar geometry of a ground robot in Habitat's Y-up world, numpy only.

World plane coordinates are (x, z). Yaw psi rotates about +Y; psi = 0 faces -Z.
forward = (-sin psi, -cos psi), left = (-cos psi, sin psi) in (x, z). Policy paths are metres (forward, left).
"""
from __future__ import annotations

import math

import numpy as np

HISTORY_M = (0., .5, 1., 2.)


def axes(yaw):
    s, c = math.sin(yaw), math.cos(yaw)
    return np.array([-s, -c]), np.array([-c, s])


def to_robot(points_xz, pos_xz, yaw):
    f, l = axes(yaw)
    d = np.asarray(points_xz, dtype=np.float64) - np.asarray(pos_xz, dtype=np.float64)
    return np.stack((d @ f, d @ l), -1)


def to_world(points_fl, pos_xz, yaw):
    f, l = axes(yaw)
    p = np.asarray(points_fl, dtype=np.float64)
    return np.asarray(pos_xz, dtype=np.float64) + p[..., :1] * f + p[..., 1:] * l


def resample(polyline, horizon=16, spacing=.5):
    """Points at arc length spacing*1..horizon along a polyline starting at the robot; past its end, the end point."""
    p = np.asarray(polyline, dtype=np.float64)
    if len(p) == 1:
        return np.repeat(p, horizon, 0)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=-1))]
    s = np.minimum(np.arange(1, horizon + 1) * spacing, arc[-1])
    return np.stack([np.interp(s, arc, p[:, i]) for i in range(2)], -1)


def history_ids(cumdist, history=HISTORY_M):
    """Latest frame at least h metres of travel before the current one (frame 0 when the episode is shorter)."""
    c = np.asarray(cumdist, dtype=np.float64)
    d = c[-1]
    ids = [max(0, int(np.searchsorted(c, d - h + 1e-9, side='right')) - 1) for h in history]
    ids[0] = len(c) - 1
    return ids


def control(path, lookahead=.5, step=.25, max_turn=math.radians(45), turn_only=math.radians(60), align=math.radians(30)):
    """One motion command (turn radians, forward metres) that follows a (forward, left) path.

    Aim at the path point `lookahead` metres along it (the end when shorter); turn at most `max_turn`, then move
    forward, scaled down linearly to zero as the heading error left after the turn reaches `align` (a differential
    drive rotating before it drives; moving while misaligned cuts corners into obstacles). No forward motion when the
    aim point is more than `turn_only` off the heading. The step never exceeds the remaining path length, so the robot
    does not overshoot a goal at the path end.
    """
    p = np.vstack(([0., 0.], np.asarray(path, dtype=np.float64)))
    seg = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    total = float(seg.sum())
    if total < 1e-6:
        return 0., 0.
    arc = np.r_[0., np.cumsum(seg)]
    s = min(lookahead, total)
    aim = np.array([np.interp(s, arc, p[:, i]) for i in range(2)])
    theta = math.atan2(aim[1], aim[0])
    turn = max(-max_turn, min(max_turn, theta))
    if abs(theta) > turn_only:
        return turn, 0.
    forward = min(step, total, float(np.linalg.norm(aim))) * max(0., 1. - abs(theta - turn) / align)
    return turn, forward


def spl(success, shortest, travelled):
    return float(success) * shortest / max(shortest, travelled, 1e-9)
