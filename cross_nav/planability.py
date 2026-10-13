"""Explicit goal feasibility and verified planning outcomes, without moving the start.

This protocol is separate from the historical nav visualization metric. Every
candidate goal stays in the audit. The primary denominator is GT-planable goals:
a full robot footprint can follow a known-free route from the exact start.
Unknown labels cannot establish either a collision or free space. A goal is
unplanable only when no route exists even if unknown labels were free.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .bev import FREE, OCC
from .plan import Field, disk, path_length


def footprint_blocked(blocked, res, radius):
    """Conservative raster footprint, with outside-grid space blocked."""
    if res <= 0 or radius < 0:
        raise ValueError('resolution must be positive and radius nonnegative')
    r = math.ceil(radius / res)
    a = np.asarray(blocked, np.uint8)
    return cv2.dilate(a, disk(r), borderType=cv2.BORDER_CONSTANT, borderValue=1) > 0 if r else a.astype(bool)


def classify_goals(gt, start, goals, res, robot_radius=0.2):
    """Return (records, known_free, possibly_free), using prediction-independent GT.

    Eight-connected routes cannot cut blocked corners. Start and goals are never
    snapped. Outside-grid goals are rejected as invalid input, not counted as
    unplanable examples. Distances and the selected candidate list must be fixed
    before comparing methods.
    """
    gt = np.asarray(gt)
    if gt.ndim != 2 or not np.isin(gt, [0, FREE, OCC]).all():
        raise ValueError('GT must be a 2D unknown/free/occupied label grid')
    start = tuple(start)
    goals = [tuple(g) for g in goals]
    for cell in [start, *goals]:
        if len(cell) != 2 or any(int(x) != x for x in cell) or not (0 <= cell[0] < gt.shape[0] and 0 <= cell[1] < gt.shape[1]):
            raise ValueError('start and goals must be in-grid integer cells')
    start = tuple(map(int, start)); goals = [tuple(map(int, g)) for g in goals]
    known = ~footprint_blocked(gt != FREE, res, robot_radius)
    possible = ~footprint_blocked(gt == OCC, res, robot_radius)
    confirmed = Field(known, start, allow_corner_cutting=False)
    optimistic = Field(possible, start, allow_corner_cutting=False)
    rows = []
    for goal in goals:
        path = confirmed.path(goal)
        status = 'planable' if path is not None else ('unknown' if optimistic.path(goal) is not None else 'unplanable')
        rows.append({'start': start, 'goal': goal, 'feasibility': status,
                     'gt_path': path, 'gt_len': path_length(path, res) if path is not None else None})
    return rows, known, possible


def evaluate_goals(gt, maps, start, goals, res, robot_radius=0.2, max_detour=3.0, costs=None,
                   diagonal_cost=False):
    """Audit all supplied candidates and score methods only on GT-planable goals.

    A successful route must have a known-free footprint throughout, no corner
    cutting, and length <= max_detour times the GT shortest route. Unverified
    paths remain in the planable-goal denominator, as unknown_path outcomes.
    This prevents a method from improving its score by abstaining or routing
    through missing labels. No GT is used to construct a method's planned route.
    """
    if max_detour < 1:
        raise ValueError('max_detour must be >= 1')
    costs = costs or {}
    if set(costs) - set(maps):
        raise ValueError('every cost field needs a matching obstacle map')
    rows, known, possible = classify_goals(gt, start, goals, res, robot_radius)
    fields = {}
    for name, obstacles in maps.items():
        if np.asarray(obstacles).shape != np.asarray(gt).shape:
            raise ValueError('method obstacle maps must match GT shape')
        ok = ~footprint_blocked(obstacles, res, robot_radius)
        fields[name] = Field(ok, tuple(map(int, start)), costs.get(name), allow_corner_cutting=False,
                             diagonal_cost=diagonal_cost)
    for row in rows:
        row['methods'] = {}
        if row['feasibility'] != 'planable':
            continue
        for name, field in fields.items():
            path = field.path(row['goal'])
            length = path_length(path, res)
            if path is None:
                status = 'no_path'
            else:
                # Include side cells on diagonal transitions: checking centers
                # alone would miss a GT corner clipped by a predicted route.
                checked = list(path)
                for a, b in zip(path, path[1:]):
                    if a[0] != b[0] and a[1] != b[1]:
                        checked.extend([(a[0], b[1]), (b[0], a[1])])
                rr, cc = np.asarray(checked).T
                if not possible[rr, cc].all(): status = 'collision'
                elif not known[rr, cc].all(): status = 'unknown_path'
                elif length > max_detour * max(row['gt_len'], res): status = 'detour'
                else: status = 'success'
            row['methods'][name] = {'status': status, 'path': path, 'len': length,
                                    'spl': row['gt_len'] / max(row['gt_len'], length, res) if status == 'success' else 0.0}
    return rows


def summarize_goals(rows, method):
    counts = {k: sum(r['feasibility'] == k for r in rows) for k in ['planable', 'unplanable', 'unknown']}
    eligible = [r['methods'][method] for r in rows if r['feasibility'] == 'planable']
    outcomes = {k: sum(r['status'] == k for r in eligible)
                for k in ['success', 'collision', 'no_path', 'unknown_path', 'detour']}
    return {'candidates': len(rows), **counts, 'outcomes': outcomes,
            'success_rate': outcomes['success'] / len(eligible) if eligible else None,
            'spl': float(np.mean([r['spl'] for r in eligible])) if eligible else None}
