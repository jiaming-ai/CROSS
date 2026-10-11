"""Training-only mining of labelled obstacles crossed by predicted routes."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .plan import Field, disk, risk_cost
from .planability import footprint_blocked


def critical_obstacles(prob, bev, res=.05, radius=.2, max_goals=32, seed=0, goals=None):
    """Return GT occupied cells touched by prediction-only routes to feasible goals.

    GT selects training goals with a known-free alternative and supplies targets.
    It never changes the predicted route's cost or traversability. Unknown cells
    are never positive targets. Goal sampling has a private RNG so mining does
    not perturb the training data or augmentation random streams.
    """
    prob, bev = np.asarray(prob), np.asarray(bev)
    if prob.shape != bev.shape or bev.ndim != 2 or not np.isin(bev, [0, 1, 2]).all():
        raise ValueError('prob and labels must be matching 2D grids')
    if max_goals < 1:
        raise ValueError('max_goals must be positive')
    known = ~footprint_blocked(bev != 1, res, radius)
    start = (bev.shape[0] // 2, bev.shape[1] // 2)
    gt_field = Field(known, start, allow_corner_cutting=False)
    empty = np.zeros_like(bev, bool)
    if not gt_field.valid:
        return empty, 0
    if goals is None:
        rr, cc = np.indices(bev.shape)
        distance = np.hypot(rr - start[0], cc - start[1]) * res
        candidates = np.argwhere(np.isfinite(gt_field.dist) & (distance >= 2) & (distance <= 8))
    else:
        candidates = np.asarray([g for g in goals if gt_field.path(tuple(g)) is not None], dtype=int)
    if not len(candidates):
        return empty, 0
    rng = np.random.default_rng(seed)
    selected = candidates[rng.choice(len(candidates), min(max_goals, len(candidates)), replace=False)]
    field = Field(~footprint_blocked(empty, res, radius), start,
                  risk_cost(prob, res, robot_radius=radius, weight=64),
                  allow_corner_cutting=False, diagonal_cost=True)
    touched = np.zeros_like(bev, np.uint8)
    for goal in selected:
        path = field.path(tuple(goal))
        if path is None:
            continue
        checked = list(path)
        for a, b in zip(path, path[1:]):
            if a[0] != b[0] and a[1] != b[1]:
                checked.extend([(a[0], b[1]), (b[0], a[1])])
        rr, cc = np.asarray(checked).T
        touched[rr, cc] = 1
    # Match the strict checker's footprint, including diagonal side cells.
    radius_cells = int(np.ceil(radius / res))
    swept = cv2.dilate(touched, disk(radius_cells)) if radius_cells else touched
    return (swept > 0) & (bev == 2), len(selected)


def route_bce(logits, bev, res=.05, radius=.2, max_goals=32, seed=0):
    """Extra positive BCE, averaged over the batch; examples without hits give zero.

    Routes/masks are detached CPU computations. Gradients only pass through the
    logits at the selected, known occupied cells. Base occupancy loss is retained
    by the caller to penalize false obstacles elsewhere.
    """
    probabilities = logits.detach().float().sigmoid().cpu().numpy()
    labels = bev.detach().cpu().numpy()
    losses, pixels, goals = [], 0, 0
    for j, (p, y) in enumerate(zip(probabilities, labels)):
        mask, n = critical_obstacles(p, y, res, radius, max_goals, seed + j)
        m = torch.as_tensor(mask, device=logits.device)
        losses.append(F.softplus(-logits[j].float()[m]).mean() if mask.any()
                      else logits[j].float().sum() * 0)
        pixels += int(mask.sum())
        goals += n
    loss = torch.stack(losses).mean()
    return loss, {'loss_route': loss.detach(), 'route_pixels': pixels, 'route_goals': goals}
