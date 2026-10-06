"""Metrics of the CROSS benchmark (benchmark/PROTOCOL.md, section 4).

Trajectories are dicts {frame index: 4x4 camera-to-world}; ground truth is an (N, 4, 4) array indexed by frame.
"""
from __future__ import annotations

import math

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False):
    """Least-squares s, R, t with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    H = S.T @ D / len(src)
    U, sig, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1.0, 1.0, d])
    R = Vt.T @ Dm @ U.T
    s = float(np.sum(sig * np.array([1.0, 1.0, d])) / max(np.mean(np.sum(S ** 2, 1)), 1e-12)) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


COMPLETENESS_RULE = "time-or-path"      # recorded in every T1 result (PROTOCOL.md section 4, T1)


def completeness(frames, gt: np.ndarray, fps: float = 10.0, window_s: float = 1.0, path_m: float = 1.0) -> float:
    """Fraction of the sequence's frames covered by the poses at `frames` (frame indices of one map).

    A frame is covered when one of these poses lies within `window_s` of it, or within `path_m` of it along the
    travelled ground-truth path (arc length).  The time window catches lost tracking; the path distance tolerates
    sparse keyframes of a slow or standing robot.  Path distance, not straight-line distance: on a revisit the poses
    of the first pass are close in space but a whole loop away along the path, so a session that lost tracking on
    its second pass is still flagged."""
    n = len(gt)
    ids = np.unique(np.asarray([i for i in frames if 0 <= i < n], dtype=int))
    if n == 0 or len(ids) == 0:
        return 0.0
    covered = np.zeros(n, bool)
    w = int(round(window_s * fps))
    for i in ids:
        covered[max(0, i - w):min(n, i + w + 1)] = True
    if path_m > 0:
        p = gt[:, :3, 3]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(step), step, 0.0))])
        k = np.searchsorted(ids, np.arange(n))           # nearest pose before / after each frame along the path
        before = np.abs(s - s[ids[np.clip(k - 1, 0, len(ids) - 1)]])
        after = np.abs(s - s[ids[np.clip(k, 0, len(ids) - 1)]])
        covered |= np.minimum(before, after) <= path_m
    return float(covered.mean())


def ate(est: dict, gt: np.ndarray, sim3: bool = False, fps: float = 10.0, window_s: float = 1.0,
        min_completeness: float = 0.8, path_m: float = 1.0, cover_frames=None) -> dict:
    """ATE RMSE of `est` (final trajectory, possibly keyframes only) against ground truth.

    completeness (see completeness()): fraction of the sequence's frames within `window_s` or `path_m` of travelled
    path of an evaluated pose (or of `cover_frames`, when the system holds poses it does not export for the ATE, such
    as CROSS's temporary keyframes on revisits); below `min_completeness` the run counts as a tracking failure."""
    ids = sorted(i for i in est if 0 <= i < len(gt) and np.isfinite(gt[i]).all())
    n = len(gt)
    cov_ids = ids if cover_frames is None else sorted(set(ids) | {int(i) for i in cover_frames})
    comp = completeness(cov_ids, gt, fps, window_s, path_m)
    if len(ids) < 3:
        return {"ate_rmse": None, "completeness": comp, "n_poses": len(ids), "failed": True,
                "align": "sim3" if sim3 else "se3", "completeness_rule": COMPLETENESS_RULE}
    src = np.array([est[i][:3, 3] for i in ids])
    dst = np.array([gt[i][:3, 3] for i in ids])
    s, R, t = umeyama(src, dst, with_scale=sim3)
    err = np.linalg.norm(s * (R @ src.T).T + t - dst, axis=1)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return {"ate_rmse": float(np.sqrt(np.mean(err ** 2))), "ate_median": float(np.median(err)), "ate_max": float(err.max()),
            "completeness": comp, "n_poses": len(ids), "n_frames": n, "scale": s, "align": "sim3" if sim3 else "se3",
            "failed": bool(comp < min_completeness), "completeness_rule": COMPLETENESS_RULE, "T_gt_from_est": T.tolist()}


def first_stable(ok: np.ndarray, hold: int = 5):
    for k in range(len(ok) - hold + 1):
        if ok[k:k + hold].all():
            return int(k)
    return None


def multisession(errors: np.ndarray, thresholds=(0.5, 1.0), loc_threshold: float = 1.0) -> dict:
    """errors: per query frame position error in the map frame after the map session's alignment (inf = no estimate)."""
    e = np.asarray(errors, dtype=float)
    fin = np.isfinite(e)
    out = {"n_frames": int(len(e)), "est_frac": float(fin.mean()) if len(e) else 0.0,
           "ms_ate": float(np.sqrt(np.mean(e[fin] ** 2))) if fin.any() else None,
           "ms_median": float(np.median(e[fin])) if fin.any() else None,
           "time_to_localize": first_stable(e < loc_threshold)}
    for x in thresholds:
        out[f"lr@{x:g}"] = float(np.mean(e < x)) if len(e) else 0.0
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))
