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


def ate(est: dict, gt: np.ndarray, sim3: bool = False, fps: float = 10.0, window_s: float = 1.0,
        min_completeness: float = 0.8) -> dict:
    """ATE RMSE of `est` (final trajectory, possibly keyframes only) against ground truth.

    completeness: fraction of the sequence's frames within `window_s` of an evaluated pose; below
    `min_completeness` the run counts as a tracking failure."""
    ids = sorted(i for i in est if 0 <= i < len(gt) and np.isfinite(gt[i]).all())
    n = len(gt)
    if len(ids) < 3:
        return {"ate_rmse": None, "completeness": len(ids) / max(n, 1), "n_poses": len(ids), "failed": True,
                "align": "sim3" if sim3 else "se3"}
    src = np.array([est[i][:3, 3] for i in ids])
    dst = np.array([gt[i][:3, 3] for i in ids])
    s, R, t = umeyama(src, dst, with_scale=sim3)
    err = np.linalg.norm(s * (R @ src.T).T + t - dst, axis=1)
    covered = np.zeros(n, bool)
    w = int(round(window_s * fps))
    idx = np.asarray(ids)
    for i in idx:
        covered[max(0, i - w):min(n, i + w + 1)] = True
    comp = float(covered.mean())
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return {"ate_rmse": float(np.sqrt(np.mean(err ** 2))), "ate_median": float(np.median(err)), "ate_max": float(err.max()),
            "completeness": comp, "n_poses": len(ids), "n_frames": n, "scale": s, "align": "sim3" if sim3 else "se3",
            "failed": bool(comp < min_completeness), "T_gt_from_est": T.tolist()}


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
