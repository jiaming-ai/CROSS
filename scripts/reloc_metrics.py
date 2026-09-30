"""Localization metrics relative to the map (shared by CROSS and external-baseline evaluation).

A map built with noisy odometry and no loop closure drifts; a system that localizes
perfectly *within that map* would still show metre-level error against ground truth.
The map-relative error therefore compares the estimate with the pose that the map
itself implies for the query frame: take the map keyframe k closest (in ground truth) to
the query frame i, and predict E_i* = M_k · G_k^-1 · G_i, where M_k is the map's
estimated pose of k and G the ground truth.  The error is E_i*^-1 · E_i.
"""

from __future__ import annotations

import numpy as np


def _inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def _proper_rotation(R):
    """Nearest rotation matrix (SVD projection).  Poses that went through many float32 quaternion
    compositions are slightly scaled / non-orthogonal; the trace formula then reports several degrees
    of error for a matrix that is 0.1 deg from the ground truth."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def _rot_deg(R):
    R = _proper_rotation(R)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def map_relative_errors(rows, meta, pose_keys=("c0", "best"), max_kf_dist=None):
    """rows: dicts with 'gt_pose' and '<key>_pose' (16 floats, map frame). meta: kf_gt / kf_est dicts."""
    ids = [k for k in meta["kf_est"] if str(k) in meta["kf_gt"] or k in meta["kf_gt"]]
    G = np.array([np.asarray(meta["kf_gt"][str(k)] if str(k) in meta["kf_gt"] else meta["kf_gt"][k]).reshape(4, 4) for k in ids])
    M = np.array([np.asarray(meta["kf_est"][k]).reshape(4, 4) for k in ids])
    for k in range(len(M)):
        M[k, :3, :3] = _proper_rotation(M[k, :3, :3])
    out = []
    for r in rows:
        g = np.asarray(r["gt_pose"]).reshape(4, 4)
        j = int(np.argmin(np.linalg.norm(G[:, :3, 3] - g[:3, 3], axis=1)))
        pred = M[j] @ _inv(G[j]) @ g
        pred[:3, :3] = _proper_rotation(pred[:3, :3])
        rec = {"nearest_kf_dist": float(np.linalg.norm(G[j, :3, 3] - g[:3, 3]))}
        for key in pose_keys:
            if f"{key}_pose" not in r or r[f"{key}_pose"] is None:
                rec[f"{key}_rel_t_err"] = np.inf
                rec[f"{key}_rel_r_err"] = np.inf
                continue
            E = np.asarray(r[f"{key}_pose"]).reshape(4, 4)
            err = _inv(pred) @ E
            rec[f"{key}_rel_t_err"] = float(np.linalg.norm(err[:3, 3]))
            rec[f"{key}_rel_r_err"] = _rot_deg(err[:3, :3])
        out.append(rec)
    return out


def summarize_errors(rows, prefix):
    e = np.array([[r.get(f"{prefix}_t_err", np.inf), r.get(f"{prefix}_r_err", np.inf)] for r in rows], dtype=float)

    def recall(t, rr):
        return float(np.mean((e[:, 0] < t) & (e[:, 1] < rr)))

    def first(t, rr, hold=5):
        ok = (e[:, 0] < t) & (e[:, 1] < rr)
        for k in range(len(ok) - hold + 1):
            if ok[k:k + hold].all():
                return int(k)
        return None

    fin = np.isfinite(e[:, 0])
    return {
        "recall_0.5m_5deg": recall(0.5, 5), "recall_1m_5deg": recall(1.0, 5), "recall_2m_10deg": recall(2.0, 10),
        "first_correct_step_1m": first(1.0, 5), "first_correct_step_2m": first(2.0, 10),
        "t_err_median": float(np.median(e[fin, 0])) if fin.any() else None,
        "r_err_median": float(np.median(e[fin, 1])) if fin.any() else None,
        "tracked_frac": float(fin.mean()),
    }


def build_trials(n_frames, trial_len, trial_stride=None, min_len=None):
    """Fixed-length relocalization trials over a query sequence (CROSS paper protocol: independent
    sub-sequences, here optionally overlapping via `trial_stride`)."""
    if not trial_len or trial_len <= 0 or trial_len >= n_frames:
        return [(0, n_frames)]
    stride = trial_stride or trial_len
    min_len = min_len or max(10, trial_len // 2)
    trials = []
    s = 0
    while s < n_frames:
        e = min(s + trial_len, n_frames)
        if e - s >= min_len:
            trials.append((s, e))
        if e == n_frames:
            break
        s += stride
    return trials


def summarize_trials(rows, prefix="c0_rel", r_d=2.0, max_age=10):
    """Relocalization success over trials: a trial succeeds if the *final* estimate of the trial is within
    r_d of the ground truth (position only, as in the CROSS paper); stricter pose variants are also reported.
    The final estimate is the system's latest pose in the trial, if it is at most `max_age` frames old (1 s at
    10 Hz): systems that only report keyframe poses have no pose on most frames.  rows need 'trial' and
    '<prefix>_t_err' / '<prefix>_r_err'."""
    trials = sorted(set(r["trial"] for r in rows))
    out = {"n_trials": len(trials), "r_d": r_d, "trials": []}
    succ_rd, succ_1m, succ_05 = [], [], []
    first_ok = []
    for t in trials:
        tr = [r for r in rows if r["trial"] == t]
        e_t = np.array([r.get(f"{prefix}_t_err", np.inf) for r in tr], dtype=float)
        e_r = np.array([r.get(f"{prefix}_r_err", np.inf) for r in tr], dtype=float)
        last = np.where(np.isfinite(e_t))[0]
        last = last[-1] if len(last) and len(e_t) - 1 - last[-1] <= max_age else len(e_t) - 1
        fin_t, fin_r = float(e_t[last]), float(e_r[last])
        s_rd = bool(fin_t < r_d)
        s_1 = bool(fin_t < 1.0 and fin_r < 5.0)
        s_05 = bool(fin_t < 0.5 and fin_r < 5.0)
        ok = np.where((e_t < 1.0) & (e_r < 5.0))[0]
        first = int(ok[0]) if len(ok) else None
        succ_rd.append(s_rd); succ_1m.append(s_1); succ_05.append(s_05); first_ok.append(first)
        out["trials"].append({"trial": int(t), "start": int(tr[0].get("frame", 0)), "n": len(tr), "final_t_err": fin_t, "final_r_err": fin_r,
                              "success_rd": s_rd, "success_1m_5deg": s_1, "success_0.5m_5deg": s_05, "first_ok_1m": first})
    out["RS"] = float(np.mean(succ_rd)) if trials else None
    out["RS_1m_5deg"] = float(np.mean(succ_1m)) if trials else None
    out["RS_0.5m_5deg"] = float(np.mean(succ_05)) if trials else None
    ft = [f for f in first_ok if f is not None]
    out["median_steps_to_1m"] = float(np.median(ft)) if ft else None
    out["frac_trials_reaching_1m"] = float(len(ft) / len(trials)) if trials else None
    return out
