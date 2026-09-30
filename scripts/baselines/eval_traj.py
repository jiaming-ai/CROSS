#!/usr/bin/env python3
"""Evaluate an external SLAM system's map + query trajectories against ground truth.

Inputs (all camera-to-world, OpenCV convention):
  --map-poses    per-frame poses of the mapping run:   idx state t00 ... t33  (state optional)
  --query-poses  per-frame poses of the query run in the *same* map frame
  --map-seq / --query-seq  SimChange sequence folders (poses_left.txt = GT)
The map trajectory is aligned to GT with SE(3) (or Sim(3) with --sim3 for monocular
systems) using its tracked frames; the same transform is applied to the query poses.
Metrics mirror scripts/map_and_reloc.py: recall at 0.5 m/5 deg, 1 m/5 deg, 2 m/10 deg,
first frame after which the pose stays within 2 m/10 deg for 5 frames, median error.
Frames without a pose (lost / not localized) count as failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reloc_metrics import map_relative_errors, summarize_errors  # noqa: E402


def load_poses(path, n_cols_prefix=2):
    rows = [l.split() for l in Path(path).read_text().strip().splitlines() if l.strip()]
    out = {}
    for r in rows:
        vals = [float(v) for v in r]
        if len(vals) >= 18:   # idx state 16 pose values [extra columns ignored]
            idx, state, T = int(vals[0]), int(vals[1]), np.array(vals[2:18]).reshape(4, 4)
        elif len(vals) == 17:
            idx, state, T = int(vals[0]), 3, np.array(vals[1:]).reshape(4, 4)
        else:
            raise ValueError(f"bad row length {len(vals)} in {path}")
        out[idx] = (state, T)
    return out


def umeyama(src, dst, with_scale=False):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    H = S.T @ D / len(src)
    U, sig, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1, 1, d])
    R = Vt.T @ Dm @ U.T
    s = float(np.sum(sig * np.array([1, 1, d])) / np.mean(np.sum(S ** 2, 1))) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def rot_err_deg(Ra, Rb):
    from reloc_metrics import _proper_rotation
    c = (np.trace(_proper_rotation(Ra).T @ _proper_rotation(Rb)) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def hold_latest(poses, valid_states, frame_range, max_age=10):
    """Causal pose per frame: a frame without its own valid pose takes the system's latest valid pose, if it is at most
    `max_age` frames old (1 s at 10 Hz).  Systems that report poses only at keyframes (MASt3R-SLAM, VGGT-SLAM) or lose
    tracking briefly are then scored like a robot that asks for its current pose.  Held poses are marked `held`."""
    out, last = {}, None
    a, b = frame_range
    for i in range(a, b):
        if i in poses and poses[i][0] in valid_states:
            out[i] = poses[i]
            last = i
        elif last is not None and i - last <= max_age:
            out[i] = (poses[last][0], poses[last][1])
    return out


def evaluate(map_poses, query_poses, gt_map, gt_query, sim3=False, valid_states=(2,), frame_range=None, max_age=10):
    if max_age:
        query_poses = hold_latest(query_poses, valid_states,
                                  frame_range if frame_range is not None else (0, len(gt_query)), max_age)
    ids = [i for i, (st, _) in map_poses.items() if st in valid_states and i < len(gt_map)]
    if len(ids) < 3:
        return None, {"error": "too few tracked map frames", "n_map_tracked": len(ids)}
    src = np.array([map_poses[i][1][:3, 3] for i in ids])
    dst = np.array([gt_map[i][:3, 3] for i in ids])
    s, R, t = umeyama(src, dst, with_scale=sim3)
    map_ate = float(np.sqrt(np.mean(np.sum((s * (R @ src.T).T + t - dst) ** 2, 1))))
    rows = []
    a, b = frame_range if frame_range is not None else (0, len(gt_query))
    for i in range(a, b):
        rec = {"frame": i, "tracked": False, "t_err": np.inf, "r_err": np.inf}
        if i in query_poses and query_poses[i][0] in valid_states:
            T = query_poses[i][1]
            p = s * R @ T[:3, 3] + t
            Rw = R @ T[:3, :3]
            rec.update({"tracked": True, "t_err": float(np.linalg.norm(p - gt_query[i][:3, 3])),
                        "r_err": rot_err_deg(Rw, gt_query[i][:3, :3])})
        rows.append(rec)
    e = np.array([[r["t_err"], r["r_err"]] for r in rows])

    def recall(t, r):
        return float(np.mean((e[:, 0] < t) & (e[:, 1] < r)))

    def first(t, r, hold=5):
        ok = (e[:, 0] < t) & (e[:, 1] < r)
        for k in range(len(ok) - hold + 1):
            if ok[k:k + hold].all():
                return int(k)
        return None

    # map-relative metric (primary): compare with the pose implied by the map's own estimate.  Estimates are first
    # expressed in the ground-truth frame with the map's alignment, so that a Sim(3)-aligned (monocular) system's
    # map-relative errors are in metres too (the transform is rigid for SE(3) and leaves those errors unchanged).
    def to_gt(T):
        out = np.eye(4)
        out[:3, :3] = R @ T[:3, :3]
        out[:3, 3] = s * R @ T[:3, 3] + t
        return out
    meta = {"kf_gt": {str(i): gt_map[i].reshape(-1).tolist() for i in ids},
            "kf_est": {str(i): to_gt(map_poses[i][1]).reshape(-1).tolist() for i in ids}}
    rrows = []
    for rec in rows:
        i = rec["frame"]
        rrows.append({"gt_pose": gt_query[i].reshape(-1).tolist(),
                      "c0_pose": to_gt(query_poses[i][1]).reshape(-1).tolist() if (i in query_poses and query_poses[i][0] in valid_states) else None})
    rel = map_relative_errors(rrows, meta, pose_keys=("c0",))
    for rec, e_ in zip(rows, rel):
        rec.update(e_)
    fin = np.isfinite(e[:, 0])
    summary = {
        "map_relative": summarize_errors(rows, "c0_rel"),
        "n_frames": len(rows), "tracked_frac": float(np.mean(fin)), "map_ate_rmse": map_ate,
        "n_map_tracked": len(ids), "map_frames": len(gt_map), "sim3_scale": s,
        "c0_recall_0.5m_5deg": recall(0.5, 5), "c0_recall_1m_5deg": recall(1.0, 5), "c0_recall_2m_10deg": recall(2.0, 10),
        "c0_first_correct_step_2m": first(2.0, 10), "c0_first_correct_step_1m": first(1.0, 5),
        "c0_t_err_median": float(np.median(e[fin, 0])) if fin.any() else None,
        "c0_r_err_median": float(np.median(e[fin, 1])) if fin.any() else None,
    }
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-poses", required=True)
    ap.add_argument("--query-poses", required=True)
    ap.add_argument("--map-seq", required=True)
    ap.add_argument("--query-seq", required=True)
    ap.add_argument("--sim3", action="store_true")
    ap.add_argument("--valid-states", default="2", help="comma list of state codes counted as localized")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gt_map = np.loadtxt(Path(args.map_seq) / "poses_left.txt").reshape(-1, 4, 4)
    gt_query = np.loadtxt(Path(args.query_seq) / "poses_left.txt").reshape(-1, 4, 4)
    vs = tuple(int(v) for v in args.valid_states.split(","))
    rows, summary = evaluate(load_poses(args.map_poses), load_poses(args.query_poses), gt_map, gt_query,
                             sim3=args.sim3, valid_states=vs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reloc_summary.json").write_text(json.dumps(summary, indent=1))
    if rows is not None:
        (out / "reloc_rows.json").write_text(json.dumps(rows, default=lambda x: None if x == np.inf else float(x)))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
