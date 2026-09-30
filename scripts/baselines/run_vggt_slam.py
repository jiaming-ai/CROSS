#!/usr/bin/env python3
"""Run VGGT-SLAM 2.0 (monocular, uncalibrated) on a map+query stream and evaluate the query traversal.

VGGT-SLAM has no map persistence, so the map traversal and the query traversal (or one trial of it) are
concatenated into one image stream, exactly as for MASt3R-SLAM (run_mast3r_slam.py): the system first maps the
scene, then has to attach the second traversal to it through its loop closures.  VGGT-SLAM only processes
keyframes (frames with enough optical-flow disparity) and logs their final poses after pose-graph optimization,
so frames that are not keyframes have no pose (counted as unlocalized).  The map part is aligned to ground
truth with Sim(3) (monocular) and the same transform evaluates the query part; all poses are expressed in the
ground-truth frame before evaluation so the map-relative errors are in metres (see sim3_to_gt).
VGGT-SLAM runs through vggt_slam_headless.py (viser viewer stubbed out, upstream code unmodified).

    python scripts/baselines/run_vggt_slam.py --map <seq> --query <seq> --out <dir> [--map-only] \
        [--trial-len N] [--trial-stride S] [--max-trials K] [--r-d R] [--timeout SEC] [--submap-size 32] [--max-loops 1]

--map-only writes map_poses.txt (idx state t00..t33, camera-to-world, state 2) and map_time.json.
Environment: VGGT_SLAM_PY (venv python), VGGT_SLAM_DIR (repo), VGGT_SLAM_TORCH_HOME (weights; default
<VGGT_SLAM_DIR>/../torch_home).  Install with scripts/baselines/install_vggt_slam.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
VS_DIR = Path(os.environ.get("VGGT_SLAM_DIR", "/home/storage/jiaming/vggt_slam/VGGT-SLAM"))
PY = Path(os.environ.get("VGGT_SLAM_PY", "/home/storage/jiaming/vggt_slam/.venv/bin/python"))
TORCH_HOME = Path(os.environ.get("VGGT_SLAM_TORCH_HOME", VS_DIR.parent / "torch_home"))
HEADLESS = Path(__file__).resolve().parent / "vggt_slam_headless.py"
sys.path.insert(0, str(ROOT / "scripts/baselines"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_traj import evaluate, umeyama  # noqa: E402
from reloc_metrics import build_trials, summarize_trials, summarize_errors  # noqa: E402


def images(seq: Path):
    return sorted((seq / ("left" if (seq / "left").is_dir() else "rgb")).glob("*.png"))


def read_traj(path: Path):
    """VGGT-SLAM pose log (GraphMap.write_poses_to_file, TUM-style): `frame_id x y z qx qy qz qw` per keyframe,
    camera-to-world, submaps in order.  frame_id is the number in the image file name (= stream index here).
    The overlap frame shared by consecutive submaps is logged twice; the first occurrence is kept."""
    poses = {}
    if not path.is_file():
        return poses
    for line in path.read_text().splitlines():
        v = line.split()
        if len(v) < 8:
            continue
        r = np.array([float(x) for x in v[:8]])
        if not np.all(np.isfinite(r)) or np.linalg.norm(r[4:8]) < 1e-9:
            continue
        idx = int(round(r[0]))
        if idx in poses:
            continue
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(r[4:8] / np.linalg.norm(r[4:8])).as_matrix()
        T[:3, 3] = r[1:4]
        poses[idx] = T
    return poses


def sim3_to_gt(poses, gt_map, n_map):
    """Express all poses in the ground-truth frame through the Sim(3) alignment of the map part (evaluate() then
    finds an identity alignment).  Needed because eval_traj.evaluate() applies its Sim(3) only to the absolute
    errors: the map-relative errors (c0_rel_*) compose the raw estimates with metric ground-truth offsets, which
    for an arbitrary-scale monocular trajectory mixes units.  After this step they are in metres."""
    ids = [i for i in sorted(poses) if i < min(n_map, len(gt_map))]
    if len(ids) < 3:
        return poses, None
    s, R, t = umeyama(np.array([poses[i][:3, 3] for i in ids]), np.array([gt_map[i][:3, 3] for i in ids]), with_scale=True)
    out = {}
    for i, T in poses.items():
        A = np.eye(4)
        A[:3, :3] = R @ T[:3, :3]
        A[:3, 3] = s * R @ T[:3, 3] + t
        out[i] = A
    return out, float(s)


def run_stream(frames, stream: Path, log: Path, traj: Path, args):
    """Run VGGT-SLAM on `frames` (symlinked as 000000.png ...); returns (rc, seconds, peak GPU GB, poses)."""
    shutil.rmtree(stream, ignore_errors=True)
    stream.mkdir(parents=True)
    for i, f in enumerate(frames):
        os.symlink(f, stream / f"{i:06d}.png")
    traj.unlink(missing_ok=True)
    cmd = [str(PY), str(HEADLESS), str(VS_DIR), "--image_folder", str(stream), "--max_loops", str(args.max_loops),
           "--submap_size", str(args.submap_size), "--log_results", "--skip_dense_log", "--log_path", str(traj)]
    env = dict(os.environ, TORCH_HOME=str(TORCH_HOME), PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        # bound every run and kill the whole process group on timeout (no GPU memory left behind)
        proc = subprocess.Popen(cmd, cwd=str(VS_DIR), stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        try:
            rc = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            rc = -9
            f.write("\nTIMEOUT\n")
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    dt = time.time() - t0
    m = re.findall(r"VGGT_SLAM_PEAK_GPU_GB ([\d.]+) ([\d.]+)", log.read_text(errors="replace"))
    peak = {"allocated": float(m[-1][0]), "reserved": float(m[-1][1])} if m else None
    poses = read_traj(traj) if rc == 0 else {}      # a killed / crashed run has no final trajectory
    shutil.rmtree(stream, ignore_errors=True)
    return rc, dt, peak, poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map-only", action="store_true", help="run on the map sequence alone (single-session accuracy)")
    ap.add_argument("--trial-len", type=int, default=0)
    ap.add_argument("--trial-stride", type=int, default=None)
    ap.add_argument("--max-trials", type=int, default=0, help="evaluate at most this many evenly spaced trials (0: all)")
    ap.add_argument("--r-d", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds per run before it is killed")
    ap.add_argument("--submap-size", type=int, default=32, help="VGGT-SLAM --submap_size (new keyframes per submap)")
    ap.add_argument("--max-loops", type=int, default=1, help="VGGT-SLAM --max_loops (1 = default, 0 disables loop closure)")
    ap.add_argument("--sim3", action="store_true", help="ignored: monocular, always evaluated with Sim(3)")
    args = ap.parse_args()
    m, q = Path(args.map).resolve(), Path(args.query).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({**vars(args), "vggt_slam_dir": str(VS_DIR), "python": str(PY)}, indent=1))
    lm = images(m)
    n_map = len(lm)
    gt_map = np.loadtxt(m / "poses_left.txt").reshape(-1, 4, 4)

    if args.map_only:
        rc, dt, peak, poses = run_stream(lm, out / "stream_map", out / "vggt_map.log", out / "vggt_map_traj.txt", args)
        with open(out / "map_poses.txt", "w") as f:
            for i in sorted(poses):
                if i < n_map:
                    f.write(f"{i} 2 " + " ".join(f"{v:.9f}" for v in poses[i].reshape(-1)) + "\n")
        info = {"rc": rc, "seconds": dt, "n_frames": n_map, "n_poses": len(poses), "peak_gpu_gb": peak}
        ids = [i for i in sorted(poses) if i < len(gt_map)]
        if len(ids) >= 3:
            src = np.array([poses[i][:3, 3] for i in ids])
            dst = np.array([gt_map[i][:3, 3] for i in ids])
            s, R, t = umeyama(src, dst, with_scale=True)
            info["map_ate_rmse_sim3"] = float(np.sqrt(np.mean(np.sum((s * (R @ src.T).T + t - dst) ** 2, 1))))
            info["sim3_scale"] = s
        (out / "map_time.json").write_text(json.dumps(info))
        print(json.dumps(info, indent=1))
        return

    lq = images(q)
    gt_query = np.loadtxt(q / "poses_left.txt").reshape(-1, 4, 4)
    trials = build_trials(len(lq), args.trial_len, args.trial_stride)
    if args.max_trials and len(trials) > args.max_trials:
        trials = [trials[i] for i in np.linspace(0, len(trials) - 1, args.max_trials).round().astype(int)]
    all_rows, per_trial_time = [], []
    for ti, (a, b) in enumerate(trials):
        rc, dt, peak, poses = run_stream(lm + lq[a:b], out / f"stream_t{ti}", out / f"vggt_t{ti}.log",
                                         out / f"vggt_t{ti}_traj.txt", args)   # map traversal, then the trial
        per_trial_time.append({"trial": ti, "rc": rc, "seconds": dt, "start": a, "end": b, "n_poses": len(poses),
                               "peak_gpu_gb": peak})
        poses, scale = sim3_to_gt(poses, gt_map, n_map)
        map_poses = {i: (2, T) for i, T in poses.items() if i < n_map}
        query_poses = {i - n_map + a: (2, T) for i, T in poses.items() if i >= n_map}
        rws, summ = evaluate(map_poses, query_poses, gt_map, gt_query, sim3=True, frame_range=(a, b))
        if rws is None:
            rws = [{"frame": i, "trial": ti, "tracked": False, "c0_rel_t_err": np.inf, "c0_rel_r_err": np.inf,
                    "t_err": np.inf, "r_err": np.inf} for i in range(a, b)]
        else:
            per_trial_time[-1].update({"map_ate_rmse": summ["map_ate_rmse"], "sim3_scale": scale,
                                       "n_map_tracked": summ["n_map_tracked"]})
            print(f"trial {ti} [{a},{b}): rc={rc} {dt:.0f}s map_ate_rmse={summ['map_ate_rmse']:.3f} "
                  f"query poses={len(query_poses)}", flush=True)
        for r in rws:
            r["trial"] = ti
        all_rows.extend(rws)
    trials_summary = summarize_trials(all_rows, "c0_rel", r_d=args.r_d)
    summary = {"trials": trials_summary, "RS": trials_summary["RS"], "RS_1m_5deg": trials_summary["RS_1m_5deg"],
               "RS_0.5m_5deg": trials_summary["RS_0.5m_5deg"], "map_relative": summarize_errors(all_rows, "c0_rel"),
               "n_frames": len(all_rows), "times": per_trial_time,
               "n_query_keyframes": int(sum(1 for r in all_rows if r.get("tracked")))}
    (out / "reloc_summary.json").write_text(json.dumps(summary, indent=1))
    (out / "reloc_rows.json").write_text(json.dumps(all_rows, default=lambda x: None))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("trials", "times")}, indent=1))


if __name__ == "__main__":
    main()
