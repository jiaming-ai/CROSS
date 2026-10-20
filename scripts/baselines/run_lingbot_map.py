#!/usr/bin/env python3
"""Run LingBot-Map (monocular, uncalibrated streaming reconstruction) on a map+query stream and evaluate the query
traversal.

LingBot-Map has no map persistence, so the map traversal and the query traversal (or one trial of it) are concatenated
into one image stream, as for MASt3R-SLAM and VGGT-SLAM 2.0 (run_vggt_slam.py): the model streams through the map
session, then continues into the query frames with its KV cache (anchor / scale frames, the last keyframes, and the
camera tokens of evicted frames as trajectory memory).  It estimates a pose for every frame.  The map part is aligned to
ground truth with Sim(3) (monocular) and the same transform evaluates the query part (see sim3_to_gt).
LingBot-Map has no relocalization or loop-closure step, so nothing tells which query poses it has linked to the map:
every query pose counts (only the protocol's coverage and moving-robot rules apply, as for MASt3R-SLAM without its
edge log), and its T2 / T3 are upper bounds.  A pose right after the jump between the traversals may be only the
stream's continuation; it is still scored against the ground truth.
The model runs through lingbot_map_stream.py (frames read one at a time, upstream model code unmodified).

    python scripts/baselines/run_lingbot_map.py --map <seq> --query <seq> --out <dir> [--map-only] \
        [--trial-len N] [--trial-stride S] [--max-trials K] [--r-d R] [--timeout SEC] [--keyframe-interval auto]

--map-only writes map_poses.txt (idx state t00..t33, camera-to-world, state 2) and map_time.json.
Environment: LINGBOT_MAP_PY (venv python), LINGBOT_MAP_DIR (repo), LINGBOT_MAP_CKPT (lingbot-map.pt; default
<LINGBOT_MAP_DIR>/../ckpt/lingbot-map.pt), LINGBOT_MAP_SDPA=1 for SDPA attention instead of FlashInfer.  Install with
scripts/baselines/install_lingbot_map.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LB_DIR = Path(os.environ.get("LINGBOT_MAP_DIR", ROOT / "third_party/lingbot_map/lingbot-map"))
PY = Path(os.environ.get("LINGBOT_MAP_PY", LB_DIR.parent / "venv/bin/python"))
CKPT = Path(os.environ.get("LINGBOT_MAP_CKPT", LB_DIR.parent / "ckpt/lingbot-map.pt"))
DRIVER = Path(__file__).resolve().parent / "lingbot_map_stream.py"
sys.path.insert(0, str(ROOT / "scripts/baselines"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_traj import evaluate, umeyama  # noqa: E402
from reloc_metrics import build_trials, summarize_errors, summarize_trials  # noqa: E402


def images(seq: Path):
    return sorted((seq / ("left" if (seq / "left").is_dir() else "rgb")).glob("*.png"))


def read_poses(path: Path):
    """lingbot_map_stream.py output: `idx` + camera-to-world 3x4 per frame (idx = stream index)."""
    poses = {}
    if not path.is_file():
        return poses
    for line in path.read_text().splitlines():
        v = line.split()
        if len(v) != 13:
            continue
        r = np.array([float(x) for x in v[1:]])
        if not np.all(np.isfinite(r)):
            continue
        T = np.eye(4)
        T[:3, :4] = r.reshape(3, 4)
        poses[int(v[0])] = T
    return poses


def sim3_to_gt(poses, gt_map, n_map):
    """Express all poses in the ground-truth frame through the Sim(3) alignment of the map part (evaluate() then
    finds an identity alignment), so that the map-relative errors are in metres (as run_vggt_slam.sim3_to_gt)."""
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


def run_stream(frames, work: Path, tag: str, args):
    """Run LingBot-Map on `frames`; returns (rc, seconds, info, poses).  info: peak GPU GB, FPS, keyframe interval."""
    work.mkdir(parents=True, exist_ok=True)
    lst, traj, log = work / f"stream_{tag}.txt", work / f"lingbot_{tag}_poses.txt", work / f"lingbot_{tag}.log"
    lst.write_text("\n".join(str(f) for f in frames) + "\n")
    traj.unlink(missing_ok=True)
    cmd = [str(PY), str(DRIVER), "--checkpoint", str(CKPT), "--images", str(lst), "--out", str(traj),
           "--keyframe-interval", str(args.keyframe_interval)]
    if args.use_sdpa:
        cmd += ["--use-sdpa"]
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               PYTHONPATH=str(LB_DIR) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    t0 = time.time()
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        # bound every run and kill the whole process group on timeout (no GPU memory left behind)
        proc = subprocess.Popen(cmd, cwd=str(LB_DIR), stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=env)
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
    text = log.read_text(errors="replace")
    m = re.findall(r"LINGBOT_PEAK_GPU_GB ([\d.]+) ([\d.]+)", text)
    info = {"peak_gpu_gb": {"allocated": float(m[-1][0]), "reserved": float(m[-1][1])} if m else None}
    for key, pat in (("fps", r"LINGBOT_FPS ([\d.]+)"), ("keyframe_interval", r"LINGBOT_KEYFRAME_INTERVAL (\d+)")):
        m = re.findall(pat, text)
        info[key] = float(m[-1]) if m else None
    poses = read_poses(traj) if rc == 0 else {}      # a killed / crashed run has no trajectory
    lst.unlink(missing_ok=True)
    return rc, dt, info, poses


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
    ap.add_argument("--keyframe-interval", default="auto", help='"auto" (upstream: 1 up to 320 frames, else '
                                                                'ceil(frames / 320)) or a fixed interval')
    ap.add_argument("--use-sdpa", action="store_true", default=os.environ.get("LINGBOT_MAP_SDPA") == "1",
                    help="SDPA attention instead of FlashInfer")
    ap.add_argument("--sim3", action="store_true", help="ignored: monocular, always evaluated with Sim(3)")
    args = ap.parse_args()
    m, q = Path(args.map).resolve(), Path(args.query).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({**vars(args), "lingbot_map_dir": str(LB_DIR), "python": str(PY),
                                               "checkpoint": str(CKPT), "settings": "streaming, 518 px crop, 8 scale "
                                               "frames, KV window 64, max_frame_num 1024, camera iterations 4, bf16 "
                                               "aggregator (lingbot_map_stream.py defaults)",
                                               "link_rule": "none: every query pose counts (upper bound)"}, indent=1))
    lm = images(m)
    n_map = len(lm)
    gt_map = np.loadtxt(m / "poses_left.txt").reshape(-1, 4, 4)

    if args.map_only:
        rc, dt, info, poses = run_stream(lm, out, "map", args)
        with open(out / "map_poses.txt", "w") as f:
            for i in sorted(poses):
                if i < n_map:
                    f.write(f"{i} 2 " + " ".join(f"{v:.9f}" for v in poses[i].reshape(-1)) + "\n")
        info = {"rc": rc, "seconds": dt, "n_frames": n_map, "n_poses": len(poses), **info}
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
        rc, dt, info, poses = run_stream(lm + lq[a:b], out, f"t{ti}", args)    # map traversal, then the trial
        per_trial_time.append({"trial": ti, "rc": rc, "seconds": dt, "start": a, "end": b, "n_poses": len(poses),
                               "map_link": None, **info})
        poses, scale = sim3_to_gt(poses, gt_map, n_map)
        map_poses = {i: (2, T) for i, T in poses.items() if i < n_map}
        query_poses = {i - n_map + a: (2, T) for i, T in poses.items() if i >= n_map}      # no link rule: upper bound
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
               "n_frames": len(all_rows), "times": per_trial_time, "link_rule": "none (upper bound)",
               "n_query_keyframes": int(sum(1 for r in all_rows if r.get("tracked")))}
    (out / "reloc_summary.json").write_text(json.dumps(summary, indent=1))
    (out / "reloc_rows.json").write_text(json.dumps(all_rows, default=lambda x: None))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("trials", "times")}, indent=1))


if __name__ == "__main__":
    main()
