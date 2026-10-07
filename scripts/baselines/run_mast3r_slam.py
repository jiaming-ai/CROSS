#!/usr/bin/env python3
"""Run MASt3R-SLAM (monocular) on a map+query stream and evaluate the query traversal.

MASt3R-SLAM has no map persistence, so the map traversal and the query traversal are
concatenated into one image stream (left images only): the system first maps the scene,
then has to register the second traversal to the first (loop closure / relocalization).
Keyframe poses are saved by MASt3R-SLAM; frames without a keyframe are counted as
unlocalized in the recall metrics.  Poses of the map part are aligned to ground truth
(SE(3); the metric checkpoint is used) and the same transform evaluates the query part.

    python scripts/baselines/run_mast3r_slam.py --map data/sim/classroom/map --query data/sim/classroom/night \
        --out outputs/sim/classroom/mast3r/night
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / "third_party/MASt3R-SLAM"
PY = ROOT / ".venv-mast3r/bin/python"
sys.path.insert(0, str(ROOT / "scripts/baselines"))
sys.path.insert(0, str(ROOT / "scripts"))
import shutil  # noqa: E402
from eval_traj import evaluate  # noqa: E402
from reloc_metrics import build_trials, first_map_link, summarize_trials, summarize_errors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim3", action="store_true", help="align with Sim(3) instead of SE(3)")
    ap.add_argument("--trial-len", type=int, default=0)
    ap.add_argument("--trial-stride", type=int, default=None)
    ap.add_argument("--r-d", type=float, default=2.0)
    ap.add_argument("--config", default="config/base.yaml", help="MASt3R-SLAM config (relative to its repo)")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per trial before the run is killed")
    ap.add_argument("--map-only", action="store_true", help="run on the map sequence alone (single-session accuracy)")
    ap.add_argument("--max-trials", type=int, default=0, help="evaluate at most this many evenly spaced trials (0: all)")
    args = ap.parse_args()
    m, q = Path(args.map).resolve(), Path(args.query).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    def images(seq):
        return sorted((seq / ("left" if (seq / "left").is_dir() else "rgb")).glob("*.png"))

    lm = images(m)
    lq = images(q)
    n_map = len(lm)
    calib = json.loads((m / "calib.json").read_text())
    K = np.asarray(calib["K"])
    (out / "calib.yaml").write_text(
        f"width: {calib['width']}\nheight: {calib['height']}\ncalibration: [{K[0,0]}, {K[1,1]}, {K[0,2]}, {K[1,2]}]\n")
    gt_map = np.loadtxt(m / "poses_left.txt").reshape(-1, 4, 4)
    gt_query = np.loadtxt(q / "poses_left.txt").reshape(-1, 4, 4)
    trials = build_trials(len(lq), args.trial_len, args.trial_stride)
    if args.max_trials and len(trials) > args.max_trials:
        trials = [trials[i] for i in np.linspace(0, len(trials) - 1, args.max_trials).round().astype(int)]
    if args.map_only:                    # the map sequence alone: one "trial" with no query frames
        trials, lq = [(0, 0)], []
    all_rows, per_trial_time = [], []
    map_poses_ref = None
    for ti, (a, b) in enumerate(trials):
        stream = out / f"stream_t{ti}"
        stream.mkdir(exist_ok=True)
        for f in stream.glob("*.png"):
            f.unlink()
        for i, f in enumerate(lm + lq[a:b]):        # map traversal followed by the trial sub-sequence
            os.symlink(f, stream / f"{i:06d}.png")
        save_as = f"cross_{out.parent.name}_{out.name}_t{ti}_{int(time.time())}"
        cmd = [str(PY), "main.py", "--dataset", str(stream), "--config", args.config, "--calib", str(out / "calib.yaml"),
               "--no-viz", "--save-as", save_as]
        t0 = time.time()
        with open(out / f"mast3r_t{ti}.log", "w") as f:
            # MASt3R-SLAM's multi-process back end occasionally deadlocks: bound every trial and kill the
            # whole process group on timeout (the back-end processes would otherwise keep the GPU memory)
            proc = subprocess.Popen(cmd, cwd=str(MS), stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                                    env=dict(os.environ, CUDA_HOME=os.environ.get("CUDA_HOME", "/usr/local/cuda-12.6")))
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
        per_trial_time.append({"trial": ti, "rc": rc, "seconds": dt, "start": a, "end": b})
        traj_all = MS / "logs" / save_as / f"stream_t{ti}_all.txt"     # every tracked frame (patched main.py)
        traj = traj_all if traj_all.is_file() else MS / "logs" / save_as / f"stream_t{ti}.txt"
        map_poses, query_poses = {}, {}
        # query frames count as localized only from the first query keyframe linked to a map keyframe by a
        # retrieval / relocalization edge (the patched main.py writes the factor graph's edges); before that their pose
        # only continues the map's trajectory across the jump between the traversals (reloc_metrics.first_map_link)
        edges_f = MS / "logs" / save_as / f"stream_t{ti}_edges.txt"
        link = None
        if edges_f.is_file() and edges_f.stat().st_size > 0:
            e = np.loadtxt(edges_f, ndmin=2)
            link = first_map_link([(r[0], r[1]) for r in e if int(r[2]) == 0], n_map)
        per_trial_time[-1].update(map_link=None if link is None else int(link) - n_map + a, edge_log=edges_f.is_file())
        if traj.is_file():
            rows_ = np.loadtxt(traj)
            rows_ = rows_.reshape(-1, rows_.shape[-1] if rows_.ndim > 1 else len(rows_))
            for r in rows_:
                idx = int(round(r[0] * 30.0))          # RGBFiles timestamps are index / 30
                if not np.all(np.isfinite(r[1:8])) or np.linalg.norm(r[4:8]) < 1e-6:
                    continue                            # degenerate pose (never tracked): counts as no pose
                T = np.eye(4)
                T[:3, :3] = Rotation.from_quat(r[4:8] / np.linalg.norm(r[4:8])).as_matrix()
                T[:3, 3] = r[1:4]
                state = 2 if (len(r) < 9 or r[8] > 0) else 1   # 1 = frame was in relocalization (lost) mode
                if idx < n_map:
                    map_poses[idx] = (state, T)
                else:
                    if link is None or idx < link:
                        state = 1                       # not linked to the map yet: no pose in the map
                    query_poses[idx - n_map + a] = (state, T)
        if args.map_only:
            with open(out / "map_poses.txt", "w") as f:
                for i in sorted(map_poses):
                    st, T = map_poses[i]
                    f.write(f"{i} {st} " + " ".join(f"{v:.9f}" for v in T.reshape(-1)) + "\n")
            (out / "map_time.json").write_text(json.dumps({"rc": rc, "seconds": dt, "n_frames": n_map}))
            shutil.rmtree(stream, ignore_errors=True)
            return
        rws, summ = evaluate(map_poses, query_poses, gt_map, gt_query, sim3=args.sim3, frame_range=(a, b))
        if rws is None:
            rws = [{"frame": i, "trial": ti, "tracked": False, "c0_rel_t_err": np.inf, "c0_rel_r_err": np.inf, "t_err": np.inf, "r_err": np.inf} for i in range(a, b)]
        for r in rws:
            r["trial"] = ti
        all_rows.extend(rws)
        shutil.rmtree(stream, ignore_errors=True)
    trials_summary = summarize_trials(all_rows, "c0_rel", r_d=args.r_d)
    summary = {"trials": trials_summary, "RS": trials_summary["RS"], "RS_1m_5deg": trials_summary["RS_1m_5deg"],
               "RS_0.5m_5deg": trials_summary["RS_0.5m_5deg"], "map_relative": summarize_errors(all_rows, "c0_rel"),
               "n_frames": len(all_rows), "times": per_trial_time, "n_query_keyframes": int(sum(1 for r in all_rows if r.get("tracked")))}
    (out / "reloc_summary.json").write_text(json.dumps(summary, indent=1))
    (out / "reloc_rows.json").write_text(json.dumps(all_rows, default=lambda x: None))
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("trials", "times")}, indent=1))


if __name__ == "__main__":
    main()
