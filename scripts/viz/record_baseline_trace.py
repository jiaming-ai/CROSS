#!/usr/bin/env python3
"""Replay traces of the external baselines (ORB-SLAM3 stereo, RTAB-Map stereo, MASt3R-SLAM) in the
format of `record_trace.py`, so that `build_trace_page.py` / `render_trace_video.py` show them next to
CROSS-stereo.

Each system maps the map sequence once and then localizes every query variant as one continuous
session (trial length 0).  Per frame the trace stores the system's pose (component 0) when the frame
is localized in the map, or no pose when the frame is lost / not merged into the map.  The stored map
is represented by its final map trajectory, sampled every `--map-sample` frames as permanent
"keyframes" joined by odometry edges (these systems do not export their keyframe graphs).

Usage:
  python scripts/viz/record_baseline_trace.py --system orbslam3 --scene lonemonk --map map_loop \
      --variants light_night reverse --out outputs/viz/lonemonk/trace_orbslam3 --baseline 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/baselines"))
sys.path.insert(0, str(ROOT / "scripts/viz"))
from eval_traj import load_poses, umeyama  # noqa: E402


def _mat_to_pose7(T: np.ndarray) -> list:
    """4x4 SE(3) -> [x, y, z, qx, qy, qz, qw], as record_trace.py writes it.  Kept local (instead of
    imported from record_trace) so that this recorder also runs in the torch-free CPU environment of
    the hosts that carry the ORB-SLAM3 / RTAB-Map builds."""
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return [round(float(v), 4) for v in np.concatenate([T[:3, 3], q])]


def _inv(T):
    o = np.eye(4); o[:3, :3] = T[:3, :3].T; o[:3, 3] = -T[:3, :3].T @ T[:3, 3]; return o


def run_system(system, m: Path, q: Path, out: Path, snr, baseline):
    """Map (once) + one continuous query session; returns (map_poses{idx:(state,T)}, query_poses{idx:(state,T)})."""
    import run_baselines as rb
    args = SimpleNamespace(map=str(m), query=str(q), snr=snr, baseline=baseline, trial_len=0, trial_stride=None, system=system)
    fps = json.loads((m / "calib.json").read_text()).get("fps", 10.0)
    if system == "orbslam3":
        mp, qf, _ = rb.run_orbslam3(args, out)
    else:
        mp, qf, _ = rb.run_rtabmap(args, out)
    map_poses = rb.apply_final_trajectory(load_poses(mp), mp, fps)
    a, b, qp = qf[0]
    try:
        query_poses = rb.apply_final_trajectory(load_poses(qp), qp, fps)
    except Exception:
        query_poses = {}
    return map_poses, query_poses


def run_mast3r(m: Path, q: Path, out: Path, config: str, timeout: float):
    """MASt3R-SLAM on the concatenated map+query stream (as scripts/baselines/run_mast3r_slam.py, one trial)."""
    import run_mast3r_slam as ms
    import os, signal, subprocess, time
    out.mkdir(parents=True, exist_ok=True)
    lm = sorted((m / "left").glob("*.png")); lq = sorted((q / "left").glob("*.png"))
    n_map = len(lm)
    calib = json.loads((m / "calib.json").read_text()); K = np.asarray(calib["K"])
    (out / "calib.yaml").write_text(f"width: {calib['width']}\nheight: {calib['height']}\ncalibration: [{K[0,0]}, {K[1,1]}, {K[0,2]}, {K[1,2]}]\n")
    stream = out / "stream_t0"
    stream.mkdir(exist_ok=True)
    for f in stream.glob("*.png"):
        f.unlink()
    for i, f in enumerate(lm + lq):
        os.symlink(f.resolve(), stream / f"{i:06d}.png")
    save_as = f"viz_{out.parent.name}_{out.name}_{int(time.time())}"
    cmd = [str(ms.PY), "main.py", "--dataset", str(stream.resolve()), "--config", config, "--calib", str((out / "calib.yaml").resolve()), "--no-viz", "--save-as", save_as]
    t0 = time.time()
    with open(out / "mast3r_t0.log", "w") as f:
        proc = subprocess.Popen(cmd, cwd=str(ms.MS), stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                                env=dict(os.environ, CUDA_HOME=os.environ.get("CUDA_HOME", "/usr/local/cuda-12.6")))
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = -9; f.write("\nTIMEOUT\n")
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    (out / "time.json").write_text(json.dumps({"rc": rc, "seconds": time.time() - t0}))
    traj_all = ms.MS / "logs" / save_as / "stream_t0_all.txt"
    traj = traj_all if traj_all.is_file() else ms.MS / "logs" / save_as / "stream_t0.txt"
    map_poses, query_poses = {}, {}
    if traj.is_file():
        rows_ = np.loadtxt(traj)
        rows_ = rows_.reshape(-1, rows_.shape[-1] if rows_.ndim > 1 else len(rows_))
        for r in rows_:
            idx = int(round(r[0] * 30.0))
            if not np.all(np.isfinite(r[1:8])) or np.linalg.norm(r[4:8]) < 1e-6:
                continue
            T = np.eye(4); T[:3, :3] = Rotation.from_quat(r[4:8] / np.linalg.norm(r[4:8])).as_matrix(); T[:3, 3] = r[1:4]
            state = 2 if (len(r) < 9 or r[8] > 0) else 1
            if idx < n_map:
                map_poses[idx] = (state, T)
            else:
                query_poses[idx - n_map] = (state, T)
    return map_poses, query_poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["orbslam3", "rtabmap_stereo", "rtabmap", "mast3r"], required=True)
    ap.add_argument("--scene", default="lonemonk")
    ap.add_argument("--data-root", default="data/sim")
    ap.add_argument("--map", default="map_loop")
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", type=float, default=0.3)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--map-sample", type=int, default=5, help="map trajectory sampling for the displayed keyframes")
    ap.add_argument("--mast3r-config", default="config/reloc_permissive.yaml")
    ap.add_argument("--mast3r-timeout", type=float, default=3600)
    args = ap.parse_args()

    root = (Path(args.data_root) / args.scene).resolve()
    m = root / args.map
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)   # the drivers change the working directory
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    gt_map = np.loadtxt(m / "poses_left.txt").reshape(-1, 4, 4)
    method = {"orbslam3": "orbslam3", "rtabmap_stereo": "rtabmap_stereo", "rtabmap": "rtabmap", "mast3r": "mast3r_slam"}[args.system]

    sessions, nodes, edges, steps = [], {}, [], []
    map_poses = None

    def add_session(name, kind, variant, seq, poses, gt, n_frames):
        sid = len(sessions)
        s = {"id": sid, "name": name, "kind": kind, "variant": variant, "seq": str(seq), "n_frames": int(n_frames),
             "start_step": len(steps), "n_steps": int(n_frames), "first_gt": _mat_to_pose7(gt[0])}
        sessions.append(s)
        for i in range(n_frames):
            st = {"s": sid, "i": i, "f": i, "gt": _mat_to_pose7(gt[i]), "w": [1.0], "rl": [True], "dt": 0.0}
            if i in poses and poses[i][0] == 2:
                st["obs"] = True
                st["mu"] = [_mat_to_pose7(poses[i][1])]
            else:
                st["obs"] = False
                st["lost"] = True
                st["mu"] = [steps[-1]["mu"][0] if steps and steps[-1]["s"] == sid else [0, 0, 0, 0, 0, 0, 1]]
            steps.append(st)
        return s

    variants = args.variants
    for k, v in enumerate(["__map__"] + variants):
        if v == "__map__":
            q, kind, name = m, "map", f"mapping ({args.map})"
        else:
            q, kind, name = root / v, "reloc", f"relocalization: {v}"
        seq_out = out / ("map" if kind == "map" else v)
        if kind != "map" and args.system != "mast3r":
            # reuse the stored map of the mapping run (as scripts/run_sim_experiments.sh does)
            seq_out.mkdir(parents=True, exist_ok=True)
            import shutil
            for f in (out / "map").glob("*"):
                if f.name.startswith(("atlas", "map.db", "map_poses", "map_time")) and f.is_file() and not (seq_out / f.name).exists():
                    shutil.copy(f, seq_out / f.name)
        if args.system == "mast3r":
            mp, qp = run_mast3r(m, q, seq_out, args.mast3r_config, args.mast3r_timeout)
            if map_poses is None:
                map_poses = mp
        else:
            mp, qp = run_system(args.system, m, q, seq_out, args.snr, args.baseline)
            if kind == "map" or map_poses is None:
                map_poses = mp
        gt = np.loadtxt(q / "poses_left.txt").reshape(-1, 4, 4)
        poses = map_poses if kind == "map" else qp
        if kind == "map":
            # displayed map: the final map trajectory sampled as keyframes with odometry edges
            tracked = sorted(i for i, (stt, _) in map_poses.items() if stt == 2)
            prev = None
            for i in tracked[::args.map_sample]:
                nodes[i] = {"s": 0, "step": i, "f": i, "perm": True, "p": _mat_to_pose7(map_poses[i][1])}
                if prev is not None:
                    edges.append({"a": prev, "b": i, "t": "odom", "s": 0, "step": i, "fc": 0, "tc": 0})
                prev = i
            # the map exists from its own step on; the page builder needs creation steps as global step ids
        add_session(name, kind, args.map if kind == "map" else v, q, poses, gt, len(gt))
        print(f"[baseline-trace] {method} {v}: {sum(1 for i in range(len(gt)) if i in poses and poses[i][0] == 2)}/{len(gt)} frames localized", flush=True)
        # save after every session
        tracked = [i for i, (stt, _) in map_poses.items() if stt == 2]
        src = np.array([map_poses[i][1][:3, 3] for i in tracked]); dst = np.array([gt_map[i][:3, 3] for i in tracked])
        s_, R, t_ = umeyama(src, dst)
        T_ume = np.eye(4); T_ume[:3, :3] = R; T_ume[:3, 3] = t_
        ate = float(np.sqrt(np.mean(np.sum(((R @ src.T).T + t_ - dst) ** 2, 1)))) if len(src) else float("nan")
        i0 = tracked[0] if tracked else 0
        T_first = gt_map[i0] @ _inv(map_poses[i0][1]) if tracked else np.eye(4)
        extra = {"scene": args.scene, "method": method, "estimator": args.system, "map_dir": str(m), "baseline": args.baseline, "snr": args.snr,
                 "T_gt_from_map_first": T_first.tolist(), "T_gt_from_map_umeyama": T_ume.tolist(), "map_ate_rmse": ate,
                 "kf_gt": {str(i): _mat_to_pose7(gt_map[i]) for i in nodes}, "map_final_nodes": {str(i): nd["p"] for i, nd in nodes.items()},
                 "n_components": 1, "obs_cadence": None, "external": True}
        data = {"sessions": sessions, "nodes": nodes, "edges": edges, "steps": steps}
        data.update(extra)
        (out / "trace.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"[baseline-trace] done -> {out / 'trace.json'}")


if __name__ == "__main__":
    main()
