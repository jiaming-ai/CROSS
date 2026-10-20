#!/usr/bin/env python3
"""Run AMB3R-SLAM (monocular, stereo or RGB-D) on a map+query stream and evaluate the query traversal.

AMB3R-SLAM (Wang & Agapito 2026, https://github.com/HengyiWang/amb3r-slam) keeps no map between runs, so the map
traversal and the query traversal (or one trial of it) are concatenated into one stream, as for VGGT-SLAM 2.0
(run_vggt_slam.py) and MASt3R-SLAM: the system first maps the scene, then has to attach the second traversal to it
through its loop closures.  It gives every frame a pose: the final poses after its pose graph (frames between its
reconstructed views interpolated).  Query frames count as localized only from the first query frame that an accepted
loop closure links to a map frame (reloc_metrics.first_map_link): amb3r_slam_driver.py logs every accepted loop
(probe frame, matched frame, the probe's submap); a loop from a query probe to a map frame links the probe's submap,
whose frames are estimated together, from its first query frame on (as for VGGT-SLAM 2.0).  Before that the query
poses only continue the map's trajectory across the jump between the traversals.  Monocular: the map part is aligned
to ground truth with Sim(3) and the same transform evaluates the query part; stereo / RGB-D (metric scale from the
sensor's depth): SE(3).  All poses are expressed in the ground-truth frame before evaluation so that the map-relative
errors are in metres (see sim3_to_gt).

    python scripts/baselines/run_amb3r_slam.py --map <seq> --query <seq> --out <dir> --sensor mono|stereo|rgbd \
        [--map-only] [--trial-len N] [--trial-stride S] [--max-trials K] [--r-d R] [--timeout SEC]

One setting for every run: the upstream configuration (configs/base.yaml + the sensor's layer), DA3 models
(DA3NESTED-GIANT-LARGE-1.1 back end, DA3-SMALL front end), bf16 encoder weights, DBoW2 loop retrieval, input 518 px on
the longer side, the sequence's intrinsics (upstream --calib), the sequence's frame rate.  Mono: the colour images
(rgb/) or the left images.  Stereo: the rectified pair, depth by SGBM on the full-resolution pair (upstream
StereoDepth) with the sequence's baseline, depth from 0.3 m (the configuration's RGB-D minimum) without a far limit
(upstream's stereo default), the same on every dataset.  Upstream's default minimum is 3 m, set for KITTI (the only
stereo data it evaluates), where SGBM measures nothing closer anyway (fx * baseline / 128 disparities = 2.96 m): there
the setting is upstream's.  On the T265 pairs (6.4 cm) a 3 m minimum leaves too little depth to measure a scale
(OpenLORIS home1-1: 5 of 47 submaps metric, ATE 0.84 m, against 43 of 47 and 0.24 m with 0.3-5 m).  RGB-D: the depth
images (uint16, calib.json depth_scale or 1000 per metre) within the configuration's 0.3-5 m.
--map-only writes map_poses.txt (idx state t00..t33, camera-to-world, state 2) and map_time.json.  Every run keeps the
driver's output (each frame's final and online pose, the accepted loops) as amb3r_map.npz / amb3r_t<trial>.npz, so the
runs can be re-scored without running the system.
Environment: AMB3R_PY (its python), AMB3R_DIR (its repository); install with scripts/baselines/install_amb3r_slam.sh.
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

ROOT = Path(__file__).resolve().parents[2]
AMB3R_DIR = Path(os.environ.get("AMB3R_DIR", ROOT / "third_party/amb3r_slam/amb3r-slam"))
PY = Path(os.environ.get("AMB3R_PY", ROOT / "third_party/amb3r_slam/env/bin/python"))
DRIVER = Path(__file__).resolve().parent / "amb3r_slam_driver.py"
sys.path.insert(0, str(ROOT / "scripts/baselines"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_traj import evaluate, umeyama  # noqa: E402
from reloc_metrics import build_trials, first_map_link, summarize_errors, summarize_trials  # noqa: E402

STEREO_DEPTH = (0.3, 0.0)          # (min, max) m of the stereo depth source; 0 = no limit (see the docstring)


def image_dir(seq: Path) -> Path:
    return seq / ("rgb" if (seq / "rgb").is_dir() else "left")


def right_dir(seq: Path) -> Path:
    if (seq / "right").is_dir():
        return seq / "right"
    calib = json.loads((seq / "calib.json").read_text())
    dirs = calib.get("right_dirs") or {}
    b = calib.get("baseline")
    key = f"{float(b):.2f}" if b is not None and f"{float(b):.2f}" in dirs else next(iter(dirs))
    return seq / dirs[key]


def stream_part(seq: Path, sensor: str, a=0, b=None):
    """The frames [a, b) of a sequence folder for the sensor: image paths, right / depth paths, camera."""
    calib = json.loads((seq / "calib.json").read_text())
    imgs = sorted(image_dir(seq).glob("*.png"))[a:b]
    part = {"images": [str(p) for p in imgs], "K": calib["K"], "width": calib["width"], "height": calib["height"],
            "fps": calib.get("fps", 10.0)}
    if sensor == "stereo":
        rd = right_dir(seq)
        part["right"] = [str(rd / p.name) for p in imgs]
        part["baseline"] = float(calib.get("baseline", np.asarray(calib["T_right_in_left"])[0, 3]))
    elif sensor == "rgbd":
        part["depth"] = [str(seq / "depth" / p.name) for p in imgs]
        part["depth_scale"] = float(calib.get("depth_scale", 1000.0))
    return part


def concat(parts):
    """One stream of several parts of the same camera (map, then query)."""
    out = dict(parts[0])
    for p in parts[1:]:
        if not np.allclose(np.asarray(p["K"]), np.asarray(out["K"]), atol=1e-3) or \
                (p["width"], p["height"]) != (out["width"], out["height"]):
            print(f"warning: the query camera differs from the map's ({p['K']} vs {out['K']}); the map's is used",
                  flush=True)
        for k in ("images", "right", "depth"):
            if k in out:
                out[k] = list(out[k]) + list(p[k])
    return out


def stereo_range(stream, args):
    """(min, max) depth (m) of the stereo depth source: STEREO_DEPTH, or --stereo-depth MIN MAX (experiments)."""
    return [float(v) for v in (args.stereo_depth or STEREO_DEPTH)]


def run_stream(stream, sensor, workdir: Path, log: Path, args, keep: Path):
    """Run AMB3R-SLAM on one stream; returns (rc, seconds, info, poses {stream index: 4x4}, loops (k, 8)).  The driver's
    output (every frame's final and online pose, the loops) is kept as `keep`, so the run can be re-scored."""
    workdir.mkdir(parents=True, exist_ok=True)
    man = workdir / "manifest.json"
    npz = workdir / "amb3r_out.npz"
    npz.unlink(missing_ok=True)
    m = dict(stream, sensor=sensor, stereo_depth=stereo_range(stream, args))
    man.write_text(json.dumps(m))
    cmd = [str(PY), str(DRIVER), str(AMB3R_DIR), str(man), str(npz)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"))
    t0 = time.time()
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        # bound every run and kill the whole process group on timeout (no GPU memory left behind)
        proc = subprocess.Popen(cmd, cwd=str(AMB3R_DIR), stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                                env=env)
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
    if rc != 0 or not npz.is_file():
        return (rc if rc != 0 else 1), dt, None, {}, np.zeros((0, 8), dtype=np.int64)
    z = np.load(npz)
    info = json.loads(str(z["info"]))
    P = z["poses"]
    ok = np.all(np.isfinite(P.reshape(len(P), -1)), axis=1) & (np.abs(np.linalg.det(P[:, :3, :3]) - 1) < 1e-2)
    poses = {i: P[i] for i in range(len(P)) if ok[i]}
    loops = z["loops"]
    man.unlink(missing_ok=True)
    shutil.move(str(npz), str(keep))
    return rc, dt, info, poses, loops


def align_to_gt(poses, gt_map, n_map, sim3: bool):
    """Express all poses in the ground-truth frame through the alignment of the map part, Sim(3) (monocular) or SE(3)
    (evaluate() then finds an identity alignment).  Needed because eval_traj.evaluate() applies its alignment only to
    the absolute errors: the map-relative errors (c0_rel_*) compose the raw estimates with metric ground-truth offsets,
    which for an arbitrary-scale monocular trajectory mixes units.  After this step they are in metres."""
    ids = [i for i in sorted(poses) if i < min(n_map, len(gt_map))]
    if len(ids) < 3:
        return poses, None
    s, R, t = umeyama(np.array([poses[i][:3, 3] for i in ids]), np.array([gt_map[i][:3, 3] for i in ids]),
                      with_scale=sim3)
    out = {}
    for i, T in poses.items():
        A = np.eye(4)
        A[:3, :3] = R @ T[:3, :3]
        A[:3, 3] = s * R @ T[:3, 3] + t
        out[i] = A
    return out, float(s)


def map_link(loops, n_map):
    """First query stream index linked to the map by an accepted loop closure from a query probe frame to a map frame:
    the probe's submap (its frames estimated together), from its first query frame on.  None without such a loop."""
    return first_map_link([(max(int(r[4]), n_map), int(r[1])) for r in loops if r[0] >= n_map and r[1] < n_map], n_map)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensor", choices=["mono", "stereo", "rgbd"], default="mono")
    ap.add_argument("--stereo-depth", nargs=2, type=float, default=None, metavar=("MIN", "MAX"),
                    help="stereo depth range in m (0 = no limit) instead of STEREO_DEPTH (experiments only)")
    ap.add_argument("--map-only", action="store_true", help="run on the map sequence alone (single-session accuracy)")
    ap.add_argument("--trial-len", type=int, default=0)
    ap.add_argument("--trial-stride", type=int, default=None)
    ap.add_argument("--max-trials", type=int, default=0, help="evaluate at most this many evenly spaced trials (0: all)")
    ap.add_argument("--r-d", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds per run before it is killed")
    ap.add_argument("--sim3", action="store_true", help="ignored: monocular is evaluated with Sim(3), metric with SE(3)")
    args = ap.parse_args()
    m, q = Path(args.map).resolve(), Path(args.query).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sim3 = args.sensor == "mono"
    (out / "args.json").write_text(json.dumps({**vars(args), "amb3r_dir": str(AMB3R_DIR), "python": str(PY),
                                               "stereo_depth": stereo_range(stream_part(m, args.sensor), args)
                                               if args.sensor == "stereo" else None,
                                               "calib": True, "model": "da3", "retrieval": "dbow"}, indent=1))
    lm = stream_part(m, args.sensor)
    n_map = len(lm["images"])
    gt_map = np.loadtxt(m / "poses_left.txt").reshape(-1, 4, 4)

    if args.map_only:
        rc, dt, info, poses, loops = run_stream(lm, args.sensor, out / "work_map", out / "amb3r_map.log", args,
                                                out / "amb3r_map.npz")
        with open(out / "map_poses.txt", "w") as f:
            for i in sorted(poses):
                if i < n_map:
                    f.write(f"{i} 2 " + " ".join(f"{v:.9f}" for v in poses[i].reshape(-1)) + "\n")
        rec = {"rc": rc, "seconds": dt, "n_frames": n_map, "n_poses": len(poses), "loops": int(len(loops)),
               "amb3r": info, "peak_gpu_gb": (info or {}).get("peak_gpu_gb")}
        ids = [i for i in sorted(poses) if i < len(gt_map)]
        if len(ids) >= 3:
            src = np.array([poses[i][:3, 3] for i in ids])
            dst = np.array([gt_map[i][:3, 3] for i in ids])
            s, R, t = umeyama(src, dst, with_scale=sim3)
            rec["map_ate_rmse_" + ("sim3" if sim3 else "se3")] = float(np.sqrt(np.mean(np.sum((s * (R @ src.T).T + t - dst) ** 2, 1))))
            rec["sim3_scale"] = s
        (out / "map_time.json").write_text(json.dumps(rec))
        shutil.rmtree(out / "work_map", ignore_errors=True)
        print(json.dumps(rec, indent=1))
        return

    lq_all = sorted(image_dir(q).glob("*.png"))
    gt_query = np.loadtxt(q / "poses_left.txt").reshape(-1, 4, 4)
    trials = build_trials(len(lq_all), args.trial_len, args.trial_stride)
    if args.max_trials and len(trials) > args.max_trials:
        trials = [trials[i] for i in np.linspace(0, len(trials) - 1, args.max_trials).round().astype(int)]
    all_rows, per_trial_time = [], []
    for ti, (a, b) in enumerate(trials):
        stream = concat([lm, stream_part(q, args.sensor, a, b)])            # map traversal, then the trial
        rc, dt, info, poses, loops = run_stream(stream, args.sensor, out / f"work_t{ti}", out / f"amb3r_t{ti}.log", args,
                                                out / f"amb3r_t{ti}.npz")
        shutil.rmtree(out / f"work_t{ti}", ignore_errors=True)
        per_trial_time.append({"trial": ti, "rc": rc, "seconds": dt, "start": a, "end": b, "n_poses": len(poses),
                               "peak_gpu_gb": (info or {}).get("peak_gpu_gb"), "fps": (info or {}).get("fps"),
                               "loops": int(len(loops))})
        poses, scale = align_to_gt(poses, gt_map, n_map, sim3)
        # query frames count as localized only from the first query submap linked to a map frame by an accepted loop
        # closure; before that their poses only continue the map's trajectory across the jump between the traversals
        link = map_link(loops, n_map)
        per_trial_time[-1]["map_link"] = None if link is None else int(link) - n_map + a
        per_trial_time[-1]["map_loops"] = int(sum(1 for r in loops if r[0] >= n_map and r[1] < n_map))
        map_poses = {i: (2, T) for i, T in poses.items() if i < n_map}
        query_poses = {i - n_map + a: (2 if link is not None and i >= link else 1, T) for i, T in poses.items() if i >= n_map}
        rws, summ = evaluate(map_poses, query_poses, gt_map, gt_query, sim3=sim3, frame_range=(a, b))
        if rws is None:
            rws = [{"frame": i, "trial": ti, "tracked": False, "c0_rel_t_err": np.inf, "c0_rel_r_err": np.inf,
                    "t_err": np.inf, "r_err": np.inf} for i in range(a, b)]
        else:
            per_trial_time[-1].update({"map_ate_rmse": summ["map_ate_rmse"], "scale": scale,
                                       "n_map_tracked": summ["n_map_tracked"]})
            print(f"trial {ti} [{a},{b}): rc={rc} {dt:.0f}s map_ate_rmse={summ['map_ate_rmse']:.3f} "
                  f"query poses={len(query_poses)} link={per_trial_time[-1]['map_link']}", flush=True)
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
