#!/usr/bin/env python3
"""Run external SLAM baselines (ORB-SLAM3 stereo, RTAB-Map RGB-D) on a map/query pair and evaluate.

    python scripts/baselines/run_baselines.py --map data/sim/classroom/map --query data/sim/classroom/night \
        --system orbslam3 --out outputs/sim/classroom/orbslam3/night --snr 10

Both systems first map the `map` traversal (saving an atlas / database) and then run the
`query` traversal in localization mode against it.  RTAB-Map receives the same noisy
odometry that CROSS receives (SNR-perturbed ground-truth deltas) and ground-truth depth
(RGB-D); ORB-SLAM3 runs pure stereo.  Outputs use the same metrics as map_and_reloc.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/baselines"))
from reloc_metrics import build_trials  # noqa: E402
BIN = ROOT / "scripts/baselines/build"
ORB = ROOT / "third_party/ORB_SLAM3"
ENV = dict(os.environ, LD_LIBRARY_PATH=f"{ROOT}/third_party/install/lib:{ORB}/lib:{ORB}/Thirdparty/DBoW2/lib:{ORB}/Thirdparty/g2o/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))


def prepare_sequence(seq: Path, snr, seed=0):
    """Write calib_pinhole.txt, depth_mm/ and odom_snr{snr}.txt next to a SimChange sequence."""
    from cross.dataloader.stereo_loader import StereoSequenceLoader
    calib = json.loads((seq / "calib.json").read_text())
    K = np.asarray(calib["K"])
    (seq / "calib_pinhole.txt").write_text(f"{K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]} {calib['width']} {calib['height']}\n")
    dmm = seq / "depth_mm"
    if not dmm.is_dir() or len(list(dmm.glob("*.png"))) != len(list((seq / "left").glob("*.png"))):
        dmm.mkdir(exist_ok=True)
        for f in sorted((seq / "depth").glob("*.npy")):
            d = np.load(f).astype(np.float32)
            cv2.imwrite(str(dmm / (f.stem + ".png")), np.clip(d * 1000.0, 0, 65535).astype(np.uint16))
    odom_file = seq / f"odom_snr{snr}.txt"
    if not odom_file.is_file():
        ds = StereoSequenceLoader(str(seq), snr=snr)
        np.random.seed(seed)
        T = np.eye(4)
        rows = []
        for i, d in enumerate(ds.replay_data()):
            if i > 0:
                T = T @ d["delta_pose"]
            rows.append(T.reshape(-1))
        np.savetxt(odom_file, np.asarray(rows), fmt="%.8f")
    return odom_file


def orb_yaml(seq: Path, path: Path, load_atlas=None, save_atlas=None, fps=10.0, n_features=None, ini_fast=None, min_fast=None, th_depth=40.0, baseline=None):
    import os
    n_features = n_features or int(os.environ.get('ORB_NFEAT', 2000))
    ini_fast = ini_fast or int(os.environ.get('ORB_INI_FAST', 12))
    min_fast = min_fast or int(os.environ.get('ORB_MIN_FAST', 4))
    calib = json.loads((seq / "calib.json").read_text())
    K = np.asarray(calib["K"])
    T = np.asarray(calib["T_right_in_left"], dtype=np.float64)
    if baseline is not None:
        T[0, 3] = baseline
    lines = ["%YAML:1.0", ""]
    if load_atlas:
        lines.append(f'System.LoadAtlasFromFile: "{load_atlas}"')
    if save_atlas:
        lines.append(f'System.SaveAtlasToFile: "{save_atlas}"')
    lines += [
        'File.version: "1.0"', 'Camera.type: "PinHole"',
        f"Camera1.fx: {K[0,0]}", f"Camera1.fy: {K[1,1]}", f"Camera1.cx: {K[0,2]}", f"Camera1.cy: {K[1,2]}",
        "Camera1.k1: 0.0", "Camera1.k2: 0.0", "Camera1.p1: 0.0", "Camera1.p2: 0.0",
        f"Camera2.fx: {K[0,0]}", f"Camera2.fy: {K[1,1]}", f"Camera2.cx: {K[0,2]}", f"Camera2.cy: {K[1,2]}",
        "Camera2.k1: 0.0", "Camera2.k2: 0.0", "Camera2.p1: 0.0", "Camera2.p2: 0.0",
        f"Camera.width: {calib['width']}", f"Camera.height: {calib['height']}", f"Camera.fps: {int(fps)}", "Camera.RGB: 0",
        f"Stereo.ThDepth: {th_depth}",
        "Stereo.T_c1_c2: !!opencv-matrix", "  rows: 4", "  cols: 4", "  dt: f",
        "  data: [" + ",".join(f"{v:.9f}" for v in T.reshape(-1)) + "]",
        f"ORBextractor.nFeatures: {n_features}", "ORBextractor.scaleFactor: 1.2", "ORBextractor.nLevels: 8",
        f"ORBextractor.iniThFAST: {ini_fast}", f"ORBextractor.minThFAST: {min_fast}",
        "Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1.0", "Viewer.GraphLineWidth: 0.9", "Viewer.PointSize: 2.0",
        "Viewer.CameraSize: 0.08", "Viewer.CameraLineWidth: 3.0", "Viewer.ViewpointX: 0.0", "Viewer.ViewpointY: -0.7",
        "Viewer.ViewpointZ: -1.8", "Viewer.ViewpointF: 500.0",
    ]
    path.write_text("\n".join(lines) + "\n")


def make_chunk(seq: Path, out: Path, start: int, end: int, snr=None) -> Path:
    """A SimChange-layout sub-sequence [start, end) made of symlinks (frames renumbered from 0)."""
    chunk = out / f"chunk_{start:05d}_{end:05d}"
    if chunk.is_dir():
        shutil.rmtree(chunk)
    chunk.mkdir(parents=True)
    calib = json.loads((seq / "calib.json").read_text())
    dirs = ["left", "depth", "depth_mm"] + list(calib.get("right_dirs", {}).values()) + (["right"] if (seq / "right").exists() else [])
    for d in dirs:
        src = seq / d
        if not src.exists():
            continue
        (chunk / d).mkdir()
        files = sorted(p for p in src.iterdir() if p.suffix in (".png", ".npy"))
        for k, f in enumerate(files[start:end]):
            os.symlink(f.resolve(), chunk / d / f"{k:06d}{f.suffix}")
    (chunk / "calib.json").write_text(json.dumps(calib))
    if (seq / "calib_pinhole.txt").is_file():
        shutil.copy(seq / "calib_pinhole.txt", chunk / "calib_pinhole.txt")
    poses = np.loadtxt(seq / "poses_left.txt").reshape(-1, 16)[start:end]
    np.savetxt(chunk / "poses_left.txt", poses, fmt="%.8f")
    if snr is not None:
        odom = np.loadtxt(seq / f"odom_snr{snr}.txt").reshape(-1, 16)[start:end]
        np.savetxt(chunk / f"odom_snr{snr}.txt", odom, fmt="%.8f")
    return chunk


def run(cmd, log, cwd=None, timeout=None):
    # BASELINE_TIMEOUT: the replay traces run whole 2300-frame traversals in one process, which takes
    # RTAB-Map far longer than the 100-frame evaluation trials the default hour was chosen for.
    timeout = timeout or float(os.environ.get("BASELINE_TIMEOUT", 3600))
    with open(log, "w") as f:
        t0 = time.time()
        try:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=cwd, timeout=timeout)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout} s\n")
            rc = -9
    return rc, time.time() - t0


def _map_run_ok(out: Path) -> bool:
    """A stored map is reused only when its mapping run finished (map_time.json with rc == 0).  A run killed by
    BASELINE_TIMEOUT leaves a truncated map_poses.txt / database behind, which a retry must not pick up (this
    is how the first HSSD RTAB-Map replay traces ended up with maps covering half of the traversal)."""
    f = out / "map_time.json"
    if not f.is_file():
        return False
    try:
        return int(json.loads(f.read_text()).get("rc", -1)) == 0
    except Exception:
        return False


def run_orbslam3(args, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    m, q = Path(args.map).resolve(), Path(args.query).resolve()
    atlas = "atlas"  # ORB-SLAM3 prefixes "./": keep it relative to cwd=out
    fps = json.loads((m / "calib.json").read_text()).get("fps", 10.0)
    voc = str(ORB / "Vocabulary/ORBvoc.txt")
    map_poses = out / "map_poses.txt"
    rd = []
    if args.baseline is not None:
        rd = ["--right-dir", json.loads((m / "calib.json").read_text())["right_dirs"][f"{args.baseline:.2f}"]]
    if not (out / "atlas.osa").is_file() or not map_poses.is_file() or not _map_run_ok(out):
        orb_yaml(m, out / "map.yaml", save_atlas=atlas, fps=fps, baseline=args.baseline)
        rc, dt = run([str(BIN / "orbslam3_reloc"), voc, str(out / "map.yaml"), str(m), str(map_poses), "--fps", str(fps)] + rd, out / "map.log", cwd=str(out))
        (out / "map_time.json").write_text(json.dumps({"rc": rc, "seconds": dt}))
    orb_yaml(q, out / "query.yaml", load_atlas=atlas, fps=fps, baseline=args.baseline)
    # multi-session mode: the query session starts a new map that ORB-SLAM3 merges into the loaded atlas
    # on place recognition (its localization-only mode never relocalizes against a loaded atlas here)
    n_q = len(list((q / "left").glob("*.png")))
    trials = build_trials(n_q, args.trial_len, args.trial_stride)
    query_files = []
    for ti, (a, b) in enumerate(trials):
        chunk = make_chunk(q, out / "trials", a, b) if len(trials) > 1 or a > 0 or b < n_q else q
        qp = out / f"query_poses_t{ti}.txt"
        rc, dt = run([str(BIN / "orbslam3_reloc"), voc, str(out / "query.yaml"), str(chunk), str(qp), "--fps", str(fps)] + rd, out / f"query_t{ti}.log", cwd=str(out))
        for _retry in range(2):
            if rc >= 0:
                break
            rc, dt = run([str(BIN / "orbslam3_reloc"), voc, str(out / "query.yaml"), str(chunk), str(qp), "--fps", str(fps)] + rd, out / f"query_t{ti}.log", cwd=str(out))
        (out / f"query_time_t{ti}.json").write_text(json.dumps({"rc": rc, "seconds": dt, "start": a, "end": b}))
        query_files.append((a, b, qp))
    return map_poses, query_files, False


def run_rtabmap(args, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    m, q = Path(args.map).resolve(), Path(args.query).resolve()
    odom_m = prepare_sequence(m, args.snr)
    odom_q = prepare_sequence(q, args.snr)
    db = out.resolve() / "map.db"
    map_poses = out / "map_poses.txt"
    extra = ["--Vis/MinInliers", "15", "--Rtabmap/DetectionRate", "0", "--Kp/DetectorStrategy", "6", "--Vis/FeatureType", "6"]
    if args.system == "rtabmap_stereo":
        # stereo mode: RTAB-Map computes its depth by stereo matching of the rectified pair (no GT depth)
        calib = json.loads((m / "calib.json").read_text())
        b = args.baseline if args.baseline is not None else float(np.asarray(calib["T_right_in_left"])[0, 3])
        rd = calib["right_dirs"][f"{b:.2f}"] if args.baseline is not None else "right"
        extra = ["--stereo", rd, str(b)] + extra
    if not db.is_file() or not map_poses.is_file() or not _map_run_ok(out):
        rc, dt = run([str(BIN / "rtabmap_reloc"), str(m), str(odom_m), str(db), str(map_poses)] + extra, out / "map.log", cwd=str(out))
        (out / "map_time.json").write_text(json.dumps({"rc": rc, "seconds": dt}))
    n_q = len(list((q / "left").glob("*.png")))
    trials = build_trials(n_q, args.trial_len, args.trial_stride)
    query_files = []
    for ti, (a, b) in enumerate(trials):
        chunk = make_chunk(q, out / "trials", a, b, snr=args.snr) if len(trials) > 1 or a > 0 or b < n_q else q
        odom_c = chunk / f"odom_snr{args.snr}.txt"
        qp = out / f"query_poses_t{ti}.txt"
        rc, dt = run([str(BIN / "rtabmap_reloc"), str(chunk), str(odom_c), str(db), str(qp), "--localization"] + extra, out / f"query_t{ti}.log", cwd=str(out))
        (out / f"query_time_t{ti}.json").write_text(json.dumps({"rc": rc, "seconds": dt, "start": a, "end": b}))
        query_files.append((a, b, qp))
    return map_poses, query_files, False


def apply_final_trajectory(poses, poses_file, fps):
    """ORB-SLAM3: replace online poses by the final trajectory (<poses_file>.final, written after shutdown:
    ts x y z qx qy qz qw map_id atlas_map_id).  A frame counts as localized in the map when its reference
    keyframe belongs to the largest (atlas) map, i.e. its session was merged, even if tracking was lost later."""
    from scipy.spatial.transform import Rotation
    fin = Path(str(poses_file) + ".final")
    if not fin.is_file():
        return poses
    poses = dict(poses)
    for line in fin.read_text().splitlines():
        v = line.split()
        if len(v) < 8:
            continue
        idx = int(round(float(v[0]) * fps))
        in_atlas = len(v) < 10 or v[8] == v[9]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([float(x) for x in v[4:8]]).as_matrix()
        T[:3, 3] = [float(x) for x in v[1:4]]
        poses[idx] = (2 if in_atlas else 6, T)
    return poses


def evaluate_trials(map_poses_file, query_files, map_seq, query_seq, out: Path, sim3=False, r_d=2.0):
    """Evaluate every trial with the map-relative metric and aggregate the relocalization success."""
    from eval_traj import evaluate, load_poses
    from reloc_metrics import summarize_trials, summarize_errors
    gt_map = np.loadtxt(Path(map_seq) / "poses_left.txt").reshape(-1, 4, 4)
    gt_query = np.loadtxt(Path(query_seq) / "poses_left.txt").reshape(-1, 4, 4)
    fps = json.loads((Path(map_seq) / "calib.json").read_text()).get("fps", 10.0)
    mp = apply_final_trajectory(load_poses(map_poses_file), map_poses_file, fps)
    all_rows = []
    for ti, (a, b, qp) in enumerate(query_files):
        try:
            qposes = apply_final_trajectory(load_poses(qp), qp, fps)
        except Exception:
            qposes = {}
        # frames in the chunk are renumbered from 0 -> shift back
        qposes = {i + a: v for i, v in qposes.items()}
        rows, summ = evaluate(mp, qposes, gt_map, gt_query, sim3=sim3, frame_range=(a, b))
        if rows is None:
            rows = [{"frame": i, "trial": ti, "tracked": False, "c0_rel_t_err": np.inf, "c0_rel_r_err": np.inf, "t_err": np.inf, "r_err": np.inf} for i in range(a, b)]
        for r in rows:
            r["trial"] = ti
        all_rows.extend(rows)
    trials = summarize_trials(all_rows, "c0_rel", r_d=r_d)
    summary = {"trials": trials, "RS": trials["RS"], "RS_1m_5deg": trials["RS_1m_5deg"], "RS_0.5m_5deg": trials["RS_0.5m_5deg"],
               "map_relative": summarize_errors(all_rows, "c0_rel"), "n_frames": len(all_rows),
               "n_map_tracked": len([1 for i, (st, _) in mp.items() if st == 2])}
    (out / "reloc_summary.json").write_text(json.dumps(summary, indent=1))
    (out / "reloc_rows.json").write_text(json.dumps(all_rows, default=lambda x: None))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("trials",)}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--system", choices=["orbslam3", "rtabmap", "rtabmap_stereo"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snr", type=float, default=10)
    ap.add_argument("--baseline", type=float, default=None, help="stereo baseline (SimChange multi-baseline renders)")
    ap.add_argument("--trial-len", type=int, default=0)
    ap.add_argument("--trial-stride", type=int, default=None)
    ap.add_argument("--r-d", type=float, default=2.0)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if args.system in ("rtabmap", "rtabmap_stereo") and False:
        pass
    if args.system == "orbslam3":
        mp, qp, sim3 = run_orbslam3(args, out)
    else:
        mp, qp, sim3 = run_rtabmap(args, out)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    evaluate_trials(mp, qp, args.map, args.query, out, sim3=sim3, r_d=args.r_d)


if __name__ == "__main__":
    main()
