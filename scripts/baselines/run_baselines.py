#!/usr/bin/env python3
"""Run external SLAM baselines (ORB-SLAM3 stereo / RGB-D / mono, RTAB-Map RGB-D / stereo) on a map/query pair and evaluate.

    python scripts/baselines/run_baselines.py --map data/sim/classroom/map --query data/sim/classroom/night \
        --system orbslam3 --out outputs/sim/classroom/orbslam3/night --snr 10

Both systems first map the `map` traversal (saving an atlas / database) and then run the
`query` traversal in localization mode against it.  RTAB-Map receives the same odometry that
CROSS receives: the sequence's own odometry (odom_left.txt, e.g. wheel odometry) when it exists,
otherwise SNR-perturbed ground-truth deltas; ORB-SLAM3 is purely visual (--orb-sensor stereo|rgbd|mono).
Sequences are SimChange folders (left/, right*/, depth/) or benchmark / posed RGB-D folders
(rgb/, depth/ uint16 mm PNG).  Outputs use the same metrics as map_and_reloc.py.

Binaries: scripts/baselines/build (or $BASELINE_BIN, e.g. a portable bundle with lib/ and ld-linux-x86-64.so.2;
set BASELINE_LDSO=1 on hosts whose system libraries differ from the build host).
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
BIN = Path(os.environ.get("BASELINE_BIN", ROOT / "scripts/baselines/build"))
ORB = ROOT / "third_party/ORB_SLAM3"
ORB_VOC = Path(os.environ.get("ORB_VOC", ORB / "Vocabulary/ORBvoc.txt"))
ENV = dict(os.environ, LD_LIBRARY_PATH=f"{BIN}/lib:{ROOT}/third_party/install/lib:{ORB}/lib:{ORB}/Thirdparty/DBoW2/lib:{ORB}/Thirdparty/g2o/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))


def binary(name):
    """Command prefix for a baseline binary (through the bundled loader when BASELINE_LDSO=1)."""
    exe = BIN / "bin" / name if (BIN / "bin" / name).is_file() else BIN / name
    if os.environ.get("BASELINE_LDSO") == "1":
        return [str(BIN / "ld-linux-x86-64.so.2"), "--library-path", str(BIN / "lib"), str(exe)]
    return [str(exe)]


def image_dir(seq: Path) -> str:
    return "left" if (seq / "left").is_dir() else "rgb"


def n_frames(seq: Path) -> int:
    return len(list((seq / image_dir(seq)).glob("*.png")))


def prepare_sequence(seq: Path, snr, seed=0, cache: Path = None):
    """Returns the odometry file of a sequence: its own odom_left.txt when it has one (real odometry, no simulated
    noise), otherwise odom_snr{snr}.txt made from the ground truth.  With `cache`, the dataset folder stays read-only:
    the simulated odometry goes to `cache` and the per-run views (make_chunk) expose depth_mm / calib_pinhole.txt.
    Without it (legacy SimChange runs), calib_pinhole.txt, depth_mm/ and the odometry are written next to the sequence."""
    from cross.dataloader.stereo_loader import StereoSequenceLoader
    if (seq / "odom_left.txt").is_file() and (cache is not None or not (seq / "left").is_dir()):
        return seq / "odom_left.txt"
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        odom_file = cache / f"odom_snr{snr}_seed{seed}.txt"
        if not odom_file.is_file():
            ds = StereoSequenceLoader(str(seq), snr=snr, baseline=_any_baseline(seq))
            np.random.seed(seed)
            T, rows = np.eye(4), []
            for i, d in enumerate(ds.replay_data()):
                if i > 0:
                    T = T @ d["delta_pose"]
                rows.append(T.reshape(-1))
            np.savetxt(odom_file, np.asarray(rows), fmt="%.8f")
        return odom_file
    calib = json.loads((seq / "calib.json").read_text())
    K = np.asarray(calib["K"])
    (seq / "calib_pinhole.txt").write_text(f"{K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]} {calib['width']} {calib['height']}\n")
    dmm = seq / "depth_mm"
    if not (seq / "depth").is_dir():
        pass
    elif next((seq / "depth").glob("*.png"), None) is not None:
        if not dmm.exists():                # current SimChange renders already store depth as uint16 millimetre PNG
            dmm.symlink_to("depth")
    elif not dmm.is_dir() or len(list(dmm.glob("*.png"))) != n_frames(seq):
        dmm.mkdir(exist_ok=True)
        for f in sorted((seq / "depth").glob("*.npy")):
            d = np.load(f).astype(np.float32)
            cv2.imwrite(str(dmm / (f.stem + ".png")), np.clip(d * 1000.0, 0, 65535).astype(np.uint16))
    if (seq / "odom_left.txt").is_file():
        return seq / "odom_left.txt"
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


def _any_baseline(seq: Path):
    """A rendered stereo baseline of a SimChange sequence (the odometry does not depend on it), else None."""
    dirs = json.loads((seq / "calib.json").read_text()).get("right_dirs", {})
    return float(next(iter(dirs))) if dirs and not (seq / "right").exists() else None


def orb_yaml(seq: Path, path: Path, load_atlas=None, save_atlas=None, fps=10.0, n_features=None, ini_fast=None, min_fast=None, th_depth=40.0, baseline=None):
    import os
    n_features = n_features or int(os.environ.get('ORB_NFEAT', 2000))
    ini_fast = ini_fast or int(os.environ.get('ORB_INI_FAST', 12))
    min_fast = min_fast or int(os.environ.get('ORB_MIN_FAST', 4))
    calib = json.loads((seq / "calib.json").read_text())
    K = np.asarray(calib["K"])
    T = np.asarray(calib.get("T_right_in_left", np.eye(4)), dtype=np.float64)
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
        f"Stereo.ThDepth: {th_depth}", "RGBD.DepthMapFactor: 1000.0",
        f"Stereo.b: {float(os.environ.get('ORB_RGBD_B', 0.05)) if abs(T[0, 3]) < 1e-9 else abs(T[0, 3])}",
        "Stereo.T_c1_c2: !!opencv-matrix", "  rows: 4", "  cols: 4", "  dt: f",
        "  data: [" + ",".join(f"{v:.9f}" for v in T.reshape(-1)) + "]",
        f"ORBextractor.nFeatures: {n_features}", "ORBextractor.scaleFactor: 1.2", "ORBextractor.nLevels: 8",
        f"ORBextractor.iniThFAST: {ini_fast}", f"ORBextractor.minThFAST: {min_fast}",
        "Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1.0", "Viewer.GraphLineWidth: 0.9", "Viewer.PointSize: 2.0",
        "Viewer.CameraSize: 0.08", "Viewer.CameraLineWidth: 3.0", "Viewer.ViewpointX: 0.0", "Viewer.ViewpointY: -0.7",
        "Viewer.ViewpointZ: -1.8", "Viewer.ViewpointF: 500.0",
    ]
    path.write_text("\n".join(lines) + "\n")


def make_chunk(seq: Path, out: Path, start: int, end: int, snr=None, odom_file=None) -> Path:
    """A SimChange-layout sub-sequence [start, end) made of symlinks (frames renumbered from 0).  Benchmark / posed
    RGB-D folders (rgb/) become left/; depth PNGs are exposed as depth_mm/; the odometry rows of `odom_file` (or of
    odom_snr{snr}.txt) are written to odom.txt (and odom_snr{snr}.txt for older callers)."""
    chunk = out / f"chunk_{start:05d}_{end:05d}"
    if chunk.is_dir():
        shutil.rmtree(chunk)
    chunk.mkdir(parents=True)
    calib = json.loads((seq / "calib.json").read_text())
    dirs = {d: d for d in ["left", "depth", "depth_mm"] + list(calib.get("right_dirs", {}).values())}
    if (seq / "right").exists():
        dirs["right"] = "right"
    if (seq / "rgb").is_dir() and not (seq / "left").is_dir():
        dirs["rgb"] = "left"
    if (seq / "depth_mm").exists():
        dirs.pop("depth", None)
    elif (seq / "depth").is_dir() and next((seq / "depth").glob("*.png"), None) is not None:
        dirs["depth"] = "depth_mm"
    for d, dst in dirs.items():
        src = seq / d
        if not src.exists():
            continue
        (chunk / dst).mkdir(exist_ok=True)
        files = sorted(p for p in src.iterdir() if p.suffix in (".png", ".npy"))
        for k, f in enumerate(files[start:end]):
            os.symlink(f.resolve(), chunk / dst / f"{k:06d}{f.suffix}")
    (chunk / "calib.json").write_text(json.dumps(calib))
    K = np.asarray(calib["K"])
    (chunk / "calib_pinhole.txt").write_text(f"{K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]} {calib['width']} {calib['height']}\n")
    poses = np.loadtxt(seq / "poses_left.txt").reshape(-1, 16)[start:end]
    np.savetxt(chunk / "poses_left.txt", poses, fmt="%.8f")
    if odom_file is None and snr is not None:
        odom_file = seq / f"odom_snr{snr}.txt"
    if odom_file is not None:
        odom = np.loadtxt(odom_file).reshape(-1, 16)[start:end]
        np.savetxt(chunk / "odom.txt", odom, fmt="%.8f")
        if snr is not None:
            np.savetxt(chunk / f"odom_snr{snr}.txt", odom, fmt="%.8f")
    return chunk


def full_view(seq: Path, out: Path, odom_file=None) -> Path:
    """The whole sequence in SimChange layout (a symlink view when the folder uses the rgb/ layout)."""
    n = n_frames(seq)
    if (seq / "left").is_dir() and odom_file is None:
        return seq
    return make_chunk(seq, out, 0, n, odom_file=odom_file)


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
    voc = str(ORB_VOC)
    map_poses = out / "map_poses.txt"
    rd = ["--sensor", args.orb_sensor]
    if args.baseline is not None:
        rd += ["--right-dir", json.loads((m / "calib.json").read_text())["right_dirs"][f"{args.baseline:.2f}"]]
    if args.orb_sensor == "rgbd":
        rd += ["--depth-dir", "depth_mm"]          # the views expose uint16 millimetre depth as depth_mm/
    orb = binary("orbslam3_reloc")
    if not (out / "atlas.osa").is_file() or not map_poses.is_file() or not _map_run_ok(out):
        if args.require_map:
            sys.exit(f"no stored ORB-SLAM3 atlas in {out} (--require-map)")
        mv = make_chunk(m, out / "views_map", 0, n_frames(m))
        orb_yaml(m, out / "map.yaml", save_atlas=atlas, fps=fps, baseline=args.baseline)
        rc, dt = run(orb + [voc, str(out / "map.yaml"), str(mv), str(map_poses), "--fps", str(fps)] + rd, out / "map.log", cwd=str(out))
        (out / "map_time.json").write_text(json.dumps({"rc": rc, "seconds": dt, "n_frames": n_frames(m)}))
    if args.map_only:
        return map_poses, [], args.orb_sensor == "mono"
    orb_yaml(q, out / "query.yaml", load_atlas=atlas, fps=fps, baseline=args.baseline)
    # multi-session mode: the query session starts a new map that ORB-SLAM3 merges into the loaded atlas
    # on place recognition (its localization-only mode never relocalizes against a loaded atlas here)
    n_q = n_frames(q)
    trials = build_trials(n_q, args.trial_len, args.trial_stride)
    query_files = []
    for ti, (a, b) in enumerate(trials):
        chunk = make_chunk(q, out / "trials", a, b)
        qp = out / f"query_poses_t{ti}.txt"
        cmd = orb + [voc, str(out / "query.yaml"), str(chunk), str(qp), "--fps", str(fps)] + rd
        rc, dt = run(cmd, out / f"query_t{ti}.log", cwd=str(out))
        for _retry in range(2):
            if rc >= 0:
                break
            rc, dt = run(cmd, out / f"query_t{ti}.log", cwd=str(out))
        (out / f"query_time_t{ti}.json").write_text(json.dumps({"rc": rc, "seconds": dt, "start": a, "end": b}))
        query_files.append((a, b, qp))
        if chunk != q and not args.keep_chunks:
            shutil.rmtree(chunk, ignore_errors=True)
    return map_poses, query_files, args.orb_sensor == "mono"


def run_rtabmap(args, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    m, q = Path(args.map).resolve(), Path(args.query).resolve()
    odom_m = prepare_sequence(m, args.snr, seed=0, cache=out / "odom")
    odom_q = prepare_sequence(q, args.snr, seed=1, cache=out / "odom") if not args.map_only else None
    db = out.resolve() / "map.db"
    map_poses = out / "map_poses.txt"
    extra = ["--Vis/MinInliers", "15", "--Rtabmap/DetectionRate", "0", "--Kp/DetectorStrategy", "6", "--Vis/FeatureType", "6"]
    fps = json.loads((m / "calib.json").read_text()).get("fps", 10.0)
    extra = ["--fps", str(fps)] + extra
    if args.system == "rtabmap_stereo":
        # stereo mode: RTAB-Map computes its depth by stereo matching of the rectified pair (no GT depth)
        calib = json.loads((m / "calib.json").read_text())
        b = args.baseline if args.baseline is not None else float(np.asarray(calib["T_right_in_left"])[0, 3])
        rd = calib["right_dirs"][f"{b:.2f}"] if args.baseline is not None else "right"
        extra = ["--stereo", rd, str(b)] + extra
    rtab = binary("rtabmap_reloc")
    if not db.is_file() or not map_poses.is_file() or not _map_run_ok(out):
        if args.require_map:
            sys.exit(f"no stored RTAB-Map database in {out} (--require-map)")
        mv = make_chunk(m, out / "views_map", 0, n_frames(m), odom_file=odom_m)
        om = mv / "odom.txt"
        rc, dt = run(rtab + [str(mv), str(om), str(db), str(map_poses)] + extra, out / "map.log", cwd=str(out))
        (out / "map_time.json").write_text(json.dumps({"rc": rc, "seconds": dt, "n_frames": n_frames(m)}))
    if args.map_only:
        return map_poses, [], False
    n_q = n_frames(q)
    trials = build_trials(n_q, args.trial_len, args.trial_stride)
    query_files = []
    for ti, (a, b) in enumerate(trials):
        chunk = make_chunk(q, out / "trials", a, b, odom_file=odom_q)
        qp = out / f"query_poses_t{ti}.txt"
        rc, dt = run(rtab + [str(chunk), str(chunk / "odom.txt"), str(db), str(qp), "--localization"] + extra, out / f"query_t{ti}.log", cwd=str(out))
        (out / f"query_time_t{ti}.json").write_text(json.dumps({"rc": rc, "seconds": dt, "start": a, "end": b}))
        query_files.append((a, b, qp))
        if not args.keep_chunks:
            shutil.rmtree(chunk, ignore_errors=True)
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
    ap.add_argument("--orb-sensor", choices=["stereo", "rgbd", "mono"], default="stereo", help="ORB-SLAM3 input")
    ap.add_argument("--map-only", action="store_true", help="map the map sequence only (single-session accuracy)")
    ap.add_argument("--keep-chunks", action="store_true", help="keep the per-trial symlink folders")
    ap.add_argument("--require-map", action="store_true", help="fail instead of mapping when --out holds no stored map")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if args.system in ("rtabmap", "rtabmap_stereo") and False:
        pass
    if args.system == "orbslam3":
        mp, qp, sim3 = run_orbslam3(args, out)
    else:
        mp, qp, sim3 = run_rtabmap(args, out)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    if not args.map_only:
        evaluate_trials(mp, qp, args.map, args.query, out, sim3=sim3, r_d=args.r_d)


if __name__ == "__main__":
    main()
