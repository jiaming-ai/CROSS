"""Two-iPhone stereo rig recording -> CROSS stereo folder (left/ right/ poses_left.txt odom_left.txt times.txt
calib.json [gnss.txt] baseline_per_frame.txt), with a rig calibration from scripts/rig/calibrate_iphone_rig.py.

Phone B is the left camera, A the right one.  Every B frame is paired with the A frame closest in time (the rig's clock
offset), and the pair is rectified with its own relative pose: the calibrated rig transform composed with B's motion
between the two exposures (ARKit), so imperfect synchronisation does not misalign the rows.  All pairs are resampled
to one pinhole camera (--size, --focal).  The left camera's rectified pose (B's ARKit pose, OpenCV axes, with the pair's
rectification rotation) is the odometry (odom_left.txt, gravity-aligned world: calib.json odom_world_up) and, without
ground truth, also poses_left.txt.  The baseline in calib.json is the median over the pairs (baseline_per_frame.txt
keeps every pair's).

  python scripts/rig/prepare_iphone_stereo.py --a rec_A --b rec_B --calib rig.yaml --out data/walk1 \
      [--rotate cw] [--size 480 640] [--focal 470] [--workers 16]

Recordings are folders written by the capture app or folders inside a capture zip ("capture.zip::inner/rec_B").
--rotate: how the raw images are turned upright (cw for phones held in portrait with the top to the left; none).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cross.rig.recording import Recording  # noqa: E402
from cross.rig.selfcalib import estimate_clock_offset, pair_frames, pair_pose  # noqa: E402

ROTATIONS = {
    "none": (np.eye(3), None),
    "cw": (np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]), cv2.ROTATE_90_CLOCKWISE),       # raw cam -> upright cam
    "ccw": (np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]]), cv2.ROTATE_90_COUNTERCLOCKWISE),
}


def rotate_K(K, W, H, how):
    """Intrinsics after turning a W x H image (cw: new u = H-1-v, new v = u; ccw: new u = v, new v = W-1-u)."""
    if how == "none":
        return K.copy()
    if how == "cw":
        return np.array([[K[1, 1], 0, H - 1 - K[1, 2]], [0, K[0, 0], K[0, 2]], [0, 0, 1.0]])
    return np.array([[K[1, 1], 0, K[1, 2]], [0, K[0, 0], W - 1 - K[0, 2]], [0, 0, 1.0]])


_RECS = None


def _init(spec_a, spec_b):
    global _RECS
    _RECS = (Recording.load(spec_a), Recording.load(spec_b))
    cv2.setNumThreads(1)


def _write(job):
    """Rectify one pair: (iB, jA, R_lr, t_lr, KL, KR, up_size, K_out, out_size, rot, dst_l, dst_r) -> R1."""
    iB, jA, R_lr, t_lr, KL, KR, up_size, K_out, out_size, rot, dst_l, dst_r = job
    recA, recB = _RECS
    R1, R2, *_ = cv2.stereoRectify(KL, np.zeros(5), KR, np.zeros(5), up_size, R_lr, t_lr,
                                   flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    flag = ROTATIONS[rot][1]
    for rec, i, K, Rr, dst in ((recB, iB, KL, R1, dst_l), (recA, jA, KR, R2, dst_r)):
        img = rec.image(i, gray=False)
        if flag is not None:
            img = cv2.rotate(img, flag)
        mx, my = cv2.initUndistortRectifyMap(K, np.zeros(5), Rr, K_out, out_size, cv2.CV_16SC2)
        cv2.imwrite(str(dst), cv2.remap(img, mx, my, cv2.INTER_AREA))
    return R1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="recording of phone A (right)")
    ap.add_argument("--b", required=True, help="recording of phone B (left)")
    ap.add_argument("--calib", type=Path, required=True, help="rig YAML of calibrate_iphone_rig.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rotate", choices=list(ROTATIONS), default="cw")
    ap.add_argument("--size", type=int, nargs=2, default=[480, 640], metavar=("W", "H"))
    ap.add_argument("--focal", type=float, default=470.0)
    ap.add_argument("--max-dt", type=float, default=0.06, help="pairs further apart in time are dropped (s)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    cal = yaml.safe_load(a.calib.read_text())
    R, T = np.asarray(cal["R"], float), np.asarray(cal["T"], float).reshape(3)
    recA, recB = Recording.load(a.a), Recording.load(a.b)
    name = Path(str(a.b).split("::")[-1].rstrip("/")).name
    d = (cal.get("clock_offsets_s") or {}).get(name)
    if d is None:
        d, corr = estimate_clock_offset(recA, recB)
        print(f"clock offset estimated: {1000 * d:+.1f} ms (correlation {corr:.3f})")
    iB, jA, dt = pair_frames(recA, recB, float(d), max_dt=a.max_dt)
    P = pair_pose(recB, R, T, float(d), recB.t[iB], recA.t[jA])          # A exposure in B exposure: x_B = R x_A + t
    Rr, how = ROTATIONS[a.rotate]
    W, H = recB.size
    up_size = (H, W) if a.rotate != "none" else (W, H)
    Wo, Ho = a.size
    K_out = np.array([[a.focal, 0, (Wo - 1) / 2], [0, a.focal, (Ho - 1) / 2], [0, 0, 1.0]])
    for sub in ("left", "right"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    jobs = []
    for k, (i, j) in enumerate(zip(iB, jA)):
        Rp, tp = P[k, :3, :3], P[k, :3, 3]
        R_lr = Rr @ Rp.T @ Rr.T                                       # left (B) -> right (A), upright cameras
        t_lr = -Rr @ Rp.T @ tp
        KL = rotate_K(recB.K(int(i)), W, H, a.rotate)
        KR = rotate_K(recA.K(int(j)), W, H, a.rotate)
        jobs.append((int(i), int(j), R_lr, t_lr, KL, KR, up_size, K_out, (Wo, Ho), a.rotate,
                     a.out / "left" / f"{k:06d}.png", a.out / "right" / f"{k:06d}.png"))
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.a, a.b)) as ex:
        R1s = list(ex.map(_write, jobs, chunksize=16))
    # left rectified camera pose: B (OpenCV axes) -> upright -> rectified
    poses = []
    for i, R1 in zip(iB, R1s):
        M = np.eye(4)
        M[:3, :3] = Rr.T @ R1.T
        poses.append(recB.T[int(i)] @ M)
    poses = np.stack(poses)
    times = recB.t[iB]
    base = np.linalg.norm(P[:, :3, 3], axis=1)
    np.savetxt(a.out / "poses_left.txt", poses.reshape(-1, 16), fmt="%.9f")
    np.savetxt(a.out / "odom_left.txt", poses.reshape(-1, 16), fmt="%.9f")
    np.savetxt(a.out / "times.txt", times - times[0], fmt="%.6f")
    np.savetxt(a.out / "baseline_per_frame.txt", np.c_[np.arange(len(base)), base, 1000 * dt], fmt="%.6f",
               header="frame baseline_m dt_ms (A exposure - B exposure)")
    loc_rel = "location.csv"
    if recB.root.exists(loc_rel):
        import io
        loc = pd.read_csv(io.BytesIO(recB.root.read(loc_rel)))
        g = loc[(loc.unix_time_s >= times[0]) & (loc.unix_time_s <= times[-1]) & (loc.horizontal_accuracy_m > 0)]
        np.savetxt(a.out / "gnss.txt", np.c_[g.unix_time_s - times[0], g.latitude, g.longitude, g.altitude_m,
                                            np.full((len(g), 3), np.nan), g.horizontal_accuracy_m], fmt="%.9f",
                   header="t lat lon alt mode num_sats hdop sigma_h (phone B, CoreLocation)")
    baseline = float(np.median(base))
    Tlr = np.eye(4)
    Tlr[0, 3] = baseline
    dur = float(times[-1] - times[0])
    calib = {"K": K_out.tolist(), "width": Wo, "height": Ho, "fps": round(len(times) / dur, 3) if dur > 0 else 10.0,
             "baseline": baseline, "T_right_in_left": Tlr.tolist(),
             "left": "phone B (upright, rectified per pair)", "right": "phone A",
             "odometry": "ARKit (phone B) camera poses", "gt": "none: poses_left.txt = ARKit odometry",
             "odom_world_up": [0.0, 1.0, 0.0],
             "clock_offset_ms": 1000 * float(d), "pair_dt_ms_abs_median": float(np.median(np.abs(1000 * dt))),
             "baseline_per_pair_p5_p95": [float(np.percentile(base, 5)), float(np.percentile(base, 95))],
             "rig_calibration": str(a.calib), "source_a": str(a.a), "source_b": str(a.b)}
    (a.out / "calib.json").write_text(json.dumps(calib, indent=1))
    print(f"{a.out}: {len(times)} pairs, {dur:.0f} s, baseline {baseline:.4f} m "
          f"(p5-p95 {calib['baseline_per_pair_p5_p95'][0]:.4f}-{calib['baseline_per_pair_p5_p95'][1]:.4f}), "
          f"|dt| median {calib['pair_dt_ms_abs_median']:.1f} ms")


if __name__ == "__main__":
    main()
