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
      [--rotate cw] [--size 480 640] [--focal 470] [--workers 16] [--rotation walk] [--rolling-shutter 0.02]

Recordings are folders written by the capture app or folders inside a capture zip ("capture.zip::inner/rec_B").
--rotate: how the raw images are turned upright (cw for phones held in portrait with the top to the left; none).
--rotation walk: the rig rotation (and translation direction) of this recording's own calibration (per_recording in
the rig YAML) instead of the combined one; the baseline length stays the combined one.
--rolling-shutter S: compensate the sensor's row readout (S seconds from the first to the last row of the raw image,
negative for bottom-to-top): every raw pixel is taken from where the camera, turning at its ARKit angular velocity,
saw that ray at its row's time; the frame time stamp is taken as the middle row's.
pairs.txt lists every pair's B and A frame ids and time gap.
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


def rolling_shutter_map(K_raw, R_raw_from_out, K_out, out_size, w_raw, readout, raw_height):
    """Float maps (into the raw, unrotated image) of a rectified view with rolling-shutter compensation: the ray of
    output pixel p is the frame-time camera ray R_raw_from_out K_out^-1 p; the row v that recorded it was exposed
    tau = (v - (H-1)/2) / H * readout later, when the camera had turned by exp(w tau) (w: angular velocity in the raw
    camera axes, rad/s), so the ray is looked up at exp(-w tau) x (two fixed-point iterations on v)."""
    W, H = out_size
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x = np.linalg.inv(K_out) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    x = R_raw_from_out @ x
    y = x
    for _ in range(3):
        p = K_raw @ y
        tau = (p[1] / p[2] - (raw_height - 1) / 2) / raw_height * readout
        th = np.outer(-w_raw, tau)                 # small rotation exp(-w tau) per pixel (first order is ample)
        y = x + np.cross(th.T, x.T).T
    p = K_raw @ y
    return (p[0] / p[2]).reshape(H, W).astype(np.float32), (p[1] / p[2]).reshape(H, W).astype(np.float32)


def _write(job):
    """Rectify one pair: (iB, jA, R_lr, t_lr, KL, KR, up_size, K_out, out_size, rot, dst_l, dst_r, rs) -> R1.
    rs: None, or (readout s, B's angular velocity in raw B axes, A's in raw A axes, raw K of B, raw K of A)."""
    iB, jA, R_lr, t_lr, KL, KR, up_size, K_out, out_size, rot, dst_l, dst_r, rs = job
    recA, recB = _RECS
    R1, R2, *_ = cv2.stereoRectify(KL, np.zeros(5), KR, np.zeros(5), up_size, R_lr, t_lr,
                                   flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    Rrot, flag = ROTATIONS[rot]
    for side, (rec, i, K, Rr, dst) in enumerate(((recB, iB, KL, R1, dst_l), (recA, jA, KR, R2, dst_r))):
        img = rec.image(i, gray=False)
        if rs is None:
            if flag is not None:
                img = cv2.rotate(img, flag)
            mx, my = cv2.initUndistortRectifyMap(K, np.zeros(5), Rr, K_out, out_size, cv2.CV_16SC2)
        else:
            readout, wB, wA, KB_raw, KA_raw = rs
            mx, my = rolling_shutter_map((KB_raw, KA_raw)[side], Rrot.T @ Rr.T, K_out, out_size, (wB, wA)[side],
                                         readout, img.shape[0])
            mx, my = cv2.convertMaps(mx, my, cv2.CV_16SC2)
        cv2.imwrite(str(dst), cv2.remap(img, mx, my, cv2.INTER_AREA))
    return R1


def angular_velocity(rec, t):
    """(n, 3) angular velocity (rad/s, camera axes) of a recording at times t: central differences of its poses,
    linearly interpolated."""
    from scipy.spatial.transform import Rotation
    dR = np.einsum("nji,njk->nik", rec.T[:-1, :3, :3], rec.T[1:, :3, :3])
    w = Rotation.from_matrix(dR).as_rotvec() / np.diff(rec.t)[:, None]
    tm = 0.5 * (rec.t[:-1] + rec.t[1:])
    return np.stack([np.interp(t, tm, w[:, k]) for k in range(3)], 1)


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
    ap.add_argument("--rotation", choices=("rig", "walk"), default="rig",
                    help="rig: the combined calibration; walk: this recording's own rotation / translation direction")
    ap.add_argument("--rolling-shutter", type=float, default=0.0, metavar="S",
                    help="row readout time of the raw image (s; 0 = no compensation)")
    a = ap.parse_args()
    cal = yaml.safe_load(a.calib.read_text())
    R, T = np.asarray(cal["R"], float), np.asarray(cal["T"], float).reshape(3)
    recA, recB = Recording.load(a.a), Recording.load(a.b)
    name = Path(str(a.b).split("::")[-1].rstrip("/")).name
    if a.rotation == "walk":
        own = (cal.get("per_recording") or {}).get(name) or {}
        if "R" not in own:
            sys.exit(f"--rotation walk: {a.calib} has no per_recording R / T for {name} (re-run calibrate_iphone_rig.py)")
        Tw = np.asarray(own["T"], float).reshape(3)
        R, T = np.asarray(own["R"], float), Tw / np.linalg.norm(Tw) * np.linalg.norm(T)
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
    if a.rolling_shutter:
        wB = angular_velocity(recB, recB.t[iB])
        wA = angular_velocity(recA, recA.t[jA])
    jobs = []
    for k, (i, j) in enumerate(zip(iB, jA)):
        Rp, tp = P[k, :3, :3], P[k, :3, 3]
        R_lr = Rr @ Rp.T @ Rr.T                                       # left (B) -> right (A), upright cameras
        t_lr = -Rr @ Rp.T @ tp
        KL = rotate_K(recB.K(int(i)), W, H, a.rotate)
        KR = rotate_K(recA.K(int(j)), W, H, a.rotate)
        jobs.append((int(i), int(j), R_lr, t_lr, KL, KR, up_size, K_out, (Wo, Ho), a.rotate,
                     a.out / "left" / f"{k:06d}.png", a.out / "right" / f"{k:06d}.png",
                     (a.rolling_shutter, wB[k], wA[k], recB.K(int(i)), recA.K(int(j))) if a.rolling_shutter else None))
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
    fid = lambda rec, idx: rec.frames.frame_id.to_numpy()[idx] if "frame_id" in rec.frames else idx  # noqa: E731
    np.savetxt(a.out / "pairs.txt", np.c_[np.arange(len(iB)), fid(recB, iB), fid(recA, jA), 1000 * dt],
               fmt=["%d", "%d", "%d", "%.3f"], header="frame frame_id_B frame_id_A dt_ms (A exposure - B exposure)")
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
             "rig_calibration": str(a.calib), "rig_rotation": a.rotation, "rolling_shutter_s": a.rolling_shutter,
             "source_a": str(a.a), "source_b": str(a.b)}
    (a.out / "calib.json").write_text(json.dumps(calib, indent=1))
    print(f"{a.out}: {len(times)} pairs, {dur:.0f} s, baseline {baseline:.4f} m "
          f"(p5-p95 {calib['baseline_per_pair_p5_p95'][0]:.4f}-{calib['baseline_per_pair_p5_p95'][1]:.4f}), "
          f"|dt| median {calib['pair_dt_ms_abs_median']:.1f} ms")


if __name__ == "__main__":
    main()
