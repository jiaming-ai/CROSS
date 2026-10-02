#!/usr/bin/env python3
"""Prepare ROVER recordings (Schmidt et al., T-RO 2025) for the CROSS benchmark, reading the frames straight from the
downloaded zip (extracting ~150k files per recording onto a network drive takes hours).

  python benchmark/datasets/prepare_rover.py <rover_root> <out_root> [--names campus_large_day_2024-09-25 ...]

<rover_root> holds <name>.zip (download_rover.sh with NO_UNZIP=1) and calibration/.  For every recording:

  <out>/<name>/rgbd/    D435i colour (undistorted to a pinhole camera) + registered depth (same undistortion, uint16 mm,
                        invalid = 0); also the monocular setup
  <out>/<name>/stereo/  T265 fisheye pair rectified to a pinhole stereo pair (640x480, 90 deg HFOV)

with poses_left.txt, times.txt, calib.json, all at 10 Hz.

Common frame.  The total station was set up anew for every recording, so each recording's ground truth is in its own
frame.  With --align-to <map recording>, the prism track is registered to the map recording's track (the robot drives the
same lawn-edge route): 2-D ICP over x, y and heading from several initial headings, plus the median height offset.  The
residual distances between the registered tracks are stored in calib.json ("gt_alignment").

Ground truth.  ROVER's ground truth is the 3-D position of a prism on the robot tracked by a total station (no
orientation).  The robot drives forward (differential drive, 0.5 m/s), so its heading is the direction of travel of
the smoothed prism track; roll and pitch are taken as zero.  The camera pose is then T_world_prism * T_prism_cam with the
extrinsics of calibration/calib_*.yaml (prism frame: x forward, y left, z up).  With the ~0.5 m prism-camera lever arm, a
2 deg heading error moves the camera by ~2 cm.  There is no wheel odometry: the benchmark simulates odometry from these
poses (SNR 10), so no odom_left.txt is written.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

RECT_W, RECT_H, RECT_HFOV_DEG = 640, 480, 90.0
FPS = 10.0
GT_LABEL = "prism position (y negated: right-handed) + heading of travel"


def _inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def read_calib(root: Path):
    d = yaml.safe_load((root / "calibration/calib_d435i.yaml").read_text())
    t = yaml.safe_load((root / "calibration/calib_t265.yaml").read_text())
    return d, t


def stamped(text: str):
    ts, paths = [], []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 2 and not line.startswith("#"):
            ts.append(float(p[0]))
            paths.append(p[1])
    o = np.argsort(ts)
    return np.asarray(ts)[o], [paths[i] for i in o]


def heading_track(t_gt, p_gt, window=0.5, min_speed=0.1):
    """Yaw of the direction of travel at the ground-truth times (unwrapped; held through slow segments)."""
    yaw = np.full(len(t_gt), np.nan)
    for i, t in enumerate(t_gt):
        a = np.searchsorted(t_gt, t - window)
        b = min(np.searchsorted(t_gt, t + window), len(t_gt) - 1)
        if b <= a:
            continue
        v = (p_gt[b, :2] - p_gt[a, :2]) / max(t_gt[b] - t_gt[a], 1e-6)
        if np.linalg.norm(v) >= min_speed:
            yaw[i] = np.arctan2(v[1], v[0])
    ok = np.isfinite(yaw)
    if ok.sum() < 2:
        raise ValueError("robot never moves")
    yaw_u = np.unwrap(yaw[ok])
    return np.interp(t_gt, t_gt[ok], yaw_u)


def prism_poses(t_gt, p_gt, yaw_gt, t):
    """T_world_prism at times t (NaN outside the ground-truth range)."""
    out = np.full((len(t), 4, 4), np.nan)
    ok = (t >= t_gt[0]) & (t <= t_gt[-1])
    for k in np.where(ok)[0]:
        y = np.interp(t[k], t_gt, yaw_gt)
        T = np.eye(4)
        T[:3, :3] = [[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]]
        T[:3, 3] = [np.interp(t[k], t_gt, p_gt[:, j]) for j in range(3)]
        out[k] = T
    return out


def read_gt(z, top):
    """Prism track (times, positions) in a right-handed frame.  The total-station coordinates in groundtruth.txt are
    left-handed: the track turns opposite to the VN-100 gyro (z up) and to visual odometry of both cameras (day: track
    +1083 deg over three laps, gyro -1089 deg), so y is negated."""
    names = set(z.namelist())
    member = f"{top}/groundtruth.txt" if f"{top}/groundtruth.txt" in names else f"{top}/groundtruth"   # night-light: no suffix
    rows = [l.split() for l in z.read(member).decode().splitlines() if l.strip() and not l.startswith("#")]
    g = np.asarray([[float(v) for v in r[:4]] for r in rows])
    g = g[np.argsort(g[:, 0])]
    p = g[:, 1:4] * np.array([1.0, -1.0, 1.0])
    check_handedness(z, top, g[:, 0], p)
    return g[:, 0], p


def check_handedness(z, top, t_gt, p_gt):
    """Fails unless the net turn of the prism track has the sign of the integrated VN-100 z gyro (prism frame, z up)."""
    rows = [l.split() for l in z.read(f"{top}/vn100/imu.txt").decode().splitlines() if l.strip() and not l.startswith("#")]
    a = np.asarray([[float(r[0]), float(r[7])] for r in rows])
    a = a[(a[:, 0] > t_gt[0]) & (a[:, 0] < t_gt[-1])]
    gyro = np.degrees(np.sum(a[:-1, 1] * np.diff(a[:, 0])))
    keep = [0]                                    # travel direction over steps of > 0.3 m
    for i in range(1, len(p_gt)):
        if np.linalg.norm(p_gt[i, :2] - p_gt[keep[-1], :2]) > 0.3:
            keep.append(i)
    d = np.diff(p_gt[keep, :2], axis=0)
    yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    track = np.degrees(yaw[-1] - yaw[0])
    if abs(gyro) > 180 and np.sign(gyro) != np.sign(track):
        raise RuntimeError(f"{top}: ground-truth track turns {track:+.0f} deg, gyro {gyro:+.0f} deg (mirrored frame?)")


def register_track(src, dst, iters=60):
    """4x4 transform (rotation about z, translation) mapping the prism track `src` onto `dst`, and the residuals (m)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(dst[:, :2])
    best = None
    for yaw0 in np.radians(np.arange(0, 360, 15)):
        R = np.array([[np.cos(yaw0), -np.sin(yaw0)], [np.sin(yaw0), np.cos(yaw0)]])
        t = dst[:, :2].mean(0) - R @ src[:, :2].mean(0)
        for _ in range(iters):
            d, j = tree.query((R @ src[:, :2].T).T + t)
            m = d < np.percentile(d, 90)
            a, b = src[m, :2], dst[j[m], :2]
            ma, mb = a.mean(0), b.mean(0)
            U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
            if np.linalg.det(Vt.T @ U.T) < 0:
                Vt[1] *= -1
            R = Vt.T @ U.T
            t = mb - R @ ma
        d, _ = tree.query((R @ src[:, :2].T).T + t)
        if best is None or np.median(d) < np.median(best[2]):
            best = (R, t, d)
    R, t, d = best
    T = np.eye(4)
    T[:2, :2] = R
    T[:2, 3] = t
    T[2, 3] = np.median(dst[:, 2]) - np.median(src[:, 2])
    return T, d


def select_10hz(ts, t0, t1):
    """Indices of the frames nearest to a 10 Hz grid over [t0, t1] (each frame at most once, |dt| < 20 ms)."""
    grid = np.arange(t0, t1, 1.0 / FPS)
    idx = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1)
    idx = np.where(np.abs(ts[idx - 1] - grid) < np.abs(ts[idx] - grid), idx - 1, idx)
    keep = np.abs(ts[idx] - grid) < 0.02
    idx = idx[keep]
    return np.unique(idx)


def write_common(out: Path, poses, times, calib):
    np.savetxt(out / "poses_left.txt", np.asarray(poses).reshape(-1, 16), fmt="%.9f")
    np.savetxt(out / "times.txt", np.asarray(times), fmt="%.6f")
    (out / "calib.json").write_text(json.dumps(calib, indent=1))     # written last: marks the folder complete


def prepare_rgbd(z, top, cal_d, gt, out: Path):
    t_rgb, p_rgb = stamped(z.read(f"{top}/realsense_D435i/rgb.txt").decode())
    t_dep, p_dep = stamped(z.read(f"{top}/realsense_D435i/depth.txt").decode())
    ci = cal_d["Cam_Intrinsics"]
    fx, fy, cx, cy = ci["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    D = np.asarray(ci["distortion_coeffs"], dtype=np.float64)        # radtan k1 k2 p1 p2
    first = cv2.imdecode(np.frombuffer(z.read(f"{top}/realsense_D435i/{p_rgb[0]}"), np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    Knew, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0.0)        # only valid pixels
    mx, my = cv2.initUndistortRectifyMap(K, D, None, Knew, (w, h), cv2.CV_32FC1)
    T_prism_cam = np.asarray(cal_d["Cam-To-Prism"], dtype=np.float64)
    t_gt, p_gt, yaw_gt = gt
    sel = select_10hz(t_rgb, max(t_rgb[0], t_gt[0]), min(t_rgb[-1], t_gt[-1]))
    P = prism_poses(t_gt, p_gt, yaw_gt, t_rgb[sel])
    for sub in ("rgb", "depth"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    poses, times, k = [], [], 0
    for n, i in enumerate(sel):
        j = int(np.argmin(np.abs(t_dep - t_rgb[i])))
        if abs(t_dep[j] - t_rgb[i]) > 0.02 or not np.isfinite(P[n]).all():
            continue
        rgb = cv2.imdecode(np.frombuffer(z.read(f"{top}/realsense_D435i/{p_rgb[i]}"), np.uint8), cv2.IMREAD_COLOR)
        dep = cv2.imdecode(np.frombuffer(z.read(f"{top}/realsense_D435i/{p_dep[j]}"), np.uint8), cv2.IMREAD_UNCHANGED)
        dep[dep >= 65535] = 0
        cv2.imwrite(str(out / "rgb" / f"{k:06d}.png"), cv2.remap(rgb, mx, my, cv2.INTER_LINEAR))
        cv2.imwrite(str(out / "depth" / f"{k:06d}.png"), cv2.remap(dep, mx, my, cv2.INTER_NEAREST))
        poses.append(P[n] @ T_prism_cam)
        times.append(t_rgb[i])
        k += 1
    write_common(out, poses, times, {
        "K": Knew.tolist(), "width": w, "height": h, "fps": FPS, "dataset": "rover", "sensor": "d435i (undistorted)",
        "source": top, "depth": "D435i depth registered to colour, uint16 mm, 0 = invalid",
        "odometry": "none recorded: simulated from ground truth by the benchmark", "gt": GT_LABEL})
    return k


def t265_member(z, top, side, p):
    """Zip member of a T265 image: the txt files say rgb/<file>, the images live in cam_left / cam_right."""
    cand = f"{top}/realsense_T265/cam_{side}/{Path(p).name}"
    return cand if cand in set(z.namelist()) else f"{top}/realsense_T265/{p}"


def stereo_rectification(z, top, cal_t):
    """Rectification of the T265 pair: (remap tables left/right, rectified K, T_prism_rect, baseline)."""
    _, pl = stamped(z.read(f"{top}/realsense_T265/cam_left.txt").decode())

    def intr(key):
        c = cal_t[key]
        fx, fy, cx, cy = c["intrinsics"]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), np.asarray(c["distortion_coeffs"], dtype=np.float64)

    K1, D1 = intr("CamLeft_Intrinsics")
    K2, D2 = intr("CamRight_Intrinsics")
    T_rl = np.asarray(cal_t["CamRight-To-CamLeft"], dtype=np.float64)    # Kalibr: x_right = R x_left + t
    first = cv2.imdecode(np.frombuffer(z.read(t265_member(z, top, "left", pl[0])), np.uint8), cv2.IMREAD_UNCHANGED)
    size = (first.shape[1], first.shape[0])
    R1, R2, _, _, _ = cv2.fisheye.stereoRectify(K1, D1, K2, D2, size, T_rl[:3, :3], T_rl[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY)
    f = (RECT_W / 2) / np.tan(np.radians(RECT_HFOV_DEG) / 2)
    K = np.array([[f, 0, (RECT_W - 1) / 2], [0, f, (RECT_H - 1) / 2], [0, 0, 1.0]])
    maps = [cv2.fisheye.initUndistortRectifyMap(Kc, Dc, Rc, K, (RECT_W, RECT_H), cv2.CV_16SC2)
            for Kc, Dc, Rc in ((K1, D1, R1), (K2, D2, R2))]
    T_cam_rect = np.eye(4)
    T_cam_rect[:3, :3] = R1.T
    T_prism_rect = np.asarray(cal_t["CamLeft-To-Prism"], dtype=np.float64) @ T_cam_rect
    baseline = float(np.linalg.norm(T_rl[:3, 3]))
    return maps, K, T_prism_rect, baseline


def prepare_stereo(z, top, cal_t, gt, out: Path):
    tl, pl = stamped(z.read(f"{top}/realsense_T265/cam_left.txt").decode())
    tr, pr = stamped(z.read(f"{top}/realsense_T265/cam_right.txt").decode())
    member = lambda side, p: t265_member(z, top, side, p)
    maps, K, T_prism_rect, baseline = stereo_rectification(z, top, cal_t)
    t_gt, p_gt, yaw_gt = gt
    sel = select_10hz(tl, max(tl[0], t_gt[0]), min(tl[-1], t_gt[-1]))
    P = prism_poses(t_gt, p_gt, yaw_gt, tl[sel])
    for sub in ("left", "right"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    poses, times, k = [], [], 0
    for n, i in enumerate(sel):
        j = int(np.argmin(np.abs(tr - tl[i])))
        if abs(tr[j] - tl[i]) > 0.02 or not np.isfinite(P[n]).all():
            continue
        for sub, side, p, (mx, my) in (("left", "left", pl[i], maps[0]), ("right", "right", pr[j], maps[1])):
            img = cv2.imdecode(np.frombuffer(z.read(member(side, p)), np.uint8), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(str(out / sub / f"{k:06d}.png"), cv2.remap(img, mx, my, cv2.INTER_LINEAR))
        poses.append(P[n] @ T_prism_rect)
        times.append(tl[i])
        k += 1
    T = np.eye(4)
    T[0, 3] = baseline
    write_common(out, poses, times, {
        "K": K.tolist(), "width": RECT_W, "height": RECT_H, "fps": FPS, "baseline": baseline, "T_right_in_left": T.tolist(),
        "dataset": "rover", "sensor": "t265 (rectified, grayscale)", "source": top,
        "odometry": "none recorded: simulated from ground truth by the benchmark", "gt": GT_LABEL})
    return k


def rewrite_poses(z, top, cal_d, cal_t, gt, o: Path, setup, T_align, info):
    """Ground-truth poses of a prepared folder at its frame times (same frames: the ground-truth range is unchanged)."""
    t = np.loadtxt(o / "times.txt").reshape(-1)
    T_prism_cam = np.asarray(cal_d["Cam-To-Prism"], dtype=np.float64) if setup == "rgbd" else stereo_rectification(z, top, cal_t)[2]
    P = prism_poses(*gt, t) @ T_prism_cam
    if not np.isfinite(P).all():
        raise RuntimeError(f"{o}: frames outside the ground-truth range")
    np.savetxt(o / "poses_left.txt", (T_align[None] @ P).reshape(-1, 16), fmt="%.9f")
    calib = json.loads((o / "calib.json").read_text())
    calib["gt"] = GT_LABEL
    if info is not None:
        calib["gt_alignment"] = info
    (o / "calib.json").write_text(json.dumps(calib, indent=1))
    print(f"{o}: {len(t)} poses rewritten" + (f" (registered to {info['to']}, residual median "
          f"{info['residual_median_m']:.2f} m)" if info else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--names", nargs="*", default=None, help="recordings (zip stems); default: every *.zip.done")
    ap.add_argument("--setups", nargs="*", default=["rgbd", "stereo"])
    ap.add_argument("--align-to", default=None, help="map recording whose ground-truth frame every recording is registered to")
    ap.add_argument("--poses-only", action="store_true",
                    help="rewrite poses_left.txt (and the registration) of prepared folders from their times.txt, keeping the images")
    a = ap.parse_args()
    root, out_root = Path(a.root), Path(a.out)
    names = a.names or sorted(p.name[:-len(".zip.done")] for p in root.glob("*.zip.done"))
    cal_d, cal_t = read_calib(root)
    ref = None
    if a.align_to:
        zr = zipfile.ZipFile(root / f"{a.align_to}.zip")
        ref = read_gt(zr, zr.namelist()[0].split("/")[0])[1]
    for name in names:
        z = zipfile.ZipFile(root / f"{name}.zip")
        top = z.namelist()[0].split("/")[0]
        t_gt, p_gt = read_gt(z, top)
        gt = (t_gt, p_gt, heading_track(t_gt, p_gt))
        T_align, info = np.eye(4), {"to": name, "residual_median_m": 0.0, "residual_p90_m": 0.0}
        if ref is not None and name != a.align_to:
            # register the shorter route onto the longer one (every point of the shorter route has a counterpart; the
            # 2023 / spring recordings drive a longer route than the September 2024 ones)
            L = lambda p: float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
            if L(p_gt) <= L(ref):
                T_align, d = register_track(p_gt, ref)
            else:
                T_inv, d = register_track(ref, p_gt)
                T_align = np.linalg.inv(T_inv)
            info = {"to": a.align_to, "method": "shorter-onto-longer", "T": T_align.tolist(),
                    "residual_median_m": float(np.median(d)), "residual_p90_m": float(np.percentile(d, 90))}
        for setup in a.setups:
            o = out_root / name / setup
            if a.poses_only:
                rewrite_poses(z, top, cal_d, cal_t, gt, o, setup, T_align, info if ref is not None else None)
                continue
            if not (o / "calib.json").is_file():
                n = (prepare_rgbd(z, top, cal_d, gt, o) if setup == "rgbd" else prepare_stereo(z, top, cal_t, gt, o))
                print(f"{name}/{setup}: {n} frames", flush=True)
            calib = json.loads((o / "calib.json").read_text())
            old = calib.get("gt_alignment", {})
            if ref is not None and (old.get("to") != info["to"] or old.get("method") != info.get("method")):
                # move the poses into the map recording's frame (undoing an earlier registration first)
                P = np.loadtxt(o / "poses_left.txt").reshape(-1, 4, 4)
                if "T" in old:
                    P = np.linalg.inv(np.asarray(old["T"]))[None] @ P
                np.savetxt(o / "poses_left.txt", (T_align[None] @ P).reshape(-1, 16), fmt="%.9f")
                calib["gt_alignment"] = info
                (o / "calib.json").write_text(json.dumps(calib, indent=1))
                print(f"{name}/{setup}: ground truth registered to {info['to']} (residual median "
                      f"{info['residual_median_m']:.2f} m, p90 {info['residual_p90_m']:.2f} m)", flush=True)


if __name__ == "__main__":
    main()
