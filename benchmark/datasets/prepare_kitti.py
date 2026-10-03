#!/usr/bin/env python3
"""Prepare the KITTI odometry sequences (from the raw synced + rectified drives) for the CROSS benchmark.

  <out>/<seq>/stereo/  left/ right/ (symlinks to image_02 / image_03, the colour pair, 0.54 m baseline),
                       poses_left.txt (KITTI odometry ground truth moved from camera 0 to camera 2),
                       odom_left.txt (dead reckoning of the OXTS velocities and yaw rate with the INS roll / pitch,
                       no GPS position),
                       times.txt, calib.json

The same folder serves the stereo, monocular (left only) and RGB-D* (left + SGBM depth from the pair) setups.

  python benchmark/datasets/prepare_kitti.py <kitti_raw_root> <out_root> [--seqs 00 05 ...]

<kitti_raw_root> holds <date>/<drive>_sync/, <date>/calib_*.txt and poses/<seq>.txt (data_odometry_poses.zip).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cross.dataloader.kitti_calib import _inv, load_kitti_calibration  # noqa: E402

# KITTI odometry devkit: sequence -> raw drive and frame range (inclusive)
SEQUENCES = {
    "00": ("2011_10_03", "2011_10_03_drive_0027", 0, 4540),
    "01": ("2011_10_03", "2011_10_03_drive_0042", 0, 1100),
    "02": ("2011_10_03", "2011_10_03_drive_0034", 0, 4660),
    "04": ("2011_09_30", "2011_09_30_drive_0016", 0, 270),
    "05": ("2011_09_30", "2011_09_30_drive_0018", 0, 2760),
    "06": ("2011_09_30", "2011_09_30_drive_0020", 0, 1100),
    "07": ("2011_09_30", "2011_09_30_drive_0027", 0, 1100),
    "08": ("2011_09_30", "2011_09_30_drive_0028", 1100, 5170),
    "09": ("2011_09_30", "2011_09_30_drive_0033", 0, 1590),
    "10": ("2011_09_30", "2011_09_30_drive_0034", 0, 1200),
}


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """xi = (v, w) * dt in the body frame -> 4x4 increment (first order in translation is enough at 10 Hz)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(xi[3:]).as_matrix()
    T[:3, 3] = xi[:3]
    return T


def _rot(axis: str, a: float) -> np.ndarray:
    return Rotation.from_euler(axis, a).as_matrix()


def oxts_dead_reckoning(oxts_dir: Path, lo: int, hi: int, mode: str = "level") -> np.ndarray:
    """IMU poses by dead reckoning of the OXTS velocities and angular rates (no GPS position).

    The OXTS velocities vf, vl, vu are given in the level frame (forward / left parallel to the earth surface, up), the
    angular rates wx, wy, wz in the body frame.  mode "level": positions integrate the level-frame velocities rotated by
    the heading integrated from the level-frame yaw rate wu; the attitude is that heading with the INS roll and pitch
    (gravity-referenced).  mode "body_rates" (the benchmark's odometry before 2026-10-03) integrated the level-frame
    velocities as body-frame velocities with the body rates: the translation direction then disagrees with the
    orientation by the vehicle's pitch / roll relative to the level (KITTI: 0.5-2.3 deg of pitch), which a camera sees
    as a constant tilt of every odometry translation (odometry ATE 9.3 m mean over 00-10 against 4.9 m)."""
    stamps = [datetime.strptime(s.strip()[:26], "%Y-%m-%d %H:%M:%S.%f").timestamp()
              for s in (oxts_dir / "timestamps.txt").read_text().splitlines()]
    files = sorted((oxts_dir / "data").glob("*.txt"))
    D = np.stack([np.loadtxt(files[i]) for i in range(lo, hi + 1)])
    t = np.asarray(stamps[lo:hi + 1])
    poses = [np.eye(4)]
    if mode == "body_rates":
        T = np.eye(4)
        for k in range(1, len(D)):
            v = 0.5 * (D[k - 1, 8:11] + D[k, 8:11])
            w = 0.5 * (D[k - 1, 17:20] + D[k, 17:20])
            T = T @ se3_exp(np.concatenate([v, w]) * (t[k] - t[k - 1]))
            poses.append(T.copy())
        return np.stack(poses), t
    if mode != "level":
        raise ValueError(f"unknown dead-reckoning mode {mode}")
    att = lambda k, psi: _rot("z", psi) @ _rot("y", D[k, 4]) @ _rot("x", D[k, 3])     # roll 3, pitch 4 (KITTI devkit order)
    T0 = np.eye(4)
    T0[:3, :3] = att(0, 0.0)
    T0_inv = np.linalg.inv(T0)
    psi, p = 0.0, np.zeros(3)
    for k in range(1, len(D)):
        dt = t[k] - t[k - 1]
        wu = 0.5 * (D[k - 1, 22] + D[k, 22])               # yaw rate about the up axis (level frame)
        v = 0.5 * (D[k - 1, 8:11] + D[k, 8:11])            # vf, vl, vu (level frame)
        p = p + _rot("z", psi + 0.5 * wu * dt) @ v * dt
        psi += wu * dt
        T = np.eye(4)
        T[:3, :3] = att(k, psi)
        T[:3, 3] = p
        poses.append(T0_inv @ T)
    return np.stack(poses), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--seqs", nargs="*", default=list(SEQUENCES))
    ap.add_argument("--odometry", choices=["level", "body_rates"], default="level",
                    help="dead reckoning of the OXTS data (see oxts_dead_reckoning); body_rates = the benchmark before 2026-10-03")
    ap.add_argument("--odometry-only", action="store_true", help="rewrite odom_left.txt only (no images, poses, calib)")
    a = ap.parse_args()
    root, out_root = Path(a.root), Path(a.out)
    for s in a.seqs:
        date, drive, lo, hi = SEQUENCES[s]
        d = root / date / f"{drive}_sync"
        if not d.is_dir():
            print(f"{s}: {d} missing, skipped")
            continue
        out = out_root / s / "stereo"
        for sub in ("left", "right"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        calib = load_kitti_calibration(root / date)
        # odometry ground truth: rectified camera 0, first frame = world
        P0 = np.loadtxt(root / "poses" / f"{s}.txt").reshape(-1, 3, 4)
        G0 = np.tile(np.eye(4), (len(P0), 1, 1))
        G0[:, :3, :] = P0
        assert len(G0) == hi - lo + 1, f"{s}: {len(G0)} poses, frames {lo}..{hi}"
        G2 = G0 @ _inv(calib["cam2_from_rect0"])          # camera 2 = rectified camera 0 shifted along x
        imu, times = oxts_dead_reckoning(d / "oxts", lo, hi, a.odometry)
        O2 = imu @ _inv(calib["cam2_from_imu"])
        if a.odometry_only:
            np.savetxt(out / "odom_left.txt", O2.reshape(-1, 16), fmt="%.9f")
            print(f"{s}: odom_left.txt rewritten ({a.odometry})")
            continue
        for k, i in enumerate(range(lo, hi + 1)):
            for sub, cam in (("left", "image_02"), ("right", "image_03")):
                dst = out / sub / f"{k:06d}.png"
                if not dst.is_symlink():
                    os.symlink(os.path.abspath(d / cam / "data" / f"{i:010d}.png"), dst)
        np.savetxt(out / "poses_left.txt", G2.reshape(-1, 16), fmt="%.9f")
        np.savetxt(out / "odom_left.txt", O2.reshape(-1, 16), fmt="%.9f")
        np.savetxt(out / "times.txt", times, fmt="%.6f")
        import cv2
        h, w = cv2.imread(str(out / "left" / "000000.png")).shape[:2]
        T_rl = np.asarray(calib["right_in_left"])
        (out / "calib.json").write_text(json.dumps({
            "K": np.asarray(calib["K_left"]).tolist(), "width": w, "height": h, "fps": 10.0,
            "baseline": float(np.linalg.norm(T_rl[:3, 3])), "T_right_in_left": T_rl.tolist(),
            "dataset": "kitti", "sensor": "cam2/cam3 colour (rectified)", "source": f"{s} = {drive} [{lo}, {hi}]",
            "odometry": "OXTS level-frame velocity + yaw-rate dead reckoning, INS roll / pitch (odom_left.txt)"}, indent=1))
        # drift of the dead reckoning over the sequence, for the record
        dist = np.linalg.norm(np.diff(G2[:, :3, 3], axis=0), axis=1).sum()
        rel = _inv(G2[0]) @ G2[-1]
        rel_o = _inv(O2[0]) @ O2[-1]
        print(f"{s}: {hi - lo + 1} frames, {dist:.0f} m, dead-reckoning end-point error "
              f"{np.linalg.norm(rel[:3, 3] - rel_o[:3, 3]):.1f} m", flush=True)


if __name__ == "__main__":
    main()
