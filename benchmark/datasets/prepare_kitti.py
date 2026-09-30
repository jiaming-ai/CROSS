#!/usr/bin/env python3
"""Prepare the KITTI odometry sequences (from the raw synced + rectified drives) for the CROSS benchmark.

  <out>/<seq>/stereo/  left/ right/ (symlinks to image_02 / image_03, the colour pair, 0.54 m baseline),
                       poses_left.txt (KITTI odometry ground truth moved from camera 0 to camera 2),
                       odom_left.txt (dead reckoning of the OXTS velocities and angular rates, no GPS),
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


def oxts_dead_reckoning(oxts_dir: Path, lo: int, hi: int) -> np.ndarray:
    """IMU poses from integrating the OXTS body-frame velocities (vf, vl, vu) and angular rates (wx, wy, wz)."""
    stamps = [datetime.strptime(s.strip()[:26], "%Y-%m-%d %H:%M:%S.%f").timestamp()
              for s in (oxts_dir / "timestamps.txt").read_text().splitlines()]
    files = sorted((oxts_dir / "data").glob("*.txt"))
    T = np.eye(4)
    poses = [T.copy()]
    for i in range(lo + 1, hi + 1):
        prev = np.loadtxt(files[i - 1])
        cur = np.loadtxt(files[i])
        dt = stamps[i] - stamps[i - 1]
        v = 0.5 * (prev[8:11] + cur[8:11])
        w = 0.5 * (prev[17:20] + cur[17:20])
        T = T @ se3_exp(np.concatenate([v, w]) * dt)
        poses.append(T.copy())
    return np.stack(poses), np.asarray(stamps[lo:hi + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--seqs", nargs="*", default=list(SEQUENCES))
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
        imu, times = oxts_dead_reckoning(d / "oxts", lo, hi)
        O2 = imu @ _inv(calib["cam2_from_imu"])
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
            "odometry": "OXTS velocity + angular-rate dead reckoning (odom_left.txt)"}, indent=1))
        # drift of the dead reckoning over the sequence, for the record
        dist = np.linalg.norm(np.diff(G2[:, :3, 3], axis=0), axis=1).sum()
        rel = _inv(G2[0]) @ G2[-1]
        rel_o = _inv(O2[0]) @ O2[-1]
        print(f"{s}: {hi - lo + 1} frames, {dist:.0f} m, dead-reckoning end-point error "
              f"{np.linalg.norm(rel[:3, 3] - rel_o[:3, 3]):.1f} m", flush=True)


if __name__ == "__main__":
    main()
