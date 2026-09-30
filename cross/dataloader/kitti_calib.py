"""KITTI Raw calibration chain and OXTS ground-truth poses (after stereo_vggt/kitti.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

EARTH_RADIUS = 6_378_137.0


def _read(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        key, sep, raw = line.partition(":")
        if not sep or key == "calib_time":
            continue
        try:
            out[key] = np.asarray([float(v) for v in raw.split()])
        except ValueError:
            pass
    return out


def _T(R, t):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R).reshape(3, 3)
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def _inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def _rpy(roll, pitch, yaw):
    sr, cr, sp, cp, sy, cy = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch), np.sin(yaw), np.cos(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def oxts_to_imu_poses(oxts_paths) -> np.ndarray:
    packets = [np.loadtxt(p) for p in oxts_paths]
    scale = np.cos(packets[0][0] * np.pi / 180.0)
    poses = []
    for pk in packets:
        lat, lon, alt, roll, pitch, yaw = pk[:6]
        tx = scale * lon * np.pi * EARTH_RADIUS / 180.0
        ty = scale * EARTH_RADIUS * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
        poses.append(_T(_rpy(roll, pitch, yaw), [tx, ty, alt]))
    return np.stack(poses)


def load_kitti_calibration(date_root: Path) -> dict:
    date_root = Path(date_root)
    cam = _read(date_root / "calib_cam_to_cam.txt")
    imu2velo = _read(date_root / "calib_imu_to_velo.txt")
    velo2cam = _read(date_root / "calib_velo_to_cam.txt")
    velo_from_imu = _T(imu2velo["R"], imu2velo["T"])
    cam0_from_velo = _T(velo2cam["R"], velo2cam["T"])
    rect0_from_cam0 = _T(cam["R_rect_00"], np.zeros(3))
    rect0_from_imu = rect0_from_cam0 @ cam0_from_velo @ velo_from_imu
    P2 = cam["P_rect_02"].reshape(3, 4)
    P3 = cam["P_rect_03"].reshape(3, 4)

    def cam_from_rect0(P):
        K = P[:, :3]
        return _T(np.eye(3), np.linalg.solve(K, P[:, 3]))

    cam2_from_rect0 = cam_from_rect0(P2)
    cam3_from_rect0 = cam_from_rect0(P3)
    right_in_left = cam2_from_rect0 @ _inv(cam3_from_rect0)
    return {
        "cam2_from_imu": cam2_from_rect0 @ rect0_from_imu,
        "right_in_left": right_in_left,
        "K_left": P2[:, :3],
        "K_right": P3[:, :3],
    }
