"""IMU stream of a prepared sequence folder (benchmark/datasets/prepare_imu.py).

Files next to the images::

    <root>/imu.txt     one sample per row: t wx wy wz ax ay az  (IMU frame; rad/s and m/s^2, specific force including
                       gravity; t on the clock of times.txt)
    <root>/imu.json    {"T_cam_imu": 4x4 pose of the IMU in the camera frame (x_cam = T x_imu),
                        "gyro_noise_density" (rad/s/sqrt(Hz)), "accel_noise_density" (m/s^2/sqrt(Hz)),
                        "gyro_random_walk" (rad/s^2/sqrt(Hz)), "accel_random_walk" (m/s^3/sqrt(Hz)),
                        "rate_hz", "source", "frame_times": file of the image timestamps on the IMU clock (default
                        times.txt), ...}
    <root>/times.txt   image timestamps (one per frame); without it, frame i is at i / fps

A frame of a loader with an IMU carries `imu` (the samples that cover the interval since the previous frame, including
one sample at or before its start and one at or after its end, so the integrator can interpolate at both ends),
`imu_t0` and `imu_t1` (the interval on the IMU clock).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ImuCalibration:
    T_cam_imu: np.ndarray = field(default_factory=lambda: np.eye(4))
    gyro_noise_density: float = 1.1e-3
    accel_noise_density: float = 1.2e-2
    gyro_random_walk: float = 6e-5
    accel_random_walk: float = 5e-4
    rate_hz: float = 200.0
    source: str = ""
    frame_times: str = "times.txt"     # the images' timestamps (KITTI: the camera's, regular; times.txt holds OXTS's)

    @classmethod
    def from_dict(cls, d: dict) -> "ImuCalibration":
        keys = {f for f in cls.__dataclass_fields__}
        out = cls(**{k: v for k, v in d.items() if k in keys})
        out.T_cam_imu = np.asarray(out.T_cam_imu, dtype=np.float64).copy()
        # the nearest rotation: calibration files print the rotation with a few digits (KITTI: singular values 1 -
        # 9e-8), and estimators that chain it through every frame compound the error
        U, _, Vt = np.linalg.svd(out.T_cam_imu[:3, :3])
        out.T_cam_imu[:3, :3] = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        return out


class ImuStream:
    """All IMU samples of a sequence and the image timestamps on the IMU clock (read on first use)."""

    def __init__(self, root: Path, calib: ImuCalibration, n_frames: int, fps: float):
        self.root, self.calib, self.n_frames, self.fps = Path(root), calib, n_frames, fps
        self._data = None

    @classmethod
    def load(cls, root, n_frames: int, fps: float) -> Optional["ImuStream"]:
        root = Path(root)
        if not (root / "imu.txt").is_file() or not (root / "imu.json").is_file():
            return None
        return cls(root, ImuCalibration.from_dict(json.loads((root / "imu.json").read_text())), n_frames, fps)

    def _read(self):
        data = np.loadtxt(self.root / "imu.txt", comments="#", dtype=np.float64).reshape(-1, 7)
        data = data[np.argsort(data[:, 0], kind="stable")]
        data = data[np.concatenate([[True], np.diff(data[:, 0]) > 0])]
        if (self.root / self.calib.frame_times).is_file():
            times = np.loadtxt(self.root / self.calib.frame_times, dtype=np.float64).reshape(-1)[:self.n_frames]
        else:
            times = np.arange(self.n_frames) / self.fps
        if len(times) != self.n_frames:
            raise ValueError(f"{self.root}: {len(times)} timestamps for {self.n_frames} frames")
        self._data = (data[:, 0], data[:, 1:4], data[:, 4:7], times)

    def _field(self, k):
        if self._data is None:
            self._read()
        return self._data[k]

    t = property(lambda self: self._field(0))
    gyro = property(lambda self: self._field(1))
    accel = property(lambda self: self._field(2))
    frame_times = property(lambda self: self._field(3))

    def window(self, prev_idx: Optional[int], idx: int) -> dict:
        """The samples covering (time of frame prev_idx, time of frame idx]; none for the first frame of a run."""
        t1 = float(self.frame_times[idx])
        if prev_idx is None or prev_idx < 0:
            return {"imu": np.zeros((0, 7)), "imu_t0": t1, "imu_t1": t1}
        t0 = float(self.frame_times[prev_idx])
        lo = max(int(np.searchsorted(self.t, t0, side="right")) - 1, 0)
        hi = min(int(np.searchsorted(self.t, t1, side="left")) + 1, len(self.t))
        sel = slice(lo, hi)
        samples = np.concatenate([self.t[sel, None], self.gyro[sel], self.accel[sel]], axis=1)
        return {"imu": samples, "imu_t0": t0, "imu_t1": t1}


def write_imu(root, t, gyro, accel, calib: dict, header: str = "", name: str = "imu"):
    """Write <name>.txt / <name>.json (default imu.txt / imu.json; calib: the ImuCalibration fields, T_cam_imu as a 4x4
    nested list, plus any notes)."""
    root = Path(root)
    data = np.concatenate([np.asarray(t, dtype=np.float64)[:, None], np.asarray(gyro), np.asarray(accel)], axis=1)
    head = "t wx wy wz ax ay az (IMU frame; rad/s, m/s^2 specific force; clock of times.txt)"
    np.savetxt(root / f"{name}.txt", data, fmt=["%.6f"] + ["%.8g"] * 6, header=head + (f"\n{header}" if header else ""))
    (root / f"{name}.json").write_text(json.dumps(calib, indent=1))
