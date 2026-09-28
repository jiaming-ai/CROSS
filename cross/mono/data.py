"""RGB-only sequence reader. No ground truth or sensor-depth access here."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class RGBFrame:
    rgb: np.ndarray
    timestamp: float
    index: int


class RGBSequence:
    def __init__(self, path, stride=1, start=0, limit=None, calibration=None, undistort=True):
        self.path = Path(path)
        if stride < 1 or start < 0 or (limit is not None and limit < 1):
            raise ValueError("Invalid frame selection")
        rows = []
        for line in (self.path / "rgb.txt").read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            stamp, filename = line.split()[:2]
            rows.append((float(stamp), self.path / filename))
        if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
            raise ValueError("rgb.txt timestamps are not strictly increasing")
        self.total_frames = len(rows)
        self.rows = list(enumerate(rows))[start::stride]
        if limit is not None:
            self.rows = self.rows[:limit]
        name = self.path.name
        if calibration is not None:
            fx, fy, cx, cy = calibration[:4]
            self.distortion = np.array(calibration[4:]) if len(calibration) > 4 else np.zeros(5)
        elif "freiburg1" in name:
            fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
            self.distortion = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])
        elif "freiburg2" in name:
            fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7
            self.distortion = np.array([0.2312, -0.7849, -0.0033, -0.0001, 0.9172])
        elif "freiburg3" in name:
            fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
            self.distortion = np.zeros(5)
        elif "bonn" in name:
            fx, fy, cx, cy = 542.822841, 542.576870, 315.593520, 237.756098
            self.distortion = np.array([0.039903, -0.099343, -0.000730, -0.000144, 0.0])
        else:
            raise ValueError("Unknown camera: supply --intrinsics fx fy cx cy [distortion]")
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.undistort = undistort
        self.maps = None

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        for index, (timestamp, path) in self.rows:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(path)
            if self.undistort and np.any(self.distortion):
                if self.maps is None:
                    self.maps = cv2.initUndistortRectifyMap(self.K, self.distortion, None, self.K,
                                                         (bgr.shape[1], bgr.shape[0]), cv2.CV_32FC1)
                bgr = cv2.remap(bgr, *self.maps, interpolation=cv2.INTER_LINEAR)
            yield RGBFrame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), timestamp, index)
