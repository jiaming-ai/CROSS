"""RGB-only sequence reader. No ground truth or sensor-depth access here."""

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, thread_time

import cv2
import numpy as np


@dataclass
class RGBFrame:
    rgb: np.ndarray
    timestamp: float
    index: int
    input_timing: dict = field(default_factory=dict)


def openloris_color_calibration(path):
    """Read only RGB calibration; package order is fx, cx, fy, cy.

    Order cross-checked against the dataset authors' CameraInfo K in
    openloris-scene-tools/dataprocess/segway_transforms.py.
    """
    handle = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        color = handle.getNode("d400_color_optical_frame")
        if color.empty() or color.getNode("model").string() != "pinhole":
            raise ValueError("Expected OpenLORIS D400 color pinhole calibration")
        if color.getNode("distortion_model").string() != "radial-tangential":
            raise ValueError("Unsupported OpenLORIS color distortion model")
        fx, cx, fy, cy = color.getNode("intrinsics").mat().reshape(4)
        distortion = color.getNode("distortion_coefficients").mat().reshape(-1)
        size = (int(color.getNode("width").real()), int(color.getNode("height").real()))
    finally:
        handle.release()
    if min(size) <= 0:
        raise ValueError("Invalid RGB calibration dimensions")
    return [fx, fy, cx, cy, *distortion], size


class RGBSequence:
    def __init__(self, path, stride=1, start=0, limit=None, calibration=None, undistort=True, sample_fps=None,
                 resize=None):
        self.path = Path(path)
        if stride < 1 or start < 0 or (limit is not None and limit < 1):
            raise ValueError("Invalid frame selection")
        if sample_fps is not None and (not np.isfinite(sample_fps) or sample_fps <= 0):
            raise ValueError("sample_fps must be finite and positive")
        manifest = self.path / "rgb.txt"
        if not manifest.exists() and (self.path / "color.txt").exists():
            manifest = self.path / "color.txt"
        self.input_size = None
        if manifest.name == "color.txt":
            color_calibration, self.input_size = openloris_color_calibration(self.path / "sensors.yaml")
            if calibration is None:
                calibration = color_calibration
        rows = []
        for line in manifest.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            stamp, filename = line.split()[:2]
            rows.append((float(stamp), self.path / filename))
        if any(not np.isfinite(row[0]) for row in rows) or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
            raise ValueError("rgb.txt timestamps are not strictly increasing")
        self.total_frames = len(rows)
        self.rows = list(enumerate(rows))[start::stride]
        if sample_fps is not None and self.rows:
            # Retain the first available image in each elapsed-time bin.
            # Empty bins stay empty; no image or timestamp is synthesized.
            # This selection reads only the RGB timestamp manifest.
            origin, previous_bin, selected = self.rows[0][1][0], -1, []
            for row in self.rows:
                time_bin = int(np.floor((row[1][0] - origin) * sample_fps))
                if time_bin > previous_bin:
                    selected.append(row)
                    previous_bin = time_bin
            self.rows = selected
        if limit is not None:
            self.rows = self.rows[:limit]
        name = self.path.name
        if calibration is not None:
            if len(calibration) not in (4, 8, 9, 12, 16, 18) or not np.isfinite(calibration).all():
                raise ValueError("Expected finite fx fy cx cy and optional OpenCV distortion coefficients")
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
        if fx <= 0 or fy <= 0:
            raise ValueError("Focal lengths must be positive")
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.input_K = self.K.copy()
        self.resize = None if resize is None else tuple(resize)
        if self.resize is not None:
            if len(self.resize) != 2 or any(int(x) != x or x < 1 for x in self.resize):
                raise ValueError("Resize dimensions must be positive integers")
            self.resize = tuple(int(x) for x in self.resize)
            if self.input_size is None:
                # Only the first selected RGB is inspected before capture,
                # exactly as in first-frame model initialization/warmup.
                first = cv2.imread(str(self.rows[0][1][1])) if self.rows else None
                if first is None:
                    raise ValueError("Cannot determine source image dimensions")
                self.input_size = (first.shape[1], first.shape[0])
            sx, sy = np.array(self.resize) / self.input_size
            # OpenCV resize maps pixel centres: u' = s * (u + .5) - .5.
            self.K[0] *= sx
            self.K[1] *= sy
            self.K[0, 2] += (sx - 1) * 0.5
            self.K[1, 2] += (sy - 1) * 0.5
        self.undistort = undistort
        self.maps = None

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        for index, (timestamp, path) in self.rows:
            wall, cpu = perf_counter(), thread_time()
            bgr = cv2.imread(str(path))
            timing = dict(read_decode_wall_seconds=perf_counter()-wall, read_decode_cpu_seconds=thread_time()-cpu)
            if bgr is None:
                raise FileNotFoundError(path)
            if self.input_size is not None and (bgr.shape[1], bgr.shape[0]) != self.input_size:
                raise ValueError("Image dimensions do not match RGB calibration")
            wall, cpu = perf_counter(), thread_time()
            if self.undistort and np.any(self.distortion):
                if self.maps is None:
                    self.maps = cv2.initUndistortRectifyMap(self.input_K, self.distortion, None, self.input_K,
                                                         (bgr.shape[1], bgr.shape[0]), cv2.CV_32FC1)
                bgr = cv2.remap(bgr, *self.maps, interpolation=cv2.INTER_LINEAR)
            timing.update(undistort_wall_seconds=perf_counter()-wall, undistort_cpu_seconds=thread_time()-cpu)
            wall, cpu = perf_counter(), thread_time()
            if self.resize is not None and self.resize != (bgr.shape[1], bgr.shape[0]):
                bgr = cv2.resize(bgr, self.resize, interpolation=cv2.INTER_LINEAR)
            timing.update(resize_wall_seconds=perf_counter()-wall, resize_cpu_seconds=thread_time()-cpu)
            wall, cpu = perf_counter(), thread_time()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            timing.update(color_wall_seconds=perf_counter()-wall, color_cpu_seconds=thread_time()-cpu,
                          opencv_threads=cv2.getNumThreads())
            yield RGBFrame(rgb, timestamp, index, timing)
