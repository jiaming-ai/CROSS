"""Sensor sources of the edge: a replay of a prepared stereo folder (the CROSS benchmark's layout), standing in for the
robot's cameras and IMU.

  <folder>/calib.json   {K, width, height, fps, baseline, T_right_in_left, right_dirs (SimChange: baseline -> folder)}
  <folder>/left, right  rectified stereo images (PNG), one per frame
  <folder>/imu_vio.txt + imu_vio.json   the IMU stream the stereo VIO was run on (benchmark/datasets/prepare_vio.py;
                        else imu.txt + imu.json): t wx wy wz ax ay az, and T_cam_imu, noise densities, frame_times
  <folder>/<frame_times>  the frames' timestamps on the IMU clock
  <folder>/poses_left.txt ground truth (c2w, 16 values per row; evaluation only)

The frames' timestamps for the back end are index / fps, as the CROSS stereo loader (cross.dataloader.stereo_loader)
gives them; the IMU clock of the frames is used only by the VIO."""

import json
from pathlib import Path

import cv2
import numpy as np


class Frame(dict):
    """A frame whose images are read when first used: rgb / rgb_right (RGB, uint8) for the upload, gray / gray_right
    (8-bit, as the VIO was fed) for the odometry."""

    def __init__(self, loaders, **values):
        super().__init__(**values)
        self._loaders = loaders

    def _load(self, key):
        if key in self._loaders and not dict.__contains__(self, key):
            dict.__setitem__(self, key, self._loaders[key]())
        return dict.get(self, key)

    def __getitem__(self, key):
        if key in self._loaders:
            return self._load(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key in self._loaders:
            v = self._load(key)
            return default if v is None else v
        return dict.get(self, key, default)


class StereoFolder:
    def __init__(self, root, baseline=None):
        self.root = Path(root).resolve()
        calib = json.loads((self.root / "calib.json").read_text())
        self.K = np.asarray(calib["K"], dtype=np.float64)
        self.width, self.height = int(calib["width"]), int(calib["height"])
        self.fps = float(calib.get("fps", 10.0))
        right_dir = self.root / "right"
        self.baseline = float(calib.get("baseline", 0.12))
        if baseline is not None:
            key = f"{float(baseline):.2f}"
            dirs = calib.get("right_dirs", {})
            if key not in dirs:
                raise ValueError(f"baseline {key} not rendered for {self.root} (available: {list(dirs)})")
            right_dir = self.root / dirs[key]
            self.baseline = float(baseline)
        self.T_right_in_left = np.asarray(calib["T_right_in_left"], dtype=np.float64)
        self.T_right_in_left[0, 3] = self.baseline
        self.left = sorted((self.root / "left").glob("*.png"))
        self.right = sorted(right_dir.glob("*.png"))
        if not self.left or len(self.left) != len(self.right):
            raise ValueError(f"{self.root}: {len(self.left)} left and {len(self.right)} right images")
        self.timestamps = np.arange(len(self.left)) / self.fps
        name = "imu_vio" if (self.root / "imu_vio.txt").is_file() else "imu"
        self.imu, self.imu_calib, self.frame_times_imu = None, None, None
        if (self.root / f"{name}.txt").is_file() and (self.root / f"{name}.json").is_file():
            self.imu = np.loadtxt(self.root / f"{name}.txt", comments="#", dtype=np.float64).reshape(-1, 7)
            self.imu_calib = json.loads((self.root / f"{name}.json").read_text())
            ft = self.root / self.imu_calib.get("frame_times", "times.txt")
            self.frame_times_imu = (np.loadtxt(ft, dtype=np.float64).reshape(-1)[:len(self.left)] if ft.is_file()
                                    else self.timestamps.copy())
        gt = self.root / "poses_left.txt"
        self.gt = np.loadtxt(gt, dtype=np.float64).reshape(-1, 4, 4) if gt.is_file() else None

    def __len__(self):
        return len(self.left)

    def odometry_file(self, name):
        """A recorded odometry file of the folder (camera c2w per frame) as an array, or None."""
        p = self.root / name
        return np.loadtxt(p, dtype=np.float64).reshape(-1, 4, 4) if p.is_file() else None

    def frame(self, i) -> Frame:
        left, right = str(self.left[i]), str(self.right[i])
        loaders = {"rgb": lambda: cv2.cvtColor(cv2.imread(left), cv2.COLOR_BGR2RGB),
                   "rgb_right": lambda: cv2.cvtColor(cv2.imread(right), cv2.COLOR_BGR2RGB),
                   "gray": lambda: cv2.imread(left, cv2.IMREAD_GRAYSCALE),
                   "gray_right": lambda: cv2.imread(right, cv2.IMREAD_GRAYSCALE)}
        return Frame(loaders, index=i, timestamp=float(self.timestamps[i]),
                     t_imu=None if self.frame_times_imu is None else float(self.frame_times_imu[i]))

    def frames(self, start=0, end=None):
        for i in range(start, len(self) if end is None else min(end, len(self))):
            yield self.frame(i)
