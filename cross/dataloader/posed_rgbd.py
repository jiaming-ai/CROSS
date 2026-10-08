"""Posed RGB-D sequence stored as a folder (e.g. the SimChange renders, or any dataset converted to this layout).

Layout::

    <root>/calib.json        {"K": 3x3, "width": W, "height": H, "fps": 10.0 (optional)}
    <root>/left/*.png        RGB images (sorted by name); `rgb/` is accepted as well
    <root>/depth/*.npy       metric depth in metres (float), or depth/*.png as uint16 millimetres
    <root>/poses_left.txt    one camera-to-world pose per image (16 values per row, OpenCV camera convention)
    <root>/odom_left.txt     optional: camera poses from the robot's own odometry (same format)
    <root>/imu.txt, imu.json optional: the IMU rigidly attached to the camera (cross/dataloader/imu.py); replayed frames
                             then carry the IMU samples since the previous frame

The ground-truth poses serve as ground truth.  The odometry is the consecutive difference of odom_left.txt when it
exists (real odometry, e.g. wheel encoders; no simulated noise is added), otherwise of the ground truth with simulated
white SNR noise and optional systematic drift (see `Dataloader`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from cross.dataloader.dataloader import Dataloader
from cross.dataloader.imu import ImuStream


def _invert(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


class PosedRGBDLoader(Dataloader):
    def __init__(self, root: str, use_depth: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.root = Path(root)
        calib = json.loads((self.root / "calib.json").read_text())
        img_dir = self.root / "left" if (self.root / "left").is_dir() else self.root / "rgb"
        self.rgb_paths = sorted(img_dir.glob("*.png")) or sorted(img_dir.glob("*.jpg"))
        depth_dir = self.root / "depth"
        self.depth_paths = (sorted(depth_dir.glob("*.npy")) or sorted(depth_dir.glob("*.png"))) if depth_dir.is_dir() else []
        self.use_depth = use_depth and len(self.depth_paths) > 0
        if self.use_depth:
            assert len(self.depth_paths) == len(self.rgb_paths), f"{root}: {len(self.depth_paths)} depth maps, {len(self.rgb_paths)} images"
        self.c2w = np.loadtxt(self.root / "poses_left.txt").reshape(-1, 4, 4)
        assert len(self.c2w) == len(self.rgb_paths), f"{root}: {len(self.c2w)} poses, {len(self.rgb_paths)} images"
        self.odom_c2w = None
        if self.odometry_path(self.root) is not None:
            self.odom_c2w = np.loadtxt(self.odometry_path(self.root)).reshape(-1, 4, 4)
            assert len(self.odom_c2w) == len(self.rgb_paths)
            self.snr = None                   # real odometry: no simulated noise
            self.odom_scale_bias = self.odom_yaw_drift = 0.0
        self.rgb_K = np.asarray(calib["K"], dtype=np.float64)
        self.rgb_width, self.rgb_height = int(calib["width"]), int(calib["height"])
        self.fps = float(calib.get("fps", 10.0))
        self.odom_vertical_world = self._world_vertical()
        self.imu = ImuStream.load(self.root, len(self.rgb_paths), self.fps)

    def _world_vertical(self) -> np.ndarray:
        """World vertical of the ground-truth frame for the heading-drift model: the camera axis whose world direction
        stays most constant over the sequence (y for a forward-looking ground robot)."""
        R = self.c2w[:, :3, :3]
        best, v = -1.0, np.array([0.0, 0.0, 1.0])
        for ax in (1, 2):
            m = R[:, :, ax].mean(0)
            if np.linalg.norm(m) > best:
                best, v = float(np.linalg.norm(m)), m / max(np.linalg.norm(m), 1e-12)
        return v

    def __len__(self):
        return len(self.rgb_paths)

    def get_sequence_frequency(self):
        return self.fps

    def get_idx_from_timestamp(self, timestamp):
        return int(round(float(timestamp) * self.fps))

    def _depth(self, idx: int) -> Optional[np.ndarray]:
        if not self.use_depth:
            return None
        p = self.depth_paths[idx]
        if p.suffix == ".npy":
            d = np.load(p).astype(np.float32)
        else:
            d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        d[~np.isfinite(d)] = 0.0
        return d

    def __getitem__(self, idx: int):
        rgb = cv2.cvtColor(cv2.imread(str(self.rgb_paths[idx])), cv2.COLOR_BGR2RGB)
        src = self.c2w if self.odom_c2w is None else self.odom_c2w
        delta = np.eye(4) if idx == 0 else _invert(src[idx - 1]) @ src[idx]
        return {
            "rgb": rgb,
            "depth": self._depth(idx),
            "conf": None,
            "delta_pose": delta,
            "world_pose": self.c2w[idx],
            "timestamp": idx / self.fps,
            "frame_idx": idx,
        }

    def replay_data(self, start_idx: int = 0, end_idx: Optional[int] = None, stride: int = 1):
        """Frames in order without sleeping (offline evaluation); with stride > 1 the odometry of the skipped frames is
        composed step by step (noise per step)."""
        end_idx = len(self) if end_idx is None else min(end_idx, len(self))
        prev = None
        for i in range(start_idx, end_idx, stride):
            item = self.get_item(i, first_item=(prev is None))
            if prev is not None and stride > 1 and self.odom_c2w is not None:
                item["delta_pose"] = _invert(self.odom_c2w[prev]) @ self.odom_c2w[i]
            elif prev is not None and stride > 1:
                T = np.eye(4)
                for j in range(prev + 1, i + 1):
                    d = _invert(self.c2w[j - 1]) @ self.c2w[j]
                    d = self._bias_delta(d, self.c2w[j])
                    if self.snr is not None:
                        d = self._noise_delta(d, self.c2w[j])
                    T = T @ d
                item["delta_pose"] = T
            if self.imu is not None:
                item.update(self.imu.window(prev, i), imu_calib=self.imu.calib)
            if getattr(self, "geo", None) is not None:     # GNSS fix / compass sample (cross/dataloader/geo.py)
                item.update(self.geo.window(prev, i, item["timestamp"]))
            prev = i
            yield item
