"""One phone's recording of an iPhone capture (ARKit + camera + LiDAR), as written by the capture app.

Folder layout (one phone):
    frames.csv      one row per saved frame: frame_id, unix_time_s, image_path, depth_path, confidence_path,
                    depth_width, depth_height, fx, fy, cx, cy (pixels of image_width x image_height), tracking_state,
                    pose_m00 .. pose_m33 (ARKit camera-to-world, row-major; camera axes x right, y up, z back)
    rgb/            images (image_path)
    depth/          LiDAR depth, float32 metres, depth_height x depth_width (depth_path), optional
    confidence/     LiDAR confidence, uint8 0..2 (confidence_path), optional

Poses are returned in OpenCV camera axes (x right, y down, z forward) in ARKit's (gravity-aligned, y up) world.
A recording can also be read straight from a capture zip: "capture.zip::path/inside/zip/recording_..._B".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import io
import zipfile

import numpy as np
import pandas as pd

# ARKit camera axes (x right, y up, z back) -> OpenCV camera axes (x right, y down, z forward)
ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


class _Files:
    """Files of a recording folder, or of a folder inside a zip ("archive.zip::inner/folder")."""

    def __init__(self, spec):
        spec = str(spec)
        if "::" in spec:
            z, inner = spec.split("::", 1)
            self.zip, self.prefix, self.root = zipfile.ZipFile(z), inner.rstrip("/") + "/", None
        else:
            self.zip, self.prefix, self.root = None, "", Path(spec)

    def read(self, rel: str) -> bytes:
        if self.zip is not None:
            return self.zip.read(self.prefix + rel)
        return (self.root / rel).read_bytes()

    def exists(self, rel: str) -> bool:
        if self.zip is not None:
            try:
                self.zip.getinfo(self.prefix + rel)
                return True
            except KeyError:
                return False
        return (self.root / rel).is_file()

    def __str__(self):
        return f"{self.zip.filename}::{self.prefix}" if self.zip is not None else str(self.root)


@dataclass
class Recording:
    root: object                   # _Files
    frames: pd.DataFrame           # rows kept (tracking, increasing time), with their original index
    t: np.ndarray                  # unix time (s), strictly increasing
    T: np.ndarray                  # (n, 4, 4) camera-to-world, OpenCV camera axes

    @classmethod
    def load(cls, root, require_tracking: bool = True) -> "Recording":
        root = _Files(root)
        d = pd.read_csv(io.BytesIO(root.read("frames.csv")))
        if require_tracking and "tracking_state" in d:
            d = d[d.tracking_state == "normal"]
        d = d.sort_values("unix_time_s", kind="stable")
        t = d.unix_time_s.to_numpy(np.float64)
        keep = np.r_[True, np.diff(t) > 1e-4]            # repeated time stamps (clock adjustments)
        d = d[keep]
        cols = [[f"pose_m{r}{c}" for c in range(4)] for r in range(4)]
        T = np.stack([d[c].to_numpy(np.float64) for c in cols], 1) @ ARKIT_TO_CV
        if len(d) < 10:
            raise ValueError(f"{root}: fewer than 10 tracked frames")
        return cls(root, d.reset_index(drop=False), d.unix_time_s.to_numpy(np.float64), T)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def size(self):
        r = self.frames.iloc[0]
        return int(r.image_width), int(r.image_height)

    def K(self, i: Optional[int] = None) -> np.ndarray:
        """Intrinsics of frame i (median over the recording when i is None)."""
        f = self.frames if i is None else self.frames.iloc[[i]]
        fx, fy, cx, cy = (float(np.median(f[c])) for c in ("fx", "fy", "cx", "cy"))
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    def image(self, i: int, gray: bool = True, scale: float = 1.0):
        import cv2
        rel = self.frames.image_path.iloc[i]
        if not self.root.exists(rel):
            raise FileNotFoundError(f"{self.root}/{rel}")
        buf = np.frombuffer(self.root.read(rel), np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"{self.root}/{rel}")
        if scale != 1.0:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return im

    def depth(self, i: int, min_confidence: int = 2):
        """(depth (h, w) float32 m with low-confidence pixels set to nan, scale from depth pixels to image pixels) or
        None without LiDAR."""
        r = self.frames.iloc[i]
        if "depth_path" not in self.frames or not isinstance(r.depth_path, str):
            return None
        if not self.root.exists(r.depth_path):
            return None
        h, w = int(r.depth_height), int(r.depth_width)
        d = np.frombuffer(self.root.read(r.depth_path), np.float32).reshape(h, w).copy()
        if "confidence_path" in self.frames and isinstance(r.confidence_path, str) and self.root.exists(r.confidence_path):
            c = np.frombuffer(self.root.read(r.confidence_path), np.uint8).reshape(h, w)
            d[c < min_confidence] = np.nan
        return d, (r.image_width / w, r.image_height / h)

    def pose_at(self, t: np.ndarray) -> np.ndarray:
        """Camera-to-world at times t (s, this phone's clock): rotations slerped, positions linear; clipped to the ends."""
        from scipy.spatial.transform import Rotation, Slerp
        t = np.clip(np.atleast_1d(np.asarray(t, np.float64)), self.t[0], self.t[-1])
        out = np.tile(np.eye(4), (len(t), 1, 1))
        out[:, :3, :3] = Slerp(self.t, Rotation.from_matrix(self.T[:, :3, :3]))(t).as_matrix()
        for k in range(3):
            out[:, k, 3] = np.interp(t, self.t, self.T[:, k, 3])
        return out
