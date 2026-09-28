import json
import time
from pathlib import Path
from typing import Optional
from zipfile import ZipFile
import pickle
import os

import lzfse
import numpy as np
import tqdm
from PIL import Image

from cross.utils.rotation import quaternion_to_rotation_matrix_numpy
from cross.dataloader.dataloader import Dataloader
# Map r3d (x right, y up, z backward) -> OpenCV (x right, y down, z forward)
APPLY_R3D_TO_OPENCV = True
R3D_TO_OPENCV = np.eye(4, dtype=np.float64)
R3D_TO_OPENCV[:3, :3] = np.diag([1.0, -1.0, -1.0])  # 180° about X
R3D_TO_OPENCV_INV = np.linalg.inv(R3D_TO_OPENCV)

class R3DDataset(Dataloader):
    """
    R3D dataloader aligned to OpenLorisLoader's interface:
      - __len__, __getitem__ -> dict with keys: rgb, depth, conf, delta_pose, world_pose, timestamp
      - get_sequence_frequency()
      - get_idx_from_timestamp(timestamp)
      - load_test_json()
      - replay_data(fps=None, start_idx=0, end_idx=None)

    Notes
    -----
    * Images are resized in the base `Dataloader`: optional target_width/height with width-first
      scaling + center crop/pad to preserve horizontal FoV; otherwise short_side scaling.
    * We store intrinsics after scaling as rgb_K/depth_K; depth is already aligned to RGB.
    * Coordinate transform: we convert both delta and world poses into the same OpenCV-ish frame
      used by your OpenLoris pipeline (right, down, forward).
    """

    def __init__(
        self,
        path: str,
        chunk_size: int = 100,
        benchmark: bool = False,
        short_side: int = 480,
        target_width: int | None = None,
        target_height: int | None = None,
        **kwargs,
    ):
        super().__init__(
            short_side=short_side,
            target_width=target_width,
            target_height=target_height,
            **kwargs,
        )
        self._raw_path = path
        self._processed_dir = Path(path).parent / (Path(path).stem + "_processed")
        self._chunk_size = chunk_size
        self._current_chunk = None
        self._current_chunk_id = None
        self._benchmark = benchmark
        self.force_reload = kwargs.pop('force_reload', False)

        # placeholders filled in by metadata
        self.rgb_width = None
        self.rgb_height = None
        self.depth_width = None
        self.depth_height = None
        self.fps = None
        self.camera_matrix = None         # original K (3x3, before scaling)
        self.rgb_K = None                 # scaled intrinsics
        self.depth_K = None               # same as rgb_K (after scaling)
        self.image_size = None
        self.poses = None                 # [qx, qy, qz, qw, px, py, pz]
        self.init_pose = None
        self.total_images = None
        self.T_rgb_depth = np.eye(4)      # depth pre-aligned to rgb after processing
        self.resize_meta = {}

        # synthetic timestamps: seconds from start; filled after we know total_images and fps
        self._timestamps = None

        self._container = None
        if self._processed_dir.exists() and not self.force_reload:
            self._load_metadata()
            if self._should_reprocess():
                self._prepare_and_process()
                self._load_metadata()
        else:
            self._prepare_and_process()
            self._load_metadata()

        # convenience mirror for OpenLoris-style names
        self.depth_width = self.rgb_width
        self.depth_height = self.rgb_height

    # ---------- internal helpers ----------

    def _open(self, relpath: str, mode: str = "r"):
        """Unified open for ZipFile / directory container."""
        if isinstance(self._container, ZipFile):
            return self._container.open(relpath, mode)
        else:
            return (Path(self._container) / relpath).open(mode)

    def _read_metadata(self):
        with self._open("metadata", "r") as f:
            md = json.load(f)

        self.rgb_width = md["w"]
        self.rgb_height = md["h"]
        self.depth_width = md["dw"]
        self.depth_height = md["dh"]
        self.fps = float(md["fps"])
        self.camera_matrix = np.array(md["K"]).reshape(3, 3).T  # match OpenLoris K layout
        self.image_size = (self.rgb_width, self.rgb_height)
        self.poses = np.array(md["poses"], dtype=np.float64)    # [qx, qy, qz, qw, px, py, pz]
        self.init_pose = np.array(md["initPose"], dtype=np.float64)
        self.total_images = len(self.poses)
        self._timestamps = np.arange(self.total_images, dtype=np.float64) / self.fps

    def _should_reprocess(self):
        """Reprocess if requested target dims differ from what is on disk."""
        meta_target_w = self.resize_meta.get("target_width")
        meta_target_h = self.resize_meta.get("target_height")
        if (self.target_width != meta_target_w or self.target_height != meta_target_h) and any(
            v is not None for v in [self.target_width, self.target_height, meta_target_w, meta_target_h]
        ):
            return True

        if self.resize_meta.get("crop_center_h") != self.crop_center_h:
            return True
        if self.resize_meta.get("crop_center_w") != self.crop_center_w:
            return True
        if self.target_width is not None and self.target_width != self.rgb_width:
            return True
        if self.target_height is not None and self.target_height != self.rgb_height:
            return True

        # If user sticks with short_side-based resizing, keep existing unless params changed.
        if self.target_width is None and self.target_height is None:
            meta_short_side = self.resize_meta.get("short_side")
            if meta_short_side is not None and meta_short_side != self.short_side:
                return True
        return False

    def _prepare_and_process(self):
        # open raw container
        if self._raw_path.endswith((".zip", ".r3d")):
            self._container = ZipFile(self._raw_path)
        else:
            self._container = Path(self._raw_path)  # if you ever dump folders

        self._read_metadata()
        self._process_and_save_chunks()

        if isinstance(self._container, ZipFile):
            self._container.close()
        self._container = None

    def _process_and_save_chunks(self):
        """Process once -> write chunked pkl + metadata.pkl (resized frames, scaled K, poses)."""
        self._processed_dir.mkdir(exist_ok=True)

        # Determine final size and scale K accordingly
        sample_rgb = self.load_image("rgbd/0.jpg")
        oh, ow = sample_rgb.shape[:2]
        resize_cfg = self._compute_resize_config(ow, oh)

        scaled_K = self._resize_intrinsics(self.camera_matrix, resize_cfg)

        # Save metadata
        meta = {
            "rgb_width": resize_cfg["final_w"],
            "rgb_height": resize_cfg["final_h"],
            "depth_width": resize_cfg["final_w"],
            "depth_height": resize_cfg["final_h"],
            "fps": self.fps,
            "camera_matrix": scaled_K,
            "image_size": (resize_cfg["final_w"], resize_cfg["final_h"]),
            "poses": self.poses,
            "init_pose": self.init_pose,
            "total_images": self.total_images,
            "T_rgb_depth": np.eye(4),
            "resize_meta": {
                "target_width": self.target_width,
                "target_height": self.target_height,
                "short_side": self.short_side,
                "strategy": "width_first" if self.target_width is not None else "short_side",
                "crop_center_h": self.crop_center_h,
                "crop_center_w": self.crop_center_w,
            },
        }
        with open(self._processed_dir / "metadata.pkl", "wb") as f:
            pickle.dump(meta, f)

        prev_world = np.eye(4)

        for chunk_start in tqdm.trange(0, self.total_images, self._chunk_size, desc="Processing r3d"):
            chunk_end = min(chunk_start + self._chunk_size, self.total_images)

            rgb_images = []
            depth_images = []
            conf_images = []
            delta_poses = []
            world_poses = []  # transformed to OpenCV-ish

            for i in range(chunk_start, chunk_end):
                rgb = self.load_image(f"rgbd/{i}.jpg")
                d = self.load_depth(f"rgbd/{i}.depth")
                c = self.load_conf(f"rgbd/{i}.conf")

                rgb = self._resize_image(rgb, resize_cfg, mode="rgb")
                depth = self._resize_image(d, resize_cfg, mode="depth")
                conf  = self._resize_image(c, resize_cfg, mode="conf")

                rgb_images.append(rgb)
                depth_images.append(depth)
                conf_images.append(conf)

                # world pose (raw)
                qx, qy, qz, qw, px, py, pz = self.poses[i]
                R = quaternion_to_rotation_matrix_numpy([qx, qy, qz, qw])
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = np.array([px, py, pz], dtype=np.float64)

                # delta in raw frame
                delta = np.linalg.inv(prev_world) @ T
                prev_world = T

                # --- map to OpenCV convention if requested ---
                if APPLY_R3D_TO_OPENCV:
                    delta_mapped = R3D_TO_OPENCV @ delta @ R3D_TO_OPENCV_INV
                    world_mapped = R3D_TO_OPENCV @ T     @ R3D_TO_OPENCV_INV
                else:
                    delta_mapped = delta
                    world_mapped = T

                delta_poses.append(delta_mapped)
                world_poses.append(world_mapped)

            chunk = {
                "rgb_images": rgb_images,
                "depth_images": depth_images,
                "conf_images": conf_images,
                "all_delta_pose": delta_poses,
                "world_poses": world_poses,
            }
            with open(self._processed_dir / f"chunk_{chunk_start:06d}.pkl", "wb") as f:
                pickle.dump(chunk, f)

            if self._benchmark:
                rgb_dir = self._processed_dir / "rgb"
                rgb_dir.mkdir(exist_ok=True)
                for j, rgb_img in enumerate(rgb_images, start=chunk_start):
                    Image.fromarray(rgb_img).save(rgb_dir / f"{j}.jpg")

        print(f"[r3d] Processed data saved to {self._processed_dir}")

    def _load_metadata(self):
        with open(self._processed_dir / "metadata.pkl", "rb") as f:
            md = pickle.load(f)

        self.rgb_width = int(md["rgb_width"])
        self.rgb_height = int(md["rgb_height"])
        self.depth_width = int(md["depth_width"])
        self.depth_height = int(md["depth_height"])
        self.fps = float(md["fps"])
        self.camera_matrix = np.array(md["camera_matrix"], dtype=np.float64)
        self.rgb_K = self.camera_matrix.copy()
        self.depth_K = self.camera_matrix.copy()
        self.T_rgb_depth = np.eye(4)  # stored in meta; keep identity

        self.image_size = tuple(md["image_size"])
        self.poses = np.array(md["poses"], dtype=np.float64)
        self.init_pose = np.array(md["init_pose"], dtype=np.float64)
        self.total_images = int(md["total_images"])
        self.resize_meta = md.get("resize_meta", {})

        # mirror names like OpenLoris
        self._timestamps = np.arange(self.total_images, dtype=np.float64) / self.fps

    # ---------- raw IO ----------

    def load_image(self, filepath):
        with self._open(filepath, "r") as f:
            return np.asarray(Image.open(f))

    def load_depth(self, filepath):
        with self._open(filepath, "r") as f:
            raw = f.read()
        arr = lzfse.decompress(raw)
        depth = np.frombuffer(arr, dtype=np.float32)
        if depth.shape[0] == 960 * 720:
            depth = depth.reshape((960, 720))
        else:
            depth = depth.reshape((256, 192))
        return depth

    def load_conf(self, filepath):
        with self._open(filepath, "r") as f:
            raw = f.read()
        arr = lzfse.decompress(raw)
        conf = np.frombuffer(arr, dtype=np.uint8)
        if conf.shape[0] == 960 * 720:
            conf = conf.reshape((960, 720))
        else:
            conf = conf.reshape((256, 192))
        return conf

    # ---------- OpenLoris-like public API ----------

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx: int):
        chunk_id = (idx // self._chunk_size) * self._chunk_size
        self._load_chunk(chunk_id)
        rel = idx % self._chunk_size

        # Confidence convention: keep boolean map (True = good depth)
        conf_map = (self._current_chunk["conf_images"][rel] == 2)

        return {
            "rgb": self._current_chunk["rgb_images"][rel],
            "depth": self._current_chunk["depth_images"][rel],
            "conf": conf_map,
            "delta_pose": self._current_chunk["all_delta_pose"][rel],
            "world_pose": self._current_chunk["world_poses"][rel],
            # "timestamp": float(self._timestamps[idx]),
        }

    def _load_chunk(self, chunk_id: int):
        if self._current_chunk_id == chunk_id:
            return
        with open(self._processed_dir / f"chunk_{chunk_id:06d}.pkl", "rb") as f:
            self._current_chunk = pickle.load(f)
        self._current_chunk_id = chunk_id

    def get_sequence_frequency(self) -> float:
        return float(self.fps)

    def get_idx_from_timestamp(self, timestamp: float) -> Optional[int]:
        # synthetic timestamps are monotonic seconds since start
        idx = int(np.searchsorted(self._timestamps, timestamp, side="right"))
        if idx >= self.total_images:
            return None
        return idx

    def load_test_json(self):
        # no test.json in r3d; return an OpenLoris-compatible default
        self._test_json = {
            "map_start_timestamp": None,
            "map_end_timestamp": None
        }
        return self._test_json

    # def replay_data(
    #     self,
    #     fps: float = None,
    #     start_idx: int = 0,
    #     end_idx: Optional[int] = None,
    # ):
    #     fps = float(fps) if fps is not None else self.get_sequence_frequency()
    #     interval = 1.0 / fps

    #     start_idx = max(0, int(start_idx))
    #     if end_idx is None:
    #         end_idx = self.total_images
    #     else:
    #         end_idx = min(int(end_idx), self.total_images)

    #     for i in range(start_idx, end_idx):
    #         yield self[i]
    #         time.sleep(interval)
