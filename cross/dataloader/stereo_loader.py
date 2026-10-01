"""Stereo sequence loaders for CROSS (KITTI Raw, TartanAir V2, Virtual KITTI 2).

Every loader yields CROSS observation dicts:
    rgb        : left image (H, W, 3) uint8
    rgb_right  : right image (H, W, 3) uint8
    depth      : metric depth of the left image (H, W) float32 or None
                 (source: "sgbm" classical stereo, "none")
    conf       : None
    delta_pose : T_{prev -> curr} of the left camera (OpenCV convention), simulated
                 odometry from ground truth; the base class adds SNR noise if requested
    world_pose : ground-truth left camera-to-world pose (OpenCV convention)
    timestamp  : seconds

and expose `rgb_K`, `rgb_width`, `rgb_height`, `T_right_in_left` (4x4) and `baseline`.
Pose conventions follow stereo_vggt/src/stereo_scale/datasets.py.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from cross.dataloader.dataloader import Dataloader


def _invert(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def _poses_from_xyz_quat(rows: np.ndarray, tartanair_camera_axes: bool) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    poses = np.repeat(np.eye(4)[None], len(rows), axis=0)
    R = Rotation.from_quat(rows[:, 3:7]).as_matrix()
    if tartanair_camera_axes:
        # TartanAir camera body: x forward, y right, z down -> OpenCV x right, y down, z forward
        ned_from_opencv = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        R = R @ ned_from_opencv
    poses[:, :3, :3] = R
    poses[:, :3, 3] = rows[:, :3]
    return poses


def _umeyama_se3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid transform T with dst ≈ T @ src for (N,3) point sets."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - R @ mu_s
    return T


def _constant_rig(left_c2w: np.ndarray, right_c2w: np.ndarray) -> np.ndarray:
    rel = np.stack([_invert(l) @ r for l, r in zip(left_c2w, right_c2w)])
    R = Rotation.from_matrix(rel[:, :3, :3]).mean().as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.median(rel[:, :3, 3], axis=0)
    return T


# ----------------------------------------------------------------------------- #
# SGBM stereo depth (classical baseline depth source)
# ----------------------------------------------------------------------------- #
class SGBMDepth:
    def __init__(self, width: int, profile: str = "fast"):
        if profile == "fast":
            num_disp = max(64, min(160, (width // 4 // 16) * 16))
            mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        else:
            num_disp = max(96, min(256, (width // 3 // 16) * 16))
            mode = cv2.STEREO_SGBM_MODE_HH4
        bs = 5
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disp, blockSize=bs, P1=8 * bs * bs, P2=32 * bs * bs,
            disp12MaxDiff=1, uniquenessRatio=8, speckleWindowSize=80, speckleRange=2, preFilterCap=31, mode=mode,
        )

    def __call__(self, left_rgb: np.ndarray, right_rgb: np.ndarray, fx: float, baseline: float) -> np.ndarray:
        l = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
        r = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
        disp = self.matcher.compute(l, r).astype(np.float32) / 16.0
        depth = np.zeros_like(disp)
        ok = disp > 0.5
        depth[ok] = fx * baseline / disp[ok]
        depth[~np.isfinite(depth)] = 0.0
        return depth


def sgbm_cache_dir(root: Path) -> Path:
    """Local cache of the SGBM depth maps of one sequence: $CROSS_CACHE_DIR or $XDG_CACHE_HOME/cross (~/.cache/cross).
    Not next to the data: a small file written to network storage costs ~1 s, 40x the SGBM itself."""
    base = os.environ.get("CROSS_CACHE_DIR") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "cross"
    key = hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:16]
    return Path(base) / "sgbm_depth" / f"{Path(root).parent.name}_{Path(root).name}_{key}"


# ----------------------------------------------------------------------------- #
class StereoSequenceLoader(Dataloader):
    """Generic stereo sequence loader; `dataset_type` in {auto, kitti, tartanair, vkitti2}."""

    def __init__(
        self,
        root: str,
        dataset_type: str = "auto",
        depth_source: str = "none",
        fps: Optional[float] = None,
        max_depth: float = 60.0,
        depth_cache: bool = True,
        baseline: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root = Path(root).resolve()
        self.requested_baseline = baseline
        self.dataset_type = self._infer_type(self.root) if dataset_type == "auto" else dataset_type
        self.depth_source = depth_source
        self.max_depth = max_depth
        self.depth_cache = depth_cache
        loader = {"kitti": self._load_kitti, "tartanair": self._load_tartanair, "vkitti2": self._load_vkitti2,
                  "simchange": self._load_simchange, "posed_rgbd": self._load_posed_rgbd}[self.dataset_type]
        self.odom_c2w = None
        loader()
        # ground-truth poses stay in the dataset world frame (OpenCV camera convention) so that
        # map and query traversals of the same scene share one frame for evaluation
        self.left_c2w = np.asarray(self.left_c2w, dtype=np.float64)
        self.baseline = 0.0 if self.T_right_in_left is None else float(np.linalg.norm(self.T_right_in_left[:3, 3]))
        self.fps = fps if fps is not None else self._default_fps
        self._timestamps = np.arange(len(self.left_paths)) / self.fps
        self.orig_width, self.orig_height = self.rgb_width, self.rgb_height
        self.orig_K = self.rgb_K.copy()
        self._resize_cfg = None
        if self.target_width is not None or self.target_height is not None:
            self._resize_cfg = self._compute_resize_config(self.orig_width, self.orig_height)
            self.rgb_K = self._resize_intrinsics(self.orig_K, self._resize_cfg)
            self.rgb_width, self.rgb_height = self._resize_cfg["final_w"], self._resize_cfg["final_h"]
        self._sgbm = SGBMDepth(self.orig_width) if depth_source == "sgbm" else None
        self.odom_vertical_world = self._world_vertical()

    def _world_vertical(self) -> np.ndarray:
        """World vertical of the GT frame, for the heading-drift model: the camera axis whose world direction stays
        most constant over the sequence."""
        R = self.left_c2w[:, :3, :3]
        best, v = -1.0, np.array([0.0, 0.0, 1.0])
        for ax in (1, 2):
            m = R[:, :, ax].mean(0)
            if np.linalg.norm(m) > best:
                best, v = float(np.linalg.norm(m)), m / max(np.linalg.norm(m), 1e-12)
        return v

    # ------------------------------------------------------------------ #
    @staticmethod
    def _infer_type(root: Path) -> str:
        if (root / "image_02" / "data").is_dir() and (root / "oxts" / "data").is_dir():
            return "kitti"
        if (root / "image_lcam_front").is_dir() and (root / "pose_lcam_front.txt").is_file():
            return "tartanair"
        if (root / "frames" / "rgb" / "Camera_0").is_dir() and (root / "extrinsic.txt").is_file():
            return "vkitti2"
        if (root / "calib.json").is_file():
            if (root / "left").is_dir():
                return "simchange"
            if (root / "rgb").is_dir():
                return "posed_rgbd"
        raise ValueError(f"Could not infer a stereo dataset layout under {root}")

    # ---- KITTI raw ---------------------------------------------------- #
    def _load_kitti(self):
        from cross.dataloader.kitti_calib import load_kitti_calibration, oxts_to_imu_poses
        date_root = self.root.parent
        self.left_paths = sorted((self.root / "image_02" / "data").glob("*.png"))
        self.right_paths = sorted((self.root / "image_03" / "data").glob("*.png"))
        oxts = sorted((self.root / "oxts" / "data").glob("*.txt"))
        assert len(self.left_paths) == len(self.right_paths) == len(oxts) > 0
        calib = load_kitti_calibration(date_root)
        imu_c2w = oxts_to_imu_poses(oxts)
        self.left_c2w = imu_c2w @ _invert(calib["cam2_from_imu"])
        self.T_right_in_left = calib["right_in_left"]
        self.rgb_K = calib["K_left"].astype(np.float64)
        img = cv2.imread(str(self.left_paths[0]))
        self.rgb_height, self.rgb_width = img.shape[:2]
        self._default_fps = 10.0

    # ---- TartanAir V2 -------------------------------------------------- #
    def _load_tartanair(self):
        self.left_paths = sorted((self.root / "image_lcam_front").glob("*.png"))
        self.right_paths = sorted((self.root / "image_rcam_front").glob("*.png"))
        left_rows = np.loadtxt(self.root / "pose_lcam_front.txt")
        right_rows = np.loadtxt(self.root / "pose_rcam_front.txt")
        assert len(self.left_paths) == len(self.right_paths) == len(left_rows)
        left_c2w = _poses_from_xyz_quat(left_rows, True)
        right_c2w = _poses_from_xyz_quat(right_rows, True)
        self.T_right_in_left = _constant_rig(left_c2w, right_c2w)
        self.left_c2w = left_c2w
        self.rgb_K = np.array([[320.0, 0, 319.5], [0, 320.0, 319.5], [0, 0, 1.0]])
        self.rgb_width = self.rgb_height = 640
        self._default_fps = 10.0

    # ---- Virtual KITTI 2 ----------------------------------------------- #
    def _load_vkitti2(self):
        self.left_paths = sorted((self.root / "frames" / "rgb" / "Camera_0").glob("*.jpg"))
        self.right_paths = sorted((self.root / "frames" / "rgb" / "Camera_1").glob("*.jpg"))
        assert len(self.left_paths) == len(self.right_paths) > 0
        frame_ids = np.asarray([int(p.stem.rsplit("_", 1)[1]) for p in self.left_paths])
        ext = np.loadtxt(self.root / "extrinsic.txt", skiprows=1)
        intr = np.loadtxt(self.root / "intrinsic.txt", skiprows=1)

        def rows(table, cam):
            out = []
            for fid in frame_ids:
                m = table[(table[:, 0].astype(int) == fid) & (table[:, 1].astype(int) == cam)]
                assert len(m) == 1, f"frame {fid} cam {cam}"
                out.append(m[0])
            return np.stack(out)

        left_w2c = rows(ext, 0)[:, 2:].reshape(-1, 4, 4)
        right_w2c = rows(ext, 1)[:, 2:].reshape(-1, 4, 4)
        left_c2w = np.stack([_invert(T) for T in left_w2c])
        right_c2w = np.stack([_invert(T) for T in right_w2c])
        self.T_right_in_left = _constant_rig(left_c2w, right_c2w)
        # the camera-yaw variants (15/30-deg-left/right) are expressed in a rotated world frame;
        # align their camera centres rigidly to the frame-aligned `clone` trajectory of the same scene
        clone_ext = self.root.parent / "clone" / "extrinsic.txt"
        if "deg" in self.root.name and clone_ext.is_file():
            cext = np.loadtxt(clone_ext, skiprows=1)
            c_w2c = rows(cext, 0)[:, 2:].reshape(-1, 4, 4)
            c_c2w = np.stack([_invert(T) for T in c_w2c])
            T_align = _umeyama_se3(left_c2w[:, :3, 3], c_c2w[:, :3, 3])
            left_c2w = np.einsum("ij,njk->nik", T_align, left_c2w)
            self.gt_alignment_rmse = float(np.sqrt(np.mean(np.sum((left_c2w[:, :3, 3] - c_c2w[:, :3, 3]) ** 2, 1))))
        self.left_c2w = left_c2w
        fx, fy, cx, cy = rows(intr, 0)[0, 2:]
        self.rgb_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        img = cv2.imread(str(self.left_paths[0]))
        self.rgb_height, self.rgb_width = img.shape[:2]
        self._default_fps = 10.0

    # ---- SimChange (Blender) ------------------------------------------- #
    def _load_simchange(self):
        import json
        calib = json.loads((self.root / "calib.json").read_text())
        self.left_paths = sorted((self.root / "left").glob("*.png"))
        right_dir = self.root / "right"
        baseline = float(calib.get("baseline", 0.12))
        if self.requested_baseline is not None:
            key = f"{self.requested_baseline:.2f}"
            dirs = calib.get("right_dirs", {})
            if key not in dirs:
                raise ValueError(f"baseline {key} not rendered for {self.root} (available: {list(dirs)})")
            right_dir = self.root / dirs[key]
            baseline = self.requested_baseline
        self.right_paths = sorted(right_dir.glob("*.png"))
        d = self.root / "depth"             # SimChange: depth/*.png (uint16 mm; current renders) or depth/*.npy (m, v1)
        self.depth_paths = (sorted(d.glob("*.png")) or sorted(d.glob("*.npy"))) if d.is_dir() else []
        assert len(self.left_paths) == len(self.right_paths) > 0
        self.left_c2w = np.loadtxt(self.root / "poses_left.txt").reshape(-1, 4, 4)
        assert len(self.left_c2w) == len(self.left_paths)
        self.T_right_in_left = np.asarray(calib["T_right_in_left"], dtype=np.float64)
        self.T_right_in_left[0, 3] = baseline
        self.rgb_K = np.asarray(calib["K"], dtype=np.float64)
        self.rgb_width, self.rgb_height = int(calib["width"]), int(calib["height"])
        self._default_fps = float(calib.get("fps", 10.0))
        self.simchange_meta = calib
        if (self.root / "odom_left.txt").is_file():   # the platform's own odometry (benchmark stereo folders)
            self.odom_c2w = np.loadtxt(self.root / "odom_left.txt", dtype=np.float64).reshape(-1, 4, 4)
            assert len(self.odom_c2w) == len(self.left_paths)
            self.snr = None
            self.odom_scale_bias = self.odom_yaw_drift = 0.0

    # ---- posed RGB-D folder (monocular; OpenLORIS / TUM RGB-D converted by the CROSS scripts) ---- #
    def _load_posed_rgbd(self):
        """rgb/*.png, optional depth/*.png (uint16 mm) or *.npy (m), poses_left.txt (ground-truth c2w, 16 values per row),
        optional odom_left.txt (the robot's own odometry as camera poses: used instead of simulated noisy odometry),
        calib.json {K, width, height, fps}.  No stereo rig (T_right_in_left None): the feed-forward estimator takes its
        metric scale from the odometry / map anchors (pose_est.ff.use_odom_anchor, use_map_anchors)."""
        import json
        calib = json.loads((self.root / "calib.json").read_text())
        self.left_paths = sorted((self.root / "rgb").glob("*.png")) or sorted((self.root / "rgb").glob("*.jpg"))
        self.mono = True
        self.right_paths = [None] * len(self.left_paths)
        d = self.root / "depth"
        self.depth_paths = (sorted(d.glob("*.png")) or sorted(d.glob("*.npy"))) if d.is_dir() else []
        self.left_c2w = np.loadtxt(self.root / "poses_left.txt", dtype=np.float64).reshape(-1, 4, 4)
        assert len(self.left_c2w) == len(self.left_paths)
        if (self.root / "odom_left.txt").is_file():
            self.odom_c2w = np.loadtxt(self.root / "odom_left.txt", dtype=np.float64).reshape(-1, 4, 4)
            assert len(self.odom_c2w) == len(self.left_paths)
            self.snr = None                        # real odometry: no simulated noise or drift
            self.odom_scale_bias = self.odom_yaw_drift = 0.0
        self.T_right_in_left = None
        self.rgb_K = np.asarray(calib["K"], dtype=np.float64)
        self.rgb_width, self.rgb_height = int(calib["width"]), int(calib["height"])
        self._default_fps = float(calib.get("fps", 10.0))

    def _read_depth_png_mm(self, path: Path) -> np.ndarray:
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(path)
        return d.astype(np.float32) / 1000.0

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.left_paths)

    def get_sequence_frequency(self):
        return float(self.fps)

    def get_idx_from_timestamp(self, timestamp):
        idx = int(np.searchsorted(self._timestamps, float(timestamp), side="right"))
        return idx if idx < len(self) else None

    def _read(self, path: Path) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

    def _depth(self, idx: int, left: np.ndarray, right: np.ndarray) -> Optional[np.ndarray]:
        if self.depth_source == "gt":
            if not getattr(self, "depth_paths", None):
                raise ValueError("ground-truth depth requested but the sequence has no depth/ folder")
            p = self.depth_paths[idx]
            depth = self._read_depth_png_mm(p) if p.suffix == ".png" else np.load(p).astype(np.float32)
            depth[~np.isfinite(depth)] = 0.0
            depth[depth > self.max_depth] = 0.0
            return depth
        if self.depth_source != "sgbm":
            return None
        name = f"{self.left_paths[idx].stem}.npy"
        cache = self.root / ".cross_sgbm_depth" / name          # written next to the data by earlier versions
        if not cache.is_file():
            cache = sgbm_cache_dir(self.root) / name
        if self.depth_cache and cache.is_file():
            depth = np.load(cache).astype(np.float32)
        else:
            depth = self._sgbm(left, right, float(self.orig_K[0, 0]), self.baseline)
            if self.depth_cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, depth.astype(np.float16))
        depth[depth > self.max_depth] = 0.0
        return depth

    def __getitem__(self, idx: int):
        left = self._read(self.left_paths[idx])
        right = None if self.right_paths[idx] is None else self._read(self.right_paths[idx])
        depth = self._depth(idx, left, right)
        if self._resize_cfg is not None:
            left = self._resize_image(left, self._resize_cfg, "rgb")
            right = self._resize_image(right, self._resize_cfg, "rgb") if right is not None else None
            depth = self._resize_image(depth, self._resize_cfg, "depth") if depth is not None else None
        src = self.left_c2w if self.odom_c2w is None else self.odom_c2w
        if idx == 0:
            delta = np.eye(4)
        else:
            delta = _invert(src[idx - 1]) @ src[idx]
        return {
            "rgb": left,
            "rgb_right": right,
            "depth": depth,
            "conf": None,
            "delta_pose": delta,
            "world_pose": self.left_c2w[idx],
            "timestamp": float(self._timestamps[idx]),
            "frame_idx": idx,
        }

    def replay_data(self, fps=None, start_timestamp=None, end_timestamp=None, start_idx=0, end_idx=None, stride=1):
        """Replay without sleeping (offline evaluation); supports frame stride."""
        if end_idx is None:
            end_idx = len(self)
        end_idx = min(end_idx, len(self))
        prev_idx = None
        for i in range(start_idx, end_idx, stride):
            item = self.get_item(i, first_item=(prev_idx is None))
            if prev_idx is not None and stride > 1 and self.odom_c2w is not None:
                item["delta_pose"] = _invert(self.odom_c2w[prev_idx]) @ self.odom_c2w[i]
            elif prev_idx is not None and stride > 1:
                # odometry over the skipped frames: compose consecutive GT deltas (with noise per step)
                T = np.eye(4)
                for j in range(prev_idx + 1, i + 1):
                    d = _invert(self.left_c2w[j - 1]) @ self.left_c2w[j]
                    d = self._bias_delta(d, self.left_c2w[j])
                    if self.snr is not None:
                        d = self._noise_delta(d, self.left_c2w[j])
                    T = T @ d
                item["delta_pose"] = T
            prev_idx = i
            yield item
