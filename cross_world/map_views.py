"""The keyframes of a saved CROSS map as posed views, the input of the reconstruction.

A permanent keyframe of a map holds its image as the mode stored it (resized / cropped to <= 512 px: what the pose
estimator saw), sensor depth in the RGB-D mode, the right image in the stereo mode, a timestamp, and its pose as a
Gaussian mixture; hypothesis 0's component is the map pose (camera-to-map, OpenCV camera axes).  Temporary keyframes
have no image and are not views.  The covisibility graph is hypothesis 0's visual edges between permanent keyframes.

With `source` (a prepared sequence folder: calib.json, left/ or rgb/, right/, depth/, times.txt, poses_left.txt) the
views use the source frames at their full resolution instead of the stored images: the keyframe's timestamp finds
its frame, the pose stays the map's (the same camera; only the intrinsics differ).  The source also gives frames that
are not keyframes, with ground-truth poses, as test views of novel viewpoints.

Maps saved before the `camera` field existed need the intrinsics from elsewhere: `source` (its calib.json and the
mode's image transform) or `K` / `size` arguments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (q / np.linalg.norm(q)).tolist()
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose7_to_mat(p: np.ndarray) -> np.ndarray:
    """[x y z qx qy qz qw] (pypose SE3) -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_R(np.asarray(p[3:7], np.float64))
    T[:3, 3] = p[:3]
    return T


@dataclass
class View:
    """One posed image.  Images load lazily (a map of 10^5 keyframes does not fit in memory decoded)."""
    id: int                              # keyframe id (or source frame index for test views)
    T_wc: np.ndarray                     # camera-to-map pose, OpenCV axes (x right, y down, z forward)
    K: np.ndarray                        # intrinsics of `image()`
    width: int
    height: int
    timestamp: Optional[float] = None
    source_index: Optional[int] = None   # frame of the source sequence
    atlas: Optional[int] = None
    kind: str = "keyframe"               # keyframe | capture (mapping.world_capture frame) | test (source frame)
    _image: Callable[[], np.ndarray] = field(default=None, repr=False)      # (H, W, 3) uint8 RGB
    _depth: Optional[Callable[[], np.ndarray]] = field(default=None, repr=False)   # (H, W) float32 metres, 0 = none
    _right: Optional[Callable[[], np.ndarray]] = field(default=None, repr=False)   # (H, W, 3) uint8 RGB

    def image(self) -> np.ndarray:
        return self._image()

    def depth(self) -> Optional[np.ndarray]:
        return self._depth() if self._depth is not None else None

    def right(self) -> Optional[np.ndarray]:
        return self._right() if self._right is not None else None

    @property
    def has_depth(self) -> bool:
        return self._depth is not None

    @property
    def has_right(self) -> bool:
        return self._right is not None

    @property
    def center(self) -> np.ndarray:
        return self.T_wc[:3, 3]


def _resize_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K = K.copy()
    K[0, :] *= sx
    K[1, :] *= sy
    return K


def _fit(img: np.ndarray, size: Tuple[int, int], nearest: bool = False) -> np.ndarray:
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


class SourceSequence:
    """A prepared sequence folder (the benchmark's layout, cross/dataloader/posed_rgbd.py and stereo_loader.py)."""

    def __init__(self, root):
        self.root = Path(root)
        calib = json.loads((self.root / "calib.json").read_text())
        self.K = np.asarray(calib["K"], np.float64)
        self.width, self.height = int(calib["width"]), int(calib["height"])
        self.T_right_in_left = np.asarray(calib["T_right_in_left"], np.float64) if calib.get("T_right_in_left") else None
        left = self.root / "left" if (self.root / "left").is_dir() else self.root / "rgb"
        self.left = sorted(left.glob("*.png")) or sorted(left.glob("*.jpg"))
        right = self.root / "right"
        self.right = (sorted(right.glob("*.png")) or sorted(right.glob("*.jpg"))) if right.is_dir() else []
        dd = self.root / "depth"
        self.depth = (sorted(dd.glob("*.npy")) or sorted(dd.glob("*.png"))) if dd.is_dir() else []
        if self.right and len(self.right) != len(self.left):
            self.right = []
        if self.depth and len(self.depth) != len(self.left):
            self.depth = []
        tf = self.root / "times.txt"
        self.times = np.loadtxt(tf).reshape(-1) if tf.exists() else None
        pf = self.root / "poses_left.txt"
        self.gt = np.loadtxt(pf).reshape(-1, 4, 4) if pf.exists() else None
        self.fps = float(calib.get("fps", 10.0))

    def __len__(self):
        return len(self.left)

    def _bases(self):
        n = len(self.left)
        out = []
        if self.times is not None and len(self.times) == n:
            out += [self.times, self.times - self.times[0]]
        out.append(np.arange(n) / self.fps)
        return out

    def calibrate(self, timestamps) -> None:
        """Pick the clock the map's timestamps use.  Loaders differ: times.txt itself, times.txt relative to its first
        entry, or frame index / fps (cross/dataloader/stereo_loader.py).  The base that matches the most timestamps
        within a quarter frame is used for every lookup (one base per sequence: a per-timestamp choice could match
        a drifting clock to the wrong frame)."""
        ts = np.asarray([t for t in timestamps if t is not None], np.float64)
        best, best_n = None, -1
        for b in self._bases():
            if not len(ts):
                break
            i = np.clip(np.searchsorted(b, ts), 1, len(b) - 1)
            d = np.minimum(np.abs(b[i] - ts), np.abs(b[i - 1] - ts))
            n_ok = int((d < 0.25 / self.fps).sum())
            if n_ok > best_n:
                best, best_n = b, n_ok
        self._base = best

    def time_of(self, i: int) -> float:
        """Timestamp of frame i on the map's clock (after calibrate)."""
        b = getattr(self, "_base", None)
        return float((b if b is not None else self._bases()[0])[i])

    def index_of(self, timestamp: float) -> Optional[int]:
        b = getattr(self, "_base", None)
        if b is None:
            b = self._bases()[0]
        i = int(np.argmin(np.abs(b - timestamp)))
        return i if abs(b[i] - timestamp) < 0.25 / self.fps else None

    def read_image(self, i: int, size=None) -> np.ndarray:
        img = cv2.imread(str(self.left[i]), cv2.IMREAD_COLOR)[..., ::-1]
        return np.ascontiguousarray(_fit(img, size) if size else img)

    def read_right(self, i: int, size=None) -> np.ndarray:
        img = cv2.imread(str(self.right[i]), cv2.IMREAD_COLOR)[..., ::-1]
        return np.ascontiguousarray(_fit(img, size) if size else img)

    def read_depth(self, i: int, size=None) -> np.ndarray:
        p = self.depth[i]
        d = np.load(p).astype(np.float32) if p.suffix == ".npy" else cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        d[~np.isfinite(d)] = 0
        return _fit(d, size, nearest=True) if size else d


@dataclass
class MapViews:
    """The posed keyframes of a map, its covisibility graph and camera rig."""
    views: List[View]
    edges: List[Tuple[int, int, float]]           # (keyframe id, keyframe id, covisibility confidence or 1)
    T_right_in_left: Optional[np.ndarray]
    mode: str                                     # rgbd | stereo | mono (what the stored keyframes hold)
    map_path: str
    source: Optional[SourceSequence] = None
    temporary_poses: Dict[int, np.ndarray] = field(default_factory=dict)   # imageless keyframes (trajectory only)

    def by_id(self) -> Dict[int, View]:
        return {v.id: v for v in self.views}

    def centers(self) -> np.ndarray:
        return np.array([v.center for v in self.views])

    def up(self) -> np.ndarray:
        """Vertical of the map: the mean 'up' (-y) of the keyframe cameras, for robots that keep the camera roughly
        level.  Used only to split large maps on the ground plane."""
        u = -np.mean([v.T_wc[:3, 1] for v in self.views], axis=0)
        return u / max(np.linalg.norm(u), 1e-9)


def _pose_component(rec: dict, comp: Optional[int]) -> np.ndarray:
    mu = rec["pose_mu"]
    mu = mu.tensor() if hasattr(mu, "tensor") else mu
    mu = np.asarray(mu.cpu().numpy() if hasattr(mu, "cpu") else mu, np.float64).reshape(-1, 7)
    if comp is None or comp >= len(mu):
        w = rec.get("pose_weights")
        comp = int(np.argmax(np.asarray(w.cpu().numpy() if hasattr(w, "cpu") else w))) if w is not None else 0
    return pose7_to_mat(mu[comp])


def _ref_loader(ref, kind: str):
    """Decode a stored keyframe image (an ImageRef or a tensor) as numpy."""
    def load():
        t = ref.load() if hasattr(ref, "load") else ref
        a = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
        if kind == "depth":
            return a.reshape(a.shape[-2:]).astype(np.float32)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
        return np.ascontiguousarray(a.transpose(1, 2, 0))
    return load


def load_map_views(map_path, source=None, *, max_side: Optional[int] = None, K: Optional[np.ndarray] = None,
                   size: Optional[Tuple[int, int]] = None, use_source_depth: bool = True,
                   use_capture: bool = True) -> MapViews:
    """The permanent keyframes of a saved map as views.

    source: prepared sequence folder (or SourceSequence) of the mapping session; its full-resolution frames replace
    the stored images (resized so that the longer side is at most `max_side`).  Without it, a keyframe whose own frame
    the map's capture directory holds (mapping.world_capture) takes that frame (full resolution, uncropped);
    otherwise the stored image is used, with the intrinsics saved in the map (or `K` / `size`)."""
    from cross.db import store
    data = store.read_map(str(map_path))
    db = data["db_data"]
    hd = data["hypo_data"]["hypotheses_data"]
    comp = hd[0].get("component_id") if 0 in hd else None
    cam = data.get("camera") or {}
    src = SourceSequence(source) if isinstance(source, (str, Path)) else source
    T_rl = cam.get("T_right_in_left") or (src.T_right_in_left.tolist() if src is not None and src.T_right_in_left is not None else None)
    T_rl = np.asarray(T_rl, np.float64) if T_rl is not None else None

    K_map = np.asarray(cam["K"], np.float64) if cam.get("K") is not None else (np.asarray(K, np.float64) if K is not None else None)
    size_map = (int(cam["width"]), int(cam["height"])) if cam.get("width") else size
    if src is None and K_map is None:
        raise ValueError(f"{map_path}: the map does not hold its intrinsics (saved before the camera field); pass the "
                         f"source folder or K / size")
    if src is not None:
        s = 1.0 if not max_side else min(1.0, max_side / max(src.width, src.height))
        w_s, h_s = int(round(src.width * s)), int(round(src.height * s))
        K_src = _resize_K(src.K, w_s / src.width, h_s / src.height)

    if src is not None:
        src.calibrate([r.get("timestamp") for r in db["keyframes"]])
    views, any_depth, any_right = [], False, False
    for rec in db["keyframes"]:
        if rec.get("temporary"):
            continue
        img_ref, dep_ref, right_ref = rec.get("raw_rgb_image"), rec.get("depth_image"), rec.get("raw_rgb_right")
        if img_ref is None:
            continue
        T = _pose_component(rec, comp)
        ts = rec.get("timestamp")
        si = src.index_of(ts) if (src is not None and ts is not None) else None
        any_depth |= dep_ref is not None
        any_right |= right_ref is not None
        if src is not None and si is not None:
            v = View(id=int(rec["id"]), T_wc=T, K=K_src, width=w_s, height=h_s, timestamp=ts, source_index=si,
                     atlas=rec.get("atlas_id"),
                     _image=(lambda i=si: src.read_image(i, (w_s, h_s))))
            if src.depth and use_source_depth:
                v._depth = lambda i=si: src.read_depth(i, (w_s, h_s))
            elif dep_ref is not None:          # stored depth, upsampled to the source resolution (nearest)
                v._depth = lambda r=dep_ref: _fit(_ref_loader(r, "depth")(), (w_s, h_s), nearest=True)
            if src.right:
                v._right = lambda i=si: src.read_right(i, (w_s, h_s))
        else:
            if K_map is None:
                raise ValueError(f"keyframe {rec['id']}: no source frame and no stored intrinsics")
            h, w = (img_ref.shape[-2:] if hasattr(img_ref, "shape") else (size_map[1], size_map[0]))
            v = View(id=int(rec["id"]), T_wc=T, K=K_map, width=int(w), height=int(h), timestamp=ts,
                     atlas=rec.get("atlas_id"), _image=_ref_loader(img_ref, "rgb"))
            if dep_ref is not None:
                v._depth = _ref_loader(dep_ref, "depth")
            if right_ref is not None:
                v._right = _ref_loader(right_ref, "rgb")
        views.append(v)

    if src is None and use_capture:
        _keyframes_from_capture(views, map_path, max_side)
    if src is not None:
        n_src = sum(v.source_index is not None for v in views)
        if n_src < len(views):
            import warnings
            warnings.warn(f"{map_path}: {len(views) - n_src} of {len(views)} keyframes have no frame in {src.root}; "
                          f"they use the stored images")
    ids = {v.id for v in views}
    edges = []
    if 0 in hd:
        for key, lst in hd[0]["visual_edges"].items():
            a, b = int(key[0]), int(key[1])
            if a in ids and b in ids and a != b and len(lst):
                conf = [e.get("conf") for e in lst if isinstance(e, dict)]
                conf = [c for c in conf if c is not None]
                edges.append((a, b, float(max(conf)) if conf else 1.0))
    temp = {}
    for rec in data["hypo_data"]["temp_keyframes"]:
        try:
            temp[int(rec["id"])] = _pose_component(rec, comp)
        except Exception:
            pass
    mode = "rgbd" if any_depth else ("stereo" if any_right else "mono")
    return MapViews(views=views, edges=edges, T_right_in_left=T_rl, mode=mode, map_path=str(map_path), source=src,
                    temporary_poses=temp)


def _keyframes_from_capture(views: List[View], map_path, max_side: Optional[int]) -> int:
    """Keyframes whose own input frame is in the map's capture directory use it (same frame and pose; full
    resolution, uncropped, with its right image / depth) instead of the stored image.  Returns how many."""
    from cross.core.world_capture import capture_dir
    d = capture_dir(map_path)
    f = d / "capture.json"
    if not f.exists():
        return 0
    idx = json.loads(f.read_text())
    W, H = int(idx["width"]), int(idx["height"])
    s = 1.0 if not max_side else min(1.0, max_side / max(W, H))
    w_s, h_s = int(round(W * s)), int(round(H * s))
    K = _resize_K(np.asarray(idx["K"], np.float64), w_s / W, h_s / H)
    by_t = {round(float(fr["timestamp"]), 6): fr for fr in idx["frames"] if fr.get("timestamp") is not None}
    fdir = d / "frames"
    missing = [v.id for v in views if v.timestamp is None or round(float(v.timestamp), 6) not in by_t]
    if missing:                 # all or none: the views of a chunk share one image size
        import warnings
        warnings.warn(f"{map_path}: {len(missing)} of {len(views)} keyframes have no captured frame; the stored "
                      f"images are used")
        return 0

    def rgb(path):
        return lambda: np.ascontiguousarray(_fit(cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1], (w_s, h_s)))

    def dep(path):
        def load():
            a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).view(np.float16).astype(np.float32)
            a[~np.isfinite(a)] = 0
            return _fit(a, (w_s, h_s), nearest=True)
        return load
    n = 0
    for v in views:
        fr = by_t.get(round(float(v.timestamp), 6)) if v.timestamp is not None else None
        if fr is None:
            continue
        files = fr["files"]
        v.K, v.width, v.height = K, w_s, h_s
        v._image = rgb(fdir / files["rgb"])
        v._depth = dep(fdir / files["depth"]) if "depth" in files else (None if v._depth is None else
                                                                        (lambda o=v._depth: _fit(o(), (w_s, h_s), nearest=True)))
        v._right = rgb(fdir / files["right"]) if "right" in files else None
        n += 1
    return n


def source_test_views(mv: MapViews, *, every: int = 1, min_gap: int = 1, max_side: Optional[int] = None,
                      limit: Optional[int] = None) -> List[View]:
    """Frames of the source sequence that are not keyframes: test views of viewpoints the map never stored.  A frame
    is posed by its ground-truth motion relative to the keyframe nearest in time, applied to that keyframe's map pose
    (T_map(f) = T_map(kf) T_gt(kf)^-1 T_gt(f)): locally as accurate as the ground truth, without the map's global
    drift.  `min_gap`: only frames at least this many frames away from every keyframe."""
    src = mv.source
    if src is None or src.gt is None:
        return []
    s = 1.0 if not max_side else min(1.0, max_side / max(src.width, src.height))
    w_s, h_s = int(round(src.width * s)), int(round(src.height * s))
    K_src = _resize_K(src.K, w_s / src.width, h_s / src.height)
    kfs = sorted(((v.source_index, v) for v in mv.views if v.source_index is not None), key=lambda r: r[0])
    if not kfs:
        return []
    kf_idx = np.array([k for k, _ in kfs])
    out = []
    for i in range(0, len(src), every):
        j = int(np.argmin(np.abs(kf_idx - i)))
        if abs(int(kf_idx[j]) - i) < min_gap:
            continue
        kv = kfs[j][1]
        T = kv.T_wc @ np.linalg.inv(src.gt[kv.source_index]) @ src.gt[i]
        v = View(id=-1 - i, T_wc=T, K=K_src, width=w_s, height=h_s,
                 timestamp=src.time_of(i), source_index=i,
                 _image=(lambda q=i: src.read_image(q, (w_s, h_s))))
        if src.depth:
            v._depth = lambda q=i: src.read_depth(q, (w_s, h_s))
        out.append(v)
    if limit and len(out) > limit:
        sel = np.linspace(0, len(out) - 1, limit).round().astype(int)
        out = [out[k] for k in sel]
    return out


def _blend_poses(Ts: List[np.ndarray], w: np.ndarray) -> np.ndarray:
    """Weighted blend of nearby poses: weighted mean position, normalised weighted mean of sign-aligned quaternions."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.stack([T[:3, :3] for T in Ts])).as_quat()
    q = q * np.where((q @ q[0]) < 0, -1.0, 1.0)[:, None]
    qm = (w[:, None] * q).sum(0)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(qm / np.linalg.norm(qm)).as_matrix()
    T[:3, 3] = (w[:, None] * np.stack([T_[:3, 3] for T_ in Ts])).sum(0)
    return T


def load_capture_views(mv: MapViews, *, max_side: Optional[int] = None, exclude_times: Optional[np.ndarray] = None,
                       exclude_dt: float = 0.15) -> List[View]:
    """The frames of the map's capture directory (cross/core/world_capture.py, mapping.world_capture) between the
    keyframes as views, posed by their anchor keyframes' poses in the saved map (T_kf T_rel, blended).  Frames whose
    timestamp is within `exclude_dt` of `exclude_times` (the evaluation frames) or of a keyframe (its own frame, which
    the keyframe view already uses) are left out."""
    from cross.core.world_capture import capture_dir
    d = capture_dir(mv.map_path)
    f = d / "capture.json"
    if not f.exists():
        return []
    idx = json.loads(f.read_text())
    W, H = int(idx["width"]), int(idx["height"])
    s = 1.0 if not max_side else min(1.0, max_side / max(W, H))
    w_s, h_s = int(round(W * s)), int(round(H * s))
    K = _resize_K(np.asarray(idx["K"], np.float64), w_s / W, h_s / H)
    poses = {v.id: v.T_wc for v in mv.views}
    poses.update(mv.temporary_poses)
    ex = np.asarray(exclude_times, np.float64) if exclude_times is not None and len(exclude_times) else None
    kf_t = np.array([v.timestamp for v in mv.views if v.timestamp is not None], np.float64)
    fdir = d / "frames"
    out = []

    def rgb_loader(path):
        return lambda: np.ascontiguousarray(_fit(cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1], (w_s, h_s)))

    def depth_loader(path):
        def load():
            a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).view(np.float16).astype(np.float32)
            a[~np.isfinite(a)] = 0
            return _fit(a, (w_s, h_s), nearest=True)
        return load
    for fr in idx["frames"]:
        t = fr.get("timestamp")
        if t is not None and ((ex is not None and np.min(np.abs(ex - t)) < exclude_dt)
                              or (len(kf_t) and np.min(np.abs(kf_t - t)) < 1e-6)):
            continue
        an = [a for a in fr["anchors"] if a["kf"] in poses]
        if not an:
            continue
        Ts = [poses[a["kf"]] @ np.asarray(a["T_rel"], np.float64).reshape(4, 4) for a in an]
        w = np.array([a["w"] for a in an], np.float64)
        T = _blend_poses(Ts, w / w.sum())
        files = fr["files"]
        v = View(id=-(10 ** 7) - int(fr["n"]), T_wc=T, K=K, width=w_s, height=h_s, timestamp=t, kind="capture",
                 _image=rgb_loader(fdir / files["rgb"]))
        if "depth" in files:
            v._depth = depth_loader(fdir / files["depth"])
        if "right" in files:
            v._right = rgb_loader(fdir / files["right"])
        out.append(v)
    return out
