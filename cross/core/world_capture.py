"""Frames kept for a later 3D reconstruction of the map (cross_world), outside the map's database.

A map stores only its permanent keyframes, at the pose estimator's resolution (<= 512 px, cropped in the stereo
mode), which is too sparse and too small for a good radiance field.  With `mapping.world_capture.enabled` the system
also keeps the input frames it observes every `min_translation` metres / `min_rotation_deg` degrees, at their input
resolution and uncropped (plus the right image / depth where the input has them).  Each captured frame's pose is held
relative to its `anchors` nearest permanent keyframes (T_kf^-1 T_frame at capture time), so it follows the graph when a
later optimisation moves the keyframes: its pose in the saved map is the weighted blend of T_kf(saved) T_kf^-1 T_frame.

The frames go to a spool directory during the run and, at save_map, next to the map:

    map.pkl.capture/
        capture.json        camera (input intrinsics, size, stereo rig), frames: timestamp, files, anchors
        frames/             <n>_l.jpg, <n>_r.jpg, <n>_d.png (16-bit PNG of fp16 depth)

The map file and its database are unchanged; a map saved without the option has no capture directory.  Frames are
captured only while hypothesis 0 is a pose in the stored map (System.session_localized); a session that loaded a
map keeps the stored map's frames and adds its own.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

SUFFIX = ".capture"


def capture_dir(map_path) -> Path:
    return Path(os.path.realpath(str(map_path)) + SUFFIX)


def _rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def _inv(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


class WorldCapture:
    def __init__(self, cfg, K: np.ndarray, width: int, height: int, T_right_in_left: Optional[np.ndarray] = None):
        self.cfg = cfg
        s = 1.0
        if cfg.max_side and max(width, height) > cfg.max_side:
            s = cfg.max_side / max(width, height)
        self.scale = s
        self.width, self.height = int(round(width * s)), int(round(height * s))
        K = np.asarray(K, np.float64).copy()
        K[:2] *= np.array([[self.width / width], [self.height / height]])
        self.K = K
        self.T_right_in_left = None if T_right_in_left is None else np.asarray(T_right_in_left, np.float64)
        self.frames: List[dict] = []                 # index entries; "dir" = where its files are now
        self.spool = Path(tempfile.mkdtemp(prefix="cross_capture_"))
        (self.spool / "frames").mkdir()
        self._last_T: Optional[np.ndarray] = None
        self._n = 0
        ext = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}.get(cfg.image_codec)
        if ext is None:
            raise ValueError(f"world_capture.image_codec: jpeg | png | webp, not {cfg.image_codec}")
        self._ext = ext

    # ------------------------------------------------------------------ live
    def reset_session(self) -> None:
        self._last_T = None

    def wants(self, T: np.ndarray, new_keyframe: bool = False) -> bool:
        """A frame every min_translation / min_rotation_deg, and every frame that became a keyframe (so that each
        keyframe has its uncropped, full-resolution image)."""
        if self._last_T is None or new_keyframe:
            return True
        d = _inv(self._last_T) @ T
        return np.linalg.norm(d[:3, 3]) >= self.cfg.min_translation or _rot_deg(d[:3, :3]) >= self.cfg.min_rotation_deg

    def _write_rgb(self, path: Path, img: np.ndarray) -> None:
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        if self.scale != 1.0:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        bgr = img[..., ::-1] if img.ndim == 3 else img
        params = [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.image_quality)] if self._ext == ".jpg" else \
            ([cv2.IMWRITE_WEBP_QUALITY, int(self.cfg.image_quality)] if self._ext == ".webp" else [])
        cv2.imwrite(str(path), np.ascontiguousarray(bgr), params)

    def add(self, T: np.ndarray, anchors: List[tuple], obs: dict, timestamp) -> None:
        """Capture one frame: pose T (camera-to-map), anchors [(keyframe id, T_kf, distance)], the raw observation."""
        rgb = obs.get("rgb")
        if rgb is None or not anchors:
            return
        n = self._n
        self._n += 1
        files = {}
        f = f"{n:06d}_l{self._ext}"
        self._write_rgb(self.spool / "frames" / f, rgb)
        files["rgb"] = f
        if self.cfg.right and obs.get("rgb_right") is not None:
            f = f"{n:06d}_r{self._ext}"
            self._write_rgb(self.spool / "frames" / f, obs["rgb_right"])
            files["right"] = f
        if self.cfg.depth and obs.get("depth") is not None:
            d = np.asarray(obs["depth"], np.float32).reshape(np.asarray(obs["depth"]).shape[:2])
            if self.scale != 1.0:
                d = cv2.resize(d, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            f = f"{n:06d}_d.png"
            cv2.imwrite(str(self.spool / "frames" / f), np.nan_to_num(d).astype(np.float16).view(np.uint16))
            files["depth"] = f
        dist = np.array([a[2] for a in anchors], np.float64)
        w = 1.0 / np.maximum(dist, 1e-3)
        w = w / w.sum()
        self.frames.append({
            "n": n, "timestamp": None if timestamp is None else float(timestamp), "files": files, "dir": str(self.spool),
            "T_capture": T.reshape(-1).tolist(),
            "anchors": [{"kf": int(k), "w": float(wi), "T_rel": (_inv(Tk) @ T).reshape(-1).tolist()}
                        for (k, Tk, _), wi in zip(anchors, w)],
        })
        self._last_T = T.copy()

    # ------------------------------------------------------------------ persistence
    def save(self, map_path) -> Path:
        """Write the frames next to the saved map (files moved from the spool, or copied from a loaded map's
        directory) and the index."""
        out = capture_dir(map_path)
        (out / "frames").mkdir(parents=True, exist_ok=True)
        for fr in self.frames:
            src_dir = Path(fr["dir"])
            if src_dir.resolve() == out.resolve():
                continue
            for name in fr["files"].values():
                src, dst = src_dir / "frames" / name, out / "frames" / name
                if src_dir == self.spool:
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copy2(src, dst)
            fr["dir"] = str(out)
        index = {"version": 1, "K": self.K.tolist(), "width": self.width, "height": self.height,
                 "T_right_in_left": None if self.T_right_in_left is None else self.T_right_in_left.tolist(),
                 "frames": [{k: v for k, v in fr.items() if k != "dir"} for fr in self.frames]}
        tmp = out / "capture.json.tmp"
        tmp.write_text(json.dumps(index))
        os.replace(tmp, out / "capture.json")
        logger.info(f"World capture: {len(self.frames)} frames saved to {out}")
        return out

    def load(self, map_path) -> None:
        """Take over the frames of a loaded map (they are copied when this session saves elsewhere)."""
        d = capture_dir(map_path)
        f = d / "capture.json"
        if not f.exists():
            return
        idx = json.loads(f.read_text())
        frames = idx.get("frames", [])
        n0 = self._n
        for fr in frames:
            fr["dir"] = str(d)
        # new frames of this session get numbers after the stored ones
        self._n = max(n0, max((fr["n"] for fr in frames), default=-1) + 1)
        self.frames = frames + [fr for fr in self.frames]
        self._last_T = None

    def cleanup(self) -> None:
        shutil.rmtree(self.spool, ignore_errors=True)


def anchor_candidates(system, T: np.ndarray, k: int, recent: int = 300) -> List[tuple]:
    """The k permanent keyframes nearest to pose T among the session's recent keyframes and the ones retrieved at
    this step: (id, T_kf (4x4), distance).  Distance: camera centres plus 1 m per 30 degrees of viewing direction."""
    hm = system.hypothesis_manager
    from cross.core.types import Keyframe
    ids = set(range(max(0, Keyframe._next_id - recent), Keyframe._next_id))
    res = getattr(system, "_last_retrieved_results", None)
    if isinstance(res, dict):
        for key in ("keyframes", "kfs"):
            for kf in res.get(key) or []:
                if hasattr(kf, "id"):
                    ids.add(int(kf.id))
    comp = 0
    h0 = hm.hypotheses.get(0) if hasattr(hm, "hypotheses") else None
    if h0 is not None:
        comp = int(getattr(h0, "component_id", 0))
    cands = []
    for i in ids:
        kf = hm.nodes.get(i)
        if kf is None or kf.temporary or not kf.has_image("raw_rgb_image"):
            continue
        Tk = kf.pose_mu[comp].matrix().detach().cpu().numpy().astype(np.float64)
        d = float(np.linalg.norm(Tk[:3, 3] - T[:3, 3])) + _rot_deg(Tk[:3, :3].T @ T[:3, :3]) / 30.0
        cands.append((i, Tk, d))
    cands.sort(key=lambda c: c[2])
    return cands[:k]
