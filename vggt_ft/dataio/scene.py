"""The scene cache: one format for every training / evaluation dataset.

    <root>/<dataset>/<scene>/scene.json
    <root>/<dataset>/<scene>/<sequence>/frames.npz      K (N,3,3) f32, E (N,4,4) f64 camera-from-world (OpenCV axes),
                                                        names (N,) str, dmed (N,) median valid depth,
                                                        valid (N,) fraction of valid depth, t (N,) seconds (optional)
    <root>/<dataset>/<scene>/<sequence>/rgb/<name>.jpg
    <root>/<dataset>/<scene>/<sequence>/depth/<name>.png  uint16, log-encoded depth (dataio/depthcodec.py; 0 = invalid)
    <root>/<dataset>/<scene>/<sequence>/vpr.npy         optional (N, D) f16 global descriptors (hard negatives)

scene.json: {"dataset", "scene", "world", "metric", "synthetic", "dynamic", "kind", "units", "posed" (optional,
             default true; false = depth-only sequences without reliable poses),
             "sequences": {name: {"session": str, "n": int}}}

A scene is one place.  Its sequences are sessions; all sequences of a scene share one world frame (`world`, also the
key that marks two scenes as the same place when a dataset splits one place into several scene folders), so
cross-session windows and GT covisibility between them are valid.  `metric`: depth and poses in metres (the scale head
is trained only on metric windows; non-metric scenes are stored rescaled to a median depth of ~2 units, `units` records
the factor).  Images are stored at a long side of at most 768 px (`vggt_ft/prep/common.py`).

`Scene.scale` multiplies the stored depth and translations at load time: a non-metric scene with a pseudo-metric factor
(`vggt_ft/prep/pseudo_metric.py`, applied by the sampler's `pseudo_metric` option) is loaded in metres.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .depthcodec import decode

cv2.setNumThreads(1)


@dataclass
class Sequence:
    scene: "Scene"
    name: str
    session: str
    K: np.ndarray
    E: np.ndarray
    names: np.ndarray
    t: Optional[np.ndarray] = None
    _centres: Optional[np.ndarray] = None
    _vpr: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def dir(self) -> Path:
        return self.scene.dir / self.name

    @property
    def centres(self) -> np.ndarray:
        """(N,3) camera centres in the world frame."""
        if self._centres is None:
            R, t = self.E[:, :3, :3], self.E[:, :3, 3]
            self._centres = -np.einsum("nji,nj->ni", R, t)
        return self._centres

    @property
    def vpr(self) -> Optional[np.ndarray]:
        if self._vpr is None:
            p = self.dir / "vpr.npy"
            if p.exists():
                # read fully (a few MB): memory-mapped files on the shared JuiceFS mount can raise SIGBUS
                self._vpr = np.load(p)
        return self._vpr

    def load(self, i: int):
        """(rgb uint8 HxWx3, depth f32 HxW, K 3x3 f64, E 4x4 f64)."""
        nm = self.names[i]
        img = cv2.imread(str(self.dir / "rgb" / f"{nm}.jpg"), cv2.IMREAD_COLOR)
        dep = cv2.imread(str(self.dir / "depth" / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
        if img is None or dep is None:
            raise FileNotFoundError(f"{self.dir} {nm}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dep = decode(dep)
        if self.scene.scale != 1.0:
            dep = dep * np.float32(self.scene.scale)
        if dep.shape != img.shape[:2]:
            dep = cv2.resize(dep, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return img, dep, self.K[i].astype(np.float64), self.E[i].astype(np.float64)


@dataclass
class Scene:
    dir: Path
    dataset: str
    name: str
    world: str
    metric: bool
    synthetic: bool
    dynamic: bool
    kind: str
    seq_info: dict
    scale: float = 1.0              # stored units -> metres (pseudo-metric scenes); applied to depth and translations
    pseudo_metric: bool = False
    posed: bool = True              # False: no reliable camera poses (depth-only sequences: scale head only)
    _seqs: dict = field(default_factory=dict)

    @property
    def world_id(self) -> int:
        return zlib.crc32(f"{self.dataset}:{self.world}".encode()) & 0x7FFFFFFF

    @property
    def n_frames(self) -> int:
        return sum(v["n"] for v in self.seq_info.values())

    @property
    def sequences(self) -> list[str]:
        return list(self.seq_info)

    def seq(self, name: str) -> Sequence:
        s = self._seqs.get(name)
        if s is None:
            z = np.load(self.dir / name / "frames.npz", allow_pickle=False)
            E = z["E"]
            if self.scale != 1.0:
                E = E.copy()
                E[:, :3, 3] *= self.scale
            s = Sequence(self, name, self.seq_info[name].get("session", name), z["K"], E, z["names"],
                         z["t"] if "t" in z else None)
            self._seqs[name] = s
        return s


def load_scene(d: Path) -> Scene:
    d = Path(d)
    m = json.load(open(d / "scene.json"))
    return Scene(d, m["dataset"], m["scene"], m.get("world", m["scene"]), bool(m.get("metric", True)),
                 bool(m.get("synthetic", False)), bool(m.get("dynamic", False)), m.get("kind", ""), m["sequences"],
                 posed=bool(m.get("posed", True)))


def is_val_scene(name: str, val_mod: int) -> bool:
    """Deterministic validation holdout of a training dataset: crc32(scene) % val_mod == 0 (val_mod 0 = none)."""
    return bool(val_mod) and zlib.crc32(name.encode()) % val_mod == 0


def list_scenes(root: Path, dataset: str, include=None, exclude=None, val_mod: int = 0, split: str = "train") -> list[Path]:
    """Scene folders of one dataset; include / exclude: lists of substrings of the scene name (None = no filter);
    val_mod > 0: split "train" drops the validation scenes (is_val_scene), split "val" keeps only them.
    An `index.json` (list of scene names, written by prep/common.py index) avoids listing a big directory."""
    base = Path(root) / dataset
    if not base.is_dir():
        return []
    idx = base / "index.json"
    names = json.load(open(idx)) if idx.exists() else sorted(p.name for p in base.iterdir()
                                                              if (p / "scene.json").exists())
    if include:
        names = [n for n in names if any(s in n for s in include)]
    if exclude:
        names = [n for n in names if not any(s in n for s in exclude)]
    if val_mod:
        names = [n for n in names if is_val_scene(n, val_mod) == (split == "val")]
    return [base / n for n in names]
