"""PointOdyssey v1.2 (aharley/pointodyssey on HF: sample / val / test / train.tar.gz*) -> scene cache.

Input: an extracted split root <raw>/<split>/<seq>/{rgbs/rgb_NNNNN.jpg (960 x 540), depths/depth_NNNNN.png (uint16,
metres = v / 65535 * 1000, 0 = invalid), anno.npz (intrinsics (N,3,3) of the stored image, extrinsics (N,4,4)
world-to-camera, OpenCV axes; the 2D / 3D tracks are unused), scene_info.json (the animated characters)}.  normals/ and
masks/ are not needed: extract with --exclude '*/normals/*' --exclude '*/masks/*'.
Conventions: utils/reprojection.py of the PointOdyssey repository (camera point = extrinsics @ world point, pixel =
K @ camera point / z, depth = z); checked with the GT overlap of nearby frames (`python -m vggt_ft.prep.common verify`).

Synthetic (Blender), dynamic (animated humans, animals and robots), metric (human-scale assets).  The indoor scenes
(`scene_*`) come with a third-person (`*_3rd`) and egocentric (`*_ego*`) cameras.  scene.json records `camera`
(ego / 3rd / other), `indoor` and the characters.  Licence: CC BY-NC-SA 4.0 (project page; MIT on the HF card).
Checked on val + test (27 scenes, 2026-10-07):
- GT overlap of nearby frames 0.72-0.94 per scene (median 0.889).
- Camera steps ~1-2.5 cm per stored frame; people at 1.5-4 m, often backlit by windows; median depth 2.5-5.9 m, under
  1 % of frames with median depth < 0.8 m (room scale, not near field).
- Volumetric fog leaves invalid (zero) depth pixels, 85-91 % valid; the valid values agree with their neighbourhood.
- `scene_recording_20210910_S05_S06_0_ego1` has non-rigid extrinsics (singular values 0.86-1.16 in 2905 of 3039
  frames, a camera parented to a deforming bone) and is rejected.

python -m vggt_ft.prep.pointodyssey <extracted_root> <cache_root> [--every 3] [--cap 600] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import traceback
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .common import SceneWriter
from .rawio import split

DATASET = "pointodyssey"


def camera_kind(seq: str) -> str:
    return "ego" if "_ego" in seq else "3rd" if seq.endswith("_3rd") else "other"


def _rotation(E: np.ndarray) -> np.ndarray:
    """float32-stored extrinsics: re-orthonormalise the rotation (a real convention / scale error is not hidden)."""
    E = E.astype(np.float64).copy()
    R = E[:3, :3]
    if np.abs(R @ R.T - np.eye(3)).max() > 1e-2:
        raise ValueError("extrinsics rotation is not orthonormal")
    U, _, Vt = np.linalg.svd(R)
    E[:3, :3] = U @ Vt
    return E


def convert(job):
    seq_dir, out, every, cap = job
    seq_dir = Path(seq_dir)
    scene = f"{seq_dir.parent.name}_{seq_dir.name}"
    if (Path(out) / DATASET / scene / "scene.json").exists():
        return scene, -1
    anno = np.load(seq_dir / "anno.npz")
    K, E = anno["intrinsics"], anno["extrinsics"]
    info = json.loads((seq_dir / "scene_info.json").read_text()) if (seq_dir / "scene_info.json").exists() else {}
    rgbs = sorted((seq_dir / "rgbs").glob("rgb_*.jpg"))[::every]
    rgbs = [p for p in rgbs if (seq_dir / "depths" / p.name.replace("rgb_", "depth_").replace(".jpg", ".png")).exists()]
    if len(rgbs) < 2:
        return scene, 0
    chars = info.get("character")
    extra = dict(camera=camera_kind(seq_dir.name), indoor=seq_dir.name.startswith("scene_"), split=seq_dir.parent.name,
                 characters=chars if isinstance(chars, list) else [chars] if chars else [])
    sw = SceneWriter(out, DATASET, scene, metric=True, synthetic=True, dynamic=True, kind=extra["camera"], extra=extra)
    for c, idx in enumerate(split(len(rgbs), cap)):
        q = sw.sequence(scene if c == 0 else f"{scene}_{c}", session=scene)
        for i in idx:
            p = rgbs[i]
            n = int(p.stem.split("_")[1])
            rgb = cv2.imread(str(p), cv2.IMREAD_COLOR)[..., ::-1]
            v = cv2.imread(str(seq_dir / "depths" / f"depth_{n:05d}.png"), cv2.IMREAD_ANYDEPTH)
            d = v.astype(np.float32) / 65535.0 * 1000.0
            d[v == 0] = 0
            q.add(f"{n:05d}", rgb, d, K[n], _rotation(E[n]))
    return scene, sw.close()


def _conv(job):
    try:
        return convert(job)
    except Exception as e:
        return Path(job[0]).name, f"ERROR {e!r} {traceback.format_exc(limit=3)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="extracted root holding <split>/<sequence>/anno.npz")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=3, help="frame stride (the renders are 30 fps)")
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    seqs = sorted(p.parent for p in Path(a.raw).glob("*/*/anno.npz"))
    print(f"{len(seqs)} sequences", flush=True)
    with Pool(a.workers) as p:
        for scene, n in p.imap_unordered(_conv, [(str(x), a.out, a.every, a.cap) for x in seqs]):
            print(scene, n, flush=True)


if __name__ == "__main__":
    main()
