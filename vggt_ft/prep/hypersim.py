"""Hypersim (HarrisonPENG/hypersim on HF: CUT3R-preprocessed) -> scene cache.

Input: <raw>/ai_XXX_YYY.tar.gz, members ai_XXX_YYY/cam_ZZ/NNNNNN_{rgb.png,depth.npy,cam.npz}; cam.npz holds
`intrinsics` (3x3, for the 1024 x 768 image) and `pose`.  Conventions (checked empirically, see `common verify`):
`pose` is camera-to-world in OpenCV axes, translations in metres; depth.npy is float32 z-depth in metres (NaN where the
renderer has no geometry, e.g. windows).  The rotations are stored with ~4 decimals, so they are re-orthonormalised.

One output scene per ai_XXX_YYY; its camera trajectories cam_ZZ are sequences (sessions) in the scene's world frame.
Every frame is kept (each trajectory has <= 100 frames with large steps).  The archive is streamed (members are in
frame order), so nothing is extracted to disk.

python -m vggt_ft.prep.hypersim <raw_dir> <cache_root> [--workers 16] [--sizes q.txt]
"""
from __future__ import annotations

import argparse
import io
import re
import tarfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .common import SceneWriter

MEMBER = re.compile(r"(ai_\d+_\d+)/(cam_\d+)/(\d+)_(rgb\.png|depth\.npy|cam\.npz)$")


def ready_files(raw: Path, pattern: str, sizes: dict) -> list[Path]:
    """Raw files with a .done marker whose size matches the expected one (when known)."""
    out = []
    for f in sorted(raw.glob(pattern)):
        if not Path(str(f) + ".done").exists():
            continue
        if f.name in sizes and f.stat().st_size != sizes[f.name]:
            print(f"size mismatch {f}: {f.stat().st_size} != {sizes[f.name]}", flush=True)
            continue
        out.append(f)
    return out


def load_sizes(queue: str | None) -> dict:
    """Expected sizes from a download queue file (lines "<kind> <repo> <path> <dataset> <bytes>")."""
    sizes = {}
    if queue and Path(queue).exists():
        for line in open(queue):
            p = line.split()
            if len(p) >= 5:
                sizes[Path(p[2]).name] = int(p[4])
    return sizes


def orthonormal(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        raise ValueError("reflection")
    return R


def cam_from_world(pose: np.ndarray) -> np.ndarray:
    T = np.asarray(pose, np.float64).copy()
    T[:3, :3] = orthonormal(T[:3, :3])
    return np.linalg.inv(T)


def convert_scene(job):
    tar_path, out_root = job
    scene = Path(tar_path).name.split(".")[0]
    if (Path(out_root) / "hypersim" / scene / "scene.json").exists():
        return scene, -1
    sw = SceneWriter(out_root, "hypersim", scene, world=scene, metric=True, synthetic=True, dynamic=False,
                     kind="indoor")
    seqs, cur = {}, {}
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            mm = MEMBER.search(m.name) if m.isfile() else None
            if not mm:
                continue
            _, cam, idx, kind = mm.groups()
            cur.setdefault((cam, idx), {})[kind] = tf.extractfile(m).read()
            fr = cur[(cam, idx)]
            if len(fr) < 3:
                continue
            del cur[(cam, idx)]
            z = np.load(io.BytesIO(fr["cam.npz"]))
            dep = np.load(io.BytesIO(fr["depth.npy"])).astype(np.float32)
            rgb = cv2.imdecode(np.frombuffer(fr["rgb.png"], np.uint8), cv2.IMREAD_COLOR)
            pose, K = z["pose"], np.asarray(z["intrinsics"], np.float64)
            if rgb is None or not np.isfinite(pose).all() or not np.isfinite(dep).any():
                continue
            if cam not in seqs:
                seqs[cam] = sw.sequence(cam, session=cam)
            seqs[cam].add(idx, rgb[..., ::-1], dep, K, cam_from_world(pose))
    return scene, sw.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sizes", help="download queue file with expected sizes (5th column)")
    ap.add_argument("--scenes", nargs="*")
    a = ap.parse_args()
    files = ready_files(Path(a.raw), "ai_*.tar.gz", load_sizes(a.sizes))
    if a.scenes:
        files = [f for f in files if f.name.split(".")[0] in a.scenes]
    jobs = [(str(f), a.out) for f in files
            if not (Path(a.out) / "hypersim" / f.name.split(".")[0] / "scene.json").exists()]
    print(f"{len(files)} archives ready, {len(jobs)} to convert", flush=True)
    if not jobs:
        return
    with Pool(min(a.workers, len(jobs))) as p:
        for scene, n in p.imap_unordered(convert_scene, jobs):
            print(scene, n, flush=True)


if __name__ == "__main__":
    main()
