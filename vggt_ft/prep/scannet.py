"""ScanNet (pllm-jt/cut3r-data on HF: processed_scannet.tar.gz split into part-a?, CUT3R preprocessing) -> scene cache.

Input: the parts read in order as one .tar.gz stream (arkitscenes.PartStream / stream_convert: each scene goes through
a local scratch folder, parts still downloading are waited for).  Members, contiguous per scene:
dust3r_data/processed_scannet/<split>/sceneXXXX_YY/{new_scene_metadata.npz (`images`: frame ids '00000'.. of the 30 Hz
video), depth/<id>.png (uint16 mm, 640 x 480, z-depth, 0 = invalid), color/<id>.jpg (640 x 480, the 1296 x 968 colour
image resized), cam/<id>.npz (`intrinsics` 3x3, `pose` 4x4 camera-to-world, OpenCV axes, metres)}.  The intrinsics are
the depth camera's (e.g. fx 577.87, c 319.5 / 239.5) and serve the resized colour image as well (CUT3R's choice; the
colour / depth registration of ScanNet is approximate).  Frames with a non-finite pose are dropped.
Conventions checked 2026-10-04 on scene0517_00 (GT overlap of frame pairs 1 / 3 / 5 apart in the stored 10 Hz
sequence, rel_tol 0.02): as stored 0.90 / 0.79 / 0.61, the depth read as ray length 0.84 / 0.38 / 0.19, the pose read
as world-to-cam 0.12 / 0.04 / 0.01.  Metric: Kinect depth (mm) and BundleFusion poses (m) agree at 2 % tolerance.

One scene per scan (sceneXXXX_YY; the scans YY of one room are not registered to each other, separate worlds), every
--every-th frame (default 3: 10 Hz), cut into consecutive chunks of <= --cap frames.  Split `scans_train` by default.

python -m vggt_ft.prep.scannet <raw_dir> <cache_root> [--workers 16] [--every 3] [--sizes q.txt ...]
python -m vggt_ft.prep.scannet --dir <extracted scene dir> <cache_root>     (one extracted scene, tests)
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

from .arkitscenes import PartStream, stream_convert
from .common import SceneWriter
from .rawio import dec_any, dec_rgb, split

DATASET = "scannet"
PREFIX = "processed_scannet.tar.gz.part-"


def make_scene_of(splits):
    def scene_of(name: str):
        p = name.split("/")
        if "processed_scannet" not in p:
            return None
        i = p.index("processed_scannet")
        if len(p) < i + 4 or p[i + 1] not in splits:
            return None
        return p[i + 2], p[i + 1]              # scan, split
    return scene_of


def make_keep(every: int):
    def keep(name: str):
        p = name.split("/")
        rest = p[p.index("processed_scannet") + 3:]
        if rest == ["new_scene_metadata.npz"]:
            return rest[0]
        if len(rest) == 2 and rest[0] in ("depth", "color", "cam"):
            stem = rest[1].rsplit(".", 1)[0]
            if stem.isdigit() and int(stem) % every == 0:
                return "/".join(rest)
        return None
    return keep


def convert_dir(d, scan, split_name, out, cap=600, every=3):
    d = Path(d)
    if (d / "new_scene_metadata.npz").exists():
        ids = [str(x) for x in np.load(d / "new_scene_metadata.npz", allow_pickle=True)["images"]]
    else:
        ids = sorted(p.stem for p in (d / "cam").glob("*.npz"))
    ids = sorted((i for i in ids if i.isdigit() and int(i) % every == 0), key=int)
    frames = []
    for i in ids:
        fc, fd, fk = d / "color" / f"{i}.jpg", d / "depth" / f"{i}.png", d / "cam" / f"{i}.npz"
        if not (fc.exists() and fd.exists() and fk.exists()):
            continue
        z = np.load(fk)
        T, K = z["pose"].astype(np.float64), z["intrinsics"].astype(np.float64)
        if np.isfinite(T).all() and np.isfinite(K).all() and abs(np.linalg.det(T[:3, :3]) - 1) < 1e-2:
            frames.append((i, T, K))
    if len(frames) < 2:
        return 0
    sw = SceneWriter(out, DATASET, scan, world=scan, metric=True, synthetic=False, dynamic=False, kind="indoor",
                     extra={"split": split_name, "depth": "kinect"})
    for c, idx in enumerate(split(len(frames), cap)):
        q = sw.sequence(scan if c == 0 else f"{scan}_{c}", session=scan)
        for k in idx:
            i, T, K = frames[k]
            rgb = dec_rgb((d / "color" / f"{i}.jpg").read_bytes())
            dep = dec_any((d / "depth" / f"{i}.png").read_bytes()).astype(np.float32) / 1000.0
            R = T[:3, :3]
            U, _, Vt = np.linalg.svd(R)                # poses stored as float32: re-orthonormalise
            T[:3, :3] = U @ Vt
            q.add(i, rgb, dep, K, np.linalg.inv(T), t=int(i) / 30.0)
    return sw.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?")
    ap.add_argument("out")
    ap.add_argument("--dir", help="convert one extracted scene folder (tests)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--splits", nargs="*", default=["scans_train"])
    ap.add_argument("--sizes", nargs="*", help="download queue files (part list and expected sizes)")
    ap.add_argument("--scratch", default="/data0/jz/scratch/scannet_stream")
    a = ap.parse_args()
    if a.dir:
        print(convert_dir(a.dir, Path(a.dir).name, Path(a.dir).parent.name, a.out, a.cap, a.every))
        return
    stream = PartStream(a.raw, PREFIX, DATASET, a.sizes)
    stream_convert(io.BufferedReader(stream, 8 << 20), make_scene_of(set(a.splits)), make_keep(a.every), convert_dir,
                   a.out, DATASET, Path(a.scratch), a.workers, extra_args=(a.cap, a.every))


if __name__ == "__main__":
    main()
