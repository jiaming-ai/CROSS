"""ScanNet++ (HarrisonPENG/scannetpp on HF, CUT3R-processed) -> scene cache.

Input: <raw>/<scene id>.tar.gz holding <id>/images/frame_NNNNNN.jpg (iPhone video frames, every 10th frame of the 60 Hz
video, 920 x 690), <id>/depth/frame_NNNNNN.png (uint16 millimetres, z-depth, 0 = invalid), <id>/refined_ins_ids/
(instance masks, unused) and <id>/scene_iphone_metadata.npz (`images` (N,) names, `intrinsics` (N,3,3) for the
920 x 690 image, `trajectories` (N,4,4) camera-to-world, OpenCV axes, metres).  The re-upload has the iPhone sequence
only (no DSLR).
Conventions checked on 0caa1ae59a (GT overlap of frame pairs 1 / 2 / 3 apart, rel_tol 0.02): as stored
0.74 / 0.54 / 0.46, the depth read as ray length 0.24 / 0.11 / 0.08, the pose read as world-to-cam 0.06 / 0.02 / 0.01.
The depth is the CUT3R / MASt3R render of the mesh (97 % valid at 920 x 690), not the 256 x 192 iPhone LiDAR.

One scene per ScanNet++ scene, one sequence `iphone` (cut into consecutive chunks of <= --cap frames, sessions of one
recording).  Every stored frame is kept by default (already ~6 Hz).

python -m vggt_ft.prep.scannetpp <raw_dir> <cache_root> [--workers 16] [--every 1] [--cap 600] [--sizes q.txt ...]
"""
from __future__ import annotations

import argparse
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import SceneWriter
from .rawio import dec_any, dec_npz, dec_rgb, iter_targz, load_sizes, ready, split

DATASET = "scannetpp"


def convert(job):
    path, out, every, cap = job
    scene = Path(path).name.replace(".tar.gz", "")
    if (Path(out) / DATASET / scene / "scene.json").exists():
        return scene, -1
    imgs, deps, meta = {}, {}, None
    for name, b in iter_targz(path):
        p = name.split("/")
        if len(p) == 3 and p[1] == "images" and p[2].endswith(".jpg"):
            imgs[p[2][:-4]] = b
        elif len(p) == 3 and p[1] == "depth" and p[2].endswith(".png"):
            deps[p[2][:-4]] = b
        elif len(p) == 2 and p[1] == "scene_iphone_metadata.npz":
            meta = dec_npz(b)
    if meta is None:
        return scene, "ERROR no scene_iphone_metadata.npz"
    stems = [str(n).rsplit(".", 1)[0] for n in meta["images"]]
    keep = [i for i, s in enumerate(stems) if s in imgs and s in deps and np.isfinite(meta["trajectories"][i]).all()]
    keep = keep[::every]
    if len(keep) < 2:
        return scene, 0
    sw = SceneWriter(out, DATASET, scene, world=scene, metric=True, synthetic=False, dynamic=False, kind="indoor",
                     extra={"camera": "iphone", "depth": "mesh_render"})
    for c, idx in enumerate(split(len(keep), cap)):
        q = sw.sequence("iphone" if c == 0 else f"iphone_{c}", session="iphone")
        for k in idx:
            i = keep[k]
            s = stems[i]
            T = np.asarray(meta["trajectories"][i], np.float64)
            d = dec_any(deps[s]).astype(np.float32) / 1000.0
            q.add(s, dec_rgb(imgs[s]), d, meta["intrinsics"][i], np.linalg.inv(T), t=int(s.split("_")[-1]) / 60.0)
    return scene, sw.close()


def safe(job):
    try:
        return convert(job)
    except Exception as e:
        return Path(job[0]).name, f"ERROR {e!r} {traceback.format_exc(limit=3)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--sizes", nargs="*", help="download queue files with the expected file sizes")
    ap.add_argument("--scenes", nargs="*")
    a = ap.parse_args()
    sizes = load_sizes(a.sizes)
    paths = sorted(p for p in Path(a.raw).glob("*.tar.gz") if ready(p, sizes, DATASET))
    if a.scenes:
        paths = [p for p in paths if p.name.replace(".tar.gz", "") in a.scenes]
    paths = [p for p in paths if not (Path(a.out) / DATASET / p.name.replace(".tar.gz", "") / "scene.json").exists()]
    print(f"{len(paths)} archives to convert", flush=True)
    if not paths:
        return
    with Pool(min(a.workers, len(paths))) as p:
        for scene, n in p.imap_unordered(safe, [(str(x), a.out, a.every, a.cap) for x in paths]):
            print(scene, n, flush=True)


if __name__ == "__main__":
    main()
