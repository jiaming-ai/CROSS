"""Spring (HarrisonPENG/spring on HF, preprocessed train split, one .tar.gz per scene) -> scene cache.

Input: <raw>/<id>.tar.gz holding <id>/rgb/NNNN.png (960 x 540, left camera), depth/NNNN.npy (float32 z-depth, metres,
from the GT disparity and the 6.5 cm stereo baseline; 0 = invalid), cam/NNNN.npz (intrinsics of the stored image,
pose = cam-to-world OpenCV 4x4); flow_forward/backward and the .safetensor duplicates are unused.
Conventions checked on scene 0027 (GT overlap of frames 1 / 2 / 3 apart, rel_tol 0.02): as stored 0.54 / 0.44 / 0.37,
the depth read as ray length 0.53 / 0.36 / 0.24, the pose read as world-to-cam 0.12 / 0.07 / 0.05 (low absolute
values: moving characters, fast camera).

Synthetic (Blender movie "Spring"), dynamic, metric (the dataset defines depth through a metric baseline; the scenes
look human-scale, e.g. the character ~1.6 m in front of the camera).  One scene per sequence.

python -m vggt_ft.prep.spring <raw_dir> <cache_root> [--every 1] [--workers 16] [--sizes <download queue files>]
"""
from __future__ import annotations

import argparse
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import SceneWriter
from .rawio import dec_npy, dec_npz, dec_rgb, iter_targz, load_sizes, ready, split

DATASET = "spring"


def convert(job):
    path, out, every, cap = job
    scene = Path(path).name.replace(".tar.gz", "")
    if (Path(out) / DATASET / scene / "scene.json").exists():
        return scene, -1
    frames: dict = {}
    for name, b in iter_targz(path):
        p = name.split("/")
        if len(p) == 3 and (p[1], p[2].rsplit(".", 1)[-1]) in (("rgb", "png"), ("depth", "npy"), ("cam", "npz")):
            frames.setdefault(p[2].rsplit(".", 1)[0], {})[p[1]] = b
    stems = sorted(s for s, f in frames.items() if len(f) == 3)[::every]
    if len(stems) < 2:
        return scene, 0
    sw = SceneWriter(out, DATASET, scene, metric=True, synthetic=True, dynamic=True, kind="movie")
    for c, idx in enumerate(split(len(stems), cap)):
        q = sw.sequence(scene if c == 0 else f"{scene}_{c}", session=scene)
        for i in idx:
            f = frames[stems[i]]
            m = dec_npz(f["cam"])
            q.add(stems[i], dec_rgb(f["rgb"]), dec_npy(f["depth"]), m["intrinsics"],
                  np.linalg.inv(m["pose"].astype(np.float64)))
    return scene, sw.close()


def _conv(job):
    try:
        return convert(job)
    except Exception as e:
        return Path(job[0]).name, f"ERROR {e!r} {traceback.format_exc(limit=3)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sizes", nargs="*", help="download queue files with the expected file sizes")
    a = ap.parse_args()
    sizes = load_sizes(a.sizes)
    paths = sorted(p for p in Path(a.raw).glob("*.tar.gz") if ready(p, sizes, DATASET)
                   and not (Path(a.out) / DATASET / p.name.replace(".tar.gz", "") / "scene.json").exists())
    print(f"{len(paths)} archives to convert", flush=True)
    with Pool(a.workers) as p:
        for scene, n in p.imap_unordered(_conv, [(str(x), a.out, a.every, a.cap) for x in paths]):
            print(scene, n, flush=True)


if __name__ == "__main__":
    main()
