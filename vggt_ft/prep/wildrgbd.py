"""WildRGB-D (HarrisonPENG/wildrgbd on HF, preprocessed, one .tar.gz per category) -> scene cache.

Input: <raw>/<category>.tar.gz holding <cat>/scenes/scene_NNN/{rgb/NNNNN.jpg (385 x 512 portrait),
depth/NNNNN.png (uint16 millimetres, z-depth, 0 = invalid), masks/ (object masks, unused),
metadata/NNNNN.npz (camera_intrinsics of the stored image, camera_pose = cam-to-world OpenCV, metres; the first frame
is the identity)}.  ~100 frames per scene (every ~4th frame of the 30 Hz iPhone video).
Conventions checked on tooth_brush scene_022 / scene_258 (GT overlap of consecutive frames, rel_tol 0.02): as stored
0.80 / 0.75, the depth read as ray length 0.75 / 0.73 (0.55 / 0.50 at 3 frames apart vs 0.74 / 0.66), the pose read
as world-to-cam 0.57 / 0.56.  Metric: iPhone LiDAR depth (mm) and poses in metres agree at 2 % tolerance.

One scene per (category, scene) (a static object on a table, hand-held orbit).

python -m vggt_ft.prep.wildrgbd <raw_dir> <cache_root> [--workers 16] [--sizes <download queue files>]
"""
from __future__ import annotations

import argparse
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import SceneWriter
from .rawio import dec_any, dec_npz, dec_rgb, iter_targz, load_sizes, ready, split

DATASET = "wildrgbd"


def convert(job):
    path, out, every, cap = job
    scenes: dict = {}
    for name, b in iter_targz(path):
        p = name.split("/")            # <cat>/scenes/scene_NNN/<kind>/<file>
        if len(p) == 5 and p[1] == "scenes" and p[3] in ("rgb", "depth", "metadata"):
            stem, ext = p[4].rsplit(".", 1)
            if (p[3], ext) in (("rgb", "jpg"), ("depth", "png"), ("metadata", "npz")):
                scenes.setdefault((p[0], p[2]), {}).setdefault(stem, {})[p[3]] = b
    res = []
    for (cat, sc), frames in sorted(scenes.items()):
        scene = f"{cat}_{sc}"
        if (Path(out) / DATASET / scene / "scene.json").exists():
            res.append((scene, -1))
            continue
        stems = sorted(s for s, f in frames.items() if len(f) == 3)[::every]
        if len(stems) < 2:
            continue
        sw = SceneWriter(out, DATASET, scene, metric=True, kind="object", extra={"category": cat})
        for c, idx in enumerate(split(len(stems), cap)):
            q = sw.sequence(sc if c == 0 else f"{sc}_{c}", session=sc)
            for i in idx:
                f = frames[stems[i]]
                m = dec_npz(f["metadata"])
                d = dec_any(f["depth"]).astype(np.float32) / 1000.0
                q.add(stems[i], dec_rgb(f["rgb"]), d, m["camera_intrinsics"], np.linalg.inv(m["camera_pose"]))
        res.append((scene, sw.close()))
    return Path(path).name, res


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
    done = Path(a.out) / DATASET / "_archives_done.txt"
    skip = set(done.read_text().split()) if done.exists() else set()
    paths = sorted(p for p in Path(a.raw).glob("*.tar.gz") if ready(p, sizes, DATASET) and p.name not in skip)
    print(f"{len(paths)} archives to convert", flush=True)
    with Pool(a.workers) as p:
        for name, res in p.imap_unordered(_conv, [(str(x), a.out, a.every, a.cap) for x in paths]):
            if isinstance(res, str):
                print(name, res, flush=True)
                continue
            print(name, len(res), "scenes", sum(n for _, n in res if n > 0), "frames", flush=True)
            done.parent.mkdir(parents=True, exist_ok=True)
            with open(done, "a") as f:
                f.write(name + "\n")


if __name__ == "__main__":
    main()
