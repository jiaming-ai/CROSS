"""BlendedMVS (HarrisonPENG/blendedmvs on HF, DUSt3R-preprocessed) -> scene cache.

Input: <raw>/<scene id>.tar.gz holding <id>/<frame>.jpg (512 x 384), <frame>.exr (z-depth, float32, 0 = invalid),
<frame>.npz (R_cam2world, t_cam2world, intrinsics of the 512 x 384 image); the .safetensor files duplicate the npz.
Conventions checked on scene 000000000000000000000009 (GT overlap of nearest-view pairs, rel_tol 0.02): pose
cam-to-world + z-depth 0.38, the depth read as ray length 0.14, the pose read as world-to-cam 0.13.

The views of a scene are an unordered collection: they are ordered by a greedy nearest-neighbour tour over camera
centre + viewing direction (rawio.nn_tour) so that frames with nearby indices overlap; tours longer than --cap frames
are cut into consecutive chunks (sequences of one scene, sharing its world frame).  Non-metric (MVS reconstructions in
arbitrary units): stored rescaled to a median depth of ~2 (units_for).

python -m vggt_ft.prep.blendedmvs <raw_dir> <cache_root> [--workers 16] [--cap 600] [--sizes <download queue files>]
"""
from __future__ import annotations

import argparse
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import SceneWriter, units_for
from .rawio import dec_any, dec_npz, dec_rgb, iter_targz, load_sizes, nn_tour, ready, split

DATASET = "blendedmvs"


def convert(job):
    path, out, cap = job
    scene = Path(path).name.replace(".tar.gz", "")
    if (Path(out) / DATASET / scene / "scene.json").exists():
        return scene, -1
    files: dict[str, dict] = {}
    for name, b in iter_targz(path):
        stem, ext = name.rsplit(".", 1)
        if ext in ("jpg", "exr", "npz"):
            files.setdefault(stem, {})[ext] = b
    stems = sorted(s for s, f in files.items() if len(f) == 3)
    if len(stems) < 2:
        return scene, 0
    K, E = [], []
    for s in stems:
        z = dec_npz(files[s]["npz"])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = z["R_cam2world"], z["t_cam2world"]
        K.append(z["intrinsics"].astype(np.float64))
        E.append(np.linalg.inv(T))
    E = np.array(E)
    probe = [dec_any(files[stems[i]]["exr"]) for i in np.linspace(0, len(stems) - 1, min(40, len(stems))).astype(int)]
    dmed = float(np.median(np.concatenate([d[d > 0].ravel()[::53] for d in probe])))
    order = nn_tour(E, dmed)
    sw = SceneWriter(out, DATASET, scene, metric=False, kind="mixed", units=units_for(probe),
                     extra={"order": "nn_tour"})
    for c, idx in enumerate(split(len(order), cap)):
        q = sw.sequence(f"c{c:02d}")
        for i in order[idx]:
            f = files[stems[i]]
            q.add(Path(stems[i]).name, dec_rgb(f["jpg"]), dec_any(f["exr"]), K[i], E[i])
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
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--sizes", nargs="*", help="download queue files with the expected file sizes")
    ap.add_argument("--limit", type=int, default=0, help="first N archives only (tests)")
    a = ap.parse_args()
    sizes = load_sizes(a.sizes)
    paths = sorted(p for p in Path(a.raw).glob("*.tar.gz") if ready(p, sizes, DATASET))
    paths = [p for p in paths if not (Path(a.out) / DATASET / p.name.replace(".tar.gz", "") / "scene.json").exists()]
    if a.limit:
        paths = paths[:a.limit]
    print(f"{len(paths)} archives to convert", flush=True)
    with Pool(a.workers) as p:
        for scene, n in p.imap_unordered(safe, [(str(x), a.out, a.cap) for x in paths]):
            print(scene, n, flush=True)


if __name__ == "__main__":
    main()
