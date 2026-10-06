"""UnrealStereo4K (HarrisonPENG/unreal4k on HF, preprocessed, one .tar.gz per scene) -> scene cache.

Input: <raw>/<id>.tar.gz holding <id>/<cam>/<frame>_rgb.png (682 x 384), <frame>_depth.npy (float32 z-depth, metres;
sky ~1e4, dropped by the depth codec's 1000 limit), <frame>.npz (intrinsics of the stored image, cam2world 4x4
OpenCV axes, but det -1: Unreal's world frame is left-handed, so the world y axis is mirrored here); cam 0 / 1 =
left / right camera of the stereo pairs.
Conventions checked on scene 00008 (GT overlap of consecutive frames, rel_tol 0.02): as stored 0.25, the depth read as
ray length 0.03, the pose read as world-to-cam 0.01.  Metric: the camera stands 1.3 - 2.6 m (quartiles) above the
surface below it; translations span tens of metres.

The frames are NOT a video (median 2.8 m / 30 degrees between consecutive frames): the views of both cameras of a
scene are ordered by a greedy nearest-neighbour tour (rawio.nn_tour) so that nearby indices overlap, and tours longer
than --cap frames are cut into consecutive chunks (sequences of one scene, one world).  Synthetic, static.

python -m vggt_ft.prep.unreal4k <raw_dir> <cache_root> [--workers 9] [--cap 600] [--sizes <download queue files>]
"""
from __future__ import annotations

import argparse
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import SceneWriter
from .rawio import dec_npy, dec_npz, dec_rgb, iter_targz, load_sizes, nn_tour, ready, split

DATASET = "unreal4k"


def convert(job):
    path, out, cap = job
    scene = Path(path).name.replace(".tar.gz", "")
    if (Path(out) / DATASET / scene / "scene.json").exists():
        return scene, -1
    frames: dict = {}
    for name, b in iter_targz(path):
        p = name.split("/")              # <id>/<cam>/<frame>{.npz,_depth.npy,_rgb.png}
        if len(p) != 3:
            continue
        for suf, k in ((".npz", "cam"), ("_depth.npy", "depth"), ("_rgb.png", "rgb")):
            if p[2].endswith(suf):
                frames.setdefault(f"{p[1]}_{p[2][:-len(suf)]}", {})[k] = b
    names = sorted(s for s, f in frames.items() if len(f) == 3)
    if len(names) < 2:
        return scene, 0
    cams = [dec_npz(frames[n]["cam"]) for n in names]
    E = np.array([np.linalg.inv(c["cam2world"].astype(np.float64)) for c in cams])
    det = np.linalg.det(E[:, :3, :3])
    if (det < 0).any():
        # Unreal's world frame is left-handed (det -1; the camera axes are OpenCV and the relative poses proper
        # rotations): mirror the world y axis, which leaves every relative pose unchanged.
        if not (det < 0).all():
            raise ValueError(f"{scene}: mixed pose handedness")
        E = E @ np.diag([1.0, -1.0, 1.0, 1.0])
    probe = [dec_npy(frames[names[i]]["depth"]) for i in np.linspace(0, len(names) - 1, 20).astype(int)]
    dmed = float(np.median(np.concatenate([d[(d > 0) & (d < 1000)].ravel()[::97] for d in probe])))
    order = nn_tour(E, dmed)
    sw = SceneWriter(out, DATASET, scene, metric=True, synthetic=True, kind="mixed", extra={"order": "nn_tour"})
    for c, idx in enumerate(split(len(order), cap)):
        q = sw.sequence(f"c{c:02d}")
        for i in order[idx]:
            f = frames[names[i]]
            q.add(names[i], dec_rgb(f["rgb"]), dec_npy(f["depth"]), cams[i]["intrinsics"], E[i])
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
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--sizes", nargs="*", help="download queue files with the expected file sizes")
    a = ap.parse_args()
    sizes = load_sizes(a.sizes)
    paths = sorted(p for p in Path(a.raw).glob("*.tar.gz") if ready(p, sizes, DATASET)
                   and not (Path(a.out) / DATASET / p.name.replace(".tar.gz", "") / "scene.json").exists())
    print(f"{len(paths)} archives to convert", flush=True)
    with Pool(a.workers) as p:
        for scene, n in p.imap_unordered(_conv, [(str(x), a.out, a.cap) for x in paths]):
            print(scene, n, flush=True)


if __name__ == "__main__":
    main()
