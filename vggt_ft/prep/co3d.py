"""Co3Dv2 (yslan/3dscene-co3d on HF: DUSt3R-preprocessed, one uncompressed tar per category) -> scene cache.

Input: <raw>/<category>.tar holding <cat>/<seq>/images/frameNNNNNN.jpg (~384 x 513), frameNNNNNN.safetensor
(camera_intrinsics of that image, camera_pose = cam-to-world OpenCV, maximum_depth) and
<cat>/<seq>/depths/frameNNNNNN.jpg.geometric.png (uint16: depth = png / 65535 * maximum_depth, z-depth, 0 = invalid).
Conventions checked on two tv sequences (GT overlap of frame pairs 2 apart, rel_tol 0.02): cam-to-world + z-depth
0.74 / 0.82, the depth read as ray length 0.70 / 0.79 (worse at every gap), the pose read as world-to-cam 0.27 / 0.48.
The repository's selected_seqs_{train,test}.json list the same sequences (different frame subsets), so every sequence
of the tars is converted.

One scene per sequence (one object; Co3D's per-sequence reconstructions are in arbitrary units: non-metric, stored with
a median depth of ~2).  The ~200 frames of a sequence are dense (overlap ~0.9 between neighbours): every --every-th
frame is kept.

python -m vggt_ft.prep.co3d <raw_dir> <cache_root> [--every 2] [--workers 16] [--sizes <download queue files>]
"""
from __future__ import annotations

import argparse
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import SceneWriter, units_for
from .rawio import dec_any, dec_rgb, dec_safetensors, load_sizes, read_at, ready, split, tar_index

DATASET = "co3d"


def sequences(path) -> list:
    """[(category, sequence, [(frame, img (off, n), cam (off, n), depth (off, n))])] of one category tar."""
    idx = tar_index(path)
    seqs: dict = {}
    for name, loc in idx.items():
        p = name.split("/")
        if len(p) == 4 and p[2] == "images" and p[3].endswith(".jpg"):
            fr = p[3][:-4]
            cam = idx.get(f"{p[0]}/{p[1]}/images/{fr}.safetensor")
            dep = idx.get(f"{p[0]}/{p[1]}/depths/{fr}.jpg.geometric.png")
            if cam and dep:
                seqs.setdefault((p[0], p[1]), []).append((fr, loc, cam, dep))
    return [(c, s, sorted(v)) for (c, s), v in sorted(seqs.items())]


def convert(job):
    path, cat, seq, frames, out, every, cap = job
    scene = f"{cat}_{seq}"
    if (Path(out) / DATASET / scene / "scene.json").exists():
        return scene, -1
    frames = frames[::every]
    if len(frames) < 2:
        return scene, 0
    with open(path, "rb") as f:
        def depth(fr):
            m = dec_safetensors(read_at(f, *fr[2]))
            d = dec_any(read_at(f, *fr[3])).astype(np.float32) / 65535.0 * float(m["maximum_depth"])
            return m, d

        probe = [depth(frames[i])[1] for i in np.linspace(0, len(frames) - 1, min(20, len(frames))).astype(int)]
        sw = SceneWriter(out, DATASET, scene, metric=False, kind="object", units=units_for(probe),
                         extra={"category": cat})
        for c, idx in enumerate(split(len(frames), cap)):
            q = sw.sequence(seq if c == 0 else f"{seq}_{c}", session=seq)
            for i in idx:
                fr = frames[i]
                m, d = depth(fr)
                rgb = dec_rgb(read_at(f, *fr[1]))
                E = np.linalg.inv(m["camera_pose"].astype(np.float64))
                q.add(fr[0], rgb, d, m["camera_intrinsics"].astype(np.float64), E)
    return scene, sw.close()


def safe(fn, job):
    try:
        return fn(job)
    except Exception as e:
        return f"{job[0]} {job[1] if len(job) > 1 else ''}", f"ERROR {e!r} {traceback.format_exc(limit=3)}"


def _seqs(path):
    return safe(lambda p: (p, sequences(p)), path)


def _conv(job):
    return safe(convert, job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sizes", nargs="*", help="download queue files with the expected file sizes")
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--max-seqs", type=int, default=0, help="per category (tests)")
    a = ap.parse_args()
    sizes = load_sizes(a.sizes)
    tars = sorted(p for p in Path(a.raw).glob("*.tar") if ready(p, sizes, DATASET)
                  and (not a.categories or p.stem in a.categories))
    print(f"{len(tars)} category tars", flush=True)
    with Pool(a.workers) as p:
        jobs = []
        for path, seqs in p.imap_unordered(_seqs, [str(t) for t in tars]):
            if isinstance(seqs, str):
                print(path, seqs, flush=True)
                continue
            seqs = seqs[:a.max_seqs] if a.max_seqs else seqs
            jobs += [(path, c, s, fr, a.out, a.every, a.cap) for c, s, fr in seqs
                     if not (Path(a.out) / DATASET / f"{c}_{s}" / "scene.json").exists()]
        print(f"{len(jobs)} sequences to convert", flush=True)
        for scene, n in p.imap_unordered(_conv, jobs):
            print(scene, n, flush=True)


if __name__ == "__main__":
    main()
