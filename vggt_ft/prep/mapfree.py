"""Map-free relocalization training scenes (CUT3R-processed re-upload HarrisonPENG/mapfree) -> scene cache.

Input: <raw>/sXXXXX.tar.gz, each `sXXXXX/dense_<k>/{rgb/frame_*.jpg, depth/frame_*.npy, cam/frame_*.safetensor,
sky_mask/...}` with `intrinsic` (3x3) and `pose` (4x4 camera-to-world) per frame; depth in metres from MVS (CUT3R zeroes
depth > 400 m and sky).  The two dense reconstructions of a scene are its two recordings of one place, registered in
one metric frame (Map-free's training scenes: reference + query sequence, scale from the phone's VIO) -> one scene with
two sequences (sessions).  Checked with `common verify` (cross column).

python -m vggt_ft.prep.mapfree <raw_root> <cache_root> [--every 2] [--workers 8]
"""
from __future__ import annotations

import argparse
import io
import tarfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from safetensors.numpy import load as st_load

from .common import SceneWriter


def convert(job):
    tgz, out_root, every = job
    scene = Path(tgz).name.split(".")[0]
    if (Path(out_root) / "mapfree" / scene / "scene.json").exists():
        return scene, -1
    files: dict[str, bytes] = {}
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf:
            if m.isfile():
                files[m.name] = tf.extractfile(m).read()
    seqs = sorted({n.split("/")[1] for n in files if n.count("/") >= 3})
    sw = SceneWriter(out_root, "mapfree", scene, world=scene, metric=True, synthetic=False, kind="outdoor")
    for sq in seqs:
        cams = sorted(n for n in files if n.startswith(f"{scene}/{sq}/cam/") and n.endswith(".safetensor"))
        q = sw.sequence(sq, session=f"{scene}_{sq}")
        for c in cams[::every]:
            stem = Path(c).name.rsplit(".", 1)[0]
            rgb_b = files.get(f"{scene}/{sq}/rgb/{stem}.jpg") or files.get(f"{scene}/{sq}/rgb/{stem}.png")
            dep_b = files.get(f"{scene}/{sq}/depth/{stem}.npy")
            if rgb_b is None or dep_b is None:
                continue
            cam = st_load(files[c])
            K = np.asarray(cam["intrinsic"], np.float64)
            T = np.asarray(cam["pose"], np.float64)
            rgb = cv2.imdecode(np.frombuffer(rgb_b, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
            dep = np.load(io.BytesIO(dep_b)).astype(np.float32)
            dep[(dep > 400) | ~np.isfinite(dep)] = 0
            sky = files.get(f"{scene}/{sq}/sky_mask/{stem}.png") or files.get(f"{scene}/{sq}/sky_mask/{stem}.jpg")
            if sky is not None:
                m = cv2.imdecode(np.frombuffer(sky, np.uint8), cv2.IMREAD_GRAYSCALE)
                if m is not None and m.shape == dep.shape:
                    dep[m >= 127] = 0
            q.add(stem, rgb, dep, K, np.linalg.inv(T))
    return scene, sw.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tgzs = sorted(str(p) for p in Path(a.raw).glob("s*.tar.gz") if Path(str(p) + ".done").exists())
    if a.limit:
        tgzs = tgzs[:a.limit]
    with Pool(a.workers) as p:
        for scene, n in p.imap_unordered(convert, [(t, a.out, a.every) for t in tgzs]):
            print(scene, n, flush=True)


if __name__ == "__main__":
    main()
