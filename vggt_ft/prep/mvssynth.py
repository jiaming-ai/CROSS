"""MVS-Synth (phuang17/MVS-Synth on HF, GTAV_540.tar.gz) -> scene cache.

Input: <raw>/GTAV_540.tar.gz with GTAV_540/<seq>/{images/NNNN.png, depths/NNNN.exr, poses/NNNN.json}; members are not
grouped by sequence, so the archive is extracted to --tmp first (deleted after a complete conversion).
poses/*.json: {"extrinsic": 4x4, "f_x", "f_y", "c_x", "c_y"} for the 960 x 540 image.
Conventions (checked empirically with a tight-tolerance overlap test, see FIX): `extrinsic` is camera-from-world, but
its rotation has det -1: the GTA world frame is left-handed.  The camera axes are OpenCV's (x right, y down, z
forward), so mirroring one WORLD axis (E @ diag(1, 1, -1, 1)) gives a proper rotation without changing any camera-frame
coordinate (images and depth stay consistent).  Depth: EXR z-depth (inf = sky -> invalid) in the units of the
translations, which are NOT metres (~0.1 m: a pedestrian is ~17 units tall, street-level cameras 15-45 units above the
road), so the data is stored as non-metric (`units_for`: median depth ~2).

One output scene per sequence (each is one place; their world frames are not related), every frame.

python -m vggt_ft.prep.mvssynth <raw_dir> <cache_root> --tmp /data0/jz/tmp/mvssynth [--workers 16]
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .common import SceneWriter, units_for
from .hypersim import load_sizes, ready_files

WORLD_FLIP = np.diag([1.0, 1.0, -1.0, 1.0])       # mirror the left-handed world (columns of E)


def read_exr(path) -> np.ndarray:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise ValueError(f"cannot read {path}")
    if d.ndim == 3:
        d = d[..., 0]
    d = d.astype(np.float32)
    d[~np.isfinite(d)] = 0
    return d


def read_pose(path):
    c = json.load(open(path))
    K = np.array([[c["f_x"], 0, c["c_x"]], [0, c["f_y"], c["c_y"]], [0, 0, 1]], np.float64)
    E = np.array(c["extrinsic"], np.float64)
    if np.linalg.det(E[:3, :3]) < 0:
        E = E @ WORLD_FLIP
    return K, E


def convert_seq(job):
    seq_dir, out_root, every = job
    seq_dir = Path(seq_dir)
    scene = seq_dir.name
    if (Path(out_root) / "mvssynth" / scene / "scene.json").exists():
        return scene, -1
    frames = [p for p in sorted((seq_dir / "images").glob("*.png"))[::every]
              if (seq_dir / "depths" / f"{p.stem}.exr").exists() and (seq_dir / "poses" / f"{p.stem}.json").exists()]
    if len(frames) < 2:
        return scene, 0
    units = units_for([read_exr(seq_dir / "depths" / f"{p.stem}.exr") for p in frames[::5]])
    sw = SceneWriter(out_root, "mvssynth", scene, world=scene, metric=False, synthetic=True, dynamic=True,
                     kind="outdoor", units=units, extra={"every": every, "unit_m": "~0.1 (estimated)"})
    q = sw.sequence(scene, session=scene)
    for p in frames:
        dp, cp = seq_dir / "depths" / f"{p.stem}.exr", seq_dir / "poses" / f"{p.stem}.json"
        K, E = read_pose(cp)
        rgb = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if rgb is None or not np.isfinite(E).all():
            continue
        q.add(p.stem, rgb[..., ::-1], read_exr(dp), K, E)
    return scene, sw.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--tmp", required=True, help="folder for the extracted archive (deleted when done)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--sizes", help="download queue file with expected sizes (5th column)")
    ap.add_argument("--keep-tmp", action="store_true")
    a = ap.parse_args()
    files = ready_files(Path(a.raw), "GTAV_540.tar.gz", load_sizes(a.sizes))
    if not files:
        print("waiting for GTAV_540.tar.gz", flush=True)
        return
    tmp = Path(a.tmp)
    if not (tmp / ".extracted").exists():
        tmp.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"pigz -dc -p 4 '{files[0]}' | tar x -C '{tmp}'", shell=True, check=True)
        (tmp / ".extracted").touch()
    seqs = sorted(p for p in (tmp / "GTAV_540").iterdir() if (p / "images").is_dir())
    jobs = [(str(s), a.out, a.every) for s in seqs]
    print(f"{len(jobs)} sequences", flush=True)
    done = 0
    with Pool(min(a.workers, len(jobs))) as p:
        for scene, n in p.imap_unordered(convert_seq, jobs):
            print(scene, n, flush=True)
            done += n != 0
    if done == len(jobs) and not a.keep_tmp:
        shutil.rmtree(tmp)
        print(f"removed {tmp}", flush=True)


if __name__ == "__main__":
    main()
