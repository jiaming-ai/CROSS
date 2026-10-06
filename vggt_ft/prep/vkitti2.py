"""Virtual KITTI 2 (nhatchung/Virtual_KITTI2 on HF, v2.0.3) -> scene cache.

Input: <raw>/vkitti_2.0.3_{rgb,depth}.tar (uncompressed; read in place through a member index) and
<raw>/vkitti_2.0.3_textgt.tar.gz.  Layout: Scene{01,02,06,18,20}/<variant>/frames/{rgb,depth}/Camera_{0,1}/
{rgb_%05d.jpg,depth_%05d.png}; <variant>/{intrinsic,extrinsic}.txt with one row per (frame, camera).
Conventions (checked with `common verify` and a tight-tolerance overlap test): extrinsic.txt rows are 4x4
camera-from-world in OpenCV axes (x right, y down, z forward); depth PNGs are uint16 z-depth in cm, 65535 = sky.
Images 1242 x 375.

One output scene per Scene0x, its variants (clone, fog, morning, overcast, rain, sunset, 15/30-deg-left/right) as
sessions in the scene's world frame (all variants of a scene share it).  Camera_0 only, every 2nd frame.  Scene20 is
held out (never converted).

python -m vggt_ft.prep.vkitti2 <raw_dir> <cache_root> [--every 2] [--workers 4]
"""
from __future__ import annotations

import argparse
import io
import tarfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .common import SceneWriter

HOLDOUT = {"Scene20"}
CAM = "Camera_0"


def tar_index(path) -> dict:
    """member name -> (data offset, size) of an uncompressed tar (headers only, seeks over the data)."""
    idx = {}
    with tarfile.open(path, "r:") as tf:
        for m in tf:
            if m.isfile():
                idx[m.name] = (m.offset_data, m.size)
    return idx


def tar_read(f, idx, name) -> bytes:
    off, n = idx[name]
    f.seek(off)
    return f.read(n)


def read_table(text: str) -> np.ndarray:
    return np.loadtxt(io.StringIO(text), skiprows=1)


def convert_scene(job):
    raw, out_root, scene, variants, every = job
    if (Path(out_root) / "vkitti2" / scene / "scene.json").exists():
        return scene, -1
    raw = Path(raw)
    ri, di = tar_index(raw / "vkitti_2.0.3_rgb.tar"), tar_index(raw / "vkitti_2.0.3_depth.tar")
    fr, fd = open(raw / "vkitti_2.0.3_rgb.tar", "rb"), open(raw / "vkitti_2.0.3_depth.tar", "rb")
    sw = SceneWriter(out_root, "vkitti2", scene, world=scene, metric=True, synthetic=True, dynamic=True,
                     kind="driving", extra={"camera": CAM, "every": every})
    for var, (intr, extr) in sorted(variants.items()):
        intr, extr = read_table(intr), read_table(extr)
        cam = int(CAM[-1])
        intr = {int(r[0]): r[2:] for r in intr if int(r[1]) == cam}
        extr = {int(r[0]): r[2:].reshape(4, 4) for r in extr if int(r[1]) == cam}
        q = sw.sequence(var, session=var)
        for f in sorted(intr)[::every]:
            rn = f"{scene}/{var}/frames/rgb/{CAM}/rgb_{f:05d}.jpg"
            dn = f"{scene}/{var}/frames/depth/{CAM}/depth_{f:05d}.png"
            if rn not in ri or dn not in di or f not in extr:
                continue
            rgb = cv2.imdecode(np.frombuffer(tar_read(fr, ri, rn), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
            draw = cv2.imdecode(np.frombuffer(tar_read(fd, di, dn), np.uint8), cv2.IMREAD_UNCHANGED)
            dep = np.where(draw == 65535, 0, draw.astype(np.float32) / 100.0)
            fx, fy, cx, cy = intr[f]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
            q.add(f"{f:05d}", rgb, dep, K, extr[f])
    return scene, sw.close()


def load_textgt(raw: Path) -> dict:
    """{scene: {variant: (intrinsic.txt, extrinsic.txt)}}"""
    out = {}
    with tarfile.open(raw / "vkitti_2.0.3_textgt.tar.gz", "r:gz") as tf:
        for m in tf:
            p = m.name.split("/")
            if len(p) == 3 and p[2] in ("intrinsic.txt", "extrinsic.txt"):
                out.setdefault(p[0], {}).setdefault(p[1], {})[p[2]] = tf.extractfile(m).read().decode()
    return {s: {v: (d["intrinsic.txt"], d["extrinsic.txt"]) for v, d in vs.items()} for s, vs in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scenes", nargs="*")
    a = ap.parse_args()
    raw = Path(a.raw)
    need = ["vkitti_2.0.3_rgb.tar", "vkitti_2.0.3_depth.tar", "vkitti_2.0.3_textgt.tar.gz"]
    if not all((raw / f"{n}.done").exists() for n in need):
        print("waiting for downloads", flush=True)
        return
    gt = load_textgt(raw)
    jobs = [(str(raw), a.out, s, v, a.every) for s, v in sorted(gt.items())
            if s not in HOLDOUT and (not a.scenes or s in a.scenes)]
    print(f"{len(jobs)} scenes", flush=True)
    with Pool(min(a.workers, len(jobs))) as p:
        for scene, n in p.imap_unordered(convert_scene, jobs):
            print(scene, n, flush=True)


if __name__ == "__main__":
    main()
