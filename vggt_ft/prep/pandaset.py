"""PandaSet (georghess/pandaset on HF, pandaset.zip) -> scene cache.

Input: <raw>/pandaset.zip with <seq>/camera/front_camera/{NN.jpg, intrinsics.json, poses.json, timestamps.json} and
<seq>/lidar/NN.pkl.gz (pandas DataFrame, columns x y z i t d: points of both LiDARs in world coordinates).
Conventions (PandaSet devkit, geometry.projection; checked with overlays and `common verify`): camera poses are
camera-to-world {"position": {x, y, z}, "heading": {w, x, y, z}} in OpenCV camera axes, metres.
Depth: the LiDAR sweep of the same frame projected into the front camera (z-depth, metres), z-buffered at the stored
resolution, with points much farther than the nearest LiDAR depth around them dropped (points seen through foreground
objects from the LiDAR's higher viewpoint).  Needs pandas (to read the pickles).

One output scene per sequence (front camera, 10 Hz, every frame).  Real, metric, dynamic.

python -m vggt_ft.prep.pandaset <raw_dir> <cache_root> [--workers 16]
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial.transform import Rotation

from .common import SceneWriter
from .hypersim import load_sizes, ready_files

CAM = "front_camera"
MAX_SIDE = 768


def pose_to_E(p) -> np.ndarray:
    h, t = p["heading"], p["position"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([h["x"], h["y"], h["z"], h["w"]]).as_matrix()
    T[:3, 3] = [t["x"], t["y"], t["z"]]
    return np.linalg.inv(T)


def lidar_depth(P: np.ndarray, K: np.ndarray, E: np.ndarray, H: int, W: int, win: int = 7,
                rel: float = 0.1) -> np.ndarray:
    """Sparse z-depth (HxW, 0 = none) from world points P (N,3): z-buffer, then drop points farther than
    (1 + rel) x the nearest depth within a win x win neighbourhood (occluded from the camera)."""
    X = P @ E[:3, :3].T + E[:3, 3]
    z = X[:, 2]
    ok = z > 0.5
    X, z = X[ok], z[ok]
    u = np.round(K[0, 0] * X[:, 0] / z + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * X[:, 1] / z + K[1, 2]).astype(np.int64)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok], v[ok], z[ok]
    d = np.full(H * W, np.inf, np.float32)
    np.minimum.at(d, v * W + u, z.astype(np.float32))
    d = d.reshape(H, W)
    near = minimum_filter(d, size=win, mode="constant", cval=np.inf)
    d[d > near * (1 + rel)] = np.inf
    d[~np.isfinite(d)] = 0
    return d


def convert_seq(job):
    zip_path, seq, out_root = job
    if (Path(out_root) / "pandaset" / seq / "scene.json").exists():
        return seq, -1
    import pandas as pd
    z = zipfile.ZipFile(zip_path)
    names = set(z.namelist())
    pre = next(p for p in (f"{seq}/", f"pandaset/{seq}/") if f"{p}camera/{CAM}/poses.json" in names)
    cpre = f"{pre}camera/{CAM}/"
    intr = json.loads(z.read(cpre + "intrinsics.json"))
    poses = json.loads(z.read(cpre + "poses.json"))
    stamps = json.loads(z.read(cpre + "timestamps.json"))
    K0 = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64)
    sw = SceneWriter(out_root, "pandaset", seq, world=seq, metric=True, synthetic=False, dynamic=True,
                     kind="driving", extra={"camera": CAM, "depth": "lidar"})
    q = sw.sequence(seq, session=seq)
    for i, pose in enumerate(poses):
        im, li = f"{cpre}{i:02d}.jpg", f"{pre}lidar/{i:02d}.pkl.gz"
        if im not in names or li not in names:
            continue
        rgb = cv2.imdecode(np.frombuffer(z.read(im), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
        H, W = rgb.shape[:2]
        s = min(1.0, MAX_SIDE / max(H, W))
        K = K0.copy()
        if s < 1:
            nw, nh = int(round(W * s)), int(round(H * s))
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            K[0, 0], K[1, 1] = K[0, 0] * nw / W, K[1, 1] * nh / H
            K[0, 2], K[1, 2] = (K[0, 2] + 0.5) * nw / W - 0.5, (K[1, 2] + 0.5) * nh / H - 0.5
            H, W = nh, nw
        df = pd.read_pickle(io.BytesIO(gzip.decompress(z.read(li))))
        P = df[["x", "y", "z"]].to_numpy(np.float64)
        E = pose_to_E(pose)
        q.add(f"{i:02d}", np.ascontiguousarray(rgb), lidar_depth(P, K, E, H, W), K, E,
              t=float(stamps[i]) - float(stamps[0]))
    return seq, sw.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sizes", help="download queue file with expected sizes (5th column)")
    ap.add_argument("--seqs", nargs="*")
    a = ap.parse_args()
    files = ready_files(Path(a.raw), "pandaset.zip", load_sizes(a.sizes))
    if not files:
        print("waiting for pandaset.zip; 0 to convert", flush=True)
        return
    names = zipfile.ZipFile(files[0]).namelist()
    seqs = sorted({n.split("/camera/")[0].split("/")[-1] for n in names if f"/camera/{CAM}/poses.json" in n})
    if a.seqs:
        seqs = [s for s in seqs if s in a.seqs]
    jobs = [(str(files[0]), s, a.out) for s in seqs
            if not (Path(a.out) / "pandaset" / s / "scene.json").exists()]
    print(f"{len(seqs)} sequences, {len(jobs)} to convert", flush=True)
    if not jobs:
        return
    with Pool(min(a.workers, len(jobs))) as p:
        for seq, n in p.imap_unordered(convert_seq, jobs):
            print(seq, n, flush=True)


if __name__ == "__main__":
    main()
