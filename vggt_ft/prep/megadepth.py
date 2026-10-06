"""MegaDepth (yslan/3dscene_megadepth on HF: DUSt3R-preprocessed) -> scene cache.

Input: <raw>/NNNN.tar (uncompressed, read in place through a member index), members
NNNN/<sub>/<photo>.jpg.{jpg,exr,safetensor}; the safetensor holds `intrinsics` (3x3, for the stored jpg) and
`cam2world` (4x4, OpenCV axes); the EXR holds the MVS depth (z-depth, 0 = invalid) in the units of the SfM model.
Conventions checked empirically (z-depth vs ray length, cam2world vs world2cam) with a tight-tolerance overlap test.

Non-metric: each subscene (one SfM / MVS model) is rescaled to a median depth of ~2 units (`units_for`).  One output
scene per (landmark, subscene), `world` = the same name: the subscenes of a landmark are separate reconstructions
with their own frames (checked: no cross-subscene overlap).  The photos are unordered, so the frames are ordered by a
greedy nearest-neighbour tour (camera centre + look-at point + viewing direction for the candidates, the GT
covisibility of low-resolution depth to pick among them) so that frames with nearby indices overlap.

python -m vggt_ft.prep.megadepth <raw_dir> <cache_root> [--workers 16] [--sizes q.txt] [--max-frames 0]
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import tarfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .common import SceneWriter, units_for
from .hypersim import load_sizes, ready_files

LOW = 32     # downsampling of the depth used for ordering


def tar_index(path) -> dict:
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


def read_cam(b: bytes):
    from safetensors.numpy import load
    z = load(b)
    return np.asarray(z["intrinsics"], np.float64), np.asarray(z["cam2world"], np.float64)


def read_exr(b: bytes) -> np.ndarray:
    d = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise ValueError("cannot decode exr")
    if d.ndim == 3:
        d = d[..., 0]
    d = d.astype(np.float32)
    d[~np.isfinite(d)] = 0
    return d


def proper_E(c2w: np.ndarray) -> np.ndarray:
    T = c2w.copy()
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = U @ Vt
    return np.linalg.inv(T)


def small(dep, K):
    """Low-resolution depth (nearest) and its intrinsics, for the ordering."""
    H, W = dep.shape
    h, w = max(8, H // LOW * 2), max(8, W // LOW * 2)
    d = cv2.resize(dep, (w, h), interpolation=cv2.INTER_NEAREST)
    K2 = K.copy()
    K2[0] *= w / W
    K2[1] *= h / H
    K2[0, 2] = (K[0, 2] + 0.5) * w / W - 0.5
    K2[1, 2] = (K[1, 2] + 0.5) * h / H - 0.5
    return d, K2


def pair_covis(deps, Ks, Es, i, js):
    """GT covisibility of frame i with each frame in js (low-res depth; frames of different sizes: per pair)."""
    import torch
    from vggt_ft.geometry import gt_covisibility
    out = np.zeros(len(js))
    for n, j in enumerate(js):
        a, b = deps[i], deps[j]
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
            Kb = Ks[j].copy()
            Kb[0] *= a.shape[1] / deps[j].shape[1]
            Kb[1] *= a.shape[0] / deps[j].shape[0]
        else:
            Kb = Ks[j]
        E = np.stack([Es[i], Es[j]])
        c0 = -E[0, :3, :3].T @ E[0, :3, 3]
        E[:, :3, 3] = E[:, :3, 3] + np.einsum("nij,j->ni", E[:, :3, :3], c0)
        dep = torch.from_numpy(np.stack([a, b]))[None]
        K = torch.from_numpy(np.stack([Ks[i], Kb])).float()[None]
        out[n] = float(gt_covisibility(dep, dep > 0, torch.from_numpy(E)[None], K, grid=16, rel_tol=0.05)[0, 0, 1])
    return out


def tour(Es, dmed, deps, Ks, k: int = 12) -> np.ndarray:
    """Greedy nearest-neighbour tour.  Candidates: the k unvisited frames closest in (centre, look-at point,
    direction); among them the one with the highest GT covisibility (geometric nearest if none overlaps)."""
    N = len(Es)
    R = Es[:, :3, :3]
    C = -np.einsum("nji,nj->ni", R, Es[:, :3, 3])
    Z = R[:, 2, :]                                   # viewing directions in the world
    s = float(np.median(dmed[dmed > 0])) if (dmed > 0).any() else 1.0
    d = np.where(dmed > 0, dmed, s)
    P = C + d[:, None] * Z                           # look-at points
    F = np.concatenate([C / s, P / s, Z * 2.0], 1)   # 1 unit ~ one median depth / ~30 deg
    start = int(np.argmax(np.linalg.norm(F - F.mean(0), axis=1)))
    order, left = [start], np.ones(N, bool)
    left[start] = False
    cur = start
    for _ in range(N - 1):
        idx = np.flatnonzero(left)
        dist = np.linalg.norm(F[idx] - F[cur], axis=1)
        cand = idx[np.argsort(dist)[:k]]
        cov = pair_covis(deps, Ks, Es, cur, cand)
        nxt = int(cand[np.argmax(cov)]) if cov.max() > 0.05 else int(cand[0])
        order.append(nxt)
        left[nxt] = False
        cur = nxt
    return np.array(order)


def convert_sub(job):
    import torch
    torch.set_num_threads(1)
    tar_path, land, sub, out_root, max_frames = job
    scene = f"{land}_{sub}"
    if (Path(out_root) / "megadepth" / scene / "scene.json").exists():
        return scene, -1
    idx = tar_index(tar_path)
    pre = f"{land}/{sub}/"
    stems = sorted(n[len(pre):-len(".jpg")] for n in idx if n.startswith(pre) and n.endswith(".jpg.jpg"))
    stems = [s for s in stems if f"{pre}{s}.exr" in idx and f"{pre}{s}.safetensor" in idx]
    f = open(tar_path, "rb")
    Ks, Es, dmed, deps, Kl, keep, samples = [], [], [], [], [], [], []
    for s in stems:
        K, c2w = read_cam(tar_read(f, idx, f"{pre}{s}.safetensor"))
        if not np.isfinite(c2w).all():
            continue
        dep = read_exr(tar_read(f, idx, f"{pre}{s}.exr"))
        v = dep[dep > 0]
        if v.size < 0.01 * dep.size:
            continue
        E = proper_E(c2w)
        ds, ks = small(dep, K)
        Ks.append(K), Es.append(E), dmed.append(float(np.median(v))), deps.append(ds), Kl.append(ks), keep.append(s)
        samples.append(v[::97])
    if len(keep) < 2:
        return scene, 0
    Es, dmed = np.array(Es), np.array(dmed)
    if max_frames and len(keep) > max_frames:
        sel = np.sort(np.random.default_rng(0).choice(len(keep), max_frames, replace=False))
        keep, Ks, Es, dmed = [keep[i] for i in sel], [Ks[i] for i in sel], Es[sel], dmed[sel]
        deps, Kl = [deps[i] for i in sel], [Kl[i] for i in sel]
    order = tour(Es, dmed, deps, Kl)
    units = units_for(samples)
    sw = SceneWriter(out_root, "megadepth", scene, world=scene, metric=False, synthetic=False, dynamic=False,
                     kind="outdoor", units=units, extra={"landmark": land, "subscene": sub, "order": "nn_tour"})
    q = sw.sequence(sub, session=f"{land}_{sub}")
    for i in order:
        s = keep[i]
        rgb = cv2.imdecode(np.frombuffer(tar_read(f, idx, f"{pre}{s}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        dep = read_exr(tar_read(f, idx, f"{pre}{s}.exr"))
        if rgb is None:
            continue
        if dep.shape != rgb.shape[:2]:
            raise ValueError(f"{scene} {s}: depth {dep.shape} vs image {rgb.shape}")
        q.add(Path(s).stem, rgb[..., ::-1], dep, Ks[i], Es[i])
    return scene, sw.close()


def subscenes(tar_path):
    subs = set()
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            p = m.name.split("/")
            if len(p) == 3 and m.isfile():
                subs.add((p[0], p[1]))
    return sorted(subs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sizes", help="download queue file with expected sizes (5th column)")
    ap.add_argument("--max-frames", type=int, default=0, help="random subset per subscene (0 = all)")
    ap.add_argument("--landmarks", nargs="*")
    a = ap.parse_args()
    files = ready_files(Path(a.raw), "[0-9]*.tar", load_sizes(a.sizes))
    if a.landmarks:
        files = [f for f in files if f.stem in a.landmarks]
    jobs = []
    for t in files:
        for land, sub in subscenes(t):
            if not (Path(a.out) / "megadepth" / f"{land}_{sub}" / "scene.json").exists():
                jobs.append((str(t), land, sub, a.out, a.max_frames))
    print(f"{len(files)} archives ready, {len(jobs)} to convert (subscenes)", flush=True)
    if not jobs:
        return
    with Pool(min(a.workers, len(jobs))) as p:
        for scene, n in p.imap_unordered(convert_sub, jobs):
            print(scene, n, flush=True)


if __name__ == "__main__":
    main()
