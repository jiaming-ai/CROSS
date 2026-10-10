"""World BEV labels of a scene from its accumulated ground truth.

All frames of all sequences of a place (scene cache, vggt_ft/dataio/scene.py; sequences of one `world` share a frame)
are accumulated, not only the past of a frame: the label of a frame then includes what other frames of the scene saw,
which the head should learn to infer (behind objects, outside the view).

    python -m cross_nav.labels --cache <root> --out <bev_root> --dataset scannet [--workers 32] [--limit N]

Steps per place:
1. gravity: mean camera up vector (handheld / robot cameras are held upright and look in all directions), refined by
   the mean normal of the near-horizontal surfaces (floor, table tops, ceiling);
2. voxels (5 cm x 5 cm x 2 cm) of all depth points within the dataset's reliable range, with the number of points whose
   normal points up (floor candidates);
3. ground: per cell the heights of the floor candidates; flood fill from the floor under each camera, stepping to a
   neighbour's candidate within `step` (slopes yes, stairs no) and across up to `gap` cells without a candidate (floor
   hidden under furniture);
4. classes per cell: occupied = enough points between h_lo and h_hi above the local ground (interpolated from the
   nearest ground cell where the floor itself was not seen), free = ground without such points, else unknown.

Output <bev_root>/<dataset>/<world>.npz: bev (H, W) uint8 (0 unknown, 1 free, 2 occupied), ground (H, W) float16 (NaN
off the ground), origin (2,), res, R_g (3, 3) world -> gravity frame (z up), up_check (deg between the camera-up and the
refined gravity), and per sequence `cam_h/<scene>/<seq>` the camera height above the ground (NaN = unknown).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from vggt_ft.dataio.depthcodec import decode
from vggt_ft.dataio.scene import list_scenes, load_scene

from .bev import FREE, OCC, UNKNOWN

try:
    from numba import njit
except ImportError:                     # pure-Python fallback (slow on large places)
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

cv2.setNumThreads(1)

# reliable depth range per dataset (m) and point subsampling (pixel step on the stored image)
DATASET_CFG = {
    "scannet": dict(max_range=5.0),           # Kinect-style structured light: noise ~ z^2
    "arkitscenes": dict(max_range=5.0),       # iPad LiDAR depth (upsampled)
    "scannetpp": dict(max_range=8.0),         # depth rendered from the laser-scan mesh
    "hypersim": dict(max_range=15.0),         # synthetic
    "tartanair2": dict(max_range=15.0, frame_step=2),
    "grandtour": dict(max_range=15.0),        # LiDAR
    "oxford_spires": dict(max_range=15.0),
    "simchange": dict(max_range=10.0),
    "openloris": dict(max_range=4.0),         # D435i
    "7scenes": dict(max_range=4.0),           # Kinect v1
    "hiroom": dict(max_range=10.0),           # synthetic
    "nrgbd": dict(max_range=8.0),             # synthetic
    "hssd_nav": dict(max_range=10.0),         # simulated (cross_nav/sim); its own labels come from the meshes
    "nclt": dict(max_range=20.0),             # Velodyne HDL-32E projected into the forward Ladybug3 camera
}
DEFAULT = dict(max_range=8.0, pix_step=4, frame_step=1, max_frames=4000, res=0.05, zres=0.02,
               normal_up_deg=25.0, step=0.05, gap=8, h_lo=0.10, h_hi=1.80, occ_min=4, free_max=1,
               seed_radius=2.5, cam_h_range=(0.08, 2.6), interp_max=2.0,
               carve_frames=1500, carve_pix=8, carve_bin=0.10, carve_ratio=0.2, carve_min_pass=5,
               sparse_normals=0.3)


# ---------------------------------------------------------------------------------------------------------- points
def frame_points(dep: np.ndarray, K: np.ndarray, E: np.ndarray, step: int, max_range: float):
    """World points (M,3) and unit normals (M,3) of a depth image on a pixel grid of `step`.  Normals from the
    neighbouring grid points (central differences), oriented towards the camera; NaN where a neighbour is missing or
    across a depth discontinuity."""
    H, W = dep.shape
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    d = dep[ys, xs].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d], -1)          # camera frame (h, w, 3)
    ok = (d > 0) & (d <= max_range)
    n = np.full_like(P, np.nan)
    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    nn_ = np.cross(dx, dy)
    nrm = np.linalg.norm(nn_, axis=-1, keepdims=True)
    dc = d[1:-1, 1:-1]
    smooth = np.ones_like(dc, bool)
    for a, b in ((d[1:-1, 2:], d[1:-1, :-2]), (d[2:, 1:-1], d[:-2, 1:-1])):
        smooth &= (a > 0) & (b > 0) & (np.abs(a - b) < 0.1 * dc * 2 + 0.02)
    nn_ = np.where(smooth[..., None] & (nrm > 1e-12), nn_ / np.maximum(nrm, 1e-12), np.nan)
    flip = np.sum(nn_ * P[1:-1, 1:-1], -1) > 0                              # face the camera
    nn_[flip] *= -1
    n[1:-1, 1:-1] = nn_
    R, t = E[:3, :3], E[:3, 3]
    Pw = (P[ok] - t) @ R                                                     # R^T (P - t)
    nw = n[ok] @ R
    return Pw, nw


def load_depth(sq, i: int):
    """(depth f32 HxW in metres, K 3x3, E 4x4) of frame i of a cache sequence without reading the image.  K is given
    for the image resolution; the depth is resized to it when stored smaller."""
    dep = cv2.imread(str(sq.dir / "depth" / f"{sq.names[i]}.png"), cv2.IMREAD_UNCHANGED)
    if dep is None:
        raise FileNotFoundError(f"{sq.dir} {sq.names[i]}")
    dep = decode(dep)
    if sq.scene.scale != 1.0:
        dep = dep * np.float32(sq.scene.scale)
    K = sq.K[i].astype(np.float64)
    W_img, H_img = 2 * (K[0, 2] + 0.5), 2 * (K[1, 2] + 0.5)          # approximate image size (centred principal point)
    if abs(dep.shape[1] - W_img) > 0.2 * W_img:                      # depth stored at another resolution
        s = dep.shape[1] / W_img
        K = K.copy()
        K[0, 0] *= s
        K[1, 1] *= s
        K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return dep, K, sq.E[i].astype(np.float64)


def camera_centres(E: np.ndarray) -> np.ndarray:
    return -np.einsum("nji,nj->ni", E[:, :3, :3], E[:, :3, 3])


def rotation_to_z(u: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector u to +z."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z)
    s, c = np.linalg.norm(v), float(u @ z)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def estimate_gravity(Es: np.ndarray, normals: np.ndarray, up_deg: float = 25.0):
    """Unit up vector in the world frame and the angle (deg) between the camera-up prior and the refined estimate."""
    ups = -Es[:, 1, :3]                                     # camera -y axis in world coordinates (rows of R_cw)
    u0 = ups.mean(0)
    u0 = u0 / np.linalg.norm(u0)
    n = normals[np.isfinite(normals).all(1)]
    u = u0.copy()
    for thr in (up_deg, 15.0, 8.0, 5.0):
        c = n @ u
        sel = np.abs(c) > math.cos(math.radians(thr))
        if sel.sum() < 100:
            break
        m = (n[sel] * np.sign(c[sel])[:, None]).sum(0)
        u = m / np.linalg.norm(m)
    return u, math.degrees(math.acos(float(np.clip(u @ u0, -1, 1))))


# ---------------------------------------------------------------------------------------------------------- ground
@njit(cache=True)
def _flood(cand, ncand, seed_iy, seed_ix, seed_h, step, gap):
    """Multi-source BFS over the cell grid.  cand (H,W,K) candidate floor heights (ncand valid per cell).  Returns the
    ground height per cell (NaN = not reached) and whether the cell's floor was seen (vs. carried across a gap)."""
    H, W, K = cand.shape
    g = np.full((H, W), np.nan, np.float32)
    seen = np.zeros((H, W), np.bool_)
    gused = np.zeros((H, W), np.int32)
    qn = H * W + len(seed_iy) + 1
    qy = np.empty(qn, np.int32)
    qx = np.empty(qn, np.int32)
    head, tail = 0, 0
    for s in range(len(seed_iy)):
        y, x = seed_iy[s], seed_ix[s]
        if np.isnan(g[y, x]):
            g[y, x] = seed_h[s]
            seen[y, x] = True
            qy[tail], qx[tail] = y, x
            tail += 1
    while head < tail:
        y, x = qy[head], qx[head]
        head += 1
        h, gu = g[y, x], gused[y, x]
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W or not np.isnan(g[ny, nx]):
                    continue
                # constant tolerance, also after hidden cells: the floor under furniture is at floor height (a
                # tolerance growing with the gap let the fill climb onto beds / sofas / car roofs next to a gap)
                tol = (step + 0.005 * gu) * (1.4142 if dy != 0 and dx != 0 else 1.0)
                best, bd = np.nan, 1e9
                for k in range(ncand[ny, nx]):
                    dd = abs(cand[ny, nx, k] - h)
                    if dd <= tol and dd < bd:
                        best, bd = cand[ny, nx, k], dd
                if not np.isnan(best):
                    g[ny, nx] = best
                    seen[ny, nx] = True
                    gused[ny, nx] = 0
                    qy[tail], qx[tail] = ny, nx
                    tail += 1
                elif gu < gap and ncand[ny, nx] == 0:
                    # nothing seen here (hidden floor): carry the height, but not into a cell with other surfaces
                    g[ny, nx] = h
                    gused[ny, nx] = gu + 1
                    qy[tail], qx[tail] = ny, nx
                    tail += 1
    return g, seen


@njit(cache=True)
def _carve(pts, cam_idx, cams, g_int, ox, oy, res, h_lo, nb, binh):
    """Height-resolved free-space carving.  For every ray (camera -> depth point), the (cell, height bin) voxels it
    crosses at band heights (h_lo .. h_lo + nb * binh above the local ground) count a pass, its end voxel a hit.
    Returns hit, pass (H, W, nb) uint16 (saturating)."""
    H, W = g_int.shape
    hit = np.zeros((H, W, nb), np.uint16)
    ps = np.zeros((H, W, nb), np.uint16)
    for k in range(pts.shape[0]):
        c = cams[cam_idx[k]]
        px, py, pz = pts[k, 0], pts[k, 1], pts[k, 2]
        ex, ey = int(np.floor((px - ox) / res)), int(np.floor((py - oy) / res))
        if 0 <= ex < W and 0 <= ey < H and not np.isnan(g_int[ey, ex]):
            b = int(np.floor((pz - g_int[ey, ex] - h_lo) / binh))
            if 0 <= b < nb and hit[ey, ex, b] < 65535:
                hit[ey, ex, b] += 1
        dx, dy, dz = px - c[0], py - c[1], pz - c[2]
        L = np.sqrt(dx * dx + dy * dy)
        n = int(L / (0.5 * res))
        last = -1
        for s in range(1, n):
            t = s / n
            if t * L > L - res:                      # stop one cell before the end point
                break
            ix = int(np.floor((c[0] + t * dx - ox) / res))
            iy = int(np.floor((c[1] + t * dy - oy) / res))
            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                continue
            g = g_int[iy, ix]
            if np.isnan(g):
                continue
            b = int(np.floor((c[2] + t * dz - g - h_lo) / binh))
            if b < 0 or b >= nb:
                continue
            cid = (iy * W + ix) * nb + b
            if cid == last:
                continue
            last = cid
            if ps[iy, ix, b] < 65535:
                ps[iy, ix, b] += 1
    return hit, ps


def floor_candidates(keys_up: np.ndarray, cnt_up: np.ndarray, H: int, W: int, zres: float, z0: float, K: int = 4,
                     min_count: int = 2):
    """Cluster the up-facing voxels of each cell along z (gaps of > 2 voxels split) -> (H,W,K) heights, (H,W) counts."""
    iy, ix, iz = keys_up
    keep = cnt_up >= 1
    iy, ix, iz, c = iy[keep], ix[keep], iz[keep], cnt_up[keep]
    cell = iy.astype(np.int64) * W + ix
    order = np.lexsort((iz, cell))
    cell, iz, c = cell[order], iz[order], c[order]
    new = np.ones(len(cell), bool)
    new[1:] = (cell[1:] != cell[:-1]) | (iz[1:] - iz[:-1] > 2)
    cid = np.cumsum(new) - 1
    tot = np.bincount(cid, weights=c)
    zm = np.bincount(cid, weights=c * (iz + 0.5)) / tot * zres + z0
    ccell = cell[new]
    good = tot >= min_count
    ccell, zm, tot = ccell[good], zm[good], tot[good]
    # top-K clusters per cell by count
    order = np.lexsort((-tot, ccell))
    ccell, zm, tot = ccell[order], zm[order], tot[order]
    first = np.ones(len(ccell), bool)
    first[1:] = ccell[1:] != ccell[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(len(ccell)), 0))
    rank = np.arange(len(ccell)) - start
    sel = rank < K
    cand = np.full((H * W, K), np.nan, np.float32)
    cand[ccell[sel], rank[sel]] = zm[sel]
    ncand = np.zeros(H * W, np.int32)
    np.add.at(ncand, ccell[sel], 1)
    return cand.reshape(H, W, K), ncand.reshape(H, W)


def ground_seeds(cams_g: np.ndarray, cand: np.ndarray, ncand: np.ndarray, origin, res, radius: float,
                 h_range=(0.08, 2.6)):
    """Floor under each camera: among the candidate heights within `radius` (horizontal) that lie h_range below the
    camera, the lowest well-supported one.  Returns seed cells / heights and the per-camera ground height (NaN)."""
    H, W, K = cand.shape
    r = int(math.ceil(radius / res))
    sy, sx, sh = [], [], []
    cam_g = np.full(len(cams_g), np.nan)
    for i, c in enumerate(cams_g):
        cx, cy = int((c[0] - origin[0]) / res), int((c[1] - origin[1]) / res)
        y0, y1, x0, x1 = max(cy - r, 0), min(cy + r + 1, H), max(cx - r, 0), min(cx + r + 1, W)
        if y0 >= y1 or x0 >= x1:
            continue
        patch = cand[y0:y1, x0:x1]
        dz = c[2] - patch
        ok = np.isfinite(patch) & (dz >= h_range[0]) & (dz <= h_range[1])
        if ok.sum() < 5:
            continue
        hs = patch[ok]
        # histogram of candidate heights (4 cm bins): lowest bin with >= 25 % of the strongest bin's support
        lo = hs.min()
        b = np.floor((hs - lo) / 0.04).astype(int)
        hist = np.bincount(b)
        thr = max(min(3, hist.max()), 0.25 * hist.max())
        kb = int(np.flatnonzero(hist >= thr)[0])
        sel = np.abs(hs - (lo + (kb + 0.5) * 0.04)) <= 0.06
        h0 = float(np.median(hs[sel]))
        cam_g[i] = h0
        yy, xx, kk = np.nonzero(ok & (np.abs(patch - h0) <= 0.04))
        sy.append(yy + y0)
        sx.append(xx + x0)
        sh.append(patch[yy, xx, kk])
    if not sy:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32), cam_g
    return (np.concatenate(sy).astype(np.int32), np.concatenate(sx).astype(np.int32),
            np.concatenate(sh).astype(np.float32), cam_g)


# ---------------------------------------------------------------------------------------------------------- place
def place_frames(scene_dirs, cfg):
    """[(scene, seq, idx)] of a place, subsampled to max_frames."""
    frames = []
    for d in scene_dirs:
        sc = load_scene(d)
        for q in sc.sequences:
            sq = sc.seq(q)
            frames += [(sc, sq, i) for i in range(0, sq.n, cfg["frame_step"])]
    if len(frames) > cfg["max_frames"]:
        idx = np.linspace(0, len(frames) - 1, cfg["max_frames"]).round().astype(int)
        frames = [frames[i] for i in idx]
    return frames


def build_place(scene_dirs, cfg: dict):
    """World BEV of one place (list of scene folders sharing a world frame).  Returns the dict saved by `save`."""
    t0 = time.time()
    frames = place_frames(scene_dirs, cfg)
    Es = np.stack([sq.E[i] for _, sq, i in frames]).astype(np.float64)
    # 1. gravity from a subset of frames' normals
    sub = frames[:: max(1, len(frames) // 150)]
    ns = []
    for sc, sq, i in sub:
        dep, K, E = load_depth(sq, i)
        _, n = frame_points(dep, K, E, cfg["pix_step"] * 2, cfg["max_range"])
        ns.append(n[:: 4])
    up, up_check = estimate_gravity(Es, np.concatenate(ns), cfg["normal_up_deg"])
    R_g = rotation_to_z(up)
    cams = camera_centres(Es) @ R_g.T
    # 2. voxels
    res, zres = cfg["res"], cfg["zres"]
    cos_up = math.cos(math.radians(cfg["normal_up_deg"]))
    keys_all, keys_up = [], []
    chunk_k, chunk_u = [], []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    carve_every = max(1, len(frames) // cfg["carve_frames"])
    sub_ray = max(1, (cfg["carve_pix"] // cfg["pix_step"]) ** 2)
    rays, ray_cam = [], []
    n_pts, n_norm = 0, 0

    def flush():
        if chunk_k:
            k = np.concatenate(chunk_k)
            u, c = np.unique(k, axis=0, return_counts=True)
            keys_all.append((u, c))
            k = np.concatenate(chunk_u)
            if len(k):
                u, c = np.unique(k, axis=0, return_counts=True)
                keys_up.append((u, c))
            chunk_k.clear()
            chunk_u.clear()

    for n_done, (sc, sq, i) in enumerate(frames):
        try:
            dep, K, E = load_depth(sq, i)
        except (OSError, FileNotFoundError):
            continue
        P, n = frame_points(dep, K, E, cfg["pix_step"], cfg["max_range"])
        P = P @ R_g.T
        n = n @ R_g.T
        q = np.floor(P / np.array([res, res, zres])).astype(np.int32)
        chunk_k.append(q)
        chunk_u.append(q[np.nan_to_num(n[:, 2], nan=0.0) > cos_up])
        n_pts += len(n)
        n_norm += int(np.isfinite(n[:, 0]).sum())
        if n_done % carve_every == 0 and len(P):
            rays.append(P[::sub_ray].astype(np.float32))
            ray_cam.append(np.full(len(rays[-1]), n_done, np.int32))
        if len(q):
            lo = np.minimum(lo, q.min(0))
            hi = np.maximum(hi, q.max(0))
        if (n_done + 1) % 100 == 0:
            flush()
    flush()

    def merge(lst):
        k = np.concatenate([u for u, _ in lst])
        c = np.concatenate([c for _, c in lst])
        u, inv = np.unique(k, axis=0, return_inverse=True)
        return u, np.bincount(inv.reshape(-1), weights=c).astype(np.int64)

    vox, vcnt = merge(keys_all)
    vup, ucnt = merge(keys_up) if keys_up else (np.zeros((0, 3), np.int32), np.zeros(0, np.int64))
    vup_n, ucnt_n = vup, ucnt                    # voxels with up-facing points (protected from carving)
    sparse = n_norm < cfg["sparse_normals"] * max(n_pts, 1)
    if sparse:
        # sparse depth (projected LiDAR): no per-pixel normals; floor candidates = the lowest surface of each cell
        cell = vox[:, 1].astype(np.int64) * (1 << 31) + vox[:, 0]
        order = np.lexsort((vox[:, 2], cell))
        cs, zs = cell[order], vox[order, 2]
        first = np.ones(len(cs), bool)
        first[1:] = cs[1:] != cs[:-1]
        zmin = np.maximum.accumulate(np.where(first, zs, np.iinfo(np.int32).min))
        low = zs - zmin <= 2
        vup, ucnt = vox[order][low], vcnt[order][low]
    # grid covering the voxels (+ margin)
    m = 4
    ix0, iy0 = int(lo[0]) - m, int(lo[1]) - m
    W, H = int(hi[0]) - ix0 + 1 + m, int(hi[1]) - iy0 + 1 + m
    z0 = 0.0
    origin = np.array([ix0 * res, iy0 * res])
    # 3. ground
    cand, ncand = floor_candidates((vup[:, 1] - iy0, vup[:, 0] - ix0, vup[:, 2]), ucnt, H, W, zres, z0)
    sy, sx, sh, cam_g = ground_seeds(cams, cand, ncand, origin, res, cfg["seed_radius"], cfg["cam_h_range"])
    if len(sy) == 0:
        raise RuntimeError("no ground under any camera")
    ground, seen = _flood(cand, ncand, sy, sx, sh, np.float32(cfg["step"]), int(cfg["gap"]))
    # interpolated ground for the obstacle band where the floor itself was not reached (nearest ground cell)
    reached = np.isfinite(ground)
    dist, (ny, nx) = ndimage.distance_transform_edt(~reached, return_indices=True)
    g_int = ground[ny, nx]
    g_int[dist * res > cfg["interp_max"]] = np.nan
    # 4. classes
    vy, vx = vox[:, 1] - iy0, vox[:, 0] - ix0
    vz = (vox[:, 2] + 0.5) * zres + z0
    gz = g_int[vy, vx]
    dz = vz - gz
    band = np.isfinite(dz) & (dz >= cfg["h_lo"]) & (dz <= cfg["h_hi"])
    occ_pts = np.zeros((H, W), np.int64)
    np.add.at(occ_pts, (vy[band], vx[band]), vcnt[band])
    # height-resolved carving: an obstacle voxel that rays mostly pass through is noise / a moving object
    nb = int(math.ceil((cfg["h_hi"] - cfg["h_lo"]) / cfg["carve_bin"]))
    occ3 = np.zeros((H, W, nb), np.int32)
    bb = np.clip(((dz[band] - cfg["h_lo"]) / cfg["carve_bin"]).astype(int), 0, nb - 1)
    np.add.at(occ3, (vy[band], vx[band], bb), vcnt[band])
    if rays:
        hit, ps = _carve(np.concatenate(rays), np.concatenate(ray_cam), cams.astype(np.float64),
                         g_int.astype(np.float64), origin[0], origin[1], res, cfg["h_lo"], nb, cfg["carve_bin"])
        hit, ps = hit.astype(np.float32), ps.astype(np.float32)
        carved = (ps >= cfg["carve_min_pass"]) & (hit < cfg["carve_ratio"] * (hit + ps))
        del hit, ps
        # horizontal surfaces (table tops, beds, seats) are grazed by rays aimed just beyond them: never carve a voxel
        # that holds up-facing points
        if len(vup_n):
            uy, ux = vup_n[:, 1] - iy0, vup_n[:, 0] - ix0
            uz = (vup_n[:, 2] + 0.5) * zres + z0 - g_int[uy, ux]
            ok = np.isfinite(uz) & (uz >= cfg["h_lo"]) & (uz <= cfg["h_hi"]) & (ucnt_n >= 2)
            ub = np.clip(((uz[ok] - cfg["h_lo"]) / cfg["carve_bin"]).astype(int), 0, nb - 1)
            carved[uy[ok], ux[ok], ub] = False
    else:
        carved = np.zeros_like(occ3, bool)
    occ_vox = (occ3 >= cfg["occ_min"]) & ~carved
    occ_kept = np.where(carved, 0, occ3).sum(2)
    n_carved = int(((occ3 >= cfg["occ_min"]) & carved).sum())
    del occ3, carved
    # cells whose surfaces lie clearly below the ground they are attached to (drops, holes) are not traversable
    below = np.isfinite(dz) & (dz < -cfg["h_lo"] - 0.05)
    below_pts = np.zeros((H, W), np.int64)
    np.add.at(below_pts, (vy[below], vx[below]), vcnt[below])
    bev = np.zeros((H, W), np.uint8)
    occ = occ_vox.any(2) | ((below_pts >= cfg["occ_min"]) & ~seen)
    free = seen & (occ_kept <= cfg["free_max"])
    bev[free] = FREE
    bev[occ] = OCC
    # camera heights above the ground at their own cell
    cy = np.clip(((cams[:, 1] - origin[1]) / res).astype(int), 0, H - 1)
    cx = np.clip(((cams[:, 0] - origin[0]) / res).astype(int), 0, W - 1)
    cam_h = cams[:, 2] - g_int[cy, cx]
    cam_h[~np.isfinite(cam_h) | (cam_h < cfg["cam_h_range"][0]) | (cam_h > cfg["cam_h_range"][1])] = np.nan
    out = {"bev": bev, "ground": np.where(reached, ground, np.nan).astype(np.float16), "origin": origin,
           "res": np.float32(res), "R_g": R_g, "up_check": np.float32(up_check),
           "cfg": json.dumps({k: v for k, v in cfg.items()})}
    per = defaultdict(lambda: None)
    for (sc, sq, i), h in zip(frames, cam_h):
        key = f"cam_h/{sc.dir.name}/{sq.name}"
        if per[key] is None:
            per[key] = np.full(sq.n, np.nan, np.float32)
        per[key][i] = h
    out.update(per)
    stats = dict(frames=len(frames), H=H, W=W, free=int(free.sum()), occ=int(occ.sum()),
                 cam_h_ok=float(np.isfinite(cam_h).mean()), cam_h_med=float(np.nanmedian(cam_h)) if np.isfinite(cam_h).any() else None,
                 up_check=round(up_check, 2), sparse=bool(sparse), carved_vox=n_carved, sec=round(time.time() - t0, 1))
    out["stats"] = json.dumps(stats)
    return out, stats


def save(out: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)


def preview(out: dict, path: Path, cams=None):
    """PNG of a world BEV: free white, occupied black, unknown grey."""
    bev = out["bev"]
    img = np.full(bev.shape + (3,), 128, np.uint8)
    img[bev == FREE] = 255
    img[bev == OCC] = 0
    img = img[::-1]                              # y up
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------------------------------------- CLI
def places(cache: Path, dataset: str):
    """{world: [scene dirs]} of a dataset."""
    by = defaultdict(list)
    for d in list_scenes(cache, dataset):
        try:
            m = json.load(open(d / "scene.json"))
        except OSError:
            continue
        if not m.get("metric", True) or not m.get("posed", True):
            continue
        by[m.get("world", m["scene"])].append(d)
    return dict(by)


def _job(args):
    world, dirs, out_path, cfg = args
    if out_path.exists():
        return world, "exists"
    try:
        out, stats = build_place(dirs, cfg)
        save(out, out_path)
        preview(out, out_path.with_suffix(".png"))
        return world, stats
    except Exception as e:                        # noqa: BLE001 (one bad place must not stop the dataset)
        return world, f"ERROR {e!r} {traceback.format_exc(limit=3)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="world names")
    ap.add_argument("--set", nargs="*", default=[], help="cfg overrides key=value")
    a = ap.parse_args()
    cfg = {**DEFAULT, **DATASET_CFG.get(a.dataset, {})}
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], tuple) else tuple(float(x) for x in v.split(","))
    pl = places(Path(a.cache), a.dataset)
    names = sorted(pl) if a.only is None else [w for w in a.only if w in pl]
    if a.limit:
        names = names[: a.limit]
    out_dir = Path(a.out) / a.dataset
    jobs = [(w, pl[w], out_dir / f"{w}.npz", cfg) for w in names]
    print(f"{a.dataset}: {len(jobs)} places, cfg {cfg}", flush=True)
    log = open(out_dir.parent / f"{a.dataset}.log.jsonl", "a") if out_dir.parent.exists() or out_dir.parent.mkdir(parents=True) is None else None
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for k, (w, st) in enumerate(pool.imap_unordered(_job, jobs)):
            print(f"[{k + 1}/{len(jobs)}] {w} {st}", flush=True)
            log.write(json.dumps({"world": w, "stats": st}) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
