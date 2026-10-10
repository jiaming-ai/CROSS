"""Complete world BEV labels of TartanGround environments from their global semantic point clouds.

TartanGround ships per environment `<Env>/<Env>_sem_pcd.zip`: points sampled from the Unreal meshes (grass and small
bushes excluded), in the trajectories' NED world frame, colour = semantic class.  The labels follow the definition of
the simulated HSSD labels (cross_nav/sim/mesh_bev.py), so they are complete where no camera looked:

* gravity is exact (NED: up = -z), R_g = diag(1, -1, -1);
* voxels (5 cm x 5 cm x 2 cm) of the points within `margin` m (horizontally) of the cameras, sky sphere removed;
  floor candidates = the tops of the voxel runs of each cell (mesh_bev.floor_candidates);
* seeds: the floor under each camera at its trajectory's `robot_height` (scene.json, from the TartanGround metadata);
* pass 1: flood fill (step 5 cm, slopes yes, kerbs / stairs no) without walls -> a local reference ground everywhere;
  walls = cells whose points cover >= `wall_cover` of the obstacle band above that local reference (unlike mesh_bev,
  which uses one storey level, outdoor terrain is not flat); pass 2: the flood fill again, not passing under walls;
* occupied = any point between ground + h_lo and ground + h_hi (nearest ground for cells off the ground), plus the
  enclosed holes of the footprint (building interiors without reachable floor, thicker ones unknown), free = ground
  with nothing in the band, unknown = everything else (as mesh_bev.build_mesh_bev).

    python -m cross_nav.sim.pcd_bev --raw /data0/jz/raw/tartanground --cache <cache root> --out <bev root> \
        [--dataset tartanground] [--label-name tartanground_pcd] [--envs A B] [--workers 4]

Output <out>/<label-name>/<world>.npz (+ .png with the trajectories): the keys of cross_nav/labels.py (bev, ground,
origin, res, R_g, up_check, cam_h/<scene>/<seq>, stats, cfg).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
import traceback
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

from cross_nav.sim.mesh_bev import CFG as MESH_CFG
try:
    from numba import njit
except ImportError:                                        # pure-Python fallback (slow)
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]
from cross_nav.sim.mesh_bev import camera_heights, floor_candidates, preview

UNKNOWN, FREE, OCC = 0, 1, 2
R_G = np.diag([1.0, -1.0, -1.0])                     # NED world -> gravity frame (z up)
CFG = {**MESH_CFG, "res": 0.05, "zres": 0.02, "margin_m": 25.0, "z_below": 20.0, "z_above": 6.0, "seed_every": 5,
       "kcand": 8, "seed_tol_h": 0.25, "gap": 3, "carry_lo": 0.5, "dens_win": 6, "dens_grow": 0.8, "dens_rmax": 3, "dens_min": 0.3}


@njit(cache=True)
def _flood_gap(cand, ncand, seed_iy, seed_ix, seed_h, step, gap, lo, hi):
    """cross_nav.labels._flood with another carry rule: the height is carried across up to `gap` cells that have no
    candidate within [h - lo, h + hi] (sparse clouds: a cell whose only point is on the ceiling, a canopy or far
    below), not only across cells without any candidate."""
    H, W, K = cand.shape
    g = np.full((H, W), np.nan, np.float32)
    gused = np.zeros((H, W), np.int32)
    qn = H * W + len(seed_iy) + 1
    qy = np.empty(qn, np.int32)
    qx = np.empty(qn, np.int32)
    head, tail = 0, 0
    for s in range(len(seed_iy)):
        y, x = seed_iy[s], seed_ix[s]
        if np.isnan(g[y, x]):
            g[y, x] = seed_h[s]
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
                tol = step * (1.4142 if dy != 0 and dx != 0 else 1.0)
                best, bd = np.nan, 1e9
                near = False
                for k in range(ncand[ny, nx]):
                    dd = abs(cand[ny, nx, k] - h)
                    if dd <= tol and dd < bd:
                        best, bd = cand[ny, nx, k], dd
                    if cand[ny, nx, k] - h >= -lo and cand[ny, nx, k] - h <= hi:
                        near = True
                if not np.isnan(best):
                    g[ny, nx] = best
                    gused[ny, nx] = 0
                    qy[tail], qx[tail] = ny, nx
                    tail += 1
                elif gu < gap and not near:
                    g[ny, nx] = h
                    gused[ny, nx] = gu + 1
                    qy[tail], qx[tail] = ny, nx
                    tail += 1
    return g


def read_pcd_bytes(buf: bytes):
    """(N, 3) float32 points and (N,) uint32 packed colours of a binary PCD with fields x y z rgb."""
    end = 0
    hdr = {}
    while True:
        nl = buf.index(b"\n", end)
        line = buf[end:nl].decode().strip()
        end = nl + 1
        if not line or line.startswith("#"):
            continue
        k, *v = line.split()
        hdr[k] = v
        if k == "DATA":
            break
    n = int(hdr["POINTS"][0])
    if hdr["DATA"][0] != "binary" or hdr["FIELDS"] != ["x", "y", "z", "rgb"]:
        raise ValueError(f"unsupported PCD header {hdr}")
    a = np.frombuffer(buf, np.float32, count=n * 4, offset=end).reshape(n, 4)
    return a[:, :3], a[:, 3].view(np.uint32)


def sky_colours(env_dir: Path, seg_rgbs: Path | None) -> set[int]:
    """Packed colours (both RGB orders) of the sky-sphere classes of an environment, from seg_labels.zip."""
    z = env_dir / "seg_labels.zip"
    if seg_rgbs is None or not z.exists():
        return set()
    try:
        nm = json.loads(zipfile.ZipFile(z).read("seg_label_map.json"))["name_map"]
    except (KeyError, ValueError):
        return set()
    rgbs = [tuple(int(v) for v in l.split(",")) for l in open(seg_rgbs).read().split("\n") if l.strip()]
    out = set()
    for name, idx in nm.items():
        if "sky" in name.lower() and idx < len(rgbs):
            r, g, b = rgbs[idx]
            out |= {(r << 16) | (g << 8) | b, (b << 16) | (g << 8) | r}
    return out


def world_frames(scene_dir: Path):
    """[(seq, E (n,4,4), robot_height)] of a cache scene (one TartanGround environment)."""
    from vggt_ft.dataio.scene import load_scene
    sc = load_scene(scene_dir)
    meta = json.load(open(scene_dir / "scene.json"))
    rh = meta.get("robot_height", {})
    # quadruped trajectories have no robot_height in their metadata: NaN (not used as seeds)
    return sc, [(q, sc.seq(q).E.astype(np.float64), float(rh[q]) if rh.get(q) is not None else float("nan"))
                for q in sc.sequences]


def build_pcd_bev(P: np.ndarray, cams: np.ndarray, cam_hs: np.ndarray, cfg: dict, seq_id=None):
    """P (N,3) points and cams (M,3) camera centres in the gravity frame, cam_hs (M,) camera heights above the floor.
    Returns (out dict, stats, aux)."""
    c = cfg
    res, zres = c["res"], c["zres"]
    q = np.floor(P / np.array([res, res, zres])).astype(np.int64)
    del P
    lo = q.min(0)
    span = q.max(0) - lo + 1
    key = ((q[:, 0] - lo[0]) * span[1] + (q[:, 1] - lo[1])) * span[2] + (q[:, 2] - lo[2])
    del q
    key, cnt = np.unique(key, return_counts=True)
    vox = np.stack([key // (span[1] * span[2]) + lo[0], (key // span[2]) % span[1] + lo[1], key % span[2] + lo[2]],
                   1).astype(np.int64)
    del key
    m = c["margin"]
    ix0, iy0 = int(vox[:, 0].min()) - m, int(vox[:, 1].min()) - m
    W, H = int(vox[:, 0].max()) - ix0 + 1 + m, int(vox[:, 1].max()) - iy0 + 1 + m
    origin = np.array([ix0 * res, iy0 * res], np.float64)
    cand, ncand = floor_candidates(vox, ix0, iy0, H, W, zres, c["kcand"])
    # ---- seeds: the candidate nearest to z - robot_height under each camera.  robot_height is missing for quadrupeds
    # and sometimes wrong (diff-drive trajectories 0.5 - 1 m off): a trajectory whose cameras mostly find no floor at
    # robot_height is seeded with the highest surface 0.2 - 2.0 m below each camera instead (the robot stands on it).
    if seq_id is None:
        seq_id = np.zeros(len(cams), int)
    cell = []
    for p in cams:
        cx, cy = int(math.floor((p[0] - origin[0]) / res)), int(math.floor((p[1] - origin[1]) / res))
        cell.append((cy, cx) if 0 <= cy < H and 0 <= cx < W and ncand[cy, cx] > 0 else None)
    pick = np.full(len(cams), np.nan)
    for i, (p, h) in enumerate(zip(cams, cam_hs)):
        if cell[i] is None or not np.isfinite(h):
            continue
        cc = cand[cell[i][0], cell[i][1], :ncand[cell[i]]]
        k = int(np.argmin(np.abs(cc - (p[2] - h))))
        if abs(cc[k] - (p[2] - h)) < c["seed_tol_h"]:
            pick[i] = cc[k]
    n_fallback = 0
    for s_ in np.unique(seq_id):
        m = seq_id == s_
        if np.isfinite(pick[m]).mean() >= 0.5:
            continue
        for i in np.flatnonzero(m):
            if cell[i] is None:
                continue
            cc = cand[cell[i][0], cell[i][1], :ncand[cell[i]]]
            dz = cams[i, 2] - cc
            ok = (dz >= 0.2) & (dz <= 2.0)
            pick[i] = cc[ok].max() if ok.any() else np.nan
        n_fallback += 1
    sy, sx, sh = [], [], []
    for i in np.flatnonzero(np.isfinite(pick)):
        sy.append(cell[i][0]), sx.append(cell[i][1]), sh.append(pick[i])
    n_seed_cams = len(sy)
    if not sy:
        raise RuntimeError("no floor seeds")
    sy, sx, sh = np.asarray(sy, np.int32), np.asarray(sx, np.int32), np.asarray(sh, np.float32)
    vy, vx, vz = (vox[:, 1] - iy0).astype(np.int64), (vox[:, 0] - ix0).astype(np.int64), (vox[:, 2] + 0.5) * zres
    band_vox = (c["h_hi"] - c["h_lo"]) / zres

    def nearest(g):
        reached = np.isfinite(g)
        dist, (ny, nx) = ndimage.distance_transform_edt(~reached, return_indices=True)
        gi = g[ny, nx]
        return gi, dist * res, reached

    # ---- pass 1 (no walls) -> local reference ground; walls relative to it
    # sparse clouds leave single cells without a floor point: the fill carries the height across up to `gap` of them
    fl = (np.float32(c["step"]), int(c["gap"]), np.float32(c["carry_lo"]), np.float32(c["h_hi"]))
    g1 = _flood_gap(cand, ncand, sy, sx, sh, *fl)
    gref, _, _ = nearest(g1)
    del g1
    dz = vz - gref[vy, vx]
    inb = (dz >= c["h_lo"]) & (dz <= c["h_hi"])
    cover = np.bincount(vy[inb] * W + vx[inb], minlength=H * W).reshape(H, W)
    wall = cover >= c["wall_cover"] * band_vox
    del cover, gref
    cand[wall, 0] = 1e4                                  # walls: one unreachable candidate (no gap carry either)
    ncand_f = np.where(wall, 1, ncand).astype(np.int32)
    keep = ~wall[sy, sx]
    ground = _flood_gap(cand, ncand_f, sy[keep], sx[keep], sh[keep], *fl)
    del cand, ncand_f
    g_int, dist, reached = nearest(ground)
    g_int[dist > c["interp_max"]] = np.nan
    # ---- band occupancy
    dz = vz - g_int[vy, vx]
    band = np.isfinite(dz) & (dz >= c["h_lo"]) & (dz <= c["h_hi"])
    occ_band = np.bincount(vy[band] * W + vx[band], minlength=H * W).reshape(H, W) > 0
    # point density of the ground (points per cell, local mean over the reached cells): the clouds are dense near the
    # trajectories and sparse far from them (~50 vs ~1 points per 5 cm cell), where objects show up as scattered
    # points.  Grow the obstacles by about one sample spacing there, and leave too sparse ground unknown.
    onfloor = np.isfinite(dz) & (np.abs(dz) <= 0.04)
    gpts = np.bincount(vy[onfloor] * W + vx[onfloor], weights=cnt[onfloor], minlength=H * W).reshape(H, W)
    k = 2 * c["dens_win"] + 1
    num = ndimage.uniform_filter(np.where(reached, gpts, 0.0), k)
    den = ndimage.uniform_filter(reached.astype(np.float64), k)
    dens = num / np.maximum(den, 1e-6)
    del gpts, num, den
    rad = np.clip(np.round(c["dens_grow"] / np.sqrt(np.maximum(dens, 1e-3))), 0, c["dens_rmax"]).astype(int)
    occ_grow = occ_band.copy()
    for r in range(1, c["dens_rmax"] + 1):
        disk = (np.add.outer(np.arange(-r, r + 1) ** 2, np.arange(-r, r + 1) ** 2) <= r * r + r)
        occ_grow |= ndimage.binary_dilation(occ_band, disk) & (rad >= r)
    sparse = reached & (dens < c["dens_min"])
    free = reached & ~occ_grow & ~sparse
    # ---- footprint: free floor + nearby obstacles, holes filled (mesh_bev.build_mesh_bev)
    d_free = ndimage.distance_transform_edt(~(free | sparse)) * res
    shell = free | sparse | (occ_grow & (d_free <= c["wall_near"]))
    foot = ndimage.binary_fill_holes(shell)
    holes = foot & ~free & ~occ_grow & ~sparse
    deep = holes & (ndimage.distance_transform_edt(holes) * res > c["hole_occ"])
    bev = np.zeros((H, W), np.uint8)
    bev[foot] = OCC
    bev[deep] = UNKNOWN
    bev[free] = FREE
    out = {"bev": bev, "ground": np.where(reached, ground, np.nan).astype(np.float16), "origin": origin,
           "res": np.float32(res), "R_g": R_G.copy(), "up_check": np.float32(0.0),
           "cfg": json.dumps({k: v for k, v in c.items()})}
    bev[sparse & ~occ_band] = UNKNOWN
    stats = dict(H=H, W=W, free=int(free.sum()), occ=int((bev == OCC).sum()), unknown_holes=int(deep.sum()),
                 sparse_cells=int(sparse.sum()), grown_cells=int((occ_grow & ~occ_band).sum()),
                 dens_med=round(float(np.median(dens[reached])), 2) if reached.any() else 0.0,
                 seeds=int(keep.sum()), seed_cams=n_seed_cams, fallback_seqs=n_fallback, ground_cells=int(reached.sum()),
                 wall_cells=int(wall.sum()), vox=int(len(vox)))
    return out, stats, {"g_int": g_int, "occ_band": occ_band, "occ_grow": occ_grow, "sparse": sparse, "holes": holes,
                        "dens": dens.astype(np.float32), "reached": reached, "wall": wall}


def build_world(env: str, raw: Path, scene_dir: Path, cfg: dict, seg_rgbs: Path | None):
    t0 = time.time()
    sc, seqs = world_frames(scene_dir)
    Es = np.concatenate([E for _, E, _ in seqs])
    hs = np.concatenate([np.full(len(E), h) for _, E, h in seqs])
    sid = np.concatenate([np.full(len(E), k) for k, (_, E, _) in enumerate(seqs)])
    C = -np.einsum("nji,nj->ni", Es[:, :3, :3], Es[:, :3, 3]) @ R_G.T
    lo, hi = C.min(0), C.max(0)
    zf = zipfile.ZipFile(raw / env / f"{env}_sem_pcd.zip")
    name = [n for n in zf.namelist() if n.endswith(".pcd")][0]
    P, col = read_pcd_bytes(zf.read(name))
    n_all = len(P)
    P = P @ R_G.T                                                            # new array (gravity frame)
    keep = ((P[:, 0] >= lo[0] - cfg["margin_m"]) & (P[:, 0] <= hi[0] + cfg["margin_m"])
            & (P[:, 1] >= lo[1] - cfg["margin_m"]) & (P[:, 1] <= hi[1] + cfg["margin_m"])
            & (P[:, 2] >= lo[2] - cfg["z_below"]) & (P[:, 2] <= hi[2] + cfg["z_above"]))
    sky = sky_colours(raw / env, seg_rgbs)
    n_sky = 0
    if sky:
        s = np.isin(col, np.array(sorted(sky), np.uint32))
        n_sky = int((s & keep).sum())
        keep &= ~s
    P = P[keep]
    del col, keep
    sel = np.arange(0, len(C), cfg["seed_every"])
    out, stats, aux = build_pcd_bev(P, C[sel], hs[sel], cfg, sid[sel])
    # density of the floor: points per ground cell near the cameras (low density = holes in the flood fill)
    stats["points"], stats["points_all"], stats["sky_removed"] = int(len(P)), int(n_all), n_sky
    for q, E, h in seqs:
        ch = camera_heights(E @ np.linalg.inv(np.block([[R_G, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])),
                            aux["g_int"], out["origin"], float(out["res"]))
        ch[~np.isfinite(ch) | (ch < 0.08) | (ch > 2.6)] = np.nan
        out[f"cam_h/{sc.dir.name}/{q}"] = ch.astype(np.float32)
        stats.setdefault("cam_h_err", {})[q] = (round(float(np.nanmedian(ch) - h), 3)
                                                if np.isfinite(ch).any() and np.isfinite(h) else None)
        stats.setdefault("cam_h_ok", {})[q] = round(float(np.isfinite(ch).mean()), 3)
    stats["sec"] = round(time.time() - t0, 1)
    out["stats"] = json.dumps(stats)
    trajs = [(-np.einsum("nji,nj->ni", E[:, :3, :3], E[:, :3, 3]) @ R_G.T)[:, :2] for _, E, _ in seqs]
    return out, stats, trajs


def _job(a):
    env, raw, scene_dir, out_path, cfg, seg_rgbs = a
    if out_path.exists():
        return env, "exists"
    try:
        out, stats, trajs = build_world(env, Path(raw), Path(scene_dir), cfg, seg_rgbs)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **out)
        os.replace(tmp, out_path)
        scale = 2 if out["bev"].size < 4e6 else 1
        preview(out, out_path.with_suffix(".png"), trajs=trajs, scale=scale)
        return env, stats
    except Exception as e:                        # noqa: BLE001
        return env, f"ERROR {e!r} {traceback.format_exc(limit=4)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="TartanGround raw root (<Env>/<Env>_sem_pcd.zip, seg_labels.zip)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="tartanground")
    ap.add_argument("--label-name", default="tartanground_pcd")
    ap.add_argument("--envs", nargs="*")
    ap.add_argument("--seg-rgbs", default=None, help="tartanairpy's seg_rgbs.txt (to drop the sky sphere)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[], help="cfg overrides key=value")
    a = ap.parse_args()
    cfg = dict(CFG)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = type(cfg[k])(v)
    root = Path(a.cache) / a.dataset
    envs = [d.name for d in sorted(root.iterdir()) if (d / "scene.json").exists()
            and (Path(a.raw) / d.name / f"{d.name}_sem_pcd.zip.done").exists()]
    if a.envs:
        envs = [e for e in envs if e in a.envs]
    out_dir = Path(a.out) / a.label_name
    jobs = [(e, a.raw, root / e, out_dir / f"{e}.npz", cfg, a.seg_rgbs) for e in envs]
    print(f"{len(jobs)} worlds", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir.parent / f"{a.label_name}.log.jsonl", "a")
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for k, (e, st) in enumerate(pool.imap_unordered(_job, jobs)):
            print(f"[{k + 1}/{len(jobs)}] {e} {st}", flush=True)
            log.write(json.dumps({"world": e, "stats": st}) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
