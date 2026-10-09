"""World BEV label of a simulated scene from its voxelised meshes (blend_geom.py), in the format of cross_nav/labels.py.

Label definition (the same classes as the accumulated real-data labels, but complete where no camera looked):
* ground = the storey's floor surfaces reachable from the seeds (the camera positions; before cameras exist, the
  dominant floor level inside the annotated rooms) by a flood fill over near-horizontal surfaces that steps at most
  `step` (5 cm) between neighbouring 5 cm cells (slopes yes, stairs / kerbs no) and does not pass under walls (cells
  whose geometry covers >= `wall_cover` of the obstacle band's height: wall faces, sides of tall closed cabinets);
* occupied = any geometry between ground + h_lo (0.10 m) and ground + h_hi (1.80 m), whatever its height (table tops,
  shelves, walls), using the nearest ground height for cells without reachable floor (walls, furniture blocks);
  plus every non-free cell enclosed by such geometry and free floor (wall interiors, closed furniture volumes, floor
  that cannot be reached), i.e. the holes of the building footprint.  Holes thicker than `hole_occ` (cells farther than
  that from the hole's border, e.g. a large unreachable courtyard) are left unknown instead of occupied;
* free = ground with nothing in the band;
* unknown = outside the footprint (beyond exterior walls, through windows).

    out, stats = build_mesh_bev(geom, seeds_xyz=None, regions=None)     # dict with bev, ground, origin, res, R_g, ...
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
from scipy import ndimage

UNKNOWN, FREE, OCC = 0, 1, 2

CFG = dict(step=0.05, h_lo=0.10, h_hi=1.80, wall_cover=0.75, wall_near=0.6, hole_occ=0.35, interp_max=2.0, kcand=8,
           floor_band=(-0.5, 0.5), seed_tol=0.03, seed_wall_dist=0.4, margin=6)

try:
    from numba import njit
except ImportError:                                        # pure-Python fallback (slow on large scenes)
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _flood(cand, ncand, seed_iy, seed_ix, seed_h, step, gap):
    """Multi-source BFS over the cell grid: a neighbour (8-connected) joins with its candidate height closest to the
    current cell's, if within `step` (x sqrt 2 diagonally).  Same rule as cross_nav.labels._flood with gap = 0 (copied
    here because cross_nav.labels imports torch, which the Blender environment does not have).  `gap` is unused."""
    H, W, K = cand.shape
    g = np.full((H, W), np.nan, np.float32)
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
        h = g[y, x]
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W or not np.isnan(g[ny, nx]):
                    continue
                tol = step * (1.4142 if dy != 0 and dx != 0 else 1.0)
                best, bd = np.nan, 1e9
                for k in range(ncand[ny, nx]):
                    dd = abs(cand[ny, nx, k] - h)
                    if dd <= tol and dd < bd:
                        best, bd = cand[ny, nx, k], dd
                if not np.isnan(best):
                    g[ny, nx] = best
                    qy[tail], qx[tail] = ny, nx
                    tail += 1
    return g, ~np.isnan(g)


def floor_candidates(vox_h: np.ndarray, ix0: int, iy0: int, H: int, W: int, zres: float, K: int):
    """Heights of the horizontal surfaces of each cell (voxel runs along z merged when the gap is <= 1 voxel; the top
    of each run), lowest first, up to K -> (H, W, K) float32 NaN-padded, (H, W) counts."""
    iy, ix, iz = vox_h[:, 1] - iy0, vox_h[:, 0] - ix0, vox_h[:, 2]
    cell = iy.astype(np.int64) * W + ix
    order = np.lexsort((iz, cell))
    cell, iz = cell[order], iz[order]
    new = np.ones(len(cell), bool)
    new[1:] = (cell[1:] != cell[:-1]) | (iz[1:] - iz[:-1] > 2)
    last = np.ones(len(cell), bool)
    last[:-1] = new[1:]
    ccell, ztop = cell[last], (iz[last] + 0.5) * zres          # top voxel of each run (the walkable surface)
    first = np.ones(len(ccell), bool)
    first[1:] = ccell[1:] != ccell[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(len(ccell)), 0))
    rank = np.arange(len(ccell)) - start
    sel = rank < K
    cand = np.full((H * W, K), np.nan, np.float32)
    cand[ccell[sel], rank[sel]] = ztop[sel]
    ncand = np.zeros(H * W, np.int32)
    np.add.at(ncand, ccell[sel], 1)
    return cand.reshape(H, W, K), ncand.reshape(H, W)


def region_mask(regions, origin, res, H, W, dilate_m=0.0):
    """Cells inside the annotated room polygons (layout.json `regions`, Blender frame)."""
    m = np.zeros((H, W), np.uint8)
    for r in regions or []:
        poly = r.get("poly") or []
        if len(poly) < 3:
            continue
        pts = np.array([[(p[0] - origin[0]) / res - 0.5, (p[1] - origin[1]) / res - 0.5] for p in poly])
        cv2.fillPoly(m, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], 1)
    if dilate_m > 0:
        k = int(round(dilate_m / res)) * 2 + 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
    return m > 0


def build_mesh_bev(geom: dict, seeds_xyz=None, seeds_h=None, regions=None, cfg: dict | None = None, frame=None):
    """geom: blend_geom.py output.  seeds_xyz (N, 3) camera centres and seeds_h (N,) their planned heights above the
    floor (the floor seed is the candidate nearest to z - h under each camera); without seeds, the dominant floor
    level inside the room polygons.  frame: optional fixed (origin, H, W) so that labels of one scene share a grid."""
    c = {**CFG, **(cfg or {})}
    res, zres = float(geom["res"]), float(geom["zres"])
    vox, vox_h = geom["vox"], geom["vox_h"]
    if frame is None:
        m = c["margin"]
        ix0, iy0 = int(vox[:, 0].min()) - m, int(vox[:, 1].min()) - m
        W, H = int(vox[:, 0].max()) - ix0 + 1 + m, int(vox[:, 1].max()) - iy0 + 1 + m
    else:
        (ix0, iy0), H, W = frame
    origin = np.array([ix0 * res, iy0 * res], np.float64)
    cand, ncand = floor_candidates(vox_h, ix0, iy0, H, W, zres, c["kcand"])
    rmask = region_mask(regions, origin, res, H, W) if regions else None
    # ---- seeds
    if seeds_xyz is None:
        lo, hi = c["floor_band"]
        z = cand[np.isfinite(cand)]
        z = z[(z >= lo) & (z <= hi)]
        hist, edges = np.histogram(z, bins=np.arange(lo, hi + 1e-6, 0.01))
        h0 = float(edges[np.argmax(hist)] + 0.005)
        ok = np.isfinite(cand) & (np.abs(cand - h0) <= c["seed_tol"])
        if rmask is not None:
            ok &= rmask[..., None]
        sy, sx, sk = np.nonzero(ok)
        sh = cand[sy, sx, sk]
    else:
        sy, sx, sh = [], [], []
        for p, h in zip(np.asarray(seeds_xyz), np.asarray(seeds_h)):
            cx, cy = int(math.floor((p[0] - origin[0]) / res)), int(math.floor((p[1] - origin[1]) / res))
            if not (0 <= cy < H and 0 <= cx < W) or ncand[cy, cx] == 0:
                continue
            cc = cand[cy, cx, :ncand[cy, cx]]
            k = int(np.argmin(np.abs(cc - (p[2] - h))))
            if abs(cc[k] - (p[2] - h)) < 0.1:
                sy.append(cy), sx.append(cx), sh.append(cc[k])
        h0 = float(np.median(sh)) if sh else float("nan")
    sy, sx, sh = np.asarray(sy, np.int32), np.asarray(sx, np.int32), np.asarray(sh, np.float32)
    if len(sy) == 0:
        raise RuntimeError("no floor seeds")
    # wall cells: vertical structure covering most of the obstacle band (relative to the storey's floor level h0);
    # the floor flood does not pass under them (mesh floors often continue under walls and closed cabinets)
    vy, vx = vox[:, 1] - iy0, vox[:, 0] - ix0
    inside = (vy >= 0) & (vy < H) & (vx >= 0) & (vx < W)
    vy, vx, viz = vy[inside], vx[inside], vox[inside, 2]
    vz = (viz + 0.5) * zres
    inb = (vz >= h0 + c["h_lo"]) & (vz <= h0 + c["h_hi"])
    cover = np.zeros((H, W), np.int32)
    np.add.at(cover, (vy[inb], vx[inb]), 1)
    wall = cover >= c["wall_cover"] * (c["h_hi"] - c["h_lo"]) / zres
    ncand_f = np.where(wall, 0, ncand).astype(np.int32)
    keep = ~wall[sy, sx]
    if seeds_xyz is None:                        # no seeds in wall interiors / closed cabinets: away from walls
        keep &= (ndimage.distance_transform_edt(~wall) * res)[sy, sx] >= c["seed_wall_dist"]
    sy, sx, sh = sy[keep], sx[keep], sh[keep]
    ground, _ = _flood(cand, ncand_f, sy, sx, sh, np.float32(c["step"]), 0)
    reached = np.isfinite(ground)
    dist, (ny, nx) = ndimage.distance_transform_edt(~reached, return_indices=True)
    g_int = ground[ny, nx]
    g_int[dist * res > c["interp_max"]] = np.nan
    # ---- band occupancy
    dz = vz - g_int[vy, vx]
    band = np.isfinite(dz) & (dz >= c["h_lo"]) & (dz <= c["h_hi"])
    occ_band = np.zeros((H, W), bool)
    occ_band[vy[band], vx[band]] = True
    free = reached & ~occ_band
    # ---- footprint: free floor + nearby obstacles, holes filled
    d_free = ndimage.distance_transform_edt(~free) * res
    shell = free | (occ_band & (d_free <= c["wall_near"]))
    foot = ndimage.binary_fill_holes(shell)
    holes = foot & ~free & ~occ_band                              # enclosed, no geometry in the band, not free floor
    d_hole = ndimage.distance_transform_edt(holes) * res          # distance to the hole's border
    deep = holes & (d_hole > c["hole_occ"])
    bev = np.zeros((H, W), np.uint8)
    bev[foot] = OCC
    bev[deep] = UNKNOWN
    bev[free] = FREE
    stats = dict(H=H, W=W, free=int(free.sum()), occ=int((bev == OCC).sum()), unknown_holes=int(deep.sum()),
                 floor_h=round(h0, 3), seeds=int(len(sy)), ground_cells=int(reached.sum()), wall_cells=int(wall.sum()))
    out = {"bev": bev, "ground": np.where(reached, ground, np.nan).astype(np.float16), "origin": origin,
           "res": np.float32(res), "R_g": np.eye(3), "up_check": np.float32(0.0),
           "cfg": json.dumps({k: v for k, v in c.items()})}
    aux = {"g_int": g_int, "rmask": rmask, "frame": ((ix0, iy0), H, W), "occ_band": occ_band}
    return out, stats, aux


def camera_heights(Es: np.ndarray, g_int: np.ndarray, origin, res) -> np.ndarray:
    """Camera centre height above the (interpolated) ground at its own cell, per frame (NaN off the grid)."""
    C = -np.einsum("nji,nj->ni", Es[:, :3, :3], Es[:, :3, 3])
    H, W = g_int.shape
    cx = np.floor((C[:, 0] - origin[0]) / res).astype(int)
    cy = np.floor((C[:, 1] - origin[1]) / res).astype(int)
    ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    h = np.full(len(C), np.nan, np.float32)
    h[ok] = C[ok, 2] - g_int[cy[ok], cx[ok]]
    return h


def preview(out: dict, path, trajs=None, scale: int = 2):
    """PNG: free white, occupied black, unknown grey, y up; trajectories (lists of (N, 2) world xy) in colour."""
    bev = out["bev"]
    img = np.full(bev.shape + (3,), 128, np.uint8)
    img[bev == FREE] = 255
    img[bev == OCC] = 0
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cols = [(0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 160, 255), (200, 0, 200), (255, 160, 0)]
    res, o = float(out["res"]), out["origin"]
    for k, xy in enumerate(trajs or []):
        p = np.stack([(xy[:, 0] - o[0]) / res * scale, (xy[:, 1] - o[1]) / res * scale], 1).round().astype(np.int32)
        cv2.polylines(img, [p.reshape(-1, 1, 2)], False, cols[k % len(cols)], 1, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in p[0]), 3, cols[k % len(cols)], -1)
    cv2.imwrite(str(path), img[::-1])
