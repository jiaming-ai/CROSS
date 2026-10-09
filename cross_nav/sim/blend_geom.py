"""Voxelised mesh geometry of a Blender scene (run with the bpy environment).

    python -m cross_nav.sim.blend_geom <scene.blend> <out_geom.npz> [--skip-categories door] [--spacing 0.02]

Every visible mesh instance (door leaves skipped: treated as open, as SimChange does) is sampled on a barycentric grid
with a point spacing of at most `spacing` along every triangle edge, so thin walls and large floor triangles leave no
gaps at the voxel size.  Output: unique voxel keys (ix, iy, iz) of cells res x res x zres (world frame of the blend,
z up) of all surfaces (`vox`) and of near-horizontal surfaces (|n_z| > cos 25 deg, `vox_h`), plus res / zres.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np


def tri_samples(a: np.ndarray, b: np.ndarray, c: np.ndarray, spacing: float, kmax: int = 2000):
    """Barycentric grid points of triangles (a, b, c: (T, 3)); k = ceil(longest edge / spacing) subdivisions per
    triangle (k = 1: the vertices).  Returns (P (M, 3), tri index (M,))."""
    e = np.maximum(np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1)),
                   np.linalg.norm(a - c, axis=1))
    k = np.clip(np.ceil(e / spacing).astype(np.int64), 1, kmax)
    out_p, out_t = [], []
    for kk in np.unique(k):
        sel = np.nonzero(k == kk)[0]
        i, j = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1), indexing="ij")
        m = (i + j) <= kk
        u, v = (i[m] / kk).astype(np.float32), (j[m] / kk).astype(np.float32)
        # chunk so that the point array stays below ~20M points
        step = max(1, int(2e7 // len(u)))
        for s0 in range(0, len(sel), step):
            ss = sel[s0:s0 + step]
            A, AB, AC = a[ss, None], (b - a)[ss, None], (c - a)[ss, None]
            P = A + u[None, :, None] * AB + v[None, :, None] * AC
            out_p.append(P.reshape(-1, 3))
            out_t.append(np.repeat(ss, len(u)))
    if not out_p:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int64)
    return np.concatenate(out_p), np.concatenate(out_t)


def main():
    import bpy

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("blend")
    ap.add_argument("out")
    ap.add_argument("--skip-categories", default="door")
    ap.add_argument("--spacing", type=float, default=0.02)
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--zres", type=float, default=0.02)
    ap.add_argument("--zmin", type=float, default=-3.0)
    ap.add_argument("--zmax", type=float, default=6.0)
    a = ap.parse_args(argv)
    t0 = time.time()
    skip = set(filter(None, a.skip_categories.split(",")))
    bpy.ops.wm.open_mainfile(filepath=a.blend)
    for o in bpy.data.objects:
        if skip and o.get("hssd_category") in skip:
            o.hide_render = True
            o.hide_viewport = True
    deps = bpy.context.evaluated_depsgraph_get()
    cos_h = math.cos(math.radians(25.0))
    q = np.array([a.res, a.res, a.zres])
    mesh_cache = {}
    keys, keys_h = [], []
    n_inst = n_pts = 0
    for inst in deps.object_instances:
        o = inst.object
        if o.type != "MESH" or (skip and o.get("hssd_category") in skip) or o.hide_render:
            continue
        tri = mesh_cache.get(o.data.name)
        if tri is None:
            try:
                me = o.to_mesh()
            except Exception:                              # noqa: BLE001
                continue
            if me is None or len(me.polygons) == 0:
                o.to_mesh_clear()
                continue
            me.calc_loop_triangles()
            V = np.empty(len(me.vertices) * 3, np.float64)
            me.vertices.foreach_get("co", V)
            T = np.empty(len(me.loop_triangles) * 3, np.int64)
            me.loop_triangles.foreach_get("vertices", T)
            tri = (V.reshape(-1, 3), T.reshape(-1, 3))
            o.to_mesh_clear()
            if not o.modifiers:                            # shared mesh data: reuse for the other instances
                mesh_cache[o.data.name] = tri
        V, T = tri
        M = np.array(inst.matrix_world)
        Vw = V @ M[:3, :3].T + M[:3, 3]
        A, B, C = Vw[T[:, 0]], Vw[T[:, 1]], Vw[T[:, 2]]
        n = np.cross(B - A, C - A)
        nn = np.linalg.norm(n, axis=1)
        ok = nn > 1e-12
        A, B, C, n, nn = A[ok], B[ok], C[ok], n[ok], nn[ok]
        horiz = np.abs(n[:, 2]) / nn > cos_h
        zlo = np.minimum(np.minimum(A[:, 2], B[:, 2]), C[:, 2])
        zhi = np.maximum(np.maximum(A[:, 2], B[:, 2]), C[:, 2])
        inz = (zhi >= a.zmin) & (zlo <= a.zmax)
        A, B, C, horiz = A[inz], B[inz], C[inz], horiz[inz]
        if not len(A):
            continue
        P, ti = tri_samples(A.astype(np.float32), B.astype(np.float32), C.astype(np.float32), a.spacing)
        k = np.floor(P / q).astype(np.int32)
        keys.append(np.unique(k, axis=0))
        if horiz.any():
            keys_h.append(np.unique(k[horiz[ti]], axis=0))
        n_inst += 1
        n_pts += len(P)
        if len(keys) > 64:
            keys = [np.unique(np.concatenate(keys), axis=0)]
            keys_h = [np.unique(np.concatenate(keys_h), axis=0)] if keys_h else []
    vox = np.unique(np.concatenate(keys), axis=0)
    vox_h = np.unique(np.concatenate(keys_h), axis=0) if keys_h else np.zeros((0, 3), np.int32)
    np.savez_compressed(a.out, vox=vox.astype(np.int32), vox_h=vox_h.astype(np.int32), res=np.float32(a.res),
                        zres=np.float32(a.zres))
    print(f"[geom] {n_inst} mesh instances, {n_pts / 1e6:.1f}M samples -> {len(vox)} voxels ({len(vox_h)} horizontal), "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
