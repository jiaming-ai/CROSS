"""Input-observation targets from depth, with explicit occlusion and height tests.

These are supervision/evaluation labels, never inputs to the learned predictor.
The current robot footprint assumption belongs to the planning audit, not this mask.
"""
import numpy as np

from .bev import bev_axes_np, camera_bev_pose, crop_world, points_to_cells


def observed_cells(depths, extrinsics, intrinsics, up_cam, bev, world, anchor_E,
                   spec, frame_valid=None, depth_tolerance=.05, ground_offset=.02):
    depths, extrinsics, intrinsics = map(np.asarray, (depths, extrinsics, intrinsics))
    if depth_tolerance < 0 or ground_offset < 0:
        raise ValueError("observation tolerances must be nonnegative")
    if frame_valid is None:
        frame_valid = np.ones(len(depths), bool)
    frame_valid = np.asarray(frame_valid, bool)
    if frame_valid.shape != (len(depths),):
        raise ValueError("one validity flag is required per input frame")
    ground = np.asarray(world["ground"], np.float32)
    Rg, origin, res = world["R_g"], world["origin"], world["res"]
    cfg = world.get("label_cfg", {})
    hlo, hhi = cfg.get("h_lo", .1), cfg.get("h_hi", 1.8)
    center = -anchor_E[:3, :3].T @ anchor_E[:3, 3]
    A = bev_axes_np(up_cam) @ extrinsics[0, :3, :3]
    pos, right, fwd, _ = camera_bev_pose(anchor_E, Rg)
    floor = crop_world(ground, origin, res, pos, right, fwd, spec, fill=float("nan"))
    x, y = spec.cell_xy()
    Pg = pos + x[..., None]*right + y[..., None]*fwd
    Pg[..., 2] = floor + ground_offset
    floor_world = Pg @ Rg - center
    free_seen = np.zeros_like(bev, bool)
    surface_seen = np.zeros_like(bev, bool)
    _, H, W = depths.shape
    ys, xs = np.mgrid[:H, :W]
    for dep, E, K, use in zip(depths, extrinsics, intrinsics, frame_valid):
        if not use:
            continue
        valid = np.isfinite(dep) & (dep > 0)
        d = dep[valid]
        P = np.stack([(xs[valid]-K[0, 2])/K[0, 0]*d,
                      (ys[valid]-K[1, 2])/K[1, 1]*d, d], 1)
        Pw = (P-E[:3, 3]) @ E[:3, :3]
        Pg = (Pw+center) @ Rg.T
        xy = np.floor((Pg[:, :2]-origin)/res).astype(int)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < ground.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < ground.shape[0])
        idx = np.flatnonzero(inside)
        g = ground[xy[idx, 1], xy[idx, 0]]
        height = Pg[idx, 2]-g
        idx = idx[np.isfinite(g) & (height >= hlo) & (height <= hhi)]
        rr, cc, ok = points_to_cells(Pw[idx] @ A.T, spec)
        surface_seen[rr[ok], cc[ok]] = True
        Pc = floor_world @ E[:3, :3].T + E[:3, 3]
        z = Pc[..., 2]
        valid = np.isfinite(Pc).all(-1) & (z > .001)
        safe_z = np.where(valid, z, 1)
        u = np.rint(K[0, 0]*np.where(valid, Pc[..., 0], 0)/safe_z+K[0, 2]).astype(int)
        v = np.rint(K[1, 1]*np.where(valid, Pc[..., 1], 0)/safe_z+K[1, 2]).astype(int)
        valid &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        rr, cc = np.where(valid)
        d = dep[v[rr, cc], u[rr, cc]]
        ok = np.isfinite(d) & (d > 0) & (z[rr, cc] <= d+depth_tolerance)
        free_seen[rr[ok], cc[ok]] = True
    return (free_seen & (bev == 1)) | (surface_seen & (bev == 2))
