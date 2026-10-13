"""Navigation map of a saved CROSS map: one BEV per permanent keyframe, fused on a gravity-aligned grid.

    python -m cross_nav.navmap --map <map.pkl> --heads <heads_XXXXXX_ema.pt> --backbone <vggt checkpoint> --out <dir>
           [--max-refs 5] [--ref-radius 3.0] [--source <prepared sequence folder>]

Per keyframe, one VGGT pass of the keyframe and its nearest keyframes (within `ref_radius` m, facing within 60 deg): the
pass's metric scale comes from the map itself (median ratio of the keyframes' map distances to the predicted ones;
the scale head when no reference is far enough), gravity is one "up" for the whole map (the cameras' mean up axis
refined by the gravity head's estimates of all keyframes, so every BEV shares one ground plane).  The BEVs stay
anchored to their keyframes (`repose` renders the map again from new keyframe poses, e.g. after loop closure) and are
fused by a distance-weighted mean of their logits (cells further than `max_range` m from a keyframe are not used).

Output <out>/navmap.npz: prob (H, W) uint8 (P(occupied) x 254, 255 = no observation), origin (2,), res, R_map (3, 3)
map -> gravity-aligned frame (z up), kf_ids, T_wc (K, 4, 4), bev (K, N, N) uint8, A (K, 3, 3) BEV axes in each
keyframe's camera; navmap.png (free white, occupied black, unobserved blue-grey, keyframes red).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vggt_ft.dataio.transforms import centred_crop_resize, target_shape

from .bev import bev_axes_np
from .labels import rotation_to_z
from .model import NavModel


def _inputs(views, idx, hw):
    imgs = []
    for i in idx:
        v = views[i]
        im = v.image()
        dummy = np.zeros(im.shape[:2], np.float32)
        im, _, _ = centred_crop_resize(im, dummy, v.K, hw, 1.0)
        imgs.append(torch.from_numpy(im).permute(2, 0, 1).float() / 255.0)
    return torch.stack(imgs)[None].cuda()


def references(C, Z, i, max_refs, radius, min_baseline):
    d = np.linalg.norm(C - C[i], axis=1)
    ok = (d >= min_baseline) & (d <= radius) & (Z @ Z[i] > math.cos(math.radians(60)))
    ok[i] = False
    j = np.flatnonzero(ok)
    return [int(x) for x in j[np.argsort(d[j])][:max_refs]]


def geom_bev(bb, ls, A, N, res, band=(0.10, 1.80), floor_tol=0.10):
    """Non-learned BEV of a pass (mode "geom"): VGGT's depth of all its views, metric, in the BEV frame; floor = the
    densest height below the camera (2 cm bins, 0.2-2.5 m below, within 3 m); 254 occupied (>= 2 weighted points in
    the band above the floor), 0 free (floor points, none above), 255 undecided (not fused)."""
    from .evaluate import lift_np, geom_baseline
    from .bev import BEVSpec
    ext = bb["extrinsics"][0].float().cpu().numpy()
    intr = bb["intrinsics"][0].float().cpu().numpy()
    dp = bb["depth"][0].float().cpu().numpy() * math.exp(ls)
    conf = bb["depth_conf"][0].float().cpu().numpy()
    P, Wt = [], []
    for s in range(dp.shape[0]):
        pts, ok = lift_np(dp[s], intr[s], ext[s], A)
        P.append(pts)
        Wt.append(np.clip((conf[s][::2, ::2][ok] - 1) / 20, 0, 1) + 0.05)
    P, Wt = np.concatenate(P), np.concatenate(Wt)
    near = (np.hypot(P[:, 0], P[:, 1]) < 3.0) & (P[:, 2] < -0.2) & (P[:, 2] > -2.5)
    if near.sum() < 50:
        return np.full((N, N), 255, np.uint8)
    hist = np.bincount(np.floor((-P[near, 2]) / 0.02).astype(int), weights=Wt[near])
    cam_h = (int(np.argmax(hist)) + 0.5) * 0.02
    g = geom_baseline(P, Wt, BEVSpec(N, res), cam_h, band, floor_tol)
    out = np.full((N, N), 255, np.uint8)
    out[g == 1] = 0
    out[g == 2] = 254
    return out


@torch.no_grad()
def build(map_path, heads, backbone, out_dir, source=None, max_refs=5, ref_radius=3.0, min_baseline=0.15,
          max_range=8.0, range_scale=4.0, area=262144, mode="head"):
    from cross_world.map_views import load_map_views
    t0 = time.time()
    mv = load_map_views(map_path, source)
    views = mv.views if hasattr(mv, "views") else mv
    ck = torch.load(heads, map_location="cpu", weights_only=False)
    model = NavModel(backbone, ck.get("bev_cfg"), ck.get("gravity_cfg")).cuda().eval()
    model.bev_head.load_state_dict(ck["bev_head"])
    model.gravity_head.load_state_dict(ck["gravity_head"])
    N, res = model.bev_head.size, model.bev_head.res
    T = np.stack([v.T_wc for v in views])
    C, R = T[:, :3, 3], T[:, :3, :3]
    Z = R[:, :, 2]
    v0 = views[0]
    hw = target_shape(v0.height / v0.width, area)
    # 1. one up vector for the map: camera-up prior refined by the gravity head (single-frame passes)
    up0 = -R[:, :, 1].mean(0)
    up0 /= np.linalg.norm(up0)
    ups = []
    for i in range(len(views)):
        bb = model.backbone(_inputs(views, [i], hw))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            u = model.gravity(bb).float()[0].cpu().numpy()
        ups.append(R[i] @ u)
    ups = np.stack(ups)
    up = up0
    for thr in (20.0, 10.0, 5.0):
        sel = ups @ up > math.cos(math.radians(thr))
        if sel.sum() < 3:
            break
        up = ups[sel].mean(0)
        up /= np.linalg.norm(up)
    # 2. one BEV per keyframe
    bevs, As, scales, nrefs = [], [], [], []
    for i in range(len(views)):
        refs = references(C, Z, i, max_refs, ref_radius, min_baseline)
        idx = [i] + refs
        bb = model.backbone(_inputs(views, idx, hw))
        ext = bb["extrinsics"][0].float().cpu().numpy()                     # cam_s from cam_0 (model gauge)
        ratios = []
        for s, j in enumerate(refs, 1):
            c_pred = -ext[s, :3, :3].T @ ext[s, :3, 3]
            c_map = R[i].T @ (C[j] - C[i])
            if np.linalg.norm(c_pred) > 1e-6 and np.linalg.norm(c_map) >= min_baseline:
                ratios.append(np.linalg.norm(c_map) / np.linalg.norm(c_pred))
        ls = math.log(float(np.median(ratios))) if ratios else float(bb["log_scale_head"][0])
        up_cam = R[i].T @ up
        if mode == "geom":
            bevs.append(geom_bev(bb, ls, bev_axes_np(up_cam), N, res))
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit = model.heads(bb, torch.tensor([ls], device="cuda"),
                                    torch.tensor(up_cam, dtype=torch.float32, device="cuda")[None]).float()[0]
            bevs.append(np.round(torch.sigmoid(logit).cpu().numpy() * 254).astype(np.uint8))
        As.append(bev_axes_np(up_cam))
        scales.append(ls)
        nrefs.append(len(ratios))
    out = {"kf_ids": np.array([v.id for v in views]), "T_wc": T,
           "source_index": np.array([-1 if v.source_index is None else v.source_index for v in views]),
           "timestamp": np.array([np.nan if v.timestamp is None else v.timestamp for v in views], np.float64), "bev": np.stack(bevs), "A": np.stack(As),
           "up": up, "res_bev": np.float32(res), "log_scale": np.array(scales), "n_scale_refs": np.array(nrefs)}
    out.update(fuse(out, res, max_range, range_scale))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "navmap.npz", **out)
    cv2.imwrite(str(out_dir / "navmap.png"), render_png(out))
    stats = {"keyframes": len(views), "with_map_scale": int((np.array(nrefs) > 0).sum()), "grid": list(out["prob"].shape),
             "up_vs_camera_prior_deg": round(math.degrees(math.acos(float(np.clip(up @ up0, -1, 1)))), 2),
             "sec": round(time.time() - t0, 1)}
    json.dump(stats, open(out_dir / "navmap.json", "w"), indent=1)
    return out, stats


def fuse(nm: dict, res: float, max_range: float = 8.0, range_scale: float = 4.0, poses: dict | None = None):
    """Fuse the keyframe BEVs (nm from build) on a grid of `res` m in the gravity frame (R_map).  poses: {kf id: T_wc}
    replaces the stored keyframe poses (repose after loop closure)."""
    R_map = rotation_to_z(np.asarray(nm["up"], np.float64))
    N = nm["bev"].shape[-1]
    rb = float(nm["res_bev"])
    i = (np.arange(N) + 0.5 - N / 2) * rb
    X, Y = np.meshgrid(i, -i)                                             # BEV frame: x right, y forward
    dist = np.hypot(X, Y)
    use = dist <= max_range
    w_cell = np.exp(-dist[use] / range_scale)
    pts_b = np.stack([X[use], Y[use], np.zeros(use.sum())], -1)
    Ts = [poses.get(int(k), T) if poses else T for k, T in zip(nm["kf_ids"], nm["T_wc"])]
    xy = []
    for T, A in zip(Ts, nm["A"]):
        p = (T[:3, 3] + (T[:3, :3] @ (A.T @ pts_b.T)).T) @ R_map.T
        xy.append(p[:, :2])
    allxy = np.concatenate(xy)
    lo = allxy.min(0) - res
    hi = allxy.max(0) + res
    W, H = (np.ceil((hi - lo) / res).astype(int) + 1)
    num = np.zeros(H * W)
    den = np.zeros(H * W)
    for p, b in zip(xy, nm["bev"]):
        bv = b[use]
        val = bv != 255                                                  # 255: undecided (geometric BEVs)
        q = np.clip(bv[val].astype(np.float64) / 254, 0.03, 0.97)
        lg = np.log(q / (1 - q))
        c = np.floor((p[val] - lo) / res).astype(int)
        k = c[:, 1] * W + c[:, 0]
        num += np.bincount(k, weights=w_cell[val] * lg, minlength=H * W)
        den += np.bincount(k, weights=w_cell[val], minlength=H * W)
    prob = np.full(H * W, 255, np.uint8)
    m = den > 0
    prob[m] = np.round(254 / (1 + np.exp(-num[m] / den[m]))).astype(np.uint8)
    return {"prob": prob.reshape(H, W), "origin": lo, "res": np.float32(res), "R_map": R_map, "weight": den.reshape(H, W).astype(np.float32)}


def render_png(nm: dict) -> np.ndarray:
    p = nm["prob"]
    img = np.empty(p.shape + (3,), np.uint8)
    g = 255 - (p.astype(np.int32) * 255 // 254)
    img[...] = np.clip(g, 0, 255)[..., None]
    img[p == 255] = (150, 130, 120)
    for T in nm["T_wc"]:
        c = (nm["R_map"] @ T[:3, 3])[:2]
        u, v = ((c - nm["origin"]) / float(nm["res"])).astype(int)
        if 0 <= v < p.shape[0] and 0 <= u < p.shape[1]:
            cv2.circle(img, (int(u), int(v)), 2, (0, 0, 255), -1)
    return img[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source")
    ap.add_argument("--max-refs", type=int, default=5)
    ap.add_argument("--ref-radius", type=float, default=3.0)
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--mode", default="head", choices=["head", "geom"], help="geom: non-learned depth projection")
    a = ap.parse_args()
    _, stats = build(a.map, a.heads, a.backbone, a.out, a.source, a.max_refs, a.ref_radius, max_range=a.max_range,
                     mode=a.mode)
    print(stats)


if __name__ == "__main__":
    main()
