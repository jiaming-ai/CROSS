"""Consistency of rendered depth with the mesh BEV label (convention check of poses, depth and label frame).

    python -m cross_nav.sim.check_render --cache <root> --dataset hssd_nav --bev <bev root> [--scenes ...] [--n 12]
        [--png-dir d] [--json out.json]

Depth points of `n` frames per scene are lifted to the world frame: points between ground + 0.15 and + 1.7 m must
fall on occupied cells, floor points (|z - ground| < 3 cm on reached ground) on free cells, and the camera height
above the label's ground must match the planned height.  A wrong axis convention, depth type (ray length vs z) or
grid offset shows up as low fractions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


def lift(dep, K, E, step=4, max_range=10.0):
    H, W = dep.shape
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    d = dep[ys, xs].astype(np.float64)
    ok = (d > 0) & (d <= max_range)
    P = np.stack([(xs - K[0, 2]) / K[0, 0] * d, (ys - K[1, 2]) / K[1, 1] * d, d], -1)[ok]
    return (P - E[:3, 3]) @ E[:3, :3]


def check_scene(L: dict, frames, png=None):
    bev, o, r = L["bev"], L["origin"], float(L["res"])
    g = L["ground"].astype(np.float32)
    H, W = bev.shape
    reached = np.isfinite(g)
    _, (ny, nx) = ndimage.distance_transform_edt(~reached, return_indices=True)
    gi = g[ny, nx]
    img = np.full(bev.shape + (3,), 128, np.uint8)
    img[bev == 1] = 255
    img[bev == 2] = 0
    n = dict(obs=0, obs_occ=0, fl=0, fl_free=0, obs_unknown=0)
    for dep, K, E in frames:
        Pw = lift(dep, K, E)
        ix = np.floor((Pw[:, 0] - o[0]) / r).astype(int)
        iy = np.floor((Pw[:, 1] - o[1]) / r).astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ix, iy, z = ix[ok], iy[ok], Pw[ok, 2]
        dz = z - gi[iy, ix]
        obs = (dz > 0.15) & (dz < 1.7)
        fl = (np.abs(dz) < 0.03) & reached[iy, ix]
        n["obs"] += int(obs.sum())
        n["obs_occ"] += int((bev[iy[obs], ix[obs]] == 2).sum())
        n["obs_unknown"] += int((bev[iy[obs], ix[obs]] == 0).sum())
        n["fl"] += int(fl.sum())
        n["fl_free"] += int((bev[iy[fl], ix[fl]] == 1).sum())
        img[iy[obs], ix[obs]] = (0, 0, 255)
        img[iy[fl], ix[fl]] = (0, 190, 0)
    res = dict(frames=len(frames), obstacle_pts=n["obs"], obstacle_on_occ=round(n["obs_occ"] / max(n["obs"], 1), 4),
               obstacle_on_unknown=round(n["obs_unknown"] / max(n["obs"], 1), 4), floor_pts=n["fl"],
               floor_on_free=round(n["fl_free"] / max(n["fl"], 1), 4))
    if png:
        cv2.imwrite(str(png), cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)[::-1])
    return res


def main():
    from vggt_ft.dataio.scene import list_scenes, load_scene
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/mnt/datasets-livsyn/jz/cache")
    ap.add_argument("--dataset", default="hssd_nav")
    ap.add_argument("--bev", default="/mnt/datasets-livsyn/jz/nav/bev")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--png-dir", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}
    for d in list_scenes(Path(a.cache), a.dataset):
        if a.scenes and d.name not in a.scenes:
            continue
        lp = Path(a.bev) / a.dataset / f"{d.name}.npz"
        if not lp.exists():
            continue
        L = dict(np.load(lp))
        sc = load_scene(d)
        frames, camh = [], []
        for q in sc.sequences:
            sq = sc.seq(q)
            for i in np.linspace(0, sq.n - 1, max(1, a.n // len(sc.sequences))).round().astype(int):
                _, dep, K, E = sq.load(int(i))
                frames.append((dep, K, E))
            camh.append(L.get(f"cam_h/{d.name}/{q}"))
        png = Path(a.png_dir) / f"{d.name}_depthcheck.png" if a.png_dir else None
        if png:
            png.parent.mkdir(parents=True, exist_ok=True)
        r = check_scene(L, frames, png)
        ch = np.concatenate([c for c in camh if c is not None]) if any(c is not None for c in camh) else np.zeros(0)
        r["cam_h_ok"] = float(np.isfinite(ch).mean()) if len(ch) else 0.0
        out[d.name] = r
        print(d.name, json.dumps(r), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
