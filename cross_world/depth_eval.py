"""Accuracy of the depth sources against sparse ground truth (e.g. KITTI LiDAR projected into the left camera).

    python -m cross_world.depth_eval --map runs/k07/map.pkl --source data/kitti/07/stereo --gt kitti07_lidar.npz \
        --methods sgbm vggt [--frames 40] [--vggt-checkpoint models/VGGT-Omega/vggt_omega_1b_512.pt]

The ground truth file holds one (N, 3) uint16 array per source frame (`f<index>`: u, v, depth in cm).  Per method:
AbsRel and delta < 1.25 on the ground-truth points the method gives a depth for, coverage (the share of ground-truth
points with a depth), and seconds per view.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cross_world.depth import StereoDepth, VGGTStereoDepth, view_depth
from cross_world.map_views import load_map_views


def score(d: np.ndarray, gt: np.ndarray, max_depth: float = 80.0) -> dict:
    u, v, z = gt[:, 0].astype(int), gt[:, 1].astype(int), gt[:, 2] / 100.0
    keep = (z > 0) & (z <= max_depth)
    u, v, z = u[keep], v[keep], z[keep]
    p = d[v, u]
    has = p > 0
    if not has.any():
        return {"coverage": 0.0}
    r = np.maximum(p[has] / z[has], z[has] / p[has])
    return {"coverage": float(has.mean()), "absrel": float(np.mean(np.abs(p[has] - z[has]) / z[has])),
            "delta1": float(np.mean(r < 1.25)), "rmse": float(np.sqrt(np.mean((p[has] - z[has]) ** 2))),
            "absrel_near20": float(np.mean((np.abs(p[has] - z[has]) / z[has])[z[has] < 20])) if (z[has] < 20).any() else None}


def make_method(name: str, args, width: int):
    if name == "sgbm":
        return "sgbm", {"stereo": StereoDepth(width)}
    if name in ("vggt", "vggt_sgbm", "fused"):
        if not hasattr(args, "_vggt"):
            args._vggt = VGGTStereoDepth(args.vggt_checkpoint, args.device)
        return name, {"vggt": args._vggt, "stereo": StereoDepth(width)}
    if name == "fstereo":
        from cross_world.depth import FoundationStereoDepth
        return "fstereo", {"fstereo": FoundationStereoDepth(args.fstereo_checkpoint, args.device, iters=args.fstereo_iters)}
    raise ValueError(name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--methods", nargs="+", default=["sgbm", "vggt"])
    ap.add_argument("--frames", type=int, default=40, help="evaluate this many keyframes, spread over the map")
    ap.add_argument("--vggt-checkpoint", default="models/VGGT-Omega/vggt_omega_1b_512.pt")
    ap.add_argument("--fstereo-checkpoint", default=None)
    ap.add_argument("--fstereo-iters", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    mv = load_map_views(args.map, args.source)
    gt = np.load(args.gt)
    views = [v for v in mv.views if v.source_index is not None and f"f{v.source_index}" in gt.files]
    views = [views[i] for i in np.linspace(0, len(views) - 1, min(args.frames, len(views))).round().astype(int)]
    res = {}
    for name in args.methods:
        source, kw = make_method(name, args, views[0].width)
        rows, t = [], 0.0
        for v in views:
            t0 = time.perf_counter()
            d = view_depth(v, mv, source, None, **kw)
            t += time.perf_counter() - t0
            rows.append(score(d, gt[f"f{v.source_index}"]))
        keys = [k for k in rows[0] if k != "coverage"] + ["coverage"]
        res[name] = {k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys
                     if any(r.get(k) is not None for r in rows)}
        res[name]["s_per_view"] = t / len(views)
        print(name, {k: round(x, 4) for k, x in res[name].items()}, flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"n_views": len(views), **res}, indent=1))


if __name__ == "__main__":
    main()
