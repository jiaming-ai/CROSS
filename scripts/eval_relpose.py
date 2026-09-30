#!/usr/bin/env python3
"""Module-level evaluation of relative pose estimators against ground truth.

For every query frame, a set of reference frames is chosen (frame gaps simulate
viewpoint change; a different scene condition of the same trajectory — vKITTI2
fog/rain/sunset/15-deg-left ... — simulates lighting/appearance/viewpoint change).
The estimator receives the references as a batch, exactly like inside CROSS, and its
outputs are compared with the ground-truth T_ref_cam.

Usage:
  python scripts/eval_relpose.py --ref data/vkitti2/Scene01/clone --query data/vkitti2/Scene01/sunset \
      --estimator ff --backend vggt_omega --gaps 0,5,10,20 --n-queries 40 --out outputs/relpose/vk_sunset_ff.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from cross.core.config import FeedForwardConfig, FFBackend, PoseEstConfig
from cross.core.types import Camera
from cross.cv.stereo_scale import invert_poses, rotation_angle_deg
from cross.dataloader.stereo_loader import StereoSequenceLoader


def build_estimator(args, ds):
    cam = Camera(K=ds.rgb_K.copy(), frame_width=ds.rgb_width, frame_height=ds.rgb_height)
    if args.estimator == "ff":
        from cross.cv.pose_est_ff import PoseEstFeedForward
        from cross.utils.camera import get_transforms_ff
        rgb_tf, depth_tf = get_transforms_ff(cam, args.resolution)
        ckpt = args.checkpoint or ("models/VGGT-Omega/vggt_omega_1b_512.pt" if args.backend == "vggt_omega" else "models/DA3-LARGE-1.1")
        cfg = FeedForwardConfig(
            backend=FFBackend(args.backend), checkpoint=ckpt, image_resolution=args.resolution,
            n_ref_anchors=args.n_ref_anchors, scale_method=args.scale_method, min_covis=args.min_covis,
            use_odom_anchor=False,
        )
        est = PoseEstFeedForward("cuda", cfg, ds.T_right_in_left)
        return est, rgb_tf, depth_tf, cam
    elif args.estimator == "pnp":
        from cross.cv.pose_est_pnp import PoseEstPnP
        from cross.utils.camera import get_transforms_target_max
        rgb_tf, depth_tf = get_transforms_target_max(cam)
        cfg = PoseEstConfig()
        cfg.kp_detector.n_keypoints = args.n_keypoints
        est = PoseEstPnP("cuda", cfg, cam)
        return est, rgb_tf, depth_tf, cam
    raise ValueError(args.estimator)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="sequence providing reference (map) frames")
    ap.add_argument("--query", default=None, help="sequence providing query frames (default: same as --ref)")
    ap.add_argument("--estimator", choices=["ff", "pnp"], default="ff")
    ap.add_argument("--backend", choices=["vggt_omega", "da3"], default="vggt_omega")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--n-ref-anchors", type=int, default=2)
    ap.add_argument("--no-curr-anchor", action="store_true", help="drop the current stereo pair (ablation)")
    ap.add_argument("--scale-method", default="adaptive")
    ap.add_argument("--min-covis", type=float, default=0.15)
    ap.add_argument("--n-keypoints", type=int, default=300)
    ap.add_argument("--gaps", default="0,5,10,20", help="reference frame offsets relative to the query")
    ap.add_argument("--n-queries", type=int, default=40)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds_ref = StereoSequenceLoader(args.ref, depth_source="sgbm" if args.estimator == "pnp" else "none")
    ds_q = ds_ref if args.query in (None, args.ref) else StereoSequenceLoader(args.query, depth_source="none")
    assert len(ds_ref) == len(ds_q), "ref/query sequences must be frame-aligned"
    est, rgb_tf, depth_tf, cam = build_estimator(args, ds_ref)

    gaps = [int(g) for g in args.gaps.split(",")]
    rng = np.random.default_rng(args.seed)
    end = args.end or len(ds_q)
    lo, hi = args.start + max(0, -min(gaps)), end - max(0, max(gaps)) - 1
    queries = np.sort(rng.choice(np.arange(lo, hi), size=min(args.n_queries, hi - lo), replace=False))

    records = []
    ref_cache = {}

    def load_ref(i):
        if i not in ref_cache:
            d = ds_ref[i]
            depth = None
            if d["depth"] is not None:
                depth = depth_tf(torch.from_numpy(d["depth"]).float().unsqueeze(0))
            ref_cache[i] = (rgb_tf(d["rgb"]), rgb_tf(d["rgb_right"]), depth)
        return ref_cache[i]

    t_warm = None
    for qi, q in enumerate(queries):
        dq = ds_q[int(q)]
        cl, cr = rgb_tf(dq["rgb"]), rgb_tf(dq["rgb_right"])
        cd = depth_tf(torch.from_numpy(dq["depth"]).float().unsqueeze(0)) if dq["depth"] is not None else None
        refs = [int(q) - g for g in gaps]
        ref_items = [load_ref(r) for r in refs]
        ref_l = torch.stack([it[0] for it in ref_items])
        ref_r = [it[1] for it in ref_items]
        ref_d = torch.cat([it[2] for it in ref_items], 0) if ref_items[0][2] is not None else None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if args.estimator == "ff":
            poses, valid, conf = est.estimate_pose(
                ref_l, None, cl, None,
                curr_image_right=None if args.no_curr_anchor else cr, ref_images_right=ref_r,
            )
            info = est.last_info
        else:
            if cd is None:  # PnP only uses reference depth (2D-3D from the map side)
                cd = torch.zeros_like(ref_d[0:1])
            poses, valid, conf = est.estimate_pose(ref_l, ref_d, cl, cd)
            info = {}
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if qi == 0:
            t_warm = dt
        k = 0
        for j, (r, g) in enumerate(zip(refs, gaps)):
            T_gt = invert_poses(ds_ref.left_c2w[r]) @ ds_q.left_c2w[int(q)]
            rec = {"query": int(q), "ref": int(r), "gap": int(g), "valid": bool(valid[j]),
                   "gt_dist": float(np.linalg.norm(T_gt[:3, 3])), "time": dt if j == 0 else None}
            if valid[j]:
                T = poses[k].matrix().cpu().numpy().astype(np.float64)
                k += 1
                err = invert_poses(T_gt) @ T
                rec.update({
                    "t_err": float(np.linalg.norm(err[:3, 3])),
                    "r_err": float(rotation_angle_deg(err[:3, :3])),
                    "conf": float(conf[k - 1]),
                    "est_dist": float(np.linalg.norm(T[:3, 3])),
                })
            if args.estimator == "ff" and info.get("valid"):
                rec["scale"] = info["scale"]["scale"]
                rec["scale_logstd"] = info["scale"]["log_std"]
                rec["n_anchors_used"] = info["scale"]["n_used"]
                rec["covis"] = info["covis"][j]
            records.append(rec)
        if qi % 10 == 0:
            logger.info(f"{qi + 1}/{len(queries)} queries, last {dt * 1e3:.0f} ms")

    # ---- summary ----
    summary = {}
    for g in gaps:
        rs = [r for r in records if r["gap"] == g]
        ok = [r for r in rs if r["valid"]]
        te = np.array([r["t_err"] for r in ok]) if ok else np.zeros(0)
        re_ = np.array([r["r_err"] for r in ok]) if ok else np.zeros(0)
        n = len(rs)
        summary[str(g)] = {
            "n": n,
            "valid_rate": len(ok) / max(n, 1),
            "t_err_median": float(np.median(te)) if len(te) else None,
            "r_err_median": float(np.median(re_)) if len(re_) else None,
            "t_err_mean": float(np.mean(te)) if len(te) else None,
            "success_0.5m_5deg": float(np.mean((te < 0.5) & (re_ < 5))) * len(ok) / max(n, 1) if len(te) else 0.0,
            "success_1m_10deg": float(np.mean((te < 1.0) & (re_ < 10))) * len(ok) / max(n, 1) if len(te) else 0.0,
            "success_0.25m_2deg": float(np.mean((te < 0.25) & (re_ < 2))) * len(ok) / max(n, 1) if len(te) else 0.0,
            "gt_dist_median": float(np.median([r["gt_dist"] for r in rs])),
        }
    times = [r["time"] for r in records if r["time"] is not None][1:]
    out = {
        "args": vars(args), "summary": summary, "records": records,
        "time_median_s": float(np.median(times)) if times else None, "time_first_s": t_warm,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    logger.info(json.dumps(summary, indent=1))
    logger.info(f"median time {out['time_median_s']:.3f}s -> {args.out}")


if __name__ == "__main__":
    main()
