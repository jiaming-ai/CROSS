#!/usr/bin/env python3
"""Observation-step cost of the feed-forward estimator as a function of the number of views.

Times PoseEstFeedForward.estimate_pose (VGGT-Omega forward pass + scale + covisibility) on real frames of a
sequence for several numbers of retrieved references and stored right-image anchors.

usage: python scripts/lc/bench_ff_views.py --seq data/sim/lonemonk/map_loop --baseline 0.3 --n-refs 1 2 4 6 8 --n-ref-anchors 0 2
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross.core.config import FeedForwardConfig, FFBackend  # noqa: E402
from cross.core.types import Camera  # noqa: E402
from cross.cv.pose_est_ff import PoseEstFeedForward  # noqa: E402
from cross.dataloader.stereo_loader import StereoSequenceLoader  # noqa: E402
from cross.utils.camera import get_transforms_ff  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--baseline", type=float, default=None)
    ap.add_argument("--n-refs", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--n-ref-anchors", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ds = StereoSequenceLoader(args.seq, baseline=args.baseline)
    camera = Camera(K=ds.rgb_K.copy(), frame_width=ds.rgb_width, frame_height=ds.rgb_height)
    tf, _ = get_transforms_ff(camera, image_resolution=args.resolution)
    idx = list(range(0, min(len(ds), 400), max(1, min(len(ds), 400) // 12)))[:12]
    items = [ds[i] for i in idx]
    lefts = [tf(it["rgb"]) for it in items]
    rights = [tf(it["rgb_right"]) for it in items]
    cfg = FeedForwardConfig(); cfg.image_resolution = args.resolution
    pe = PoseEstFeedForward("cuda", cfg, ds.T_right_in_left)
    res = []
    for n_ref in args.n_refs:
        for n_anch in args.n_ref_anchors:
            cfg.n_ref_anchors = n_anch
            refs = torch.stack(lefts[1:1 + n_ref]); rr = rights[1:1 + n_ref]
            for _ in range(2):
                pe.estimate_pose(refs, None, lefts[0], None, curr_image_right=rights[0], ref_images_right=rr)
            torch.cuda.synchronize()
            ts, tm = [], []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                pe.estimate_pose(refs, None, lefts[0], None, curr_image_right=rights[0], ref_images_right=rr)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0); tm.append(pe.last_info.get("t_model", float("nan")))
            n_views = pe.last_info.get("n_views")
            r = {"n_refs": n_ref, "n_ref_anchors": n_anch, "n_views": n_views, "hw": list(pe.last_info.get("view_tags", [])) and None,
                 "total_ms_median": 1e3 * float(np.median(ts)), "model_ms_median": 1e3 * float(np.median(tm))}
            res.append(r)
            print(f"refs={n_ref} ref_anchors={n_anch} views={n_views}: total {r['total_ms_median']:.0f} ms, model {r['model_ms_median']:.0f} ms")
    print("image size", tuple(lefts[0].shape[-2:]), "GPU", torch.cuda.get_device_name(0))
    if args.out:
        Path(args.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "image_hw": list(lefts[0].shape[-2:]), "results": res}, indent=1))


if __name__ == "__main__":
    main()
