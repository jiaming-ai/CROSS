"""Check the CROSS integration of the BEV head: the same windows through PoseEstFeedForward (pose_est.ff.nav_bev_head,
scale from the scale head, eager and CUDA-graph paths) and through cross_nav.model.NavModel must give the same BEV.

    python scripts/nav_cross_check.py --windows <windows.json> --heads <heads.pt> --checkpoint <vggt ckpt> [--n 5]
"""
import argparse
import json
import time

import numpy as np
import torch

from cross.core.config import FeedForwardConfig
from cross.cv.pose_est_ff import PoseEstFeedForward
from cross_nav.data import NavStream, WorldBEVs
from cross_nav.evaluate import load_window
from cross_nav.model import NavModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    W = json.load(open(a.windows))
    st = NavStream({"bev_root": W["bev_root"], "datasets": {}, "cam_h_range": W.get("cam_h_range") or [0.1, 2.2]})
    st.bevs, st.label_ds = WorldBEVs(W["bev_root"]), {}
    scenes = {}
    items = []
    for w in W["windows"]:
        if len(w["frames"]) < 2:
            continue
        it = load_window(w, st, scenes)
        if it is not None:
            items.append(it)
        if len(items) >= a.n:
            break
    ck = torch.load(a.heads, map_location="cpu", weights_only=False)
    ref = NavModel(a.checkpoint, ck.get("bev_cfg"), ck.get("gravity_cfg")).cuda().eval()
    ref.bev_head.load_state_dict(ck["bev_head"])
    ref.gravity_head.load_state_dict(ck["gravity_head"])
    for graphs in (False, True):
        cfg = FeedForwardConfig(checkpoint=a.checkpoint, nav_bev_head=a.heads, scale_source="head", token_cache=64 if graphs else 0,
                                cuda_graphs=graphs, compile=False, quantize_input=False, principal_point_correction=False)
        est = PoseEstFeedForward("cuda", cfg)
        for k, it in enumerate(items):
            imgs = it["images"].cuda()
            t0 = time.perf_counter()
            est.estimate_pose(imgs[1:], None, imgs[0], None)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            bev = est.last_bev
            if bev is None:
                print("no BEV (scale invalid)", est.last_info.get("scale"))
                continue
            with torch.no_grad():
                bb = ref.backbone(imgs[None])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    up = ref.gravity(bb).float()
                    p = torch.sigmoid(ref.heads(bb, torch.tensor([bev["log_scale"]], device="cuda"), up).float())[0]
            p = p.cpu().numpy()
            q = bev["prob"].astype(np.float32)
            agree = float(((p > 0.5) == (q > 0.5)).mean())
            print(f"graphs={graphs} window {k}: S={imgs.shape[0]} log_scale {bev['log_scale']:.3f} (ref head "
                  f"{float(bb['log_scale_head'][0]):.3f}) | prob |diff| mean {np.abs(p - q).mean():.4f} max "
                  f"{np.abs(p - q).max():.3f}, class agreement {agree:.4f}, up {np.round(bev['up_cam'], 3)} vs "
                  f"{np.round(up[0].cpu().numpy(), 3)}, pass {dt * 1e3:.0f} ms")
        del est
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
