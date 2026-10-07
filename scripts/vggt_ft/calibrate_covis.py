#!/usr/bin/env python3
"""Platt calibration of a checkpoint's covisibility head on held-out training-domain windows, folded into the head.

Windows are drawn by the training sampler from the validation scenes (data.val_mod of the config, split "val") of the
config's datasets; the GT covisibility of every pair is the training target.  p = sigmoid(a * logit + b) is fitted by
soft-target BCE, then folded into the head's last layer (weight * a, a * bias + b), so CROSS loads the calibrated head
unchanged.

  calibrate_covis.py --config configs/vggt_ft/<run>.yaml --ckpt <ckpt.pt> --out <ckpt_cal.pt> [--windows 400]
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from vggt_ft.dataio.transforms import target_shape
from vggt_ft.dataio.windows import DatasetPool, WindowSampler, materialise
from vggt_ft.evaluate import load_model
from vggt_ft.geometry import gt_covisibility


def bins(p, y):
    out = []
    for lo, hi in ((0, .05), (.05, .15), (.15, .3), (.3, .5), (.5, 1.01)):
        m = (y >= lo) & (y < hi)
        if m.any():
            out.append(f"[{lo:.2f},{hi:.2f}) n={m.sum()} p {p[m].mean():.3f} gt {y[m].mean():.3f} tpr@.15 {np.mean(p[m] >= .15):.2f}")
    return "  ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--target", choices=["symmetric", "oneway"], default="symmetric",
                    help="symmetric: the head's training target (min of both directions); oneway: the overlap of one view "
                         "in the other (CROSS's geometric gate measures reference -> current), both directions per pair")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dc = cfg["data"]
    pools = []
    for n, c in dc["datasets"].items():
        if c.get("weight", 1.0) <= 0:
            continue
        try:
            pools.append(DatasetPool(dc["root"], n, {"val_mod": dc.get("val_mod", 20), "split": "val", **c}, a.seed))
        except (RuntimeError, FileNotFoundError) as e:
            print(f"skip {n}: {e}")
    sampler = WindowSampler(pools)
    model = load_model(a.ckpt)
    rng = np.random.default_rng(a.seed)
    L, Y = [], []
    lo, hi = max(3, dc["frames"][0]), dc["frames"][1]
    done = 0
    while done < a.windows:
        S = int(rng.integers(lo, hi + 1))
        hw = target_shape(float(rng.uniform(*dc["aspect"])), dc.get("area", 512 * 512))
        try:
            pool, frames, negs, kind = sampler.sample(rng, S)
            it = materialise(frames, hw, None, None, (1.0, 1.0))
        except (OSError, ValueError, RuntimeError) as e:
            print("skip window:", e)
            continue
        b = {k: v[None].cuda() for k, v in it.items() if torch.is_tensor(v)}
        with torch.no_grad():
            o = model(b["images"], need_depth=False)
            gt = gt_covisibility(b["depths"], b["masks"], b["extrinsics"], b["intrinsics"], b["world_id"],
                                 symmetric=a.target == "symmetric")
        iu = torch.triu_indices(S, S, 1)
        lgw = o["covis_logits"][0].float().cpu()
        L.append(lgw[iu[0], iu[1]])
        Y.append(gt[0][iu[0], iu[1]].float().cpu())
        if a.target == "oneway":           # the other direction of each pair, same (symmetric) logit
            L.append(lgw[iu[1], iu[0]])
            Y.append(gt[0][iu[1], iu[0]].float().cpu())
        done += 1
        if done % 50 == 0:
            print(f"{done} windows, {sum(len(x) for x in L)} pairs", flush=True)
    lg, y = torch.cat(L), torch.cat(Y)
    ab = torch.tensor([1.0, 0.0], requires_grad=True)
    opt = torch.optim.LBFGS([ab], lr=0.5, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(ab[0] * lg + ab[1], y)
        loss.backward()
        return loss
    opt.step(closure)
    a_, b_ = float(ab[0]), float(ab[1])
    p0, p1 = torch.sigmoid(lg).numpy(), torch.sigmoid(a_ * lg + b_).numpy()
    print(f"pairs {len(y)}; Platt a {a_:.4f} b {b_:.4f}")
    print("before:", bins(p0, y.numpy()))
    print("after: ", bins(p1, y.numpy()))
    if a.out:
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        m = sd["model"] if "model" in sd else sd
        w, bias = "covis_head.pair.2.weight", "covis_head.pair.2.bias"
        dt = m[w].dtype
        m[w] = (m[w].float() * a_).to(dt)
        m[bias] = (m[bias].float() * a_ + b_).to(dt)
        if "model" in sd:
            sd["covis_platt"] = {"a": a_, "b": b_, "target": a.target}
        torch.save(sd, a.out)
        print("saved", a.out)


if __name__ == "__main__":
    main()
