#!/usr/bin/env python3
"""Platt calibration of a checkpoint's covisibility head on held-out training-domain windows, folded into the head.

Windows are drawn by the training sampler from the validation scenes (data.val_mod of the config, split "val") of the
config's datasets; the GT covisibility of every pair is the training target.  p = sigmoid(a * logit + b) is fitted by
soft-target BCE, then folded into the head's last layer (weight * a, a * bias + b), so CROSS loads the calibrated head
unchanged.

  calibrate_covis.py --config vggt_ft/configs/<run>.yaml --ckpt <ckpt.pt> --out <ckpt_cal.pt> [--windows 400]
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from vggt_omega.utils.pose_enc import encoding_to_camera

from vggt_ft import load_yaml
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


def geo_pairs(o, b, gt, lg, hw):
    """(head logit, geometric score reference -> other, symmetric GT) of every ordered pair of one window; the
    geometric score is CROSS's (cross/cv/pose_est_ff.covisibility_scores) on the model's depth and poses."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_covis_reloc import covis  # CROSS's geometric score (the fine-tune env has no pypose for cross)
    E, Ki = encoding_to_camera(o["pose_enc"].float(), hw)
    S = E.shape[1]
    w2c = np.tile(np.eye(4), (S, 1, 1))
    w2c[:, :3] = E[0].cpu().numpy()
    pp = [(u, v) for u in range(S) for v in range(S) if u != v]
    sc = covis(o["depth"][0, ..., 0].float(), o["depth_conf"][0].float(), np.linalg.inv(w2c), Ki[0].cpu().numpy(),
               [p[0] for p in pp], [p[1] for p in pp])
    rows = [(float(lg[u, v]), float(c), float(gt[u, v])) for (u, v), c in zip(pp, sc)]
    return np.array(rows)


def match_geo(G, a):
    """Shift b so that sigmoid(logit + b) >= 0.15 has the geometric gate's false-positive rate (GT < 0.05)."""
    R = np.concatenate(G)
    lg, geo, y = R[:, 0], R[:, 1], R[:, 2]
    neg, pos = y < 0.05, y >= 0.15
    fpr_geo = float(np.mean(geo[neg] >= 0.15))
    t_logit = float(np.quantile(lg[neg], 1 - fpr_geo))           # head logit with the same false-positive rate
    b_ = float(np.log(0.15 / 0.85)) - t_logit
    p = 1 / (1 + np.exp(-(lg + b_)))
    print(f"ordered pairs {len(y)} (neg {neg.sum()}, pos {pos.sum()}); geometric gate: FPR {fpr_geo:.3f} TPR "
          f"{np.mean(geo[pos] >= .15):.3f}; head at the matched threshold (raw p {1 / (1 + np.exp(-t_logit)):.4f}): "
          f"FPR {np.mean(p[neg] >= .15):.3f} TPR {np.mean(p[pos] >= .15):.3f}; shift b {b_:.4f}")
    if a.out:
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        m = sd["model"] if "model" in sd else sd
        bias = "covis_head.pair.2.bias"
        m[bias] = (m[bias].float() + b_).to(m[bias].dtype)
        if "model" in sd:
            sd["covis_platt"] = {"a": 1.0, "b": b_, "target": "match_geo_fpr", "fpr": fpr_geo}
        torch.save(sd, a.out)
        print("saved", a.out)


def isotonic(x, y, n_bins: int = 200):
    """Monotone (non-decreasing) fit of E[y | x]: means over quantile bins of x, pooled where they decrease
    (pool-adjacent-violators, weighted by the bin counts). Returns knots (x, y) for np.interp."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    xs = np.array([x[idx == k].mean() for k in range(len(edges) - 1) if (idx == k).any()])
    ys = np.array([y[idx == k].mean() for k in range(len(edges) - 1) if (idx == k).any()])
    ws = np.array([(idx == k).sum() for k in range(len(edges) - 1) if (idx == k).any()], float)
    blocks = []                                   # [weighted mean, weight, x sum (weighted)]
    for xi, yi, wi in zip(xs, ys, ws):
        blocks.append([yi, wi, xi * wi])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            y2, w2, x2 = blocks.pop(); y1, w1, x1 = blocks.pop()
            blocks.append([(y1 * w1 + y2 * w2) / (w1 + w2), w1 + w2, x1 + x2])
    kx = np.array([b[2] / b[1] for b in blocks]); ky = np.array([b[0] for b in blocks])
    return np.concatenate([[0.0], kx, [1.0]]), np.concatenate([[0.0], ky, [ky[-1]]])   # 0 -> 0, flat above the last knot


def map_geo(G, a):
    """covis_head_map: the raw head probability -> the expected one-way geometric score (CROSS's gate scale, the
    scale its thresholds and fix weights were tuned on), fitted by isotonic regression on the held-out windows."""
    R = np.concatenate(G)
    lg, geo, y = R[:, 0], R[:, 1], R[:, 2]
    p = 1 / (1 + np.exp(-lg))
    kx, ky = isotonic(p, geo)
    m = np.interp(p, kx, ky)
    neg, pos = y < 0.05, y >= 0.15
    print(f"ordered pairs {len(y)} (neg {neg.sum()}, pos {pos.sum()}), {len(kx)} knots")
    for name, s in (("geometric", geo), ("head raw", p), ("head mapped", m)):
        print(f"  {name:12s} TPR@.15 {np.mean(s[pos] >= .15):.3f} FPR@.15 {np.mean(s[neg] >= .15):.3f}  "
              f"quantiles 25/50/75/90 {np.round(np.quantile(s, [.25, .5, .75, .9]), 3).tolist()}")
    if a.out:
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        sd["covis_head_map"] = {"x": kx.tolist(), "y": ky.tolist(), "target": "one-way geometric score (CROSS gate)",
                                "windows": a.windows, "config": a.config}
        torch.save(sd, a.out)
        print("saved", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--match_geo", action="store_true",
                    help="instead of Platt: shift the head so that its 0.15 gate has the false-positive rate of the "
                         "model's own geometric score at 0.15 (CROSS's default gate) on the same windows (a = 1)")
    ap.add_argument("--map_geo", action="store_true",
                    help="store covis_head_map in the checkpoint: an isotonic map from the raw head probability to the "
                         "expected one-way geometric score on the same windows (the head's weights are unchanged)")
    ap.add_argument("--target", choices=["symmetric", "oneway"], default="symmetric",
                    help="symmetric: the head's training target (min of both directions); oneway: the overlap of one view "
                         "in the other (CROSS's geometric gate measures reference -> current), both directions per pair")
    a = ap.parse_args()
    cfg = load_yaml(a.config)
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
    L, Y, G = [], [], []
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
            o = model(b["images"], need_depth=a.match_geo or a.map_geo)
            gt = gt_covisibility(b["depths"], b["masks"], b["extrinsics"], b["intrinsics"], b["world_id"],
                                 symmetric=a.target == "symmetric")
        iu = torch.triu_indices(S, S, 1)
        lgw = o["covis_logits"][0].float().cpu()
        if a.match_geo or a.map_geo:
            G.append(geo_pairs(o, b, gt[0], lgw, hw))
        L.append(lgw[iu[0], iu[1]])
        Y.append(gt[0][iu[0], iu[1]].float().cpu())
        if a.target == "oneway":           # the other direction of each pair, same (symmetric) logit
            L.append(lgw[iu[1], iu[0]])
            Y.append(gt[0][iu[1], iu[0]].float().cpu())
        done += 1
        if done % 50 == 0:
            print(f"{done} windows, {sum(len(x) for x in L)} pairs", flush=True)
    if a.map_geo:
        return map_geo(G, a)
    if a.match_geo:
        return match_geo(G, a)
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
