"""Held-out evaluation of a BEV head on fixed windows.

    # once: fixed windows of the validation scenes (val_mod holdout) / of a test cache
    python -m cross_nav.evaluate make --config <run.yaml> --out windows.json [--per-dataset 150] [--split val]
           [--root <cache>] [--bev-root <bev labels>] [--datasets openloris simchange]
    # per checkpoint
    python -m cross_nav.evaluate run --windows windows.json --heads <heads_XXXXXX_ema.pt> --out result.json
           [--figs <dir>] [--n-figs 6]

Conditions (what the pass knows about metric scale and gravity, as in CROSS's modes):
  oracle  GT scale, GT gravity                              (upper bound of the head)
  stereo  GT scale x log-normal 3 %, gravity head            (stereo without IMU: scale from the baseline)
  mono    scale head, gravity head                           (single camera without odometry or IMU)
Baselines without learning (on the same windows, GT scale / gravity and the GT floor height):
  geom    VGGT's predicted depth of all window frames, occupied = >= 2 points in [floor+0.1, floor+1.8] m, free =
          points within 0.1 m of the floor and none above; undecided cells count as errors
  gtgeom  the same with the windows' GT depth (what the frames show with perfect depth, no priors)
Metrics over labelled cells: IoU occupied / free, mean IoU, accuracy, coverage (decided cells), split into cells seen
by the window (a GT depth point of a window frame) and unseen, and by distance (< 3, 3-6, > 6 m).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from vggt_ft.dataio.scene import load_scene
from vggt_ft.dataio.transforms import target_shape

from .bev import FREE, OCC, BEVSpec, bev_axes
from .data import NavStream
from .model import NavModel

RANGES = ((0.0, 3.0), (3.0, 6.0), (6.0, 1e9))


# ---------------------------------------------------------------------------------------------------------- windows
def make_windows(cfg, out, per_dataset, split, root=None, bev_root=None, datasets=None, seed=123):
    dc = dict(cfg["data"])
    if root:
        dc["root"] = root
    if bev_root:
        dc["bev_root"] = bev_root
    if datasets:
        # "name" or "name=label_folder" (labels under another folder name, e.g. tartanground=tartanground_mix)
        base = next(iter(dc["datasets"].values()))
        sel = {}
        for d in datasets:
            name, _, lab = d.partition("=")
            c = dict(dc["datasets"].get(name, {**base, "weight": 1.0, "p_cross": 0.3, "stride": [1, 4]}))
            if lab:
                c["label_dataset"] = lab
            sel[name] = c
        dc["datasets"] = sel
    if split == "test":
        dc["val_mod"] = 0
    st = NavStream(dc, seed=seed, split="val" if split == "val" else "train")
    st._build()
    rng = np.random.default_rng(seed)
    hw = target_shape(0.75, dc.get("area", 512 * 512))
    wins = []
    for pool in st.sampler.pools:
        sub = type(st.sampler)([pool])
        k, tries = 0, 0
        while k < per_dataset and tries < per_dataset * 20:
            tries += 1
            S = (1, 4, 8)[k % 3]
            try:
                pool_, frames, negs, kind = sub.sample(rng, S)
            except RuntimeError:
                continue
            r = st._frame0(rng, pool_, frames)
            if r is None:
                continue
            frames = r[0]
            wins.append({"dataset": pool.name, "label_dataset": st.label_ds[pool.name], "kind": kind,
                         "frames": [[str(sc.dir), sq.name, int(i)] for sc, sq, i in frames], "hw": list(hw)})
            k += 1
        print(f"{pool.name}: {k} windows", flush=True)
    json.dump({"bev_root": dc["bev_root"], "bev": dc.get("bev", {}), "cam_h_range": dc.get("cam_h_range"),
               "windows": wins}, open(out, "w"))


def load_window(w, stream, scenes):
    frames = []
    for d, q, i in w["frames"]:
        sc = scenes.get(d)
        if sc is None:
            sc = scenes[d] = load_scene(Path(d))
        frames.append((sc, sc.seq(q), i))

    class P:                                             # minimal pool for make_item
        name = w["dataset"]
        cfg = {"zoom": (1.0, 1.0)}
    stream.label_ds[P.name] = w["label_dataset"]
    return stream.make_item(np.random.default_rng(0), P, frames, [], w["kind"], tuple(w["hw"]), aug=False)


# ---------------------------------------------------------------------------------------------------------- baselines
def geom_baseline(pts, w, spec: BEVSpec, cam_h, band=(0.10, 1.80), floor_tol=0.10, min_occ=2.0):
    """pts (M,3) BEV-frame points (heights relative to the camera), w (M,) weights -> (N,N) prediction:
    0 undecided, 1 free, 2 occupied."""
    N = spec.size
    col = np.floor(pts[:, 0] / spec.res + N / 2).astype(np.int64)
    row = np.floor(N / 2 - pts[:, 1] / spec.res).astype(np.int64)
    ok = (col >= 0) & (col < N) & (row >= 0) & (row < N) & (w > 0)
    hz = pts[:, 2] + cam_h
    occ = ok & (hz >= band[0]) & (hz <= band[1])
    flo = ok & (np.abs(hz) <= floor_tol)
    a = np.bincount(row[occ] * N + col[occ], weights=w[occ], minlength=N * N)
    f = np.bincount(row[flo] * N + col[flo], weights=w[flo], minlength=N * N)
    out = np.zeros(N * N, np.uint8)
    out[(f > 0) & (a < min_occ)] = FREE
    out[a >= min_occ] = OCC
    return out.reshape(N, N)


def lift_np(depth, K, E_c_from_0, A):
    """depth (H,W) -> BEV-frame points of a frame with camera-from-frame-0 E (3x4 or 4x4) and BEV-from-frame-0 A."""
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:2, 0:W:2]
    d = depth[ys, xs]
    ok = d > 0
    P = np.stack([(xs[ok] + 0.5 - K[0, 2]) / K[0, 0] * d[ok], (ys[ok] + 0.5 - K[1, 2]) / K[1, 1] * d[ok], d[ok]], -1)
    P0 = (P - E_c_from_0[:3, 3]) @ E_c_from_0[:3, :3]
    return P0 @ A.T, ok


# ---------------------------------------------------------------------------------------------------------- metrics
class Meter:
    def __init__(self):
        self.c = defaultdict(lambda: np.zeros(5))     # tp, fp, fn, tn, undecided (occupied = positive)

    def add(self, key, pred, gt, sel):
        lab = ((gt == FREE) | (gt == OCC)) & sel
        po, pf = pred == OCC, pred == FREE
        go = gt == OCC
        self.c[key] += [(po & go & lab).sum(), (po & ~go & lab).sum(), (pf & go & lab).sum(), (pf & ~go & lab).sum(),
                        (~po & ~pf & lab).sum()]

    def result(self):
        out = {}
        for k, (tp, fp, fn, tn, und) in self.c.items():
            n = tp + fp + fn + tn + und
            if n == 0:
                continue
            # undecided cells are errors for both classes: an occupied cell missed (fn), a free cell missed (fp of free)
            go, gf = None, None
            iou_o = tp / max(tp + fp + fn + und, 1)
            iou_f = tn / max(tn + fp + fn + und, 1)
            out[k] = {"iou_occ": round(float(iou_o), 4), "iou_free": round(float(iou_f), 4),
                      "miou": round(float((iou_o + iou_f) / 2), 4), "acc": round(float((tp + tn) / n), 4),
                      "coverage": round(float(1 - und / n), 4), "cells": int(n)}
        return out


def colour(bev):
    img = np.full(bev.shape + (3,), 128, np.uint8)
    img[bev == FREE] = 255
    img[bev == OCC] = 0
    return img


def prob_rgb(p, gt):
    img = (np.clip(1 - p, 0, 1) * 255).astype(np.uint8)
    img = np.repeat(img[..., None], 3, -1)
    img[gt == 0] = (img[gt == 0] * 0.7 + np.array([90, 60, 60]) * 0.3).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------------------------------------- run
@torch.no_grad()
def run(windows_path, heads_path, out, figs=None, n_figs=6, backbone=None, conditions=("oracle", "stereo", "mono"),
        baselines=True, limit=0):
    W = json.load(open(windows_path))
    ck = torch.load(heads_path, map_location="cpu", weights_only=False)
    model = NavModel(backbone or ck["backbone"], ck.get("bev_cfg"), ck.get("gravity_cfg")).cuda().eval()
    model.bev_head.load_state_dict(ck["bev_head"])
    model.gravity_head.load_state_dict(ck["gravity_head"])
    spec = BEVSpec(model.bev_head.size, model.bev_head.res)
    st = NavStream({"bev_root": W["bev_root"], "bev": W.get("bev", {}), "datasets": {},
                    "cam_h_range": W.get("cam_h_range") or [0.1, 2.2]})
    from .data import WorldBEVs
    st.bevs, st.label_ds = WorldBEVs(W["bev_root"]), {}
    scenes = {}
    x, y = spec.cell_xy()
    dist = np.hypot(x, y)
    meters = defaultdict(Meter)
    grav = defaultdict(list)
    scale_err = defaultdict(list)
    nfig = defaultdict(int)
    gen = torch.Generator().manual_seed(0)
    wins = W["windows"][:limit] if limit else W["windows"]
    for wi, w in enumerate(wins):
        try:
            it = load_window(w, st, scenes)
        except (OSError, KeyError, ValueError) as e:
            print("skip", e)
            continue
        if it is None:
            continue
        ds = w["dataset"]
        imgs = it["images"][None].cuda()
        bb = model.backbone(imgs)
        gt = it["bev"].numpy()
        vis = it["vis"].numpy()
        depths = it["depths"][None].cuda()
        from .train import gt_log_scale
        ls_gt = gt_log_scale(bb["depth"], depths, depths > 0)
        up_gt = it["up_cam"][None].cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            up_pred = model.gravity(bb).float()
        ang = math.degrees(math.acos(float((up_pred * up_gt).sum().clamp(-1, 1))))
        grav[ds].append(ang)
        if "log_scale_head" in bb:
            scale_err[ds].append(abs(float(bb["log_scale_head"][0] - ls_gt[0])))
        S = imgs.shape[1]
        sels = {"all": np.ones_like(gt, bool), "seen": (vis & 2) > 0, "unseen": (vis & 2) == 0}
        for lo, hi in RANGES:
            sels[f"r{lo:g}-{hi:g}" if hi < 1e8 else f"r>{lo:g}"] = (dist >= lo) & (dist < hi)
        preds = {}
        for cond in conditions:
            if cond == "oracle":
                ls, up = ls_gt, up_gt
            elif cond == "stereo":
                ls = ls_gt + 0.03 * torch.randn(1, generator=gen).cuda()
                up = up_pred
            else:
                ls = bb["log_scale_head"].float() if "log_scale_head" in bb else ls_gt
                up = up_pred
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit = model.heads(bb, ls, up).float()[0].cpu().numpy()
            p = 1 / (1 + np.exp(-logit))
            pred = np.where(p > 0.5, OCC, FREE).astype(np.uint8)
            preds[cond] = p
            for sk, sm in sels.items():
                meters[ds].add(f"{cond}/{sk}", pred, gt, sm)
                meters["ALL"].add(f"{cond}/{sk}", pred, gt, sm)
            meters[ds].add(f"{cond}/S{S}", pred, gt, sels["all"])
        if baselines:
            cam_h = float(it["cam_h"])
            A = bev_axes(up_gt)[0].cpu().numpy()
            ext = bb["extrinsics"][0].cpu().numpy()
            intr = bb["intrinsics"][0].cpu().numpy()
            dp = bb["depth"][0].float().cpu().numpy() * math.exp(float(ls_gt[0]))
            conf = bb["depth_conf"][0].float().cpu().numpy()
            P, Wt = [], []
            for s in range(S):
                pts, ok = lift_np(dp[s], intr[s], ext[s], A)
                P.append(pts)
                Wt.append(np.clip((conf[s][::2, ::2][ok] - 1) / 20, 0, 1) + 0.05)
            bg = geom_baseline(np.concatenate(P), np.concatenate(Wt), spec, cam_h)
            # GT depth with the GT cameras (camera-from-frame-0 from the window's extrinsics)
            Es = it["extrinsics"].numpy().astype(np.float64)
            Ks = it["intrinsics"].numpy().astype(np.float64)
            E0inv = np.linalg.inv(Es[0])
            Ag = A
            P, Wt = [], []
            for s in range(S):
                Erel = Es[s] @ E0inv
                pts, ok = lift_np(it["depths"][s].numpy(), Ks[s], Erel, Ag)
                P.append(pts)
                Wt.append(np.ones(len(pts)))
            bgt = geom_baseline(np.concatenate(P), np.concatenate(Wt), spec, cam_h, min_occ=2.0)
            for name, pred in (("geom", bg), ("gtgeom", bgt)):
                for sk, sm in sels.items():
                    meters[ds].add(f"{name}/{sk}", pred, gt, sm)
                    meters["ALL"].add(f"{name}/{sk}", pred, gt, sm)
        if figs and nfig[ds] < n_figs:
            nfig[ds] += 1
            Path(figs).mkdir(parents=True, exist_ok=True)
            img = (it["images"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[..., ::-1]
            img = cv2.resize(img, (int(img.shape[1] * 400 / img.shape[0]), 400))
            tiles = [img, colour(gt)] + [prob_rgb(preds[c], gt) for c in conditions[:2]]
            if baselines:
                tiles.append(colour(bg))
            row = np.concatenate([cv2.copyMakeBorder(t, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(0, 0, 255))
                                  for t in tiles], 1)
            cv2.imwrite(str(Path(figs) / f"{ds}_{nfig[ds]:02d}.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if (wi + 1) % 50 == 0:
            print(f"{wi + 1}/{len(wins)}", flush=True)
    res = {"heads": str(heads_path), "windows": str(windows_path),
           "metrics": {ds: m.result() for ds, m in meters.items()},
           "gravity_deg": {ds: round(float(np.mean(v)), 3) for ds, v in grav.items()},
           "gravity_deg_median": {ds: round(float(np.median(v)), 3) for ds, v in grav.items()},
           "scale_head_abs_log_err": {ds: round(float(np.mean(v)), 4) for ds, v in scale_err.items()}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("make")
    a.add_argument("--config", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--per-dataset", type=int, default=150)
    a.add_argument("--split", default="val", choices=["val", "test"])
    a.add_argument("--root")
    a.add_argument("--bev-root")
    a.add_argument("--datasets", nargs="*")
    b = sp.add_parser("run")
    b.add_argument("--windows", required=True)
    b.add_argument("--heads", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--figs")
    b.add_argument("--n-figs", type=int, default=6)
    b.add_argument("--backbone")
    b.add_argument("--no-baselines", action="store_true")
    b.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "make":
        make_windows(yaml.safe_load(open(args.config)), args.out, args.per_dataset, args.split, args.root,
                     args.bev_root, args.datasets)
    else:
        r = run(args.windows, args.heads, args.out, args.figs, args.n_figs, args.backbone,
                baselines=not args.no_baselines, limit=args.limit)
        m = r["metrics"].get("ALL", {})
        for k in sorted(m):
            if k.endswith(("/all", "/seen", "/unseen")):
                print(k, m[k])


if __name__ == "__main__":
    main()
