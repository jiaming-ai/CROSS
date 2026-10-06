"""Metric scale of off-the-shelf monocular metric-depth models, as candidate pseudo-labelers for the scale head.

On the held-out windows of evaluate.py, each labeler predicts metric depth L for every frame from the image and its
intrinsics (known in CROSS and at training time).  Its window scale for the fine-tuned model's depth d is
s_L = median(L / d) over the labeler's valid pixels (no GT used), compared with the scale head's s = exp(log_scale) on
the same windows.  Per test set and source: m_abs (AbsRel of s * d against GT metric depth, as evaluate.py), lse
(|log s - log s*|, s* = median GT / d), ratio (median of s / s*: > 1 means scenes predicted too large), and for the
labelers their own per-frame dense AbsRel against GT (dense_abs).

python scripts/vggt_ft/eval_metric_labelers.py --ckpt <fine-tuned ckpt> --suite configs/vggt_ft/eval_suite.yaml \
    --weights <dir with DA3METRIC-LARGE/, unidepth-v2-vitl14/, moge-2-vitl-normal/> \
    --labelers da3metric unidepth moge2 --max_windows 40 --out <json>
Labeler packages (depth_anything_3, unidepth, moge) must be importable (PYTHONPATH).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from vggt_ft.dataio.scene import load_scene
from vggt_ft.dataio.transforms import target_shape
from vggt_ft.dataio.windows import materialise
from vggt_ft.evaluate import build_windows, load_model
from vggt_ft.geometry import connected_to_ref, gt_covisibility
from vggt_ft.labelers import LABELERS, load_labelers



def scale_metrics(s, pdep, gt, m):
    pm, gm = s * pdep[m], gt[m]
    s_star = (gm / pdep[m]).median()
    return {"m_abs": float(((pm - gm).abs() / gm).mean()), "lse": float((torch.log(s) - torch.log(s_star)).abs()),
            "ratio": float(s / s_star)}


@torch.no_grad()
def run_set(model, labelers, ts, windows, device="cuda"):
    hw = target_shape(ts.get("aspect", 0.75), ts.get("area", 512 * 512))
    acc = {name: {"m_abs": [], "lse": [], "ratio": [], "dense_abs": []} for name in ["head", *labelers]}
    cache = {}
    for spec in windows:
        frames = []
        for d, qn, i in spec:
            if d not in cache:
                cache[d] = load_scene(Path(d))
            frames.append((cache[d], cache[d].seq(qn), i))
        b = materialise(frames, hw)
        if not bool(b["metric"]) or not bool(b["masks"].any()):
            continue
        img = b["images"].to(device)
        pred = model(img[None])
        pdep = pred["depth"][0, ..., 0].float()
        gt, mask, K = b["depths"].to(device), b["masks"].to(device), b["intrinsics"].to(device)
        if img.shape[0] > 1:
            cov = gt_covisibility(gt[None], mask[None], b["extrinsics"][None].to(device), K[None],
                                  b["world_id"][None].to(device))
            conn = connected_to_ref(cov, 0.05)[0]
        else:
            conn = torch.ones(1, dtype=torch.bool, device=device)
        m = mask & conn[:, None, None] & (pdep > 0)
        if m.sum() < 50:
            continue
        if "log_scale" in pred:
            for k, v in scale_metrics(torch.exp(pred["log_scale"][0].float()), pdep, gt, m).items():
                acc["head"][k].append(v)
        for name, lab in labelers.items():
            L = torch.stack([lab(img[f], K[f]) for f in range(img.shape[0])])
            ok = torch.isfinite(L) & (L > 0) & (pdep > 0)
            if ok.sum() < 50:
                continue
            s_l = (L[ok] / pdep[ok]).median()
            for k, v in scale_metrics(s_l, pdep, gt, m).items():
                acc[name][k].append(v)
            md = m & ok
            if md.sum() > 50:
                acc[name]["dense_abs"].append(float(((L[md] - gt[md]).abs() / gt[md]).mean()))
    res = {}
    for name, a in acc.items():
        if a["m_abs"]:
            res[name] = {"n": len(a["m_abs"]), "m_abs": float(np.mean(a["m_abs"])), "lse": float(np.mean(a["lse"])),
                         "ratio_med": float(np.median(a["ratio"]))}
            if a["dense_abs"]:
                res[name]["dense_abs"] = float(np.mean(a["dense_abs"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--labelers", nargs="+", default=list(LABELERS))
    ap.add_argument("--sets", nargs="*")
    ap.add_argument("--max_windows", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    suite = yaml.safe_load(open(a.suite))
    model = load_model(a.ckpt)
    labelers = load_labelers(a.labelers, a.weights)
    wdir = Path(suite.get("windows_dir", Path(a.suite).with_suffix("")))
    out = json.load(open(a.out)) if Path(a.out).exists() else {}
    out["ckpt"] = a.ckpt
    for ts in suite["sets"]:
        if a.sets and ts["name"] not in a.sets:
            continue
        wf = wdir / f"{ts['name']}.json"
        wins = json.load(open(wf)) if wf.exists() else build_windows(suite["root"], ts)
        if len(wins) > a.max_windows:     # evenly spread over the set (the window lists are ordered by scene)
            wins = [wins[i] for i in np.linspace(0, len(wins) - 1, a.max_windows).round().astype(int)]
        t0 = time.time()
        res = run_set(model, labelers, dict(ts, root=ts.get("root", suite["root"])), wins)
        if res:
            out[ts["name"]] = res
            print(ts["name"], f"{time.time() - t0:.0f}s", json.dumps({k: {m: round(v, 3) for m, v in r.items()}
                                                                      for k, r in res.items()}), flush=True)
            json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
