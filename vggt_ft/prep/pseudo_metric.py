"""Pseudo-metric scale of the scenes of a dataset from monocular metric-depth models (vggt_ft/labelers.py).

The stored depth and poses of a non-metric scene (SfM / MVS reconstructions) are correct up to one factor per scene,
so one number makes the scene metric:  k = median over sampled frames of median(labeler metric depth / stored depth)
over the frame's valid pixels.  Averaging over frames cancels much of a labeler's per-image error.  The ensemble
("ens") takes, per frame, the median of the labelers' log ratios.  On a metric dataset the same procedure measures the
labelers' error, since there k should be 1.

python -m vggt_ft.prep.pseudo_metric <root> <dataset> --weights <dir> [--labelers da3metric unidepth moge2] \
    [--frames 24] [--max_scenes N] [--shard i/n] --out <json>
-> {scene: {"n": frames used, "k": {labeler: k, ..., "ens": k}, "iqr": {labeler: [q25, q75] of per-frame ratios}}}
(--max_scenes takes evenly spaced scenes; shards are merged by reading all outputs).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vggt_ft.dataio.scene import list_scenes, load_scene
from vggt_ft.labelers import load_labelers


def sample_frames(scene, n, rng, min_valid=0.02):
    """Up to n (sequence, index) pairs with valid depth (fraction > min_valid; lidar depth covers ~10 % of the image),
    spread over the scene's sequences."""
    cand = []
    for q in scene.sequences:
        z = np.load(scene.dir / q / "frames.npz", allow_pickle=False)
        valid = z["valid"] if "valid" in z else np.ones(len(z["names"]))
        cand += [(q, i) for i in np.flatnonzero(valid > min_valid)]
    if len(cand) > n:
        cand = [cand[i] for i in np.sort(rng.choice(len(cand), n, replace=False))]
    return cand


@torch.no_grad()
def scene_scale(scene, labelers, n_frames, rng, device="cuda"):
    logs = {name: [] for name in labelers}
    for q, i in sample_frames(scene, n_frames, rng):
        rgb, dep, K, _ = scene.seq(q).load(i)
        img = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
        Kt = torch.from_numpy(K).float().to(device)
        D = torch.from_numpy(dep).to(device)
        for name, lab in labelers.items():
            L = lab(img, Kt)
            if L.shape != D.shape:          # seen rarely (an output of shape (H, 1)): skip this labeler on the frame
                print(f"  {scene.name} {q} {i}: {name} returned {tuple(L.shape)} for {tuple(D.shape)}", flush=True)
                continue
            ok = (D > 0) & torch.isfinite(L) & (L > 0)
            if ok.sum() > 200:
                logs[name].append(float(torch.log(L[ok] / D[ok]).median()))
    n = min(len(v) for v in logs.values()) if logs else 0
    if n == 0:
        return None
    out = {"n": n, "k": {}, "iqr": {}}
    for name, v in logs.items():
        v = np.array(v)
        out["k"][name] = float(np.exp(np.median(v)))
        out["iqr"][name] = [float(np.exp(np.quantile(v, 0.25))), float(np.exp(np.quantile(v, 0.75)))]
    if len(labelers) > 1:
        per_frame = np.median(np.stack([np.array(logs[k][:n]) for k in labelers]), axis=0)
        out["k"]["ens"] = float(np.exp(np.median(per_frame)))
        out["iqr"]["ens"] = [float(np.exp(np.quantile(per_frame, 0.25))), float(np.exp(np.quantile(per_frame, 0.75)))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("dataset")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--labelers", nargs="+", default=["da3metric", "unidepth", "moge2"])
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scenes = list_scenes(Path(a.root), a.dataset)
    if a.max_scenes and len(scenes) > a.max_scenes:
        scenes = [scenes[i] for i in np.linspace(0, len(scenes) - 1, a.max_scenes).round().astype(int)]
    si, sn = map(int, a.shard.split("/"))
    scenes = scenes[si::sn]
    out = json.load(open(a.out)) if Path(a.out).exists() else {}
    labelers = load_labelers(a.labelers, a.weights)
    t0 = time.time()
    for j, d in enumerate(scenes):
        if d.name in out:
            continue
        try:
            r = scene_scale(load_scene(d), labelers, a.frames, np.random.default_rng(0))
        except (FileNotFoundError, KeyError, ValueError, IndexError, RuntimeError) as e:
            print(f"skip {d.name}: {e}", flush=True)
            continue
        if r is not None:
            out[d.name] = r
        if (j + 1) % 10 == 0 or j + 1 == len(scenes):
            json.dump(out, open(a.out, "w"))
            print(f"{a.dataset} {j + 1}/{len(scenes)} {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
