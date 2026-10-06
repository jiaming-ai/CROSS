"""Global appearance descriptors of every frame of the scene cache, for appearance-mined hard negatives.

Descriptor = L2-normalised [class token, GeM(p=3) of the patch tokens] of the released VGGT-Omega's DINO encoder
(no extra model needed) on the image resized to 224 px height; stored as <sequence>/vpr.npy (N, 2048) float16.  The
sampler (dataio/windows.py, p_hard_neg) draws a negative group's anchor among the frames most similar to frame 0 that
are geometrically far from it: the look-alike places CROSS's retrieval proposes and its covisibility gate must reject.

python -m vggt_ft.prep.vpr <cache_root> [--datasets a b] --ckpt <vggt_omega_1b_512.pt> [--batch 128]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from vggt_ft.dataio.scene import list_scenes, load_scene


def load_encoder(ckpt: str, device="cuda"):
    from vggt_omega.models.aggregator import _build_patch_embed
    enc = _build_patch_embed(16, 1024)
    sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    sub = {k[len("aggregator.patch_embed."):]: v for k, v in sd.items() if k.startswith("aggregator.patch_embed.")}
    enc.load_state_dict(sub)
    return enc.to(device).eval()


@torch.no_grad()
def describe(enc, imgs: list[np.ndarray], device="cuda") -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(np.stack(imgs)).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = (x - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = enc.forward_features(x)
    cls = F.normalize(out["x_norm_clstoken"].float(), dim=-1)
    p = out["x_norm_patchtokens"].float().clamp(min=1e-6)
    gem = F.normalize(p.pow(3).mean(1).pow(1 / 3), dim=-1)
    return F.normalize(torch.cat([cls, gem], -1), dim=-1).half().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hw", type=int, nargs=2, default=[224, 288])
    a = ap.parse_args()
    enc = load_encoder(a.ckpt)
    dsets = a.datasets or sorted(p.name for p in Path(a.root).iterdir() if p.is_dir())
    h, w = a.hw
    for ds in dsets:
        n_seq = 0
        for d in list_scenes(a.root, ds):
            sc = load_scene(d)
            for qn in sc.sequences:
                q = sc.seq(qn)
                out = q.dir / "vpr.npy"
                if out.exists():
                    continue
                desc = []
                for s in range(0, q.n, a.batch):
                    imgs = []
                    for nm in q.names[s:s + a.batch]:
                        im = cv2.imread(str(q.dir / "rgb" / f"{nm}.jpg"), cv2.IMREAD_COLOR)
                        imgs.append(cv2.resize(np.ascontiguousarray(im[..., ::-1]), (w, h), interpolation=cv2.INTER_AREA))
                    desc.append(describe(enc, imgs))
                tmp = q.dir / "vpr.tmp.npy"
                np.save(tmp, np.concatenate(desc))
                tmp.rename(out)
                n_seq += 1
        print(f"{ds}: {n_seq} sequences described", flush=True)


if __name__ == "__main__":
    main()
