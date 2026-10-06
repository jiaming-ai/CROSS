"""Weight interpolation between the released VGGT-Omega and a fine-tuned checkpoint (WiSE-FT, Wortsman et al. 2022):
theta = (1 - alpha) * released + alpha * fine-tuned for every weight both hold; weights only the fine-tune has (the scale
and covisibility heads) are kept as they are.

python scripts/vggt_ft/interpolate.py --released <vggt_omega_1b_512.pt> --ft <ckpt_*_bf16.pt> --alpha 0.5 0.75 \
    --out_dir <dir>      ->  <dir>/ckpt_a050_bf16.pt, <dir>/ckpt_a075_bf16.pt (the layout evalwatch.sh evaluates)
"""
import argparse
from pathlib import Path

import torch


def state_dict(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    meta = {k: v for k, v in sd.items() if k != "model"} if isinstance(sd, dict) and "model" in sd else {}
    return (sd["model"] if "model" in sd else sd), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--alpha", type=float, nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="", help="file name prefix, e.g. s4000 -> ckpt_s4000a050_bf16.pt")
    a = ap.parse_args()
    rel, _ = state_dict(a.released)
    ft, meta = state_dict(a.ft)
    shared = [k for k in ft if k in rel and ft[k].is_floating_point() and ft[k].shape == rel[k].shape]
    print(f"{len(shared)} interpolated, {len(ft) - len(shared)} kept from the fine-tune")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for alpha in a.alpha:
        sd = dict(ft)
        for k in shared:
            sd[k] = ((1 - alpha) * rel[k].float() + alpha * ft[k].float()).to(torch.bfloat16)
        torch.save({"model": sd, **meta, "alpha": alpha, "source": str(a.ft)},
                   out / f"ckpt_{a.tag}a{round(alpha * 100):03d}_bf16.pt")
        print("wrote", out / f"ckpt_{a.tag}a{round(alpha * 100):03d}_bf16.pt")


if __name__ == "__main__":
    main()
