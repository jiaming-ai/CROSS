"""Dump a few training windows (frame 0 image, BEV label, visibility) as PNGs to check the labels by eye.

    python scripts/nav_dump_samples.py --config cross_nav/configs/train/nav_v1.yaml --out <dir> [--n 12] [--dataset scannet]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from cross_nav.data import NavStream
from vggt_ft.dataio.transforms import target_shape


def label_rgb(bev, vis=None):
    img = np.full(bev.shape + (3,), 128, np.uint8)
    img[bev == 1] = 255
    img[bev == 2] = 0
    if vis is not None:
        img[(vis & 1) > 0] = (img[(vis & 1) > 0] * 0.6 + np.array([0, 0, 255]) * 0.4).astype(np.uint8)
        m = ((vis & 2) > 0) & ((vis & 1) == 0)
        img[m] = (img[m] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)
    n = bev.shape[0]
    cv2.circle(img, (n // 2, n // 2), 4, (0, 200, 0), -1)
    cv2.line(img, (n // 2, n // 2), (n // 2, n // 2 - 20), (0, 200, 0), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--dataset", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))["data"]
    if a.dataset:
        cfg["datasets"] = {a.dataset: cfg["datasets"][a.dataset]}
    st = NavStream(cfg, seed=1)
    st._build()
    rng = np.random.default_rng(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k = 0
    while k < a.n:
        S = int(rng.integers(1, 4))
        hw = target_shape(0.75, 262144)
        pool, frames, negs, kind = st.sampler.sample(rng, S)
        it = st.make_item(rng, pool, frames, negs, kind, hw, aug=False)
        if it is None:
            continue
        img = (it["images"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[..., ::-1]
        lab = label_rgb(it["bev"].numpy(), it["vis"].numpy())
        img = cv2.resize(img, (int(img.shape[1] * 400 / img.shape[0]), 400))
        cv2.putText(img, f"{pool.name} S={S} h={float(it['cam_h']):.2f} up={np.round(it['up_cam'].numpy(), 2)}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imwrite(str(out / f"{k:02d}_{pool.name}.jpg"), np.concatenate([img, lab], 1))
        k += 1


if __name__ == "__main__":
    main()
