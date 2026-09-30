#!/usr/bin/env python3
"""Variant gallery for the HSSD scenes (report/figures/hssd_examples_<scene>.png).

The same left frame of the map traversal rendered under every path-preserving variant, as
scripts/make_sim_figures.py:fig_examples does for the Blender scenes.  Must run on a host that has the
rendered traversals.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

VARIANTS = [("map", "map"), ("rearr_25", "rearrange 25 %"), ("rearr_50", "rearrange 50 %"), ("rearr_100", "rearrange 100 %"),
            ("rearr_50_remove", "remove 50 %"), ("rearr_100_relocate", "relocate 100 %"), ("light_morning", "morning"), ("light_evening", "evening"),
            ("light_overcast", "overcast"), ("light_night", "night"), ("offset_1.0", "offset 1 m"), ("yaw_45", "yaw 45 deg")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["hssd_house", "hssd_restaurant"])
    ap.add_argument("--data-root", default="data/sim")
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--frame", type=int, default=600)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        ims = []
        for v, label in VARIANTS:
            files = sorted((Path(args.data_root) / scene / v / "left").glob("*.png"))
            if not files:
                print(f"  ! {scene}/{v}: no frames")
                continue
            im = cv2.resize(cv2.imread(str(files[min(args.frame, len(files) - 1)])), (320, 240))
            cv2.rectangle(im, (0, 0), (320, 30), (0, 0, 0), -1)
            cv2.putText(im, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            ims.append(im)
        if not ims:
            continue
        while len(ims) % args.cols:
            ims.append(np.zeros_like(ims[0]))
        rows = [np.concatenate(ims[i:i + args.cols], 1) for i in range(0, len(ims), args.cols)]
        p = out / f"hssd_examples_{scene}.png"
        cv2.imwrite(str(p), np.concatenate(rows, 0))
        print("wrote", p)


if __name__ == "__main__":
    main()
