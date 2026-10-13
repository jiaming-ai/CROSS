"""Rectification check of a prepared stereo folder (left/ right/): matched features of a rectified pair lie on the
same row, so the vertical offset of ORB matches (after a robust filter) should be ~0 px.  Prints per-folder statistics
of the per-frame median offset and of all matches.

  python scripts/rig/check_rectification.py data/walk1 [data/walk2 ...] [--frames 100]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def frame_offsets(left, right, n_features=1500):
    orb = cv2.ORB_create(n_features)
    kl, dl = orb.detectAndCompute(left, None)
    kr, dr = orb.detectAndCompute(right, None)
    if dl is None or dr is None:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dl, dr)
    if len(m) < 20:
        return None
    pl = np.float32([kl[x.queryIdx].pt for x in m])
    pr = np.float32([kr[x.trainIdx].pt for x in m])
    dv = pr[:, 1] - pl[:, 1]
    du = pl[:, 0] - pr[:, 0]                       # disparity: positive for points in front
    ok = (du > 0) & (np.abs(dv - np.median(dv)) < 8)
    return dv[ok] if ok.sum() >= 10 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--frames", type=int, default=100)
    a = ap.parse_args()
    for f in a.folders:
        names = sorted((f / "left").glob("*.png"))
        pick = [names[i] for i in np.linspace(0, len(names) - 1, min(a.frames, len(names))).astype(int)]
        per, allv = [], []
        for p in pick:
            L = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            R = cv2.imread(str(f / "right" / p.name), cv2.IMREAD_GRAYSCALE)
            if L is None or R is None:
                continue
            dv = frame_offsets(L, R)
            if dv is not None:
                per.append(float(np.median(dv)))
                allv.append(dv)
        allv = np.concatenate(allv) if allv else np.zeros(0)
        per = np.asarray(per)
        print(json.dumps({"folder": str(f), "frames": int(len(per)),
                          "frame_offset_px_median": round(float(np.median(per)), 2) if len(per) else None,
                          "frame_offset_px_p5_p95": [round(float(np.percentile(per, q)), 2) for q in (5, 95)] if len(per) else None,
                          "frame_offset_abs_px_median": round(float(np.median(np.abs(per))), 2) if len(per) else None,
                          "match_offset_abs_px_median": round(float(np.median(np.abs(allv))), 2) if len(allv) else None}))


if __name__ == "__main__":
    main()
