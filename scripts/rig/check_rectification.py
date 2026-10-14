"""Rectification check of a prepared stereo folder (left/ right/): matched features of a rectified pair lie on the
same row, so the vertical offset of matches should be ~0 px.  Per pair, SIFT matches (sub-pixel positions, ratio
test, positive disparity) are fitted robustly with dv = a + b * (u - cx) + c * (v - cy): a is the pair's offset at the
image centre, b and c the residual rotation about the optical axis and about the baseline (px per px).  Prints
per-folder statistics over the sampled pairs.  --detector orb gives the earlier check (ORB, median offset only; ORB
positions are quantised to the pyramid levels, ~1.2 px steps).

  python scripts/rig/check_rectification.py data/walk1 [data/walk2 ...] [--frames 100] [--json per_pair.json]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _matches(left, right, detector="sift"):
    if detector == "orb":
        f = cv2.ORB_create(1500)
        kl, dl = f.detectAndCompute(left, None)
        kr, dr = f.detectAndCompute(right, None)
        if dl is None or dr is None:
            return None
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dl, dr)
    else:
        f = cv2.SIFT_create(3000)
        kl, dl = f.detectAndCompute(left, None)
        kr, dr = f.detectAndCompute(right, None)
        if dl is None or dr is None or len(dl) < 2 or len(dr) < 2:
            return None
        m = [p[0] for p in cv2.BFMatcher(cv2.NORM_L2).knnMatch(dl, dr, k=2)
             if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(m) < 20:
        return None
    pl = np.float32([kl[x.queryIdx].pt for x in m])
    pr = np.float32([kr[x.trainIdx].pt for x in m])
    return pl, pr


def pair_offsets(left, right, detector="sift"):
    """{"a", "b", "c", "n", "resid"} of one rectified pair (None with too few matches); "median" = plain median dv."""
    mt = _matches(left, right, detector)
    if mt is None:
        return None
    pl, pr = mt
    dv = (pr[:, 1] - pl[:, 1]).astype(np.float64)
    du = pl[:, 0] - pr[:, 0]                       # disparity: positive for points in front
    ok = (du > 0) & (np.abs(dv - np.median(dv)) < 8)
    if ok.sum() < 15:
        return None
    H, W = left.shape
    A = np.c_[np.ones(ok.sum()), pl[ok, 0] - (W - 1) / 2, pl[ok, 1] - (H - 1) / 2]
    y = dv[ok]
    w = np.ones(len(y))
    for _ in range(10):                            # Huber IRLS (1 px)
        x, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        r = y - A @ x
        w = np.sqrt(np.minimum(1.0, 1.0 / np.maximum(np.abs(r), 1e-9)))
    return {"a": float(x[0]), "b": float(x[1]), "c": float(x[2]), "n": int(ok.sum()),
            "resid": float(np.median(np.abs(r))), "median": float(np.median(y))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--detector", choices=("sift", "orb"), default="sift")
    ap.add_argument("--json", type=Path, help="write the per-pair fits of every folder")
    a = ap.parse_args()
    dump = {}
    for f in a.folders:
        names = sorted((f / "left").glob("*.png"))
        pick = [names[i] for i in np.linspace(0, len(names) - 1, min(a.frames, len(names))).astype(int)]
        fits = []
        for p in pick:
            L = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            R = cv2.imread(str(f / "right" / p.name), cv2.IMREAD_GRAYSCALE)
            if L is None or R is None:
                continue
            fit = pair_offsets(L, R, a.detector)
            if fit is not None:
                fits.append({"frame": p.stem, **fit})
        dump[str(f)] = fits
        off = np.asarray([x["a" if a.detector == "sift" else "median"] for x in fits])
        rnd = lambda v: round(float(v), 2)  # noqa: E731
        print(json.dumps({"folder": str(f), "frames": int(len(off)),
                          "frame_offset_px_median": rnd(np.median(off)) if len(off) else None,
                          "frame_offset_px_p5_p95": [rnd(np.percentile(off, q)) for q in (5, 95)] if len(off) else None,
                          "frame_offset_abs_px_median": rnd(np.median(np.abs(off))) if len(off) else None,
                          "frame_offset_spread_px": rnd(np.median(np.abs(off - np.median(off)))) if len(off) else None,
                          "match_residual_px_median": rnd(np.median([x["resid"] for x in fits])) if fits else None}))
    if a.json:
        a.json.write_text(json.dumps(dump))


if __name__ == "__main__":
    main()
