"""Agreement of the mesh BEV labels with the accumulated-depth labels (cross_nav.labels) of the same renders.

    python -m cross_nav.sim.compare_acc --mesh <bev>/hssd_nav --acc <bev_acc>/hssd_nav [--out stats.json] [--png-dir d]

The accumulated label is resampled into the mesh label's grid (mesh world = gravity frame, z up; the accumulated
label's R_g is its own gravity estimate, applied to the mesh cell centres at the floor height).  Reported per scene and
in total: cells labelled by both, agreement on them, the confusion matrix (rows mesh free / occupied, columns
accumulated free / occupied), and the number of cells only the mesh labels (vs. only the accumulation).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

UNKNOWN, FREE, OCC = 0, 1, 2


def resample_acc(mesh: dict, acc: dict) -> np.ndarray:
    H, W = mesh["bev"].shape
    res = float(mesh["res"])
    o = mesh["origin"]
    xs = o[0] + (np.arange(W) + 0.5) * res
    ys = o[1] + (np.arange(H) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    g = mesh["ground"].astype(np.float32)
    z0 = float(np.nanmedian(g)) if np.isfinite(g).any() else 0.0
    P = np.stack([X, Y, np.full_like(X, z0)], -1) @ np.asarray(acc["R_g"]).T      # world (= mesh) -> acc gravity
    ra, oa = float(acc["res"]), acc["origin"]
    u = ((P[..., 0] - oa[0]) / ra - 0.5).astype(np.float32)
    v = ((P[..., 1] - oa[1]) / ra - 0.5).astype(np.float32)
    return cv2.remap(acc["bev"], u, v, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=UNKNOWN)


def compare(mesh: dict, acc_rs: np.ndarray) -> dict:
    m = mesh["bev"]
    both = (m != UNKNOWN) & (acc_rs != UNKNOWN)
    conf = [[int(((m == a) & (acc_rs == b)).sum()) for b in (FREE, OCC)] for a in (FREE, OCC)]
    agree = conf[0][0] + conf[1][1]
    return dict(both=int(both.sum()), agree=round(agree / max(both.sum(), 1), 4), conf=conf,
                mesh_only=int(((m != UNKNOWN) & (acc_rs == UNKNOWN)).sum()),
                acc_only=int(((m == UNKNOWN) & (acc_rs != UNKNOWN)).sum()),
                mesh_known=int((m != UNKNOWN).sum()), acc_known=int((acc_rs != UNKNOWN).sum()))


def overlay(mesh: dict, acc_rs: np.ndarray, path: Path):
    """Left: mesh label; middle: accumulated (resampled); right: disagreement (red: mesh occupied / acc free,
    blue: mesh free / acc occupied, green: mesh only)."""
    def col(b):
        img = np.full(b.shape + (3,), 128, np.uint8)
        img[b == FREE] = 255
        img[b == OCC] = 0
        return img
    m = mesh["bev"]
    d = np.full(m.shape + (3,), 200, np.uint8)
    d[(m == FREE) & (acc_rs == FREE)] = 255
    d[(m == OCC) & (acc_rs == OCC)] = 0
    d[(m == OCC) & (acc_rs == FREE)] = (0, 0, 255)
    d[(m == FREE) & (acc_rs == OCC)] = (255, 0, 0)
    d[(m != UNKNOWN) & (acc_rs == UNKNOWN)] = (120, 200, 120)
    sep = np.full((m.shape[0], 4, 3), (0, 128, 255), np.uint8)
    cv2.imwrite(str(path), np.concatenate([col(m), sep, col(acc_rs), sep, d], 1)[::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--acc", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--png-dir", default=None)
    a = ap.parse_args()
    rows, tot = {}, None
    for p in sorted(Path(a.mesh).glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        q = Path(a.acc) / p.name
        if not q.exists():
            continue
        mesh, acc = dict(np.load(p)), dict(np.load(q))
        rs = resample_acc(mesh, acc)
        r = compare(mesh, rs)
        rows[p.stem] = r
        if a.png_dir:
            Path(a.png_dir).mkdir(parents=True, exist_ok=True)
            overlay(mesh, rs, Path(a.png_dir) / f"{p.stem}_mesh_vs_acc.png")
        if tot is None:
            tot = {k: (np.array(v) if isinstance(v, list) else v) for k, v in r.items() if k != "agree"}
        else:
            for k, v in r.items():
                if k != "agree":
                    tot[k] = tot[k] + (np.array(v) if isinstance(v, list) else v)
        print(p.stem, json.dumps(r), flush=True)
    if tot is not None:
        c = tot["conf"]
        tot = {k: (v.tolist() if isinstance(v, np.ndarray) else int(v)) for k, v in tot.items()}
        tot["agree"] = round((c[0][0] + c[1][1]) / max(tot["both"], 1), 4)
        tot["mesh_vs_acc_known"] = round(tot["mesh_known"] / max(tot["acc_known"], 1), 3)
        tot["scenes"] = len(rows)
        print("TOTAL", json.dumps(tot), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"total": tot, "scenes": rows}, indent=1))


if __name__ == "__main__":
    main()
