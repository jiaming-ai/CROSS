"""Merge two world BEV label sets of the same worlds: the primary label where it is known, the secondary one filling
the primary's unknown cells (in the primary's grid and gravity frame).

Used for TartanGround: primary = the complete labels from the global point clouds (cross_nav/sim/pcd_bev.py,
`tartanground_pcd`), secondary = the depth-accumulated labels (cross_nav/labels.py, `tartanground`), which cover the
few places the point clouds miss (areas left out of a cloud, water, very sparse terrain).

    python -m cross_nav.sim.bev_merge --bev <bev root> --primary tartanground_pcd --secondary tartanground \
        --out tartanground_mix [--workers 8]

Per world: bev = primary where known, else the secondary label at the cell centre (nearest cell, mapped through the
two frames' R_g: both are gravity frames, so the mapping is a rotation about z plus a shift; heights are not needed);
cam_h/<scene>/<seq> = primary where finite, else secondary; ground = primary's; other keys from the primary.  Cells are
not filled when the two gravity directions differ by more than 2 deg (the depth-accumulated labels estimate gravity from
the surface normals, which slopes bias: 3 - 9 deg in hilly TartanGround scenes).
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cross_nav.sim.mesh_bev import preview

UNKNOWN = 0


def merge(p: dict, s: dict, max_tilt: float = 2.0):
    bev_p, bev_s = p["bev"], s["bev"]
    res_p, res_s = float(p["res"]), float(s["res"])
    H, W = bev_p.shape
    M = s["R_g"] @ p["R_g"].T                        # primary gravity frame -> secondary gravity frame
    unk = np.flatnonzero(bev_p.reshape(-1) == UNKNOWN)
    iy, ix = np.divmod(unk, W)
    x = p["origin"][0] + (ix + 0.5) * res_p
    y = p["origin"][1] + (iy + 0.5) * res_p
    xs = M[0, 0] * x + M[0, 1] * y
    ys = M[1, 0] * x + M[1, 1] * y
    jx = np.floor((xs - s["origin"][0]) / res_s).astype(np.int64)
    jy = np.floor((ys - s["origin"][1]) / res_s).astype(np.int64)
    ok = (jx >= 0) & (jx < bev_s.shape[1]) & (jy >= 0) & (jy < bev_s.shape[0])
    tilt = float(np.degrees(np.arccos(np.clip(M[2, 2], -1, 1))))
    if tilt > max_tilt:          # the frames' gravity disagree (secondary's estimate biased by slopes): no cell fill
        ok[:] = False
    out = dict(p)
    bev = bev_p.copy().reshape(-1)
    bev[unk[ok]] = bev_s[jy[ok], jx[ok]]
    out["bev"] = bev.reshape(H, W)
    n_cam = n_fill = 0
    for k in p:
        if k.startswith("cam_h/") and k in s:
            a, b = p[k].copy(), s[k]
            fill = ~np.isfinite(a) & np.isfinite(b)
            a[fill] = b[fill]
            out[k] = a
            n_cam += a.size
            n_fill += int(fill.sum())
    stats = dict(known_primary=int((bev_p > 0).sum()), known=int((out["bev"] > 0).sum()),
                 filled=int((out["bev"] > 0).sum() - (bev_p > 0).sum()), cam_h_filled=n_fill, cam_h_n=n_cam,
                 tilt_deg=round(tilt, 2))
    st = json.loads(str(p["stats"])) if "stats" in p else {}
    out["stats"] = json.dumps({"merge": stats, "primary": st})
    return out, stats


def _job(a):
    w, bev, prim, sec, outn = a
    root = Path(bev)
    po, so, oo = root / prim / f"{w}.npz", root / sec / f"{w}.npz", root / outn / f"{w}.npz"
    if oo.exists():
        return w, "exists"
    if not so.exists():
        return w, "no secondary"
    p = dict(np.load(po))
    s = dict(np.load(so))
    out, stats = merge(p, s)
    tmp = oo.with_name(oo.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, oo)
    preview(out, oo.with_suffix(".png"), scale=2 if out["bev"].size < 4e6 else 1)
    return w, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bev", required=True)
    ap.add_argument("--primary", default="tartanground_pcd")
    ap.add_argument("--secondary", default="tartanground")
    ap.add_argument("--out", default="tartanground_mix")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = Path(a.bev)
    (root / a.out).mkdir(parents=True, exist_ok=True)
    worlds = sorted(f.stem for f in (root / a.primary).glob("*.npz") if not f.name.endswith(".tmp.npz"))
    ex = root / a.primary / "exclude.txt"
    if ex.exists():
        (root / a.out / "exclude.txt").write_text(ex.read_text())
    with Pool(a.workers) as pool:
        for w, st in pool.imap_unordered(_job, [(w, a.bev, a.primary, a.secondary, a.out) for w in worlds]):
            print(w, st, flush=True)


if __name__ == "__main__":
    main()
