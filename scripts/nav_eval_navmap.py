"""Score a fused navigation map (cross_nav.navmap) against a world BEV label (mesh / accumulated GT).

    python scripts/nav_eval_navmap.py --navmap <dir>/navmap.npz --label <world>.npz --poses <seq>/poses_left.txt

The map frame is aligned to the label's world by a rigid fit (Umeyama, no scale) of the keyframe centres to their GT
camera centres (poses_left.txt row = source frame index, camera-to-world 4x4).  Every labelled GT cell (centre at the
local ground height) is looked up in the fused map; reported: IoU occupied / free and mIoU over the cells the map
observed, the share of labelled cells it observed, and the same within 2 m / 4 m of a keyframe."""
import argparse
import json

import numpy as np


def umeyama_rigid(A, B):
    """R, t with B ~ R A + t."""
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - mb).T @ (A - ma))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, mb - R @ ma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navmap", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    nm = np.load(a.navmap)
    lab = np.load(a.label)
    P = np.loadtxt(a.poses).reshape(-1, 4, 4)
    si = nm["source_index"]
    ok = si >= 0
    Cm = nm["T_wc"][ok, :3, 3]
    Cg = P[si[ok], :3, 3]
    R, t = umeyama_rigid(Cm, Cg)
    rms = float(np.sqrt(((Cm @ R.T + t - Cg) ** 2).sum(1).mean()))
    bev, res, org = lab["bev"], float(lab["res"]), lab["origin"]
    R_g = lab["R_g"]
    ground = lab["ground"].astype(np.float64)
    gz = float(np.nanmedian(ground))
    iy, ix = np.nonzero(bev > 0)
    xg = org[0] + (ix + 0.5) * res
    yg = org[1] + (iy + 0.5) * res
    zg = np.where(np.isfinite(ground[iy, ix]), ground[iy, ix], gz)
    pg = np.stack([xg, yg, zg], -1) @ R_g                                # gravity frame -> label world
    pm = (pg - t) @ R                                                    # label world -> CROSS map frame
    q = pm @ nm["R_map"].T
    c = np.floor((q[:, :2] - nm["origin"]) / float(nm["res"])).astype(int)
    H, W = nm["prob"].shape
    inside = (c[:, 0] >= 0) & (c[:, 0] < W) & (c[:, 1] >= 0) & (c[:, 1] < H)
    p = np.full(len(c), 255, np.int32)
    p[inside] = nm["prob"][c[inside, 1], c[inside, 0]]
    gt_occ = bev[iy, ix] == 2
    obs = p < 255
    pred_occ = p >= 127
    # distance of each labelled cell to the nearest keyframe (map frame, horizontal)
    kq = (nm["T_wc"][:, :3, 3] @ nm["R_map"].T)[:, :2]
    from scipy.spatial import cKDTree
    d = cKDTree(kq).query(q[:, :2])[0]
    out = {"align_rms_m": round(rms, 3), "labelled": int(len(c))}
    for name, sel in (("all", np.ones(len(c), bool)), ("<2m", d < 2), ("<4m", d < 4)):
        m = sel & obs
        tp = (pred_occ & gt_occ & m).sum()
        fp = (pred_occ & ~gt_occ & m).sum()
        fn = (~pred_occ & gt_occ & m).sum()
        tn = (~pred_occ & ~gt_occ & m).sum()
        io, iff = tp / max(tp + fp + fn, 1), tn / max(tn + fp + fn, 1)
        out[name] = {"iou_occ": round(float(io), 4), "iou_free": round(float(iff), 4), "miou": round(float(io + iff) / 2, 4),
                     "observed": round(float(m.sum() / max(sel.sum(), 1)), 4), "cells": int(sel.sum())}
    print(json.dumps(out))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
