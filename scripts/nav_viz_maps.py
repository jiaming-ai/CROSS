"""Fused navigation map vs ground truth in the same grid, as images for the comparison page.

    python scripts/nav_viz_maps.py --navmap <dir>/navmap.npz --label <world>.npz --poses <poses_left.txt> --out <dir> --name house

Writes <out>/<name>_pred.jpg (P(occupied), unobserved transparent-free grey), <name>_gt.png, <name>_err.png and
<name>.json (alignment, metrics within 2 / 4 m and overall, keyframe pixels for the trajectory)."""
import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location("ev", str(Path(__file__).with_name("nav_eval_navmap.py")))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
from nav_viz_export import C_FN, C_FP, C_FREE, C_OCC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navmap", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    nm, lab = np.load(a.navmap), np.load(a.label)
    P = np.loadtxt(a.poses).reshape(-1, 4, 4)
    si = nm["source_index"]
    ok = si >= 0
    R, t = ev.umeyama_rigid(nm["T_wc"][ok, :3, 3], P[si[ok], :3, 3])
    H, W = nm["prob"].shape
    res, org, R_map = float(nm["res"]), nm["origin"], nm["R_map"]
    kq = nm["T_wc"][:, :3, 3] @ R_map.T
    z0 = float(np.median(kq[:, 2])) - 1.0
    ys, xs = np.mgrid[0:H, 0:W]
    q = np.stack([org[0] + (xs + 0.5) * res, org[1] + (ys + 0.5) * res, np.full(xs.shape, z0)], -1).reshape(-1, 3)
    pm = q @ R_map                                        # gravity map frame -> CROSS map frame
    pg = pm @ R.T + t                                     # -> label world
    g = pg @ lab["R_g"].T                                 # -> label gravity frame
    lres, lorg = float(lab["res"]), lab["origin"]
    li = np.floor((g[:, :2] - lorg) / lres).astype(int)
    LH, LW = lab["bev"].shape
    inside = (li[:, 0] >= 0) & (li[:, 0] < LW) & (li[:, 1] >= 0) & (li[:, 1] < LH)
    gt = np.zeros(H * W, np.uint8)
    gt[inside] = lab["bev"][li[inside, 1], li[inside, 0]]
    gt = gt.reshape(H, W)
    p = nm["prob"]
    obs = p < 255
    prob = np.where(obs, p / 254.0, 0.5)
    v = np.clip(0.94 - 0.82 * prob, 0, 1)
    pred = np.dstack([(v * 255).astype(np.uint8)] * 3)
    pred[~obs] = (205, 200, 196)
    po = obs & (p >= 127)
    labm = (gt == 1) | (gt == 2)
    err = np.zeros((H, W, 4), np.uint8)
    go = gt == 2
    err[labm & obs & ~po & ~go] = C_FREE + (255,)
    err[labm & obs & po & go] = C_OCC + (255,)
    err[labm & obs & po & ~go] = C_FP + (255,)
    err[labm & obs & ~po & go] = C_FN + (255,)
    gimg = np.zeros((H, W, 4), np.uint8)
    gimg[gt == 1] = C_FREE + (255,)
    gimg[gt == 2] = C_OCC + (255,)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    flip = lambda x: x[::-1]                              # image y up
    cv2.imwrite(str(out / f"{a.name}_pred.jpg"), flip(pred), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(out / f"{a.name}_gt.png"), flip(gimg), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(str(out / f"{a.name}_err.png"), flip(err), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    kp = ((kq[:, :2] - org) / res)
    traj = [[round(float(x) / W, 4), round(float(1 - y / H), 4)] for x, y in kp]
    js = json.load(open(Path(a.navmap).with_name("eval.json"))) if Path(a.navmap).with_name("eval.json").exists() else {}
    json.dump({"name": a.name, "size": [W, H], "res": res, "traj": traj, "eval": js}, open(out / f"{a.name}.json", "w"))
    print(a.name, js.get("<2m"), js.get("all"))


if __name__ == "__main__":
    main()
