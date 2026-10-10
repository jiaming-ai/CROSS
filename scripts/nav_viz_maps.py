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
from cross_nav.plan import evaluate_planning, summarize  # noqa: E402


def norm(p, W, H, step=3):
    """Path cells -> [x, y] in [0, 1] of the y-up image (None stays None)."""
    if p is None:
        return None
    q = p[::step] + ([p[-1]] if (len(p) - 1) % step else [])
    return [[round(c / W, 4), round(1 - r / H, 4)] for r, c in q]


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
    # planning on the fused map: starts at keyframes (every 10th), goals 5-20 m away; unobserved cells blocked for a
    # learned map (prob 255), geometric maps both ways (undecided free = optimistic, occupied = pessimistic)
    kc = np.floor((kq[:, :2] - org) / res).astype(int)
    starts = [(int(c[1]), int(c[0])) for c in kc[::10] if 0 <= c[1] < H and 0 <= c[0] < W]
    maps = {"map": ~obs | po, "map_free_unknown": obs & po}
    recs = evaluate_planning(gt, maps, res, n_goals=4, dist_range=(5.0, 20.0), starts=starts, seed=0)
    plan = {"summary": {m: summarize(recs, m) for m in maps},
            "goals": [[r["goal"][1] / W, 1 - r["goal"][0] / H] for r in recs[:12]],
            "starts": [[r["start"][1] / W, 1 - r["start"][0] / H] for r in recs[:12]],
            "status": [r["methods"]["map"]["status"] for r in recs[:12]],
            "paths": [norm(r["methods"]["map"]["path"], W, H) for r in recs[:12]],
            "gt": [norm(r["gt_path"], W, H) for r in recs[:12]]}
    kp = ((kq[:, :2] - org) / res)
    traj = [[round(float(x) / W, 4), round(float(1 - y / H), 4)] for x, y in kp]
    js = json.load(open(Path(a.navmap).with_name("eval.json"))) if Path(a.navmap).with_name("eval.json").exists() else {}
    json.dump({"name": a.name, "size": [W, H], "res": res, "traj": traj, "eval": js, "plan": plan},
              open(out / f"{a.name}.json", "w"))
    print(a.name, js.get("<2m"), js.get("all"), plan["summary"])


if __name__ == "__main__":
    main()
