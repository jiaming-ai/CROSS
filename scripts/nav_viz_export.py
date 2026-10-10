"""Export BEV predictions and ground truth of fixed evaluation windows as images + manifest.json for a comparison page.

    python scripts/nav_viz_export.py --heads <heads.pt> --out <dir> --per-dataset 6 <windows.json> [<windows.json> ...]

Per window: frame 0 (f0.jpg) and the other frames (r<k>.jpg), GT label (gt.png: free / occupied, unknown transparent),
P(occupied) with GT scale + gravity (po.jpg) and with the scale head + gravity head (pm.jpg), error map on labelled
cells (err.png: correct free / correct occupied / false occupied / missed occupied), cells a GT depth point of the
window falls in (seen.png).  manifest.json: dataset, scene, frames, metrics per window.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from cross_nav.bev import FREE, OCC
from cross_nav.data import NavStream, WorldBEVs
from cross_nav.evaluate import load_window
from cross_nav.model import NavModel
from cross_nav.train import gt_log_scale

C_FREE = (238, 241, 243)        # BGR
C_OCC = (54, 47, 43)
C_FP = (60, 90, 230)            # predicted occupied, actually free (warm red)
C_FN = (230, 150, 40)           # missed occupied (blue)


def gt_png(gt):
    img = np.zeros(gt.shape + (4,), np.uint8)
    img[gt == FREE] = C_FREE + (255,)
    img[gt == OCC] = C_OCC + (255,)
    return img


def err_png(pred_occ, gt):
    img = np.zeros(gt.shape + (4,), np.uint8)
    lab = (gt == FREE) | (gt == OCC)
    go = gt == OCC
    img[lab & ~pred_occ & ~go] = C_FREE + (255,)
    img[lab & pred_occ & go] = C_OCC + (255,)
    img[lab & pred_occ & ~go] = C_FP + (255,)
    img[lab & ~pred_occ & go] = C_FN + (255,)
    return img


def prob_jpg(p):
    v = np.clip(0.94 - 0.82 * p, 0, 1)                      # light = free, dark = occupied
    return (v * 255).astype(np.uint8)


def miou(pred_occ, gt, sel):
    lab = ((gt == FREE) | (gt == OCC)) & sel
    go = gt == OCC
    tp = (pred_occ & go & lab).sum(); fp = (pred_occ & ~go & lab).sum()
    fn = (~pred_occ & go & lab).sum(); tn = (~pred_occ & ~go & lab).sum()
    if lab.sum() == 0:
        return None, None, None
    io, iff = tp / max(tp + fp + fn, 1), tn / max(tn + fp + fn, 1)
    return round(float((io + iff) / 2), 3), round(float(io), 3), round(float(iff), 3)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-dataset", type=int, default=6)
    ap.add_argument("windows", nargs="+")
    a = ap.parse_args()
    ck = torch.load(a.heads, map_location="cpu", weights_only=False)
    model = NavModel(ck["backbone"], ck.get("bev_cfg"), ck.get("gravity_cfg")).cuda().eval()
    model.bev_head.load_state_dict(ck["bev_head"])
    model.gravity_head.load_state_dict(ck["gravity_head"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for wpath in a.windows:
        W = json.load(open(wpath))
        st = NavStream({"bev_root": W["bev_root"], "datasets": {}, "cam_h_range": W.get("cam_h_range") or [0.1, 2.2]})
        st.bevs, st.label_ds = WorldBEVs(W["bev_root"]), {}
        scenes = {}
        by_ds = {}
        for w in W["windows"]:
            by_ds.setdefault(w["dataset"], []).append(w)
        setname = Path(wpath).stem
        for ds, ws in by_ds.items():
            # spread over the set, prefer multi-frame windows and different scenes
            picks, seen_scenes = [], set()
            order = sorted(range(len(ws)), key=lambda k: (len(ws[k]["frames"]) == 1, k % 7, k))
            for k in order:
                sc = ws[k]["frames"][0][0]
                if sc in seen_scenes and len(seen_scenes) < len({w["frames"][0][0] for w in ws}):
                    continue
                picks.append(ws[k]); seen_scenes.add(sc)
                if len(picks) >= a.per_dataset:
                    break
            for j, w in enumerate(picks):
                it = load_window(w, st, scenes)
                if it is None:
                    continue
                key = f"{setname}_{ds}_{j}"
                d = out / key
                d.mkdir(exist_ok=True)
                imgs = it["images"][None].cuda()
                bb = model.backbone(imgs)
                ls_gt = gt_log_scale(bb["depth"], it["depths"][None].cuda(), it["depths"][None].cuda() > 0)
                up_gt = it["up_cam"][None].cuda()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    up_pred = model.gravity(bb).float()
                    lo = model.heads(bb, ls_gt, up_gt).float()[0].cpu().numpy()
                    lm = model.heads(bb, bb["log_scale_head"].float(), up_pred).float()[0].cpu().numpy()
                po, pm = 1 / (1 + np.exp(-lo)), 1 / (1 + np.exp(-lm))
                gt, vis = it["bev"].numpy(), it["vis"].numpy()
                S = imgs.shape[1]
                for s in range(S):
                    im = (it["images"][s].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[..., ::-1]
                    wdt = 360 if s == 0 else 120
                    im = cv2.resize(im, (wdt, int(im.shape[0] * wdt / im.shape[1])), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(d / (f"f0.jpg" if s == 0 else f"r{s}.jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 82])
                cv2.imwrite(str(d / "gt.png"), gt_png(gt), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(str(d / "po.jpg"), prob_jpg(po), [cv2.IMWRITE_JPEG_QUALITY, 85])
                cv2.imwrite(str(d / "pm.jpg"), prob_jpg(pm), [cv2.IMWRITE_JPEG_QUALITY, 85])
                cv2.imwrite(str(d / "err.png"), err_png(po > 0.5, gt), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(str(d / "errm.png"), err_png(pm > 0.5, gt), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                sm = np.zeros(gt.shape + (4,), np.uint8)
                sm[(vis & 2) > 0] = (40, 200, 250, 120)
                cv2.imwrite(str(d / "seen.png"), sm, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                seen, uns = (vis & 2) > 0, (vis & 2) == 0
                mo = {k: miou(po > 0.5, gt, sel) for k, sel in (("seen", seen), ("unseen", uns), ("all", np.ones_like(seen)))}
                mm = {k: miou(pm > 0.5, gt, sel) for k, sel in (("seen", seen), ("unseen", uns), ("all", np.ones_like(seen)))}
                ang = math.degrees(math.acos(float((up_pred * up_gt).sum().clamp(-1, 1))))
                items.append({"key": key, "set": setname, "dataset": ds, "scene": Path(w["frames"][0][0]).name,
                              "seq": w["frames"][0][1], "frame": w["frames"][0][2], "S": S, "kind": w["kind"],
                              "cam_h": round(float(it["cam_h"]), 2), "oracle": mo, "mono": mm,
                              "gravity_err_deg": round(ang, 2),
                              "scale_err": round(abs(float(bb["log_scale_head"][0] - ls_gt[0])), 3),
                              "labelled": round(float(((gt == FREE) | (gt == OCC)).mean()), 3),
                              "seen_frac": round(float((((gt > 0) & seen).sum()) / max((gt > 0).sum(), 1)), 3)})
                print(key, mo["all"], flush=True)
    json.dump({"heads": a.heads, "items": items}, open(out / "manifest.json", "w"))


if __name__ == "__main__":
    main()
