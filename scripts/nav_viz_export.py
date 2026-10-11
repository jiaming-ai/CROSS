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

from cross_nav.bev import FREE, OCC, BEVSpec, bev_axes
from cross_nav.data import NavStream, WorldBEVs
from cross_nav.evaluate import geom_baseline, lift_np, load_window
from cross_nav.model import NavModel
from cross_nav.plan import evaluate_planning, summarize, risk_cost
from cross_nav.train import gt_log_scale
from cross_nav.uncertainty import error_metrics, uncertainty_score

SKIP = {"tartanair2"}           # drone trajectories: not ground navigation (user, 2026-10-10)

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


def geom_map(bb, ls, up, N, res, cam_h=None):
    """Geometric BEV of a pass: VGGT depth of all views lifted with scale exp(ls) and gravity up; floor height = the
    given camera height, else the densest height 0.2-2.5 m below the camera within 3 m.  0 undecided, 1 free, 2 occ."""
    A = bev_axes(up)[0].cpu().numpy()
    ext = bb["extrinsics"][0].float().cpu().numpy()
    intr = bb["intrinsics"][0].float().cpu().numpy()
    dp = bb["depth"][0].float().cpu().numpy() * math.exp(float(ls))
    conf = bb["depth_conf"][0].float().cpu().numpy()
    P, Wt = [], []
    for s in range(dp.shape[0]):
        pts, ok = lift_np(dp[s], intr[s], ext[s], A)
        P.append(pts)
        Wt.append(np.clip((conf[s][::2, ::2][ok] - 1) / 20, 0, 1) + 0.05)
    P, Wt = np.concatenate(P), np.concatenate(Wt)
    if cam_h is None:
        near = (np.hypot(P[:, 0], P[:, 1]) < 3.0) & (P[:, 2] < -0.2) & (P[:, 2] > -2.5)
        if near.sum() < 50:
            return np.zeros((N, N), np.uint8)
        hist = np.bincount(np.floor(-P[near, 2] / 0.02).astype(int), weights=Wt[near])
        cam_h = (int(np.argmax(hist)) + 0.5) * 0.02
    return geom_baseline(P, Wt, BEVSpec(N, res), cam_h)


def lab_png(m):
    """Binary / ternary map image: 1 free light, 2 occupied dark, 0 transparent."""
    img = np.zeros(m.shape + (4,), np.uint8)
    img[m == 1] = C_FREE + (255,)
    img[m == 2] = C_OCC + (255,)
    return img


def path_pts(p, N, step=3):
    if p is None:
        return None
    q = p[::step] + ([p[-1]] if (len(p) - 1) % step else [])
    return [[round(c / N, 4), round(r / N, 4)] for r, c in q]


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
    ap.add_argument("--backbone", help="Override checkpoint backbone location")
    ap.add_argument("--save-arrays", action="store_true", help="Cache float probabilities and GT for CPU planner sweeps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-dataset", type=int, default=6, help="0 = all windows")
    ap.add_argument("--no-images", action="store_true", help="planning statistics only")
    ap.add_argument("--skip-planning", action="store_true",
                    help="Export predictions without path searches; score saved arrays in a separate CPU job")
    ap.add_argument("--goals", type=int, default=5)
    ap.add_argument("windows", nargs="+")
    a = ap.parse_args()
    if a.skip_planning and not (a.save_arrays and a.no_images):
        ap.error("--skip-planning requires --save-arrays and --no-images")
    ck = torch.load(a.heads, map_location="cpu", weights_only=False)
    model = NavModel(a.backbone or ck["backbone"], ck.get("bev_cfg"), ck.get("gravity_cfg")).cuda().eval()
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
            if ds in SKIP:
                continue
            # spread over the set, prefer multi-frame windows and different scenes
            picks, seen_scenes = ([], set()) if a.per_dataset else (list(ws), {None})
            order = sorted(range(len(ws)), key=lambda k: (len(ws[k]["frames"]) == 1, k % 7, k)) if a.per_dataset else []
            for k in order:
                sc = ws[k]["frames"][0][0]
                if sc in seen_scenes and len(seen_scenes) < len({w["frames"][0][0] for w in ws}):
                    continue
                picks.append(ws[k]); seen_scenes.add(sc)
                if a.per_dataset and len(picks) >= a.per_dataset:
                    break
            for j, w in enumerate(picks):
                it = load_window(w, st, scenes)
                if it is None:
                    continue
                key = f"{setname}_{ds}_{j}"
                d = out / key
                if not a.no_images:
                    d.mkdir(exist_ok=True)
                imgs = it["images"][None].cuda()
                bb = model.backbone(imgs)
                ls_gt = gt_log_scale(bb["depth"], it["depths"][None].cuda(), it["depths"][None].cuda() > 0)
                up_gt = it["up_cam"][None].cuda()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    up_pred = model.gravity(bb).float()
                    pred_o = model.heads(bb, ls_gt, up_gt, return_aux=True)
                    pred_m = model.heads(bb, bb["log_scale_head"].float(), up_pred, return_aux=True)
                    lo = pred_o["logits"].float()[0].cpu().numpy()
                    lm = pred_m["logits"].float()[0].cpu().numpy()
                    uo = (uncertainty_score(pred_o["error_logits"][0], model.bev_head.uncertainty_mode).cpu().numpy()
                          if pred_o["error_logits"] is not None else None)
                    um = (uncertainty_score(pred_m["error_logits"][0], model.bev_head.uncertainty_mode).cpu().numpy()
                          if pred_m["error_logits"] is not None else None)
                po, pm = 1 / (1 + np.exp(-lo)), 1 / (1 + np.exp(-lm))
                gt, vis = it["bev"].numpy(), it["vis"].numpy()
                S = imgs.shape[1]
                N, res = gt.shape[0], model.bev_head.res
                go = geom_map(bb, ls_gt[0], up_gt, N, res, float(it["cam_h"]))
                gm = geom_map(bb, bb["log_scale_head"][0], up_pred, N, res, None)
                # planning: thresholded head maps; geometric maps with undecided cells free (optimistic) and occupied
                if a.save_arrays:
                    arrays = out / "arrays"
                    arrays.mkdir(exist_ok=True)
                    extra = {"uo": uo, "um": um} if uo is not None else {}
                    np.savez_compressed(arrays / (key + ".npz"), po=po, pm=pm, gt=gt, vis=vis,
                                        go=go, gm=gm, res=res, seed=j, **extra)
                plans = {}
                for cond, ph, gg, uu in (("oracle", po, go, uo), ("mono", pm, gm, um)):
                    maps = {"head": ph > 0.5, "head_t40": ph > 0.4, "head_t30": ph > 0.3, "geom": gg == 2,
                            "geom_pess": gg != 1}
                    maps["soft_risk"] = np.zeros_like(gt, dtype=bool)
                    costs = {"soft_risk": risk_cost(ph, res)}
                    if uu is not None:
                        maps["soft_uncertainty"] = np.zeros_like(gt, dtype=bool)
                        costs["soft_uncertainty"] = risk_cost(ph, res, uncertainty=uu, uncertainty_weight=4.0)
                    recs = ([] if a.skip_planning else
                            evaluate_planning(gt, maps, res, (N // 2, N // 2), n_goals=a.goals, seed=j, costs=costs))
                    plans[cond] = {"summary": {m: summarize(recs, m) for m in maps},
                                   "goals": [[round(r["goal"][1] / N, 4), round(r["goal"][0] / N, 4)] for r in recs],
                                   "gt": [path_pts(r["gt_path"], N) for r in recs],
                                   "status": {m: [r["methods"][m]["status"] for r in recs] for m in maps},
                                   "paths": {m: [path_pts(r["methods"][m]["path"], N) for r in recs] for m in maps}}
                if a.no_images:
                    items.append({"key": key, "set": setname, "dataset": ds, "S": S,
                                  "plan": {c: plans[c]["summary"] for c in plans},
                                  "plan_status": {c: plans[c]["status"] for c in plans}})
                    print(key, plans["oracle"]["summary"]["head"], plans["oracle"]["summary"]["geom"], flush=True)
                    continue
                for s in range(S if not a.no_images else 0):
                    im = (it["images"][s].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[..., ::-1]
                    wdt = 360 if s == 0 else 120
                    im = cv2.resize(im, (wdt, int(im.shape[0] * wdt / im.shape[1])), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(d / (f"f0.jpg" if s == 0 else f"r{s}.jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if uo is not None:
                    for name, uu, ph in (("uo", uo, po), ("um", um, pm)):
                        # Fixed scale: black = zero predicted error, yellow = one.
                        heat = cv2.applyColorMap(np.uint8(np.clip(uu, 0, 1) * 255), cv2.COLORMAP_INFERNO)
                        cv2.imwrite(str(d / (name + ".png")), heat)
                        selective = np.where(uu <= 0.2, np.where(ph > 0.5, 2, 1), 0)
                        cv2.imwrite(str(d / (name + "_selective.png")), lab_png(selective))
                cv2.imwrite(str(d / "gt.png"), gt_png(gt), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(str(d / "po.jpg"), prob_jpg(po), [cv2.IMWRITE_JPEG_QUALITY, 85])
                cv2.imwrite(str(d / "pm.jpg"), prob_jpg(pm), [cv2.IMWRITE_JPEG_QUALITY, 85])
                cv2.imwrite(str(d / "pt.png"), lab_png(np.where(po > 0.5, 2, 1)), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(str(d / "ptm.png"), lab_png(np.where(pm > 0.5, 2, 1)), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(str(d / "geo.png"), lab_png(go), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(str(d / "geom.png"), lab_png(gm), [cv2.IMWRITE_PNG_COMPRESSION, 9])
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
                              "seen_frac": round(float((((gt > 0) & seen).sum()) / max((gt > 0).sum(), 1)), 3),
                              "geom": {"oracle": {k: miou(go == 2, gt, sel & (go > 0)) for k, sel in (("all", np.ones_like(seen)),)},
                                       "mono": {k: miou(gm == 2, gt, sel & (gm > 0)) for k, sel in (("all", np.ones_like(seen)),)},
                                       "coverage": {"oracle": round(float(((go > 0) & (gt > 0)).sum() / max((gt > 0).sum(), 1)), 3),
                                                    "mono": round(float(((gm > 0) & (gt > 0)).sum() / max((gt > 0).sum(), 1)), 3)}},
                              "uncertainty": ({"oracle": error_metrics(po, uo, gt, seen, model.bev_head.uncertainty_mode),
                                               "mono": error_metrics(pm, um, gt, seen, model.bev_head.uncertainty_mode)} if uo is not None else None),
                              "occupancy_fraction": float((gt == OCC).sum() / max((gt > 0).sum(), 1)),
                              "plan": plans})
                print(key, mo["all"], flush=True)
    json.dump({"heads": a.heads, "uncertainty_kind": model.bev_head.uncertainty_mode, "planning_skipped": a.skip_planning, "items": items}, open(out / "manifest.json", "w"))


if __name__ == "__main__":
    main()
