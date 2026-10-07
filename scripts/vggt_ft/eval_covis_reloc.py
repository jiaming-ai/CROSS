#!/usr/bin/env python3
"""Covisibility head vs ground truth in CROSS-like relocalization windows (evaluation only).

For every `step`-th frame of each query session: window = [query i, query i-3, the `n_map` ground-truth-nearest map
frames facing the same way (view cos > 0.3), one far map frame], and a second copy of the window with `n_dist`
consecutive frames of another scene's map appended (distractors, covisibility 0 by construction).  Ground-truth
covisibility comes from the sensor depth and poses with the same function as CROSS's geometric gate.

Records per model: head covisibility (sigmoid of the logits, optionally Platt-calibrated: --platt a b) and the model's
geometric covisibility for query-map, map-map and distractor pairs; the rotation / translation-direction error of the
query against each map frame, with and without the distractors.

  eval_covis_reloc.py --data <openloris root> --out res.json --ckpt name=path [name=path ...]
"""
from __future__ import annotations

import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from vggt_ft.dataio.transforms import centred_crop_resize, target_shape
from vggt_ft.evaluate import load_model
from vggt_omega.utils.pose_enc import encoding_to_camera

QUERIES = {"office": ["office1-2", "office1-3", "office1-4", "office1-5", "office1-6", "office1-7"],
           "home": ["home1-2", "home1-3", "home1-4", "home1-5"], "cafe": ["cafe1-2"]}
OTHER = {"office": "home1-1", "home": "cafe1-1", "cafe": "office1-1"}       # distractor source per scene


def covis(depth, conf, c2w, K, src, dst, grid=48, tol=0.15, q=0.3):
    """cross.cv.pose_est_ff.covisibility_scores for a list of (source, destination) pairs; depth (S,H,W) torch."""
    S, H, W = depth.shape
    dev = depth.device
    ys, xs = torch.linspace(0, H - 1, grid, device=dev), torch.linspace(0, W - 1, grid, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gy, gx = gy.reshape(-1), gx.reshape(-1)
    a, b = torch.as_tensor(src, device=dev), torch.as_tensor(dst, device=dev)
    n = len(src)
    norm = torch.stack([gx / (W - 1) * 2 - 1, gy / (H - 1) * 2 - 1], -1).view(1, 1, -1, 2).expand(n, 1, -1, 2)
    d = F.grid_sample(depth[a][:, None], norm, align_corners=True).view(n, -1)
    valid = torch.isfinite(d) & (d > 1e-6)
    if conf is not None:
        c = F.grid_sample(conf[a][:, None], norm, align_corners=True).view(n, -1)
        thr = torch.quantile(conf[a].flatten(1)[:, :: max(1, (H * W) // 20000)], q, dim=1)
        valid &= c >= thr[:, None]
    c2w_t = torch.as_tensor(c2w, device=dev, dtype=torch.float32)
    K_t = torch.as_tensor(K, device=dev, dtype=torch.float32)
    Ks, Kd = K_t[a], K_t[b]
    x = (gx - Ks[:, 0, 2:3]) / Ks[:, 0, 0:1] * d
    y = (gy - Ks[:, 1, 2:3]) / Ks[:, 1, 1:2] * d
    P = torch.stack([x, y, d, torch.ones_like(d)], 1)
    Pd = (torch.linalg.inv(c2w_t[b]) @ c2w_t[a] @ P)[:, :3]
    z = Pd[:, 2]
    u = Kd[:, 0, 0:1] * Pd[:, 0] / z.clamp(min=1e-6) + Kd[:, 0, 2:3]
    v = Kd[:, 1, 1:2] * Pd[:, 1] / z.clamp(min=1e-6) + Kd[:, 1, 2:3]
    inside = (z > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    un = torch.stack([u / (W - 1) * 2 - 1, v / (H - 1) * 2 - 1], -1)[:, None]
    zd = F.grid_sample(depth[b][:, None], un, align_corners=True).view(n, -1)
    ok = inside & ((z - zd).abs() <= tol * zd.clamp(min=1e-6)) & valid
    nv = valid.sum(1)
    return torch.where(nv >= 16, ok.sum(1).double() / nv.clamp(min=1).double(), torch.zeros_like(nv, dtype=torch.float64)).cpu().numpy()


def rot_err(R1, R2):
    return float(np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1))))


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True, help="name=path; a path may be followed by :a,b (Platt)")
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--n_map", type=int, default=4)
    ap.add_argument("--n_dist", type=int, default=2)
    a = ap.parse_args()
    hw = target_shape(480 / 848)

    def seq(name):
        d = f"{a.data}/{name}/rgbd"
        names = sorted(os.listdir(f"{d}/rgb"))
        return d, names, np.loadtxt(f"{d}/poses_left.txt").reshape(-1, 4, 4), np.array(json.load(open(f"{d}/calib.json"))["K"])

    def load(d, names, i, K):
        im = cv2.cvtColor(cv2.imread(f"{d}/rgb/{names[i]}"), cv2.COLOR_BGR2RGB)
        dep = cv2.imread(f"{d}/depth/{names[i]}", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        im2, dep2, K2 = centred_crop_resize(im, dep, K.copy(), hw, 1.0)
        return torch.from_numpy(np.ascontiguousarray(im2)).permute(2, 0, 1).float() / 255.0, dep2, K2

    models = {}
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        platt = None
        if ":" in path:
            path, ab = path.rsplit(":", 1)
            platt = tuple(float(x) for x in ab.split(","))
        models[name] = (load_model(path), platt)
    rng = random.Random(0)
    recs = []
    for scene, qs in QUERIES.items():
        md, mn, MP, Km = seq(f"{scene}1-1")
        od, on, OP, Ko = seq(OTHER[scene])
        for q in qs:
            qd, qn, QP, Kq = seq(q)
            for i in range(6, len(qn), a.step):
                d = np.linalg.norm(MP[:, :3, 3] - QP[i][:3, 3], axis=1)
                cosv = MP[:, :3, 2] @ QP[i][:3, 2]
                cand = np.where(cosv > 0.3)[0]
                cand = cand[np.argsort(d[cand])]
                js = []
                for j in cand:
                    if all(abs(j - o) > 10 for o in js):
                        js.append(int(j))
                    if len(js) == a.n_map:
                        break
                if len(js) < 2:
                    continue
                far = [j for j in range(len(mn)) if cosv[j] < 0 or d[j] > 4]
                if far:
                    js.append(int(rng.choice(far)))
                ids = [("q", i), ("q", max(i - 3, 0))] + [("m", j) for j in js]
                o0 = rng.randrange(0, len(on) - 6)
                dist = [("o", o0 + 3 * k) for k in range(a.n_dist)]
                L = {}
                for s, x in ids + dist:
                    src = {"q": (qd, qn, Kq), "m": (md, mn, Km), "o": (od, on, Ko)}[s]
                    L[(s, x)] = load(src[0], src[1], x, src[2])
                GT = np.stack([QP[x] if s == "q" else MP[x] for s, x in ids])
                n = len(ids)
                gt_dep = torch.from_numpy(np.stack([L[k][1] for k in ids]))
                gt_K = np.stack([L[k][2] for k in ids])
                pairs = [(u, v) for u in range(n) for v in range(n) if u != v and not (u < 2 and v < 2)]
                gtc = covis(gt_dep, None, GT, gt_K, [p[0] for p in pairs], [p[1] for p in pairs])
                G = {p: c for p, c in zip(pairs, gtc)}
                rec = {"scene": scene, "query": q, "frame": i, "map": js, "far": len(js) > a.n_map,
                       "gt": {f"{u},{v}": float(min(G[(u, v)], G[(v, u)])) for u, v in pairs if u < v},
                       "dist_m": {str(k): float(np.linalg.norm(GT[k][:3, 3] - GT[0][:3, 3])) for k in range(2, n)},
                       "dist_mm": {f"{u},{v}": float(np.linalg.norm(GT[u][:3, 3] - GT[v][:3, 3])) for u, v in pairs if 2 <= u < v},
                       "view_ang": {str(k): rot_err(GT[k][:3, :3], GT[0][:3, :3]) for k in range(2, n)}}
                for name, (m, platt) in models.items():
                    for with_d in (False, True):
                        keys = ids + (dist if with_d else [])
                        views = torch.stack([L[k][0] for k in keys])[None].cuda()
                        with torch.no_grad():
                            o = m(views)
                        E, Ki = encoding_to_camera(o["pose_enc"].float(), hw)
                        w2c = np.tile(np.eye(4), (len(keys), 1, 1))
                        w2c[:, :3] = E[0].cpu().numpy()
                        c2w = np.linalg.inv(w2c)
                        r = {}
                        if "covis_logits" in o:
                            lg = o["covis_logits"][0].float()
                            if platt is not None:
                                lg = platt[0] * lg + platt[1]
                            r["head"] = torch.sigmoid(lg).cpu().numpy().round(4).tolist()
                        dep = o["depth"][0, ..., 0].float()
                        cf = o["depth_conf"][0].float()
                        pp = [(u, v) for u in range(len(keys)) for v in range(len(keys)) if u != v]
                        gc = covis(dep, cf, c2w, Ki[0].cpu().numpy(), [p[0] for p in pp], [p[1] for p in pp])
                        Cg = np.ones((len(keys), len(keys)))
                        for (u, v), c in zip(pp, gc):
                            Cg[u, v] = c
                        r["geo"] = Cg.round(4).tolist()
                        re, te = {}, {}
                        for k in range(2, n):
                            Tg = np.linalg.inv(GT[k]) @ GT[0]
                            Tp = np.linalg.inv(c2w[k]) @ c2w[0]
                            re[str(k)] = rot_err(Tg[:3, :3], Tp[:3, :3])
                            if np.linalg.norm(Tg[:3, 3]) > 0.2:
                                te[str(k)] = ang(Tg[:3, 3], Tp[:3, 3])
                        r["rot"], r["tdir"] = re, te
                        rec[f"{name}{'+d' if with_d else ''}"] = r
                recs.append(rec)
            print(q, "done", len(recs), flush=True)
    json.dump(recs, open(a.out, "w"))


if __name__ == "__main__":
    main()
