"""Held-out evaluation: relative pose, depth, metric scale and covisibility on fixed windows of scene caches.

    python -m vggt_ft.evaluate --ckpt <ckpt.pt | released.pt> --suite configs/vggt_ft/eval_suite.yaml --out res.json

Protocols (per test set in the suite):
  seq     S frames of one sequence, `stride` frames apart (consecutive frames overlap), deterministic starts.
  random  S frames drawn at random from one sequence (VGGT-Omega / DA3 protocol, S = 10).
  cross   a query frame of one session + `n_ref` frames of other sessions of the same place nearest to it (position and
          heading) + `n_neg` frames of other sessions far from it: cross-session pose and covisibility (CROSS's
          relocalization input).
  single  one frame (monocular metric depth).

Metrics: pose AUC@3/5/10/30 of max(rotation error, translation-direction error) over all frame pairs connected by
shared scene (VGGT protocol); depth AbsRel / delta<1.25 after per-frame median alignment (`d_*`) and after one scale per
window (`dw_*`, multi-view consistency); metric (scale head, no alignment): `m_absrel`, `m_d125`, `log_scale_err`
(|log s_pred - log s*|, s* = median GT / predicted depth); translation scale `t_ratio` (metric predicted / GT length of
relative translations > 0.1 units); covisibility AUROC of the head (`cov_auroc`) and of CROSS's geometric score from the
predicted depth and poses (`geo_auroc`), positives = GT overlap > 0.1, negatives = GT overlap < 0.02.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from vggt_omega.utils.pose_enc import encoding_to_camera

from .dataio.scene import list_scenes, load_scene
from .dataio.windows import materialise
from .dataio.transforms import target_shape
from .geometry import connected_to_ref, gt_covisibility
from .model import VGGTOmegaFT


# ---------------------------------------------------------------------------------------------------- windows
def build_windows(root, ts: dict, seed: int = 0) -> list[list[tuple[str, str, int]]]:
    """Deterministic window specs [(scene_dir, sequence, index), ...] for one test set."""
    rng = np.random.default_rng(seed)
    dirs = list_scenes(ts.get("root", root), ts["dataset"], ts.get("include"), ts.get("exclude"), ts.get("val_mod", 0),
                       "val")
    scenes = [load_scene(d) for d in dirs]
    proto, S = ts["protocol"], ts.get("frames", 8)
    per_seq = ts.get("per_seq", 4)
    wins = []
    if proto in ("seq", "random", "single"):
        for sc in scenes:
            for qn in sc.sequences:
                q = sc.seq(qn)
                for k in range(per_seq):
                    if proto == "seq":
                        st = ts.get("stride", 3)
                        span = st * (S - 1)
                        if q.n <= span:
                            continue
                        s0 = int(round((q.n - span - 1) * (k + 0.5) / per_seq))
                        idx = s0 + st * np.arange(S)
                    elif proto == "random":
                        if q.n < S:
                            continue
                        idx = np.sort(rng.choice(q.n, S, replace=False))
                    else:
                        idx = [int(rng.integers(q.n))]
                    wins.append([(str(sc.dir), qn, int(i)) for i in idx])
    elif proto == "cross":
        n_ref, n_neg = ts.get("n_ref", 5), ts.get("n_neg", 2)
        by_world = {}
        for sc in scenes:
            for qn in sc.sequences:
                by_world.setdefault(sc.world_id, []).append((sc, qn))
        for members in by_world.values():
            if len(members) < 2:
                continue
            for k in range(per_seq * len(members)):
                sa, qa = members[rng.integers(len(members))]
                A = sa.seq(qa)
                i = int(rng.integers(A.n))
                others = [(s, n) for s, n in members if not (s is sa and n == qa)]
                cand = []
                for s, n in others:
                    B = s.seq(n)
                    d = np.linalg.norm(B.centres - A.centres[i], axis=1)
                    ang = np.degrees(np.arccos(np.clip((np.einsum("ij,nij->n", A.E[i, :3, :3], B.E[:, :3, :3]) - 1) / 2,
                                                       -1, 1)))
                    unit = max(1e-3, float(np.median(np.load(B.dir / "frames.npz")["dmed"])))
                    cost = d / unit + ang / 30.0
                    for j in np.argsort(cost)[: 4 * n_ref]:
                        cand.append((cost[j], d[j] / unit, str(s.dir), n, int(j)))
                cand.sort()
                refs = cand[: n_ref * 3]
                if len(refs) < n_ref:
                    continue
                pick = [refs[j] for j in sorted(rng.choice(len(refs), n_ref, replace=False))]
                far = []
                for s, n in others:
                    B = s.seq(n)
                    unit = max(1e-3, float(np.median(np.load(B.dir / "frames.npz")["dmed"])))
                    d = np.linalg.norm(B.centres - A.centres[i], axis=1) / unit
                    far += [(str(s.dir), n, int(j)) for j in np.flatnonzero(d > ts.get("neg_dist", 4.0))]
                negs = [far[j] for j in rng.choice(len(far), min(n_neg, len(far)), replace=False)] if far else []
                wins.append([(str(sa.dir), qa, i)] + [(p[2], p[3], p[4]) for p in pick] + negs)
    if ts.get("max_windows") and len(wins) > ts["max_windows"]:
        sel = np.sort(rng.choice(len(wins), ts["max_windows"], replace=False))
        wins = [wins[i] for i in sel]
    return wins


# ---------------------------------------------------------------------------------------------------- metrics
def _rot_err(Ra, Rb):
    c = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.arccos(c.clamp(-1, 1)))


def pair_errors(E_pred, E_gt, valid):
    """Relative rotation / translation-direction errors (deg) of all ordered pairs i<j with valid[i] & valid[j]."""
    S = E_pred.shape[0]
    rre, rte, tg_len, tp_len = [], [], [], []
    for i in range(S):
        for j in range(i + 1, S):
            if not (valid[i] and valid[j]):
                continue
            Tp = E_pred[j] @ torch.linalg.inv(E_pred[i])
            Tg = E_gt[j] @ torch.linalg.inv(E_gt[i])
            rre.append(_rot_err(Tp[:3, :3], Tg[:3, :3]))
            tp, tg = Tp[:3, 3], Tg[:3, 3]
            c = (tp @ tg) / (tp.norm() * tg.norm()).clamp(min=1e-9)
            rte.append(torch.rad2deg(torch.arccos(c.clamp(-1, 1))))
            tg_len.append(tg.norm())
            tp_len.append(tp.norm())
    if not rre:
        return None
    return torch.stack(rre), torch.stack(rte), torch.stack(tg_len), torch.stack(tp_len)


def auc(errors: np.ndarray, max_t: int) -> float:
    """VGGT-style AUC: mean over integer thresholds 1..max_t of the fraction of pairs with error < t."""
    if errors.size == 0:
        return float("nan")
    return float(np.mean([(errors < t).mean() for t in range(1, max_t + 1)]))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def depth_metrics(pred, gt, mask, per_frame=True):
    """AbsRel and delta<1.25 after median alignment (per frame, or one scale for all frames)."""
    out_abs, out_d = [], []
    if per_frame:
        for p, g, m in zip(pred, gt, mask):
            if m.sum() < 50:
                continue
            s = (g[m] / p[m]).median()
            a = (s * p[m] - g[m]).abs() / g[m]
            r = torch.maximum(s * p[m] / g[m], g[m] / (s * p[m]))
            out_abs.append(a.mean())
            out_d.append((r < 1.25).float().mean())
    else:
        if mask.sum() >= 50:
            s = (gt[mask] / pred[mask]).median()
            a = (s * pred[mask] - gt[mask]).abs() / gt[mask]
            r = torch.maximum(s * pred[mask] / gt[mask], gt[mask] / (s * pred[mask]))
            out_abs.append(a.mean())
            out_d.append((r < 1.25).float().mean())
    if not out_abs:
        return None
    return float(torch.stack(out_abs).mean()), float(torch.stack(out_d).mean())


# ---------------------------------------------------------------------------------------------------- run
@torch.no_grad()
def evaluate_set(model, root, ts, windows, device="cuda", log_every=0, known_focal=True):
    root = ts.get("root", root)
    hw = target_shape(ts.get("aspect", 0.75), ts.get("area", 512 * 512))
    acc = {k: [] for k in ("rre", "rte", "d_abs", "d_d125", "dw_abs", "dw_d125", "m_abs", "m_d125", "lse", "t_ratio",
                           "cov_pos", "cov_neg", "geo_pos", "geo_neg")}
    cache = {}
    t0 = time.time()
    for wi, spec in enumerate(windows):
        frames = []
        for d, qn, i in spec:
            if d not in cache:
                cache[d] = load_scene(Path(d))
            sc = cache[d]
            frames.append((sc, sc.seq(qn), i))
        b = materialise(frames, hw)
        img = b["images"][None].to(device)
        # calibrated intrinsics (as in CROSS) for a canonical-camera scale head; known_focal=False: its predicted FoV
        pred = model(img, intrinsics=b["intrinsics"][None].to(device) if known_focal else None)
        depth_gt, mask = b["depths"].to(device), b["masks"].to(device)
        E_gt, K_gt = b["extrinsics"].to(device).double(), b["intrinsics"].to(device)
        S = img.shape[1]
        cov_gt = gt_covisibility(depth_gt[None], mask[None], E_gt[None].float(), K_gt[None], b["world_id"][None].to(device))[0]
        has_depth = bool(mask.any())
        # without GT depth (KITTI) connectivity is unknown: every frame of the sequence window counts
        conn = connected_to_ref(cov_gt[None], 0.05)[0].cpu().numpy() if has_depth else np.ones(S, bool)
        Ep, Kp = encoding_to_camera(pred["pose_enc"].float(), hw)
        Ep4 = torch.eye(4, device=device, dtype=torch.float64).repeat(S, 1, 1)
        Ep4[:, :3] = Ep[0].double()
        pdep = pred["depth"][0, ..., 0].float()
        if S > 1:
            pe = pair_errors(Ep4, E_gt, conn)
            if pe is not None:
                rre, rte, tgl, tpl = pe
                acc["rre"].append(rre.cpu().numpy())
                acc["rte"].append(rte.cpu().numpy())
                if "log_scale" in pred:
                    s = torch.exp(pred["log_scale"][0].double())
                    ok = tgl > 0.1 * float(depth_gt[mask].median()) if has_depth else tgl > 0.1
                    if b["metric"] and ok.any():
                        acc["t_ratio"].append((s * tpl[ok] / tgl[ok]).cpu().numpy())
            iu = torch.triu_indices(S, S, 1)
            g = cov_gt[iu[0], iu[1]].cpu().numpy() if has_depth else np.full(len(iu[0]), 0.05)   # no labels
            if "covis_logits" in pred:
                p = torch.sigmoid(pred["covis_logits"][0])[iu[0], iu[1]].cpu().numpy()
                acc["cov_pos"].append(p[g > 0.1])
                acc["cov_neg"].append(p[g < 0.02])
            geo = gt_covisibility(pdep[None], (pdep > 0)[None], Ep4[None].float(), Kp, None)[0]
            q = geo[iu[0], iu[1]].cpu().numpy()
            acc["geo_pos"].append(q[g > 0.1])
            acc["geo_neg"].append(q[g < 0.02])
        if has_depth:
            m = mask & torch.from_numpy(conn).to(device)[:, None, None]
            r = depth_metrics(pdep, depth_gt, m)
            if r:
                acc["d_abs"].append(r[0])
                acc["d_d125"].append(r[1])
            r = depth_metrics(pdep, depth_gt, m, per_frame=False)
            if r:
                acc["dw_abs"].append(r[0])
                acc["dw_d125"].append(r[1])
            if "log_scale" in pred and b["metric"] and m.sum() > 50:
                s = torch.exp(pred["log_scale"][0].float())
                pm, gm = s * pdep[m], depth_gt[m]
                acc["m_abs"].append(float(((pm - gm).abs() / gm).mean()))
                acc["m_d125"].append(float((torch.maximum(pm / gm, gm / pm) < 1.25).float().mean()))
                acc["lse"].append(float((torch.log(s) - torch.log(gm / pdep[m]).median()).abs()))
        if log_every and (wi + 1) % log_every == 0:
            print(f"  {ts['name']}: {wi + 1}/{len(windows)} {time.time() - t0:.0f}s", flush=True)
    res = {"windows": len(windows)}
    if acc["rre"]:
        err = np.maximum(np.concatenate(acc["rre"]), np.concatenate(acc["rte"]))
        res.update({f"auc{t}": auc(err, t) for t in (3, 5, 10, 30)})
        res["rre_med"] = float(np.median(np.concatenate(acc["rre"])))
        res["rte_med"] = float(np.median(np.concatenate(acc["rte"])))
    for k in ("d_abs", "d_d125", "dw_abs", "dw_d125", "m_abs", "m_d125", "lse"):
        if acc[k]:
            res[k] = float(np.mean(acc[k]))
    if acc["t_ratio"]:
        tr = np.concatenate(acc["t_ratio"])
        res["t_ratio_med"] = float(np.median(tr))
        res["t_ratio_absrel"] = float(np.median(np.abs(tr - 1)))
    for src in ("cov", "geo"):
        if acc[f"{src}_pos"]:
            pos, neg = np.concatenate(acc[f"{src}_pos"]), np.concatenate(acc[f"{src}_neg"])
            res[f"{src}_auroc"] = auroc(pos, neg)
            res[f"{src}_n"] = [int(pos.size), int(neg.size)]
            if src == "cov":
                res["cov_tpr@0.4"] = float((pos >= 0.4).mean()) if pos.size else float("nan")
                res["cov_fpr@0.4"] = float((neg >= 0.4).mean()) if neg.size else float("nan")
    return res


def load_model(ckpt: str, device="cuda"):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    head_cfg = sd.get("scale_head_cfg") if isinstance(sd, dict) else None
    model = VGGTOmegaFT(scale_head_cfg=head_cfg)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    missing, _ = model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()},
                                       strict=False)
    if any(m.startswith("scale_head.") for m in missing):
        model.scale_head = None
    if any(m.startswith("covis_head.") for m in missing):
        model.covis_head = None
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="*")
    ap.add_argument("--pred_focal", action="store_true", help="canonical-camera scale head: predicted, not known focal")
    a = ap.parse_args()
    suite = yaml.safe_load(open(a.suite))
    model = load_model(a.ckpt)
    wdir = Path(suite.get("windows_dir", Path(a.suite).with_suffix("")))
    wdir.mkdir(parents=True, exist_ok=True)
    out = json.load(open(a.out)) if Path(a.out).exists() else {}
    out.setdefault("ckpt", a.ckpt)
    for ts in suite["sets"]:
        if a.sets and ts["name"] not in a.sets:
            continue
        wf = wdir / f"{ts['name']}.json"
        if wf.exists():
            wins = json.load(open(wf))
        else:
            wins = build_windows(suite["root"], ts)
            json.dump(wins, open(wf, "w"))
        t0 = time.time()
        res = evaluate_set(model, suite["root"], ts, wins, log_every=100, known_focal=not a.pred_focal)
        res["sec"] = round(time.time() - t0, 1)
        out[ts["name"]] = res
        print(ts["name"], json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}),
              flush=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
