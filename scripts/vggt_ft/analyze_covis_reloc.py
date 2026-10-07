#!/usr/bin/env python3
"""Summary of eval_covis_reloc.py results: covisibility of query-map pairs (CROSS's gate: reference -> query), map-map
pairs (the anchor filter: both ways) and distractor pairs (ground truth 0) for each model's head and geometric score,
and the query's pose error against the overlapping map frames with and without distractors.

  analyze_covis_reloc.py res.json [res2.json ...]   (records of several files are merged per model name)
"""
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

recs = {}
for f in sys.argv[1:]:
    for r in json.load(open(f)):
        recs.setdefault((r["query"], r["frame"]), {}).update(r)
recs = list(recs.values())
models = sorted({k for r in recs for k in r if k not in ("scene", "query", "frame", "map", "far", "gt", "dist_m",
                                                          "dist_mm", "view_ang") and not k.endswith("+d")})
POS, NEG, GATE = 0.15, 0.05, 0.15


def summarise(y, s, sel_name, extra=None):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = y >= POS, y < NEG
    out = {"n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
    if pos.sum() and neg.sum():
        m = pos | neg
        out["auc"] = roc_auc_score(pos[m], s[m])
    out["tpr"] = float(np.mean(s[pos] >= GATE)) if pos.sum() else float("nan")
    out["fpr"] = float(np.mean(s[neg] >= GATE)) if neg.sum() else float("nan")
    return out


def table(title, rows):
    print(f"\n== {title}")
    for name, d in rows:
        print(f"   {name:28s} " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items()))


for m in models:
    qm = {"y": [], "head": [], "geo": [], "dist": [], "ang": []}
    mm = {"y": [], "head": [], "geo": [], "dist": []}
    dneg = {"head": [], "geo": []}
    pose = {"rot": [], "rot_d": [], "tdir": [], "tdir_d": []}
    calib = []
    for r in recs:
        if m not in r:
            continue
        n = 2 + len(r["map"])
        R, Rd = r[m], r.get(m + "+d")
        for k in range(2, n):
            y = r["gt"][f"0,{k}"]
            qm["y"].append(y)
            qm["dist"].append(r["dist_m"][str(k)])
            qm["ang"].append(r["view_ang"][str(k)])
            qm["geo"].append(R["geo"][k][0])
            if "head" in R:
                qm["head"].append(R["head"][0][k])
            if y >= POS:
                pose["rot"].append(R["rot"][str(k)])
                if str(k) in R["tdir"]:
                    pose["tdir"].append(R["tdir"][str(k)])
                if Rd is not None:
                    pose["rot_d"].append(Rd["rot"][str(k)])
                    if str(k) in Rd["tdir"]:
                        pose["tdir_d"].append(Rd["tdir"][str(k)])
        for u in range(2, n):
            for v in range(u + 1, n):
                mm["y"].append(r["gt"][f"{u},{v}"])
                mm["dist"].append(r["dist_mm"][f"{u},{v}"])
                mm["geo"].append(min(R["geo"][u][v], R["geo"][v][u]))
                if "head" in R:
                    mm["head"].append(R["head"][u][v])
        if Rd is not None:
            nd = len(Rd["geo"])
            for dd in range(n, nd):
                for k in range(n):
                    dneg["geo"].append(Rd["geo"][dd][k] if k >= 2 else Rd["geo"][dd][k])
                    if "head" in Rd:
                        dneg["head"].append(Rd["head"][dd][k])
    print(f"\n######## {m}")
    for src in ("head", "geo"):
        if not qm[src]:
            continue
        y, s = np.array(qm["y"]), np.array(qm[src])
        dist, angv = np.array(qm["dist"]), np.array(qm["ang"])
        rows = [("all", summarise(y, s, "all")),
                ("dist < 1 m", summarise(y[dist < 1], s[dist < 1], "")),
                ("dist >= 1 m", summarise(y[dist >= 1], s[dist >= 1], "")),
                ("view angle >= 40 deg", summarise(y[angv >= 40], s[angv >= 40], ""))]
        table(f"{m} {src}: query-map (reference -> query)", rows)
        bins = [(0, .05), (.05, .15), (.15, .3), (.3, .5), (.5, 1.01)]
        print("   calibration (GT bin: mean score / mean GT): " + "  ".join(
            f"[{lo:.2f},{hi:.2f}) {s[(y >= lo) & (y < hi)].mean():.3f}/{y[(y >= lo) & (y < hi)].mean():.3f}"
            for lo, hi in bins if ((y >= lo) & (y < hi)).any()))
        if mm[src]:
            y2, s2, d2 = np.array(mm["y"]), np.array(mm[src]), np.array(mm["dist"])
            table(f"{m} {src}: map-map (both ways)", [("all", summarise(y2, s2, "")),
                                                     ("dist >= 1 m", summarise(y2[d2 >= 1], s2[d2 >= 1], ""))])
        if dneg[src]:
            s3 = np.array(dneg[src])
            print(f"   distractor pairs (GT 0): n {len(s3)}  FPR@{GATE} {np.mean(s3 >= GATE):.3f}  mean {s3.mean():.3f}"
                  f"  p95 {np.quantile(s3, .95):.3f}")
    if pose["rot"]:
        r0, r1 = np.array(pose["rot"]), np.array(pose["rot_d"]) if pose["rot_d"] else None
        t0, t1 = np.array(pose["tdir"]), np.array(pose["tdir_d"]) if pose["tdir_d"] else None
        msg = f"   pose (overlapping map frames, n {len(r0)}): rot med {np.median(r0):.2f} >10deg {np.mean(r0 > 10):.3f}" \
              f" tdir med {np.median(t0):.2f} >30deg {np.mean(t0 > 30):.3f}"
        if r1 is not None and len(r1):
            msg += f" | with distractors: rot med {np.median(r1):.2f} >10deg {np.mean(r1 > 10):.3f}" \
                   f" tdir med {np.median(t1):.2f} >30deg {np.mean(t1 > 30):.3f}"
        print(msg)
