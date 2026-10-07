"""Where should the descriptor projection come from?  Recall and score fidelity of PCA projections of the BoQ
descriptor, fitted on the dataset itself, on the other datasets (leave-one-out: a shipped generic projection) or on
NCLT, against the full 16384-d descriptor.

    python scripts/retrieval/projection_study.py --desc openloris.npz rover.npz kitti.npz simchange.npz [nclt.npz] \
        --dims 128 256 512 1024 --out study.json

Per dataset, database = the first sequence of each scene (OpenLORIS x-1, ROVER day, SimChange map; KITTI and NCLT: the
first folder), queries = the other sequences of the scene.  A query is correct at radius r when one of its top-k
database frames is within r (r per dataset).  Fidelity: over each query's 50 best database frames (full descriptor),
mean |cos_projected - cos_full|, and how often the two agree on the database's score threshold 0.3.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cross.db.index import PCAProjection   # noqa: E402

MODES = {"uc": {}, "pca": dict(center=True, normalize=True),
         "pcaw": dict(center=True, normalize=True, whiten_power=0.5)}
RADIUS = {"openloris": 2.0, "simchange": 2.0, "rover": 5.0, "kitti": 10.0, "nclt": 10.0}


def scene_of(label, folder):
    parts = folder.rstrip("/").split("/")
    if label == "openloris":
        name = [p for p in parts if re.match(r"^[a-z]+\d-\d", p)][0]
        return re.match(r"^([a-z]+)", name).group(1), name
    if label == "simchange":
        return parts[-2], parts[-1]
    if label == "rover":
        name = [p for p in parts if p.startswith("campus")][0]
        return "campus_large", name
    return label, parts[-1]


def split(d):
    """(db_index, query_index) arrays and per-row scene ids: per scene the map / day / x-1 sequence (else the first)
    is the database."""
    label = str(d["label"])
    folders = [str(f) for f in d["folders"]]
    info = [scene_of(label, f) for f in folders]
    scenes = sorted({s for s, _ in info})

    def rank(name):
        return 0 if (name == "map" or "day" in name or name.endswith("-1")) else 1

    first = {}
    for i, (s, name) in enumerate(info):
        if s not in first or rank(name) < rank(info[first[s]][1]):
            first[s] = i
    seq = d["seq"]
    is_db = np.isin(seq, list(first.values()))
    scene_row = np.array([scenes.index(info[i][0]) for i in seq])
    return np.where(is_db)[0], np.where(~is_db)[0], scene_row


def evaluate(Q, X, qpos, xpos, qscene, xscene, r, ks=(1, 5, 10), chunk=2048):
    """Recall@k of queries Q against database X (rows L2-normalized, torch on device), same-scene database only."""
    hits = {k: 0 for k in ks}
    n_ok = 0
    qpos_t = torch.as_tensor(qpos, device=X.device, dtype=torch.float32)
    xpos_t = torch.as_tensor(xpos, device=X.device, dtype=torch.float32)
    qs = torch.as_tensor(qscene, device=X.device)
    xs = torch.as_tensor(xscene, device=X.device)
    for i in range(0, Q.shape[0], chunk):
        S = Q[i:i + chunk] @ X.T
        S[qs[i:i + chunk, None] != xs[None]] = -2
        D = torch.cdist(qpos_t[i:i + chunk], xpos_t)
        D[qs[i:i + chunk, None] != xs[None]] = 1e9
        answerable = (D.min(1).values <= r)
        top = S.topk(max(ks), dim=1).indices
        close = torch.gather(D, 1, top) <= r
        for k in ks:
            hits[k] += int((close[:, :k].any(1) & answerable).sum())
        n_ok += int(answerable.sum())
    return {f"R@{k}": hits[k] / max(n_ok, 1) for k in ks} | {"n_queries": n_ok}


def fidelity(Qf, Xf, Qp, Xp, qscene, xscene, n=50, thr=0.3, chunk=1024):
    errs, agree, tot = [], 0, 0
    qs = torch.as_tensor(qscene, device=Xf.device)
    xs = torch.as_tensor(xscene, device=Xf.device)
    for i in range(0, Qf.shape[0], chunk):
        S = Qf[i:i + chunk] @ Xf.T
        S[qs[i:i + chunk, None] != xs[None]] = -2
        top = S.topk(n, dim=1).indices
        sf = torch.gather(S, 1, top)
        sp = torch.gather(Qp[i:i + chunk] @ Xp.T, 1, top)
        errs.append((sp - sf).abs().flatten())
        agree += int(((sp > thr) == (sf > thr)).sum())
        tot += sf.numel()
    e = torch.cat(errs)
    return {"mae_top50": float(e.mean()), "p95_top50": float(e.quantile(0.95)) if e.numel() < 2 ** 24 else float(e[:2 ** 24].quantile(0.95)),
            "thr_agree": agree / max(tot, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", nargs="+", required=True)
    ap.add_argument("--dims", nargs="+", type=int, default=[128, 256, 512, 1024])
    ap.add_argument("--modes", nargs="+", default=["uc", "pca"],
                    help="uc: uncentred codes, scores kept (CROSS default); pca: centred + renormalized; pcaw: + whitening")
    ap.add_argument("--fit-max", type=int, default=60000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-generic", default="", help="write the projections fitted on all datasets here (dir)")
    a = ap.parse_args()
    dev = "cuda"
    data = {}
    for p in a.desc:
        d = np.load(p, allow_pickle=False)
        label = str(d["label"])
        X = torch.nn.functional.normalize(torch.from_numpy(d["desc"].astype(np.float32)), dim=-1)
        dbi, qi, scene_row = split(d)
        data[label] = dict(X=X, pos=d["pos"], db=dbi, q=qi, scene=scene_row)
        print(f"{label}: {X.shape[0]} descriptors, db {len(dbi)}, queries {len(qi)}", flush=True)
    results = []
    g = torch.Generator().manual_seed(0)

    def fit_rows(labels):
        rows = torch.cat([data[l]["X"] for l in labels])
        if rows.shape[0] > a.fit_max:
            rows = rows[torch.randperm(rows.shape[0], generator=g)[: a.fit_max]]
        return rows.to(dev)

    for label, D in data.items():
        X = D["X"].to(dev)
        Xq, Xd = X[D["q"]], X[D["db"]]
        qpos, xpos = D["pos"][D["q"]], D["pos"][D["db"]]
        qsc, xsc = D["scene"][D["q"]], D["scene"][D["db"]]
        r = RADIUS.get(label, 5.0)
        t0 = time.perf_counter()
        base = evaluate(Xq, Xd, qpos, xpos, qsc, xsc, r)
        results.append(dict(dataset=label, source="full", dim=X.shape[1], mode="full", **base,
                            t_eval=time.perf_counter() - t0))
        print(label, "full", base, flush=True)
        sources = {"self": [label], "others": [l for l in data if l != label]}
        if "nclt" in data and label != "nclt":
            sources["nclt"] = ["nclt"]
        for src, labels in sources.items():
            if not labels:
                continue
            F = fit_rows(labels) if src != "self" else Xd
            for mode in a.modes:
                for dim in a.dims:
                    if dim >= min(F.shape):
                        continue
                    proj = PCAProjection.fit(F, dim, **MODES[mode])
                    Pq, Pd = proj.apply(Xq), proj.apply(Xd)
                    rec = evaluate(Pq, Pd, qpos, xpos, qsc, xsc, r)
                    fid = fidelity(Xq, Xd, Pq, Pd, qsc, xsc)
                    row = dict(dataset=label, source=src, dim=dim, mode=mode, explained=proj.meta["explained"], **rec, **fid)
                    results.append(row)
                    print(row, flush=True)
    if a.save_generic:
        os.makedirs(a.save_generic, exist_ok=True)
        F = fit_rows(list(data))
        for mode in a.modes:
            for dim in a.dims:
                proj = PCAProjection.fit(F, dim, meta={"fit_on": sorted(data)}, **MODES[mode])
                proj.save(os.path.join(a.save_generic, f"boq_{mode}{dim}.npz"))
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
