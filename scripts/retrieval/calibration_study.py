"""Calibration of projected (code) scores to the full-descriptor cosine, for maps whose full descriptors are gone
(loaded maps).  Fit on the map's own pairs, evaluate on held-out map pairs and on queries of another session.

    python scripts/retrieval/calibration_study.py --nclt <desc> --nclt-prepared <prepared> --map 2012-01-08 \
        --query 2012-08-04 --cams 5 --dims 512 1024 2048 --out calib.json

Models (s = code inner product, e = |code|^2 of a unit descriptor = explained energy, r = sqrt(1 - e)):
  raw        s
  iso        isotonic map of s fitted on map pairs
  resid      s + beta r_q r_y cos(z_q, z_y) + c, (beta, c) least squares on map pairs (the residual inner product
             modelled as a fraction of its bound)
Pairs per query: its 50 best full-descriptor neighbours (outside +-20 frames for map queries) and 50 random rows.
Metrics: MAE (all pairs / best neighbours), agreement of the decisions s > 0.3 and s > 0.75 with the full cosine.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from cross.db.index import PCAProjection, ScoreCalibration   # noqa: E402
from stress_test import load_nclt, concat, to_rows   # noqa: E402



def pair_scores(Qf, X, Zq, Z, k=50, n_rand=50, exclude=None, seed=0, chunk=128):
    """Full and code scores (+ energies) of each query with its k best full neighbours and n_rand random rows."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ez, eq_all = Z.pow(2).sum(1), Zq.pow(2).sum(1)
    out = {k_: [] for k_ in ("full", "s", "eq", "ey", "top")}
    for i in range(0, Qf.shape[0], chunk):
        S = Qf[i:i + chunk] @ X.T
        Sm = S.clone()
        if exclude is not None:
            Sm[exclude[i:i + chunk]] = -9
        top = Sm.topk(k, dim=1).indices
        rnd = torch.randint(0, X.shape[0], (S.shape[0], n_rand), generator=g).to(S.device)
        xi = torch.cat([top, rnd], 1)
        out["full"].append(torch.gather(S, 1, xi).reshape(-1))
        out["s"].append(torch.gather(Zq[i:i + chunk] @ Z.T, 1, xi).reshape(-1))
        out["eq"].append(eq_all[i:i + chunk][:, None].expand_as(xi).reshape(-1))
        out["ey"].append(ez[xi].reshape(-1))
        t = torch.zeros_like(xi, dtype=torch.bool)
        t[:, :k] = True
        out["top"].append(t.reshape(-1))
    return {k_: torch.cat(v) for k_, v in out.items()}


def metrics(pred, full, is_top):
    e = (pred - full).abs()
    out = {"mae_all": float(e.mean()), "mae_top": float(e[is_top].mean()), "p95_top": float(e[is_top].quantile(0.95)),
           "bias_top": float((pred - full)[is_top].mean())}
    for t in (0.3, 0.75):
        out[f"agree_{t}"] = float(((pred > t) == (full > t)).float().mean())
        out[f"agree_{t}_top"] = float(((pred > t) == (full > t))[is_top].float().mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nclt", required=True)
    ap.add_argument("--nclt-prepared", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--cams", nargs="+", type=int, default=[5])
    ap.add_argument("--dims", nargs="+", type=int, default=[512, 1024, 2048])
    ap.add_argument("--n-q", type=int, default=1500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(0)
    M = concat(load_nclt(a.nclt, a.nclt_prepared, [a.map], a.cams))
    Q = concat(load_nclt(a.nclt, a.nclt_prepared, [a.query], a.cams))
    X = to_rows(M["desc"], np.arange(M["desc"].shape[0]), dev)
    qi = np.sort(rng.choice(Q["desc"].shape[0], a.n_q, replace=False))
    Xq = to_rows(Q["desc"], qi, dev)
    mi = np.sort(rng.choice(X.shape[0], a.n_q, replace=False))      # map rows used as queries (held out of the fit)
    fit_rows = np.setdiff1d(np.arange(X.shape[0]), mi)
    excl = (torch.as_tensor(mi, device=dev)[:, None] - torch.arange(X.shape[0], device=dev)[None]).abs() <= 20
    results = []
    for d in a.dims:
        proj = PCAProjection.fit(X[fit_rows[rng.permutation(len(fit_rows))[:16384]]], d)
        Z = proj.apply(X)
        cal = ScoreCalibration.fit(X[fit_rows], Z[fit_rows])            # on the map's own pairs
        for name, Qf, ex in (("map_heldout", X[mi], excl), ("other_session", Xq, None)):
            Zq = proj.apply(Qf)
            P = pair_scores(Qf, X, Zq, Z, exclude=ex)
            full, s, eq, ey, top = P["full"], P["s"], P["eq"], P["ey"], P["top"]
            row = {"dim": d, "set": name, "e_map": float(Z.pow(2).sum(1).mean()), "e_query": float(Zq.pow(2).sum(1).mean())}
            for model in ("raw", "iso", "resid"):
                pred = cal.apply(s, eq, ey, model=model)
                row[model] = metrics(pred, full, top)
            results.append(row)
            print(json.dumps(row), flush=True)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
