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


def pairs(Qf, Xf, k=50, n_rand=50, exclude=None, seed=0):
    """(qi, xi) index pairs: k best full neighbours and n_rand random rows per query."""
    S = Qf @ Xf.T
    if exclude is not None:
        S[exclude] = -9
    top = S.topk(k, dim=1).indices
    g = torch.Generator(device="cpu").manual_seed(seed)
    rnd = torch.randint(0, Xf.shape[0], (Qf.shape[0], n_rand), generator=g).to(Qf.device)
    xi = torch.cat([top, rnd], 1)
    qi = torch.arange(Qf.shape[0], device=Qf.device)[:, None].expand_as(xi)
    is_top = torch.zeros_like(xi, dtype=torch.bool)
    is_top[:, :k] = True
    return qi.reshape(-1), xi.reshape(-1), is_top.reshape(-1)


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
            a_i, x_i, top = pairs(Qf, X, exclude=ex)
            full = (Qf[a_i] * X[x_i]).sum(1)
            s = (Zq[a_i] * Z[x_i]).sum(1)
            eq = Zq[a_i].pow(2).sum(1)
            ey = Z[x_i].pow(2).sum(1)
            row = {"dim": d, "set": name, "e_map": float(Z.pow(2).sum(1).mean()), "e_query": float(Zq.pow(2).sum(1).mean())}
            for model in ("raw", "iso", "resid"):
                pred = cal.apply(s, eq, ey, model=model)
                row[model] = metrics(pred, full, top)
            results.append(row)
            print(json.dumps(row), flush=True)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
