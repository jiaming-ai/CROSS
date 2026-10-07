"""Latency of KeyframeDatabase.query (what one retrieval of the system costs, without the query's VPR embedding) vs
database size: full descriptors (the original database, while it fits), map-projected codes with exact re-scoring
from the full descriptors in RAM (mapping session) and codes only (a loaded map), the IVF backend, and a locality
query restricted to the rows near a position.

    python scripts/retrieval/db_query_bench.py --sizes 1e4 1e5 1e6 --dim 2048 --out db_query_bench.json
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cross.db.db import KeyframeDatabase   # noqa: E402
from cross.db.index import DescriptorIndex, PCAProjection, SpatialIndex   # noqa: E402


def make_db(X, dev, projection=None, keep_full=False, backend="exact"):
    """KeyframeDatabase over rows X (unit, float16 on CPU) without a VPR model; embedding of image i = X[i]."""
    n, D = X.shape
    db = KeyframeDatabase.__new__(KeyframeDatabase)
    db.device, db.top_k = dev, 10
    db.score_threshold_high = db.score_threshold_low = 0.3
    db.index = DescriptorIndex(D, device=dev, initial_capacity=n, projection=projection, backend=backend,
                               ivf_min_rows=0, ivf_nprobe=16)
    if projection is None:
        db.index.set_rows(X.to(dev, torch.float32), range(n))
    else:
        codes = torch.cat([projection.apply(X[i:i + 65536].to(dev).float()) for i in range(0, n, 65536)])
        db.index.set_rows(codes, range(n))
        if keep_full:      # a mapping session: the full descriptors of every row are in RAM
            db.index._full, db.index._full_ok = X, torch.ones(n, dtype=torch.bool)
            from cross.db.index import ScoreCalibration
            sel = torch.linspace(0, n - 1, min(n, 8192)).long()
            db.index.calibration = ScoreCalibration.fit(X[sel].to(dev).float(), db.index.buf[sel.to(dev)].float(), exclude=0)
            print("residual ratio q999", db.index.calibration.meta["resid_ratio_q999"], flush=True)
    kfs = [SimpleNamespace(id=i) for i in range(n)]
    db._row_kf, db._id_to_row = kfs, {i: i for i in range(n)}
    db._keyframe_by_atlas, db._index_to_atlas_idx, db._atlas_to_indices = {None: kfs}, {}, {None: []}
    db.spatial = SpatialIndex()
    return db


def timed(fn, n=30, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=float, default=[1e4, 1e5, 1e6])
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--full-gpu-max", type=float, default=1.2e5)
    ap.add_argument("--nclt", default="", help="NCLT desc root: real descriptors of --sessions (all cameras)")
    ap.add_argument("--sessions", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    g = torch.Generator().manual_seed(0)
    D = 16384
    rows = []
    base = torch.randn(256, D, generator=g)
    for size in a.sizes:
        n = int(size)
        if a.nclt:
            if "real" not in locals():
                parts = [np.load(os.path.join(a.nclt, s_, f"Cam{c}.npy"), mmap_mode="r") for s_ in a.sessions
                         for c in (1, 2, 3, 4, 5) if os.path.exists(os.path.join(a.nclt, s_, f"Cam{c}.npy"))]
                real = torch.from_numpy(np.concatenate([np.asarray(p_) for p_ in parts]))
                print(f"{real.shape[0]} real descriptors", flush=True)
            if n > real.shape[0]:
                continue
            X = real[torch.randperm(real.shape[0], generator=g)[:n]]
            q = X[n // 3].float().to(dev)
            X = torch.cat([X[:n // 3], X[n // 3 + 1:]])
            n = X.shape[0]
        else:
            # clustered unit descriptors (pessimistic: little energy in any subspace), float16 on the CPU
            X = torch.empty((n, D), dtype=torch.float16)
            for i in range(0, n, 65536):
                m = min(65536, n - i)
                c = torch.randint(0, 256, (m,), generator=g)
                X[i:i + m] = torch.nn.functional.normalize(base[c] + 2.0 * torch.randn(m, D, generator=g), dim=-1).half()
            q = torch.nn.functional.normalize(X[7].float() + 0.3 * torch.randn(D, generator=g), dim=-1).to(dev)
        proj = PCAProjection.fit(X[torch.randperm(n, generator=g)[:16384]].to(dev).float(), a.dim)
        row = {"n": n, "dim": a.dim}
        if n <= a.full_gpu_max:
            db = make_db(X, dev)
            db.vpr_model = SimpleNamespace(get_embedding=lambda _: q)
            row["full_query"] = timed(lambda: db.query(0))
            row["full_query_map_split"] = timed(lambda: db.query(0, max_kf_id=n // 2))
            del db
        for label, kw in (("codes_rescored", dict(keep_full=True)), ("codes_loaded", dict()),
                          ("codes_ivf_loaded", dict(backend="ivf"))):
            db = make_db(X, dev, projection=proj, **kw)
            db.vpr_model = SimpleNamespace(get_embedding=lambda _: q)
            row[label] = timed(lambda: db.query(0))
            if label == "codes_rescored":
                row[label + "_map_split"] = timed(lambda: db.query(0, max_kf_id=n // 2))
                near = np.arange(0, min(n, 3000))
                row[label + "_locality_3000rows"] = timed(lambda: db.query(0, rows=near, top_k=5))
            del db
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)
        json.dump({"gpu": torch.cuda.get_device_name(0), "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
