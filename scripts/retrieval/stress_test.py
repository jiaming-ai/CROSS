"""Scaling of keyframe retrieval: memory, build time, query latency and recall from 10^3 to millions of keyframes, for
the full BoQ descriptor, projected codes (cross.db.index.PCAProjection), the IVF backend, and with location priors
(the belief's predicted position with drift, or a GPS fix with its quality).

    python scripts/retrieval/stress_test.py --db db.npz [more ...] --queries q.npz --projection boq_uc512.npz \
        --sizes 1000 10000 100000 1000000 0 --out results.json

A descriptor set is an .npz (scripts/retrieval/extract_descriptors.py: desc, pos[, gps, gps_sigma, session]) or a
directory with desc.npy (memory-mapped, float16, N x 16384) and meta.npz (pos, gps, gps_sigma, session).  Size 0 = the
whole database.  Smaller databases are random subsets (same area, lower density), drawn once per size.

Methods (each evaluated on the same queries):
  full        exact search of the full descriptors (GPU, database streamed in chunks)
  code        exact search of the projected codes (GPU fp16 / CPU fp32)
  ivf         the IVF backend of DescriptorIndex on the codes (nprobe cells)
  belief:S    code search + locality slots around a predicted position = truth + N(0, S^2) per horizontal axis
              (radius r_min + 3 S), as System._locality_merge does (slots local, rest global)
  gps         code search + locality slots around the query's GPS fix (radius r_min + 3 sigma_gps; no fix: global only)
Recall@k: a query counts when the database has a frame within the radius; correct when one of the top k is.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cross.db.index import DescriptorIndex, PCAProjection, SpatialIndex   # noqa: E402


# --------------------------------------------------------------------------------------------------------------------
def load_set(path):
    """dict(desc (N, D) float16 array or memmap, pos (N, 3), gps (N, 3) or None, gps_sigma (N,) or None, session (N,))."""
    if os.path.isdir(path):
        desc = np.load(os.path.join(path, "desc.npy"), mmap_mode="r")
        m = np.load(os.path.join(path, "meta.npz"), allow_pickle=False)
    else:
        m = np.load(path, allow_pickle=False)
        desc = m["desc"]
    n = desc.shape[0]
    get = lambda k: m[k] if k in m.files else None   # noqa: E731
    session = get("session")
    if session is None:
        session = get("seq") if get("seq") is not None else np.zeros(n, int)
    return dict(desc=desc, pos=np.asarray(m["pos"], float)[:, :3], gps=get("gps"), gps_sigma=get("gps_sigma"),
                session=np.asarray(session), name=os.path.basename(path.rstrip("/")))


def gps_sigma_model(t_frames, gnss_t, fix_dt, max_dt=0.5, gap=2.0, young=5.0, sigma_ok=8.0, sigma_young=40.0):
    """Consumer GPS without a quality field (NCLT Garmin): sigma from the age of the fix since the receiver
    reacquired (a gap > `gap` s without rows): the first `young` s after a reacquisition are much worse (NCLT
    2012-01-08: median 11 m / p90 66 m vs 4 / 15 m later).  No fix within max_dt: NaN."""
    starts = gnss_t[np.r_[0, np.where(np.diff(gnss_t) > gap)[0] + 1]]
    i = np.searchsorted(starts, t_frames, side="right") - 1
    age = t_frames - starts[np.clip(i, 0, len(starts) - 1)]
    sig = np.where(age < young, sigma_young, sigma_ok).astype(float)
    sig[~(fix_dt < max_dt)] = np.nan
    return sig


def load_nclt(desc_root, prepared_root, sessions, cams):
    """NCLT descriptor sets (benchmark/datasets/nclt_descriptors.py): positions = ground-truth camera positions
    (local NED), GPS = the nearest consumer fix in the same frame with gps_sigma_model."""
    sets = []
    for si, ses in enumerate(sessions):
        g = np.loadtxt(os.path.join(prepared_root, ses, "gnss.txt"))
        for cam in cams:
            f = os.path.join(desc_root, ses, f"Cam{cam}.npy")
            if not os.path.exists(f):
                continue
            desc = np.load(f, mmap_mode="r")
            m = np.load(os.path.join(desc_root, ses, f"Cam{cam}_meta.npz"))
            # horizontal positions only (NED x, y; z = 0): consumer-GPS altitude is much worse than its position
            pos = np.asarray(m["T_world_cam"][:, :3, 3], float).copy()
            pos[:, 2] = 0.0
            gps = np.asarray(m["gps_xyz"], float).copy()
            gps[:, 2] = 0.0
            sig = gps_sigma_model(np.asarray(m["t"], float), g[:, 0], np.asarray(m["gps_dt"], float))
            gps[~np.isfinite(sig)] = np.nan
            ok = np.isfinite(pos).all(1)
            sets.append(dict(desc=MultiRows([(desc, np.where(ok)[0])]), pos=pos[ok], gps=gps[ok],
                             gps_sigma=sig[ok], session=np.full(int(ok.sum()), si), name=f"{ses}/Cam{cam}"))
    return sets


class MultiRows:
    """Rows of several (memory-mapped) arrays as one indexable array, without loading them."""

    def __init__(self, parts):
        self.parts = []                          # (array, rows)
        for arr, rows in parts:
            if isinstance(arr, MultiRows):
                self.parts += [(a, r[rows] if rows is not None else r) for a, r in arr.parts]
            else:
                self.parts.append((arr, np.arange(arr.shape[0]) if rows is None else np.asarray(rows)))
        self.offsets = np.cumsum([0] + [len(r) for _, r in self.parts])
        self.shape = (int(self.offsets[-1]), self.parts[0][0].shape[1])

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        out = np.empty((len(idx), self.shape[1]), dtype=self.parts[0][0].dtype)
        which = np.searchsorted(self.offsets, idx, side="right") - 1
        for k in np.unique(which):
            sel = which == k
            arr, rows = self.parts[k]
            local = rows[idx[sel] - self.offsets[k]]
            order = np.argsort(local)
            got = np.asarray(arr[local[order]])
            tmp = np.empty_like(got)
            tmp[order] = got
            out[sel] = tmp
        return out


def concat(sets):
    out = dict(desc=MultiRows([(s["desc"], None) for s in sets]) if len(sets) > 1 else sets[0]["desc"])
    for k in ("pos", "session"):
        out[k] = np.concatenate([s[k] for s in sets])
    for k in ("gps", "gps_sigma"):
        out[k] = np.concatenate([s[k] for s in sets]) if all(s[k] is not None for s in sets) else None
    return out


def to_rows(desc, idx, dev, chunk=65536):
    """float32 normalized rows of desc[idx] on dev (chunked reads of a memmap)."""
    out = []
    for i in range(0, len(idx), chunk):
        x = torch.from_numpy(np.asarray(desc[np.sort(idx[i:i + chunk])], dtype=np.float32))
        out.append(torch.nn.functional.normalize(x, dim=-1).to(dev))
    return torch.cat(out)


def encode_rows(desc, idx, proj, dev, chunk=32768):
    """Projected codes (float16) of desc[idx] (sorted idx), chunked."""
    out = []
    for i in range(0, len(idx), chunk):
        x = torch.from_numpy(np.asarray(desc[idx[i:i + chunk]], dtype=np.float32)).to(dev)
        out.append(proj.apply(torch.nn.functional.normalize(x, dim=-1)).half())
    return torch.cat(out)


def topk_full(desc, db_idx, Q, k, dev, chunk=20000):
    """Exact top-k of queries Q (full, normalized, on dev) over desc[db_idx], database streamed in chunks."""
    best_s = torch.full((Q.shape[0], k), -9.0, device=dev)
    best_i = torch.zeros((Q.shape[0], k), dtype=torch.long, device=dev)
    Qh = Q.half()
    for i in range(0, len(db_idx), chunk):
        rows = db_idx[i:i + chunk]
        X = torch.from_numpy(np.asarray(desc[rows], dtype=np.float16)).to(dev)   # stored L2-normalized
        S = (Qh @ X.T).float()
        s, j = S.topk(min(k, S.shape[1]), dim=1)
        cand_s = torch.cat([best_s, s], 1)
        cand_i = torch.cat([best_i, torch.as_tensor(rows, device=dev)[j]], 1)
        best_s, o = cand_s.topk(k, dim=1)
        best_i = torch.gather(cand_i, 1, o)
    return best_s, best_i


def recall(top_idx, qpos, dbpos, radius, ks):
    """top_idx (Q, K) database indices (global, -1 = none) -> recall@k over answerable queries."""
    from scipy.spatial import cKDTree
    tree = cKDTree(dbpos[:, :2])
    answerable = np.array([len(x) > 0 for x in tree.query_ball_point(qpos[:, :2], radius)])
    d = np.where(top_idx >= 0, np.linalg.norm(dbpos_full_cache[top_idx.clip(0)][..., :2] - qpos[:, None, :2], axis=-1), np.inf)
    out = {}
    for k in ks:
        ok = (d[:, :k] <= radius).any(1) & answerable
        out[f"R@{k}"] = float(ok.sum() / max(answerable.sum(), 1))
    out["answerable"] = int(answerable.sum())
    return out


def latency(fn, n=200, warm=10):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95))}


# --------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", nargs="+", required=True)
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--projection", default="", help=".npz projection; default: fit uc512 on 50k database rows")
    ap.add_argument("--fit-dim", type=int, default=512)
    ap.add_argument("--fit-first", type=int, default=0,
                    help="fit the projection on the first N database rows (as the system does at fit_at) instead of a "
                         "random sample of 50k")
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 100000, 0])
    ap.add_argument("--n-queries", type=int, default=3000)
    ap.add_argument("--radii", nargs="+", type=float, default=[10.0, 25.0])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--methods", nargs="+", default=["full", "code", "ivf", "belief:5", "belief:20", "belief:50", "gps"])
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--r-min", type=float, default=5.0)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--latency-cpu-threads", type=int, default=8)
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nclt", default="", help="NCLT desc root: --db / --queries are then session names")
    ap.add_argument("--nclt-prepared", default="")
    ap.add_argument("--db-cams", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--q-cams", nargs="+", type=int, default=[5])
    a = ap.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(0)
    if a.nclt:
        DB = concat(load_nclt(a.nclt, a.nclt_prepared, a.db, a.db_cams))
        QS = concat(load_nclt(a.nclt, a.nclt_prepared, a.queries, a.q_cams))
    else:
        DB = concat([load_set(p) for p in a.db])
        QS = concat([load_set(p) for p in a.queries])
    N = DB["desc"].shape[0]
    global dbpos_full_cache
    dbpos_full_cache = DB["pos"]
    qi = np.sort(rng.choice(QS["desc"].shape[0], min(a.n_queries, QS["desc"].shape[0]), replace=False))
    Qfull = to_rows(QS["desc"], qi, dev)
    qpos = QS["pos"][qi]
    print(f"database {N} rows, {len(qi)} queries", flush=True)

    t0 = time.perf_counter()
    if a.projection:
        proj = PCAProjection.load(a.projection)
    else:
        fit = np.arange(min(N, a.fit_first)) if a.fit_first else np.sort(rng.choice(N, min(N, 50000), replace=False))
        proj = PCAProjection.fit(to_rows(DB["desc"], fit, dev), a.fit_dim)
    t_fit = time.perf_counter() - t0
    Qc = proj.apply(Qfull)
    results = {"N": int(N), "n_queries": int(len(qi)), "projection": proj.meta, "t_fit_s": t_fit, "rows": []}
    K = max(max(a.ks), 2 * a.slots)

    for size in a.sizes:
        n = N if size <= 0 or size >= N else size
        db_idx = np.sort(rng.choice(N, n, replace=False)) if n < N else np.arange(N)
        dbpos = DB["pos"][db_idx]
        row = {"size": int(n), "bytes_full_fp16": int(n * proj.dim_in * 2), "bytes_code_fp16": int(n * proj.dim * 2)}
        t0 = time.perf_counter()
        codes = encode_rows(DB["desc"], db_idx, proj, dev)
        row["code_explained_db"] = float(codes[: min(n, 20000)].float().pow(2).sum(1).mean())
        row["code_explained_q"] = float(Qc.pow(2).sum(1).mean())
        torch.cuda.synchronize()
        row["t_encode_s"] = time.perf_counter() - t0
        index = DescriptorIndex(proj.dim_in, device=dev, initial_capacity=n, projection=proj, backend="exact")
        index.set_rows(codes, db_idx)
        res = {}

        def evaluate(name, top_global):
            for r in a.radii:
                res[f"{name}@{int(r)}m"] = recall(top_global, qpos, dbpos, r, a.ks)

        if "full" in a.methods and n * proj.dim_in * 2 < 200e9:
            t0 = time.perf_counter()
            ts_full, ti = topk_full(DB["desc"], db_idx, Qfull, K, dev)
            row["t_full_eval_s"] = time.perf_counter() - t0
            evaluate("full", ti.cpu().numpy())
            # score fidelity of the codes on the K best full matches of every query (as projection_study.py)
            loc = torch.as_tensor(np.searchsorted(db_idx, ti.cpu().numpy()), device=dev)
            sc = torch.stack([(codes[loc[j]].float() @ Qc[j]) for j in range(len(qi))])
            err = (sc - ts_full).abs()
            row["code_score_mae_topK"] = float(err.mean())
            row["code_score_p95_topK"] = float(err.flatten().quantile(0.95))
            row["code_thr03_agree"] = float(((sc > 0.3) == (ts_full > 0.3)).float().mean())
            zq = Qc.norm(dim=1).clamp(1e-6, 1)
            zy = codes.float().norm(dim=1).clamp(1e-6, 1)
            gain = 1 + torch.sqrt((1 - zq[:, None] ** 2) * (1 - zy[loc] ** 2)) / (zq[:, None] * zy[loc])
            scc = sc * gain
            errc = (scc - ts_full).abs()
            row["codec_score_mae_topK"] = float(errc.mean())
            row["codec_score_p95_topK"] = float(errc.flatten().quantile(0.95))
            row["codec_thr03_agree"] = float(((scc > 0.3) == (ts_full > 0.3)).float().mean())
        tops = []
        for i in range(0, len(qi), 256):   # chunked over queries
            tops.append(((Qc[i:i + 256].half() @ codes.T).float()).topk(min(K, n), dim=1).indices)
        code_top = torch.cat(tops)
        code_top_np = code_top.cpu().numpy()
        if "codec" in a.methods:   # residual-corrected code scores (ranking by them)
            zy_all = codes.float().norm(dim=1).clamp(1e-6, 1)
            ry_all = torch.sqrt(1 - zy_all ** 2)
            tops = []
            for i in range(0, len(qi), 256):
                q = Qc[i:i + 256]
                zq = q.norm(dim=1).clamp(1e-6, 1)
                d = (q.half() @ codes.T).float()
                g = 1 + torch.sqrt(1 - zq[:, None] ** 2) * ry_all[None] / (zq[:, None] * zy_all[None])
                tops.append((d * g).topk(min(K, n), dim=1).indices)
            evaluate("codec", db_idx[torch.cat(tops).cpu().numpy()])
        if "code" in a.methods:
            evaluate("code", db_idx[code_top_np])
        if "ivf" in a.methods and n >= 20000:
            t0 = time.perf_counter()
            ivf_index = DescriptorIndex(proj.dim_in, device=dev, initial_capacity=n, projection=proj, backend="ivf",
                                        ivf_nprobe=a.nprobe, ivf_min_rows=0)
            ivf_index.set_rows(codes, db_idx)
            torch.cuda.synchronize()
            row["t_ivf_train_s"] = time.perf_counter() - t0
            row["ivf_nlist"] = int(ivf_index._ivf.centroids.shape[0])
            tops, visited = [], []
            for j in range(len(qi)):
                rows = ivf_index.ann_rows(Qc[j])
                visited.append(int(rows.numel()))
                s = ivf_index.scores(Qc[j], rows)
                t = rows[s.topk(min(K, s.numel())).indices]
                tops.append(np.pad(t.cpu().numpy(), (0, K - t.numel()), constant_values=-1))
            T = np.stack(tops)
            evaluate("ivf", np.where(T >= 0, db_idx[T.clip(0)], -1))
            row["ivf_visited_frac"] = float(np.mean(visited) / n)

        # location priors: locality slots (best in-region by code score) + the global ranking without them
        sp = SpatialIndex()
        sp.rebuild(np.arange(n), dbpos, epoch=0)

        def prior_merge(centers, radii, label):
            out = np.full((len(qi), K), -1, dtype=np.int64)
            n_local = []
            for j in range(len(qi)):
                glob = list(code_top_np[j])
                if centers[j] is None:
                    out[j, :len(glob)] = glob[:K]
                    n_local.append(0)
                    continue
                rows = sp.query(centers[j][None], np.array([radii[j]]))
                if len(rows):
                    rt = torch.as_tensor(rows, device=dev)
                    s = (codes[rt].float() @ Qc[j]).float()
                    loc = rows[s.topk(min(a.slots, len(rows))).indices.cpu().numpy()].tolist()
                else:
                    loc = []
                merged = loc + [g for g in glob if g not in set(loc)]
                out[j, :min(K, len(merged))] = merged[:K]
                n_local.append(len(rows))
            row[f"{label}_region_mean"] = float(np.mean(n_local))
            top = np.where(out >= 0, db_idx[out.clip(0)], -1)
            evaluate(label, top)
            return top

        for m in a.methods:
            if m.startswith("belief:"):
                sig = float(m.split(":")[1])
                c = qpos.copy()
                c[:, :2] += rng.normal(0, sig, size=(len(qi), 2))
                prior_merge(list(c), np.full(len(qi), a.r_min + 3 * sig), m)
            if m == "gps" and QS["gps"] is not None:
                g = QS["gps"][qi]
                gs = QS["gps_sigma"][qi]
                ok = np.isfinite(g).all(1) & np.isfinite(gs)
                centers = [g[j] if ok[j] else None for j in range(len(qi))]
                gtop = prior_merge(centers, a.r_min + 3 * np.where(ok, gs, 0), "gps")
                row["gps_valid_frac"] = float(ok.mean())
                ctop = db_idx[code_top_np]
                for split_name, mask in (("fix", ok), ("nofix", ~ok)):
                    if mask.sum():
                        for r in a.radii:
                            res[f"code@{int(r)}m_{split_name}"] = recall(ctop[mask], qpos[mask], dbpos, r, a.ks)
                            res[f"gps@{int(r)}m_{split_name}"] = recall(gtop[mask], qpos[mask], dbpos, r, a.ks)

        # single-query latency (what one retrieval costs the system), GPU fp16 codes and CPU fp32 codes
        if not a.no_latency:
            q1 = Qc[0]
            row["lat_gpu_code"] = latency(lambda: (codes @ q1.half()).float().topk(10))
            if "ivf" in a.methods and n >= 20000:
                row["lat_gpu_ivf"] = latency(lambda: ivf_index.scores(q1, ivf_index.ann_rows(q1)).topk(10))
            torch.set_num_threads(a.latency_cpu_threads)
            cpu_codes = codes.float().cpu()
            q1c = q1.float().cpu()
            row["lat_cpu_code"] = latency(lambda: (cpu_codes @ q1c).topk(10), n=50 if n > 1e6 else 200)
            if "ivf" in a.methods and n >= 20000:
                ivc = DescriptorIndex(proj.dim_in, device="cpu", initial_capacity=n, projection=proj, backend="ivf",
                                      ivf_nprobe=a.nprobe, ivf_min_rows=0, store_dtype="float32")
                ivc.buf = cpu_codes
                ivc.ids = torch.as_tensor(db_idx)
                ivc.n = n
                ivc._ivf = type(ivf_index._ivf)(ivf_index._ivf.centroids.float().cpu())
                ivc._ivf.cell = ivf_index._ivf.cell.cpu()
                row["lat_cpu_ivf"] = latency(lambda: ivc.scores(q1c, ivc.ann_rows(q1c)).topk(10), n=50 if n > 1e6 else 200)
            if n <= 200000:
                full_rows = to_rows(DB["desc"], db_idx, "cpu").half() if n * proj.dim_in * 2 < 8e9 else None
                if full_rows is not None:
                    qf = Qfull[0].half().cpu()
                    row["lat_cpu_full_fp32"] = latency(lambda: (full_rows.float() @ qf.float()).topk(10), n=20, warm=2) \
                        if n <= 20000 else None
                    gpu_full = full_rows.to(dev)
                    qfg = Qfull[0].half()
                    row["lat_gpu_full"] = latency(lambda: (gpu_full @ qfg).float().topk(10))
                    del gpu_full
            # incremental add (one keyframe) to the code index
            one = torch.nn.functional.normalize(torch.randn(proj.dim_in, device=dev), dim=-1)
            row["lat_add_code"] = latency(lambda: (index.add(one, -1), index.remove_row(index.n - 1)), n=100)
        row["recall"] = res
        results["rows"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "recall"}), flush=True)
        for k, v in res.items():
            print(f"  {k}: {v}", flush=True)
        json.dump(results, open(a.out, "w"), indent=1)
        del codes, index
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
