"""Latency and memory of one retrieval vs database size (synthetic unit descriptors: the cost does not depend on content).

    python scripts/retrieval/latency_scaling.py --sizes 1e3 1e4 1e5 1e6 1e7 --out latency.json

Per size: the full 16384-d float32 buffer (the original database; GPU while it fits), projected float16 codes
(512 / 256 dims) with exact search on the GPU and float32 codes on the CPU (--threads), and the IVF backend
(nlist 4 sqrt(n), nprobe 16) on GPU and CPU.  One query = scores + top-10 (what the database's query does besides
thresholds and bookkeeping).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cross.db.index import _IVF   # noqa: E402


def timed(fn, n, warm=5, cuda=True):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95))}


def rand_unit(n, d, dev, dtype, chunk=1 << 20):
    out = torch.empty((n, d), device=dev, dtype=dtype)
    g = torch.Generator(device=dev).manual_seed(0)
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        out[i:i + m] = torch.nn.functional.normalize(torch.randn(m, d, device=dev, generator=g), dim=-1).to(dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=float, default=[1e3, 1e4, 1e5, 1e6, 1e7])
    ap.add_argument("--dims", nargs="+", type=int, default=[512, 256])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--gpu-mem-gb", type=float, default=16.0, help="largest buffer tried on the GPU")
    ap.add_argument("--cpu-mem-gb", type=float, default=48.0)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = "cuda"
    rows = []
    info = {"gpu": torch.cuda.get_device_name(0), "threads": a.threads, "cpu_count": os.cpu_count()}
    for size in a.sizes:
        n = int(size)
        row = {"n": n}
        # the original database: full 16384-d float32 on the GPU
        if n * 16384 * 4 <= a.gpu_mem_gb * 1e9:
            X = rand_unit(n, 16384, dev, torch.float32)
            q = X[0].clone()
            row["full_fp32_gpu"] = timed(lambda: (X @ q.unsqueeze(-1)).squeeze(-1).topk(10), 50)
            del X
            torch.cuda.empty_cache()
        row["bytes_full_fp32"] = n * 16384 * 4
        for d in a.dims:
            row[f"bytes_code{d}_fp16"] = n * d * 2
            if n * d * 2 <= a.gpu_mem_gb * 1e9:
                C = rand_unit(n, d, dev, torch.float16)
                q = C[0].clone()
                row[f"code{d}_fp16_gpu"] = timed(lambda: (C @ q).float().topk(10), 100)
                if n >= 20000:
                    t0 = time.perf_counter()
                    ivf = _IVF.train(C.float() if n * d * 4 <= a.gpu_mem_gb * 1e9 else C, int(min(65536, 4 * math.sqrt(n))))
                    torch.cuda.synchronize()
                    row[f"ivf{d}_train_s"] = time.perf_counter() - t0
                    qf = q.float()

                    def ivf_q():
                        rws = ivf.probe(qf, a.nprobe, n)
                        return (C.index_select(0, rws) @ q).float().topk(10)
                    row[f"ivf{d}_gpu"] = timed(ivf_q, 100)
                    cell_cpu = ivf.cell.cpu()
                    cent_cpu = ivf.centroids.float().cpu()
                else:
                    ivf = None
                Ccpu = C.float().cpu() if n * d * 4 <= a.cpu_mem_gb * 1e9 else None
                del C
                torch.cuda.empty_cache()
            else:
                Ccpu, ivf = None, None
            if Ccpu is not None:
                qc = Ccpu[0].clone()
                row[f"code{d}_fp32_cpu"] = timed(lambda: (Ccpu @ qc).topk(10), 20 if n >= 1e6 else 100, cuda=False)
                if ivf is not None:
                    cpu_ivf = _IVF(cent_cpu)
                    cpu_ivf.cell = cell_cpu

                    def ivf_c():
                        rws = cpu_ivf.probe(qc, a.nprobe, n)
                        return (Ccpu.index_select(0, rws) @ qc).topk(10)
                    row[f"ivf{d}_cpu"] = timed(ivf_c, 50, cuda=False)
                del Ccpu
        rows.append(row)
        print(json.dumps(row), flush=True)
        json.dump({"info": info, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
