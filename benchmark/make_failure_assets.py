#!/usr/bin/env python3
"""Images for the failure cases of the results page: for the T3 runs that build_site.py lists as failure cases, the
query frame at the end of up to two failed trials next to the map frame nearest to it in ground truth.

  python benchmark/make_failure_assets.py --data $BENCH_DATA [--results benchmark/results/results.json]

Writes benchmark/site/assets/failures/<run id>_<k>_q.jpg and _m.jpg (320 px wide); run where the dataset folders are.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

import build_site as bs

ROOT = Path(__file__).resolve().parents[1]


def image(seq: Path, i: int):
    d = seq / ("left" if (seq / "left").is_dir() else "rgb")
    files = sorted(d.glob("*.png")) or sorted(d.glob("*.jpg"))
    if not files:
        return None
    im = cv2.imread(str(files[min(i, len(files) - 1)]))
    return None if im is None else cv2.resize(im, (320, int(320 * im.shape[0] / im.shape[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", default=str(ROOT / "benchmark/results/results.json"))
    ap.add_argument("--per-run", type=int, default=2)
    a = ap.parse_args()
    ds = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())
    results = json.loads(Path(a.results).read_text())["results"]
    runs_by_key = {id(r): bs.run_id(r) for r in results}
    by_id = {bs.run_id(r): r for r in results}
    fail = bs.failures(results, runs_by_key)
    out = ROOT / "benchmark/site/assets/failures"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for k, v in fail.items():
        for rid in v["t3"]:
            r = by_id[rid]
            cfg = ds[r["dataset"]]
            setup_dir = cfg["setups"][r["setup"]]
            q = Path(a.data) / r["dataset"] / r["query"] / setup_dir
            m = Path(a.data) / r["dataset"] / r["map"] / setup_dir
            if not (q / "poses_left.txt").is_file() or not (m / "poses_left.txt").is_file():
                continue
            Gq = np.loadtxt(q / "poses_left.txt").reshape(-1, 4, 4)
            Gm = np.loadtxt(m / "poses_left.txt").reshape(-1, 4, 4)
            failed = [t for t in r.get("trials", []) if not t["success"]]
            failed = sorted(failed, key=lambda t: -(t["final_err"] or 1e9))[:a.per_run]
            for j, t in enumerate(failed):
                i = min(int(t["start"]) + cfg["trial_len"] - 1, len(Gq) - 1)
                mi = int(np.argmin(np.linalg.norm(Gm[:, :3, 3] - Gq[i, :3, 3], axis=1)))
                qi, mim = image(q, i), image(m, mi)
                if qi is None or mim is None:
                    continue
                cv2.imwrite(str(out / f"{rid}_{j}_q.jpg"), qi, [cv2.IMWRITE_JPEG_QUALITY, 80])
                cv2.imwrite(str(out / f"{rid}_{j}_m.jpg"), mim, [cv2.IMWRITE_JPEG_QUALITY, 80])
                n += 1
    print(f"{n} failure-case image pairs in {out}")


if __name__ == "__main__":
    main()
