#!/usr/bin/env python3
"""Accuracy of the odometry sources of prepared benchmark folders over trial-length windows (ground truth for scoring).

For every folder: the simulated odometry the benchmark uses without an odometry file (ground-truth increments with the
loader's SNR noise, seeded as in a run), odom_left.txt (wheel / OXTS dead reckoning) and every odom_vio*.txt.

  python benchmark/datasets/eval_odometry.py $BENCH_DATA/kitti/*/stereo [--snr 10] [--window 100] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark" / "datasets"))
from prepare_vio import drift_stats  # noqa: E402


def simulated(gt: np.ndarray, snr: float, seed: int) -> np.ndarray:
    from cross.dataloader.dataloader import Dataloader

    class _D(Dataloader):              # only the noise model of the base class
        def __len__(self): return len(gt)
        def __getitem__(self, i): raise NotImplementedError
        def get_sequence_frequency(self): return 10.0
        def get_idx_from_timestamp(self, t): return None
    d = _D(snr=snr, seed=seed)
    out = [gt[0]]
    for j in range(1, len(gt)):
        delta = np.linalg.inv(gt[j - 1]) @ gt[j]
        out.append(out[-1] @ d._noise_delta(delta, gt[j]))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = {}
    for f in map(Path, a.folders):
        if not (f / "poses_left.txt").is_file():
            continue
        gt = np.loadtxt(f / "poses_left.txt").reshape(-1, 4, 4)
        srcs = {f"sim_snr{a.snr:g}": simulated(gt, a.snr, 0)}
        for p in sorted(f.glob("odom_*.txt")):
            o = np.loadtxt(p).reshape(-1, 4, 4)
            if len(o) == len(gt):
                srcs[p.stem] = o
        res[str(f)] = {}
        for k, o in srcs.items():
            s = drift_stats(o, gt, window=a.window, stride=a.window // 2)
            res[str(f)][k] = s
            t, r, y = s["trans_m_mean_med_p95"], s["rot_deg_mean_med_p95"], s["yaw_deg_mean_med_p95"]
            if t is None:                      # shorter than one window
                continue
            pct = s["trans_pct_mean_med_p95"]
            print(f"{f}  {k:22s} trans {t[0]:7.3f} m (p95 {t[2]:7.3f})  rot {r[0]:6.2f} deg  yaw {y[0]:6.2f} deg "
                  f"(p95 {y[2]:6.2f})  {pct[0] if pct else float('nan'):6.2f} %", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
