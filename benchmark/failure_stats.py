#!/usr/bin/env python3
"""Features of every T3 trial for the failure analysis (benchmark/failure_analysis.py), computed from the prepared data:

  motion     path_m, speed (m/s), rot_deg (summed frame-to-frame rotation), rate_max (deg/s over 1 s), still (share of
             frames slower than 0.05 m/s)
  viewpoint  end_view: smallest angle between the last frame's optical axis and that of a map-session frame within the
             near radius (1 m indoors, 3 m outdoors; 180 when none is that near); best_view: the same minimised over the
             trial; end_map_dist (m); n_map_near (map frames within the near radius of the last frame)
  image      bright (mean grey level), corners (FAST corners at 320 px), end_corners (mean of the last 10 frames)
  matchability  SIFT matches surviving a fundamental-matrix RANSAC between a query frame and the map frame shown for it
             (within the near radius the most similar viewing direction, else the nearest position): every 5th frame and
             the last; inl_end (last frame), inl_max (best frame of the trial), inl_med

and, with --runs, the per-frame tracking state of ORB-SLAM3's mapping runs (T1 loss onsets against the turn rate).

  python benchmark/failure_stats.py --data $BENCH_DATA [--runs <run root> ...] [--jobs 32]

Writes benchmark/results/failure_trials.json.  Run where the dataset folders are (it reads every trial's images).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
_W = {}


def seq_dir(data: Path, ds_cfg, dataset, seq, setup_dir):
    if dataset == "simchange":
        return data / dataset / seq
    return data / ds_cfg[dataset].get("data", dataset) / seq / setup_dir


def frames_of(d: Path):
    sub = d / "left" if (d / "left").is_dir() else d / "rgb"
    return sorted(sub.glob("*.png")) or sorted(sub.glob("*.jpg"))


def rot_angle(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.einsum("...ij,...ij->...", Ra, Rb) - 1) / 2, -1, 1)))


def gray(path, width):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    if im.shape[1] != width:
        im = cv2.resize(im, (width, int(round(width * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA)
    return im


def sift_inliers(a, b):
    sift = _W.setdefault("sift", cv2.SIFT_create(nfeatures=1500))
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
    if len(good) < 8:
        return len(good) // 4
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.0, 0.999)
    return int(mask.sum()) if mask is not None else 0


def trial_job(t):
    """Features of one trial: t = (dataset, map dir, query dir, start, len, near radius, fps)."""
    dataset, mdir, qdir, start, tl, near, fps = t
    Gq = np.loadtxt(Path(qdir) / "poses_left.txt").reshape(-1, 4, 4)
    Gm = np.loadtxt(Path(mdir) / "poses_left.txt").reshape(-1, 4, 4)
    fq, fm = frames_of(Path(qdir)), frames_of(Path(mdir))
    end = min(start + tl, len(Gq), len(fq))
    P, R = Gq[start:end, :3, 3], Gq[start:end, :3, :3]
    n = len(P)
    if n < 2:
        return None
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    w = min(10, n - 1)
    f = {"path_m": float(step.sum()), "speed": float(step.sum() / ((n - 1) / fps)),
         "rot_deg": float(rot_angle(R[1:], R[:-1]).sum()), "rate_max": float((rot_angle(R[w:], R[:-w]) * fps / w).max()),
         "still": float((step * fps < 0.05).mean())}
    mp, mz = Gm[:len(fm), :3, 3], Gm[:len(fm), :3, 2]
    tree = cKDTree(mp)
    f["end_map_dist"] = float(tree.query(P[-1])[0])
    z = R[:, :, 2]

    def view(i):
        idx = tree.query_ball_point(P[i], near)
        return float(np.degrees(np.arccos(np.clip(mz[idx] @ z[i], -1, 1))).min()) if idx else 180.0
    f["end_view"] = view(n - 1)
    f["best_view"] = min(view(i) for i in range(0, n, 2))
    f["n_map_near"] = len(tree.query_ball_point(P[-1], near))
    # image statistics (every 2nd frame) and matchability (every 5th frame and the last)
    fast = _W.setdefault("fast", cv2.FastFeatureDetector_create(threshold=20))
    br, co = [], []
    for i in range(start, end, 2):
        im = gray(fq[i], 320)
        if im is not None:
            br.append(float(im.mean())); co.append(len(fast.detect(im, None)))
    ends = [len(fast.detect(im, None)) for im in (gray(fq[i], 320) for i in range(max(start, end - 10), end)) if im is not None]
    f.update(bright=float(np.mean(br)) if br else None, corners=float(np.mean(co)) if co else None,
             end_corners=float(np.mean(ends)) if ends else None)
    inl = []
    for i in sorted(set(list(range(start, end, 5)) + [end - 1])):
        d = np.linalg.norm(mp - Gq[i, :3, 3], axis=1)
        ang = np.degrees(np.arccos(np.clip(mz @ Gq[i, :3, 2], -1, 1)))
        c = np.where(d < near)[0]
        j = int(c[np.argmin(ang[c])]) if len(c) else int(np.argmin(d))
        a, b = gray(fq[i], 640), gray(fm[j], 640)
        inl.append(sift_inliers(a, b) if a is not None and b is not None else 0)
    f.update(inl_end=float(inl[-1]), inl_max=float(max(inl)), inl_med=float(np.median(inl)))
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()}


def orb_tracking(roots, ds_cfg, data: Path):
    """Tracking state per frame of ORB-SLAM3's mapping runs (map_poses.txt: idx state pose; 2 = tracking)."""
    out = []
    seen = set()
    for root in roots:
        for f in sorted(Path(root).glob("*/*/orbslam3/*/s0/maps/**/map_poses.txt")):
            rel = f.relative_to(root).parts
            dataset, scene, setup = rel[0], rel[1], rel[3]
            seq = "/".join(rel[6:-1])
            if (dataset, setup, seq) in seen or dataset not in ds_cfg:
                continue
            seen.add((dataset, setup, seq))
            rows = np.loadtxt(f, ndmin=2)
            gt = seq_dir(data, ds_cfg, dataset, seq, ds_cfg[dataset]["setups"][setup]) / "poses_left.txt"
            if not gt.is_file() or rows.size == 0:
                continue
            G = np.loadtxt(gt).reshape(-1, 4, 4)
            tracked = np.zeros(len(G), bool)
            for i, st in zip(rows[:, 0].astype(int), rows[:, 1].astype(int)):
                if i < len(G) and st == 2:
                    tracked[i] = True
            w = 5
            rate = np.zeros(len(G))
            rate[w // 2:w // 2 + len(G) - w] = rot_angle(G[w:, :3, :3], G[:-w, :3, :3]) * 10 / w
            onsets = [i for i in range(5, len(G) - 5) if tracked[i - 5:i].all() and not tracked[i:i + 5].any()]
            out.append({"dataset": dataset, "setup": setup, "sequence": seq, "tracked": float(tracked.mean()),
                        "rate_p90": float(np.percentile(rate, 90)),
                        "onsets": [[i, round(float(rate[max(0, i - 5):i + 1].max()), 1)] for i in onsets]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="*", default=[], help="run roots with ORB-SLAM3 mapping runs (T1 tracking losses)")
    ap.add_argument("--results", default=str(ROOT / "benchmark/results/results.json"))
    ap.add_argument("--out", default=str(ROOT / "benchmark/results/failure_trials.json"))
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    data = Path(a.data)
    ds_cfg = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())
    jobs = {}
    for r in json.loads(Path(a.results).read_text())["results"]:
        if r.get("track") != "t3" or not r.get("trials") or ds_cfg.get(r["dataset"], {}).get("dev"):
            continue
        cfg = ds_cfg[r["dataset"]]
        su = cfg["setups"][r["setup"]]
        qd, md = seq_dir(data, ds_cfg, r["dataset"], r["query"], su), seq_dir(data, ds_cfg, r["dataset"], r["map"], su)
        if not (qd / "poses_left.txt").is_file():
            continue
        near = 1.0 if cfg["environment"] == "indoor" else 3.0
        for t in r["trials"]:
            k = (r["dataset"], r["query"], su, int(t["start"]))
            jobs.setdefault(k, (r["dataset"], str(md), str(qd), int(t["start"]), int(cfg["trial_len"]), near, 10.0))
    keys = sorted(jobs)
    print(f"{len(keys)} trials", flush=True)
    trials = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for k, (key, f) in enumerate(zip(keys, ex.map(trial_job, [jobs[k] for k in keys], chunksize=4))):
            if f is not None:
                trials.append({"dataset": key[0], "query": key[1], "setup_dir": key[2], "start": key[3], **f})
            if k % 200 == 0:
                print(k, flush=True)
    out = {"trials": trials, "orb_tracking": orb_tracking(a.runs, ds_cfg, data) if a.runs else []}
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {a.out}: {len(trials)} trials, {len(out['orb_tracking'])} ORB-SLAM3 mapping runs")


if __name__ == "__main__":
    main()
