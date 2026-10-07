#!/usr/bin/env python3
"""Statistics of the prepared benchmark sequences: frames, duration, path length and relocalization trials per session,
the frames the map covers and the intervals in which the robot stands still (PROTOCOL.md, T2 / T3).

  python benchmark/dataset_stats.py --data $BENCH_DATA [--datasets openloris kitti rover simchange] [--out FILE]

Merges into benchmark/results/datasets.json (one entry per dataset / scene / sequence), which make_tables.py and
build_site.py turn into the dataset section.  Run it where the prepared folders are.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "benchmark" / "eval"))
from metrics import moving_frames, moving_rule, still_intervals  # noqa: E402
from reloc_metrics import build_trials  # noqa: E402


def seq_stats(folder: Path, fps: float):
    P = np.loadtxt(folder / "poses_left.txt").reshape(-1, 4, 4)
    calib = json.loads((folder / "calib.json").read_text())
    n = len(P)
    fps = float(calib.get("fps", fps))
    return {"frames": n, "duration_s": n / fps, "path_m": float(np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1).sum()),
            "extent_m": float(np.linalg.norm(P[:, :3, 3].max(0) - P[:, :3, 3].min(0)))}


def motion(folder: Path, rule: dict):
    """Per-frame moving mask of a session (PROTOCOL.md, moving robot): (moving fraction, still intervals, mask)."""
    P = np.loadtxt(folder / "poses_left.txt").reshape(-1, 4, 4)
    fps = json.loads((folder / "calib.json").read_text()).get("fps", 10.0)
    m = moving_frames(P, fps, **rule)
    return float(m.mean()) if len(m) else 0.0, still_intervals(m), m


def coverage(query_folder: Path, map_folder: Path, radius: float):
    """Frames of a query session whose ground-truth position lies within `radius` of the map session's ground-truth path:
    (fraction covered, list of [first, last] frame intervals that are NOT covered)."""
    from scipy.spatial import cKDTree
    q = np.loadtxt(query_folder / "poses_left.txt").reshape(-1, 4, 4)[:, :3, 3]
    m = np.loadtxt(map_folder / "poses_left.txt").reshape(-1, 4, 4)[:, :3, 3]
    d, _ = cKDTree(m).query(q)
    ok = d < radius
    gaps, i = [], 0
    while i < len(ok):
        if not ok[i]:
            j = i
            while j + 1 < len(ok) and not ok[j + 1]:
                j += 1
            gaps.append([int(i), int(j)])
            i = j + 1
        else:
            i += 1
    return float(ok.mean()), gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--datasets", nargs="*", default=["openloris", "kitti", "rover", "simchange"])
    ap.add_argument("--out", default=str(ROOT / "benchmark/results/datasets.json"))
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())
    out = Path(a.out)
    stats = json.loads(out.read_text()) if out.is_file() else {}
    for d in a.datasets:
        c = cfg[d]
        ds = stats.setdefault(d, {})
        for scene, sc in c["scenes"].items():
            rows = {}
            for role, names in (("map", [sc["map"]]), ("query", sc.get("queries", []))):
                for name in names:
                    for setup, sub in c["setups"].items():
                        f = Path(a.data) / d / name / sub
                        if not (f / "calib.json").is_file():
                            continue
                        key = name if sub in (".", "") else f"{name}"
                        r = rows.setdefault(key, {"role": role, "setups": {}})
                        if sub not in r["setups"]:
                            r["setups"][sub] = seq_stats(f, 10.0)
                            mov, still, mask = motion(f, moving_rule(c))
                            r["setups"][sub].update({"moving": mov, "still": still})
                            mf = Path(a.data) / d / sc["map"] / sub
                            if role == "query" and (mf / "calib.json").is_file():
                                frac, gaps = coverage(f, mf, c["thresholds"][1])
                                n = len(mask)
                                cov = np.ones(n, bool)
                                for g0, g1 in gaps:
                                    cov[g0:g1 + 1] = False
                                r["setups"][sub].update({"covered": frac, "uncovered": gaps,
                                                         "evaluated": float((cov & mask).mean()) if n else 0.0})
                    if name in rows and role == "query" and c.get("trial_len"):
                        n = min(v["frames"] for v in rows[name]["setups"].values())
                        rows[name]["trials"] = len(build_trials(n, c["trial_len"], c["trial_stride"]))
            if rows:
                ds[scene] = rows
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=1))
    print(f"wrote {out}: " + ", ".join(f"{d} {len(v)} scenes" for d, v in stats.items()))


if __name__ == "__main__":
    main()
