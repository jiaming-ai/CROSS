#!/usr/bin/env python3
"""Merge the result.json files of benchmark runs into benchmark/results/results.json, and keep every run in
benchmark/results/history.json.

  python benchmark/collect.py <results_root> [<results_root> ...] [--out benchmark/results/results.json]
  python benchmark/collect.py --backfill-git        # history.json from every committed version of results.json

Results of the same cell (track, dataset, scene, system, setup, seed, sequence / map+query) found in several roots
(e.g. a rerun) are resolved by the newest `time`; --prefer ROOT makes the cells of that root win (a rerun with new
code that finished before another rerun of the old code).  Per-frame curves and trajectories stay in the merged file
(downsampled by run.py), so the web page needs nothing else.

history.json holds every run ever collected (one entry per cell, commit and run time; without the trajectories and
error curves), so the page can show the results of each code version (`commit` of the run) next to each other.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = "benchmark/results/results.json"
HISTORY = ROOT / "benchmark/results/history.json"
HEAVY = ("traj_est", "traj_gt", "err_curve")       # per-frame data, only kept for the current cells


def key(r):
    return (r.get("track"), r.get("dataset"), r.get("scene"), r.get("system"), r.get("setup"), r.get("seed"),
            r.get("sequence"), r.get("map"), r.get("query"))


def run_key(r):
    """One run: its cell, code version and time."""
    return key(r) + (r.get("commit"), r.get("time"))


def slim(r):
    return {k: v for k, v in r.items() if k not in HEAVY}


def update_history(runs, path=HISTORY):
    """Add runs to history.json (entries already there are kept; a run is identified by run_key)."""
    path = Path(path)
    hist = {}
    if path.is_file():
        for r in json.loads(path.read_text())["runs"]:
            hist[run_key(r)] = r
    n0 = len(hist)
    for r in runs:
        hist.setdefault(run_key(r), slim(r))
    out = sorted(hist.values(), key=lambda r: [str(x) for x in run_key(r)])
    path.write_text(json.dumps({"runs": out}, separators=(",", ":")))
    print(f"history: {len(hist) - n0} new runs, {len(hist)} in {path}")


def backfill_git(path=HISTORY):
    """Every run of every committed version of results.json (oldest first)."""
    revs = subprocess.run(["git", "log", "--reverse", "--format=%h", "--", RESULTS], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.split()
    for rev in revs:
        txt = subprocess.run(["git", "show", f"{rev}:{RESULTS}"], cwd=ROOT, capture_output=True, text=True).stdout
        try:
            runs = json.loads(txt)["results"]
        except Exception as e:      # noqa: BLE001
            print(f"skip {rev}: {e}")
            continue
        print(f"{rev}: {len(runs)} cells")
        update_history(runs, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--out", default=str(ROOT / RESULTS))
    ap.add_argument("--merge", action="store_true", help="keep the entries already in --out")
    ap.add_argument("--history", default=str(HISTORY), help="history file ('' to leave it alone)")
    ap.add_argument("--backfill-git", action="store_true", help="add the runs of every committed results.json to the history")
    ap.add_argument("--prefer", nargs="*", default=[], help="roots (also given as roots) whose cells win regardless of time")
    a = ap.parse_args()
    if a.backfill_git:
        backfill_git(a.history)
        if not a.roots:
            return
    cells = {}
    out = Path(a.out)
    if a.merge and out.is_file():
        for r in json.loads(out.read_text())["results"]:
            cells[key(r)] = r
    n, found = 0, []
    preferred = set()
    for root in sorted(a.roots, key=lambda x: x in a.prefer):     # preferred roots last
        pref = root in a.prefer
        for f in Path(root).rglob("result.json"):
            try:
                r = json.loads(f.read_text())
            except Exception as e:      # noqa: BLE001
                print(f"skip {f}: {e}")
                continue
            n += 1
            found.append(r)
            k = key(r)
            newer = k not in cells or (r.get("time") or "") >= (cells[k].get("time") or "")
            if newer or (pref and k not in preferred):          # a preferred root's first result of a cell always wins
                cells[k] = r
                if pref:
                    preferred.add(k)
    res = sorted(cells.values(), key=lambda r: [str(x) for x in key(r)])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": res}, separators=(",", ":")))
    print(f"{n} files -> {len(res)} cells in {out}")
    if a.history:
        update_history(found + res, a.history)


if __name__ == "__main__":
    main()
