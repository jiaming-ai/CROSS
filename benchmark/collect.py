#!/usr/bin/env python3
"""Merge the result.json files of benchmark runs into benchmark/results/results.json.

  python benchmark/collect.py <results_root> [<results_root> ...] [--out benchmark/results/results.json]

Results of the same cell (track, dataset, scene, system, setup, seed, sequence / map+query) found in several roots
(e.g. a rerun) are resolved by the newest `time`.  Per-frame curves and trajectories stay in the merged file
(downsampled by run.py), so the web page needs nothing else.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def key(r):
    return (r.get("track"), r.get("dataset"), r.get("scene"), r.get("system"), r.get("setup"), r.get("seed"),
            r.get("sequence"), r.get("map"), r.get("query"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "benchmark/results/results.json"))
    ap.add_argument("--merge", action="store_true", help="keep the entries already in --out")
    a = ap.parse_args()
    cells = {}
    out = Path(a.out)
    if a.merge and out.is_file():
        for r in json.loads(out.read_text())["results"]:
            cells[key(r)] = r
    n = 0
    for root in a.roots:
        for f in Path(root).rglob("result.json"):
            try:
                r = json.loads(f.read_text())
            except Exception as e:      # noqa: BLE001
                print(f"skip {f}: {e}")
                continue
            n += 1
            k = key(r)
            if k not in cells or (r.get("time") or "") >= (cells[k].get("time") or ""):
                cells[k] = r
    res = sorted(cells.values(), key=lambda r: [str(x) for x in key(r)])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": res}, separators=(",", ":")))
    print(f"{n} files -> {len(res)} cells in {out}")


if __name__ == "__main__":
    main()
