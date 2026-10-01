#!/usr/bin/env python3
"""List re-run jobs (benchmark/run.py argument lines) for every failed result under a results root.

  python benchmark/redo_failed.py <results_root> [--systems ...] [--datasets ...] [--apply] > redo.txt

- a failed map (T1 of the scene's map session): with --apply its folder and T1 result are removed, and a `--task map` line
  is printed (the query jobs of the scene then re-run too, as their results depend on the map);
- a failed T1 of another session: `--task t1 --seq S --force`;
- failed T2 / T3 cells: `--task query --query Q --only <tracks> --redo <tracks>`.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--apply", action="store_true", help="remove failed maps so that they are rebuilt")
    a = ap.parse_args()
    ds = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())
    maps, t1, queries = [], [], defaultdict(set)
    failed_maps = set()
    for f in sorted(Path(a.root).rglob("result.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:      # noqa: BLE001
            continue
        if r.get("status") != "failed":
            continue
        if (a.systems and r.get("system") not in a.systems) or (a.datasets and r.get("dataset") not in a.datasets):
            continue
        base = f"--dataset {r['dataset']} --scene {r['scene']} --system {r['system']} --setup {r['setup']} --seed {r.get('seed', 0)}"
        scene = ds[r["dataset"]]["scenes"][r["scene"]]
        if r["track"] == "t1":
            if r.get("sequence") == scene["map"]:
                run_root = f.parents[2]                     # <root>/.../s<seed>/t1/<seq>/result.json
                failed_maps.add((base, run_root, scene["map"]))
            else:
                t1.append(f"{base} --task t1 --seq {r['sequence']} --force")
        elif r.get("error") == "mapping run failed":
            continue                                        # re-run with its map
        else:
            queries[(base, r["query"])].add(r["track"])
    for base, run_root, m in sorted(failed_maps, key=lambda x: x[0]):
        if a.apply:
            shutil.rmtree(run_root / "maps" / m, ignore_errors=True)
            (run_root / "t1" / m / "result.json").unlink(missing_ok=True)
            for track in ("t2", "t3"):                       # the scene's query results used the failed map
                shutil.rmtree(run_root / track, ignore_errors=True)
        maps.append(f"{base} --task map")
        for q in ds_queries(ds, base):
            queries[(base, q)] |= {"t2", "t3"}
    lines = maps + t1 + [f"{b} --task query --query {q} --only {' '.join(sorted(t))} --redo {' '.join(sorted(t))}"
                         for (b, q), t in sorted(queries.items())]
    print("\n".join(lines))


def ds_queries(ds, base):
    parts = base.split()
    d, scene = parts[parts.index("--dataset") + 1], parts[parts.index("--scene") + 1]
    return ds[d]["scenes"][scene].get("queries", [])


if __name__ == "__main__":
    main()
