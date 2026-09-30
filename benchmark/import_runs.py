#!/usr/bin/env python3
"""Import relocalization runs made directly with scripts/map_and_reloc(_rgbd).py (outside benchmark/run.py) as T3
result.json cells, e.g. the SimChange v2 evaluation laid out as <root>/<scene>/<mode>/s<seed>/<variant>/reloc_summary.json
(the map traversal is <root>/<scene>/<mode>/s<seed>/map).

  python benchmark/import_runs.py <root> <out_dir> --dataset simchange --modes rgbd=cross_rgbd:rgbd stereo=cross_stereo:stereo

Only runs whose trial settings match the dataset's protocol (trial length) are imported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark" / "eval"))
from metrics import wilson  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--dataset", default="simchange")
    ap.add_argument("--modes", nargs="+", required=True, help="mode=system:setup")
    ap.add_argument("--source", default="imported", help="note stored with every cell")
    ap.add_argument("--calibrated", nargs="*", default=[],
                    help="scenes whose runs used a noise model calibrated without ground truth (PROTOCOL section 2)")
    a = ap.parse_args()
    dcfg = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())[a.dataset]
    sy = yaml.safe_load((ROOT / "benchmark/configs/systems.yaml").read_text())
    modes = {m.split("=")[0]: m.split("=")[1].split(":") for m in a.modes}
    n = 0
    for f in sorted(Path(a.root).glob("*/*/s*/*/reloc_summary.json")):
        variant, seed_dir, mode, scene = f.parent.name, f.parent.parent.name, f.parent.parent.parent.name, f.parent.parent.parent.parent.name
        if mode not in modes or variant == "map":
            continue
        system, setup = modes[mode]
        summ = json.loads(f.read_text())
        if summ.get("trial_len") not in (None, dcfg["trial_len"]):
            continue
        tr = summ["trials"]
        k = int(round((tr["RS"] or 0) * tr["n_trials"]))
        trials = [{"start": t.get("start"), "success": t["success_rd"], "success_strict": t["success_1m_5deg"],
                   "final_err": t["final_t_err"]} for t in tr["trials"]]
        fails = sorted(t["final_err"] for t in trials if not t["success"] and t["final_err"] is not None)
        res = {"dataset": a.dataset, "scene": scene, "system": system, "setup": setup, "seed": int(seed_dir[1:]),
               "label": sy[system]["label"], "uses_odometry": sy[system].get("uses_odometry", False), "track": "t3",
               "map": "map", "query": variant, "n_trials": tr["n_trials"], "n_success": k, "rs": tr["RS"],
               "rs_strict": tr["RS_1m_5deg"], "rs_ci95": wilson(k, tr["n_trials"]),
               "fail_err_median": fails[len(fails) // 2] if fails else None, "trials": trials, "r_d": tr.get("r_d", dcfg["r_d"]),
               "trial_len": dcfg["trial_len"], "status": "ok", "source": a.source, "calibrated": scene in a.calibrated,
               "time": __import__("time").strftime("%Y-%m-%d %H:%M:%S", __import__("time").localtime(f.stat().st_mtime))}
        o = Path(a.out) / a.dataset / scene / system / setup / seed_dir / "t3" / f"map__{variant}" / "result.json"
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(res, indent=1, default=lambda x: None))
        n += 1
    print(f"imported {n} runs")


if __name__ == "__main__":
    main()
