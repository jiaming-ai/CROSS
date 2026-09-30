#!/usr/bin/env python3
"""Print the job lines of the benchmark (arguments of benchmark/run.py), one per line: map jobs first, then single-
session (T1) jobs of the other sequences, then query (T2 + T3) jobs.

  python benchmark/jobs.py --dataset openloris --systems cross_rgbd orbslam3 --setups rgbd stereo > queue.txt
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--setups", nargs="*", default=None, help="restrict to these setups")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--tracks", nargs="*", default=["t1", "t2t3"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ds = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())[a.dataset]
    sy = yaml.safe_load((ROOT / "benchmark/configs/systems.yaml").read_text())
    maps, t1, queries = [], [], []
    for system in a.systems:
        for setup in sy[system]["setups"]:
            if (a.setups and setup not in a.setups) or setup not in ds["setups"] or \
                    f"{a.dataset}/{setup}" in sy[system].get("skip", []):
                continue
            for scene, sc in ds["scenes"].items():
                if a.scenes and scene not in a.scenes:
                    continue
                base = f"--dataset {a.dataset} --scene {scene} --system {system} --setup {setup} --seed {a.seed}"
                maps.append(f"{base} --task map")
                if "t1" in a.tracks and ds.get("t1") == "all":
                    for q in sc.get("queries", []):
                        t1.append(f"{base} --task t1 --seq {q}")
                if "t2t3" in a.tracks:
                    for q in sc.get("queries", []):
                        queries.append(f"{base} --task query --query {q}")
    print("\n".join(maps + t1 + queries))


if __name__ == "__main__":
    main()
