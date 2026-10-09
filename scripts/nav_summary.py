"""Summarise CROSS Nav evaluations: one table per window set, rows = checkpoints, columns = datasets x condition.

    python scripts/nav_summary.py <eval_root>/<run> [--metric miou] [--sel all|seen|unseen] [--json out.json]
"""
import argparse
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--metric", default="miou")
    ap.add_argument("--sel", default="all")
    ap.add_argument("--conds", default="oracle,stereo,mono,geom,gtgeom")
    ap.add_argument("--json")
    a = ap.parse_args()
    conds = a.conds.split(",")
    dump = {}
    for wdir in sorted(p for p in Path(a.run_dir).iterdir() if p.is_dir()):
        rows = []
        for f in sorted(wdir.glob("heads_*.json"), key=lambda p: int(re.findall(r"\d+", p.stem)[0])):
            r = json.load(open(f))
            rows.append((f.stem, r))
        if not rows:
            continue
        dss = [d for d in rows[-1][1]["metrics"] if d != "ALL"] + ["ALL"]
        print(f"\n## {wdir.name}  ({a.metric}, {a.sel} labelled cells)")
        hdr = ["ckpt"] + [f"{d}:{c}" for d in dss for c in conds]
        print(" | ".join(hdr))
        for name, r in rows:
            vals = []
            for d in dss:
                for c in conds:
                    v = r["metrics"].get(d, {}).get(f"{c}/{a.sel}", {}).get(a.metric)
                    vals.append("-" if v is None else f"{v:.3f}")
            print(" | ".join([name] + vals))
            dump.setdefault(wdir.name, {})[name] = r
        last = rows[-1][1]
        print("gravity err (deg, mean):", last.get("gravity_deg"))
        print("scale head |log err|:", last.get("scale_head_abs_log_err"))
    if a.json:
        json.dump(dump, open(a.json, "w"))


if __name__ == "__main__":
    main()
