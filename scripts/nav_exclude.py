"""Write <bev_root>/<dataset>/exclude.txt: places whose label statistics fail a check (occupied fraction of the labelled
cells > --max-occ, or fewer than --min-free free cells).  python scripts/nav_exclude.py <labels log> <bev_root>/<dataset>"""
import argparse
import ast
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("out_dir")
ap.add_argument("--max-occ", type=float, default=0.85)
ap.add_argument("--min-free", type=int, default=500)
a = ap.parse_args()
bad = []
for line in open(a.log):
    m = re.search(r"\] (\S+) (\{.*\})$", line.strip())
    if not m:
        continue
    try:
        s = ast.literal_eval(m.group(2))
    except (ValueError, SyntaxError):
        continue
    if s.get("free") is None:
        continue
    if s["occ"] / max(1, s["free"] + s["occ"]) > a.max_occ or s["free"] < a.min_free:
        bad.append(m.group(1))
Path(a.out_dir, "exclude.txt").write_text("\n".join(sorted(set(bad))) + "\n")
print(f"{a.out_dir}: {len(set(bad))} excluded")
