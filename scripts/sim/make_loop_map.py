#!/usr/bin/env python3
"""Make a closed-loop map sequence <scene>/map_loop from <scene>/map by appending a replay of the first
`n_extra` frames (same direction), so that the mapping session revisits its start and can close the loop.
Frames are symlinked; poses/calib/meta are rewritten.  Usage: make_loop_map.py data/sim/lonemonk 150"""
import json, os, sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1]).resolve(); n_extra = int(sys.argv[2]) if len(sys.argv) > 2 else 150
src, dst = root / "map", root / "map_loop"
poses = np.loadtxt(src / "poses_left.txt")
n = len(poses)
order = list(range(n)) + list(range(min(n_extra, n)))
dirs = [d.name for d in src.iterdir() if d.is_dir() and not d.name.startswith("_")]
dst.mkdir(exist_ok=True)
for d in dirs:
    (dst / d).mkdir(exist_ok=True)
    files = sorted((src / d).iterdir())
    if len(files) != n:
        continue
    for k, i in enumerate(order):
        link = dst / d / f"{k:06d}{files[i].suffix}"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(files[i], link.parent), link)
np.savetxt(dst / "poses_left.txt", poses[order], fmt="%.9f")
for f in ("calib.json", "meta.json"):
    if (src / f).is_file():
        d = json.loads((src / f).read_text())
        if f == "meta.json":
            d["n_frames"] = len(order); d["loop_replay"] = n_extra
        (dst / f).write_text(json.dumps(d, indent=1))
print("map_loop:", len(order), "frames from", n, "+", n_extra, "in", dirs)
