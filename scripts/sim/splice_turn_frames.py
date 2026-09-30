#!/usr/bin/env python3
"""Splice the in-place rotation frames (rendered with --turn-frames-only) into a traversal rendered on the original
station grid.  The result has one frame per station of the new grid; original frames are hard-linked.

    python scripts/sim/splice_turn_frames.py --old <variant_dir> --turns <variant_dir>__turns --out <new_dir>
"""
import argparse, json, os, shutil
from pathlib import Path
import numpy as np


def link(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True); ap.add_argument("--turns", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    old, turns, out = Path(a.old), Path(a.turns), Path(a.out)
    st = json.loads((turns / "stations.json").read_text())
    orig = np.asarray(st["orig_index"]); n_new = len(orig)
    fidx = np.atleast_1d(np.loadtxt(turns / "frame_index.txt", dtype=int)) if (turns / "frame_index.txt").is_file() else np.array([], int)
    turn_pos = {int(k): j for j, k in enumerate(fidx)}                 # new station -> file index in the turns render
    inserted = [int(k) for k in np.nonzero(orig < 0)[0]]
    missing = [k for k in inserted if k not in turn_pos]
    if missing:
        raise SystemExit(f"turn render incomplete: {len(missing)} inserted stations missing (e.g. {missing[:5]})")
    calib = json.loads((old / "calib.json").read_text())
    n_old = calib["n_frames"]
    if n_old != int((orig >= 0).sum()):
        raise SystemExit(f"station grid mismatch: old render has {n_old} frames, new grid keeps {(orig >= 0).sum()} original stations")
    dirs = [d.name for d in old.iterdir() if d.is_dir() and not d.name.startswith("_") and d.name != "right" and d.name != "ids"]
    out.mkdir(parents=True, exist_ok=True)
    for d in dirs:
        (out / d).mkdir(exist_ok=True)
        ext = next(iter(p.suffix for p in (old / d).iterdir()), ".png")
        for k in range(n_new):
            if orig[k] >= 0:
                link(old / d / f"{orig[k]:06d}{ext}", out / d / f"{k:06d}{ext}")
            else:
                link(turns / d / f"{turn_pos[k]:06d}{ext}", out / d / f"{k:06d}{ext}")
    if (old / "right").is_symlink():
        tgt = os.readlink(old / "right")
        if (out / "right").is_symlink(): (out / "right").unlink()
        os.symlink(tgt, out / "right")
    shutil.copy2(turns / "poses_left_all.txt", out / "poses_left.txt")
    for f in ("plan.json", "path_xy.npy", "path_heading.npy", "stations.json"):
        if (turns / f).is_file():
            shutil.copy2(turns / f, out / f)
    calib["n_frames"] = n_new; calib["turn_frames_inserted"] = len(inserted); calib["max_turn_deg_per_frame"] = 10.0
    (out / "calib.json").write_text(json.dumps(calib, indent=1))
    meta = json.loads((old / "meta.json").read_text())
    meta_t = json.loads((turns / "meta.json").read_text()) if (turns / "meta.json").is_file() else {}
    meta["turn_frames"] = meta_t.get("turn_frames", {"inserted": len(inserted)})
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"spliced {out}: {n_new} stations ({n_old} original + {len(inserted)} rotation frames)")


if __name__ == "__main__":
    main()
