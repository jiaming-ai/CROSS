#!/usr/bin/env python3
"""Write the clipped sequences of the development split (`clips:` of the dev entries in benchmark/configs/datasets.yaml).

A clip is the first N frames of a prepared sequence, written next to it as <data>/<dataset>/<clip>/<setup_dir>/:
per-frame folders hold symlinks to the first N frames, per-frame text files (times, poses, odometry) their first N
lines, and the other files (calibration) are copied.  The clip is therefore a normal benchmark sequence.

  python benchmark/datasets/make_dev.py --data $BENCH_DATA
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FRAME_DIRS = ("left", "rgb")          # the folder whose sorted files define the frames


def clip_setup(src: Path, dst: Path, n: int):
    ref = next((src / d for d in FRAME_DIRS if (src / d).is_dir()), None)
    if ref is None:
        raise FileNotFoundError(f"no frame folder ({FRAME_DIRS}) in {src}")
    frames = sorted(p for p in ref.iterdir() if not p.name.startswith("."))
    if len(frames) < n:
        raise ValueError(f"{src} has {len(frames)} frames, fewer than {n}")
    stems = {p.stem for p in frames[:n]}
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        out = dst / item.name
        if item.is_dir():           # per-frame folder (images, depth, the SGBM cache): link the first n frames
            out.mkdir(exist_ok=True)
            for f in item.iterdir():
                if f.stem in stems and not (out / f.name).exists():
                    os.symlink(f.resolve(), out / f.name)
        elif item.suffix == ".txt" and _n_lines(item) == len(frames):     # one line per frame
            out.write_text("".join(item.read_text().splitlines(keepends=True)[:n]))
        else:
            shutil.copy2(item, out)


def _n_lines(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("BENCH_DATA"), required=os.environ.get("BENCH_DATA") is None)
    ap.add_argument("--force", action="store_true", help="rewrite existing clips")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())
    for name, d in cfg.items():
        if not d.get("dev"):
            continue
        base = Path(a.data) / d.get("data", name)
        for clip, (seq, n) in (d.get("clips") or {}).items():
            out = base / clip
            if out.exists() and not a.force:
                print(f"exists: {out}")
                continue
            shutil.rmtree(out, ignore_errors=True)
            for setup_dir in sorted(set(d["setups"].values())):
                src = base / seq / setup_dir
                if src.is_dir():
                    clip_setup(src, out / setup_dir, int(n))
            print(f"wrote {out} ({seq}, first {n} frames)")


if __name__ == "__main__":
    main()
