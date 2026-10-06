"""uCO3D VGGT-Omega labels (facebook/uco3d on HF, vggt_omega_anno/*.tar, CC BY 4.0) -> scene cache.

Input: <raw>/vggt_omega_anno/<shard>.tar, an uncompressed tar of uncompressed per-sequence tars <seq>.tar, each holding
<seq>/cam_from_worlds.npy (N,3,4 camera-from-world, OpenCV), intrinsics.npy (N,3,3 for the stored image),
image_names.json (N names "00000.png" ...), images/<name>.png (1067 x 1898 portrait) and depths/<stem>.exr (float32
z-depth, 0 = invalid; ~25 % of the pixels valid).  Conventions checked on 1-15085-766-... (GT overlap of frames 3 / 6 /
9 apart, rel_tol 0.02): as stored 0.95 / 0.91 / 0.87, the depth read as ray length 0.95 / 0.88 / 0.85 (the narrow
field of view, f = 2000 px, makes the two close), the pose inverted 0.62 / 0.59 / 0.49.

One scene per sequence (one object, a turntable-like video of ~335 frames; arbitrary units: non-metric, stored with a
median depth of ~2).  Neighbouring frames overlap ~0.95: every --every-th frame is kept.  The shards are read with
random access (no extraction); a shard is converted once it is complete (`.done` + expected size).

python -m vggt_ft.prep.uco3d <raw_dir> <cache_root> [--every 2] [--workers 16] [--sizes <download queue files>]
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import SceneWriter, units_for
from .rawio import Window, dec_any, dec_npy, dec_rgb, load_sizes, ready, split, tar_index

DATASET = "uco3d"


def convert(job):
    shard, member, off, size, out, every, cap = job
    seq = Path(member).name.replace(".tar", "")
    if (Path(out) / DATASET / seq / "scene.json").exists():
        return seq, -1
    with open(shard, "rb") as f:
        t = tarfile.open(fileobj=io.BufferedReader(Window(f, off, size), 1 << 20), mode="r:")
        mem = {m.name: m for m in t if m.isfile()}

        def rd(rel):
            return t.extractfile(mem[f"{seq}/{rel}"]).read()

        E = dec_npy(rd("cam_from_worlds.npy")).astype(np.float64)
        K = dec_npy(rd("intrinsics.npy")).astype(np.float64)
        names = json.loads(rd("image_names.json"))
        sel = [i for i in range(0, len(names), every)
               if f"{seq}/images/{names[i]}" in mem and f"{seq}/depths/{Path(names[i]).stem}.exr" in mem]
        if len(sel) < 2:
            return seq, 0

        def depth(i):
            return dec_any(rd(f"depths/{Path(names[i]).stem}.exr"))

        probe = [depth(sel[k]) for k in np.linspace(0, len(sel) - 1, min(12, len(sel))).astype(int)]
        sw = SceneWriter(out, DATASET, seq, metric=False, kind="object", units=units_for(probe),
                         extra={"shard": Path(shard).name})
        for c, idx in enumerate(split(len(sel), cap)):
            q = sw.sequence(seq if c == 0 else f"{seq}_{c}", session=seq)
            for k in idx:
                i = sel[k]
                T = np.eye(4)
                T[:3, :4] = E[i, :3, :4]
                q.add(Path(names[i]).stem, dec_rgb(rd(f"images/{names[i]}")), depth(i), K[i], T)
    return seq, sw.close()


def _conv(job):
    try:
        return convert(job)
    except Exception as e:
        return job[1], f"ERROR {e!r} {traceback.format_exc(limit=3)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="folder holding vggt_omega_anno/")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sizes", nargs="*", help="download queue files with the expected file sizes")
    ap.add_argument("--max-seqs", type=int, default=0, help="per shard (tests)")
    a = ap.parse_args()
    sizes = load_sizes(a.sizes)
    shards = sorted(p for p in (Path(a.raw) / "vggt_omega_anno").glob("*.tar")
                    if ready(p, sizes, DATASET, f"vggt_omega_anno/{p.name}"))
    jobs = []
    for s in shards:
        idx = [(n, o, z) for n, (o, z) in tar_index(s).items() if n.endswith(".tar")]
        idx = idx[:a.max_seqs] if a.max_seqs else idx
        jobs += [(str(s), n, o, z, a.out, a.every, a.cap) for n, o, z in idx
                 if not (Path(a.out) / DATASET / Path(n).name.replace(".tar", "") / "scene.json").exists()]
    print(f"{len(shards)} shards, {len(jobs)} sequences to convert", flush=True)
    with Pool(a.workers) as p:
        for seq, n in p.imap_unordered(_conv, jobs):
            print(seq, n, flush=True)


if __name__ == "__main__":
    main()
