"""Convert the KTX2 (Basis Universal) textures of downloaded hssd-hab glb files to PNG in place, once, in parallel
(SimChange glb_debasis.convert; Blender's glTF importer cannot read KHR_texture_basisu).  Doing it here, before any
Blender conversion, avoids two scene conversions rewriting the same shared object file at the same time.

    LD_LIBRARY_PATH=<ktx>/lib python -m cross_nav.sim.hssd_debasis --root <raw> --ktx <ktx>/bin/ktx [--workers 32]

A marker file <root>/.debasis/<md5 of path> records converted / skipped files, so reruns only touch new downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from multiprocessing import Pool
from pathlib import Path


def _sc():
    root = os.environ.get("SIMCHANGE_ROOT", "/data2/projects/cross_v2/SimChange")
    sys.path.insert(0, os.path.join(root, "simchange"))
    import glb_debasis                                  # noqa: E402
    return glb_debasis


def _one(args):
    path, ktx, mark_dir = args
    m = Path(mark_dir) / hashlib.md5(path.encode()).hexdigest()
    if m.exists():
        return "cached"
    try:
        r = _sc().convert(path, ktx, keep=False)
        m.write_text(r)
        return "skip" if r == "skip" else "converted"
    except Exception as e:                              # noqa: BLE001
        return f"FAILED {path} {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ktx", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    root = Path(a.root)
    mark = root / ".debasis"
    mark.mkdir(exist_ok=True)
    files = [str(p) for d in ("objects", "stages") if (root / d).exists() for p in (root / d).rglob("*.glb")
             if not p.name.endswith((".collider.glb", ".filteredSupportSurface.glb", ".basisu.glb"))]
    cnt = {}
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(_one, [(f, a.ktx, str(mark)) for f in files], chunksize=8):
            k = r.split()[0]
            cnt[k] = cnt.get(k, 0) + 1
            if k == "FAILED":
                print(r, flush=True)
    print(f"{len(files)} glb files: {cnt}", flush=True)


if __name__ == "__main__":
    main()
