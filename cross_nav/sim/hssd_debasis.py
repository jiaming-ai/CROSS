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
    root = os.environ["SIMCHANGE_ROOT"]      # SimChange checkout
    sys.path.insert(0, os.path.join(root, "simchange"))
    import glb_debasis                                  # noqa: E402
    return glb_debasis


_GRAY_PNG = None


def _gray_png() -> bytes:
    global _GRAY_PNG
    if _GRAY_PNG is None:
        import struct
        import zlib
        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
        raw = b"".join(b"\x00" + bytes([128, 128, 128, 255]) * 4 for _ in range(4))
        _GRAY_PNG = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 6, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return _GRAY_PNG


def convert_tolerant(path, ktx):
    """SimChange glb_debasis.convert, but an image that ktx cannot transcode (non-conformant level index in a few
    hssd-hab files) becomes a 4x4 grey PNG instead of failing the whole object (whose geometry would then be lost)."""
    import json
    import shutil
    import subprocess
    import tempfile
    gd = _sc()
    js, bin_ = gd.read_glb(path)
    if "KHR_texture_basisu" not in (js.get("extensionsUsed") or []):
        return "skip"
    images, bvs = js.get("images", []), js.setdefault("bufferViews", [])
    tmp = Path(tempfile.mkdtemp(prefix="debasis_"))
    n_conv = n_grey = 0
    try:
        for i, im in enumerate(images):
            if im.get("mimeType") != "image/ktx2":
                continue
            bv = bvs[im["bufferView"]]
            data = bytes(bin_[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]])
            src, dst = tmp / f"{i}.ktx2", tmp / f"{i}.png"
            src.write_bytes(data)
            r = subprocess.run([ktx, "extract", "--transcode", "rgba8", "--level", "0", str(src), str(dst)],
                               capture_output=True, text=True)
            if r.returncode == 0 and dst.is_file():
                png = dst.read_bytes()
                n_conv += 1
            else:
                png = _gray_png()
                n_grey += 1
            while len(bin_) % 4:
                bin_.append(0)
            bvs.append({"buffer": 0, "byteOffset": len(bin_), "byteLength": len(png)})
            bin_.extend(png)
            images[i] = {"mimeType": "image/png", "bufferView": len(bvs) - 1, **({"name": im["name"]} if "name" in im else {})}
        for tex in js.get("textures", []):
            ext = (tex.get("extensions") or {}).pop("KHR_texture_basisu", None)
            if ext is not None:
                tex["source"] = ext["source"]
            if "extensions" in tex and not tex["extensions"]:
                del tex["extensions"]
        for key in ("extensionsUsed", "extensionsRequired"):
            if key in js:
                js[key] = [e for e in js[key] if e != "KHR_texture_basisu"]
                if not js[key]:
                    del js[key]
        js["buffers"][0]["byteLength"] = len(bin_)
        out = Path(str(path) + ".tmpglb")
        gd.write_glb(out, js, bin_)
        os.replace(out, path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return f"converted {n_conv} images, {n_grey} grey"


def _one(args):
    path, ktx, mark_dir = args
    m = Path(mark_dir) / hashlib.md5(path.encode()).hexdigest()
    if m.exists():
        return "cached"
    try:
        r = convert_tolerant(path, ktx)
        m.write_text(r)
        return "skip" if r == "skip" else ("grey" if not r.endswith(" 0 grey") else "converted")
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
