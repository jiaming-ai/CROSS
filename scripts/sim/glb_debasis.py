#!/usr/bin/env python3
"""Rewrite glb files whose textures use KHR_texture_basisu (KTX2 / Basis Universal) into plain PNG textures so that
Blender's glTF importer can read them.  Needs the Khronos `ktx` CLI (KTX-Software >= 4.3).

    python scripts/sim/glb_debasis.py --ktx ~/opt/ktx/bin/ktx path/to/*.glb
The original file is kept as <name>.basisu.glb.
"""
import argparse, json, os, shutil, struct, subprocess, sys, tempfile
from pathlib import Path

JSON_T, BIN_T = 0x4E4F534A, 0x004E4942


def read_glb(path):
    b = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<III", b, 0)
    assert magic == 0x46546C67, "not a glb"
    off = 12; js = None; bin_ = b""
    while off < length:
        clen, ctype = struct.unpack_from("<II", b, off); off += 8
        chunk = b[off:off + clen]; off += clen
        if ctype == JSON_T:
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == BIN_T:
            bin_ = chunk
    return js, bytearray(bin_)


def write_glb(path, js, bin_):
    jb = json.dumps(js, separators=(",", ":")).encode("utf-8")
    jb += b" " * ((4 - len(jb) % 4) % 4)
    bb = bytes(bin_) + b"\0" * ((4 - len(bin_) % 4) % 4)
    total = 12 + 8 + len(jb) + 8 + len(bb)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(jb), JSON_T)); f.write(jb)
        f.write(struct.pack("<II", len(bb), BIN_T)); f.write(bb)


def convert(path, ktx, keep=True):
    js, bin_ = read_glb(path)
    if "KHR_texture_basisu" not in (js.get("extensionsUsed") or []):
        return "skip"
    images = js.get("images", [])
    bvs = js.setdefault("bufferViews", [])
    tmp = Path(tempfile.mkdtemp(prefix="debasis_"))
    n_conv = 0
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
            if r.returncode != 0 or not dst.is_file():
                raise RuntimeError(f"ktx extract failed for image {i}: {r.stderr.strip()[:300]}")
            png = dst.read_bytes()
            while len(bin_) % 4:
                bin_.append(0)
            bvs.append({"buffer": 0, "byteOffset": len(bin_), "byteLength": len(png)})
            bin_.extend(png)
            images[i] = {"mimeType": "image/png", "bufferView": len(bvs) - 1, **({"name": im["name"]} if "name" in im else {})}
            n_conv += 1
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
        if keep:
            backup = Path(str(path)[:-4] + ".basisu.glb")
            if not backup.exists():
                shutil.copy2(path, backup)
        write_glb(path, js, bin_)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return f"converted {n_conv} images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ktx", default="ktx")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    n_ok = n_skip = n_fail = 0
    for f in args.files:
        try:
            r = convert(f, args.ktx)
            if r == "skip":
                n_skip += 1
            else:
                n_ok += 1
                print(f, r, flush=True)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print("FAILED", f, e, flush=True)
    print(f"converted {n_ok}, skipped {n_skip}, failed {n_fail}")


if __name__ == "__main__":
    main()
