"""Convert a saved CROSS map between the storage formats (cross/db/store.py), e.g. an old single-pickle map to format
v2 (graph in map.pkl, images and descriptors in map.pkl.store/), or re-encode a v2 map with another codec.

    python scripts/convert_map.py old/map.pkl new/map.pkl                         # v2, lossless (default codecs)
    python scripts/convert_map.py old/map.pkl new/map.pkl --image-codec jpeg --image-quality 95
    python scripts/convert_map.py new/map.pkl back/map.pkl --format pickle       # the old single file
    python scripts/convert_map.py old/map.pkl new/map.pkl --verify               # check the round trip (exact codecs)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cross.core.config import StorageConfig  # noqa: E402
from cross.db import store  # noqa: E402


def same(a, b, path="") -> list:
    """Differences between two map dicts (images compared decoded; tensors by type, dtype, ltype, value)."""
    import pypose as pp
    if store.is_ref(a):
        a = a.load()
    if store.is_ref(b):
        b = b.load()
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return [f"{path}: {type(a).__name__} vs {type(b).__name__}"]
        if type(a) is not type(b) or a.dtype != b.dtype or a.shape != b.shape:
            return [f"{path}: {type(a).__name__} {a.dtype} {tuple(a.shape)} vs {type(b).__name__} {b.dtype} {tuple(b.shape)}"]
        if isinstance(a, pp.LieTensor) and type(a.ltype) is not type(b.ltype):      # old pickles hold ltype copies
            return [f"{path}: ltype {a.ltype} vs {b.ltype}"]
        at, bt = (a.tensor() if isinstance(a, pp.LieTensor) else a), (b.tensor() if isinstance(b, pp.LieTensor) else b)
        return [] if torch.equal(at.cpu(), bt.cpu()) else [f"{path}: values differ"]
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a.keys()) != list(b.keys()):
            return [f"{path}: keys {list(a.keys())[:5]} vs {list(b.keys())[:5]}"]
        out = []
        for k in a:
            out += same(a[k], b[k], f"{path}/{k}")
            if len(out) > 20:
                break
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return [f"{path}: {type(a).__name__}[{len(a)}] vs {type(b).__name__}[{len(b)}]"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += same(x, y, f"{path}[{i}]")
            if len(out) > 20:
                break
        return out
    if type(a) is not type(b) or a != b:
        return [f"{path}: {a!r} vs {b!r}"]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--format", default="v2", choices=["v2", "pickle"])
    ap.add_argument("--image-codec", default=StorageConfig.image_codec)
    ap.add_argument("--image-quality", type=int, default=StorageConfig.image_quality)
    ap.add_argument("--png-level", type=int, default=StorageConfig.png_level)
    ap.add_argument("--depth-codec", default=StorageConfig.depth_codec)
    ap.add_argument("--descriptor-dtype", default=StorageConfig.descriptor_dtype)
    ap.add_argument("--depth-drop-bits", type=int, default=StorageConfig.depth_drop_bits)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", action="store_true", help="read both back and compare every field")
    ap.add_argument("--stats", help="write the size / time statistics (JSON) here")
    a = ap.parse_args()

    cfg = StorageConfig(format=a.format, image_codec=a.image_codec, image_quality=a.image_quality,
                        png_level=a.png_level, depth_codec=a.depth_codec, descriptor_dtype=a.descriptor_dtype,
                        depth_drop_bits=a.depth_drop_bits,
                        encode_workers=a.workers)
    t0 = time.perf_counter()
    data = store.read_map(a.src)
    t_read = time.perf_counter() - t0
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if a.format == "v2":
        st = store.write_map(a.dst, data, cfg)
        st.pop("pack", None)
    else:
        import pickle
        with open(a.dst, "wb") as f:
            pickle.dump(store.materialize(data), f)
        st = {}
    st.update(read_src_s=round(t_read, 3), write_s=round(time.perf_counter() - t0, 3),
              src_bytes=store.map_bytes(a.src), dst_bytes=store.map_bytes(a.dst))
    print(json.dumps(st, indent=1))
    if a.stats:
        Path(a.stats).write_text(json.dumps(st, indent=1))
    if a.verify:
        t0 = time.perf_counter()
        back = store.read_map(a.dst)
        print(f"read back in {time.perf_counter() - t0:.2f} s")
        diff = same(data, back)
        print("round trip exact" if not diff else "differences:\n  " + "\n  ".join(diff[:20]))
        sys.exit(0 if not diff else 1)


if __name__ == "__main__":
    main()
