"""Compact map storage (format v2) and lazily decoded keyframe images.

A saved map is `map.pkl` plus a sidecar directory next to the real file (symlinks are resolved first, so a symlinked
`map.pkl` finds its data):

    map.pkl                     the graph: keyframe poses, odometry / visual edges, metadata and the image index, as
                                numpy columns (no per-edge dicts of small tensors), and the config
    map.pkl.store/
        images-<uid>.pack       encoded keyframe images, appended one after another
        descriptors.npy         retrieval descriptors (N, D), float32 or float16 (StorageConfig.descriptor_dtype)

`map.pkl` holds the image index (keyframe id, image kind, offset, length, codec, shape), so it is the consistent snapshot
of the map; a pack is only appended to, so a map saved again where it was loaded from only adds the new keyframes.
Packs that no saved map references any more are deleted after the save.

Colour images are stored with `StorageConfig.image_codec` (png / webp_lossless: exact; jpeg / webp: lossy at
`image_quality`), depth (fp16) as its bit pattern in a 16-bit PNG (exact).  A loaded keyframe keeps an `ImageRef` in
place of its tensor; `Keyframe.raw_rgb_image` etc. decode it on access through one LRU cache of decoded images
(`StorageConfig.decode_cache`), so a map with 10^5-10^6 keyframes does not hold its images in memory.
`ImageSpool` does the same during a live run (`StorageConfig.max_ram_images`): images beyond the newest N are encoded
into a spool pack and dropped from memory.

Maps written by older code (one pickle with every tensor) still load: `read_map` returns their dict unchanged.
"""
from __future__ import annotations

import atexit
import os
import pickle
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from cross.core.types import PackedTensor

try:
    import cv2
except ImportError:          # pragma: no cover - opencv is a dependency of the package
    cv2 = None

FORMAT = "cross-map"
FORMAT_VERSION = 2
SIDECAR_SUFFIX = ".store"
IMAGE_FIELDS = ("raw_rgb_image", "depth_image", "raw_rgb_right")
COLOUR_CODECS = ("png", "webp_lossless", "jpeg", "webp", "raw")
LOSSLESS_CODECS = ("png", "webp_lossless", "png16", "raw", "zstd")
_CODEC_IDS = {"raw": 0, "png": 1, "webp_lossless": 2, "jpeg": 3, "webp": 4, "png16": 5, "zstd": 6}
_CODEC_NAMES = {v: k for k, v in _CODEC_IDS.items()}


def sidecar_dir(map_path) -> Path:
    """Sidecar directory of a map file (next to the real file: a symlinked map.pkl finds the data of its target)."""
    return Path(os.path.realpath(str(map_path)) + SIDECAR_SUFFIX)


def map_bytes(map_path) -> int:
    """Bytes on disk of a saved map: the map file and its sidecar directory."""
    real = Path(os.path.realpath(str(map_path)))
    n = real.stat().st_size if real.exists() else 0
    side = sidecar_dir(real)
    if side.is_dir():
        n += sum(p.stat().st_size for p in side.rglob("*") if p.is_file())
    return n


def remove_sidecar(map_path) -> None:
    """Delete the sidecar directory of a map file (a map rewritten in the old single-file format)."""
    side = sidecar_dir(map_path)
    if side.is_dir():
        shutil.rmtree(side, ignore_errors=True)


def remove_map(map_path) -> None:
    """Delete a saved map: the map file and its sidecar directory."""
    p = Path(map_path)
    side = sidecar_dir(p) if p.exists() else None
    if p.is_symlink() or p.exists():
        p.unlink()
    if side is not None and side.is_dir():
        shutil.rmtree(side, ignore_errors=True)


# --------------------------------------------------------------------------------------------------------------------
# codecs
# --------------------------------------------------------------------------------------------------------------------

def _np_dtype(name: str):
    return np.dtype(name)


def encode_array(arr: np.ndarray, codec: str, quality: int = 95, png_level: int = 3, depth_drop_bits: int = 0) -> bytes:
    """Encode one image array: colour (3, H, W) uint8 (RGB), depth (1, H, W) float16, anything else with raw / zstd.
    depth_drop_bits (png16): round away that many low mantissa bits of the fp16 depth (relative error <= 2^(b-11));
    the noise in the low bits of resized sensor depth is what keeps it from compressing."""
    if codec == "raw":
        return np.ascontiguousarray(arr).tobytes()
    if codec == "zstd":
        import zstandard
        return zstandard.ZstdCompressor(level=3).compress(np.ascontiguousarray(arr).tobytes())
    if codec == "png16":
        if arr.dtype != np.float16:
            raise ValueError("png16 stores float16 depth")
        a = np.ascontiguousarray(arr.reshape(arr.shape[-2:])).view(np.uint16)
        if depth_drop_bits > 0:
            b = int(depth_drop_bits)
            keep = np.uint16(0xFFFF ^ ((1 << b) - 1))
            finite = (a & np.uint16(0x7C00)) != np.uint16(0x7C00)          # leave inf / nan alone
            r = ((a.astype(np.uint32) + (1 << (b - 1))) & keep).astype(np.uint16)
            a = np.where(finite, r, a)
        ok, buf = cv2.imencode(".png", a, [cv2.IMWRITE_PNG_COMPRESSION, png_level])
    else:
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[0] not in (1, 3):
            raise ValueError(f"{codec} stores uint8 images (C, H, W), got {arr.dtype} {arr.shape}")
        hwc = np.ascontiguousarray(arr.transpose(1, 2, 0)[..., ::-1] if arr.shape[0] == 3 else arr[0])   # RGB -> BGR
        if codec == "png":
            ok, buf = cv2.imencode(".png", hwc, [cv2.IMWRITE_PNG_COMPRESSION, png_level])
        elif codec == "webp_lossless":
            ok, buf = cv2.imencode(".webp", hwc, [cv2.IMWRITE_WEBP_QUALITY, 101])
        elif codec == "webp":
            ok, buf = cv2.imencode(".webp", hwc, [cv2.IMWRITE_WEBP_QUALITY, int(min(quality, 100))])
        elif codec == "jpeg":
            ok, buf = cv2.imencode(".jpg", hwc, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        else:
            raise ValueError(f"unknown image codec {codec}")
    if not ok:
        raise RuntimeError(f"encoding with {codec} failed")
    return buf.tobytes()


def decode_array(blob, codec: str, shape, dtype: str) -> np.ndarray:
    """Inverse of encode_array: an array of `shape` / `dtype`."""
    dt = _np_dtype(dtype)
    if codec == "raw":
        return np.frombuffer(blob, dt).reshape(shape).copy()
    if codec == "zstd":
        import zstandard
        return np.frombuffer(zstandard.ZstdDecompressor().decompress(blob), dt).reshape(shape).copy()
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"decoding a {codec} image failed")
    if codec == "png16":
        return img.view(np.float16).reshape(shape)
    if img.ndim == 2:
        return img.reshape(shape)
    return np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))                                   # BGR -> RGB, CHW


def codec_for(kind: str, dtype, cfg) -> str:
    """The codec of one stored image (by its field and dtype): the configured one when it can hold the image (exactly,
    or lossy by choice); float images (store_images_uint8 off) and other depth only with an exact byte codec."""
    dt = str(dtype).replace("torch.", "")
    if kind == "depth_image":
        dc = getattr(cfg, "depth_codec", "png16")
        if dt == "float16" and dc == "png16":
            return "png16"
        return "zstd" if dc == "zstd" else "raw"
    if dt != "uint8":
        return "raw"
    c = getattr(cfg, "image_codec", "webp_lossless")
    if c not in COLOUR_CODECS:
        raise ValueError(f"storage.image_codec: one of {COLOUR_CODECS}, not {c}")
    return c


# --------------------------------------------------------------------------------------------------------------------
# packs, references, decoded cache
# --------------------------------------------------------------------------------------------------------------------

class _LRU:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.d: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key):
        with self.lock:
            v = self.d.get(key)
            if v is not None:
                self.d.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
            return v

    def put(self, key, value):
        if self.capacity <= 0:
            return
        with self.lock:
            self.d[key] = value
            self.d.move_to_end(key)
            while len(self.d) > self.capacity:
                self.d.popitem(last=False)

    def clear(self):
        with self.lock:
            self.d.clear()


_DECODED = _LRU(1024)


def set_decode_cache(capacity: int) -> None:
    """Number of decoded keyframe images kept in memory (process-wide)."""
    _DECODED.capacity = int(capacity)
    with _DECODED.lock:
        while len(_DECODED.d) > _DECODED.capacity:
            _DECODED.d.popitem(last=False)


def decode_cache_stats() -> dict:
    return {"capacity": _DECODED.capacity, "size": len(_DECODED.d), "hits": _DECODED.hits, "misses": _DECODED.misses}


class ImagePack:
    """An append-only file of encoded images, identified by a uid (in its file name)."""

    _open: Dict[str, "ImagePack"] = {}
    _open_lock = threading.Lock()

    def __init__(self, path: Path, uid: str):
        self.path = Path(path)
        self.uid = uid
        self._lock = threading.Lock()
        self._fd = None
        self._pid = None

    @classmethod
    def get(cls, path, uid: str) -> "ImagePack":
        """The process-wide pack object of a file (one file handle, shared by every reference into it)."""
        key = os.path.realpath(str(path))
        with cls._open_lock:
            p = cls._open.get(key)
            if p is None or p.uid != uid:
                p = ImagePack(Path(key), uid)
                cls._open[key] = p
            return p

    @classmethod
    def create(cls, directory: Path) -> "ImagePack":
        uid = uuid.uuid4().hex[:16]
        path = Path(directory) / f"images-{uid}.pack"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return cls.get(path, uid)

    def _read_fd(self):
        if self._fd is None or self._pid != os.getpid():
            self._fd = os.open(str(self.path), os.O_RDONLY)
            self._pid = os.getpid()
        return self._fd

    def read(self, offset: int, length: int) -> bytes:
        data = os.pread(self._read_fd(), int(length), int(offset))
        if len(data) != int(length):
            raise IOError(f"{self.path}: short read at {offset} ({len(data)} of {length} bytes)")
        return data

    def append(self, blobs: List[bytes]) -> List[int]:
        """Append encoded images; returns their offsets."""
        offsets = []
        with self._lock:
            with open(self.path, "ab") as f:
                off = f.seek(0, os.SEEK_END)
                for b in blobs:
                    offsets.append(off)
                    f.write(b)
                    off += len(b)
        return offsets

    def size(self) -> int:
        return self.path.stat().st_size

    def __getstate__(self):
        return {"path": str(self.path), "uid": self.uid}

    def __setstate__(self, state):
        self.__init__(Path(state["path"]), state["uid"])


class ImageRef:
    """A keyframe image stored in a pack; decoded on `load()` (the Keyframe image fields call it on access)."""
    __slots__ = ("pack", "offset", "length", "codec", "shape", "dtype", "device")
    _cross_image_ref = True

    def __init__(self, pack: ImagePack, offset: int, length: int, codec: str, shape, dtype: str, device="cpu"):
        self.pack = pack
        self.offset = int(offset)
        self.length = int(length)
        self.codec = codec
        self.shape = tuple(int(s) for s in shape)
        self.dtype = str(dtype)
        self.device = str(device)

    def to(self, device) -> "ImageRef":
        return ImageRef(self.pack, self.offset, self.length, self.codec, self.shape, self.dtype, device)

    def blob(self) -> bytes:
        return self.pack.read(self.offset, self.length)

    def load(self) -> torch.Tensor:
        key = (self.pack.uid, self.offset, self.device)
        t = _DECODED.get(key)
        if t is None:
            arr = decode_array(self.blob(), self.codec, self.shape, self.dtype)
            t = torch.from_numpy(arr)
            if self.device != "cpu":
                t = t.to(self.device)
            _DECODED.put(key, t)
        return t

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * _np_dtype(self.dtype).itemsize

    def __repr__(self):
        return f"ImageRef({self.pack.path.name}@{self.offset}+{self.length}, {self.codec}, {self.shape})"

    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for s in self.__slots__:
            setattr(self, s, state[s])


def is_ref(v) -> bool:
    return getattr(type(v), "_cross_image_ref", False)


def _to_numpy_image(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().contiguous().numpy()


# --------------------------------------------------------------------------------------------------------------------
# live spool: images of a running session beyond the newest N are encoded and dropped from memory
# --------------------------------------------------------------------------------------------------------------------

class ImageSpool:
    """Live-run handling of keyframe images (format v2), in background threads:
    - encode_ahead: every new keyframe's images are encoded right away into a spool pack; the keyframe keeps its
      tensor and holds the reference as `_pre_<field>`, so save_map copies bytes instead of encoding (a 10^4-keyframe
      session saves in seconds, not minutes).  Results are unchanged: the tensors stay what they were.
    - max_ram > 0: only the newest max_ram keyframes keep their images as tensors; older ones are replaced by their
      encoded references (decoded again on access) and dropped from memory.
    The spool lives in `spill_dir` (default: a temporary directory removed at exit); after a save the encoded images
    are re-pointed to the map's pack and later ones are written there (retarget)."""

    def __init__(self, cfg, max_ram: int, encode_ahead: bool = True):
        self.cfg = cfg
        self.max_ram = int(max_ram)
        self.encode_ahead = bool(encode_ahead)
        root = getattr(cfg, "spill_dir", None)
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
            self.dir = Path(tempfile.mkdtemp(prefix="cross_spool_", dir=root))
        else:
            self.dir = Path(tempfile.mkdtemp(prefix="cross_spool_"))
        atexit.register(shutil.rmtree, str(self.dir), True)
        self.pack = ImagePack.create(self.dir)
        self.resident = OrderedDict()              # kf id -> (keyframe whose images are still tensors, its encoding)
        # FIFO: a keyframe's encoding is taken before its eviction, so an eviction waiting for it cannot deadlock
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cross-spool")
        self.pending = []

    def add(self, kf) -> None:
        # host copies taken here, in the caller's thread: the background threads never touch CUDA (a device copy from
        # another thread can break a CUDA-graph capture of the pose estimator)
        items = [(f, _to_numpy_image(t), str(t.device), t.dtype) for f in IMAGE_FIELDS
                 for t in [kf.stored_image(f)] if torch.is_tensor(t)]
        if not items:
            return
        fut = self.pool.submit(self._encode, kf, items)
        self.pending.append(fut)
        if self.max_ram > 0:
            self.resident[kf.id] = (kf, fut)
            while len(self.resident) > self.max_ram:
                _, (old, ofut) = self.resident.popitem(last=False)
                self.pending.append(self.pool.submit(self._evict, old, ofut))
        self.pending = [p for p in self.pending if not p.done()]

    def _encode(self, kf, items) -> None:
        for f, arr, device, dtype in items:
            codec = codec_for(f, dtype, self.cfg)
            blob = encode_array(arr, codec, getattr(self.cfg, "image_quality", 95), getattr(self.cfg, "png_level", 3),
                                getattr(self.cfg, "depth_drop_bits", 0))
            off = self.pack.append([blob])[0]
            kf.__dict__["_pre_" + f] = ImageRef(self.pack, off, len(blob), codec, arr.shape, str(arr.dtype), device)

    def _evict(self, kf, encoding) -> None:
        encoding.result()
        for f in IMAGE_FIELDS:
            pre = kf.__dict__.pop("_pre_" + f, None)
            if pre is not None and torch.is_tensor(kf.stored_image(f)):
                setattr(kf, f, pre)

    def flush(self) -> None:
        for p in self.pending:
            p.result()
        self.pending = []

    def retarget(self, pack: ImagePack) -> None:
        """Spill into a saved map's pack from now on (after save_map re-pointed the spooled images to it): later saves
        to the same map keep them instead of copying the spool again.  The old spool file is deleted."""
        self.flush()
        if pack.uid == self.pack.uid:
            return
        old, self.pack = self.pack, pack
        if old.path.parent == self.dir:
            old.path.unlink(missing_ok=True)


# --------------------------------------------------------------------------------------------------------------------
# columns: lists of homogeneous records <-> numpy arrays
# --------------------------------------------------------------------------------------------------------------------

_LTYPES = {"SE3Type": "SE3_type", "se3Type": "se3_type", "SO3Type": "SO3_type", "so3Type": "so3_type",
           "Sim3Type": "Sim3_type", "sim3Type": "sim3_type", "RxSO3Type": "RxSO3_type", "rxso3Type": "rxso3_type"}


_decode_opts = threading.local()


def _tensor_kind(v):
    """('lie', ltype) / ('tensor', None) / None for a value that can go into a stacked numeric column.  The ltype is
    matched by its class: a LieTensor unpickled from an old map carries a copy of the ltype object, decoding gives the
    pypose singleton (as a live session has)."""
    import pypose as pp
    if isinstance(v, pp.LieTensor):
        name = _LTYPES.get(type(v.ltype).__name__)
        return ("lie", name) if name is not None else None
    if type(v) is torch.Tensor:
        return ("tensor", None)
    return None


_NP_OK = {torch.float16, torch.float32, torch.float64, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
          torch.bool}


def _no_torch_function():
    """Context in which tensor subclasses (pypose LieTensor) behave as plain tensors."""
    import contextlib
    ctx = getattr(torch._C, "DisableTorchFunctionSubclass", None)
    return ctx() if ctx is not None else contextlib.nullcontext()


def lie_tensor(arr: np.ndarray, ltype_name: str):
    """pp.LieTensor(torch.from_numpy(arr), ltype=pp.<ltype_name>) without its torch-function overhead (3x faster; a
    map holds two LieTensors per edge)."""
    import pypose as pp
    t = torch.Tensor.as_subclass(torch.from_numpy(arr), pp.LieTensor)
    t.__dict__["ltype"] = getattr(pp, ltype_name)
    return t


def to_device(t, device):
    """t.to(device), skipped when t is already there (.to returns t itself then; a LieTensor's .to costs ~60 us)."""
    if t is None:
        return t
    if type(t) is PackedTensor:
        return t.to(device)
    if torch.is_tensor(t) and device is not None:
        with _no_torch_function():                       # .device of a LieTensor otherwise costs ~18 us
            same = t.device == torch.device(device)
        if same:
            return t
    return t.to(device)


def _encode_column(vals: list):
    """One column of values -> a compact representation (exact)."""
    present = [v for v in vals if v is not None]
    if not present:
        return {"k": "none", "n": len(vals)}
    mask = None if len(present) == len(vals) else np.array([v is not None for v in vals], dtype=bool)
    v0 = present[0]
    tk = _tensor_kind(v0)
    if tk is not None and all(_tensor_kind(v) == tk for v in present):
        # tensor attributes and conversions without the LieTensor torch-function overhead (~50 us per access)
        with _no_torch_function():
            meta0 = (v0.dtype, tuple(v0.shape))
            if v0.dtype in _NP_OK and all((v.dtype, tuple(v.shape)) == meta0 and not v.requires_grad for v in present):
                arr = np.stack([v.cpu().numpy() if v.device.type != "cpu" else v.numpy() for v in present])
                return {"k": tk[0], "ltype": tk[1], "mask": mask, "a": arr}
    if all(type(v) is bool for v in present):
        return {"k": "py", "t": "bool", "mask": mask, "a": np.array(present, dtype=bool)}
    if all(type(v) is int for v in present):
        a = np.array(present, dtype=np.int64)
        return {"k": "py", "t": "int", "mask": mask, "a": a}
    if all(type(v) is float for v in present):
        return {"k": "py", "t": "float", "mask": mask, "a": np.array(present, dtype=np.float64)}
    if all(type(v) is str for v in present) and len(set(present)) <= 64:
        cats = sorted(set(present))
        idx = {c: i for i, c in enumerate(cats)}
        return {"k": "cat", "cats": cats, "mask": mask, "a": np.array([idx[v] for v in present], dtype=np.uint8)}
    return {"k": "list", "v": vals}


def _decode_column(col, n: int) -> list:
    k = col["k"]
    if k == "none":
        return [None] * n
    if k == "list":
        return col["v"]
    mask = col.get("mask")
    a = col["a"]
    if k in ("tensor", "lie"):
        # one owned tensor per row (no views of a shared buffer: a view would pickle / deep-copy the whole column)
        if getattr(_decode_opts, "packed", False):
            # System.load_map: rows stay numpy arrays (PackedTensor), taken by the compact keyframe / edge fields
            import pypose as pp
            lt = getattr(pp, col["ltype"]) if k == "lie" else None
            vals = [PackedTensor(np.array(x), lt) for x in a]
        elif k == "lie":
            vals = [lie_tensor(np.array(x), col["ltype"]) for x in a]
        else:
            vals = [torch.from_numpy(np.array(x)) for x in a]
    elif k == "py":
        conv = {"bool": bool, "int": int, "float": float}[col["t"]]
        vals = [conv(x) for x in a.tolist()]
    elif k == "cat":
        cats = col["cats"]
        vals = [cats[i] for i in a.tolist()]
    else:
        raise ValueError(f"unknown column kind {k}")
    if mask is None:
        return vals
    out, it = [], iter(vals)
    for m in mask.tolist():
        out.append(next(it) if m else None)
    return out


def encode_records(records: list) -> dict:
    """A list of dicts with the same keys -> columns.  Other lists are kept as they are."""
    if not records or not all(isinstance(r, dict) for r in records):
        return {"__records__": False, "v": records}
    keys = list(records[0].keys())
    if not all(list(r.keys()) == keys for r in records):
        return {"__records__": False, "v": records}
    return {"__records__": True, "n": len(records), "keys": keys,
            "cols": {k: _encode_column([r[k] for r in records]) for k in keys}}


def decode_records(enc: dict) -> list:
    if not enc.get("__records__"):
        return enc["v"]
    n, keys = enc["n"], enc["keys"]
    cols = [_decode_column(enc["cols"][k], n) for k in keys]
    return [dict(zip(keys, vals)) for vals in zip(*cols)] if keys else [{} for _ in range(n)]


def _encode_keys(keys: list):
    """Dict keys -> an int array when they are ints or equal-length tuples of ints (the edge keys), else a list."""
    if keys and all(type(k) is int for k in keys):
        return {"k": "int", "a": np.array(keys, dtype=np.int64)}
    if keys and all(type(k) is tuple and len(k) == len(keys[0]) and all(type(x) is int for x in k) for k in keys):
        return {"k": "tuple", "a": np.array(keys, dtype=np.int64).reshape(len(keys), len(keys[0]))}
    return {"k": "list", "v": keys}


def _decode_keys(enc) -> list:
    if enc["k"] == "int":
        return [int(x) for x in enc["a"].tolist()]
    if enc["k"] == "tuple":
        return [tuple(int(x) for x in row) for row in enc["a"].tolist()]
    return enc["v"]


def encode_dict_of_records(d: dict) -> dict:
    """{key: record dict} (odometry edges) -> columns."""
    return {"__dor__": True, "keys": _encode_keys(list(d.keys())), "recs": encode_records(list(d.values()))}


def decode_dict_of_records(enc: dict) -> dict:
    return dict(zip(_decode_keys(enc["keys"]), decode_records(enc["recs"])))


def encode_dict_of_lists(d: dict, records: bool) -> dict:
    """{key: [items]} (visual edges: lists of record dicts; adjacency / index maps: lists of ints) -> flat columns."""
    keys = list(d.keys())
    counts = np.array([len(v) for v in d.values()], dtype=np.int64)
    flat = [x for v in d.values() for x in v]
    if records:
        items = encode_records(flat)
    elif flat and all(type(x) is int for x in flat):
        items = {"k": "int", "a": np.array(flat, dtype=np.int64)}
    else:
        items = {"k": "list", "v": flat}
    return {"__dol__": True, "keys": _encode_keys(keys), "counts": counts, "records": records, "items": items}


def decode_dict_of_lists(enc: dict) -> dict:
    keys = _decode_keys(enc["keys"])
    if enc["records"]:
        flat = decode_records(enc["items"])
    elif enc["items"]["k"] == "int":
        flat = [int(x) for x in enc["items"]["a"].tolist()]
    else:
        flat = enc["items"]["v"]
    out, i = {}, 0
    for k, c in zip(keys, enc["counts"].tolist()):
        out[k] = flat[i:i + c]
        i += c
    return out


# --------------------------------------------------------------------------------------------------------------------
# map files
# --------------------------------------------------------------------------------------------------------------------

def _encode_images(records: list, directory: Path, cfg, stats: dict, on_written=None):
    """Pull the image fields out of keyframe records into one pack; returns the image index (columns)."""
    # the pack to append to: one this map already references (incremental save), else a new one
    pack = None
    for r in records:
        for f in IMAGE_FIELDS:
            v = r.get(f)
            if is_ref(v) and v.pack.path.parent == directory and v.pack.path.exists():
                pack = v.pack
                break
        if pack is not None:
            break
    if pack is None:
        pack = ImagePack.create(directory)
    quality, level = getattr(cfg, "image_quality", 95), getattr(cfg, "png_level", 3)

    jobs = []           # (row, field, kind of work, payload)
    for i, r in enumerate(records):
        for f in IMAGE_FIELDS:
            v = r.get(f)
            if v is None:
                continue
            if is_ref(v):
                if v.pack.uid == pack.uid:
                    jobs.append((i, f, "keep", v))
                elif v.codec == codec_for(f, v.dtype, cfg):
                    jobs.append((i, f, "copy", v))
                else:
                    jobs.append((i, f, "encode", v.load()))
            elif torch.is_tensor(v):
                jobs.append((i, f, "encode", v))
            else:
                raise TypeError(f"keyframe image field {f}: {type(v)}")

    def work(job):
        i, f, how, v = job
        if how == "keep":
            return None, v.offset, v.length, v.codec, v.shape, v.dtype
        if how == "copy":
            return v.blob(), None, v.length, v.codec, v.shape, v.dtype
        arr = _to_numpy_image(v)
        codec = codec_for(f, v.dtype, cfg)
        blob = encode_array(arr, codec, quality, level, getattr(cfg, "depth_drop_bits", 0))
        return blob, None, len(blob), codec, arr.shape, str(arr.dtype)

    t0 = time.perf_counter()
    workers = max(1, int(getattr(cfg, "encode_workers", 4)))
    # encoded in batches and appended batch by batch: memory stays at one batch of blobs for any map size
    results, offsets_list = [], []
    ex = ThreadPoolExecutor(max_workers=workers) if workers > 1 and len(jobs) > 8 else None
    try:
        for b0 in range(0, len(jobs), 1024):
            batch = jobs[b0:b0 + 1024]
            res = list(ex.map(work, batch, chunksize=16)) if ex is not None else [work(j) for j in batch]
            new_blobs = [r[0] for r in res if r[0] is not None]
            offsets_list += pack.append(new_blobs) if new_blobs else []
            results += [(None if r[0] is None else True,) + tuple(r[1:]) for r in res]       # drop the blob
    finally:
        if ex is not None:
            ex.shutdown()
    offsets = iter(offsets_list)
    n = len(jobs)
    idx = {"row": np.zeros(n, np.int64), "field": np.zeros(n, np.uint8), "offset": np.zeros(n, np.int64),
           "length": np.zeros(n, np.int64), "codec": np.zeros(n, np.uint8), "shape": np.zeros((n, 3), np.int32),
           "dtype": []}
    for j, ((i, f, how, _), (blob, off, length, codec, shape, dtype)) in enumerate(zip(jobs, results)):
        idx["row"][j] = i
        idx["field"][j] = IMAGE_FIELDS.index(f)
        idx["offset"][j] = off if blob is None else next(offsets)
        idx["length"][j] = length
        idx["codec"][j] = _CODEC_IDS[codec]
        sh = tuple(shape) + (1,) * (3 - len(shape))
        idx["shape"][j] = sh[:3]
        idx["dtype"].append(dtype)
        if on_written is not None:
            on_written(i, f, ImageRef(pack, int(idx["offset"][j]), int(length), codec, shape, dtype, "cpu"))
        stats.setdefault(f"{f}_bytes", 0)
        stats[f"{f}_bytes"] += int(length)
        stats.setdefault(f"{f}_count", 0)
        stats[f"{f}_count"] += 1
    stats["encoded"] = sum(1 for j in jobs if j[2] == "encode")
    stats["copied"] = sum(1 for j in jobs if j[2] == "copy")
    stats["kept"] = sum(1 for j in jobs if j[2] == "keep")
    stats["encode_s"] = round(time.perf_counter() - t0, 3)
    dts = sorted(set(idx["dtype"]))
    idx["dtype_names"] = dts
    idx["dtype"] = np.array([dts.index(d) for d in idx["dtype"]], dtype=np.uint8)
    idx["ndim"] = np.array([len(r[4]) for r in results], dtype=np.uint8)
    return pack, idx


def _attach_images(records: list, idx: dict, pack: ImagePack) -> None:
    dts = idx["dtype_names"]
    for j in range(len(idx["row"])):
        shape = tuple(int(s) for s in idx["shape"][j][: int(idx["ndim"][j])])
        ref = ImageRef(pack, int(idx["offset"][j]), int(idx["length"][j]), _CODEC_NAMES[int(idx["codec"][j])],
                       shape, dts[int(idx["dtype"][j])], "cpu")
        records[int(idx["row"][j])][IMAGE_FIELDS[int(idx["field"][j])]] = ref


def _encode_hypo(hypo: dict) -> dict:
    out = dict(hypo)
    out["temp_keyframes"] = encode_records(hypo["temp_keyframes"])
    out["odom_edges"] = encode_dict_of_records(hypo["odom_edges"])
    hd = {}
    for k, h in hypo["hypotheses_data"].items():
        h = dict(h)
        h["visual_edges"] = encode_dict_of_lists(h["visual_edges"], records=True)
        h["visual_adjacency"] = encode_dict_of_lists(h["visual_adjacency"], records=False)
        hd[k] = h
    out["hypotheses_data"] = hd
    return out


# ---------------------------------------------------------------- lazy records (System.load_map of a large map)
# read_map(packed=True) hands the hypothesis manager its records one at a time, built from the columns as they are
# iterated: a list of a million record dicts (with their per-row objects) would be a transient as large as the graph
# itself, and its freed memory stays in the process.  Tensor fields are PackedTensor views of the column arrays.

def _column_getter(col, n: int):
    """Function row -> value of one encoded column (the same values as _decode_column)."""
    k = col["k"]
    if k == "none":
        return lambda i: None
    if k == "list":
        return col["v"].__getitem__
    a = col["a"]
    if k in ("tensor", "lie"):
        import pypose as pp
        lt = getattr(pp, col["ltype"]) if k == "lie" else None
        get = lambda j: PackedTensor(a[j], lt)          # noqa: E731  (a view of the column, no copy)
    elif k == "py":
        conv = {"bool": bool, "int": int, "float": float}[col["t"]]
        vals = [conv(x) for x in a.tolist()]
        get = vals.__getitem__
    elif k == "cat":
        cats, codes = col["cats"], a.tolist()
        get = lambda j: cats[codes[j]]                   # noqa: E731
    else:
        raise ValueError(f"unknown column kind {k}")
    mask = col.get("mask")
    if mask is None:
        return get
    present = mask.tolist()
    pos = (np.cumsum(mask) - 1).tolist()
    return lambda i: get(pos[i]) if present[i] else None


class LazyRecords:
    """A list of records (dicts) decoded from columns on access.  `enc` (the columns) is kept for the bulk restore
    of System.load_map (cross.core.bulk_load)."""

    def __init__(self, enc: dict):
        self.enc = enc
        self.n, self.keys = enc["n"], list(enc["keys"])
        self._get = [_column_getter(enc["cols"][k], self.n) for k in self.keys]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self.n))]
        if i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise IndexError(i)
        return {k: g(i) for k, g in zip(self.keys, self._get)}

    def __iter__(self):
        for i in range(self.n):
            yield {k: g(i) for k, g in zip(self.keys, self._get)}


class KeyframeRecords(LazyRecords):
    """The database keyframe records of a v2 map: LazyRecords plus the stored images (ImageRefs) of each row."""

    def __init__(self, enc: dict, images: dict, pack: "ImagePack"):
        super().__init__(enc)
        self.images = (images, pack)
        self._refs = None

    def _row_refs(self):
        if self._refs is None:
            refs = [{} for _ in range(self.n)]
            _attach_images(refs, self.images[0], self.images[1])
            self._refs = refs
        return self._refs

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self.n))]
        rec = super().__getitem__(i)
        rec.update(self._row_refs()[i if i >= 0 else i + self.n])
        return rec

    def __iter__(self):
        refs = self._row_refs()
        for i, rec in enumerate(super().__iter__()):
            rec.update(refs[i])
            yield rec


def _lazy_records(enc: dict):
    return LazyRecords(enc) if enc.get("__records__") else enc["v"]


def _key_getter(enc):
    if enc["k"] == "int":
        a = enc["a"]
        return len(a), lambda i: int(a[i])
    if enc["k"] == "tuple":
        a = enc["a"]
        return len(a), lambda i: tuple(int(x) for x in a[i])
    v = enc["v"]
    return len(v), v.__getitem__


class _LazyMap:
    """Read-only mapping over encoded keys and per-key values (items decoded while iterating)."""

    def __init__(self, n, key, value):
        self._n, self._key, self._value, self._index = n, key, value, None

    def __len__(self):
        return self._n

    def __iter__(self):
        return (self._key(i) for i in range(self._n))

    def keys(self):
        return list(iter(self))

    def items(self):
        return ((self._key(i), self._value(i)) for i in range(self._n))

    def values(self):
        return (self._value(i) for i in range(self._n))

    def __contains__(self, key):
        return self._lookup().__contains__(key)

    def __getitem__(self, key):
        return self._value(self._lookup()[key])

    def get(self, key, default=None):
        i = self._lookup().get(key)
        return default if i is None else self._value(i)

    def _lookup(self):
        if self._index is None:
            self._index = {self._key(i): i for i in range(self._n)}
        return self._index


def _lazy_dict_of_records(enc: dict):
    n, key = _key_getter(enc["keys"])
    recs = _lazy_records(enc["recs"])
    m = _LazyMap(n, key, recs.__getitem__)
    m.enc = enc                                          # bulk restore (cross.core.bulk_load)
    return m


def _lazy_dict_of_lists(enc: dict):
    n, key = _key_getter(enc["keys"])
    if not enc["records"]:
        if enc["items"]["k"] != "int":
            return decode_dict_of_lists(enc)
        flat = enc["items"]["a"]
        starts = np.concatenate([[0], np.cumsum(enc["counts"])]).tolist()
        m = _LazyMap(n, key, lambda i: flat[starts[i]:starts[i + 1]].tolist())
    else:
        recs = _lazy_records(enc["items"])
        starts = np.concatenate([[0], np.cumsum(enc["counts"])]).tolist()
        m = _LazyMap(n, key, lambda i: recs[starts[i]:starts[i + 1]])
    m.enc = enc                                          # bulk restore (cross.core.bulk_load)
    return m


def _decode_hypo(enc: dict) -> dict:
    if getattr(_decode_opts, "packed", False):
        out = dict(enc)
        out["temp_keyframes"] = _lazy_records(enc["temp_keyframes"])
        out["odom_edges"] = _lazy_dict_of_records(enc["odom_edges"])
        hd = {}
        for k, h in enc["hypotheses_data"].items():
            h = dict(h)
            h["visual_edges"] = _lazy_dict_of_lists(h["visual_edges"])
            h["visual_adjacency"] = _lazy_dict_of_lists(h["visual_adjacency"])
            hd[k] = h
        out["hypotheses_data"] = hd
        return out
    out = dict(enc)
    out["temp_keyframes"] = decode_records(enc["temp_keyframes"])
    out["odom_edges"] = decode_dict_of_records(enc["odom_edges"])
    hd = {}
    for k, h in enc["hypotheses_data"].items():
        h = dict(h)
        h["visual_edges"] = decode_dict_of_lists(h["visual_edges"])
        h["visual_adjacency"] = decode_dict_of_lists(h["visual_adjacency"])
        hd[k] = h
    out["hypotheses_data"] = hd
    return out


def _encode_index_map(d: dict) -> dict:
    """{buffer index: (atlas id, list index)} -> columns."""
    keys = list(d.keys())
    vals = list(d.values())
    if all(type(k) is int for k in keys) and all(type(v) is tuple and len(v) == 2 and all(type(x) is int for x in v) for v in vals):
        return {"__im__": True, "keys": np.array(keys, np.int64), "vals": np.array(vals, np.int64).reshape(-1, 2)}
    return {"__im__": False, "v": d}


def _decode_index_map(enc: dict) -> dict:
    if not enc.get("__im__"):
        return enc["v"]
    return {int(k): (int(a), int(b)) for k, (a, b) in zip(enc["keys"].tolist(), enc["vals"].tolist())}


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_map(save_path, save_data: dict, cfg, on_written=None) -> dict:
    """Write a map in format v2: the graph into `save_path`, images and descriptors into its sidecar directory.
    `on_written(row, field, ImageRef)` is called with the stored reference of every keyframe image (row of
    save_data["db_data"]["keyframes"]).  Returns size / time statistics (stats["pack"]: the map's ImagePack)."""
    t0 = time.perf_counter()
    real = Path(os.path.realpath(str(save_path)))
    real.parent.mkdir(parents=True, exist_ok=True)
    side = sidecar_dir(real)
    side.mkdir(parents=True, exist_ok=True)
    stats = {}

    data = dict(save_data)
    db = dict(save_data["db_data"])
    records = [dict(r) for r in db["keyframes"]]
    pack, img_idx = _encode_images(records, side, cfg, stats, on_written)
    for r in records:
        for f in IMAGE_FIELDS:
            if f in r:
                r[f] = None
    db["keyframes"] = encode_records(records)

    emb = db.get("embeddings")
    desc_file = None
    if torch.is_tensor(emb):
        dt = np.float16 if getattr(cfg, "descriptor_dtype", "float32") == "float16" else None
        arr = emb.detach().cpu().numpy()
        src_dtype = str(arr.dtype)
        if dt is not None:
            arr = arr.astype(dt)
        desc_file = "descriptors.npy"
        tmp = side / f"descriptors.npy.tmp{os.getpid()}.npy"
        np.save(tmp, arr)
        os.replace(tmp, side / desc_file)
        db["embeddings"] = {"__file__": desc_file, "dtype": src_dtype, "shape": tuple(arr.shape)}
        stats["descriptor_bytes"] = int(arr.nbytes)
    # the descriptor projection of a large map (cross/db/index.py: D x d float16, up to 64 MB) goes to the data
    # directory too, so that map.pkl stays a small graph
    di = db.get("descriptor_index")
    if isinstance(di, dict) and isinstance(di.get("projection"), dict) and \
            isinstance(di["projection"].get("components"), np.ndarray):
        proj = dict(di["projection"])
        tmp = side / f"projection.npy.tmp{os.getpid()}.npy"
        np.save(tmp, proj["components"])
        os.replace(tmp, side / "projection.npy")
        stats["projection_bytes"] = int(proj["components"].nbytes)
        proj["components"] = {"__file__": "projection.npy"}
        db["descriptor_index"] = dict(di, projection=proj)
    if "index_to_atlas_idx" in db:
        db["index_to_atlas_idx"] = _encode_index_map(db["index_to_atlas_idx"])
    if "atlas_to_indices" in db:
        db["atlas_to_indices"] = encode_dict_of_lists(db["atlas_to_indices"], records=False)
    data["db_data"] = db
    data["hypo_data"] = _encode_hypo(save_data["hypo_data"])
    data["__format__"] = FORMAT
    data["__version__"] = FORMAT_VERSION
    data["__store__"] = {"pack": pack.path.name, "uid": pack.uid, "images": img_idx, "descriptors": desc_file,
                         "image_codec": getattr(cfg, "image_codec", None)}
    blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_bytes(real, blob)
    stats["graph_bytes"] = len(blob)

    # packs no map references any more (a map overwritten by another one) are removed
    for p in side.glob("images-*.pack"):
        if p.name != pack.path.name:
            p.unlink(missing_ok=True)
    stats["pack_bytes"] = pack.size()
    stats["write_s"] = round(time.perf_counter() - t0, 3)
    stats["pack"] = pack
    return stats


def read_map(load_path, descriptor_mmap: bool = False, packed: bool = False) -> dict:
    """Read a map (format v2 or the old single pickle) into the dict System.load_map restores from.  Images of a v2
    map come back as ImageRefs (decoded on access).  `packed`: tensor fields of the records come back as
    PackedTensor (numpy rows) instead of one tensor object each (System.load_map: millions of fields)."""
    with open(load_path, "rb") as f:
        data = pickle.load(f)
    if data.get("__format__") != FORMAT:
        return data                                                    # old single-file map
    _decode_opts.packed = packed
    try:
        return _read_v2(load_path, data, descriptor_mmap)
    finally:
        _decode_opts.packed = False


def _read_v2(load_path, data: dict, descriptor_mmap: bool) -> dict:
    if int(data.get("__version__", 0)) > FORMAT_VERSION:
        raise ValueError(f"{load_path}: map format {data['__version__']} is newer than this code ({FORMAT_VERSION})")
    side = sidecar_dir(load_path)
    st = data.pop("__store__")
    data.pop("__format__"), data.pop("__version__")
    if not side.is_dir():
        raise FileNotFoundError(f"{load_path}: its data directory {side} is missing (copy map.pkl together with "
                                f"map.pkl{SIDECAR_SUFFIX}/)")
    pack = ImagePack.get(side / st["pack"], st["uid"])
    db = dict(data["db_data"])
    if getattr(_decode_opts, "packed", False) and db["keyframes"].get("__records__"):
        db["keyframes"] = KeyframeRecords(db["keyframes"], st["images"], pack)     # columns: bulk restore
    else:
        records = decode_records(db["keyframes"])
        _attach_images(records, st["images"], pack)
        db["keyframes"] = records
    emb = db.get("embeddings")
    if isinstance(emb, dict) and "__file__" in emb:
        arr = np.load(side / emb["__file__"], mmap_mode="r" if descriptor_mmap else None)
        if str(arr.dtype) != emb["dtype"]:
            arr = arr.astype(emb["dtype"])
        db["embeddings"] = torch.from_numpy(np.ascontiguousarray(arr))
    di = db.get("descriptor_index")
    if isinstance(di, dict) and isinstance(di.get("projection"), dict) and \
            isinstance(di["projection"].get("components"), dict):
        proj = dict(di["projection"])
        proj["components"] = np.load(side / proj["components"]["__file__"])
        db["descriptor_index"] = dict(di, projection=proj)
    if isinstance(db.get("index_to_atlas_idx"), dict) and "__im__" in db["index_to_atlas_idx"]:
        db["index_to_atlas_idx"] = _decode_index_map(db["index_to_atlas_idx"])
    if isinstance(db.get("atlas_to_indices"), dict) and db["atlas_to_indices"].get("__dol__"):
        db["atlas_to_indices"] = decode_dict_of_lists(db["atlas_to_indices"])
    data["db_data"] = db
    data["hypo_data"] = _decode_hypo(data["hypo_data"])
    return data


def materialize(save_data: dict) -> dict:
    """The old single-pickle dict: image references decoded to CPU tensors (StorageConfig.format = pickle)."""
    db = dict(save_data["db_data"])
    recs = []
    for r in db["keyframes"]:
        r = dict(r)
        for f in IMAGE_FIELDS:
            v = r.get(f)
            if is_ref(v):
                r[f] = v.load().cpu()
            elif torch.is_tensor(v):
                r[f] = v.cpu()
        recs.append(r)
    db["keyframes"] = recs
    out = dict(save_data)
    out["db_data"] = db
    return out
