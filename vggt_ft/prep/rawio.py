"""Helpers of the converters that read downloaded archives (blendedmvs, co3d, uco3d, wildrgbd, spring, unreal4k).

* `ready(path, sizes)`: a raw file is complete when its `<file>.done` marker exists, no `<file>.chunks` (download in
  progress) does, and its size is the expected one (from the download queue files "<kind> <repo> <path> <dataset>
  <bytes>", see `load_sizes`), or > 1 MB without them.
* `tar_index` / `read_at`: random access to the members of an uncompressed tar (one pass over the headers).
* `iter_targz`: the regular files of a .tar.gz, streamed.
* `nn_tour`: an order of an unordered image collection in which neighbouring indices overlap (greedy nearest-neighbour
  tour over camera centre + viewing direction), `split` cuts a long sequence into consecutive chunks.
"""
from __future__ import annotations

import io
import json
import os
import struct
import tarfile
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402


def load_sizes(files) -> dict:
    """{(dataset, path_in_repo): bytes} from download queue files."""
    out = {}
    for f in files or []:
        for line in open(f):
            p = line.split()
            if len(p) >= 5 and p[4].isdigit():
                out[(p[3], p[2])] = int(p[4])
    return out


def ready(path, sizes: dict | None = None, dataset: str | None = None, rel: str | None = None) -> bool:
    path = Path(path)
    if not Path(str(path) + ".done").exists() or not path.exists() or Path(str(path) + ".chunks").exists():
        return False       # (<file>.chunks: a download in progress; it pre-sizes the file to its full length)
    n = path.stat().st_size
    exp = (sizes or {}).get((dataset, rel if rel is not None else path.name))
    return n == exp if exp else n > (1 << 20)


def tar_index(path) -> dict:
    """{member name: (offset of the data, size)} of the regular files of an uncompressed tar."""
    with tarfile.open(path, "r:") as t:
        return {m.name: (m.offset_data, m.size) for m in t if m.isfile()}


class Window(io.RawIOBase):
    """Read-only view of bytes [off, off + size) of an open binary file (a tar nested in a tar)."""

    def __init__(self, f, off: int, size: int):
        self.f, self.off, self.size, self.pos = f, off, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, pos, whence=0):
        self.pos = {0: pos, 1: self.pos + pos, 2: self.size + pos}[whence]
        return self.pos

    def readinto(self, b):
        n = max(0, min(len(b), self.size - self.pos))
        if n == 0:
            return 0
        self.f.seek(self.off + self.pos)
        data = self.f.read(n)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def read_at(f, off: int, size: int) -> bytes:
    f.seek(off)
    return f.read(size)


def iter_targz(path):
    """(name, bytes) of the regular files of a .tar.gz, in archive order."""
    with tarfile.open(path, "r|gz") as t:
        for m in t:
            if m.isfile():
                yield m.name, t.extractfile(m).read()


def dec_rgb(b: bytes) -> np.ndarray:
    a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if a is None:
        raise ValueError("cannot decode image")
    return a[..., ::-1]


def dec_any(b: bytes) -> np.ndarray:
    """PNG / EXR as stored (uint16, float32 ...)."""
    a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError("cannot decode image")
    return a


def dec_npy(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


def dec_npz(b: bytes) -> dict:
    z = np.load(io.BytesIO(b), allow_pickle=False)
    return {k: z[k] for k in z.files}


_ST = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32, "U8": np.uint8}


def dec_safetensors(b: bytes) -> dict:
    """Tensors of a .safetensors file as numpy arrays (header: u64 length + JSON)."""
    n = struct.unpack("<Q", b[:8])[0]
    head = json.loads(b[8:8 + n])
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        o0, o1 = v["data_offsets"]
        out[k] = np.frombuffer(b[8 + n + o0:8 + n + o1], _ST[v["dtype"]]).reshape(v["shape"])
    return out


def centres_dirs(E: np.ndarray):
    """Camera centres and viewing directions (world frame) of camera-from-world poses (N,4,4)."""
    R, t = E[:, :3, :3], E[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t), R[:, 2, :]


def nn_tour(E: np.ndarray, scale: float) -> np.ndarray:
    """Greedy nearest-neighbour tour; cost between two views = centre distance / scale + angle between the viewing
    directions (radians).  `scale` ~ the median scene depth, so 0.5 x depth of baseline costs as much as ~29 degrees.
    Starts at the view farthest (in that cost) from the others' mean so that the tour sweeps from one end."""
    c, v = centres_dirs(E)
    n = len(c)
    if n <= 2:
        return np.arange(n)
    D = np.linalg.norm(c[:, None] - c[None], axis=-1) / max(scale, 1e-9)
    D += np.arccos(np.clip(v @ v.T, -1, 1))
    order = [int(np.argmax(D.mean(1)))]
    left = np.ones(n, bool)
    left[order[0]] = False
    for _ in range(n - 1):
        d = np.where(left, D[order[-1]], np.inf)
        j = int(np.argmin(d))
        order.append(j)
        left[j] = False
    return np.array(order)


def split(n: int, cap: int) -> list[np.ndarray]:
    """Indices 0..n-1 cut into ceil(n / cap) consecutive chunks of (nearly) equal size."""
    k = max(1, -(-n // cap))
    return [a for a in np.array_split(np.arange(n), k) if len(a)]
