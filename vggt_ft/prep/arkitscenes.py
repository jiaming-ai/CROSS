"""ARKitScenes (pllm-jt/cut3r-data on HF: processed_arkitscenes.tar.gz split into part-aa, -ab, -ac; DUSt3R / CUT3R
preprocessing) -> scene cache.

Input: the parts, read in order as one .tar.gz stream (nothing is extracted in full: each scene is written to a local
scratch folder, converted by a worker and deleted; parts still downloading are waited for).  Members:
dust3r_data/processed_arkitscenes/<split>/<video id>/{new_scene_metadata.npz, scene_metadata.npz,
lowres_depth/<id>_<timestamp>.png, vga_wide/<id>_<timestamp>.jpg}.  new_scene_metadata.npz (CUT3R): `images` (N,) in
time order (10 Hz), `trajectories` (N,4,4) camera-to-world, OpenCV axes, metres, `intrinsics` (N,6) = [w, h, fx, fy,
cx, cy] of the stored image (portrait videos are stored upright, 480 x 640), plus pair lists (unused);
scene_metadata.npz is DUSt3R's shuffled version of the same.  Depth: uint16 millimetres, the iPhone LiDAR depth
upsampled to the image size (z-depth, 0 = invalid).
Conventions checked 2026-10-04 (GT overlap of frame pairs 1 / 3 / 5 apart, rel_tol 0.02): video 43896169 (portrait):
as stored 0.71 / 0.64 / 0.54, the depth read as ray length 0.61 / 0.25 / 0.15, the pose read as world-to-cam
0.11 / 0.04 / 0.01; video 48458612 (landscape): 0.80 / 0.67 / 0.56 vs 0.70 / 0.45 / 0.36 vs 0.37 / 0.07 / 0.03.
Metric: LiDAR depth (mm) and ARKit poses (m) agree at 2 % tolerance.

One scene per video (each video has its own ARKit world frame; several videos of one venue are NOT registered to each
other here, so they are separate worlds), one sequence cut into consecutive chunks of <= --cap frames.  Every frame is
kept by default (10 Hz).  Split `Training` only by default (--splits).

python -m vggt_ft.prep.arkitscenes <raw_dir> <cache_root> [--workers 16] [--sizes q.txt] [--scratch /data0/jz/scratch]
python -m vggt_ft.prep.arkitscenes --dir <extracted scene dir> <cache_root>     (one extracted scene, tests)
"""
from __future__ import annotations

import argparse
import io
import shutil
import tarfile
import time
import traceback
import zlib
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import SceneWriter
from .rawio import dec_any, dec_rgb, load_sizes, ready, split

DATASET = "arkitscenes"
PREFIX = "processed_arkitscenes.tar.gz.part-"


# ---------------------------------------------------------------------------------------------- split-archive stream
class PartStream(io.RawIOBase):
    """The parts <raw>/<prefix>* of a split archive as one stream, in name order.  Each part is opened once it is
    complete (rawio.ready: .done marker, no .chunks, size = the queue's); the part list comes from the download queue
    files (entries of `dataset` whose file name starts with `prefix`, re-read when the listed parts are used up, so
    parts added to a queue later are picked up) or, without queue entries, from the files on disk."""

    def __init__(self, raw, prefix: str, dataset: str, queues=None, poll: int = 120, max_wait: float = 36 * 3600):
        self.raw, self.prefix, self.dataset, self.queues = Path(raw), prefix, dataset, queues or []
        self.poll, self.max_wait = poll, max_wait
        self.i, self.f = 0, None

    def readable(self):
        return True

    def parts(self) -> list[tuple[str, int | None]]:
        sizes = load_sizes(self.queues)
        q = sorted((Path(p).name, n) for (d, p), n in sizes.items() if d == self.dataset
                   and Path(p).name.startswith(self.prefix))
        if q:
            return q
        return [(p.name, None) for p in sorted(self.raw.glob(self.prefix + "*"))
                if not p.name.endswith((".done", ".chunks"))]

    def _open_next(self) -> bool:
        parts = self.parts()
        if self.i >= len(parts):
            return False
        name, size = parts[self.i]
        path = self.raw / name
        t0 = time.time()
        while not ready(path, {(self.dataset, name): size} if size else None, self.dataset):
            if time.time() - t0 > self.max_wait:
                raise TimeoutError(f"{path} not complete after {self.max_wait / 3600:.0f} h")
            print(f"waiting for {name}", flush=True)
            time.sleep(self.poll)
        print(f"reading {name}", flush=True)
        self.f = open(path, "rb", buffering=8 << 20)
        return True

    def readinto(self, b) -> int:
        while True:
            if self.f is None and not self._open_next():
                return 0
            n = self.f.readinto(b)
            if n:
                return n
            self.f.close()
            self.f, self.i = None, self.i + 1


def stream_convert(stream, scene_of, keep, convert_dir, out: str, dataset: str, scratch: Path, workers: int,
                   extra_args=(), max_pending: int | None = None):
    """Stream a .tar.gz whose members of one scene are contiguous.  scene_of(name) -> (scene id, info) or None (member
    not wanted); keep(name) -> path relative to the scene's scratch folder or None.  A finished scene is converted by
    convert_dir(folder, scene, info, out, *extra_args) in a worker, then its folder is deleted; scenes whose scene.json
    exists are skipped.  A truncated stream (archive still incomplete) ends the pass; its last scene is dropped."""
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    max_pending = max_pending or 2 * workers
    pending: deque = deque()
    seen: set = set()
    stats = {"converted": 0, "skipped": 0, "failed": 0}

    def report(block_until: int):
        while pending and (len(pending) > block_until or pending[0][1].ready()):
            sc, res = pending.popleft()
            r = res.get()
            print(sc, r, flush=True)
            stats["converted" if isinstance(r, int) and r > 0 else "failed"] += 1

    cur, cur_dir, skip = None, None, True
    pool = None             # created with the first scene (no idle workers while waiting for a download)

    def submit():
        nonlocal pool
        if pool is None:
            pool = Pool(workers)
        while sum(not r.ready() for _, r in pending) >= max_pending:
            time.sleep(0.5)
        pending.append((cur[0], pool.apply_async(_safe, (convert_dir, str(cur_dir), cur[0], cur[1], out,
                                                         *extra_args))))
        report(4 * max_pending)

    try:
        with tarfile.open(fileobj=stream, mode="r|gz", bufsize=4 << 20) as t:
            for m in t:
                if not m.isfile():
                    continue
                key = scene_of(m.name)
                if key is None:
                    continue
                if cur is None or key[0] != cur[0]:
                    if not skip:
                        submit()
                    done = (Path(out) / dataset / key[0] / "scene.json").exists()
                    if key[0] in seen:
                        print(f"WARNING {key[0]}: members not contiguous, later ones ignored", flush=True)
                    stats["skipped"] += done
                    cur, skip = key, done or key[0] in seen
                    seen.add(key[0])
                    cur_dir = scratch / key[0]
                    if not skip:
                        shutil.rmtree(cur_dir, ignore_errors=True)
                        cur_dir.mkdir(parents=True)
                if skip:
                    continue
                rel = keep(m.name)
                if rel:
                    p = cur_dir / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with open(p, "wb") as f:
                        f.write(t.extractfile(m).read())
        if not skip:
            submit()
    except (tarfile.ReadError, EOFError, zlib.error) as e:
        print(f"stream ended early ({e!r}); last scene {cur and cur[0]} dropped", flush=True)
        if cur_dir is not None and not skip:
            shutil.rmtree(cur_dir, ignore_errors=True)
    report(0)
    if pool is not None:
        pool.close()
        pool.join()
    print(f"== pass done: {stats}", flush=True)


def _safe(fn, d, sc, info, out, *args):
    try:
        return fn(d, sc, info, out, *args)
    except Exception as e:
        return f"ERROR {e!r} {traceback.format_exc(limit=3)}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------- ARKitScenes
def make_scene_of(splits):
    def scene_of(name: str):
        p = name.split("/")
        if "processed_arkitscenes" not in p:
            return None
        i = p.index("processed_arkitscenes")
        if len(p) < i + 4 or p[i + 1] not in splits:
            return None
        return p[i + 2], p[i + 1]              # video id, split
    return scene_of


def keep(name: str):
    p = name.split("/")
    rest = p[p.index("processed_arkitscenes") + 3:]
    if rest in (["new_scene_metadata.npz"], ["scene_metadata.npz"]):
        return rest[0]
    if len(rest) == 2 and rest[0] in ("lowres_depth", "vga_wide"):
        return "/".join(rest)
    return None


def convert_dir(d, vid, split_name, out, cap=600, every=1):
    """One extracted ARKitScenes video folder -> one scene (name = video id)."""
    d = Path(d)
    meta = d / "new_scene_metadata.npz"
    if not meta.exists():
        meta = d / "scene_metadata.npz"
    if not meta.exists():
        return 0 if not any(d.iterdir()) else "ERROR no metadata"
    z = np.load(meta, allow_pickle=True)
    names, traj, intr = [str(x) for x in z["images"]], z["trajectories"], z["intrinsics"]
    order = np.argsort([float(n.rsplit("_", 1)[1].rsplit(".", 1)[0]) for n in names])
    keep = [i for i in order if (d / "vga_wide" / names[i].replace(".png", ".jpg")).exists()
            and (d / "lowres_depth" / names[i]).exists() and np.isfinite(traj[i]).all()][::every]
    if len(keep) < 2:
        return 0
    sw = SceneWriter(out, DATASET, vid, world=vid, metric=True, synthetic=False, dynamic=False, kind="indoor",
                     extra={"split": split_name, "depth": "lidar"})
    for c, idx in enumerate(split(len(keep), cap)):
        q = sw.sequence(vid if c == 0 else f"{vid}_{c}", session=vid)
        for k in idx:
            i = keep[k]
            stem = names[i].rsplit(".", 1)[0]
            rgb = dec_rgb((d / "vga_wide" / f"{stem}.jpg").read_bytes())
            dep = dec_any((d / "lowres_depth" / names[i]).read_bytes()).astype(np.float32) / 1000.0
            w, h, fx, fy, cx, cy = intr[i]
            if rgb.shape[:2] != (int(h), int(w)):
                raise ValueError(f"{stem}: image {rgb.shape[:2]} vs intrinsics {(h, w)}")
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
            q.add(stem.split("_", 1)[1], rgb, dep, K, np.linalg.inv(traj[i]), t=float(stem.split("_", 1)[1]))
    return sw.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?")
    ap.add_argument("out")
    ap.add_argument("--dir", help="convert one extracted scene folder (tests)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--splits", nargs="*", default=["Training"])
    ap.add_argument("--sizes", nargs="*", help="download queue files (part list and expected sizes)")
    ap.add_argument("--scratch", default="/data0/jz/scratch/arkitscenes_stream")
    a = ap.parse_args()
    if a.dir:
        print(convert_dir(a.dir, Path(a.dir).name, Path(a.dir).parent.name, a.out, a.cap, a.every))
        return
    stream = PartStream(a.raw, PREFIX, DATASET, a.sizes)
    stream_convert(io.BufferedReader(stream, 8 << 20), make_scene_of(set(a.splits)), keep, convert_dir, a.out,
                   DATASET, Path(a.scratch), a.workers, extra_args=(a.cap, a.every))


if __name__ == "__main__":
    main()
