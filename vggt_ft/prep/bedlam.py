"""BEDLAM / BEDLAM2 (Intelligent-Systems/BEDLAM, BEDLAM-depth, BEDLAM2, BEDLAM2-depth on HF) -> scene cache.

Input: <raw>/<batch>/ for each rendered batch (one Unreal level, 250 sequences of 117-477 frames at 30 fps, 1280 x 720
RGBA PNG; people moving, camera orbiting / dollying / zooming):
* png/<batch>_png.<k>.tar: <batch>/png/seq_XXXXXX/seq_XXXXXX_FFFF.png (shards k = 0, 1, ...).
* BEDLAM:  depth/<batch>_depth.<k>.tar: <batch>/depth/seq_XXXXXX/seq_XXXXXX_FFFF_depth.exr (one float32 channel
  "Depth", centimetres).  ground_truth/<batch>_gt.tar.gz: <batch>/ground_truth/camera/seq_XXXXXX_camera.csv.
* BEDLAM2: exr_depth/<batch>_exr_depth.<k>.tar: <batch>/exr_depth/seq_XXXXXX/seq_XXXXXX_FFFF.exr (multi-layer EXR:
  beauty RGBA, ActorHitProxyMask*, FinalImageMovieRenderQueue_WorldDepth.{R,G,B} half floats, centimetres).
  ground_truth/<batch>_gt_centersubframe_exr_meta_csv.tar.gz: <batch>/ground_truth/meta_exr_csv/seq_XXXXXX_camera.csv
  (the centre sub-frame of the 7 motion-blur sub-frames; the per-frame JSONs of ..._exr_meta.tar.gz hold the same
  values and are the fallback).
Camera CSV (both): name, x, y, z, yaw, pitch, roll, focal_length, sensor_width, sensor_height, hfov per frame: the
camera in the Unreal world (left-handed: x forward, y right, z up; centimetres), FRotator angles in degrees, focal length
and sensor in mm (36 x 20.25, the 16:9 of the image; BEDLAM2 zooms, so K changes per frame).

Conventions (see `c2w_unreal`, `read_depth`):
* Rotation: Unreal's FRotationMatrix (roll about x, then pitch about y, then yaw about z; rows = the camera's forward,
  right, up axes in the world).  OpenCV camera axes: x = right, y = -up, z = forward.  The world is Unreal's with y
  negated (right-handed), in metres.
* Intrinsics: fx = fy = focal_length / sensor_width x W, principal point at the image centre ((W - 1) / 2 with pixel
  centres at integer coordinates).
* Depth: BEDLAM "Depth" and BEDLAM2 WorldDepth.R are planar z-depth (Unreal SceneDepth); BEDLAM2 WorldDepth.G is the
  ray length and WorldDepth.B the world height (z) of the surface.  Sky / far background: BEDLAM 1e6-1e8 cm (beyond
  the codec's 1000 m: invalid), BEDLAM2 saturates at 65504 cm (half max; stored invalid).
Checked 2026-10-04 (3 sequences per batch, all 6 batches):
* GT overlap (rel_tol 0.02) of frame pairs 1 / 2 / 3 apart (after --every 2): bigOffice as stored 0.92 / 0.88 / 0.83,
  depth read as ray length 0.92 / 0.86 / 0.79, pose inverted 0.51 / 0.31 / 0.21, roll negated 0.92 / 0.87 / 0.80,
  pitch negated 0.88 / 0.78 / 0.70, yaw negated 0.60 / 0.50 / 0.41; stadium 0.90 / 0.88 / 0.86 vs ray 0.89 / 0.87 /
  0.75, inverted 0.46 / 0.24 / 0.17, yaw negated 0.71 / 0.56 / 0.43; rome (BEDLAM2) 0.88 / 0.85 / 0.79 vs ray
  0.88 / 0.83 / 0.75, inverted 0.47 / 0.40 / 0.31, pitch negated 0.86 / 0.80 / 0.74.
* BEDLAM2: WorldDepth.G / R / sqrt(1 + x^2 + y^2) = 1.0000 (p1-p99 0.9993-1.0007), so R is z and G the ray length.
  World height of the R-unprojected pixels under the pose above minus WorldDepth.B: median |error| 0.06-0.08 cm
  (busstation, rome; roll negated 0.6-7.7 cm, pitch negated 1-3 m); chemicalplant a constant -3 to -7 cm along each
  sequence (B is half float: at the level's z ~ -125 m its step is 8 cm).
* Time alignment (PNG and depth are separate render jobs): PNG j warped into frame i (4 raw frames apart) with depth i
  and the poses, median abs grey error on textured pixels, PNG pair shifted by s = -2..2: best at s = 0 in every batch
  (bigOffice 3.1 / 3.0 / 2.3 / 2.7 / 3.0, rome 12.3 / 8.7 / 5.2 / 9.6 / 13.8); the pose of j shifted by +-1 frame:
  8.8-22 -> PNG, depth and camera CSV are frame-aligned.

One scene per sequence (people differ between sequences of a batch, so sequences of one level are separate worlds),
`metric`, `synthetic`, `dynamic`.  Every --every-th frame (default 2: 15 Hz), at most --cap frames per sequence (longer
ones are cut into consecutive chunks, sessions of one recording).  Only sequences whose every frame is present in both
the PNG and the depth shards on disk are converted (the others wait for more shards); sequences whose camera centre
moves less than --min-move metres are skipped (no parallax).

python -m vggt_ft.prep.bedlam <raw_dir> <cache_root> [--workers 16] [--every 2] [--cap 300] [--sizes q.txt ...]
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import traceback  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .common import SceneWriter  # noqa: E402
from .rawio import dec_rgb, iter_targz, load_sizes, ready, split, tar_index  # noqa: E402

DATASET = "bedlam"
FPS = 30.0
CSV_KEYS = ("x", "y", "z", "yaw", "pitch", "roll", "focal_length", "sensor_width", "sensor_height")
INDOOR = ("office", "archviz", "room", "kitchen", "house", "apartment", "studio", "gym")


# ------------------------------------------------------------------------------------------------------ conventions
def rot_unreal(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Unreal FRotationMatrix (degrees): rows = the rotated x (forward), y (right), z (up) axes in the world."""
    p, y, r = np.radians([pitch, yaw, roll])
    sp, cp, sy, cy, sr, cr = np.sin(p), np.cos(p), np.sin(y), np.cos(y), np.sin(r), np.cos(r)
    return np.array([[cp * cy, cp * sy, sp],
                     [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
                     [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp]])


FLIP_Y = np.diag([1.0, -1.0, 1.0])


def c2w_unreal(x, y, z, yaw, pitch, roll) -> np.ndarray:
    """4x4 camera-to-world, OpenCV camera axes, world = Unreal with y negated, metres."""
    M = rot_unreal(yaw, pitch, roll)
    T = np.eye(4)
    T[:3, :3] = FLIP_Y @ np.stack([M[1], -M[2], M[0]], 1)
    T[:3, 3] = FLIP_Y @ (np.array([x, y, z], np.float64) / 100.0)
    return T


def intrinsics(focal_mm: float, sensor_w_mm: float, W: int, H: int) -> np.ndarray:
    f = focal_mm / sensor_w_mm * W
    return np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]], np.float64)


def exr_channels(b: bytes, names) -> dict:
    """float32 arrays of the named channels of an EXR (bytes)."""
    import Imath
    import OpenEXR
    f = OpenEXR.InputFile(io.BytesIO(b))
    dw = f.header()["dataWindow"]
    W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    return {c: np.frombuffer(f.channel(c, pt), np.float32).reshape(H, W) for c in names}


def read_depth(b: bytes, v2: bool) -> np.ndarray:
    """z-depth in metres (0 = invalid) of a BEDLAM (v2 False) or BEDLAM2 depth EXR."""
    if v2:
        d = exr_channels(b, ["FinalImageMovieRenderQueue_WorldDepth.R"])["FinalImageMovieRenderQueue_WorldDepth.R"]
        d = np.where(d >= 65000, 0, d)                 # half-float saturation (sky, > 650 m)
    else:
        d = exr_channels(b, ["Depth"])["Depth"]
    d = d.astype(np.float32) / 100.0
    d[~np.isfinite(d) | (d <= 0)] = 0
    return d


# ------------------------------------------------------------------------------------------------------- raw layout
def read_cams(batch_dir: Path, sizes: dict | None) -> tuple[dict, bool] | None:
    """{seq: {frame: (x, y, z, yaw, pitch, roll, focal, sensor_w, sensor_h)}}, is_bedlam2; None if no GT archive."""
    b = batch_dir.name
    gt = batch_dir / "ground_truth"
    cands = [(gt / f"{b}_gt.tar.gz", False), (gt / f"{b}_gt_centersubframe_exr_meta_csv.tar.gz", True),
             (gt / f"{b}_gt_centersubframe_exr_meta.tar.gz", True)]
    for path, v2 in cands:
        if not _complete(path, sizes, f"{b}/ground_truth/{path.name}"):
            continue
        cams: dict = {}
        for name, data in iter_targz(path):
            p = name.split("/")
            if len(p) >= 2 and p[-2] in ("camera", "meta_exr_csv") and p[-1].endswith("_camera.csv"):
                seq = p[-1][:-len("_camera.csv")]
                for r in csv.DictReader(io.StringIO(data.decode())):
                    fr = int(r["name"].rsplit(".", 1)[0].rsplit("_", 1)[1])
                    cams.setdefault(seq, {})[fr] = tuple(float(r[k]) for k in CSV_KEYS)
            elif len(p) >= 2 and p[-1].endswith("_meta.json") and "meta_exr" in p:
                m = json.loads(data)
                seq, fr = re.match(r"(seq_\d+)_(\d+)_meta\.json", p[-1]).groups()
                g = lambda k: float(m[f"unreal/camera/{k}"])  # noqa: E731
                cams.setdefault(seq, {})[int(fr)] = (
                    g("curPos/x"), g("curPos/y"), g("curPos/z"), g("curRot/yaw"), g("curRot/pitch"),
                    g("curRot/roll"), g("FinalImage/focalLength"), g("FinalImage/sensorWidth"),
                    g("FinalImage/sensorHeight"))
        if cams:
            return cams, v2
    return None


def _complete(path: Path, sizes: dict | None, rel: str) -> bool:
    """rawio.ready for small files: the .done marker, no .chunks, and the queue's size when it is known."""
    if not Path(str(path) + ".done").exists() or not path.exists() or Path(str(path) + ".chunks").exists():
        return False
    exp = (sizes or {}).get((DATASET, rel))
    return path.stat().st_size == exp if exp else True


_FRAME = re.compile(r"(seq_\d+)/(seq_\d+)_(\d+)(?:_depth)?\.(png|exr)$")


def index_shards(batch_dir: Path, sub: str, sizes: dict | None) -> tuple[dict, list[str]]:
    """{(seq, frame): (tar path, offset, size)} over the complete shards <batch>/<sub>/*.tar; missing / incomplete
    shard names."""
    out, waiting = {}, []
    d = batch_dir / sub
    if not d.is_dir():
        return out, waiting
    for t in sorted(d.glob("*.tar")):
        if not ready(t, sizes, DATASET, f"{batch_dir.name}/{sub}/{t.name}"):
            waiting.append(t.name)
            continue
        for name, (off, n) in tar_index(t).items():
            m = _FRAME.search(name)
            if m and m.group(1) == m.group(2):
                out[(m.group(1), int(m.group(3)))] = (str(t), off, n)
    return out, waiting


def scene_name(batch: str, seq: str) -> str:
    return f"{batch}-{seq}"


def plan(raw: Path, out: Path, sizes: dict | None, min_move: float, every: int):
    """Jobs (one per sequence to convert) and a per-batch summary."""
    jobs, summary = [], {}
    for bd in sorted(p for p in Path(raw).iterdir() if p.is_dir()):
        r = read_cams(bd, sizes)
        if r is None:
            summary[bd.name] = "no ground truth yet"
            continue
        cams, v2 = r
        png, wp = index_shards(bd, "png", sizes)
        dep, wd = index_shards(bd, "exr_depth" if v2 else "depth", sizes)
        st = dict(v2=v2, seqs=len(cams), done=0, todo=0, incomplete=0, static=0, shards_waiting=wp + wd)
        for seq in sorted(cams):
            frames = sorted(cams[seq])
            sc = scene_name(bd.name, seq)
            if (Path(out) / DATASET / sc / "scene.json").exists():
                st["done"] += 1
                continue
            if not all((seq, f) in png and (seq, f) in dep for f in frames):
                st["incomplete"] += 1
                continue
            keep = frames[::every]
            c = np.array([cams[seq][f][:3] for f in keep]) / 100.0
            if np.linalg.norm(c.max(0) - c.min(0)) < min_move:
                st["static"] += 1
                continue
            st["todo"] += 1
            jobs.append(dict(batch=bd.name, seq=seq, v2=v2, frames=[(f, cams[seq][f], png[(seq, f)], dep[(seq, f)])
                                                                     for f in keep]))
        summary[bd.name] = st
    return jobs, summary


# -------------------------------------------------------------------------------------------------------- conversion
def _read(loc) -> bytes:
    path, off, n = loc
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(n)


def convert(job, out: str, cap: int) -> int:
    batch, seq, v2 = job["batch"], job["seq"], job["v2"]
    kind = "indoor" if any(k in batch.lower() for k in INDOOR) else "outdoor"
    sc = scene_name(batch, seq)
    sw = SceneWriter(out, DATASET, sc, world=sc, metric=True, synthetic=True, dynamic=True, kind=kind,
                     extra={"source": "bedlam2" if v2 else "bedlam", "batch": batch, "sequence": seq,
                            "depth": "render"})
    fr = job["frames"]
    for c, idx in enumerate(split(len(fr), cap)):
        q = sw.sequence(seq if c == 0 else f"{seq}_{c}", session=seq)
        for i in idx:
            f, cam, lp, ld = fr[i]
            rgb = dec_rgb(_read(lp))
            H, W = rgb.shape[:2]
            d = read_depth(_read(ld), v2)
            if d.shape != (H, W):
                raise ValueError(f"{sc} {f}: depth {d.shape} vs image {(H, W)}")
            x, y, z, yaw, pitch, roll, fl, sw_mm, sh_mm = cam
            if abs(sw_mm / sh_mm - W / H) > 0.01:
                raise ValueError(f"{sc} {f}: sensor {sw_mm} x {sh_mm} vs image {W} x {H}")
            q.add(f"{f:04d}", rgb, d, intrinsics(fl, sw_mm, W, H), np.linalg.inv(c2w_unreal(x, y, z, yaw, pitch, roll)),
                  t=f / FPS)
    return sw.close()


def _safe(args):
    job, out, cap = args
    sc = scene_name(job["batch"], job["seq"])
    try:
        return sc, convert(job, out, cap)
    except Exception as e:
        return sc, f"ERROR {e!r} {traceback.format_exc(limit=3)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--min-move", type=float, default=0.1, help="skip sequences whose camera centres span less (m)")
    ap.add_argument("--sizes", nargs="*", help="download queue files with the expected file sizes")
    ap.add_argument("--batches", nargs="*", help="batch names (substrings) to convert")
    ap.add_argument("--seqs", nargs="*", help="sequence ids to convert (tests)")
    ap.add_argument("--plan", action="store_true", help="print the plan only")
    a = ap.parse_args()
    sizes = load_sizes(a.sizes)
    jobs, summary = plan(Path(a.raw), Path(a.out), sizes, a.min_move, a.every)
    if a.batches:
        jobs = [j for j in jobs if any(s in j["batch"] for s in a.batches)]
    if a.seqs:
        jobs = [j for j in jobs if j["seq"] in a.seqs]
    for b, st in summary.items():
        print(b, json.dumps(st), flush=True)
    print(f"{len(jobs)} sequences to convert ({sum(len(j['frames']) for j in jobs)} frames)", flush=True)
    if not jobs or a.plan:
        return
    with Pool(min(a.workers, len(jobs))) as p:
        for sc, n in p.imap_unordered(_safe, [(j, a.out, a.cap) for j in jobs]):
            print(sc, n, flush=True)


if __name__ == "__main__":
    main()
