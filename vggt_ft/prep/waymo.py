"""Waymo Open Dataset (HarrisonPENG/waymo on HF, CUT3R / DUSt3R-processed) -> scene cache.

Input: <raw>/segment-<id>_with_camera_labels.tfrecord.tar.gz holding <segment>/<FFFFF>_<c>.{jpg,exr,npz} for every
frame F (10 Hz, ~198 per segment) and camera c in 1..5 (FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT; images
downscaled to a long side of 512: 512 x 341 front, 512 x 236 side).  npz: `intrinsics` (3x3 of the stored image),
`cam2world` (4x4, OpenCV axes, metres, = vehicle-to-world x camera-to-vehicle: all cameras of a segment in one world
frame), `distortion` (k1 k2 p1 p2 k3 of the original camera; the images are NOT undistorted).  exr: sparse LiDAR
z-depth in metres (the points Waymo projected into the camera, with its distortion model; 0 = none, ~11-17 % valid;
each LiDAR point is kept in its first camera projection only, so neighbouring cameras share no LiDAR points).

Undistortion (default on): the image is remapped to the pinhole camera with the same K (barrel distortion, so no
border is lost) and every LiDAR pixel is moved to its undistorted position (cv2.undistortPoints, z-buffered); the depth
values (z) are unchanged.  Up to ~6 px (1 %) at the corners of the 512 px images.

Conventions checked 2026-10-04 on segments 10017090168044687777 (5.6 m/s) and 10023947602400723454 (stationary):
* GT overlap (rel_tol 0.02) of FRONT frames 1 / 2 / 3 apart: as stored 0.88 / 0.80 / 0.72, the pose read as
  world-to-cam 0.00 / 0.01 / 0.00 (z vs ray length is barely testable with sparse depth, 0.87 / 0.79 / 0.70 as ray;
  the depth is z by construction in the CUT3R / DUSt3R script).
* Photometric alignment (LiDAR edge pixels of camera A moved into camera B at the same frame, best extra shift of the
  B samples): stationary segment, undistorted: FRONT->FRONT_LEFT / FRONT->FRONT_RIGHT / FRONT_LEFT->SIDE_LEFT best at
  du 0.0 / +0.5 / 0.0 px, error at zero shift 8.9 / 9.9 / 9.5 grey levels; as stored (distorted): du -1.0 / +1.5 /
  -1.5, error 10.8 / 13.3 / 11.9 -> undistortion is right.  Consecutive frames of one camera align at 0 in both.
* Moving segment: residual cross-camera shifts of 1.5-4 px remain (same camera: 0): the poses of all cameras are the
  vehicle pose at the frame time, not at each camera's capture time (and rolling shutter); a LiDAR-sweep trigger
  model (tau_c = tau0 +- azimuth / 360 x 100 ms) did not reduce the error, so it is left uncorrected.  Windows mixing
  cameras of a moving vehicle are therefore a few px inconsistent; windows within one camera are not affected.

One scene per segment; its five cameras are sequences (one rig, one world frame; session = camera).  Every frame is
kept (10 Hz).  Real, metric, dynamic.

python -m vggt_ft.prep.waymo <raw_dir> <cache_root> [--workers 16] [--every 1] [--sizes q.txt] [--no-undistort]
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse  # noqa: E402
import traceback  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .common import SceneWriter  # noqa: E402
from .rawio import dec_any, dec_npz, dec_rgb, iter_targz, load_sizes, ready  # noqa: E402

DATASET = "waymo"
CAMS = {"1": "FRONT", "2": "FRONT_LEFT", "3": "FRONT_RIGHT", "4": "SIDE_LEFT", "5": "SIDE_RIGHT"}


def scene_name(path) -> str:
    return Path(path).name.replace("_with_camera_labels.tfrecord.tar.gz", "").replace(".tar.gz", "")


class Undistort:
    """Pinhole (same K) version of an image and of its sparse depth, for one camera model."""

    def __init__(self, K, dist, W, H):
        self.K, self.dist = np.asarray(K, np.float64), np.asarray(dist, np.float64)
        self.W, self.H = W, H
        self.mx, self.my = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, (W, H), cv2.CV_32FC1)

    def same(self, K, dist, W, H) -> bool:
        return (W, H) == (self.W, self.H) and np.allclose(K, self.K) and np.allclose(dist, self.dist)

    def rgb(self, img):
        return cv2.remap(img, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def depth(self, d):
        y, x = np.nonzero(d > 0)
        out = np.full((self.H, self.W), np.inf, np.float32)
        if len(x):
            p = cv2.undistortPoints(np.stack([x, y], -1).astype(np.float64)[:, None], self.K, self.dist, P=self.K)[:, 0]
            u, v = np.round(p[:, 0]).astype(np.int64), np.round(p[:, 1]).astype(np.int64)
            ok = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
            np.minimum.at(out, (v[ok], u[ok]), d[y[ok], x[ok]])
        out[~np.isfinite(out)] = 0
        return out


def convert(job):
    path, out, every, undistort = job
    scene = scene_name(path)
    if (Path(out) / DATASET / scene / "scene.json").exists():
        return scene, -1
    frames: dict = {}
    for name, b in iter_targz(path):
        stem, ext = Path(name).name.rsplit(".", 1)
        if ext in ("jpg", "exr", "npz") and "_" in stem:
            frames.setdefault(stem, {})[ext] = b
    by_cam: dict = {}
    for stem, f in frames.items():
        fi, c = stem.split("_")
        if len(f) == 3 and c in CAMS:
            by_cam.setdefault(c, []).append((int(fi), stem))
    if not by_cam:
        return scene, 0
    sw = SceneWriter(out, DATASET, scene, world=scene, metric=True, synthetic=False, dynamic=True, kind="driving",
                     extra={"depth": "lidar", "undistorted": bool(undistort)})
    for c in sorted(by_cam):
        q = sw.sequence(CAMS[c], session=CAMS[c])
        und = None
        for fi, stem in sorted(by_cam[c])[::every]:
            f = frames[stem]
            m = dec_npz(f["npz"])
            T = np.asarray(m["cam2world"], np.float64)
            if not np.isfinite(T).all():
                continue
            rgb, d = dec_rgb(f["jpg"]), dec_any(f["exr"]).astype(np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            K = np.asarray(m["intrinsics"], np.float64)
            if undistort:
                H, W = rgb.shape[:2]
                if d.shape != (H, W):
                    raise ValueError(f"{stem}: depth {d.shape} vs image {(H, W)}")
                if und is None or not und.same(K, m["distortion"], W, H):
                    und = Undistort(K, m["distortion"], W, H)
                rgb, d = und.rgb(rgb), und.depth(d)
            q.add(f"{fi:05d}", rgb, d, K, np.linalg.inv(T), t=fi / 10.0)
    return scene, sw.close()


def safe(job):
    try:
        return convert(job)
    except Exception as e:
        return scene_name(job[0]), f"ERROR {e!r} {traceback.format_exc(limit=3)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--no-undistort", dest="undistort", action="store_false")
    ap.add_argument("--sizes", nargs="*", help="download queue files with the expected file sizes")
    ap.add_argument("--scenes", nargs="*", help="segment ids (substrings) to convert")
    a = ap.parse_args()
    sizes = load_sizes(a.sizes)
    paths = sorted(p for p in Path(a.raw).glob("segment-*.tar.gz") if ready(p, sizes, DATASET))
    if a.scenes:
        paths = [p for p in paths if any(s in p.name for s in a.scenes)]
    paths = [p for p in paths if not (Path(a.out) / DATASET / scene_name(p) / "scene.json").exists()]
    print(f"{len(paths)} archives to convert", flush=True)
    if not paths:
        return
    with Pool(min(a.workers, len(paths))) as p:
        for scene, n in p.imap_unordered(safe, [(str(x), a.out, a.every, a.undistort) for x in paths]):
            print(scene, n, flush=True)


if __name__ == "__main__":
    main()
