#!/usr/bin/env python3
"""Convert a TUM RGB-D sequence to the posed RGB-D folder layout of cross/dataloader/posed_rgbd.py.

  python scripts/datasets/convert_tum.py <tum_sequence_dir> <out_dir> [--stride 1] [--max-dt 0.02]

RGB, depth and ground truth are associated by nearest timestamp (|dt| <= max_dt for depth and ground truth);
frames without a match are dropped.  Depth is written as uint16 millimetre PNGs (TUM stores depth * 5000), images are
symlinked.  The ground-truth trajectory (motion capture, camera optical centre in the room frame) is written as
camera-to-world matrices; sequences recorded in the same room share that frame, so one can serve as the map and another
as the query (e.g. freiburg2_desk -> freiburg2_desk_with_person).
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

INTRINSICS = {  # fx, fy, cx, cy (TUM RGB-D ROS default calibration per sensor)
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}


def read_list(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        rows.append((float(parts[0]), parts[1:]))
    return rows


def nearest(ts: np.ndarray, t: float):
    i = int(np.searchsorted(ts, t))
    cands = [j for j in (i - 1, i) if 0 <= j < len(ts)]
    j = min(cands, key=lambda j: abs(ts[j] - t))
    return j, abs(ts[j] - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-dt", type=float, default=0.02)
    a = ap.parse_args()
    seq, out = Path(a.seq), Path(a.out)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    rgb = read_list(seq / "rgb.txt")
    dep = read_list(seq / "depth.txt")
    gt = read_list(seq / "groundtruth.txt")
    dts = np.array([t for t, _ in dep])
    gts = np.array([t for t, _ in gt])
    poses, k = [], 0
    for i, (t, (fn,)) in enumerate(rgb):
        if i % a.stride:
            continue
        jd, dd = nearest(dts, t)
        jg, dg = nearest(gts, t)
        if dd > a.max_dt or dg > a.max_dt:
            continue
        v = np.array([float(x) for x in gt[jg][1]])        # tx ty tz qx qy qz qw
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(v[3:7]).as_matrix()
        T[:3, 3] = v[:3]
        poses.append(T.reshape(-1))
        dst = out / "rgb" / f"{k:06d}.png"
        if not dst.exists():
            os.symlink(os.path.abspath(seq / fn), dst)
        d = cv2.imread(str(seq / dep[jd][1][0]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5.0   # -> millimetres
        cv2.imwrite(str(out / "depth" / f"{k:06d}.png"), np.clip(d, 0, 65535).astype(np.uint16))
        k += 1
    np.savetxt(out / "poses_left.txt", np.asarray(poses), fmt="%.9f")
    sensor = next(s for s in INTRINSICS if s in seq.name)
    fx, fy, cx, cy = INTRINSICS[sensor]
    img = cv2.imread(str(out / "rgb" / "000000.png"))
    calib = {"K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "width": img.shape[1], "height": img.shape[0],
             "fps": 30.0 / a.stride, "dataset": "tum_rgbd", "source": str(seq)}
    (out / "calib.json").write_text(json.dumps(calib, indent=1))
    print(f"{seq.name}: {k} frames -> {out}")


if __name__ == "__main__":
    main()
