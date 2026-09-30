#!/usr/bin/env python3
"""Convert an OpenLORIS-Scene sequence (package format) to the posed RGB-D folder layout of
cross/dataloader/posed_rgbd.py, keeping the robot's real wheel odometry.

  python scripts/datasets/convert_openloris.py <openloris_sequence_dir> <out_dir> [--stride 3] [--max-dt 0.05]

Output: rgb/ (symlinks to the D435i colour images), depth/ (symlinks to the aligned depth, uint16 millimetres),
poses_left.txt (ground-truth camera-to-world, 16 values per row: ground-truth base pose composed with the base -> colour
camera extrinsic of trans_matrix.yaml), odom_left.txt (the same for the wheel-odometry base pose; the loader takes its
consecutive differences as the odometry of the run, so no simulated noise is needed) and calib.json.
Colour frames are associated with the nearest aligned depth frame (|dt| <= max_dt) and with the ground truth and
odometry interpolated at the colour timestamp (frames outside their time ranges are dropped).  Sequences of one scene
share the ground-truth frame (motion capture in office, LiDAR SLAM elsewhere), so one sequence can be the map and the
others the queries.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def read_stamped(path: Path, ncols: int):
    ts, vals = [], []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        ts.append(float(p[0]))
        vals.append(p[1:1 + ncols])
    return np.asarray(ts), vals


def interp_poses(ts, poses7, query):
    """poses7: (N, 7) tx ty tz qx qy qz qw; returns (M, 4, 4) at query times (NaN outside the range)."""
    order = np.argsort(ts)
    ts, poses7 = ts[order], poses7[order]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    ts, poses7 = ts[keep], poses7[keep]
    slerp = Slerp(ts, Rotation.from_quat(poses7[:, 3:7]))
    out = np.full((len(query), 4, 4), np.nan)
    ok = (query >= ts[0]) & (query <= ts[-1])
    if ok.any():
        q = query[ok]
        R = slerp(q).as_matrix()
        t = np.stack([np.interp(q, ts, poses7[:, k]) for k in range(3)], 1)
        T = np.tile(np.eye(4), (len(q), 1, 1))
        T[:, :3, :3] = R
        T[:, :3, 3] = t
        out[ok] = T
    return out


def read_extrinsic(seq: Path, child: str) -> np.ndarray:
    fs = cv2.FileStorage(str(seq / "trans_matrix.yaml"), cv2.FILE_STORAGE_READ)
    node = fs.getNode("trans_matrix")
    for i in range(node.size()):
        n = node.at(i)
        if n.getNode("parent_frame").string() == "base_link" and n.getNode("child_frame").string() == child:
            return np.asarray(n.getNode("matrix").mat(), dtype=np.float64)
    raise KeyError(f"no base_link -> {child} in {seq}/trans_matrix.yaml")


def read_intrinsics(seq: Path, sensor: str):
    fs = cv2.FileStorage(str(seq / "sensors.yaml"), cv2.FILE_STORAGE_READ)
    n = fs.getNode(sensor)
    fx, cx, fy, cy = [float(v) for v in n.getNode("intrinsics").mat().reshape(-1)]
    return fx, fy, cx, cy, int(n.getNode("width").real()), int(n.getNode("height").real())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=3, help="keep every n-th colour frame (30 Hz -> 10 Hz)")
    ap.add_argument("--max-dt", type=float, default=0.05)
    ap.add_argument("--gt", default=None, help="ground-truth file (default: <seq>/groundtruth.txt)")
    a = ap.parse_args()
    seq, out = Path(a.seq), Path(a.out)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    t_rgb, rgb = read_stamped(seq / "color.txt", 1)
    t_dep, dep = read_stamped(seq / "aligned_depth.txt", 1)
    t_gt, gt = read_stamped(Path(a.gt) if a.gt else seq / "groundtruth.txt", 7)
    t_od, od = read_stamped(seq / "odom.txt", 7)
    T_bc = read_extrinsic(seq, "d400_color_optical_frame")
    sel = np.arange(0, len(t_rgb), a.stride)
    G = interp_poses(t_gt, np.asarray(gt, dtype=np.float64), t_rgb[sel])
    O = interp_poses(t_od, np.asarray(od, dtype=np.float64), t_rgb[sel])
    poses, odoms, k = [], [], 0
    for n, i in enumerate(sel):
        j = int(np.argmin(np.abs(t_dep - t_rgb[i])))
        if abs(t_dep[j] - t_rgb[i]) > a.max_dt or not np.isfinite(G[n]).all() or not np.isfinite(O[n]).all():
            continue
        poses.append((G[n] @ T_bc).reshape(-1))
        odoms.append((O[n] @ T_bc).reshape(-1))
        for sub, src in (("rgb", rgb[i][0]), ("depth", dep[j][0])):
            dst = out / sub / f"{k:06d}.png"
            if not dst.exists():
                os.symlink(os.path.abspath(seq / src), dst)
        k += 1
    np.savetxt(out / "poses_left.txt", np.asarray(poses), fmt="%.9f")
    np.savetxt(out / "odom_left.txt", np.asarray(odoms), fmt="%.9f")
    fx, fy, cx, cy, w, h = read_intrinsics(seq, "d400_color_optical_frame")
    calib = {"K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "width": w, "height": h, "fps": 30.0 / a.stride,
             "dataset": "openloris", "source": str(seq), "odometry": "wheel (odom_left.txt)"}
    (out / "calib.json").write_text(json.dumps(calib, indent=1))
    print(f"{seq.name}: {k} frames -> {out}")


if __name__ == "__main__":
    main()
