#!/usr/bin/env python3
"""Prepare OpenLORIS-Scene sequences (package format) for the CROSS benchmark.

For every sequence two folders are written, both at 10 Hz and in the scene's shared ground-truth frame:

  <out>/<seq>/rgbd/    D435i colour + aligned depth (symlinks), the layout of cross/dataloader/posed_rgbd.py
                       (also the monocular setup: the colour images only)
  <out>/<seq>/stereo/  T265 fisheye pair rectified to a pinhole stereo pair (left/, right/; 640x480, 90 deg HFOV)

each with poses_left.txt (ground-truth camera-to-world of the left / colour camera, 16 values per row),
odom_left.txt (the robot's wheel odometry expressed as the same camera's poses), times.txt (image timestamps) and
calib.json.  Ground truth and odometry are interpolated at the image timestamps; frames outside their time ranges are
dropped.

  python benchmark/datasets/prepare_openloris.py <openloris_root> <out_root> [--seqs office1-1 ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "datasets"))
from convert_openloris import interp_poses, read_extrinsic, read_intrinsics, read_stamped  # noqa: E402

RECT_W, RECT_H, RECT_HFOV_DEG = 640, 480, 90.0


def _inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def read_fisheye(seq: Path, sensor: str):
    fs = cv2.FileStorage(str(seq / "sensors.yaml"), cv2.FILE_STORAGE_READ)
    n = fs.getNode(sensor)
    fx, cx, fy, cy = [float(v) for v in n.getNode("intrinsics").mat().reshape(-1)]
    D = np.asarray(n.getNode("distortion_coefficients").mat(), dtype=np.float64).reshape(-1)[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return K, D, (int(n.getNode("width").real()), int(n.getNode("height").real()))


def read_child_extrinsic(seq: Path, parent: str, child: str) -> np.ndarray:
    fs = cv2.FileStorage(str(seq / "trans_matrix.yaml"), cv2.FILE_STORAGE_READ)
    node = fs.getNode("trans_matrix")
    for i in range(node.size()):
        n = node.at(i)
        if n.getNode("parent_frame").string() == parent and n.getNode("child_frame").string() == child:
            return np.asarray(n.getNode("matrix").mat(), dtype=np.float64)
    raise KeyError(f"no {parent} -> {child} in {seq}/trans_matrix.yaml")


def fisheye_rectifier(seq: Path):
    """Maps that rectify the T265 pair to a pinhole stereo pair; returns (maps, K_rect, baseline, T_cam1_rect)."""
    K1, D1, size = read_fisheye(seq, "t265_fisheye1_optical_frame")
    K2, D2, _ = read_fisheye(seq, "t265_fisheye2_optical_frame")
    T12 = read_child_extrinsic(seq, "t265_fisheye1_optical_frame", "t265_fisheye2_optical_frame")  # cam2 in cam1
    T21 = _inv(T12)                                           # x2 = R x1 + t
    R1, R2, _, _, _ = cv2.fisheye.stereoRectify(K1, D1, K2, D2, size, T21[:3, :3], T21[:3, 3],
                                                flags=cv2.CALIB_ZERO_DISPARITY)
    f = (RECT_W / 2) / np.tan(np.radians(RECT_HFOV_DEG) / 2)
    K = np.array([[f, 0, (RECT_W - 1) / 2], [0, f, (RECT_H - 1) / 2], [0, 0, 1.0]])
    maps = [cv2.fisheye.initUndistortRectifyMap(Kc, Dc, Rc, K, (RECT_W, RECT_H), cv2.CV_16SC2)
            for Kc, Dc, Rc in ((K1, D1, R1), (K2, D2, R2))]
    T_cam1_rect = np.eye(4)
    T_cam1_rect[:3, :3] = R1.T                                # rectified left camera in the fisheye1 frame
    baseline = float(np.linalg.norm(T21[:3, 3]))
    return maps, K, baseline, T_cam1_rect


def write_common(out: Path, poses, odoms, times, calib):
    np.savetxt(out / "poses_left.txt", np.asarray(poses).reshape(-1, 16), fmt="%.9f")
    np.savetxt(out / "odom_left.txt", np.asarray(odoms).reshape(-1, 16), fmt="%.9f")
    np.savetxt(out / "times.txt", np.asarray(times), fmt="%.6f")
    (out / "calib.json").write_text(json.dumps(calib, indent=1))


def prepare_rgbd(seq: Path, out: Path, G_fn, O_fn, stride: int, max_dt: float) -> int:
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("rgb", "depth"):
        (out / sub).mkdir(exist_ok=True)
    t_rgb, rgb = read_stamped(seq / "color.txt", 1)
    t_dep, dep = read_stamped(seq / "aligned_depth.txt", 1)
    T_bc = read_extrinsic(seq, "d400_color_optical_frame")
    sel = np.arange(0, len(t_rgb), stride)
    G, O = G_fn(t_rgb[sel]), O_fn(t_rgb[sel])
    poses, odoms, times, k = [], [], [], 0
    for n, i in enumerate(sel):
        j = int(np.argmin(np.abs(t_dep - t_rgb[i])))
        if abs(t_dep[j] - t_rgb[i]) > max_dt or not np.isfinite(G[n]).all() or not np.isfinite(O[n]).all():
            continue
        poses.append(G[n] @ T_bc)
        odoms.append(O[n] @ T_bc)
        times.append(t_rgb[i])
        for sub, src in (("rgb", rgb[i][0]), ("depth", dep[j][0])):
            dst = out / sub / f"{k:06d}.png"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(os.path.abspath(seq / src), dst)
        k += 1
    fx, fy, cx, cy, w, h = read_intrinsics(seq, "d400_color_optical_frame")
    write_common(out, poses, odoms, times, {
        "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "width": w, "height": h, "fps": 30.0 / stride,
        "dataset": "openloris", "sensor": "d435i", "source": seq.name, "odometry": "wheel (odom_left.txt)",
        "depth": "D435i aligned depth, uint16 mm"})
    return k


def prepare_stereo(seq: Path, out: Path, G_fn, O_fn, stride: int, max_dt: float) -> int:
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("left", "right"):
        (out / sub).mkdir(exist_ok=True)
    t1, f1 = read_stamped(seq / "fisheye1.txt", 1)
    t2, f2 = read_stamped(seq / "fisheye2.txt", 1)
    maps, K, baseline, T_cam1_rect = fisheye_rectifier(seq)
    T_b1 = read_extrinsic(seq, "t265_fisheye1_optical_frame")
    T_brect = T_b1 @ T_cam1_rect
    sel = np.arange(0, len(t1), stride)
    G, O = G_fn(t1[sel]), O_fn(t1[sel])
    poses, odoms, times, k = [], [], [], 0
    for n, i in enumerate(sel):
        j = int(np.argmin(np.abs(t2 - t1[i])))
        if abs(t2[j] - t1[i]) > max_dt or not np.isfinite(G[n]).all() or not np.isfinite(O[n]).all():
            continue
        for sub, src, (mx, my) in (("left", f1[i][0], maps[0]), ("right", f2[j][0], maps[1])):
            dst = out / sub / f"{k:06d}.png"
            if not dst.exists():
                img = cv2.imread(str(seq / src), cv2.IMREAD_UNCHANGED)
                cv2.imwrite(str(dst), cv2.remap(img, mx, my, cv2.INTER_LINEAR))
        poses.append(G[n] @ T_brect)
        odoms.append(O[n] @ T_brect)
        times.append(t1[i])
        k += 1
    T_rl = np.eye(4)
    T_rl[0, 3] = baseline
    write_common(out, poses, odoms, times, {
        "K": K.tolist(), "width": RECT_W, "height": RECT_H, "fps": 30.0 / stride, "baseline": baseline,
        "T_right_in_left": T_rl.tolist(), "dataset": "openloris", "sensor": "t265 (rectified, grayscale)",
        "source": seq.name, "odometry": "wheel (odom_left.txt)"})
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="OpenLORIS package root (one folder per sequence, e.g. office1-1/)")
    ap.add_argument("out")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--setups", nargs="*", default=["rgbd", "stereo"])
    ap.add_argument("--stride", type=int, default=3, help="30 Hz -> 10 Hz")
    ap.add_argument("--max-dt", type=float, default=0.02)
    a = ap.parse_args()
    root, out_root = Path(a.root), Path(a.out)
    seqs = a.seqs or sorted(p.name for p in root.iterdir() if (p / "color.txt").is_file())
    for s in seqs:
        seq = root / s
        t_gt, gt = read_stamped(seq / "groundtruth.txt", 7)
        t_od, od = read_stamped(seq / "odom.txt", 7)
        gt, od = np.asarray(gt, dtype=np.float64), np.asarray(od, dtype=np.float64)

        def G_fn(t, t_gt=t_gt, gt=gt):
            return interp_poses(t_gt, gt, t)

        def O_fn(t, t_od=t_od, od=od):
            return interp_poses(t_od, od, t)

        for setup in a.setups:
            fn = prepare_rgbd if setup == "rgbd" else prepare_stereo
            n = fn(seq, out_root / s / setup, G_fn, O_fn, a.stride, a.max_dt)
            print(f"{s}/{setup}: {n} frames", flush=True)


if __name__ == "__main__":
    main()
