#!/usr/bin/env python3
"""Stereo visual-inertial odometry (VIO) for the prepared benchmark sequences: writes odom_vio.txt next to the images.

Datasets without wheel odometry (KITTI, ROVER, SimChange) get their odometry from a stereo-inertial VIO run on the
platform's own stereo pair and IMU, instead of ground-truth-derived odometry (KITTI: INS velocities; ROVER, SimChange:
noisy ground truth).  The VIO runs causally over the whole recording; the pose it outputs for a frame is the estimate
at the time that frame was processed (no smoothing, no loop closure), i.e. what an online VIO would hand CROSS.

  <seq>/odom_vio.txt   one row per frame: 4x4 camera-to-world pose of the left camera (16 values, the format of
                       odom_left.txt / poses_left.txt); frames before the VIO's first pose hold that pose
  <seq>/odom_vio.json  method, inputs, calibration, frames without an estimate, runtime

Inputs per dataset (the stereo pair is the one of the benchmark's stereo setup; the IMU is rigidly attached to it):

  kitti      rectified colour pair cam 2/3 (10 Hz, as greyscale) + the OXTS RT3003 accelerations and angular rates at
             100 Hz from the raw *_extract drives (the synced drives have 10 Hz only; fetch_kitti_oxts.py downloads the
             oxts folders alone); the camera's own timestamps (image_02 of the synced drive)
  rover      T265 fisheye pair rectified as in prepare_rover.py (10 Hz) + the T265's BMI055 IMU (~200 Hz gyroscope),
             Kalibr extrinsics / noise of calibration/calib_t265.yaml; the rgbd folder (D435i colour) gets the same
             odometry, interpolated to its frame times and moved to its camera through the prism calibration
  simchange  rendered pair (baseline of the benchmark config) + the simulated BMI055 IMU of prepare_imu.py (200 Hz)
  openloris  (--write-imu only: OpenLORIS keeps its wheel odometry) T265 fisheye pair rectified as in
             prepare_openloris.py + the T265's BMI055 IMU (gyroscope 200 Hz, accelerometer 62 Hz interpolated to the
             gyroscope times; factory intrinsics of sensors.yaml applied, extrinsics of trans_matrix.yaml)

--write-imu NAME writes the stereo pair's IMU stream without running a VIO: <seq>/NAME.txt / NAME.json (format of
cross/dataloader/imu.py; T_cam_imu = pose of the IMU in the rectified left camera) and NAME_frames.txt (the frames'
times on the IMU clock), for the baselines that run with an IMU (ORB-SLAM3 inertial, RTAB-Map with IMU).

VIO back ends (built by scripts/vio/install_basalt.sh, scripts/vio/install_okvis2.sh):

  basalt     Basalt VIO (Usenko et al., RA-L 2020): KLT patch tracking + square-root sliding-window optimisation
  okvis2     OKVIS2-X (Boche et al., T-RO 2025) in VIO mode (loop closures and final BA off), causal trajectory

  python benchmark/datasets/prepare_vio.py kitti  $BENCH_DATA/kitti  --raw /path/kitti_raw --extract /path/kitti_extract
  python benchmark/datasets/prepare_vio.py rover  $BENCH_DATA/rover  --raw /path/rover
  python benchmark/datasets/prepare_vio.py simchange $BENCH_DATA/simchange --baseline 0.3
      [--vio basalt|okvis2] [--seqs ...] [--work /tmp/vio_work] [--out-name odom_vio.txt]
  python benchmark/datasets/prepare_vio.py openloris $BENCH_DATA/openloris --raw /path/openloris --write-imu imu_vio
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark" / "datasets"))

BASALT = os.environ.get("BASALT_VIO", "basalt_vio")
OKVIS2 = os.environ.get("OKVIS2_APP", "okvis_app_synchronous")


@dataclass
class ViSequence:
    """A stereo pair + IMU in the frame of the rectified left camera of a prepared benchmark folder."""
    name: str
    out: Path                       # the prepared folder (odom_vio.txt is written here)
    left: list
    right: list
    t_frames: np.ndarray            # s, on the IMU clock
    K: np.ndarray
    size: tuple                     # (width, height)
    baseline: float
    T_cam_imu: np.ndarray           # x_cam = T x_imu (rectified left camera)
    imu_t: np.ndarray
    gyro: np.ndarray                # rad/s, IMU frame
    acc: np.ndarray                 # m/s^2 specific force, IMU frame
    noise: dict                     # gyro/accel noise densities and random walks (continuous time)
    source: dict = field(default_factory=dict)
    # other prepared folders of the same recording that get the odometry too: (folder, T_cam_target = pose of the
    # target camera in the rectified left camera, frame times of the target on the IMU clock)
    targets: list = field(default_factory=list)


def _clean_imu(t, w, a):
    o = np.argsort(t, kind="stable")
    t, w, a = t[o], w[o], a[o]
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    return t[keep], w[keep], a[keep]


def _utc(stamp: str) -> float:
    """KITTI timestamp string (local wall time without zone) -> seconds, independent of the machine's time zone."""
    s = stamp.strip()
    dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    frac = float("0" + s[19:]) if len(s) > 19 else 0.0
    return calendar.timegm(dt.timetuple()) + frac


# ---------------------------------------------------------------------------------------------------- KITTI
# OXTS RT3003 (datasheet): gyro angular random walk 0.2 deg/sqrt(h), bias stability 2 deg/h; accelerometer noise
# ~0.1 mg/sqrt(Hz), bias stability 5 ug.  The densities below are inflated by ~20x (gyro) / ~20x (accel), the usual
# margin for vehicle vibration and the unmodelled 100 Hz sample jitter of the raw logs; not tuned on any sequence.
KITTI_NOISE = {"gyro_noise_density": 1.2e-3, "accel_noise_density": 2.0e-2,
               "gyro_random_walk": 2.0e-5, "accel_random_walk": 2.0e-3}


def kitti(bench: Path, raw: Path, extract: Path, seqs):
    from prepare_kitti import SEQUENCES
    from cross.dataloader.kitti_calib import load_kitti_calibration
    out = []
    for s in seqs or list(SEQUENCES):
        date, drive, lo, hi = SEQUENCES[s]
        folder = bench / s / "stereo"
        ox = extract / date / f"{drive}_extract" / "oxts"
        sync = raw / date / f"{drive}_sync"
        if not folder.is_dir() or not ox.is_dir():
            print(f"kitti {s}: no prepared folder or no extract oxts ({ox}), skipped")
            continue
        stamps = (ox / "timestamps.txt").read_text().splitlines()
        files = sorted((ox / "data").glob("*.txt"))
        assert len(files) == len(stamps), f"{ox}: {len(files)} files, {len(stamps)} stamps"
        t = np.asarray([_utc(x) for x in stamps])
        o = np.stack([np.loadtxt(f) for f in files])
        # 11-13 ax ay az, 17-19 wx wy wz: vehicle body frame (x forward, y left, z up), specific force incl. gravity
        t, w, a = _clean_imu(t, o[:, 17:20], o[:, 11:14])
        cam = [_utc(x) for x in (sync / "image_02" / "timestamps.txt").read_text().splitlines()][lo:hi + 1]
        calib = json.loads((folder / "calib.json").read_text())
        T_cam_imu = np.asarray(load_kitti_calibration(raw / date)["cam2_from_imu"], dtype=np.float64)
        left = sorted((folder / "left").iterdir())
        right = sorted((folder / "right").iterdir())
        assert len(left) == len(right) == len(cam), (len(left), len(right), len(cam))
        out.append(ViSequence(f"kitti/{s}", folder, left, right, np.asarray(cam), np.asarray(calib["K"], float),
                              (int(calib["width"]), int(calib["height"])), float(calib["baseline"]), T_cam_imu,
                              t, w, a, dict(KITTI_NOISE),
                              {"imu": f"{drive}_extract/oxts (OXTS RT3003, {1 / np.median(np.diff(t)):.0f} Hz)",
                               "frame_times": f"{drive}_sync/image_02/timestamps.txt"}))
    return out


# ---------------------------------------------------------------------------------------------------- ROVER
def rover(bench: Path, raw: Path, seqs):
    import yaml
    from prepare_rover import read_calib, stereo_rectification
    cal_d, cal_t = read_calib(raw)
    T_camleft_imu = np.asarray(cal_t["IMU-To-CamLeft"], dtype=np.float64)       # Kalibr: x_camleft = T x_imu
    nz = cal_t["IMU_Intrinsics"]
    noise = {"gyro_noise_density": nz["noise_gyro"], "accel_noise_density": nz["noise_acc"],
             "gyro_random_walk": nz["walk_gyro"], "accel_random_walk": nz["walk_acc"]}
    names = seqs or sorted(p.name for p in bench.iterdir() if (p / "stereo" / "times.txt").is_file())
    out = []
    for name in names:
        folder = bench / name / "stereo"
        if not (raw / f"{name}.zip").is_file() or not (folder / "times.txt").is_file():
            print(f"rover {name}: no zip or no prepared stereo folder, skipped")
            continue
        z = zipfile.ZipFile(raw / f"{name}.zip")
        top = z.namelist()[0].split("/")[0]
        _, K, T_prism_rect, baseline = stereo_rectification(z, top, cal_t)
        T_prism_cam = np.asarray(cal_t["CamLeft-To-Prism"], dtype=np.float64)
        T_cam_rect = np.linalg.inv(T_prism_cam) @ T_prism_rect                     # rectified -> original left camera
        T_rect_imu = np.linalg.inv(T_cam_rect) @ T_camleft_imu
        rows = [l.split(",") for l in z.read(f"{top}/realsense_T265/imu/imu.txt").decode().splitlines()
                if l.strip() and not l.startswith("#")]
        d = np.asarray(rows, dtype=np.float64)                                     # t, ax, ay, az, wx, wy, wz
        t, w, a = _clean_imu(d[:, 0], d[:, 4:7], d[:, 1:4])
        calib = json.loads((folder / "calib.json").read_text())
        left = sorted((folder / "left").iterdir())
        right = sorted((folder / "right").iterdir())
        tf = np.loadtxt(folder / "times.txt", dtype=np.float64).reshape(-1)
        assert len(left) == len(right) == len(tf)
        targets = []
        rgbd = bench / name / "rgbd"
        if (rgbd / "times.txt").is_file():
            T_rect_d435 = np.linalg.inv(T_prism_rect) @ np.asarray(cal_d["Cam-To-Prism"], dtype=np.float64)
            targets.append((rgbd, T_rect_d435, np.loadtxt(rgbd / "times.txt", dtype=np.float64).reshape(-1)))
        out.append(ViSequence(f"rover/{name}", folder, left, right, tf, np.asarray(calib["K"], float),
                              (int(calib["width"]), int(calib["height"])), float(calib["baseline"]), T_rect_imu,
                              t, w, a, dict(noise),
                              {"imu": f"realsense_T265/imu ({1 / np.median(np.diff(t)):.0f} Hz), calib_t265.yaml"},
                              targets))
    return out


# ---------------------------------------------------------------------------------------------------- OpenLORIS
def openloris(bench: Path, raw: Path, seqs):
    from prepare_imu import _openloris_intrinsic
    from prepare_openloris import fisheye_rectifier, read_child_extrinsic
    from cross.imu.simulate import BMI055
    names = seqs or sorted(p.name for p in bench.iterdir() if (p / "stereo" / "times.txt").is_file())
    out = []
    for name in names:
        folder = bench / name / "stereo"
        src = raw / name
        if not (folder / "times.txt").is_file() or not (src / "t265_gyroscope.txt").is_file():
            print(f"openloris {name}: no prepared stereo folder or no T265 IMU, skipped")
            continue
        gyro = np.loadtxt(src / "t265_gyroscope.txt", comments="#", dtype=np.float64)
        acc = np.loadtxt(src / "t265_accelerometer.txt", comments="#", dtype=np.float64)
        S_g, b_g = _openloris_intrinsic(src, "t265_gyroscope")
        S_a, b_a = _openloris_intrinsic(src, "t265_accelerometer")
        tg = gyro[:, 0]
        inside = (tg >= acc[0, 0]) & (tg <= acc[-1, 0])
        a_raw = np.stack([np.interp(tg, acc[:, 0], acc[:, k]) for k in (1, 2, 3)], axis=1)
        w = gyro[:, 1:4] @ S_g.T - b_g
        a = a_raw @ S_a.T - b_a
        t, w, a = _clean_imu(tg[inside], w[inside], a[inside])
        _, K, baseline, T_cam1_rect = fisheye_rectifier(src)          # rectified left camera in the fisheye1 frame
        T_fish1_imu = read_child_extrinsic(src, "t265_fisheye1_optical_frame", "t265_gyroscope")
        T_rect_imu = np.linalg.inv(T_cam1_rect) @ T_fish1_imu
        calib = json.loads((folder / "calib.json").read_text())
        left = sorted((folder / "left").iterdir())
        right = sorted((folder / "right").iterdir())
        tf = np.loadtxt(folder / "times.txt", dtype=np.float64).reshape(-1)
        assert len(left) == len(right) == len(tf)
        out.append(ViSequence(f"openloris/{name}", folder, left, right, tf, np.asarray(calib["K"], float),
                              (int(calib["width"]), int(calib["height"])), float(calib["baseline"]), T_rect_imu,
                              t, w, a, dict(BMI055),
                              {"imu": f"OpenLORIS {name} T265 IMU ({1 / np.median(np.diff(t)):.0f} Hz, factory intrinsics)"}))
    return out


def write_imu_stream(seq: ViSequence, name: str):
    """The stereo pair's IMU as <name>.txt / <name>.json + <name>_frames.txt (frame times on the IMU clock)."""
    from cross.dataloader.imu import write_imu
    keep = (seq.imu_t >= seq.t_frames[0] - 1.0) & (seq.imu_t <= seq.t_frames[-1] + 1.0)
    np.savetxt(seq.out / f"{name}_frames.txt", np.asarray(seq.t_frames, dtype=np.float64), fmt="%.6f")
    write_imu(seq.out, seq.imu_t[keep], seq.gyro[keep], seq.acc[keep],
              {"T_cam_imu": np.asarray(seq.T_cam_imu).tolist(), **seq.noise, "rate_hz": _imu_rate(seq),
               "frame_times": f"{name}_frames.txt", "source": seq.source.get("imu", ""),
               "camera": "rectified left camera of the stereo folder"}, name=name)
    g = np.linalg.norm(seq.acc[keep], axis=1)
    print(f"{seq.name}: {int(keep.sum())} IMU samples ({_imu_rate(seq):.0f} Hz), |a| median {np.median(g):.3f}, "
          f"frames {seq.t_frames[0]:.3f}-{seq.t_frames[-1]:.3f}, IMU {seq.imu_t[keep][0]:.3f}-{seq.imu_t[keep][-1]:.3f}",
          flush=True)


# ---------------------------------------------------------------------------------------------------- SimChange
def simchange(bench: Path, baseline: float, seqs):
    from cross.dataloader.imu import ImuCalibration
    folders = [bench / s for s in seqs] if seqs else sorted(p.parent for p in bench.glob("*/*/poses_left.txt"))
    out = []
    for folder in folders:
        calib = json.loads((folder / "calib.json").read_text())
        rdir = calib.get("right_dirs", {}).get(f"{baseline:.2f}", "right")
        ic = ImuCalibration.from_dict(json.loads((folder / "imu.json").read_text()))
        d = np.loadtxt(folder / "imu.txt", comments="#", dtype=np.float64).reshape(-1, 7)
        t, w, a = _clean_imu(d[:, 0], d[:, 1:4], d[:, 4:7])
        left = sorted((folder / "left").glob("*.png")) or sorted((folder / "left").glob("*.jpg"))
        right = sorted((folder / rdir).glob("*.png")) or sorted((folder / rdir).glob("*.jpg"))
        fps = float(calib.get("fps", 10.0))
        tf = (np.loadtxt(folder / "times.txt").reshape(-1) if (folder / "times.txt").is_file()
              else np.arange(len(left)) / fps)
        assert len(left) == len(right) == len(tf), (folder, len(left), len(right))
        noise = {k: getattr(ic, k) for k in ("gyro_noise_density", "accel_noise_density", "gyro_random_walk",
                                             "accel_random_walk")}
        out.append(ViSequence(f"simchange/{folder.parent.name}/{folder.name}", folder, left, right, tf,
                              np.asarray(calib["K"], float), (int(calib["width"]), int(calib["height"])), baseline,
                              ic.T_cam_imu, t, w, a, noise, {"imu": ic.source, "right": rdir}))
    return out


# ---------------------------------------------------------------------------------------------------- EuRoC export
def _ns(t):
    return np.round(np.asarray(t) * 1e9).astype(np.int64)


def export_euroc(seq: ViSequence, work: Path, threads=16) -> np.ndarray:
    """mav0/{cam0,cam1,imu0} with greyscale PNGs named by their timestamp (ns); returns the frame timestamps (ns)."""
    mav = work / "mav0"
    ns = _ns(seq.t_frames)
    assert np.all(np.diff(ns) > 0), "frame timestamps must increase"
    for ci, paths in enumerate((seq.left, seq.right)):
        d = mav / f"cam{ci}" / "data"
        d.mkdir(parents=True, exist_ok=True)

        def conv(i, paths=paths, d=d):
            dst = d / f"{ns[i]}.png"
            if not dst.is_file():
                img = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(paths[i])
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(conv, range(len(ns))))
        with open(mav / f"cam{ci}" / "data.csv", "w") as f:
            f.write("#timestamp [ns],filename\n")
            f.writelines(f"{n},{n}.png\n" for n in ns)
    (mav / "imu0").mkdir(parents=True, exist_ok=True)
    keep = (seq.imu_t >= seq.t_frames[0] - 1.0) & (seq.imu_t <= seq.t_frames[-1] + 1.0)
    with open(mav / "imu0" / "data.csv", "w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for n, w, a in zip(_ns(seq.imu_t[keep]), seq.gyro[keep], seq.acc[keep]):
            f.write(f"{n},{w[0]:.9f},{w[1]:.9f},{w[2]:.9f},{a[0]:.9f},{a[1]:.9f},{a[2]:.9f}\n")
    return ns


def _T_imu_cams(seq: ViSequence):
    T_imu_c0 = np.linalg.inv(seq.T_cam_imu)
    T_c0_c1 = np.eye(4)
    T_c0_c1[0, 3] = seq.baseline                    # rectified pair: right camera at +baseline along x
    return T_imu_c0, T_imu_c0 @ T_c0_c1


def _imu_rate(seq):
    return float(1.0 / np.median(np.diff(seq.imu_t)))


# ---------------------------------------------------------------------------------------------------- Basalt
def basalt_calib(seq: ViSequence) -> dict:
    def pose(T):
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        return {"px": T[0, 3], "py": T[1, 3], "pz": T[2, 3], "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3]}
    fx, fy, cx, cy = seq.K[0, 0], seq.K[1, 1], seq.K[0, 2], seq.K[1, 2]
    intr = {"camera_type": "pinhole", "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy}}
    n = seq.noise
    rate = _imu_rate(seq)
    # Basalt's *_noise_std / *_bias_std are the continuous-time densities (basalt/calibration/calibration.hpp)
    return {"value0": {
        "T_imu_cam": [pose(T) for T in _T_imu_cams(seq)],
        "intrinsics": [intr, dict(intr)],
        "resolution": [list(seq.size), list(seq.size)],
        # no vignetting model (flat spline, the format of Basalt's calibration files)
        "vignette": [{"value0": 0, "value1": 50000000000, "value2": [[1.0]] * 15} for _ in range(2)],
        "calib_accel_bias": [0.0] * 9, "calib_gyro_bias": [0.0] * 12,
        "imu_update_rate": rate,
        "accel_noise_std": [n["accel_noise_density"]] * 3,
        "gyro_noise_std": [n["gyro_noise_density"]] * 3,
        "accel_bias_std": [n["accel_random_walk"]] * 3,
        "gyro_bias_std": [n["gyro_random_walk"]] * 3,
        "T_mocap_world": pose(np.eye(4)), "T_imu_marker": pose(np.eye(4)),
        "mocap_time_offset_ns": 0, "mocap_to_imu_offset_ns": 0, "cam_time_offset_ns": 0}}


def basalt_config(template: Path, overrides: dict) -> dict:
    cfg = json.loads(template.read_text())
    cfg["value0"].update({f"config.{k}": v for k, v in overrides.items()})
    return cfg


def run_basalt(seq: ViSequence, work: Path, args) -> tuple[dict, dict]:
    calib = work / "basalt_calib.json"
    calib.write_text(json.dumps(basalt_calib(seq), indent=1))
    tmpl = Path(args.basalt_config) if args.basalt_config else Path(BASALT).resolve().parents[2] / "data/euroc_config.json"
    over = dict(json.loads(args.basalt_overrides)) if args.basalt_overrides else {}
    cfg = work / "basalt_config.json"
    cfg.write_text(json.dumps(basalt_config(tmpl, over), indent=1))
    cmd = [BASALT, "--dataset-path", str(work), "--cam-calib", str(calib), "--dataset-type", "euroc",
           "--config-path", str(cfg), "--show-gui", "0", "--save-trajectory", "tum", "--use-imu", "1",
           "--num-threads", str(args.threads)]
    rc, dt = _run(cmd, work, work / "vio.log", args.timeout)
    traj = {}
    if (work / "trajectory.txt").is_file():
        for row in np.loadtxt(work / "trajectory.txt", ndmin=2):
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
            T[:3, 3] = row[1:4]
            traj[int(round(row[0] * 1e9))] = T                            # T_world_imu at the frame's time
    return traj, {"rc": rc, "seconds": dt, "cmd": " ".join(cmd), "config_overrides": over}


# ---------------------------------------------------------------------------------------------------- OKVIS2-X
def okvis_config(seq: ViSequence, template: Path) -> str:
    """The template (OpenCV FileStorage YAML, which PyYAML cannot round-trip) with the cameras block replaced and the
    IMU noise, VIO mode (no loop closure, no final BA) and headless output set by key."""
    import re
    text = template.read_text()
    cams = []
    for T in _T_imu_cams(seq):
        rows = ",\n          ".join(", ".join(f"{x:.12g}" for x in r) for r in T)
        cams.append(f"     - {{T_SC:\n        [ {rows} ],\n"
                    f"        image_dimension: [{seq.size[0]}, {seq.size[1]}],\n"
                    f"        distortion_coefficients: [0.0, 0.0, 0.0, 0.0],\n"
                    f"        distortion_type: radialtangential,\n"
                    f"        focal_length: [{seq.K[0, 0]:.9g}, {seq.K[1, 1]:.9g}],\n"
                    f"        principal_point: [{seq.K[0, 2]:.9g}, {seq.K[1, 2]:.9g}],\n"
                    f"        cam_model: pinhole,\n        camera_type: gray,\n        mapping: false,\n"
                    f"        mapping_rectification: false,\n        slam_use: okvis}}\n")
    a, b = text.index("cameras:"), text.index("# additional camera parameters")
    text = text[:a] + "cameras:\n" + "\n".join(cams) + "\n" + text[b:]
    n = seq.noise
    sets = {"sigma_g_c": n["gyro_noise_density"], "sigma_a_c": n["accel_noise_density"],
            "sigma_gw_c": n["gyro_random_walk"], "sigma_aw_c": n["accel_random_walk"],
            "a0": "[ 0.0, 0.0, 0.0 ]", "g0": "[ 0.0, 0.0, 0.0 ]", "g_max": 20.0, "a_max": 200.0,
            "do_loop_closures": "false", "do_final_ba": "false", "enforce_realtime": "false",
            "display_topview": "false", "display_matches": "false", "display_overhead": "false",
            "enable_submapping": "false"}
    for k, v in sets.items():
        text, cnt = re.subn(rf"^(\s*{k}:)\s*[^#\n]*", lambda m: f"{m.group(1)} {v} ", text, flags=re.M)
        assert cnt == 1, (k, cnt)
    return text


def run_okvis2(seq: ViSequence, work: Path, args) -> tuple[dict, dict]:
    tmpl = Path(args.okvis_config) if args.okvis_config else Path(OKVIS2).resolve().parents[1] / "config/euroc/okvis2.yaml"
    cfg = work / "okvis2.yaml"
    cfg.write_text(okvis_config(seq, tmpl))
    res = work / "okvis_out"
    res.mkdir(exist_ok=True)
    cmd = [OKVIS2, str(cfg), str(work / "mav0") + "/", str(res) + "/"]
    rc, dt = _run(cmd, work, work / "vio.log", args.timeout)
    traj = {}
    csvs = sorted(res.glob("*trajectory.csv"))
    causal = [c for c in csvs if "final" not in c.name]
    if causal:
        import csv
        with open(causal[0]) as f:
            r = csv.reader(f)
            head = next(r)
            for row in r:
                v = [float(x) for x in row[:8]]
                T = np.eye(4)
                T[:3, 3] = v[1:4]
                T[:3, :3] = Rotation.from_quat(v[4:8]).as_matrix()       # qx qy qz qw (OKVIS csv order)
                traj[int(round(v[0]))] = T                                 # T_world_sensor(IMU)
    return traj, {"rc": rc, "seconds": dt, "cmd": " ".join(cmd), "trajectory_file": causal[0].name if causal else None}


def _run(cmd, cwd, log, timeout):
    t0 = time.time()
    with open(log, "w") as f:
        try:
            rc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    return rc, time.time() - t0


# ---------------------------------------------------------------------------------------------------- odometry
def to_odometry(traj: dict, ns: np.ndarray, T_cam_imu: np.ndarray, tol_ns=2_000_000):
    """Camera c2w poses at every frame from the VIO's IMU poses: matched by timestamp (within tol); a frame without an
    estimate keeps the previous frame's pose (zero motion), frames before the first estimate hold the first one."""
    keys = np.asarray(sorted(traj))
    T_imu_cam = np.linalg.inv(T_cam_imu)
    poses = np.full((len(ns), 4, 4), np.nan)
    have = np.zeros(len(ns), bool)
    if len(keys):
        j = np.clip(np.searchsorted(keys, ns), 1, len(keys) - 1) if len(keys) > 1 else np.zeros(len(ns), int)
        for i, n in enumerate(ns):
            cands = [keys[k] for k in {int(j[i]) - 1, int(j[i])} if 0 <= k < len(keys)]
            best = min(cands, key=lambda c: abs(c - n))
            if abs(best - n) <= tol_ns:
                poses[i] = traj[best] @ T_imu_cam
                have[i] = True
    if not have.any():
        return None, have
    first = int(np.argmax(have))
    poses[:first] = poses[first]
    for i in range(first + 1, len(ns)):
        if not have[i]:
            poses[i] = poses[i - 1]
    return poses, have


def interpolate_poses(t_src: np.ndarray, poses: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Poses at times t from poses at increasing times t_src: rotation slerp, linear translation (clamped at the ends)."""
    from scipy.spatial.transform import Slerp
    tc = np.clip(t, t_src[0], t_src[-1])
    out = np.tile(np.eye(4), (len(t), 1, 1))
    out[:, :3, :3] = Slerp(t_src, Rotation.from_matrix(poses[:, :3, :3]))(tc).as_matrix()
    for k in range(3):
        out[:, k, 3] = np.interp(tc, t_src, poses[:, k, 3])
    return out


def write_targets(seq: ViSequence, poses: np.ndarray, args, meta: dict):
    for folder, T_cam_target, times in seq.targets:
        P = interpolate_poses(seq.t_frames, poses, times) @ T_cam_target
        np.savetxt(folder / args.out_name, P.reshape(-1, 16), fmt="%.9f")
        m = dict(meta, target_of=str(seq.out), T_cam_target=T_cam_target.tolist(),
                 note="the stereo VIO's camera poses interpolated to this folder's frame times, moved to its camera")
        gt = folder / "poses_left.txt"
        if gt.is_file():
            m["eval"] = drift_stats(P, np.loadtxt(gt).reshape(-1, 4, 4))
        (folder / (Path(args.out_name).stem + ".json")).write_text(json.dumps(m, indent=1))
        print(f"{seq.name} -> {folder}: {json.dumps(m.get('eval', {}))}", flush=True)


def process(seq: ViSequence, args):
    work = Path(args.work) / seq.name.replace("/", "__")
    out_txt = seq.out / args.out_name
    if args.targets_only:
        if out_txt.is_file() and seq.targets:
            meta = json.loads((seq.out / (Path(args.out_name).stem + ".json")).read_text())
            meta.pop("eval", None)
            write_targets(seq, np.loadtxt(out_txt).reshape(-1, 4, 4), args, meta)
        return
    if out_txt.is_file() and not args.force:
        print(f"{seq.name}: {out_txt.name} exists, skipped", flush=True)
        return
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ns = export_euroc(seq, work)
    t_export = time.time() - t0
    traj, info = (run_basalt if args.vio == "basalt" else run_okvis2)(seq, work, args)
    poses, have = to_odometry(traj, ns, seq.T_cam_imu)
    meta = {"vio": args.vio, "sequence": seq.name, "n_frames": len(ns), "n_estimated": int(have.sum()),
            "first_estimate": int(np.argmax(have)) if have.any() else None,
            "n_missing_after_first": int((~have[int(np.argmax(have)):]).sum()) if have.any() else len(ns),
            "export_seconds": t_export, "vio_fps": len(ns) / max(info["seconds"], 1e-9), **info,
            "imu_rate_hz": _imu_rate(seq), "noise": seq.noise, "T_cam_imu": seq.T_cam_imu.tolist(),
            "baseline": seq.baseline, "source": seq.source}
    if poses is None:
        meta["failed"] = True
        print(f"{seq.name}: VIO produced no poses (rc {info['rc']}), see {work / 'vio.log'}", flush=True)
    else:
        np.savetxt(out_txt, poses.reshape(-1, 16), fmt="%.9f")
        gt = seq.out / "poses_left.txt"
        if gt.is_file():
            meta["eval"] = drift_stats(poses, np.loadtxt(gt).reshape(-1, 4, 4))
        print(f"{seq.name}: {meta['n_estimated']}/{len(ns)} frames, {meta['vio_fps']:.0f} fps, "
              f"{json.dumps(meta.get('eval', {}))}", flush=True)
        write_targets(seq, poses, args, meta)
    (seq.out / (Path(args.out_name).stem + ".json")).write_text(json.dumps(meta, indent=1))
    if not args.keep_work:
        shutil.rmtree(work / "mav0", ignore_errors=True)


def drift_stats(est: np.ndarray, gt: np.ndarray, window=100, stride=50):
    """Relative error over `window`-frame segments (the benchmark's trial length): translation error in % of the
    distance travelled and in m, rotation error in deg (ground truth used for reporting only)."""
    errs_t, errs_r, errs_y, pct = [], [], [], []
    # heading error: rotation angle about the world vertical (ground truth's most constant camera axis) expressed in the
    # window's first camera frame, estimate vs ground truth; robust to ground truth without roll / pitch (ROVER)
    R = gt[:, :3, :3]
    v = max((R[:, :, ax].mean(0) for ax in (1, 2)), key=np.linalg.norm)
    v = v / np.linalg.norm(v)

    def yaw(Rrel, u):
        a = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = Rrel @ a
        b = b - u * (b @ u)
        return np.arctan2(np.cross(a, b) @ u, a @ b)
    for s in range(0, len(gt) - window, stride):
        e = np.linalg.inv(est[s]) @ est[s + window]
        g = np.linalg.inv(gt[s]) @ gt[s + window]
        d = np.linalg.inv(g) @ e
        dist = float(np.sum(np.linalg.norm(np.diff(gt[s:s + window + 1, :3, 3], axis=0), axis=1)))
        errs_t.append(float(np.linalg.norm(d[:3, 3])))
        errs_r.append(float(np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))))
        u = gt[s, :3, :3].T @ v                         # vertical in the window's first camera frame
        dy = yaw(e[:3, :3], u) - yaw(g[:3, :3], u)
        errs_y.append(float(abs(np.degrees((dy + np.pi) % (2 * np.pi) - np.pi))))
        if dist > 1.0:
            pct.append(100 * errs_t[-1] / dist)
    f = lambda a: (float(np.mean(a)), float(np.median(a)), float(np.percentile(a, 95))) if a else None
    return {"window": window, "trans_m_mean_med_p95": f(errs_t), "rot_deg_mean_med_p95": f(errs_r),
            "yaw_deg_mean_med_p95": f(errs_y), "trans_pct_mean_med_p95": f(pct)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["kitti", "rover", "simchange", "openloris"])
    ap.add_argument("bench", help="prepared benchmark root of the dataset ($BENCH_DATA/<dataset>)")
    ap.add_argument("--raw", help="kitti: raw root with <date>/<drive>_sync; rover: root with <name>.zip, calibration/; "
                                  "openloris: raw root with <seq>/t265_*.txt, sensors.yaml, trans_matrix.yaml")
    ap.add_argument("--write-imu", default=None, metavar="NAME", help="only write the stereo pair's IMU stream as "
                                                                     "NAME.txt / NAME.json (no VIO run)")
    ap.add_argument("--extract", help="kitti: root with <date>/<drive>_extract/oxts (fetch_kitti_oxts.py)")
    ap.add_argument("--baseline", type=float, default=0.3, help="simchange: rendered stereo baseline (m)")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--vio", choices=["basalt", "okvis2"], default="basalt")
    ap.add_argument("--work", default="/tmp/cross_vio_work")
    ap.add_argument("--out-name", default="odom_vio.txt")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=1, help="sequences in parallel")
    ap.add_argument("--timeout", type=float, default=4 * 3600)
    ap.add_argument("--basalt-config", default=None, help="Basalt config template (default: data/euroc_config.json)")
    ap.add_argument("--basalt-overrides", default=None, help='JSON of config keys without "config.", e.g. '
                                                             '\'{"vio_max_kfs": 7}\'')
    ap.add_argument("--okvis-config", default=None, help="OKVIS2-X config template (default: config/euroc/okvis2.yaml)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--targets-only", action="store_true", help="only (re)write the other folders' odometry (rover rgbd) "
                                                                "from an existing odometry file")
    a = ap.parse_args()
    bench = Path(a.bench)
    if a.dataset == "kitti":
        seqs = kitti(bench, Path(a.raw), Path(a.extract), a.seqs)
    elif a.dataset == "rover":
        seqs = rover(bench, Path(a.raw), a.seqs)
    elif a.dataset == "openloris":
        if not a.write_imu:
            sys.exit("openloris: only --write-imu (OpenLORIS keeps its wheel odometry)")
        seqs = openloris(bench, Path(a.raw), a.seqs)
    else:
        seqs = simchange(bench, a.baseline, a.seqs)
    if a.write_imu:
        for s in seqs:
            write_imu_stream(s, a.write_imu)
        return
    if a.jobs > 1:
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(lambda s: process(s, a), seqs))
    else:
        for s in seqs:
            process(s, a)


if __name__ == "__main__":
    main()
