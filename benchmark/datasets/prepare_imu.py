#!/usr/bin/env python3
"""Write the IMU stream (imu.txt, imu.json; format in cross/dataloader/imu.py) of prepared benchmark sequences.

The IMU is the one rigidly attached to the camera of the monocular setup, expressed in its own frame, with its pose in
that camera's frame (T_cam_imu) and its noise densities:

  openloris  D435i IMU (gyroscope 400 Hz, accelerometer 250 Hz interpolated to the gyroscope times; the factory
             intrinsics of sensors.yaml applied: corrected = S raw - b), camera = D435i colour (rgbd folder)
  rover      D435i IMU (realsense_D435i/imu/imu.txt in the recording's zip), camera = D435i colour (rgbd folder);
             extrinsics and noise from calibration/calib_d435i.yaml (Kalibr).  The dusk recording has no D435i IMU:
             the VN-100 (vn100/imu.txt; VN100-To-Cam of calib_d435i.yaml, noise of calib_vn100.yaml) instead
  kitti      OXTS RT3003 accelerations and angular rates in the vehicle body frame (ax ay az, wx wy wz, 10 Hz in the
             synced drives; gravity included), camera = cam 2 (stereo folder); no velocities and no GPS.  The OXTS
             timestamps jitter by +-5 ms (times.txt holds them) while the camera runs at a regular 10 Hz: the frames'
             times for the IMU are the camera's own (image_02/timestamps.txt -> frame_times.txt; with the OXTS times
             the scale estimate collapsed to 0.2-0.4 of the true one on ground-truth poses)
  simchange  simulated: the 10 Hz ground truth is interpolated by a C2 spline and differentiated at 200 Hz, with the
             noise and bias of a BMI055 (the D435i IMU; ROVER's Kalibr values) and a seed derived from the sequence name

  python benchmark/datasets/prepare_imu.py openloris /path/to/openloris $BENCH_DATA/openloris [--seqs office1-1 ...]
  python benchmark/datasets/prepare_imu.py rover     /path/to/rover     $BENCH_DATA/rover
  python benchmark/datasets/prepare_imu.py kitti     /path/to/kitti_raw $BENCH_DATA/kitti
  python benchmark/datasets/prepare_imu.py simchange -                  $BENCH_DATA/simchange
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cross.dataloader.imu import write_imu  # noqa: E402
from cross.imu.simulate import BMI055, simulate_imu  # noqa: E402


def _frames(out: Path):
    return np.loadtxt(out / "times.txt", dtype=np.float64).reshape(-1)


def _clip(t, *arrays, t_frames, margin=1.0):
    keep = (t >= t_frames[0] - margin) & (t <= t_frames[-1] + margin)
    return (t[keep],) + tuple(a[keep] for a in arrays)


# ---------------------------------------------------------------------------------------------------- OpenLORIS
def _openloris_intrinsic(seq: Path, sensor: str):
    import cv2
    fs = cv2.FileStorage(str(seq / "sensors.yaml"), cv2.FILE_STORAGE_READ)
    m = np.asarray(fs.getNode(sensor).getNode("imu_intrinsic").mat(), dtype=np.float64).reshape(3, 4)
    return m[:, :3], m[:, 3]


def openloris(raw: Path, bench: Path, seqs):
    sys.path.insert(0, str(ROOT / "benchmark" / "datasets"))
    from prepare_openloris import read_child_extrinsic
    seqs = seqs or sorted(p.name for p in bench.iterdir() if (p / "rgbd" / "times.txt").is_file())
    for name in seqs:
        out = bench / name / "rgbd"
        src = raw / name.split("_f")[0] if not (raw / name).is_dir() else raw / name
        if not (out / "times.txt").is_file() or not (src / "d400_gyroscope.txt").is_file():
            print(f"{name}: no prepared rgbd folder or no D435i IMU, skipped")
            continue
        gyro = np.loadtxt(src / "d400_gyroscope.txt", comments="#", dtype=np.float64)
        acc = np.loadtxt(src / "d400_accelerometer.txt", comments="#", dtype=np.float64)
        S_g, b_g = _openloris_intrinsic(src, "d400_gyroscope")
        S_a, b_a = _openloris_intrinsic(src, "d400_accelerometer")
        tg = gyro[:, 0]
        a_raw = np.stack([np.interp(tg, acc[:, 0], acc[:, k]) for k in (1, 2, 3)], axis=1)
        inside = (tg >= acc[0, 0]) & (tg <= acc[-1, 0])
        w = gyro[:, 1:4] @ S_g.T - b_g
        a = a_raw @ S_a.T - b_a
        t, w, a = _clip(tg[inside], w[inside], a[inside], t_frames=_frames(out))
        T_cam_imu = read_child_extrinsic(src, "d400_color_optical_frame", "d400_accelerometer")
        write_imu(out, t, w, a, {"T_cam_imu": T_cam_imu.tolist(), **BMI055, "rate_hz": 400.0,
                                 "source": f"OpenLORIS {src.name} D435i IMU (factory intrinsics applied)"})
        print(f"{name}: {len(t)} samples, |a| median {np.median(np.linalg.norm(a, axis=1)):.3f}", flush=True)


# ---------------------------------------------------------------------------------------------------- ROVER
def rover(raw: Path, bench: Path, names):
    import yaml
    cal = yaml.safe_load((raw / "calibration/calib_d435i.yaml").read_text())
    T_cam_imu = np.asarray(cal["IMU-To-Cam"], dtype=np.float64)
    noise = cal["IMU_Intrinsics"]
    names = names or sorted(p.name for p in bench.iterdir() if (p / "rgbd" / "times.txt").is_file())
    for name in names:
        out = bench / name / "rgbd"
        if not (raw / f"{name}.zip").is_file() or not (out / "times.txt").is_file():
            print(f"{name}: no zip or no prepared rgbd folder, skipped")
            continue
        z = zipfile.ZipFile(raw / f"{name}.zip")
        top = z.namelist()[0].split("/")[0]
        member = f"{top}/realsense_D435i/imu/imu.txt"
        if member in set(z.namelist()):
            rows = [l.split(",") for l in z.read(member).decode().splitlines() if l.strip() and not l.startswith("#")]
            d = np.asarray(rows, dtype=np.float64)              # t, ax, ay, az, wx, wy, wz
            t, w, a = _clip(d[:, 0], d[:, 4:7], d[:, 1:4], t_frames=_frames(out))
            T, nz, src = T_cam_imu, noise, "realsense_D435i/imu (Kalibr calib_d435i.yaml)"
        else:
            # VN-100 rows: t_start t_end ax ay az wx wy wz mag(3) yaw pitch roll; time = end of the sample
            rows = [l.split() for l in z.read(f"{top}/vn100/imu.txt").decode().splitlines()
                    if l.strip() and not l.startswith("#")]
            d = np.asarray(rows, dtype=np.float64)
            t, w, a = _clip(d[:, 1], d[:, 5:8], d[:, 2:5], t_frames=_frames(out))
            T = np.asarray(cal["VN100-To-Cam"], dtype=np.float64)
            nz = yaml.safe_load((raw / "calibration/calib_vn100.yaml").read_text())["Intrinsics"]
            src = "vn100 (no D435i IMU in this recording; VN100-To-Cam of calib_d435i.yaml, calib_vn100.yaml)"
        write_imu(out, t, w, a, {"T_cam_imu": T.tolist(), "gyro_noise_density": nz["noise_gyro"],
                                 "accel_noise_density": nz["noise_acc"], "gyro_random_walk": nz["walk_gyro"],
                                 "accel_random_walk": nz["walk_acc"], "rate_hz": float(1 / np.median(np.diff(t))),
                                 "source": f"ROVER {top} {src}"})
        print(f"{name}: {len(t)} samples ({1 / np.median(np.diff(t)):.0f} Hz), "
              f"|a| median {np.median(np.linalg.norm(a, axis=1)):.3f}", flush=True)


# ---------------------------------------------------------------------------------------------------- KITTI
def kitti(raw: Path, bench: Path, seqs):
    sys.path.insert(0, str(ROOT / "benchmark" / "datasets"))
    from prepare_kitti import SEQUENCES
    from cross.dataloader.kitti_calib import load_kitti_calibration
    for s in seqs or list(SEQUENCES):
        date, drive, lo, hi = SEQUENCES[s]
        d = raw / date / f"{drive}_sync" / "oxts"
        out = bench / s / "stereo"
        if not d.is_dir() or not (out / "times.txt").is_file():
            print(f"{s}: no drive or no prepared folder, skipped")
            continue
        stamps = np.asarray([datetime.strptime(x.strip()[:26], "%Y-%m-%d %H:%M:%S.%f").timestamp()
                             for x in (d / "timestamps.txt").read_text().splitlines()])
        files = sorted((d / "data").glob("*.txt"))
        o = np.stack([np.loadtxt(files[i]) for i in range(lo, hi + 1)])
        t = stamps[lo:hi + 1]
        cam = [datetime.strptime(x.strip()[:26], "%Y-%m-%d %H:%M:%S.%f").timestamp()
               for x in (d.parent / "image_02" / "timestamps.txt").read_text().splitlines()]
        np.savetxt(out / "frame_times.txt", np.asarray(cam[lo:hi + 1]), fmt="%.6f")
        calib = load_kitti_calibration(raw / date)
        # OXTS: 11-13 ax ay az (body frame x forward, y left, z up; specific force incl. gravity), 17-19 wx wy wz
        write_imu(out, t, o[:, 17:20], o[:, 11:14], {
            "T_cam_imu": np.asarray(calib["cam2_from_imu"]).tolist(),
            # 10 Hz point samples of a navigation-grade unit: the noise is the discretization of the vehicle's motion
            # between samples, not the sensor's (an effective density for 0.1 s intervals)
            "gyro_noise_density": 0.005, "accel_noise_density": 0.1, "gyro_random_walk": 1e-5,
            "accel_random_walk": 1e-3, "rate_hz": 10.0, "frame_times": "frame_times.txt",
            "source": f"KITTI {drive}_sync oxts ax ay az wx wy wz (OXTS RT3003, 10 Hz)"})
        print(f"{s}: {len(t)} samples, |a| median {np.median(np.linalg.norm(o[:, 11:14], axis=1)):.3f}", flush=True)


# ---------------------------------------------------------------------------------------------------- SimChange
def simchange(bench: Path, seqs):
    folders = [bench / s for s in seqs] if seqs else sorted(p.parent for p in bench.glob("*/*/poses_left.txt"))
    for out in folders:
        name = f"{out.parent.name}/{out.name}"
        calib = json.loads((out / "calib.json").read_text())
        fps = float(calib.get("fps", 10.0))
        poses = np.loadtxt(out / "poses_left.txt", dtype=np.float64).reshape(-1, 4, 4)
        seed = zlib.crc32(name.encode())
        t, w, a = simulate_imu(poses, fps, seed)
        write_imu(out, t, w, a, {"T_cam_imu": np.eye(4).tolist(), **BMI055, "rate_hz": 200.0,
                                 "source": f"simulated from the ground truth (C2 spline, BMI055 noise, seed {seed})"})
        print(f"{name}: {len(t)} samples", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["openloris", "rover", "kitti", "simchange"])
    ap.add_argument("raw", help="raw dataset root ('-' for simchange)")
    ap.add_argument("bench", help="prepared benchmark root of the dataset ($BENCH_DATA/<dataset>)")
    ap.add_argument("--seqs", nargs="*", default=None)
    a = ap.parse_args()
    raw, bench = Path(a.raw), Path(a.bench)
    if a.dataset == "openloris":
        openloris(raw, bench, a.seqs)
    elif a.dataset == "rover":
        rover(raw, bench, a.seqs)
    elif a.dataset == "kitti":
        kitti(raw, bench, a.seqs)
    else:
        simchange(bench, a.seqs)


if __name__ == "__main__":
    main()
