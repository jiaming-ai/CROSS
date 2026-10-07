"""Basalt's stereo-inertial VIO (Usenko et al., RA-L 2020) running live on the edge: the basalt_live program
(edge/native/basalt_live.cpp, built by edge/native/install_basalt_live.sh) fed with the IMU samples and the stereo
frames as they come, its states read back per frame.

The calibration and configuration are those of the benchmark's recorded Basalt odometry (CROSS
benchmark/datasets/prepare_vio.py: odom_vio.txt): pinhole intrinsics of the rectified pair, the IMU-camera transform
and noise densities of the IMU stream, Basalt's EuRoC configuration; the camera pose of a frame is Basalt's IMU pose
at the frame's time moved to the left camera, T_w_cam = T_w_imu T_imu_cam.  Basalt processes a frame once an IMU
sample after its time has arrived, so the IMU is fed one sample ahead of each frame."""

import json
import os
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .geometry import matrix_from_quat, quat_from_matrix

DEFAULT_BINARY = os.environ.get("BASALT_LIVE", "basalt_live")


def _pose(T):
    q = quat_from_matrix(T[:3, :3])
    return {"px": float(T[0, 3]), "py": float(T[1, 3]), "pz": float(T[2, 3]),
            "qx": float(q[0]), "qy": float(q[1]), "qz": float(q[2]), "qw": float(q[3])}


def basalt_calibration(K, size, baseline, T_cam_imu, noise, imu_rate_hz) -> dict:
    """Basalt's camera + IMU calibration (cereal JSON) of a rectified stereo pair (right camera at +baseline along x)
    and its IMU: as benchmark/datasets/prepare_vio.py basalt_calib."""
    T_imu_c0 = np.linalg.inv(np.asarray(T_cam_imu, dtype=np.float64))
    T_c0_c1 = np.eye(4)
    T_c0_c1[0, 3] = float(baseline)
    K = np.asarray(K, dtype=np.float64)
    intr = {"camera_type": "pinhole", "intrinsics": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}}
    return {"value0": {
        "T_imu_cam": [_pose(T_imu_c0), _pose(T_imu_c0 @ T_c0_c1)],
        "intrinsics": [intr, dict(intr)],
        "resolution": [list(size), list(size)],
        "vignette": [{"value0": 0, "value1": 50000000000, "value2": [[1.0]] * 15} for _ in range(2)],
        "calib_accel_bias": [0.0] * 9, "calib_gyro_bias": [0.0] * 12,
        "imu_update_rate": float(imu_rate_hz),
        # Basalt's *_noise_std / *_bias_std are the continuous-time densities
        "accel_noise_std": [noise["accel_noise_density"]] * 3,
        "gyro_noise_std": [noise["gyro_noise_density"]] * 3,
        "accel_bias_std": [noise["accel_random_walk"]] * 3,
        "gyro_bias_std": [noise["gyro_random_walk"]] * 3,
        "T_mocap_world": _pose(np.eye(4)), "T_imu_marker": _pose(np.eye(4)),
        "mocap_time_offset_ns": 0, "mocap_to_imu_offset_ns": 0, "cam_time_offset_ns": 0}}


def _ns(t):
    return int(round(float(t) * 1e9))


class BasaltOdometry:
    """Live Basalt for a sensor source with an IMU stream (cross_edge.sensors.StereoFolder or a robot driver giving
    the same fields: K, width, height, baseline, imu (N x 7: t gyro accel), imu_calib, frames with t_imu and gray /
    gray_right images).

    binary    basalt_live (default $BASALT_LIVE)
    config    Basalt configuration JSON (default: Basalt's data/euroc_config.json next to the binary's source tree)
    threads   Basalt's worker threads (TBB)
    wait      seconds to wait for a frame's state before reporting no estimate for it"""

    def __init__(self, source, binary=None, config=None, threads=4, wait=5.0, overrides=None, log=None):
        if source.imu is None:
            raise ValueError("Basalt needs the IMU stream of the stereo folder (imu_vio.txt / imu.txt + .json)")
        self.source, self.wait = source, float(wait)
        self.binary = str(binary or DEFAULT_BINARY)
        cal = source.imu_calib
        self.T_imu_cam = np.linalg.inv(np.asarray(cal["T_cam_imu"], dtype=np.float64))
        rate = float(1.0 / np.median(np.diff(source.imu[:, 0])))
        noise = {k: float(cal[k]) for k in ("gyro_noise_density", "accel_noise_density", "gyro_random_walk",
                                             "accel_random_walk")}
        self.work = Path(tempfile.mkdtemp(prefix="cross_edge_basalt_"))
        calib = self.work / "calib.json"
        calib.write_text(json.dumps(basalt_calibration(source.K, (source.width, source.height), source.baseline,
                                                       cal["T_cam_imu"], noise, rate), indent=1))
        if config is None:
            config = Path(self.binary if os.sep in self.binary else _which(self.binary)).resolve().parents[2] / "data" / "euroc_config.json"
        cfg = json.loads(Path(config).read_text())
        cfg["value0"].update({f"config.{k}": v for k, v in (overrides or {}).items()})
        cfg_path = self.work / "config.json"
        cfg_path.write_text(json.dumps(cfg, indent=1))
        self.log = open(log or (self.work / "basalt_live.log"), "w")
        self.proc = subprocess.Popen([self.binary, "--cam-calib", str(calib), "--config-path", str(cfg_path),
                                      "--num-threads", str(int(threads))],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self._states = {}                        # t_ns -> (T_w_i, v_w)
        self._cv = threading.Condition()
        self._done = False
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self._imu_next = 0                       # the next IMU sample to send
        self.stats = {"frames": 0, "estimates": 0, "missing": 0, "read_s": 0.0, "send_s": 0.0, "wait_s": 0.0}

    def _read(self):
        for line in self.proc.stdout:
            f = line.split()
            if not f or f[0] != b"S":
                continue
            v = [float(x) for x in f[2:]]
            T = np.eye(4)
            T[:3, :3] = matrix_from_quat(v[3:7])
            T[:3, 3] = v[0:3]
            with self._cv:
                self._states[int(f[1])] = (T, np.asarray(v[7:10]))
                self._cv.notify_all()
        with self._cv:
            self._done = True
            self._cv.notify_all()

    def _send_imu_until(self, t_ns_frame):
        """The IMU samples up to the frame's time and the first one after it (Basalt integrates up to the frame)."""
        imu, out = self.source.imu, []
        past = False
        while self._imu_next < len(imu):
            t = _ns(imu[self._imu_next, 0])
            out.append(struct.pack("<cq6d", b"I", t, *imu[self._imu_next, 1:7]))
            self._imu_next += 1
            if t > t_ns_frame:
                past = True
                break
        if out:
            self.proc.stdin.write(b"".join(out))
        return past

    def pose(self, frame):
        """The left camera's pose (c2w) at this frame in Basalt's world frame, or None (no estimate yet / in time)."""
        t_ns = _ns(frame["t_imu"])
        t_in = time.monotonic()
        gl, gr = frame["gray"], frame["gray_right"]
        t_read = time.monotonic()
        h, w = gl.shape
        self.proc.stdin.write(struct.pack("<cqII", b"F", t_ns, w, h) + gl.tobytes() + gr.tobytes())
        past = self._send_imu_until(t_ns)
        self.proc.stdin.flush()
        self.stats["frames"] += 1
        t0 = time.monotonic()
        self.stats["read_s"] += t_read - t_in          # the images (a replay: PNG decoding)
        self.stats["send_s"] += t0 - t_read
        # without an IMU sample after the frame (the end of a recording) Basalt cannot process it: do not wait long
        wait = self.wait if past else min(self.wait, 0.2)
        with self._cv:
            while t_ns not in self._states and not self._done and time.monotonic() - t0 < wait:
                self._cv.wait(timeout=wait)
            got = self._states.pop(t_ns, None)
            for k in [k for k in self._states if k < t_ns]:
                del self._states[k]
        self.stats["wait_s"] += time.monotonic() - t0
        if got is None:
            self.stats["missing"] += 1
            return None
        self.stats["estimates"] += 1
        return got[0] @ self.T_imu_cam

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(b"E")
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            self.proc.wait(timeout=60)
        self.log.close()


def _which(name):
    import shutil
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"{name} not found: build it with edge/native/install_basalt_live.sh and set $BASALT_LIVE")
    return path
