#!/usr/bin/env python3
"""Prepare the NCLT dataset (University of Michigan North Campus Long-Term Vision and Lidar Dataset, Carlevaris-Bianco,
Ushani and Eustice, IJRR 2016) for CROSS: a Segway driving indoors and outdoors on one campus, 27 sessions over 15
months, with consumer GPS, RTK GPS, an IMU with magnetometer (compass), wheel + fibre-optic-gyro odometry, ground
truth from lidar SLAM fused with RTK GPS, and a Ladybug3 omnidirectional camera (5 horizontal cameras + 1 up).

  python benchmark/datasets/nclt_download.py <raw> --kinds calib gt cov sensors           # small files, all sessions
  python benchmark/datasets/prepare_nclt.py sensors <raw> <prepared> [--sessions ...]      # standardised sensors
  python benchmark/datasets/prepare_nclt.py stats <prepared>                               # GPS / compass checks
  python benchmark/datasets/prepare_nclt.py images <prepared> --session S [--tar T | --stream] [--cams 1 2 3 4 5]
  python benchmark/datasets/prepare_nclt.py posed <prepared> --session S [--cam 5]         # CROSS posed folder

Conventions (NCLT paper, Sec. 3; checked against the RTK GPS and the magnetometer, see `stats`):
- local frame: x north, y east, z down (NED), linearised at lat 42.293227, lon -83.709657, alt 270 m (`to_local`);
- body frame: on the wheel axle, x forward, y right, z down; sensor extrinsics x_body,sensor (Table 4) in
  `EXTRINSICS`; 6-DOF vectors [x, y, z, roll, pitch, yaw] (deg) -> R = Rz(yaw) Ry(pitch) Rx(roll) (`ssc_to_T`);
- time: microseconds since the epoch in the raw files, seconds (float) in the prepared files.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np

LAT0, LON0, ALT0 = 42.293227, -83.709657, 270.0
R_EQ, R_POL = 6378135.0, 6356750.0
# x_body,sensor of the NCLT paper (Table 4): x y z (m), roll pitch yaw (deg); None angles = position only
EXTRINSICS = {
    "velodyne": [0.002, -0.004, -0.957, 0.807, 0.166, -90.703],
    "lb3": [0.035, 0.002, -1.23, -179.93, -0.23, 0.50],
    "imu": [-0.11, -0.18, -0.71, 0.0, 0.0, 0.0],       # Microstrain 3DM-GX3-45 ("ms25" files)
    "fog": [0.0, -0.25, -0.49, 0.0, 0.0, 0.0],
    "gps": [0.0, -0.25, -0.51, None, None, None],      # Garmin 18x 5 Hz (consumer)
    "rtk": [-0.24, 0.0, -1.24, None, None, None],      # NovAtel DL-4 plus
    "h30": [0.28, 0.0, -0.44, 180.0, 0.0, 0.0],
    "h04": [0.31, 0.0, -0.38, 180.0, -40.0, 0.0],
}
# forward-looking camera of the Ladybug3 (optical axis 0.7 deg from the body x axis; Cam1-4 at -71, -143, 145, 73 deg)
FORWARD_CAM = 5


def _local_radii(lat0_rad: float):
    d = (R_EQ * np.cos(lat0_rad)) ** 2 + (R_POL * np.sin(lat0_rad)) ** 2
    return (R_EQ * R_POL) ** 2 / d ** 1.5, R_EQ ** 2 / np.sqrt(d)


def to_local(lat_deg, lon_deg, alt_m=None):
    """WGS84 (deg, m) -> NCLT local frame (x north, y east, z down, metres), the paper's linearisation (Eq. 2)."""
    lat0, lon0 = np.radians(LAT0), np.radians(LON0)
    rns, rew = _local_radii(lat0)
    lat, lon = np.radians(np.asarray(lat_deg, float)), np.radians(np.asarray(lon_deg, float))
    z = ALT0 - np.asarray(alt_m, float) if alt_m is not None else np.zeros_like(lat)
    return np.stack([np.sin(lat - lat0) * rns, np.sin(lon - lon0) * rew * np.cos(lat0), z], -1)


def from_local(xyz):
    """NCLT local frame -> WGS84 (lat deg, lon deg, alt m) (Eq. 3)."""
    lat0, lon0 = np.radians(LAT0), np.radians(LON0)
    rns, rew = _local_radii(lat0)
    xyz = np.asarray(xyz, float)
    lat = np.arcsin(xyz[..., 0] / rns) + lat0
    lon = np.arcsin(xyz[..., 1] / (rew * np.cos(lat0))) + lon0
    return np.degrees(lat), np.degrees(lon), ALT0 - xyz[..., 2]


def ssc_to_T(x) -> np.ndarray:
    """[x, y, z, roll, pitch, yaw] (m, deg) -> 4x4, R = Rz(yaw) Ry(pitch) Rx(roll) (NCLT devkit ssc_to_homo)."""
    T = np.eye(4)
    T[:3, 3] = x[:3]
    if x[3] is not None:
        r, p, h = np.radians(x[3:6])
        cr, sr, cp, sp, ch, sh = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(h), np.sin(h)
        T[:3, :3] = [[ch * cp, -sh * cr + ch * sp * sr, sh * sr + ch * sp * cr],
                     [sh * cp, ch * cr + sh * sp * sr, -ch * sr + sh * sp * cr],
                     [-sp, cp * sr, cp * cr]]
    return T


def euler_to_R(roll, pitch, yaw) -> np.ndarray:
    """(N,) arrays of radians -> (N, 3, 3) with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, ch, sh = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    R = np.empty((len(roll), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = ch * cp, -sh * cr + ch * sp * sr, sh * sr + ch * sp * cr
    R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = sh * cp, ch * cr + sh * sp * sr, -ch * sr + sh * sp * cr
    R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = -sp, cp * sr, cp * cr
    return R


def R_to_quat(R) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()          # x y z w


def read_csv(path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",")


def camera_params(cam_zip: Path) -> dict:
    """K (1616x1232 stored image, after undistortion) and x_lb3,c of every Ladybug3 camera from cam_params.zip."""
    out = {}
    with zipfile.ZipFile(cam_zip) as z:
        names = {Path(n).name: n for n in z.namelist()}
        for c in range(6):
            K = np.loadtxt(io.StringIO(z.read(names[f"K_cam{c}.csv"]).decode()), delimiter=",")
            x = np.loadtxt(io.StringIO(z.read(names[f"x_lb3_c{c}.csv"]).decode()), delimiter=",")
            out[c] = {"K": K.tolist(), "x_lb3_c": x.tolist()}
    return out


def T_body_cam(cam: int, cams: dict) -> np.ndarray:
    """Pose of camera `cam` (stored, unrotated image: x down, y left for the horizontal cameras) in the body frame."""
    return ssc_to_T(EXTRINSICS["lb3"]) @ ssc_to_T(cams[cam]["x_lb3_c"])


# ---------------------------------------------------------------------------------------------------------- images
STORED_W, STORED_H = 1616, 1232          # Ladybug3 images as stored: rotated by 90 deg (image x = down)
OUT_W, OUT_H = 640, 480                   # prepared images: upright pinhole 640 x 480
CROP_HW, CROP_HH = 392, 294               # half size of the centred 4:3 crop of the upright undistorted image (88 x 71 deg)


def u2d_maps(u2d_dir: Path, cam: int):
    """NCLT undistortion map of a camera (U2D_Cam<k>_1616X1232.txt, from U2D_ALL_1616X1232.tar.gz): for every pixel of
    the undistorted (pinhole, K_cam<k>) stored image, its position in the distorted image; cached as .npz."""
    cache = u2d_dir / f"u2d_cam{cam}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["mapu"], d["mapv"]
    txt = next(u2d_dir.rglob(f"U2D_Cam{cam}_1616X1232.txt"))
    a = np.loadtxt(txt, skiprows=1, dtype=np.float32)
    mapu = np.zeros((STORED_H, STORED_W), np.float32)
    mapv = np.zeros((STORED_H, STORED_W), np.float32)
    r, c = a[:, 0].astype(int), a[:, 1].astype(int)
    mapu[r, c], mapv[r, c] = a[:, 3], a[:, 2]            # as the NCLT devkit (undistort.py)
    np.savez(cache, mapu=mapu, mapv=mapv)
    return mapu, mapv


class LB3Rectifier:
    """Raw Ladybug3 image of one camera -> upright undistorted pinhole image OUT_W x OUT_H (one remap + one resize).

    The stored image is undistorted with the NCLT U2D map (pinhole, K_cam<k>), turned upright (90 deg clockwise: the
    horizontal cameras are mounted on their side), cropped to a 4:3 window centred on the principal point that is valid
    for every camera (88 x 71 deg), and resized with area interpolation.  Upright camera frame = OpenCV (x right, y down,
    z forward); its pose in the body frame is `T_body_cam_upright`."""

    def __init__(self, u2d_dir: Path, cam: int, cams: dict):
        mapu, mapv = u2d_maps(u2d_dir, cam)
        K = np.asarray(cams[cam]["K"], float)
        f, cx_s, cy_s = K[0, 0], K[0, 2], K[1, 2]
        # upright image B (H' = STORED_W rows, W' = STORED_H cols): B[r, c] = A[STORED_H - 1 - c, r]
        cxu, cyu = (STORED_H - 1) - cy_s, cx_s
        self.x0, self.y0 = int(round(cxu)) - CROP_HW, int(round(cyu)) - CROP_HH
        c = np.arange(2 * CROP_HW)[None, :] + self.x0          # upright column
        r = np.arange(2 * CROP_HH)[:, None] + self.y0          # upright row
        su, sv = r + 0 * c, (STORED_H - 1) - c + 0 * r         # stored undistorted pixel (u, v)
        self.map_x = np.ascontiguousarray(mapu[sv, su])
        self.map_y = np.ascontiguousarray(mapv[sv, su])
        valid = (self.map_x >= 0) & (self.map_x <= STORED_W - 1) & (self.map_y >= 0) & (self.map_y <= STORED_H - 1)
        self.valid_fraction = float(valid.mean())
        s = OUT_W / (2 * CROP_HW)
        # pixel centres: x_out + 0.5 = s (x_crop + 0.5)
        self.K = np.array([[f * s, 0, (cxu - self.x0 + 0.5) * s - 0.5],
                           [0, f * s, (cyu - self.y0 + 0.5) * s - 0.5],
                           [0, 0, 1.0]])
        # upright axes in the stored camera frame: x' = -y, y' = x, z' = z
        R_stored_upright = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = R_stored_upright
        self.T_body_cam_upright = T_body_cam(cam, cams) @ T

    def __call__(self, raw_bgr: np.ndarray) -> np.ndarray:
        import cv2
        crop = cv2.remap(raw_bgr, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)


_RECT = {}


def _init_worker(u2d_dir, cams, out_dir, quality):
    global _RECT, _OUT, _Q
    _RECT = {c: LB3Rectifier(Path(u2d_dir), c, cams) for c in range(1, 6)}
    _OUT, _Q = Path(out_dir), quality


def _process_member(args):
    import cv2
    cam, utime, data = args
    raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if raw is None or raw.shape[:2] != (STORED_H, STORED_W):
        return cam, utime, False
    img = _RECT[cam](raw)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _Q])
    (_OUT / f"Cam{cam}" / f"{utime}.jpg").write_bytes(buf.tobytes())
    return cam, utime, True


def prepare_images(prep: Path, session: str, source, u2d_dir: Path, cams_used=(1, 2, 3, 4, 5), workers: int = 8,
                   quality: int = 95, connections: int = 8):
    """Stream the session's image archive (a local .tar.gz path, or None = straight from S3 with parallel range
    requests, nothing stored), rectify every frame of the chosen cameras with LB3Rectifier, write
    <prep>/<session>/lb3/Cam<k>/<utime>.jpg and lb3/frames.txt (cam utime ok)."""
    import multiprocessing as mp
    from nclt_download import BASE, RangeStream
    cams = {int(k): v for k, v in json.loads((prep / session / "geo_ref.json").read_text())["cameras"].items()}
    out = prep / session / "lb3"
    for c in cams_used:
        (out / f"Cam{c}").mkdir(parents=True, exist_ok=True)
    done = {(int(p.parent.name[3:]), int(p.stem)) for c in cams_used for p in (out / f"Cam{c}").glob("*.jpg")}
    rect = {c: LB3Rectifier(u2d_dir, c, cams) for c in cams_used}
    (out / "cameras.json").write_text(json.dumps({f"cam{c}": {"K": r.K.tolist(), "width": OUT_W, "height": OUT_H,
                                                             "T_body_cam": r.T_body_cam_upright.tolist(),
                                                             "crop_valid_fraction": r.valid_fraction}
                                                  for c, r in rect.items()}, indent=1))
    fileobj = RangeStream(f"{BASE}/images/{session}_lb3.tar.gz", connections=connections) if source is None else open(source, "rb")
    n_seen, n_ok, t0 = {}, 0, time.time()
    rows = []
    pool = mp.Pool(workers, initializer=_init_worker, initargs=(str(u2d_dir), cams, str(out), quality))

    def members():
        nonlocal n_ok
        with tarfile.open(fileobj=fileobj, mode="r|gz", bufsize=1 << 20) as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".tiff"):
                    continue
                cam, utime = int(Path(m.name).parent.name[3:]), int(Path(m.name).stem)
                n_seen[cam] = n_seen.get(cam, 0) + 1
                if cam not in cams_used or (cam, utime) in done:
                    continue
                yield cam, utime, tf.extractfile(m).read()

    pending = []
    for res in pool.imap_unordered(_process_member, members(), chunksize=4):
        rows.append(res)
        n_ok += res[2]
        if len(rows) % 5000 == 0:
            rate = fileobj.rate() if hasattr(fileobj, "rate") else float("nan")
            print(f"{session}: {len(rows)} frames ({n_seen}), {rate:.1f} MB/s, {time.time() - t0:.0f} s", flush=True)
    pool.close()
    pool.join()
    rows += [(c, u, True) for (c, u) in done]
    rows.sort()
    with open(out / "frames.txt", "w") as f:
        f.write("# cam utime_us ok\n")
        for c, u, ok in rows:
            f.write(f"{c} {u} {int(ok)}\n")
    print(f"{session}: done, {n_ok} new frames, per camera seen {n_seen}, {time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------------------------------------------- sensors
def _write(path: Path, header: str, rows: np.ndarray, fmt):
    np.savetxt(path, rows, fmt=fmt, header=header, comments="# ")


def prepare_sensors(raw: Path, out_root: Path, session: str, cams: dict):
    out = out_root / session
    out.mkdir(parents=True, exist_ok=True)
    tgz = raw / f"{session}_sen.tar.gz"
    files = {}
    with tarfile.open(tgz) as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".csv"):
                files[Path(m.name).name] = read_csv(tf.extractfile(m))
    us = 1e-6
    # consumer GPS (Garmin 18x): utime, msg (2 / 3: two NMEA sentences per fix, the 3 carries the altitude; NOT a fix
    # quality), num_sats (always 0: not reported), lat, lon (rad), alt (m, nan in the 2 rows), track (rad), speed (m/s)
    g = files["gps.csv"]
    _write(out / "gnss.txt", "t_s lat_deg lon_deg alt_m msg num_sats track_rad speed_mps  (Garmin 18x consumer GPS; "
           "msg 2/3 = NMEA sentence pair, not fix quality; num_sats not reported; no rows while there is no fix)",
           np.column_stack([g[:, 0] * us, np.degrees(g[:, 3]), np.degrees(g[:, 4]), g[:, 5], g[:, 1], g[:, 2], g[:, 6], g[:, 7]]),
           ["%.6f", "%.9f", "%.9f", "%.3f", "%d", "%d", "%.5f", "%.4f"])
    r = files["gps_rtk.csv"]
    e = files.get("gps_rtk_err.csv")
    err = np.interp(r[:, 0], e[:, 0], e[:, 1]) if e is not None and len(e) else np.full(len(r), np.nan)
    _write(out / "gnss_rtk.txt", "t_s lat_deg lon_deg alt_m mode num_sats track_rad speed_mps err_m  (NovAtel DL-4 "
           "plus; mode 3 = 3-D solution; err_m = receiver error estimate (gps_rtk_err.csv))",
           np.column_stack([r[:, 0] * us, np.degrees(r[:, 3]), np.degrees(r[:, 4]), r[:, 5], r[:, 1], r[:, 2], r[:, 6], r[:, 7], err]),
           ["%.6f", "%.9f", "%.9f", "%.3f", "%d", "%d", "%.5f", "%.4f", "%.3f"])
    m = files["ms25.csv"]
    _write(out / "imu.txt", "t_s mag_x mag_y mag_z (Gauss) acc_x acc_y acc_z (m/s^2, specific force) gyro_x gyro_y "
           "gyro_z (rad/s)  (Microstrain 3DM-GX3-45, axes = body axes: x fwd, y right, z down)",
           np.column_stack([m[:, 0] * us, m[:, 1:10]]), ["%.6f"] + ["%.7g"] * 9)
    a = files["ms25_euler.csv"]
    _write(out / "ahrs.txt", "t_s roll pitch heading (rad; the IMU's own magnetic AHRS, heading w.r.t. magnetic north)",
           np.column_stack([a[:, 0] * us, a[:, 1:4]]), ["%.6f"] + ["%.7g"] * 3)
    o = files["odometry_mu_100hz.csv"]
    Ro = euler_to_R(o[:, 4], o[:, 5], o[:, 6])
    _write(out / "odom.txt", "t_s x y z qx qy qz qw  (body pose in the odometry frame: wheel + FOG + IMU EKF, 100 Hz; "
           "starts at the identity)", np.column_stack([o[:, 0] * us, o[:, 1:4], R_to_quat(Ro)]), ["%.6f"] + ["%.6f"] * 3 + ["%.8f"] * 4)
    if "kvh.csv" in files:
        k = files["kvh.csv"]
        _write(out / "fog.txt", "t_s heading_rad (KVH single-axis FOG, integrated)", np.column_stack([k[:, 0] * us, k[:, 1]]), ["%.6f", "%.7g"])
    if "wheels.csv" in files:
        w = files["wheels.csv"]
        _write(out / "wheels.txt", "t_s v_left v_right (m/s)", np.column_stack([w[:, 0] * us, w[:, 1:3]]), ["%.6f", "%.5f", "%.5f"])
    gt = read_csv(raw / f"groundtruth_{session}.csv")
    gt = gt[np.all(np.isfinite(gt), axis=1)]
    Rg = euler_to_R(gt[:, 4], gt[:, 5], gt[:, 6])
    _write(out / "gt_body.txt", "t_s x y z qx qy qz qw  (body pose in the NCLT local frame: x north, y east, z down; "
           "lidar SLAM + RTK nodes every ~8 m, odometry in between)",
           np.column_stack([gt[:, 0] * us, gt[:, 1:4], R_to_quat(Rg)]), ["%.6f"] + ["%.5f"] * 3 + ["%.8f"] * 4)
    cov = raw / f"cov_{session}.csv"
    if cov.exists():
        c = read_csv(cov)
        _write(out / "gt_nodes.txt", "t_s  (times of the SLAM graph nodes = the precise ground-truth poses; the full "
               "covariance rows are in the raw cov_<session>.csv)", c[:, :1] * us, ["%.6f"])
    write_geo_ref(out / "geo_ref.json", cams)


def write_geo_ref(path: Path, cams: dict):
    T_bc = {f"cam{c}": T_body_cam(c, cams).tolist() for c in range(6)}
    ref = {
        "local_frame": "NED: x north, y east, z down (metres); linearised WGS84 (NCLT paper Eq. 2-3)",
        "origin": {"lat_deg": LAT0, "lon_deg": LON0, "alt_m": ALT0},
        "earth_radii_m": {"equatorial": R_EQ, "polar": R_POL},
        "to_local": "x = sin(lat - lat0) r_ns, y = sin(lon - lon0) r_ew cos(lat0), z = alt0 - alt",
        "body_frame": "wheel axle centre; x forward, y right, z down",
        "euler_convention": "R = Rz(yaw) Ry(pitch) Rx(roll); 6-DOF vectors [x y z roll pitch yaw] (m, deg)",
        "x_body_sensor": EXTRINSICS,
        "T_body_cam_stored": T_bc,
        "stored_camera_axes": "horizontal cameras as stored (1616 x 1232): image x = body down, image y = body left",
        "cameras": cams,
        "forward_cam": FORWARD_CAM,
        "magnetic_declination_deg": "about -6.9 (Ann Arbor, 2012); measured offset of the magnetometer heading "
                                    "atan2(-m_y, m_x) (levelled) minus GT yaw: see stats.json",
    }
    path.write_text(json.dumps(ref, indent=1))


# ----------------------------------------------------------------------------------------------------------- stats
def _interp_gt(gt, t):
    """GT position (linear) and rotation (nearest) at times t; ok = inside the GT time span."""
    from scipy.spatial.transform import Rotation, Slerp
    ok = (t >= gt[0, 0]) & (t <= gt[-1, 0])
    tc = np.clip(t, gt[0, 0], gt[-1, 0])
    p = np.stack([np.interp(tc, gt[:, 0], gt[:, k]) for k in (1, 2, 3)], -1)
    R = Slerp(gt[:, 0], Rotation.from_quat(gt[:, 4:8]))(tc).as_matrix()
    return p, R, ok


def session_stats(prep: Path, session: str) -> dict:
    d = prep / session
    gt = np.loadtxt(d / "gt_body.txt")
    g = np.loadtxt(d / "gnss.txt")
    r = np.loadtxt(d / "gnss_rtk.txt")
    imu = np.loadtxt(d / "imu.txt")
    odom = np.loadtxt(d / "odom.txt")
    st = {"session": session}
    st["duration_s"] = float(odom[-1, 0] - odom[0, 0])
    st["gt_span_s"] = float(gt[-1, 0] - gt[0, 0])
    gaps = np.diff(gt[:, 0])
    st["gt_coverage"] = float(1 - gaps[gaps > 1.0].sum() / max(st["duration_s"], 1e-9))
    st["path_length_m"] = float(np.linalg.norm(np.diff(gt[:, 1:3], axis=0), axis=1).sum())
    # GPS vs GT (lever arms of Table 4)
    out = {}
    for name, arr, lever in (("gps", g, np.array(EXTRINSICS["gps"][:3])), ("rtk", r, np.array(EXTRINSICS["rtk"][:3]))):
        pl = to_local(arr[:, 1], arr[:, 2], np.where(np.isfinite(arr[:, 3]), arr[:, 3], np.nan))
        p, R, ok = _interp_gt(gt, arr[:, 0])
        e = pl[:, :2] - (p + R @ lever)[:, :2]
        eh = np.linalg.norm(e, axis=1)
        res = {"n": int(ok.sum()), "err_h_m": {f"p{q}": float(np.percentile(eh[ok], q)) for q in (50, 90, 95, 99)},
               "mean_e_ne_m": e[ok].mean(0).round(3).tolist()}
        if name == "rtk":
            for mode in (2, 3):
                s = ok & (arr[:, 4] == mode)
                if s.sum():
                    res[f"mode{mode}"] = {"n": int(s.sum()), "p50": float(np.percentile(eh[s], 50)), "p90": float(np.percentile(eh[s], 90))}
            for lo, hi in ((0, 4), (4, 6), (6, 8), (8, 99)):
                s = ok & (arr[:, 5] >= lo) & (arr[:, 5] < hi)
                if s.sum():
                    res[f"sats{lo}-{hi - 1}"] = {"n": int(s.sum()), "p50": float(np.percentile(eh[s], 50)), "p90": float(np.percentile(eh[s], 90))}
            res["err_est_vs_true_corr"] = float(np.corrcoef(np.nan_to_num(arr[ok, 8]), eh[ok])[0, 1])
        else:
            # error vs time since the fix was (re)acquired after a gap > 2 s
            tg = arr[:, 0]
            starts = np.r_[0, np.where(np.diff(tg) > 2.0)[0] + 1]
            since = tg - tg[starts[np.searchsorted(starts, np.arange(len(tg)), side="right") - 1]]
            for lo, hi in ((0, 5), (5, 30), (30, 1e9)):
                s = ok & (since >= lo) & (since < hi)
                if s.sum():
                    res[f"since_reacq_{lo}-{int(min(hi, 9999))}s"] = {"n": int(s.sum()), "p50": float(np.percentile(eh[s], 50)), "p90": float(np.percentile(eh[s], 90))}
        out[name] = res
    st.update(out)
    # consumer GPS availability: no fix = no rows; intervals without a fix for > 2 s (inside the GT span)
    tg = g[:, 0]
    tg = tg[(tg >= gt[0, 0]) & (tg <= gt[-1, 0])]
    edges = np.r_[gt[0, 0], tg, gt[-1, 0]]
    gaps = np.diff(edges)
    nofix = gaps > 2.0
    st["gps_nofix_fraction"] = float(gaps[nofix].sum() / max(st["gt_span_s"], 1e-9))
    st["gps_nofix_intervals"] = int(nofix.sum())
    st["gps_nofix_intervals_gt10s"] = int((gaps > 10).sum())
    st["gps_nofix_longest_s"] = float(gaps.max())
    st["gps_rate_hz"] = float(len(tg) / max(st["gt_span_s"], 1e-9))
    # compass: levelled magnetometer heading vs GT yaw, with / without a GPS fix within 1 s
    t = imu[::5, 0]
    mag = imu[::5, 1:4]
    p, R, ok = _interp_gt(gt, t)
    t, mag, R = t[ok], mag[ok], R[ok]
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    pitch = -np.arcsin(np.clip(R[:, 2, 0], -1, 1))
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    Rl = euler_to_R(roll, pitch, np.zeros_like(roll))          # level <- body
    ml = np.einsum("nij,nj->ni", Rl, mag)
    hm = np.arctan2(-ml[:, 1], ml[:, 0])
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    dd = wrap(hm - yaw)
    gi = np.clip(np.searchsorted(g[:, 0], t), 1, len(g) - 1)
    fix = np.minimum(np.abs(g[gi, 0] - t), np.abs(g[gi - 1, 0] - t)) < 1.0
    off = float(np.median(dd[fix]))
    nrm = np.linalg.norm(mag, axis=1)
    st["compass"] = {"offset_deg": float(np.degrees(off))}
    for lab, s in (("with_fix", fix), ("no_fix", ~fix)):
        if s.sum() > 10:
            ae = np.degrees(np.abs(wrap(dd[s] - off)))
            st["compass"][lab] = {"frac": float(s.mean()), "abs_err_deg_p50": float(np.percentile(ae, 50)),
                                  "abs_err_deg_p90": float(np.percentile(ae, 90)),
                                  "field_G_p5_p50_p95": np.percentile(nrm[s], [5, 50, 95]).round(3).tolist()}
    # odometry drift: horizontal ATE of the odometry track aligned (SE2) to GT at the start, per 100 m
    po, _, oko = _interp_gt(gt, odom[::100, 0])
    st["odom_vs_gt_end_err_m"] = None
    try:
        tt = odom[::100, 0][oko]
        q = odom[::100, 1:3][oko]
        pg = po[oko][:, :2]
        n0 = min(50, len(q) - 1)
        A = q[:n0] - q[:n0].mean(0)
        B = pg[:n0] - pg[:n0].mean(0)
        U, _, Vt = np.linalg.svd(A.T @ B)
        Rz = (U @ Vt).T
        al = (q - q[:n0].mean(0)) @ Rz.T + pg[:n0].mean(0)
        st["odom_vs_gt_end_err_m"] = float(np.linalg.norm(al[-1] - pg[-1]))
        st["odom_drift_pct"] = float(100 * np.linalg.norm(al - pg, axis=1).max() / st["path_length_m"])
    except Exception:
        pass
    return st


def cmd_stats(prep: Path, sessions):
    allst = []
    for s in sessions:
        if not (prep / s / "gt_body.txt").exists():
            continue
        st = session_stats(prep, s)
        (prep / s / "stats.json").write_text(json.dumps(st, indent=1))
        allst.append(st)
        print(f"{s}: {st['duration_s'] / 60:5.1f} min {st['path_length_m'] / 1000:5.2f} km GTcov {st['gt_coverage']:.3f} "
              f"gps p50/p90 {st['gps']['err_h_m']['p50']:.1f}/{st['gps']['err_h_m']['p90']:.1f} m rtk p50 "
              f"{st['rtk']['err_h_m']['p50']:.2f} nofix {st['gps_nofix_fraction']:.3f} ({st['gps_nofix_intervals_gt10s']} gaps>10s) "
              f"compass p50 {st['compass'].get('with_fix', {}).get('abs_err_deg_p50', float('nan')):.1f}/"
              f"{st['compass'].get('no_fix', {}).get('abs_err_deg_p50', float('nan')):.1f} deg", flush=True)
    (prep / "stats_all.json").write_text(json.dumps(allst, indent=1))


def main():
    from nclt_download import SESSIONS  # noqa: E402  (same folder)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("sensors")
    s1.add_argument("raw", type=Path)
    s1.add_argument("prepared", type=Path)
    s1.add_argument("--sessions", nargs="+", default=SESSIONS)
    s2 = sub.add_parser("stats")
    s2.add_argument("prepared", type=Path)
    s2.add_argument("--sessions", nargs="+", default=SESSIONS)
    s3 = sub.add_parser("images")
    s3.add_argument("prepared", type=Path)
    s3.add_argument("--session", required=True)
    s3.add_argument("--tar", type=Path, default=None, help="local <session>_lb3.tar.gz (default: stream from S3)")
    s3.add_argument("--u2d", type=Path, required=True, help="folder with U2D_Cam<k>_1616X1232.txt (or cached .npz)")
    s3.add_argument("--cams", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    s3.add_argument("--workers", type=int, default=8)
    s3.add_argument("--connections", type=int, default=8)
    s3.add_argument("--quality", type=int, default=95)
    a = ap.parse_args()
    if a.cmd == "images":
        prepare_images(a.prepared, a.session, a.tar, a.u2d, tuple(a.cams), a.workers, a.quality, a.connections)
    elif a.cmd == "sensors":
        cams = camera_params(a.raw / "cam_params.zip")
        for s in a.sessions:
            if not (a.raw / f"{s}_sen.tar.gz").exists() or not (a.raw / f"groundtruth_{s}.csv").exists():
                print(f"{s}: raw files missing, skipped", flush=True)
                continue
            t0 = time.time()
            prepare_sensors(a.raw, a.prepared, s, cams)
            print(f"{s}: sensors prepared ({time.time() - t0:.0f} s)", flush=True)
    elif a.cmd == "stats":
        cmd_stats(a.prepared, a.sessions)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
