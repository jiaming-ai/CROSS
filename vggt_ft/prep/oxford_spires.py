"""Oxford Spires (Tao et al., IJRR 2025; HF ori-drs/oxford_spires_dataset, CC BY-NC-SA 4.0) -> scene cache.

Handheld multi-sensor unit around Oxford landmarks (colleges, gardens, a palace, a library): three global-shutter colour
cameras (equidistant fisheye, 1440x1080), a 64-beam LiDAR and a millimetre-accurate terrestrial LiDAR map.  Per processed
sequence this uses
- colmap/images.zip: the cameras' images (one every ~0.3 s), named <timestamp>.jpg per camera folder;
- lidar-depths/depths_euc.zip: LiDAR depth of each image (uint16 PNG, metres x 256, Euclidean range on the raw fisheye
  grid, from the dataset's scripts/generate_depth.py with hidden-point removal);
- colmap/transforms_colmap_scaled.json: per-image camera-to-world poses (nerfstudio / OpenGL axes) of COLMAP, scaled to
  metres by the dataset (evo alignment to the ground-truth trajectory), and the fisheye intrinsics.
Images are undistorted to a pinhole camera (default 768x576, 90 deg horizontal field of view); every LiDAR pixel is
unprojected through the fisheye model and re-projected into the pinhole camera with a z-buffer (range -> z exactly).
Poses (--poses gt, default): the ground-truth trajectory (gt-tum.txt: the device base in the site's terrestrial-scanner
map, from LiDAR registration) + the camera calibration, so all sequences of a site share one world (cross-session
windows); --poses colmap: COLMAP scaled to metres (per-sequence world).  One scene per sequence, world = site.

python -m vggt_ft.prep.oxford_spires <sequence processed dir> <root> [--hfov 90] [--size 768 576] [--every 1]
"""
import argparse
import io
import json
import math
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np

from vggt_ft.prep.common import SceneWriter

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

# configs/sensor.yaml of github.com/ori-drs/oxford_spires_dataset (the calibration the depth images were made with):
# camera-from-LiDAR (t_xyz, q_xyzw) per camera and base-from-LiDAR
T_CAM_LIDAR = {
    "cam0": [0.00035060884033447846, -0.079574053851818, -0.053826061545697405,
             0.5034725628949727, 0.5003337192107279, -0.4947539264483171, 0.5013981452864594],
    "cam1": [3.388762063794015e-05, -0.07994358595369325, -0.05600121569499821,
             0.005222551021921984, 0.7078744176660065, -0.7063115244548896, 0.0032502610734581453],
    "cam2": [-0.0006070783260079121, -0.07992015242752262, -0.054798558347406996,
             0.7066174394729171, -0.001174547005383393, 0.0015977924498378243, 0.7075930057111628],
}
T_BASE_LIDAR = [0.0, 0.0, 0.124, 0.0, 0.0, 1.0, 0.0]


def se3(t_q):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(t_q[3:7]).as_matrix()
    T[:3, 3] = t_q[:3]
    return T


def gt_poses(tum_path):
    """Ground-truth base poses in the site's terrestrial-scanner map: (timestamps, world-from-base 4x4)."""
    g = np.loadtxt(tum_path)
    return g[:, 0], np.stack([se3(r[1:8]) for r in g])


def gt_cam_from_world(ts, gt, cam, max_dt=0.025):
    """Camera-from-world from the ground-truth base pose nearest in time (None when none within max_dt)."""
    gt_t, gt_T = gt
    i = int(np.argmin(np.abs(gt_t - ts)))
    if abs(gt_t[i] - ts) > max_dt:
        return None
    cam_from_base = se3(T_CAM_LIDAR[cam]) @ np.linalg.inv(se3(T_BASE_LIDAR))
    return cam_from_base @ np.linalg.inv(gt_T[i])


def frame_key(path: str) -> str:
    """'images/alphasense_driver_ros_cam2_debayered_image_compressed/1710255933.69.jpg' or 'depths_euc/cam2/...png'
    -> 'cam2/1710255933.69' (the depth archive names cameras cam0..cam2)."""
    m = re.search(r"cam(\d)", Path(path).parent.name)
    return f"cam{m.group(1)}/{Path(path).stem}" if m else Path(path).parent.name + "/" + Path(path).stem


def fisheye_params(frame, meta):
    get = lambda k: frame.get(k, meta.get(k))
    K = np.array([[get("fl_x"), 0, get("cx")], [0, get("fl_y"), get("cy")], [0, 0, 1]], np.float64)
    D = np.array([get("k1") or 0, get("k2") or 0, get("k3") or 0, get("k4") or 0], np.float64)
    return K, D, int(get("w")), int(get("h"))


def lidar_to_pinhole(depth_png, K, D, Kp, out_wh):
    """Sparse Euclidean depth on the fisheye grid -> z-depth on the pinhole grid (z-buffer, nearest pixel)."""
    r = depth_png.astype(np.float32) / 256.0
    v, u = np.nonzero(r > 0)
    if u.size == 0:
        return np.zeros(out_wh[::-1], np.float32)
    pts = np.stack([u, v], -1).astype(np.float64)[:, None]
    xy = cv2.fisheye.undistortPoints(pts, K, D)[:, 0]               # normalised pinhole coordinates of the rays
    rays = np.concatenate([xy, np.ones((len(xy), 1))], 1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    P = rays * r[v, u][:, None]                                      # 3D points in the camera frame
    z = P[:, 2]
    keep = z > 0.1
    P, z = P[keep], z[keep]
    up = np.round(Kp[0, 0] * P[:, 0] / z + Kp[0, 2]).astype(int)
    vp = np.round(Kp[1, 1] * P[:, 1] / z + Kp[1, 2]).astype(int)
    W, H = out_wh
    ok = (up >= 0) & (up < W) & (vp >= 0) & (vp < H)
    up, vp, z = up[ok], vp[ok], z[ok]
    out = np.full((H, W), np.inf, np.float32)
    np.minimum.at(out, (vp, up), z.astype(np.float32))
    out[~np.isfinite(out)] = 0
    return out


def convert(seq_dir: Path, root: Path, hfov: float, size, every: int, poses: str = "gt"):
    seq = seq_dir.parent.name if seq_dir.name == "processed" else seq_dir.name
    site = "-".join(seq.split("-")[3:-1])                            # 2024-03-12-keble-college-02 -> keble-college
    meta = json.load(open(seq_dir / "colmap" / "transforms_colmap_scaled.json"))
    imgs = zipfile.ZipFile(seq_dir / "colmap" / "images.zip")
    deps = zipfile.ZipFile(seq_dir / "lidar-depths" / "depths_euc.zip")
    dep_names = {frame_key(n): n for n in deps.namelist() if n.endswith(".png")}
    img_names = {frame_key(n): n for n in imgs.namelist() if n.endswith(".jpg")}
    gt = gt_poses(seq_dir / "trajectory" / "gt-tum.txt") if poses == "gt" else None
    W, H = size
    f = W / 2 / math.tan(math.radians(hfov) / 2)
    Kp = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]])
    # ground-truth poses: every sequence of a site is in the site's scanner map, so sessions of a site share a world
    sw = SceneWriter(root, "oxford_spires", seq, world=site if gt is not None else seq, metric=True, synthetic=False,
                     dynamic=False,
                     kind="outdoor_walk", extra={"site": site, "source": "HF ori-drs/oxford_spires_dataset (processed)",
                                                 "depth_src": "64-beam LiDAR projected per image (depths_euc), "
                                                              "re-projected to the pinhole camera",
                                                 "poses": "ground truth (gt-tum: base in the scanner map) + camera "
                                                          "calibration" if gt is not None else
                                                          "COLMAP scaled to metres (transforms_colmap_scaled.json)"})
    frames = sorted(meta["frames"], key=lambda fr: fr["file_path"])
    by_cam = {}
    for fr in frames:
        key = frame_key(fr["file_path"])
        by_cam.setdefault(key.split("/")[0], []).append((key, fr))
    stats = {"frames": 0, "no_depth": 0}
    for cam, lst in sorted(by_cam.items()):
        q = sw.sequence(cam, session=seq)
        for key, fr in lst[::every]:
            if key not in dep_names or key not in img_names:
                stats["no_depth"] += 1
                continue
            K, D, w, h = fisheye_params(fr, meta)
            img = cv2.imdecode(np.frombuffer(imgs.read(img_names[key]), np.uint8), cv2.IMREAD_COLOR)
            dep = cv2.imdecode(np.frombuffer(deps.read(dep_names[key]), np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None or dep is None or img.shape[:2] != (h, w) or dep.shape != (h, w):
                stats["no_depth"] += 1
                continue
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Kp, (W, H), cv2.CV_32FC1)
            rgb = cv2.cvtColor(cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), cv2.COLOR_BGR2RGB)
            z = lidar_to_pinhole(dep, K, D, Kp, (W, H))
            if (z > 0).sum() < 200:
                stats["no_depth"] += 1
                continue
            if gt is not None:
                E = gt_cam_from_world(float(Path(key).name), gt, cam)
                if E is None:
                    stats["no_pose"] = stats.get("no_pose", 0) + 1
                    continue
            else:
                E = np.linalg.inv(np.array(fr["transform_matrix"], np.float64) @ GL_TO_CV)
            q.add(Path(key).name, rgb, z, Kp, E)
            stats["frames"] += 1
    n = sw.close()
    return n, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir", nargs="+")
    ap.add_argument("--root", required=True)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--size", type=int, nargs=2, default=[768, 576])
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--poses", choices=["gt", "colmap"], default="gt")
    a = ap.parse_args()
    for d in a.seq_dir:
        n, st = convert(Path(d), Path(a.root), a.hfov, a.size, a.every, a.poses)
        print(d, n, st, flush=True)


if __name__ == "__main__":
    main()
