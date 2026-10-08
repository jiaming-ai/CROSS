"""GrandTour (Frey, Tuna et al., ETH Zurich RSL 2025; HF leggedrobotics/grand_tour_dataset, zarr) -> scene cache.

ANYmal-D legged robot carrying the Boxi sensor head through forests, meadows, mountain trails, villages and industrial
sites (49 missions of 3-16 min).  Used here: the front HDR colour camera `hdr_front` (1920x1280, 10 Hz, equidistant
fisheye model), the motion-compensated scans of the Hesai XT32 `hesai_points_undistorted` (spin axis vertical,
-16..+15 degrees: a band around the horizon across the whole image) and of the Livox Mid-360 `livox_points_undistorted`
(mounted upside down, looking down: the near ground below the Hesai's band), and the DLIO LiDAR-inertial odometry
`dlio_map_odometry` (pose of `hesai_lidar` in `dlio_map`, one per Hesai scan).

Input: one mission folder as on HF: <mission>/data/<topic>.tar (one zarr group per topic: arrays `timestamp` (s),
`points` (scans x max points x 3, first `valid[i, 0]` rows used), `pose_pos` / `pose_orien` (xyzw); group attributes
`frame_id`, `camera_info`, and the static transforms in `tf`; points are stored in chunks of 256 scans, read one
chunk at a time), <mission>/images/hdr_front.tar (<index>.jpeg, index = row of the topic's `timestamp`).  The needed
tars are extracted into a work folder (removed afterwards unless --keep-extracted).

Conventions (a static transform entry maps its parent frame's coordinates to its child frame's, p_child = T p_parent,
see StaticTf; the odometry is the pose of `hesai_lidar` in `dlio_map`, p_map = T p_lidar):
  camera-to-world(t) = T_map_odomchild(t) T_odomchild_box T_box_cam, interpolated at the image timestamp (linear /
  slerp); LiDAR points of a scan at t_s go to the world with T_map_odomchild(t_s) T_odomchild_box T_box_lidar.
The image is undistorted to the pinhole camera with the same K (88.6 x 66.1 degrees; the fisheye's periphery is
cropped, no black border), resized to the cache resolution, and the LiDAR depth is rendered at that resolution: z-depth
of the scans nearest in time (default 3 Hesai within 0.15 s and 5 Livox within 0.3 s), z-buffered together, points
showing through gaps of nearer surfaces dropped (drop_see_through).
Frames are kept when the camera moved >= --min-dist m or turned >= --min-deg degrees since the last kept frame.  One
scene per mission (world = the mission's DLIO map), one sequence `hdr_front`.  Real, metric, mostly static (people
walk through some missions).

Checks (2026-10-08 on missions 2024-11-04-10-57-34 (village) and 2024-11-14-14-36-02 (forest); scripts and numbers in
the conversion report):
* Odometry child frame: Hesai scans 2 s apart moved to the map agree to a median nearest-neighbour distance of
  0.04-0.09 m with child = hesai_lidar (box_base 0.17-1.7 m, base 0.14-1.4 m); single scans registered to the DLIO map:
  Hesai 0.012-0.018 m, Livox 0.014-0.020 m.
* Static transforms: see StaticTf (the ROS reading puts the Hesai's points on clear sky: 14-50 % vs 0.1-2 %).
* Image alignment: share of the Hesai points near the boundary of clear blue sky that land on the sky, for pose time
  offsets of -0.1..0.1 s: minimum at 0 in both missions (camera and LiDAR clocks agree); for pixel shifts: minimum within
  1-3 px of zero at 768 px width.
* GT overlap of frames 1 / 2 / 3 apart: 0.85 / 0.76 / 0.69 (village), 0.78 / 0.67 / 0.57 (forest) with the poses as
  stored, 0.77 / 0.61 / 0.53 and 0.59 / 0.33 / 0.36 inverted.  The test is flat for depth scaled by 0.9-1.1 and for ray
  vs z depth (sparse depth), so it says nothing about the depth's scale; the depth is z by construction, and depth and
  poses come from the same LiDARs.
* The DLIO map (~3e7 points) as depth: 2-4x denser, but it disagrees with the current scans on 19-39 % of the pixels
  (|log ratio| > 0.1; Livox vs Hesai 7-20 %): far points show through gaps of vegetation the map holds only sparsely,
  and leafless canopy fills the sky.  Not used.
* Camera height above the ground (Livox ground points 0.8-2 m from the base): median 0.62 m (forest) / 0.75 m (village).

python -m vggt_ft.prep.grandtour <mission dir> [...] --root <cache root> --work <scratch dir> [--workers 4]
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import traceback
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from vggt_ft.prep.common import SceneWriter
from vggt_ft.prep.projection import drop_see_through, resize_for_cache, zbuffer

DATASET = "grandtour"
CAMERA = "hdr_front"
LIDARS = ("hesai_points_undistorted", "livox_points_undistorted")
ODOM = "dlio_map_odometry"
ODOM_CHILD = "hesai_lidar"            # checked: see the module docstring


def se3(t, q):
    """4x4 from a translation and a quaternion (dicts x y z (w) or arrays, quaternion xyzw)."""
    from scipy.spatial.transform import Rotation
    if isinstance(t, dict):
        t, q = [t["x"], t["y"], t["z"]], [q["x"], q["y"], q["z"], q["w"]]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(q, np.float64)).as_matrix()
    T[:3, 3] = np.asarray(t, np.float64)
    return T


class StaticTf:
    """Static transforms of the `tf` group.  An entry (base_frame_id P, child_frame_id C, translation, rotation) maps P
    coordinates to C coordinates, p_C = T p_P (the dataset's own scripts use it so; read as the ROS pose of C in P
    instead, the camera's optical axis points along the robot's -y, its image-down axis along +x, and 14-50 % of the
    Hesai points of clear-sky images land on the sky; with this reading: optical axis +x, image-down -z, Hesai spin axis
    vertical, 0.1-2 % on the sky)."""

    def __init__(self, tf: dict):
        self.tf = tf

    def in_base(self, frame: str) -> np.ndarray:
        """p_base = T p_frame (base: the robot base; box_base is a child of base)."""
        if frame == "base":
            return np.eye(4)
        e = self.tf[frame]
        return self.in_base(e["base_frame_id"]) @ np.linalg.inv(se3(e["translation"], e["rotation"]))

    def between(self, parent: str, child: str) -> np.ndarray:
        """p_parent = T p_child."""
        return np.linalg.inv(self.in_base(parent)) @ self.in_base(child)


class Trajectory:
    """Poses of the odometry's child frame in the map, interpolated (linear / slerp) at given times."""

    def __init__(self, t, pos, quat_xyzw):
        from scipy.spatial.transform import Rotation, Slerp
        t = np.asarray(t, np.float64)
        o = np.argsort(t)
        t, pos, quat = t[o], np.asarray(pos, np.float64)[o], np.asarray(quat_xyzw, np.float64)[o]
        keep = np.concatenate([[True], np.diff(t) > 0])
        self.t, self.pos = t[keep], pos[keep]
        self.slerp = Slerp(self.t, Rotation.from_quat(quat[keep]))

    def __call__(self, t: float):
        if t < self.t[0] or t > self.t[-1]:
            return None
        T = np.eye(4)
        T[:3, :3] = self.slerp([t]).as_matrix()[0]
        for k in range(3):
            T[k, 3] = np.interp(t, self.t, self.pos[:, k])
        return T


def extract(mission: Path, work: Path, topics) -> Path:
    """Extract the zarr groups of the topics (data/<topic>.tar, data/.zgroup) and the camera's JPEGs into work."""
    (work / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(mission / "data" / ".zgroup", work / "data" / ".zgroup")
    for name in topics:
        if not (work / "data" / name / ".zgroup").exists():
            with tarfile.open(mission / "data" / f"{name}.tar") as tf:
                tf.extractall(work / "data")
    if not (work / "images" / CAMERA).is_dir():
        with tarfile.open(mission / "images" / f"{CAMERA}.tar") as tf:
            tf.extractall(work / "images")
    return work


def select_frames(cam_t, traj, T_odom_cam, min_dist: float, min_deg: float):
    """Indices of the frames whose camera moved >= min_dist or turned >= min_deg since the last kept one."""
    keep, last = [], None
    for i, t in enumerate(cam_t):
        T = traj(float(t))
        if T is None:
            continue
        c2w = T @ T_odom_cam
        if last is not None:
            d = np.linalg.norm(c2w[:3, 3] - last[:3, 3])
            ang = np.degrees(np.arccos(np.clip((np.trace(last[:3, :3].T @ c2w[:3, :3]) - 1) / 2, -1, 1)))
            if d < min_dist and ang < min_deg:
                continue
        keep.append(i)
        last = c2w
    return keep


class Scans:
    """The motion-compensated scans of one LiDAR topic, in the world frame on demand (one decompressed chunk kept)."""

    def __init__(self, group, T_odom_lidar, traj, n: int, max_dt: float):
        self.g, self.T, self.traj, self.n, self.max_dt = group, T_odom_lidar, traj, n, max_dt
        self.t = np.asarray(group["timestamp"][:], np.float64).ravel()
        self.valid = np.asarray(group["valid"][:]).reshape(len(self.t), -1)[:, 0].astype(np.int64)
        self.cs = int(group["points"].chunks[0])
        self.block, self.block_i = None, -1

    def scan(self, i: int) -> np.ndarray:
        c = i // self.cs
        if c != self.block_i:
            self.block, self.block_i = np.asarray(self.g["points"][c * self.cs:(c + 1) * self.cs]), c
        p = np.asarray(self.block[i - c * self.cs, :self.valid[i]], np.float64)[:, :3]
        return p[np.isfinite(p).all(1) & (np.abs(p).sum(1) > 1e-6)]

    def world_points(self, t: float) -> np.ndarray:
        out = []
        for i in sorted(np.argsort(np.abs(self.t - t))[:self.n]):
            if abs(self.t[i] - t) > self.max_dt or self.valid[i] <= 0:
                continue
            T = self.traj(float(self.t[i]))
            if T is None:
                continue
            M = T @ self.T
            out.append(self.scan(int(i)) @ M[:3, :3].T + M[:3, 3])
        return np.concatenate(out) if out else np.zeros((0, 3))


class Undistort:
    """Remap of the camera's image to the pinhole camera with the same K (equidistant or plumb-bob model)."""

    def __init__(self, ci: dict):
        self.K = np.asarray(ci["K"], np.float64).reshape(3, 3)
        D = np.asarray(ci["D"], np.float64)
        self.W, self.H = int(ci["width"]), int(ci["height"])
        model = ci.get("distortion_model", "plumb_bob")
        if model == "equidistant":
            self.mx, self.my = cv2.fisheye.initUndistortRectifyMap(self.K, D[:4], np.eye(3), self.K, (self.W, self.H),
                                                                   cv2.CV_32FC1)
        elif model in ("plumb_bob", "radtan"):
            self.mx, self.my = cv2.initUndistortRectifyMap(self.K, D, None, self.K, (self.W, self.H), cv2.CV_32FC1)
        else:
            raise ValueError(f"unexpected distortion model {model}")

    def __call__(self, img):
        return cv2.remap(img, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def convert(job):
    mission, root, work, a = job
    mission = Path(mission)
    name = mission.name
    if (Path(root) / DATASET / name / "scene.json").exists():
        return name, -1, {}
    import torch
    import zarr
    torch.set_num_threads(4)
    w = extract(mission, Path(work) / name, [CAMERA, ODOM, "tf", *LIDARS])
    z = zarr.open_group(str(w / "data"), mode="r")
    st = StaticTf(dict(z["tf"].attrs["tf"]))
    od = z[ODOM]
    traj = Trajectory(od["timestamp"][:], od["pose_pos"][:], od["pose_orien"][:])
    cam = z[CAMERA]
    und = Undistort(dict(cam.attrs["camera_info"]))
    K, W, H = und.K, und.W, und.H
    T_odom_cam = st.between(ODOM_CHILD, cam.attrs.get("frame_id", CAMERA))
    cam_t = np.asarray(cam["timestamp"][:], np.float64).ravel()
    scans = [Scans(z[k], st.between(ODOM_CHILD, z[k].attrs["frame_id"]), traj, n, dt)
             for k, n, dt in zip(LIDARS, (a.hesai_scans, a.livox_scans), (a.hesai_dt, a.livox_dt)) if n > 0]
    keep = select_frames(cam_t, traj, T_odom_cam, a.min_dist, a.min_deg)
    sw = SceneWriter(root, DATASET, name, world=name, metric=True, synthetic=False, dynamic=False,
                     kind="legged_robot_outdoor",
                     extra={"source": "HF leggedrobotics/grand_tour_dataset (zarr)", "camera": CAMERA,
                            "depth": f"LiDAR scans nearest in time (Hesai XT32 {a.hesai_scans}, Livox Mid-360 "
                                     f"{a.livox_scans}), z-buffered",
                            "poses": f"DLIO odometry ({ODOM}) + static calibration", "undistorted": True})
    q = sw.sequence(CAMERA, session=name)
    stats = {"candidates": len(cam_t), "kept": len(keep), "frames": 0, "no_depth": 0}
    for i in keep:
        img = cv2.imread(str(w / "images" / CAMERA / f"{i:06d}.jpeg"), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (H, W):
            stats["no_image"] = stats.get("no_image", 0) + 1
            continue
        rgb, Kc = resize_for_cache(cv2.cvtColor(und(img), cv2.COLOR_BGR2RGB), K, a.max_side)
        c2w = traj(float(cam_t[i])) @ T_odom_cam
        E = np.linalg.inv(c2w)
        pts = np.concatenate([s.world_points(float(cam_t[i])) for s in scans] or [np.zeros((0, 3))])
        d = zbuffer(pts, E, Kc, rgb.shape[:2], near=a.near) if len(pts) else np.zeros(rgb.shape[:2], np.float32)
        d = drop_see_through(d)
        d[d > a.max_depth] = 0
        if (d > 0).sum() < a.min_points:
            stats["no_depth"] += 1
            continue
        q.add(f"{i:06d}", rgb, d, Kc, E, t=float(cam_t[i]))
        stats["frames"] += 1
    n = sw.close()
    if not a.keep_extracted:
        shutil.rmtree(w, ignore_errors=True)
    return name, n, stats


def safe(job):
    try:
        return convert(job)
    except Exception as e:
        return Path(job[0]).name, f"ERROR {e!r} {traceback.format_exc(limit=4)}", {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("missions", nargs="+", help="mission folders (<mission>/data, <mission>/images)")
    ap.add_argument("--root", required=True, help="cache root")
    ap.add_argument("--work", required=True, help="scratch folder for the extracted zarr groups and images")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-dist", type=float, default=0.3, help="m between kept frames ...")
    ap.add_argument("--min-deg", type=float, default=8.0, help="... or degrees of rotation")
    ap.add_argument("--hesai-scans", type=int, default=3, help="Hesai scans nearest in time per image (0: none)")
    ap.add_argument("--hesai-dt", type=float, default=0.15, help="s: Hesai scans further from the image time are not used")
    ap.add_argument("--livox-scans", type=int, default=5, help="Livox scans nearest in time per image (0: none)")
    ap.add_argument("--livox-dt", type=float, default=0.3)
    ap.add_argument("--max-depth", type=float, default=80.0)
    ap.add_argument("--near", type=float, default=0.5, help="m: points nearer to the camera are dropped")
    ap.add_argument("--min-points", type=int, default=500, help="frames with fewer depth pixels are skipped")
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--keep-extracted", action="store_true")
    a = ap.parse_args()
    jobs = [(m, a.root, a.work, a) for m in a.missions]
    with Pool(min(a.workers, len(jobs))) as p:
        for name, n, st in p.imap_unordered(safe, jobs):
            print(name, n, st, flush=True)


if __name__ == "__main__":
    main()
