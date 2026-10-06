"""TartanAir v2 (theairlabcmu/tartanair2 on HF) -> scene cache.

Input: <raw>/<Env>/Data_{easy,hard}/{image,depth}_lcam_front.zip (one zip holds every trajectory P000.. of that
difficulty; poses in image zip's P00x/pose_lcam_front.txt).  Camera: 640 x 640 pinhole, f = 320, centred.
Poses: per line "x y z qx qy qz qw", body-to-world in NED with the body axes x forward, y right, z down (TartanAir
convention); OpenCV camera = body rotated by R_body_cam.  Depth: float32 packed in RGBA PNGs, z-depth in metres.

One output scene per (environment, difficulty, trajectory); all trajectories of an environment share its world frame
(`world` = environment), which gives cross-session windows (easy / hard trajectories through the same places).
Variant environments (OldTownFall / ...Winter, SeasonalForest*, *Day / *Night) are the same trajectories rendered
under other light / season (checked 2026-10-04: ArchVizTinyHouseDay vs Night, matched frames 0.00 m / 0.0 deg apart,
overlap 0.997), so they share a world (`--no-variant-worlds` to keep them apart): appearance-change sessions.

python -m vggt_ft.prep.tartanair2 <raw_root> <cache_root> [--envs A B] [--every 1] [--workers 16]
"""
from __future__ import annotations

import argparse
import io
import re
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .common import SceneWriter

K = np.array([[320.0, 0, 319.5], [0, 320.0, 319.5], [0, 0, 1]])
R_BODY_CAM = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], np.float64)   # columns: camera x, y, z in NED body axes
VARIANTS = {"OldTownFall": "OldTown", "OldTownNight": "OldTown", "OldTownSummer": "OldTown", "OldTownWinter": "OldTown",
            "SeasonalForestAutumn": "SeasonalForest", "SeasonalForestSpring": "SeasonalForest",
            "SeasonalForestSummerNight": "SeasonalForest", "SeasonalForestWinter": "SeasonalForest",
            "SeasonalForestWinterNight": "SeasonalForest", "ArchVizTinyHouseDay": "ArchVizTinyHouse",
            "ArchVizTinyHouseNight": "ArchVizTinyHouse", "OldBrickHouseDay": "OldBrickHouse",
            "OldBrickHouseNight": "OldBrickHouse", "WaterMillDay": "WaterMill", "WaterMillNight": "WaterMill"}


def pose_to_E(line_vals):
    x, y, z, qx, qy, qz, qw = line_vals
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix() @ R_BODY_CAM
    T[:3, 3] = [x, y, z]
    return np.linalg.inv(T)


def convert_traj(job):
    img_zip, dep_zip, env, diff, traj, out_root, every, variant_worlds = job
    scene = f"{env}_{diff.replace('Data_', '')}_{traj}"
    if (Path(out_root) / "tartanair2" / scene / "scene.json").exists():
        return scene, -1
    zi, zd = zipfile.ZipFile(img_zip), zipfile.ZipFile(dep_zip)
    pre = f"{env}/{diff}/{traj}"
    poses = np.loadtxt(io.BytesIO(zi.read(f"{pre}/pose_lcam_front.txt")))
    names = sorted(n for n in zi.namelist() if n.startswith(f"{pre}/image_lcam_front/") and n.endswith(".png"))
    dnames = set(zd.namelist())
    world = VARIANTS.get(env, env) if variant_worlds else env
    sw = SceneWriter(out_root, "tartanair2", scene, world=world, metric=True, synthetic=True, kind="mixed",
                     extra={"env": env, "difficulty": diff})
    q = sw.sequence(traj, session=f"{env}_{diff}_{traj}")
    for n in names[::every]:
        idx = int(re.match(r"(\d+)_lcam_front", Path(n).name).group(1))
        dn = f"{pre}/depth_lcam_front/{idx:06d}_lcam_front_depth.png"
        if dn not in dnames or idx >= len(poses):
            continue
        rgb = cv2.imdecode(np.frombuffer(zi.read(n), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
        draw = cv2.imdecode(np.frombuffer(zd.read(dn), np.uint8), cv2.IMREAD_UNCHANGED)
        dep = np.ascontiguousarray(draw).view("<f4")[..., 0]
        dep = np.where(dep > 400, 0, dep)          # sky / far background
        q.add(f"{idx:06d}", rgb, dep, K, pose_to_E(poses[idx]))
    return scene, sw.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--envs", nargs="*")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-variant-worlds", dest="variant_worlds", action="store_false")
    ap.add_argument("--max-trajs", type=int, default=0, help="per zip (for quick tests)")
    a = ap.parse_args()
    jobs = []
    for img_zip in sorted(Path(a.raw).glob("*/Data_*/image_lcam_front.zip")):
        env, diff = img_zip.parts[-3], img_zip.parts[-2]
        dep_zip = img_zip.with_name("depth_lcam_front.zip")
        if a.envs and env not in a.envs:
            continue
        if not (Path(str(img_zip) + ".done").exists() and Path(str(dep_zip) + ".done").exists()):
            continue
        trajs = sorted({n.split("/")[2] for n in zipfile.ZipFile(img_zip).namelist() if n.count("/") >= 3})
        if a.max_trajs:
            trajs = trajs[:a.max_trajs]
        jobs += [(str(img_zip), str(dep_zip), env, diff, t, a.out, a.every, a.variant_worlds) for t in trajs]
    print(f"{len(jobs)} trajectories", flush=True)
    with Pool(a.workers) as p:
        for scene, n in p.imap_unordered(convert_traj, jobs):
            print(scene, n, flush=True)


if __name__ == "__main__":
    main()
