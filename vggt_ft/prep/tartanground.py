"""TartanGround (theairlabcmu/TartanGround on HF, CC-BY-4.0) -> scene cache.

Ground-robot counterpart of TartanAir v2: omnidirectional (P00xx), differential-drive (P10xx) and quadruped (P20xx)
trajectories in 63 Unreal environments; camera height above the ground = the trajectory's `robot_height` (0.5 - 1.5 m,
metadata), 10 Hz, ~0.1 - 0.25 m between frames.

Input (one zip per trajectory and modality, as on HF; a `<zip>.done` marker per finished download):
    <raw>/<Env>/Data_{omni,diff,anymal}/<Pxxxx>/{image_lcam_front,depth_lcam_front,metadata}.zip
image zip: image_lcam_front/<idx>_lcam_front.png; depth zip: depth_lcam_front/<idx>_lcam_front_depth.png (float32
z-depth in metres packed in RGBA); metadata zip: pose_lcam_front.txt ("x y z qx qy qz qw" per line, body-to-world in
NED, TartanAir convention) and <Pxxxx>_metadata.json (robot_height, ...).  Camera: 640 x 640 pinhole, f = 320.

Output: one scene per environment (scene = world = <Env>), one sequence per trajectory (session = trajectory), frames
subsampled to >= `--spacing` m of travel or >= `--rot` deg of rotation since the last kept frame.  The per-trajectory
jobs only write sequences; scene.json (listing every converted sequence of the environment) is written by the main
process when the environment's jobs are done, so environments can be converted incrementally.

python -m vggt_ft.prep.tartanground <raw_root> <cache_root> [--envs A B] [--spacing 0.25] [--rot 15] [--workers 16]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .common import SceneWriter, SeqWriter, _atomic_write_bytes

K = np.array([[320.0, 0, 319.5], [0, 320.0, 319.5], [0, 0, 1]])
R_BODY_CAM = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], np.float64)   # columns: camera x, y, z in NED body axes
SKY = 400.0                                                             # depth beyond this = sky / far background
ROBOT = {"Data_omni": "omni", "Data_diff": "diff", "Data_anymal": "anymal"}
# kind per environment (TartanGround metadata sheet categories, checked on the label previews / sky fraction; an
# "indoor" environment with >= 2.5 % sky pixels becomes "mixed"; FactoryWeather is an outdoor street block)
KIND = {
    # large indoor (halls, wards, classrooms, hangars, factory floors)
    "Hospital": "indoor_large", "Office": "indoor_large", "AbandonedSchool": "indoor_large",
    "IndustrialHangar": "indoor_large", "AbandonedFactory": "indoor_large", "AbandonedFactory2": "indoor_large",
    "Prison": "mixed",                  # prison yard + buildings (sky fraction 0: near sky box) "Restaurant": "indoor_large", "Slaughter": "indoor_large",
    "Supermarket": "indoor_large", "CarWelding": "indoor_large",
    # tunnels
    "CoalMine": "underground", "Sewerage": "underground",
}
NATURE = {"ForestEnv", "SeasonalForestSpring", "SeasonalForestAutumn", "SeasonalForestWinter", "GreatMarsh",
          "OldScandinavia", "Gascola", "BrushifyMoon"}


def pose_to_E(vals):
    x, y, z, qx, qy, qz, qw = vals
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix() @ R_BODY_CAM
    T[:3, 3] = [x, y, z]
    return np.linalg.inv(T)


def subsample(poses: np.ndarray, spacing: float, rot_deg: float) -> list[int]:
    """Indices of frames >= spacing m or >= rot_deg deg away from the last kept frame."""
    keep = [0]
    R = Rotation.from_quat(poses[:, 3:7])
    for i in range(1, len(poses)):
        j = keep[-1]
        if (np.linalg.norm(poses[i, :3] - poses[j, :3]) >= spacing
                or np.degrees((R[j].inv() * R[i]).magnitude()) >= rot_deg):
            keep.append(i)
    return keep


def env_kind(env: str, sky_frac: float) -> str:
    k = KIND.get(env)
    if k is not None:
        return k if sky_frac < 0.025 else "mixed"          # windows: ~1-2 % sky in the indoor scenes
    return "outdoor_nature" if env in NATURE else "outdoor"


def convert_traj(job):
    tdir, env, robot_dir, traj, out_root, dataset, spacing, rot = job
    tdir = Path(tdir)
    seq_dir = Path(out_root) / dataset / env / traj
    if (seq_dir / "frames.npz").exists() and (seq_dir / "info.json").exists():
        return env, traj, json.load(open(seq_dir / "info.json"))
    zm = zipfile.ZipFile(tdir / "metadata.zip")
    poses = np.loadtxt(io.BytesIO(zm.read("pose_lcam_front.txt")))
    meta = json.loads(zm.read(f"{traj}_metadata.json"))
    zi, zd = zipfile.ZipFile(tdir / "image_lcam_front.zip"), zipfile.ZipFile(tdir / "depth_lcam_front.zip")
    imgs = {int(re.match(r"(\d+)_lcam_front", Path(n).name).group(1)): n for n in zi.namelist() if n.endswith(".png")}
    deps = {int(re.match(r"(\d+)_lcam_front_depth", Path(n).name).group(1)): n
            for n in zd.namelist() if n.endswith(".png")}
    sw = SceneWriter(out_root, dataset, env, world=env, metric=True, synthetic=True)
    q = SeqWriter(sw, traj, f"{env}_{traj}")
    sky = []
    for idx in subsample(poses, spacing, rot):
        if idx not in imgs or idx not in deps:
            continue
        rgb = cv2.imdecode(np.frombuffer(zi.read(imgs[idx]), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
        draw = cv2.imdecode(np.frombuffer(zd.read(deps[idx]), np.uint8), cv2.IMREAD_UNCHANGED)
        dep = np.ascontiguousarray(draw).view("<f4")[..., 0].copy()
        far = ~(dep <= SKY)
        sky.append(float(far.mean()))
        dep[far] = 0
        q.add(f"{idx:06d}", rgb, dep, K, pose_to_E(poses[idx]), t=idx * 0.1)
    n = q.close()
    info = dict(n=n, n_raw=len(poses), robot=ROBOT[robot_dir], robot_height=meta.get("robot_height"),
                path_length=meta.get("path_length"), type=meta.get("type"), sky_frac=float(np.mean(sky)) if sky else 0)
    if n:
        _atomic_write_bytes(seq_dir / "info.json", json.dumps(info).encode())
    return env, traj, info


def write_scene(out_root, dataset: str, env: str, test: bool = False):
    """scene.json of an environment from its converted sequences (folders with frames.npz + info.json)."""
    d = Path(out_root) / dataset / env
    seqs, info = {}, {}
    for s in sorted(p for p in d.iterdir() if (p / "frames.npz").exists() and (p / "info.json").exists()):
        i = json.load(open(s / "info.json"))
        if i["n"] >= 2:
            seqs[s.name] = {"session": f"{env}_{s.name}", "n": i["n"]}
            info[s.name] = i
    if not seqs:
        return 0
    w = np.array([i["n"] for i in info.values()], float)
    sky = float(np.average([i["sky_frac"] for i in info.values()], weights=w))
    meta = dict(dataset=dataset, scene=env, world=env, metric=True, synthetic=True, dynamic=False,
                kind=env_kind(env, sky), units=1.0, env=env, sky_frac=round(sky, 4), test=test,
                robot={k: i["robot"] for k, i in info.items()},
                robot_height={k: i["robot_height"] for k, i in info.items()}, sequences=seqs)
    _atomic_write_bytes(d / "scene.json", json.dumps(meta, indent=1).encode())
    return sum(v["n"] for v in seqs.values())


def find_trajs(raw: Path, envs=None):
    """[(traj dir, env, robot dir, traj)] whose three zips are downloaded."""
    out = []
    for m in sorted(Path(raw).glob("*/Data_*/P*/metadata.zip")):
        tdir = m.parent
        env, robot_dir, traj = tdir.parts[-3], tdir.parts[-2], tdir.parts[-1]
        if envs and env not in envs:
            continue
        if all((tdir / f"{z}.zip.done").exists() for z in ("metadata", "image_lcam_front", "depth_lcam_front")):
            out.append((str(tdir), env, robot_dir, traj))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--envs", nargs="*")
    ap.add_argument("--dataset", default="tartanground")
    ap.add_argument("--spacing", type=float, default=0.25)
    ap.add_argument("--rot", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--test", action="store_true", help="mark the scenes as held-out test scenes in scene.json")
    a = ap.parse_args()
    trajs = find_trajs(Path(a.raw), a.envs)
    jobs = [(t, e, r, p, a.out, a.dataset, a.spacing, a.rot) for t, e, r, p in trajs]
    print(f"{len(jobs)} trajectories", flush=True)
    left = defaultdict(int)
    for j in jobs:
        left[j[1]] += 1
    with Pool(a.workers) as pool:
        for env, traj, info in pool.imap_unordered(convert_traj, jobs):
            print(env, traj, info, flush=True)
            left[env] -= 1
            if left[env] == 0:
                print("scene", env, write_scene(a.out, a.dataset, env, a.test), flush=True)


if __name__ == "__main__":
    main()
