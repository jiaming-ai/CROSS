"""NCLT (University of Michigan North Campus Long-Term; Carlevaris-Bianco, Ushani, Eustice, IJRR 2016; ODbL) -> scene cache.

Segway robot driven through the North Campus of the University of Michigan, indoors (halls, corridors, atriums of
several buildings) and outdoors (paths, plazas, car parks), 27 sessions over 15 months.  Used here: the forward
Ladybug3 camera Cam5 (1.23 m above the wheel axle), the Velodyne HDL-32E (0.96 m above the axle, +10.7..-30.7 degrees)
and the ground truth (LiDAR SLAM + RTK GPS; body pose in the NCLT local NED frame, odometry between the SLAM nodes).

Inputs (from benchmark/datasets/prepare_nclt.py and `fetch` below):
- <prepared>/<session>/lb3/Cam5/<utime>.jpg: rectified frames (NCLT U2D maps, upright, centred 88 x 71 degree crop,
  640 x 480), <prepared>/<session>/lb3/cameras.json (K, T_body_cam of the rectified camera);
- <prepared>/<session>/gt_body.txt: `t x y z qx qy qz qw`, body (x forward, y right, z down) in the local NED frame;
- <vel>/<session>/velodyne_hits.bin: every Velodyne packet (`fetch`: the first member of the session's archive).
  Packet: 4 x uint16 magic 44444, uint32 n, uint64 utime, 4 bytes padding, n x (uint16 x, y, z (5 mm, offset -100 m),
  uint8 intensity, uint8 laser).  Points are in the Velodyne frame, x_body_velodyne = (0.002, -0.004, -0.957 m,
  roll 0.807, pitch 0.166, yaw -90.703 deg) (NCLT paper Table 4; checked with `check-frame`).

Depth: the packets within --window s of the image time, each moved to the world with the ground-truth body pose at the
packet's own time (motion compensation and accumulation in one step), then into the camera at the image time,
z-buffered at the stored resolution, points showing through gaps of nearer surfaces dropped (drop_see_through).
Frames are kept when the camera moved >= --min-dist m or turned >= --min-deg degrees since the last kept one.

Worlds: all sessions share the NCLT local frame, but the campus (~450 x 850 m) is too large for one 5 cm label grid,
so it is cut into square tiles of --tile m (default 200 m, origin --tile-origin): world `t<ix>_<iy>` = tile column
ix (north, x) / row iy (east, y).  Each frame belongs to the tile of its camera centre.  One scene per (session, tile)
`<session>_<world>`, one sequence per contiguous run of frames inside the tile (`r<k>`, >= --min-run frames).

python -m vggt_ft.prep.nclt fetch <vel_root> --sessions 2012-08-04 ...
python -m vggt_ft.prep.nclt convert --prepared <prepared> --vel <vel_root> --root <cache root> --sessions ... [--workers N]
python -m vggt_ft.prep.nclt check-frame --prepared <prepared> --vel <vel_root> --session <s>   (Velodyne frame test)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tarfile
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

BASE = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"
DATASET = "nclt"
X_BODY_VEL = [0.002, -0.004, -0.957, 0.807, 0.166, -90.703]
HIT = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("i", "u1"), ("l", "u1")])

try:
    from numba import njit
except ImportError:
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


# ---------------------------------------------------------------------------------------------------------- fetch
def _nclt_download():
    """benchmark/datasets/nclt_download.py by path ('benchmark' is shadowed by a package of the same name in some envs)."""
    import importlib.util
    p = Path(__file__).resolve().parents[2] / "benchmark" / "datasets" / "nclt_download.py"
    spec = importlib.util.spec_from_file_location("nclt_download", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fetch_hits(session: str, out_root: Path, connections: int = 12):
    RangeStream = _nclt_download().RangeStream
    dest = out_root / session / "velodyne_hits.bin"
    if dest.exists():
        return dest
    part = dest.with_name(dest.name + ".part")
    if part.exists() and time.time() - part.stat().st_mtime < 300:
        print(f"{session}: being fetched by another process, skipped", flush=True)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    part.touch()                                    # claim the session before the (slow) stream start
    rs = RangeStream(f"{BASE}/velodyne_data/{session}_vel.tar.gz", connections=connections)
    t0 = time.time()
    with tarfile.open(fileobj=rs, mode="r|gz") as tf:
        for m in tf:
            if m.name.endswith("velodyne_hits.bin"):
                src = tf.extractfile(m)
                with open(part, "wb") as f:
                    shutil.copyfileobj(src, f, 16 << 20)
                part.rename(dest)
                break
    rs.close()
    print(f"{session}: {dest.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f} s "
          f"({rs.consumed / 1e9:.2f} GB fetched, {rs.rate():.1f} MB/s)", flush=True)
    return dest


# ---------------------------------------------------------------------------------------------------------- geometry
def ssc_to_T(x):
    """NCLT 6-DOF [x y z roll pitch yaw] (m, deg), R = Rz(yaw) Ry(pitch) Rx(roll) -> 4x4."""
    r, p, h = np.radians(x[3:6])
    cr, sr, cp, sp, ch, sh = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(h), math.sin(h)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = x[:3]
    return T


class GT:
    """Ground-truth body poses (world-from-body), interpolated (linear / slerp) at any time inside the span."""

    def __init__(self, path: Path):
        from scipy.spatial.transform import Rotation, Slerp
        g = np.loadtxt(path, comments="#")
        keep = np.concatenate([[True], np.diff(g[:, 0]) > 1e-6])
        g = g[keep]
        self.t, self.p = g[:, 0], g[:, 1:4]
        self.slerp = Slerp(self.t, Rotation.from_quat(g[:, 4:8]))

    def __call__(self, ts):
        """(N,4,4) world-from-body at times ts (s); NaN outside the span."""
        ts = np.atleast_1d(np.asarray(ts, np.float64))
        ok = (ts >= self.t[0]) & (ts <= self.t[-1])
        T = np.full((len(ts), 4, 4), np.nan)
        if ok.any():
            tt = ts[ok]
            M = np.tile(np.eye(4), (len(tt), 1, 1))
            M[:, :3, :3] = self.slerp(tt).as_matrix()
            for k in range(3):
                M[:, k, 3] = np.interp(tt, self.t, self.p[:, k])
            T[ok] = M
        return T


# ---------------------------------------------------------------------------------------------------------- velodyne
@njit(cache=True)
def _scan_packets(buf):
    n = buf.shape[0]
    cap = n // 32 + 1
    off = np.empty(cap, np.int64)
    ut = np.empty(cap, np.int64)
    nh = np.empty(cap, np.int32)
    k, pos, skipped = 0, 0, 0
    while pos + 24 <= n:
        ok = True
        for j in range(8):
            if buf[pos + j] != (0x9C if j % 2 == 0 else 0xAD):         # 44444 = 0xAD9C, little endian
                ok = False
                break
        if not ok:                                                   # resynchronise on the next magic
            pos += 1
            skipped += 1
            continue
        h = np.int64(buf[pos + 8]) | (np.int64(buf[pos + 9]) << 8) | (np.int64(buf[pos + 10]) << 16) | \
            (np.int64(buf[pos + 11]) << 24)
        u = np.int64(0)
        for j in range(8):
            u |= np.int64(buf[pos + 12 + j]) << (8 * j)
        if pos + 24 + 8 * h > n:
            break
        off[k], ut[k], nh[k] = pos + 24, u, h
        k += 1
        pos += 24 + 8 * h
    return off[:k], ut[:k], nh[:k], skipped


class Velodyne:
    """Packets of velodyne_hits.bin (memory-mapped) with an index of (offset, utime, hits) cached next to the file."""

    def __init__(self, path: Path):
        self.buf = np.memmap(path, np.uint8, "r")
        idx = path.with_name(path.name + ".idx.npz")
        z = np.load(idx) if idx.exists() else None
        if z is not None and int(z["size"]) == len(self.buf):
            self.off, self.ut, self.nh = z["off"], z["ut"], z["nh"]
        else:
            assert self.buf[0] == 0x9C and self.buf[1] == 0xAD, f"{path}: no packet magic"
            self.off, self.ut, self.nh, skipped = _scan_packets(self.buf)
            order = np.argsort(self.ut, kind="stable")
            self.off, self.ut, self.nh = self.off[order], self.ut[order], self.nh[order]
            np.savez(idx, off=self.off, ut=self.ut, nh=self.nh, size=np.int64(len(self.buf)),
                     skipped=np.int64(skipped))
        self.t = self.ut * 1e-6

    def window(self, t0: float, t1: float):
        """Points (M,3) float64 in the Velodyne frame and their packet times (M,) of the packets in [t0, t1]."""
        a, b = np.searchsorted(self.t, [t0, t1])
        if b <= a:
            return np.zeros((0, 3)), np.zeros(0)
        parts = [np.frombuffer(self.buf, HIT, int(self.nh[k]), int(self.off[k])) for k in range(a, b)]
        h = np.concatenate(parts)
        tp = np.repeat(self.t[a:b], self.nh[a:b])
        P = np.stack([h["x"], h["y"], h["z"]], -1).astype(np.float64) * 0.005 - 100.0
        return P, tp


def world_points(vel: Velodyne, gt: GT, T_body_vel: np.ndarray, t: float, window: float, rmin=1.0, rmax=80.0):
    """Velodyne points within +-window s of t in the world frame, each with the body pose at its packet time."""
    P, tp = vel.window(t - window, t + window)
    r = np.linalg.norm(P, axis=1)
    keep = (r > rmin) & (r < rmax)
    P, tp = P[keep], tp[keep]
    if not len(P):
        return P
    Pb = P @ T_body_vel[:3, :3].T + T_body_vel[:3, 3]
    ut, inv = np.unique(tp, return_inverse=True)
    T = gt(ut)
    ok = np.isfinite(T[:, 0, 0])[inv]
    Pw = np.einsum("nij,nj->ni", T[inv, :3, :3], Pb) + T[inv, :3, 3]
    return Pw[ok]


# ---------------------------------------------------------------------------------------------------------- frames
def load_camera(prepared: Path, session: str, cam: int = 5):
    c = json.load(open(prepared / session / "lb3" / "cameras.json"))[f"cam{cam}"]
    return np.array(c["K"], np.float64), np.array(c["T_body_cam"], np.float64), (int(c["width"]), int(c["height"]))


def frame_times(prepared: Path, session: str, cam: int = 5):
    """utimes (int) of the rectified frames of a camera."""
    f = np.loadtxt(prepared / session / "lb3" / "frames.txt", comments="#", dtype=np.int64)
    f = f[(f[:, 0] == cam) & (f[:, 2] == 1)]
    return np.unique(f[:, 1])


def select(ut, gt: GT, T_body_cam, min_dist: float, min_deg: float):
    """Indices of frames that moved >= min_dist or turned >= min_deg since the last kept one; their cam-to-world."""
    T = gt(ut * 1e-6) @ T_body_cam
    keep, last = [], None
    for i in range(len(ut)):
        if not np.isfinite(T[i, 0, 0]):
            continue
        if last is not None:
            d = np.linalg.norm(T[i, :3, 3] - last[:3, 3])
            c = (np.trace(last[:3, :3].T @ T[i, :3, :3]) - 1) / 2
            if d < min_dist and math.degrees(math.acos(min(1.0, max(-1.0, c)))) < min_deg:
                continue
        keep.append(i)
        last = T[i]
    keep = np.array(keep, np.int64)
    return keep, T[keep]


def tile_of(xy, tile: float, origin):
    return np.floor((np.asarray(xy) - np.asarray(origin)) / tile).astype(int)


def world_name(ix: int, iy: int) -> str:
    return f"t{ix}_{iy}"


def runs_by_tile(c2w, tile: float, origin, min_run: int):
    """{world: [index arrays (into c2w) of contiguous runs]} of frames by the tile of their camera centre."""
    tl = tile_of(c2w[:, :2, 3], tile, origin)
    out = {}
    s = 0
    for i in range(1, len(tl) + 1):
        if i == len(tl) or (tl[i] != tl[s]).any():
            if i - s >= min_run:
                out.setdefault(world_name(*tl[s]), []).append(np.arange(s, i))
            s = i
    return out


# ---------------------------------------------------------------------------------------------------------- convert
def depth_for(vel, gt, T_body_vel, E, K, hw, t, a):
    from vggt_ft.prep.projection import drop_see_through, zbuffer
    P = world_points(vel, gt, T_body_vel, t, a.window)
    d = zbuffer(P, E, K, hw, near=a.near) if len(P) else np.zeros(hw, np.float32)
    d = drop_see_through(d)
    d[d > a.max_depth] = 0
    return d


def convert_scene(job):
    """One (session, world) scene: all runs of the session inside the tile."""
    from vggt_ft.prep.common import SceneWriter, _atomic_write_bytes
    import torch
    torch.set_num_threads(2)
    session, world, runs, a = job
    root = Path(a.root)
    scene = f"{session}_{world}"
    if (root / DATASET / scene / "scene.json").exists():
        return scene, -1, {}
    prepared, vel_root = Path(a.prepared), Path(a.vel)
    K, T_body_cam, (W, H) = load_camera(prepared, session)
    gt = GT(prepared / session / "gt_body.txt")
    vel = Velodyne(vel_root / session / "velodyne_hits.bin")
    T_body_vel = ssc_to_T(X_BODY_VEL)
    ut = frame_times(prepared, session)
    sw = SceneWriter(root, DATASET, scene, world=world if a.world_mode == "tile" else scene, metric=True, synthetic=False, dynamic=False,
                     kind="campus_robot", max_side=a.max_side,
                     extra={"session": session, "tile": world, "tile_m": a.tile, "tile_origin": list(a.tile_origin),
                            "source": "NCLT (robots.engin.umich.edu/nclt), rectified Ladybug3 Cam5 "
                                      "(prepare_nclt.py images)",
                            "depth": f"Velodyne HDL-32E packets within +-{a.window} s, each with the ground-truth "
                                     "pose at its time, z-buffered, see-through points dropped",
                            "poses": "NCLT ground truth (LiDAR SLAM + RTK, local NED frame) + Ladybug3 calibration",
                            "frame": "NCLT local NED (x north, y east, z down)"})
    st = {"frames": 0, "no_depth": 0, "no_image": 0, "valid": []}
    for k, idx in enumerate(runs):
        q = sw.sequence(f"r{k:02d}", session=session)
        for i in idx:
            u = int(ut[i])
            src = prepared / session / "lb3" / "Cam5" / f"{u}.jpg"
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (H, W):
                st["no_image"] += 1
                continue
            E = np.linalg.inv(gt([u * 1e-6])[0] @ T_body_cam)
            d = depth_for(vel, gt, T_body_vel, E, K, (H, W), u * 1e-6, a)
            if (d > 0).sum() < a.min_points:
                st["no_depth"] += 1
                continue
            q.add(str(u), cv2.cvtColor(img, cv2.COLOR_BGR2RGB), d, K, E, t=u * 1e-6)
            if max(H, W) <= a.max_side:
                # keep the source JPEG (~50 KB, the same pixels) instead of SceneWriter's quality-95 re-encoding (~100 KB)
                _atomic_write_bytes(q.dir / "rgb" / f"{u}.jpg", src.read_bytes())
            st["frames"] += 1
            st["valid"].append(float((d > 0).mean()))
    n = sw.close()
    st["valid"] = round(float(np.median(st["valid"])), 4) if st["valid"] else 0
    return scene, n, st


def _safe(job):
    try:
        return convert_scene(job)
    except Exception as e:  # noqa: BLE001
        return f"{job[0]}_{job[1]}", f"ERROR {e!r} {traceback.format_exc(limit=4)}", {}


def plan_session(session: str, a):
    prepared = Path(a.prepared)
    K, T_body_cam, _ = load_camera(prepared, session)
    gt = GT(prepared / session / "gt_body.txt")
    ut = frame_times(prepared, session)
    keep, c2w = select(ut, gt, T_body_cam, a.min_dist, a.min_deg)
    tiles = runs_by_tile(c2w, a.tile, a.tile_origin, a.min_run)
    return [(session, w, [keep[r] for r in runs], a) for w, runs in sorted(tiles.items())], len(ut), len(keep)


def convert(a):
    jobs = []
    for s in a.sessions:
        if not (Path(a.vel) / s / "velodyne_hits.bin").exists():
            print(f"{s}: no velodyne_hits.bin, skipped", flush=True)
            continue
        Velodyne(Path(a.vel) / s / "velodyne_hits.bin")           # build the packet index once
        js, n_all, n_keep = plan_session(s, a)
        print(f"{s}: {n_all} frames, {n_keep} kept, {len(js)} tiles "
              f"{ {j[1]: sum(len(r) for r in j[2]) for j in js} }", flush=True)
        jobs += js
    if a.only:
        jobs = [j for j in jobs if j[1] in a.only]
    jobs.sort(key=lambda j: -sum(len(r) for r in j[2]))
    with Pool(a.workers, maxtasksperchild=8) as pool:
        for scene, n, st in pool.imap_unordered(_safe, jobs):
            print(scene, n, json.dumps(st), flush=True)


# ---------------------------------------------------------------------------------------------------------- checks
def check_frame(a):
    """Which frame are the hits in?  Points of +-1 s moved to the world with each hypothesis: the right one gives the
    sharpest cloud (fewest occupied 10 cm voxels per point) and the ground at the expected height below the axle."""
    prepared, s = Path(a.prepared), a.session
    gt = GT(prepared / s / "gt_body.txt")
    vel = Velodyne(Path(a.vel) / s / "velodyne_hits.bin")
    print(f"{len(vel.t)} packets, {vel.t[-1] - vel.t[0]:.0f} s, hits/packet median {np.median(vel.nh)}", flush=True)
    hyps = {"velodyne(x_body_vel)": ssc_to_T(X_BODY_VEL), "body(identity)": np.eye(4),
            "velodyne_yaw+90": ssc_to_T(X_BODY_VEL[:5] + [89.297])}
    rng = np.random.default_rng(0)
    ts = vel.t[len(vel.t) // 20::max(1, len(vel.t) // 40)][:25]
    for name, T in hyps.items():
        occ, gz = [], []
        for t in ts:
            Pw = world_points(vel, gt, T, t, 1.0)
            if len(Pw) < 1000:
                continue
            Pw = Pw[rng.choice(len(Pw), min(len(Pw), 150000), replace=False)]
            v = np.unique(np.floor(Pw / 0.1).astype(np.int64), axis=0)
            occ.append(len(v) / len(Pw))
            Tb = gt([t])[0]
            Pb = (Pw - Tb[:3, 3]) @ Tb[:3, :3]
            near = Pb[(np.abs(Pb[:, 0]) < 6) & (np.abs(Pb[:, 1]) < 6)]
            if len(near):
                gz.append(np.percentile(near[:, 2], 90))
        print(f"{name}: voxels/point {np.median(occ):.4f}, ground z in body (90th pct within 6 m) "
              f"{np.median(gz):.3f} m", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("vel_root", type=Path)
    f.add_argument("--sessions", nargs="+", required=True)
    f.add_argument("--connections", type=int, default=12)
    for name in ("convert", "check-frame"):
        c = sub.add_parser(name)
        c.add_argument("--prepared", required=True)
        c.add_argument("--vel", required=True)
        if name == "check-frame":
            c.add_argument("--session", required=True)
            continue
        c.add_argument("--root", required=True)
        c.add_argument("--sessions", nargs="+", required=True)
        c.add_argument("--only", nargs="*", default=None, help="worlds (tiles) to convert")
        c.add_argument("--workers", type=int, default=16)
        c.add_argument("--min-dist", type=float, default=0.3, help="m between kept frames ...")
        c.add_argument("--min-deg", type=float, default=10.0, help="... or degrees of rotation")
        c.add_argument("--tile", type=float, default=200.0)
        c.add_argument("--tile-origin", type=float, nargs=2, default=[-350.0, -730.0])
        c.add_argument("--min-run", type=int, default=8)
        c.add_argument("--world-mode", choices=["tile", "session"], default="tile",
                       help="world (label grid) = the tile (all sessions accumulated) or the (session, tile) scene")
        c.add_argument("--window", type=float, default=0.2, help="s of Velodyne packets on each side of the image")
        c.add_argument("--near", type=float, default=0.5)
        c.add_argument("--max-depth", type=float, default=80.0)
        c.add_argument("--min-points", type=int, default=500)
        c.add_argument("--max-side", type=int, default=768)
    a = ap.parse_args()
    if a.cmd == "fetch":
        for s in a.sessions:
            try:
                fetch_hits(s, a.vel_root, a.connections)
            except Exception as e:  # noqa: BLE001
                print(f"{s}: FAILED {e!r}", file=sys.stderr, flush=True)
    elif a.cmd == "check-frame":
        check_frame(a)
    else:
        convert(a)


if __name__ == "__main__":
    main()
