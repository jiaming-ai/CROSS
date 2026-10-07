#!/usr/bin/env python3
"""BoQ place-recognition descriptors of prepared NCLT frames, for retrieval stress tests at the million-image scale.

  PYTHONPATH=<CROSS repo> python benchmark/datasets/nclt_descriptors.py <prepared> <desc_root> --session 2012-01-08

For every rectified frame of `<prepared>/<session>/lb3/Cam<k>/` (prepare_nclt.py images) the descriptor is computed
exactly as CROSS computes a keyframe's (cross/db/boq.py: BoQ ResNet50, 16384-d, L2-normalised) on the image as the
feed-forward modes see it (cross.utils.camera.get_transforms_ff: 640x480 -> 512x384, bicubic, antialiased).  Output
per session and camera, in `<desc_root>/<session>/`:

  Cam<k>.npy        (N, 16384) float16 descriptors, rows in frame-time order
  Cam<k>_meta.npz   utime (us), t (s), T_world_cam (N, 4, 4; ground truth, NaN outside its span), T_world_body,
                    gps_lat/lon (deg, nearest consumer fix within 0.5 s, NaN if none), gps_dt (s), gps_xyz (local NED),
                    rtk_ok (an RTK solution within 0.5 s), in_gt (bool)

and `<desc_root>/index.json` lists sessions, cameras, counts and paths.  The world frame is NCLT's local NED frame
(x north, y east, z down; geo_ref.json); camera frames are OpenCV (x right, y down, z forward).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_nclt import to_local  # noqa: E402


def interp_poses(gt: np.ndarray, t: np.ndarray):
    """gt rows (t x y z qx qy qz qw) -> 4x4 body poses at t (linear / slerp); NaN outside the GT span."""
    from scipy.spatial.transform import Rotation, Slerp
    ok = (t >= gt[0, 0]) & (t <= gt[-1, 0])
    tc = np.clip(t, gt[0, 0], gt[-1, 0])
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, 3] = np.stack([np.interp(tc, gt[:, 0], gt[:, k]) for k in (1, 2, 3)], -1)
    T[:, :3, :3] = Slerp(gt[:, 0], Rotation.from_quat(gt[:, 4:8]))(tc).as_matrix()
    T[~ok] = np.nan
    return T, ok


def nearest(ts: np.ndarray, t: np.ndarray):
    i = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)
    j = np.where(np.abs(ts[i - 1] - t) < np.abs(ts[i] - t), i - 1, i)
    return j, np.abs(ts[j] - t)


def frame_meta(prep: Path, session: str, cam: int, utimes: np.ndarray) -> dict:
    d = prep / session
    cams = json.loads((d / "lb3" / "cameras.json").read_text())
    T_bc = np.asarray(cams[f"cam{cam}"]["T_body_cam"])
    t = utimes * 1e-6
    gt = np.loadtxt(d / "gt_body.txt")
    T_wb, in_gt = interp_poses(gt, t)
    g = np.loadtxt(d / "gnss.txt")
    j, dt = nearest(g[:, 0], t)
    has = dt < 0.5
    lat = np.where(has, g[j, 1], np.nan)
    lon = np.where(has, g[j, 2], np.nan)
    r = np.loadtxt(d / "gnss_rtk.txt")
    jr, dtr = nearest(r[:, 0], t)
    return {"utime": utimes.astype(np.int64), "t": t, "T_world_body": T_wb.astype(np.float32),
            "T_world_cam": (T_wb @ T_bc).astype(np.float32), "in_gt": in_gt,
            "gps_lat": lat, "gps_lon": lon, "gps_dt": dt.astype(np.float32),
            "gps_xyz": to_local(lat, lon).astype(np.float32),
            "rtk_ok": (dtr < 0.5) & (r[jr, 4] == 3)}


class _Frames:
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import cv2
        import torch
        rgb = cv2.cvtColor(cv2.imread(str(self.paths[i])), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prepared", type=Path)
    ap.add_argument("desc_root", type=Path)
    ap.add_argument("--session", required=True)
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    from torchvision import transforms
    from cross.core.types import Camera
    from cross.db.boq import BoQ
    from cross.utils.camera import get_transforms_ff

    out = a.desc_root / a.session
    out.mkdir(parents=True, exist_ok=True)
    model = BoQ(backbone_name="resnet50", device=a.device, enable_cache=False)
    lb3 = a.prepared / a.session / "lb3"
    cams = json.loads((lb3 / "cameras.json").read_text())
    for cam in a.cams:
        if (out / f"Cam{cam}.npy").exists():
            continue
        paths = sorted((lb3 / f"Cam{cam}").glob("*.jpg"), key=lambda p: int(p.stem))
        if not paths:
            continue
        c = cams[f"cam{cam}"]
        camera = Camera(K=np.asarray(c["K"]), frame_width=c["width"], frame_height=c["height"])
        tf, _ = get_transforms_ff(camera)                        # ToTensor + Resize (as System.rgb_transform)
        resize = transforms.Compose(tf.transforms[1:])
        loader = torch.utils.data.DataLoader(_Frames(paths), batch_size=a.batch, num_workers=a.workers, pin_memory=True)
        desc = np.empty((len(paths), model.get_embed_dim()), np.float16)
        t0, k = time.time(), 0
        with torch.inference_mode():
            for x in loader:
                x = x.to(a.device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
                x = resize(x)                                    # (B, 3, 384, 512)
                e = model.get_embedding(x)
                desc[k:k + len(e)] = e.half().cpu().numpy()
                k += len(e)
        np.save(out / f"Cam{cam}.npy", desc)
        utimes = np.array([int(p.stem) for p in paths], np.int64)
        np.savez(out / f"Cam{cam}_meta.npz", **frame_meta(a.prepared, a.session, cam, utimes))
        print(f"{a.session} Cam{cam}: {len(paths)} descriptors, {len(paths) / (time.time() - t0):.0f} img/s", flush=True)
    # index of everything in desc_root
    idx = {}
    for sdir in sorted(p for p in a.desc_root.iterdir() if p.is_dir()):
        entry = {}
        for f in sorted(sdir.glob("Cam?.npy")):
            n = int(np.load(f, mmap_mode="r").shape[0])
            entry[f.stem] = {"n": n, "desc": f"{sdir.name}/{f.name}", "meta": f"{sdir.name}/{f.stem}_meta.npz"}
        if entry:
            idx[sdir.name] = entry
    total = sum(v["n"] for e in idx.values() for v in e.values())
    (a.desc_root / "index.json").write_text(json.dumps({"total": total, "dim": 16384, "dtype": "float16",
                                                        "sessions": idx}, indent=1))
    print(f"index: {total} descriptors in {len(idx)} sessions", flush=True)


if __name__ == "__main__":
    main()
