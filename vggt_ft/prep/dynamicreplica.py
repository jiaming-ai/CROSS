"""Dynamic Replica, point-tracking validation subset (HF mirror ZhengGuangze/DynamicReplica: dynamicreplica.tar.gz) ->
scene cache, with sparse depth from the tracked 3D points.

Input: <valid>/<seq>_source_left/images/<seq>_source_left-NNNN.png (1280 x 720), trajectories/NNNNNN.pth (per frame:
traj_3d_world (P, 3) world positions of the tracked points at that frame, verts_inds_vis (P,) visibility, instances
(P,): 0 = points sampled on the static room, > 0 = the animated humans / animals, traj_2d, img), and
frame_annotations_valid.jgz (Implicitron frame annotations, left and right cameras).  The mirror holds the left camera
only, and no depth / mask / flow files (the annotations point to them).

Cameras: viewpoint R, T in PyTorch3D's row-vector convention (X_view = X_world @ R + T, +X left, +Y up), intrinsics
`ndc_isotropic` (f_px = f_ndc * min(W, H) / 2, c_px = (W/2, H/2) - p_ndc * min(W, H) / 2), as in dynamic_stereo's
_get_pytorch3d_camera.  OpenCV: R_cv = diag(-1,-1,1) R^T, t_cv = diag(-1,-1,1) T.  Checked: the visible points
projected this way reproduce traj_2d to 0.000 px (median, 3 frames), without the axis flip ~450-620 px off.

Depth: the visible tracked points (static room + actors at their positions in that frame) z-buffered at the stored
resolution: sparse, ~2-6 % of the pixels (like the sparse lidar of Waymo).  Synthetic, dynamic, metric (Replica
reconstructions, human-scale actors).  Licence: Meta's Dynamic Replica licence (CC BY-NC), whatever the mirror's tag.

python -m vggt_ft.prep.dynamicreplica <valid_dir> <cache_root> [--every 3] [--workers 8]
"""
from __future__ import annotations

import argparse
import gzip
import json
import traceback
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import SceneWriter
from .projection import resize_for_cache, zbuffer

DATASET = "dynamicreplica"
FLIP = np.diag([-1.0, -1.0, 1.0])


def camera(vp: dict, hw) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV K (for the annotation's image size) and 4x4 camera-from-world of an Implicitron viewpoint."""
    H, W = hw
    if vp.get("intrinsics_format", "ndc_isotropic").lower() == "ndc_isotropic":
        sx = sy = min(W, H) / 2.0
    else:                                   # ndc_norm_image_bounds
        sx, sy = W / 2.0, H / 2.0
    K = np.array([[vp["focal_length"][0] * sx, 0, W / 2.0 - vp["principal_point"][0] * sx],
                  [0, vp["focal_length"][1] * sy, H / 2.0 - vp["principal_point"][1] * sy], [0, 0, 1]])
    E = np.eye(4)
    E[:3, :3] = FLIP @ np.asarray(vp["R"], np.float64).T
    E[:3, 3] = FLIP @ np.asarray(vp["T"], np.float64)
    return K, E


def convert(job):
    valid, seq, cam, frames, out, every = job
    scene = f"{seq}_{cam}"
    if (Path(out) / DATASET / scene / "scene.json").exists():
        return scene, -1, None
    d = Path(valid) / f"{seq}_source_{cam}"
    if not (d / "images").exists():
        return scene, 0, {"missing": str(d)}
    rows, actors, near = [], 0, []
    for fa in sorted(frames, key=lambda x: x["frame_number"])[::every]:
        img = cv2.imread(str(Path(valid) / fa["image"]["path"]), cv2.IMREAD_COLOR)
        tp = Path(valid) / fa["trajectories"]["path"]
        if img is None or not tp.exists():
            continue
        K0, E = camera(fa["viewpoint"], fa["image"]["size"])
        rgb, K = resize_for_cache(img[..., ::-1], K0)
        t = torch.load(tp, weights_only=False, map_location="cpu")
        vis = t["verts_inds_vis"].numpy().astype(bool)
        ins = t["instances"].numpy()
        X = t["traj_3d_world"].numpy()[vis]
        dep = zbuffer(X, E, K, rgb.shape[:2])
        actors += int((ins[vis] > 0).sum() >= 100)
        v = dep[dep > 0]
        near.append(float(np.median(v)) if v.size else 0.0)
        rows.append((fa["frame_number"], fa.get("frame_timestamp"), rgb, dep, K, E))
    if len(rows) < 2:
        return scene, 0, None
    extra = dict(camera=cam, frames_with_actors=actors, stride=every, depth="sparse tracked points")
    sw = SceneWriter(out, DATASET, scene, metric=True, synthetic=True, dynamic=True, kind="handheld", extra=extra)
    q = sw.sequence(scene, session=seq)
    for n, ts, rgb, dep, K, E in rows:
        q.add(f"{n:04d}", rgb, dep, K, E, t=ts)
    nm = np.array(near)
    return scene, sw.close(), {"actors": actors, "frames": len(rows), "dmed_lt_0.8": float(np.mean((nm > 0) & (nm < 0.8)))}


def _conv(job):
    try:
        return convert(job)
    except Exception as e:
        return job[1], f"ERROR {e!r} {traceback.format_exc(limit=3)}", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("valid", help="directory holding frame_annotations_valid.jgz and the <seq>_source_<cam> folders")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    annots = json.load(gzip.open(Path(a.valid) / "frame_annotations_valid.jgz", "rt"))
    by = defaultdict(list)
    for fa in annots:
        by[(fa["sequence_name"], fa.get("camera_name") or "left")].append(fa)
    js = [(a.valid, s, c, fr, a.out, a.every) for (s, c), fr in sorted(by.items())]
    print(f"{len(js)} sequence-cameras", flush=True)
    with Pool(a.workers) as p:
        for scene, n, info in p.imap_unordered(_conv, js):
            print(scene, n, json.dumps(info) if info else "", flush=True)


if __name__ == "__main__":
    main()
