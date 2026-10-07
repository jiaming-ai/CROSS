"""HOI4D (HF mirror yinloonga/HOI4D: HOI4D_release.zip with align_rgb/image.mp4 only, HOI4D_annotations.zip,
camera_params.zip) -> scene cache, with depth rendered from the sequence's reconstructed scene cloud.

Input (extracted): <rel>/HOI4D_release/<cam>/H*/C*/N*/S*/s*/T*/align_rgb/image.mp4 (1920 x 1080, 300 frames, 15 fps);
<ann>/HOI4D_annotations/<same>/3Dseg/{output.log (Open3D trajectory log: 300 camera-to-world poses, frame 0 = identity),
raw_pc.pcd (static scene cloud, binary x y z rgb, ~1e5 points), label.pcd}, 2Dseg/mask/NNNNN.png (per-frame colour
masks: black = background, the other colours = hand / manipulated object); <cam>/camera_params/<cam>/intrin.npy
(3 x 3 for 1920 x 1080).

Depth: the cloud projected with a z-buffer at the stored resolution (prep/projection.py), points showing through gaps of
nearer surfaces removed, and the pixels of the 2D masks (dilated) invalidated: hands and the manipulated object move,
the cloud is static.  No depth video in the mirror.  Licence: CC BY-NC 4.0 (HOI4D), whatever the mirror's tag says.

Checked on 8 sequences of different categories:
- output.log is camera-to-world: the grey level of the projected cloud points correlates with the video 0.63-0.97 this
  way vs -0.22-0.87 the other way.
- The trajectory index is the video frame (offset 0 best of -6..+6).
- Another scene's cloud gives -0.44-0.2; two exceptions at 0.5-0.7 are for sequences that themselves score 0.35-0.51
  and are visibly misaligned.
- Metric: the objpose sizes from the depth sensor (mug 10.0 x 12.7 x 8.8 cm, kettle 25.8 x 23.5 x 16.7 cm) are real
  sizes, at z 0.88-0.96 m in the first camera like the projected cloud (0.84-0.93 m).

A sequence is kept when the median colour agreement over its kept frames is >= --min-agree (0.6), using only frames
with a 2D mask; without one, the hands' and object's pixels would get the background depth. 1022 of the 2973
sequences have masks. Mask colours (VOC palette, BGR): (0, 128, 0) = hand, the others = object parts.

python -m vggt_ft.prep.hoi4d <release_root> <annotations_root> <camera_params_root> <cache_root> [--every 5]
    [--workers 32] [--min-agree 0.6]
"""
from __future__ import annotations

import argparse
import json
import traceback
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import SceneWriter
from .projection import colour_agreement, drop_see_through, resize_for_cache, zbuffer

DATASET = "hoi4d"


def read_pcd(path, colours: bool = False):
    """x y z (N, 3) of a PCD file (ascii or binary); with colours also the points' RGB (N, 3) uint8 (packed `rgb`)."""
    raw = Path(path).read_bytes()
    head, i = {}, 0
    while True:
        j = raw.index(b"\n", i)
        line = raw[i:j].decode("ascii", "replace").strip()
        i = j + 1
        if line and not line.startswith("#"):
            k, *v = line.split()
            head[k.upper()] = v
            if k.upper() == "DATA":
                break
    fields, sizes, types = head["FIELDS"], [int(s) for s in head["SIZE"]], head["TYPE"]
    counts = [int(c) for c in head.get("COUNT", ["1"] * len(fields))]
    n = int(head["POINTS"][0])
    if head["DATA"][0] == "ascii":
        a = np.loadtxt(raw[i:].decode().splitlines(), dtype=np.float64, ndmin=2)
        return a[:, :3].astype(np.float32)
    if head["DATA"][0] != "binary":
        raise ValueError(f"{path}: PCD {head['DATA'][0]} not supported")
    dt = np.dtype([(f, {("F", 4): "<f4", ("F", 8): "<f8", ("U", 4): "<u4", ("U", 1): "u1", ("I", 4): "<i4"}[(t, s)],
                    (c,) if c > 1 else ()) for f, s, t, c in zip(fields, sizes, types, counts)])
    a = np.frombuffer(raw, dt, count=n, offset=i)
    xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float32)
    ok = np.isfinite(xyz).all(1)
    if not colours:
        return xyz[ok]
    packed = a["rgb"].view(np.uint32) if "rgb" in a.dtype.names else np.zeros(n, np.uint32)
    rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], 1).astype(np.uint8)
    return xyz[ok], rgb[ok]


def read_log(path) -> np.ndarray:
    """Open3D trajectory .log: blocks of '<i> <j> <n>' + 4 rows; returns (F, 4, 4) camera-to-world."""
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    out = []
    for b in range(0, len(rows) - 4, 5):
        out.append(np.array(rows[b + 1:b + 5], np.float64))
    return np.stack(out)


def mask_pixels(mask_bgr: np.ndarray, hw, dilate_px: int = 6) -> np.ndarray:
    """Non-background pixels of an HOI4D 2D mask at the stored resolution (dilated), bool HxW."""
    m = (mask_bgr.max(axis=2) > 0).astype(np.uint8)
    m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    if dilate_px:
        m = cv2.dilate(m, np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8))
    return m > 0


def convert(job):
    rel_seq, ann_seq, intr_path, out, every, min_agree = job
    rel_seq, ann_seq = Path(rel_seq), Path(ann_seq)
    parts = rel_seq.parts[rel_seq.parts.index("HOI4D_release") + 1:]
    scene = "_".join(parts)
    if (Path(out) / DATASET / scene / "scene.json").exists():
        return scene, -1, None
    poses = read_log(ann_seq / "3Dseg" / "output.log")
    pts, col = read_pcd(ann_seq / "3Dseg" / "raw_pc.pcd", colours=True)
    K0 = np.load(intr_path).astype(np.float64)
    cap = cv2.VideoCapture(str(rel_seq / "align_rgb" / "image.mp4"))
    frames, f = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if f % every == 0 and f < len(poses):
            frames.append((f, bgr))
        f += 1
    cap.release()
    if len(frames) < 2:
        return scene, 0, None
    rows, agree, hands, objs = [], [], 0, 0
    for f, bgr in frames:
        rgb, K = resize_for_cache(bgr[..., ::-1], K0)
        hw = rgb.shape[:2]
        mp = ann_seq / "2Dseg" / "mask" / f"{f:05d}.png"
        mbgr = cv2.imread(str(mp), cv2.IMREAD_COLOR) if mp.exists() else None
        if mbgr is None:
            continue                        # no mask: hands / moving object cannot be removed from the depth
        mask = mask_pixels(mbgr, hw)
        hand = (mbgr == (0, 128, 0)).all(-1)
        hands += int(hand.sum() >= 1000)
        objs += int(((mbgr.max(-1) > 0) & ~hand).sum() >= 1000)
        E = np.linalg.inv(poses[f])
        d, idx = zbuffer(pts, E, K, hw, return_index=True)
        a = colour_agreement(rgb, idx, col, mask)
        if np.isfinite(a):
            agree.append(a)
        d = drop_see_through(d)
        d[mask] = 0
        rows.append((f, rgb, d, K, E))
    a_med = float(np.median(agree)) if agree else float("nan")
    info = {"agree": round(a_med, 3), "frames": len(rows), "hand_frames": hands, "object_frames": objs}
    if len(rows) < 2 or not (a_med >= min_agree):
        return scene, 0, info
    extra = dict(camera_id=parts[0], colour_agreement=a_med, n_points=int(len(pts)), hand_frames=hands,
                 object_frames=objs, stride=every)
    sw = SceneWriter(out, DATASET, scene, metric=True, synthetic=False, dynamic=True, kind="egocentric_hoi", extra=extra)
    q = sw.sequence(scene)
    for f, rgb, d, K, E in rows:
        q.add(f"{f:05d}", rgb, d, K, E, t=f / 15.0)
    return scene, sw.close(), info


def _single_thread():
    """One thread per worker: torch / OpenCV / OpenMP thread pools in every worker oversubscribed the CPUs ~50x."""
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def _conv(job):
    try:
        return convert(job)
    except Exception as e:
        return Path(job[0]).name, f"ERROR {e!r} {traceback.format_exc(limit=3)}", None


def jobs(rel_root, ann_root, cam_root, out, every, min_agree, limit=None):
    out_jobs = []
    for mp4 in sorted(Path(rel_root).glob("HOI4D_release/*/H*/C*/N*/S*/s*/T*/align_rgb/image.mp4")):
        seq = mp4.parent.parent
        rel = seq.relative_to(Path(rel_root) / "HOI4D_release")
        ann = Path(ann_root) / "HOI4D_annotations" / rel
        intr = Path(cam_root) / "camera_params" / rel.parts[0] / "intrin.npy"
        if (ann / "3Dseg" / "output.log").exists() and (ann / "3Dseg" / "raw_pc.pcd").exists() and intr.exists() \
                and (ann / "2Dseg" / "mask").is_dir():
            out_jobs.append((str(seq), str(ann), str(intr), out, every, min_agree))
    return out_jobs[:limit] if limit else out_jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rel")
    ap.add_argument("ann")
    ap.add_argument("cam")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--min-agree", type=float, default=0.6, help="median colour agreement of a kept sequence")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    js = jobs(a.rel, a.ann, a.cam, a.out, a.every, a.min_agree, a.limit)
    print(f"{len(js)} sequences", flush=True)
    with Pool(a.workers, initializer=_single_thread) as p:
        for scene, n, info in p.imap_unordered(_conv, js):
            print(scene, n, json.dumps(info) if info else "", flush=True)


if __name__ == "__main__":
    main()
