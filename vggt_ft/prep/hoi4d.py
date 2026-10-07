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

python -m vggt_ft.prep.hoi4d <release_root> <annotations_root> <camera_params_root> <cache_root> [--every 5]
    [--workers 32] [--min-align 0.0]
"""
from __future__ import annotations

import argparse
import json
import traceback
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .common import SceneWriter
from .projection import drop_see_through, edge_alignment, resize_for_cache, zbuffer

DATASET = "hoi4d"


def read_pcd(path) -> np.ndarray:
    """x y z of a PCD file (ascii or binary, float32 x y z first), (N, 3)."""
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
    return xyz[np.isfinite(xyz).all(1)]


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


def render(points, pose_c2w, K, hw, mask=None):
    E = np.linalg.inv(pose_c2w)
    d = drop_see_through(zbuffer(points, E, K, hw))
    if mask is not None:
        d[mask] = 0
    return d, E


def convert(job):
    rel_seq, ann_seq, intr_path, out, every, min_align = job
    rel_seq, ann_seq = Path(rel_seq), Path(ann_seq)
    parts = rel_seq.parts[rel_seq.parts.index("HOI4D_release") + 1:]
    scene = "_".join(parts)
    if (Path(out) / DATASET / scene / "scene.json").exists():
        return scene, -1, None
    poses = read_log(ann_seq / "3Dseg" / "output.log")
    pts = read_pcd(ann_seq / "3Dseg" / "raw_pc.pcd")
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
    rows, align, hands = [], [], 0
    for f, bgr in frames:
        rgb, K = resize_for_cache(bgr[..., ::-1], K0)
        hw = rgb.shape[:2]
        mp = ann_seq / "2Dseg" / "mask" / f"{f:05d}.png"
        mbgr = cv2.imread(str(mp), cv2.IMREAD_COLOR) if mp.exists() else None
        if mbgr is None:
            continue                        # no mask: hands / moving object cannot be removed from the depth
        mask = mask_pixels(mbgr, hw)
        hands += int(mbgr.max() > 0)
        d, E = render(pts, poses[f], K, hw, mask)
        a = edge_alignment(rgb, d)
        if np.isfinite(a):
            align.append(a)
        rows.append((f, rgb, d, K, E))
    a_med = float(np.median(align)) if align else float("nan")
    if len(rows) < 2 or not (a_med >= min_align):
        return scene, 0, {"align": a_med, "frames": len(rows)}
    extra = dict(camera_id=parts[0], align=a_med, n_points=int(len(pts)), masked_frames=hands, stride=every)
    sw = SceneWriter(out, DATASET, scene, metric=True, synthetic=False, dynamic=True, kind="egocentric_hoi", extra=extra)
    q = sw.sequence(scene)
    for f, rgb, d, K, E in rows:
        q.add(f"{f:05d}", rgb, d, K, E, t=f / 15.0)
    return scene, sw.close(), {"align": a_med, "frames": len(rows), "masked": hands}


def _conv(job):
    try:
        return convert(job)
    except Exception as e:
        return Path(job[0]).name, f"ERROR {e!r} {traceback.format_exc(limit=3)}", None


def jobs(rel_root, ann_root, cam_root, out, every, min_align, limit=None):
    out_jobs = []
    for mp4 in sorted(Path(rel_root).glob("HOI4D_release/*/H*/C*/N*/S*/s*/T*/align_rgb/image.mp4")):
        seq = mp4.parent.parent
        rel = seq.relative_to(Path(rel_root) / "HOI4D_release")
        ann = Path(ann_root) / "HOI4D_annotations" / rel
        intr = Path(cam_root) / "camera_params" / rel.parts[0] / "intrin.npy"
        if (ann / "3Dseg" / "output.log").exists() and (ann / "3Dseg" / "raw_pc.pcd").exists() and intr.exists():
            out_jobs.append((str(seq), str(ann), str(intr), out, every, min_align))
    return out_jobs[:limit] if limit else out_jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rel")
    ap.add_argument("ann")
    ap.add_argument("cam")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--min-align", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    js = jobs(a.rel, a.ann, a.cam, a.out, a.every, a.min_align, a.limit)
    print(f"{len(js)} sequences", flush=True)
    with Pool(a.workers) as p:
        for scene, n, info in p.imap_unordered(_conv, js):
            print(scene, n, json.dumps(info) if info else "", flush=True)


if __name__ == "__main__":
    main()
