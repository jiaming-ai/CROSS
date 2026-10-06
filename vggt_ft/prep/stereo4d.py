"""Stereo4D (Jin et al., CVPR 2025: internet VR180 videos) -> scene cache with metric depth from FoundationStereo.

Inputs
- rectified 512x512, 60 deg HFOV perspective videos of both eyes (HF KevinMathew/stereo4d-{left,right}eye-perspective,
  made with Stereo4D's rectify.py): <raw>/left/<clip>-left_rectified.mp4, <raw>/right/<clip>-right_rectified.mp4;
- camera poses of the rectified left camera, one per video frame (Stereo4D annotations, gs://stereo4d/<split>/<clip>.npz,
  `camera2world`; vggt_ft does not need the 3D tracks): an npz with arrays <clip>__camera2world, <clip>__timestamps.
Depth: FoundationStereo disparity of the left view, left-right consistent pixels with disparity >= --min_disp px,
z = fx * 0.063 / d (0.063 m: the VR180 inter-camera distance Stereo4D assumes).  Intrinsics: fx = fy = 256 / tan(30 deg),
principal point at the image centre.  One scene per clip (a clip is one place and one session).

Scale check (--check_weights): the 63 mm baseline is an assumption, and VR180 cameras / edits differ.  Per clip, the
median DA3-metric / stereo depth ratio (every 3rd written frame) and the camera speed (from the poses, which use the
same baseline) are stored in scene.json["check"].  A clip stays metric when the ratio is within --max_ratio of 1 or its
straight-line speed is a walk (0.7-2.5 m/s, independent of the depth prior); otherwise it is written with metric=False
(pose / depth losses on normalised GT still apply, the scale head is not trained on it).  On 39 test clips the ratio
was 0.96 median, 54 % within 1.25x, 79 % within 1.5x; walking clips gave 1.4-1.8 m/s (DA3 scale 1.0-1.5 m/s: DA3
under-estimates outdoor scale), the large disagreements (DA3 1.5-3.1x farther) were all static clips.

python -m vggt_ft.prep.stereo4d <raw> <poses.npz> <root> --fs_code <FoundationStereo dir> --fs_ckpt <dir with cfg.yaml>
    [--stride 4] [--max_frames 40] [--shard i/n] [--batch 8] [--check_weights <labeler weights dir>]
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

from vggt_ft.prep.common import SceneWriter, write_index
from vggt_ft.stereo import FoundationStereo, disparity_to_depth

BASELINE = 0.063
SIZE = 512
FX = SIZE / 2 / math.tan(math.radians(30))
K = np.array([[FX, 0, (SIZE - 1) / 2], [0, FX, (SIZE - 1) / 2], [0, 0, 1]])


def read_frames(path: Path, idx) -> list:
    cap = cv2.VideoCapture(str(path))
    want, out, i = set(int(j) for j in idx), {}, 0
    while len(out) < len(want):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return [out.get(int(j)) for j in idx]


def n_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def clip_check(written, c2w, ts, da3):
    """DA3-metric / stereo median depth ratio of every 3rd written frame, path and straight-line speed (m/s)."""
    import torch
    out = {}
    if da3 is not None:
        r = []
        for rgb, depth in written[::3]:
            with torch.no_grad():
                L = da3(torch.from_numpy(rgb).cuda().permute(2, 0, 1).float() / 255,
                        torch.from_numpy(K).float().cuda()).cpu().numpy()
            m = (depth > 0) & np.isfinite(L) & (L > 0)
            if m.sum() > 200:
                r.append(float(np.median(L[m] / depth[m])))
        if r:
            out["da3_over_stereo"] = float(np.median(r))
    if ts is not None and len(ts) > 1:
        t = c2w[:, :, 3]
        dur = max(float(ts[-1] - ts[0]) * 1e-6, 1e-3)
        out["speed"] = float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum() / dur)
        out["straight_speed"] = float(np.linalg.norm(t[-1] - t[0]) / dur)
    return out


def convert_clip(clip, raw, poses, root, fs, stride, max_frames, min_disp, batch, da3=None, max_ratio=1.5):
    left, right = raw / "left" / f"{clip}-left_rectified.mp4", raw / "right" / f"{clip}-right_rectified.mp4"
    c2w = poses[f"{clip}__camera2world"]
    T = len(c2w)
    nl, nr = n_frames(left), n_frames(right)
    if min(nl, nr) < T - 2 or abs(nl - nr) > 2:          # the videos have one frame per annotated timestamp
        return 0, f"frames {nl}/{nr} vs {T} poses"
    n = min(nl, nr, T)
    idx = np.arange(0, n, stride)
    if len(idx) > max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, max_frames).round().astype(int)]
    L, R = read_frames(left, idx), read_frames(right, idx)
    keep = [k for k in range(len(idx)) if L[k] is not None and R[k] is not None and L[k].shape[:2] == (SIZE, SIZE)]
    if len(keep) < 2:
        return 0, "unreadable"
    ts = poses.get(f"{clip}__timestamps")
    sw = SceneWriter(root, "stereo4d", clip, metric=True, synthetic=False, dynamic=True, kind="wild",
                     extra={"source": "Stereo4D (HF KevinMathew/stereo4d-*eye-perspective, gs://stereo4d poses)",
                            "depth_src": f"FoundationStereo 23-51-11, left-right consistent, disparity >= {min_disp} px, "
                                         f"baseline {BASELINE} m (assumed VR180)"})
    q = sw.sequence("clip")
    written = []
    for b0 in range(0, len(keep), batch):
        ks = keep[b0:b0 + batch]
        disp, ok = fs.disparity(np.stack([L[k] for k in ks]), np.stack([R[k] for k in ks]))
        for j, k in enumerate(ks):
            i = int(idx[k])
            depth = disparity_to_depth(disp[j], ok[j], FX, BASELINE, min_disp)
            if (depth > 0).mean() < 0.05:
                continue
            E = np.eye(4)
            E[:3, :4] = c2w[i]
            E = np.linalg.inv(E)                               # camera-from-world (Stereo4D's extrs_rectified)
            t = None if ts is None else float(ts[i]) * 1e-6
            q.add(f"{i:05d}", L[k], depth, K, E, t)
            written.append((L[k], depth))
    chk = clip_check(written, c2w, ts, da3)
    r, walk = chk.get("da3_over_stereo"), 0.7 <= chk.get("straight_speed", 0) <= 2.5
    chk["metric_ok"] = bool(walk or (r is not None and abs(math.log(r)) <= math.log(max_ratio)))
    sw.meta["check"] = chk
    sw.meta["metric"] = chk["metric_ok"]
    return sw.close(), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("poses")
    ap.add_argument("root")
    ap.add_argument("--fs_code", required=True)
    ap.add_argument("--fs_ckpt", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=40)
    ap.add_argument("--min_disp", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument("--check_weights", default="", help="labeler weights dir (vggt_ft/labelers.py) for the scale check")
    ap.add_argument("--max_ratio", type=float, default=1.5)
    a = ap.parse_args()
    raw, root = Path(a.raw), Path(a.root)
    poses = dict(np.load(a.poses))
    clips = sorted({k.split("__")[0] for k in poses if k.endswith("__camera2world")})
    clips = [c for c in clips if (raw / "left" / f"{c}-left_rectified.mp4").exists()
             and (raw / "right" / f"{c}-right_rectified.mp4").exists()]
    if a.max_clips:
        clips = clips[:a.max_clips]
    si, sn = map(int, a.shard.split("/"))
    clips = clips[si::sn]
    fs = FoundationStereo(a.fs_code, a.fs_ckpt)
    da3 = None
    if a.check_weights:
        from vggt_ft.labelers import load_labelers
        da3 = load_labelers(["da3metric"], a.check_weights)["da3metric"]
    t0, frames, done, skipped = time.time(), 0, 0, []
    for j, clip in enumerate(clips):
        if (root / "stereo4d" / clip / "scene.json").exists():
            continue
        n, why = convert_clip(clip, raw, poses, root, fs, a.stride, a.max_frames, a.min_disp, a.batch, da3, a.max_ratio)
        frames += n
        done += n > 0
        if not n:
            skipped.append((clip, why))
        if (j + 1) % 20 == 0 or j + 1 == len(clips):
            print(f"shard {a.shard}: {j + 1}/{len(clips)} clips, {done} written, {frames} frames, "
                  f"{len(skipped)} skipped, {time.time() - t0:.0f}s", flush=True)
    if skipped:
        print("skipped (first 10):", skipped[:10], flush=True)
    if sn == 1:
        write_index(root / "stereo4d")


if __name__ == "__main__":
    main()
