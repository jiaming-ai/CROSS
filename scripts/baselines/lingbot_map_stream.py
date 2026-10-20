#!/usr/bin/env python3
"""LingBot-Map streaming driver of the CROSS benchmark: per-frame camera poses of one image stream.

Runs with the LingBot-Map environment's python (scripts/baselines/install_lingbot_map.sh); called by
run_lingbot_map.py.  It reproduces GCTStream.inference_streaming (lingbot_map/models/gct_stream.py: the first
`num_scale_frames` frames as one block, then one frame at a time with the KV cache; frames that are not keyframes attend
to the cache but are not appended to it) with two changes that leave the poses unchanged:
  - frames are read and preprocessed one at a time (demo.py's load_and_preprocess_images, crop mode) instead of all up
    front: a 14k-frame stream would otherwise hold ~34 GB of float images;
  - only the camera head's pose encoding is kept and the depth head is not built (its output feeds nothing else).
--reference runs upstream inference_streaming on the whole stream instead (with the depth head), to check both.
Settings: demo.py's and the upstream benchmark's defaults (benchmark/configs/methods/lingbot_map.yaml: streaming,
518 px, 8 scale frames, KV sliding window 64, max_frame_num 1024, keyframe interval "auto" = 1 up to 320 frames, else
ceil(frames / 320)); the aggregator in bf16 (fp16 before Ampere) with fp32 heads, as demo.py; FlashInfer attention
unless --use-sdpa.

    python lingbot_map_stream.py --checkpoint lingbot-map.pt --images list.txt --out poses.txt [--use-sdpa]

--images: one image path per line, in stream order.  Writes `idx` + the camera-to-world 3x4 matrix (row-major, OpenCV
camera axes) per frame, and prints LINGBOT_KEYFRAME_INTERVAL <k>, LINGBOT_FPS <frames / inference seconds> and
LINGBOT_PEAK_GPU_GB <allocated> <reserved>.
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")    # as demo.py
os.environ.setdefault("TQDM_DISABLE", "1")          # load_and_preprocess_images shows a progress bar per call

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lingbot_map.utils.load_fn import load_and_preprocess_images  # noqa: E402
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


def keyframe_interval(value, n_frames: int, threshold: int) -> int:
    """Upstream "auto" rule (benchmark/methods/lingbot_map.py _resolve_keyframe_interval, demo.py)."""
    if value in (None, 0, "0", "auto"):
        return 1 if n_frames <= threshold else (n_frames + threshold - 1) // threshold
    return int(value)


def build_model(a, device):
    """demo.py load_model (streaming), with the depth head only when asked for."""
    from lingbot_map.models.gct_stream import GCTStream
    model = GCTStream(img_size=a.image_size, patch_size=a.patch_size, enable_3d_rope=True, max_frame_num=a.max_frame_num,
                      kv_cache_sliding_window=a.kv_cache_sliding_window, kv_cache_scale_frames=a.num_scale_frames,
                      kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True, use_sdpa=a.use_sdpa,
                      camera_num_iterations=a.camera_num_iterations, enable_depth=a.with_depth or a.reference)
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    print(f"checkpoint loaded: {len(missing)} missing, {len(unexpected)} unexpected keys", flush=True)
    model = model.to(device).eval()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model.aggregator = model.aggregator.to(dtype=dtype)       # demo.py: heads stay fp32
    return model, dtype


def frames(paths, a, depth=32):
    """Preprocessed frames [1, 3, H, W] in stream order, read ahead by a background thread."""
    q: queue.Queue = queue.Queue(maxsize=depth)

    def work():
        for p in paths:
            q.put(load_and_preprocess_images([p], mode="crop", image_size=a.image_size, patch_size=a.patch_size))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (x := q.get()) is not None:
        yield x


@torch.no_grad()
def stream_poses(model, dtype, paths, kfi, a, device):
    """GCTStream.inference_streaming, pose encodings only: [S, 9]."""
    S = len(paths)
    scale = min(a.num_scale_frames, S)
    it = frames(paths, a)
    model.clean_kv_cache()
    out = []
    with torch.amp.autocast("cuda", dtype=dtype):
        first = torch.cat([next(it) for _ in range(scale)]).unsqueeze(0).to(device)
        torch.compiler.cudagraph_mark_step_begin()
        r = model.forward(first, num_frame_for_scale=scale, num_frame_per_block=scale, causal_inference=True)
        out.append(r["pose_enc"].float().cpu())
        shape = first.shape[-2:]
        del r, first
        for i in range(scale, S):
            img = next(it).unsqueeze(0).to(device, non_blocking=True)
            key = kfi <= 1 or (i - scale) % kfi == 0
            if not key:
                model._set_skip_append(True)
            torch.compiler.cudagraph_mark_step_begin()
            r = model.forward(img, num_frame_for_scale=scale, num_frame_per_block=1, causal_inference=True)
            if not key:
                model._set_skip_append(False)
            out.append(r["pose_enc"].float().cpu())
            del r
    model.clean_kv_cache()
    return torch.cat(out, dim=1), shape


@torch.no_grad()
def reference_poses(model, dtype, paths, kfi, a):
    images = load_and_preprocess_images(paths, mode="crop", image_size=a.image_size, patch_size=a.patch_size)
    with torch.amp.autocast("cuda", dtype=dtype):
        pred = model.inference_streaming(images, num_scale_frames=a.num_scale_frames, keyframe_interval=kfi,
                                         output_device=torch.device("cpu"))
    return pred["pose_enc"].float(), images.shape[-2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True, help="text file: one image path per line, in stream order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--num-scale-frames", type=int, default=8)
    ap.add_argument("--max-frame-num", type=int, default=1024)
    ap.add_argument("--kv-cache-sliding-window", type=int, default=64)
    ap.add_argument("--camera-num-iterations", type=int, default=4)
    ap.add_argument("--keyframe-interval", default="auto", help='"auto" (upstream default) or a fixed interval')
    ap.add_argument("--auto-keyframe-threshold", type=int, default=320)
    ap.add_argument("--use-sdpa", action="store_true", help="SDPA attention instead of FlashInfer")
    ap.add_argument("--with-depth", action="store_true", help="build the depth head too (check: poses unchanged)")
    ap.add_argument("--reference", action="store_true", help="upstream inference_streaming on the whole stream (check)")
    a = ap.parse_args()
    paths = [p for p in open(a.images).read().split("\n") if p]
    device = torch.device("cuda")
    t0 = time.time()
    model, dtype = build_model(a, device)
    kfi = keyframe_interval(a.keyframe_interval, len(paths), a.auto_keyframe_threshold)
    print(f"LINGBOT_KEYFRAME_INTERVAL {kfi}", flush=True)
    print(f"model loaded in {time.time() - t0:.1f} s; {len(paths)} frames", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    if a.reference:
        pose_enc, shape = reference_poses(model, dtype, paths, kfi, a)
    else:
        pose_enc, shape = stream_poses(model, dtype, paths, kfi, a, device)
    torch.cuda.synchronize()
    dt = time.time() - t1
    print(f"LINGBOT_FPS {len(paths) / max(dt, 1e-9):.3f}", flush=True)
    print(f"LINGBOT_PEAK_GPU_GB {torch.cuda.max_memory_allocated() / 1e9:.3f} {torch.cuda.max_memory_reserved() / 1e9:.3f}",
          flush=True)
    print(f"LINGBOT_IMAGE_HW {shape[0]} {shape[1]}", flush=True)
    c2w, _ = pose_encoding_to_extri_intri(pose_enc, tuple(shape))     # camera-to-world [1, S, 3, 4]
    c2w = c2w[0].double().numpy()
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        for i, T in enumerate(c2w):
            f.write(f"{i} " + " ".join(f"{v:.9f}" for v in np.asarray(T).reshape(-1)) + "\n")
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
