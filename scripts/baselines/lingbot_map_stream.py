#!/usr/bin/env python3
"""LingBot-Map streaming driver of the CROSS benchmark: per-frame camera poses of one image stream.

Runs with the LingBot-Map environment's python (scripts/baselines/install_lingbot_map.sh); called by
run_lingbot_map.py.  It reproduces GCTStream.inference_streaming (lingbot_map/models/gct_stream.py: the first
`num_scale_frames` frames as one block, then one frame at a time with the KV cache; frames that are not keyframes attend
to the cache but are not appended to it) with two changes that leave the poses unchanged:
  - frames are read and preprocessed one at a time (demo.py's load_and_preprocess_images, crop mode) instead of all up
    front: a 14k-frame stream would otherwise hold ~34 GB of float images;
  - only the camera head's pose encoding is kept and the depth head is not built (its output feeds nothing else).
--reference runs upstream inference_streaming on the whole stream instead (with the depth head), to check both
(2026-10-20, OpenLORIS home1-4_f400, keyframe interval 2: all 400 poses bit-identical).
--trial-images: several streams that share their first frames (a map session followed by each T3 trial): the shared
part is computed once and the streaming state restored before each trial, which gives every trial the poses of its own
full stream, at a fraction of the compute (SDPA backend only; the state of the FlashInfer cache is not copied).
Settings: demo.py's and the upstream benchmark's defaults (benchmark/configs/methods/lingbot_map.yaml: streaming,
518 px, 8 scale frames, KV sliding window 64, max_frame_num 1024, keyframe interval "auto" = 1 up to 320 frames, else
ceil(frames / 320)); the aggregator in bf16 (fp16 before Ampere) with fp32 heads, as demo.py; FlashInfer attention
unless --use-sdpa.

    python lingbot_map_stream.py --checkpoint lingbot-map.pt --images list.txt --out poses.txt [--use-sdpa]
    python lingbot_map_stream.py --checkpoint lingbot-map.pt --images map.txt --use-sdpa \
        --trial-images t0.txt t1.txt --trial-out poses_t0.txt poses_t1.txt

--images: one image path per line, in stream order.  Writes `idx` + the camera-to-world 3x4 matrix (row-major, OpenCV
camera axes) per frame of the stream (with --trial-images: of the map + trial stream, per trial), and prints
LINGBOT_KEYFRAME_INTERVAL <k>, LINGBOT_FPS <frames / inference seconds> and LINGBOT_PEAK_GPU_GB <allocated> <reserved>.
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
def stream_poses(model, dtype, paths, kfi, a, device, start=0, scale=None):
    """GCTStream.inference_streaming, pose encodings only: [1, len(paths), 9].  start > 0 continues a stream whose first
    `start` frames the model has processed (its caches hold them, see snapshot()); `scale`: that stream's number of scale
    frames."""
    S = len(paths)
    it = frames(paths, a)
    out, shape = [], None
    with torch.amp.autocast("cuda", dtype=dtype):
        if start == 0:
            model.clean_kv_cache()
            scale = min(a.num_scale_frames, S)
            first = torch.cat([next(it) for _ in range(scale)]).unsqueeze(0).to(device)
            torch.compiler.cudagraph_mark_step_begin()
            r = model.forward(first, num_frame_for_scale=scale, num_frame_per_block=scale, causal_inference=True)
            out.append(r["pose_enc"].float().cpu())
            shape = first.shape[-2:]
            del r, first
        for i in range(max(start, scale), start + S):
            img = next(it).unsqueeze(0).to(device, non_blocking=True)
            shape = img.shape[-2:]
            key = kfi <= 1 or (i - scale) % kfi == 0
            if not key:
                model._set_skip_append(True)
            torch.compiler.cudagraph_mark_step_begin()
            r = model.forward(img, num_frame_for_scale=scale, num_frame_per_block=1, causal_inference=True)
            if not key:
                model._set_skip_append(False)
            out.append(r["pose_enc"].float().cpu())
            del r
    return torch.cat(out, dim=1), shape


def _tree(x, device):
    """Copy of a nest of dicts / lists / tuples of tensors, the tensors on `device`."""
    if torch.is_tensor(x):
        return x.detach().to(device, copy=True)
    if isinstance(x, dict):
        return {k: _tree(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_tree(v, device) for v in x)
    return x


def snapshot(model):
    """The streaming state (SDPA backend): the aggregator's KV cache and frame count, the camera head's KV cache and
    frame index; on the CPU."""
    return {"agg_kv": _tree(model.aggregator.kv_cache, "cpu"), "agg_n": model.aggregator.total_frames_processed,
            "cam_kv": _tree(model.camera_head.kv_cache, "cpu"), "cam_i": model.camera_head.frame_idx}


def restore(model, snap, device):
    model.aggregator.kv_cache = _tree(snap["agg_kv"], device)
    model.aggregator.total_frames_processed = snap["agg_n"]
    model.camera_head.kv_cache = _tree(snap["cam_kv"], device)
    model.camera_head.frame_idx = snap["cam_i"]


def write_poses(path, pose_enc, shape):
    c2w, _ = pose_encoding_to_extri_intri(pose_enc, tuple(shape))     # camera-to-world [1, S, 3, 4]
    c2w = c2w[0].double().numpy()
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        for i, T in enumerate(c2w):
            f.write(f"{i} " + " ".join(f"{v:.9f}" for v in np.asarray(T).reshape(-1)) + "\n")
    os.replace(tmp, path)


def run_trials(model, dtype, paths, a, device):
    """Map + trial streams that share the map frames (--trial-images): the map part is computed once per keyframe
    interval and the streaming state restored before each trial.  The model is causal and the state is restored
    exactly, so every trial's poses are those of its own map + trial stream (same keyframe interval, the auto rule on
    the length of that stream)."""
    trials = [[p for p in open(t).read().split("\n") if p] for t in a.trial_images]
    groups = {}
    for k, t in enumerate(trials):
        groups.setdefault(keyframe_interval(a.keyframe_interval, len(paths) + len(t), a.auto_keyframe_threshold), []).append(k)
    n = 0
    for kfi, ks in groups.items():
        print(f"LINGBOT_KEYFRAME_INTERVAL {kfi} (trials {ks})", flush=True)
        scale = min(a.num_scale_frames, len(paths))
        enc_map, shape = stream_poses(model, dtype, paths, kfi, a, device)
        n += len(paths)
        snap = snapshot(model)
        size = sum(v.numel() * v.element_size() for v in _leaves(snap))
        print(f"state snapshot {size / 1e9:.2f} GB", flush=True)
        for k in ks:
            model.clean_kv_cache()              # free the previous trial's state before the copy (peak memory)
            torch.cuda.empty_cache()
            restore(model, snap, device)
            enc_t, _ = stream_poses(model, dtype, trials[k], kfi, a, device, start=len(paths), scale=scale)
            n += len(trials[k])
            write_poses(a.trial_out[k], torch.cat([enc_map, enc_t], dim=1), shape)
        del snap
        model.clean_kv_cache()
    return n, shape


def _leaves(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _leaves(v)


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
    ap.add_argument("--out", help="pose file of the --images stream (without --trial-images)")
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
    ap.add_argument("--trial-images", nargs="*", default=[], help="trial streams (lists like --images) that follow the "
                    "--images frames, each written to the --trial-out file at its position (map part computed once)")
    ap.add_argument("--trial-out", nargs="*", default=[])
    a = ap.parse_args()
    paths = [p for p in open(a.images).read().split("\n") if p]
    assert len(a.trial_images) == len(a.trial_out), "--trial-images and --trial-out differ in length"
    assert a.trial_images or a.out, "--out is needed without --trial-images"
    assert not a.trial_images or a.use_sdpa, "trial streams restore the SDPA backend's state: use --use-sdpa"
    device = torch.device("cuda")
    t0 = time.time()
    model, dtype = build_model(a, device)
    print(f"model loaded in {time.time() - t0:.1f} s; {len(paths)} frames", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    if a.trial_images:
        n, shape = run_trials(model, dtype, paths, a, device)
    else:
        kfi = keyframe_interval(a.keyframe_interval, len(paths), a.auto_keyframe_threshold)
        print(f"LINGBOT_KEYFRAME_INTERVAL {kfi}", flush=True)
        if a.reference:
            pose_enc, shape = reference_poses(model, dtype, paths, kfi, a)
        else:
            pose_enc, shape = stream_poses(model, dtype, paths, kfi, a, device)
        n = len(paths)
        write_poses(a.out, pose_enc, shape)
    torch.cuda.synchronize()
    dt = time.time() - t1
    print(f"LINGBOT_FPS {n / max(dt, 1e-9):.3f}", flush=True)
    print(f"LINGBOT_PEAK_GPU_GB {torch.cuda.max_memory_allocated() / 1e9:.3f} {torch.cuda.max_memory_reserved() / 1e9:.3f}",
          flush=True)
    print(f"LINGBOT_IMAGE_HW {shape[0]} {shape[1]}", flush=True)


if __name__ == "__main__":
    main()
