"""Run image-only CROSS; evaluate separately with cross.mono.evaluate."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .config import MonoConfig, ScaleConfig
from .data import RGBSequence
from .frontend import MonoFrontend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontend-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--metric-resolution", type=int, default=504)
    parser.add_argument("--anchor-interval", type=int, default=30)
    parser.add_argument("--metric-interval", type=int, default=30)
    parser.add_argument("--mapping-interval", type=int, default=5)
    parser.add_argument("--pose-model", default="depth-anything/DA3-SMALL")
    parser.add_argument("--metric-model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--scale-mode", choices=["filtered", "direct", "initial", "relative"], default="filtered")
    parser.add_argument("--intrinsics", nargs="+", type=float)
    parser.add_argument("--no-undistort", action="store_true")
    parser.add_argument("--load-map", type=Path)
    parser.add_argument("--save-map", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cv2.setRNGSeed(args.seed)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    if os.environ.get("CROSS_TORCH_HUB"):
        torch.hub.set_dir(os.environ["CROSS_TORCH_HUB"])
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite previous results: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    config = MonoConfig(pose_model=args.pose_model, metric_model=args.metric_model,
                        resolution=args.resolution, metric_resolution=args.metric_resolution,
                        anchor_interval=args.anchor_interval, mapping_interval=args.mapping_interval,
                        scale=ScaleConfig(interval=args.metric_interval, mode=args.scale_mode))
    sequence = RGBSequence(args.sequence, args.stride, args.start, args.frames, args.intrinsics, not args.no_undistort)
    if not len(sequence):
        raise ValueError("No selected images")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = os.environ.get("CROSS_SOURCE_COMMIT", "unavailable")
    metadata = {"config": asdict(config), "command": vars(args).copy(), "source_commit": commit,
                "hostname": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
                "selected_frames": len(sequence), "dataset_rgb_frames": sequence.total_frames,
                "input_modalities": ["RGB", "timestamps", "camera_intrinsics"],
                "ground_truth_used_for_inference": False, "sensor_depth_used": False,
                "calibration": sequence.K.tolist(), "distortion": sequence.distortion.tolist()}
    source_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((source_root / "cross").rglob("*.py")):
        digest.update(str(path.relative_to(source_root)).encode())
        digest.update(path.read_bytes())
    metadata["python_source_sha256"] = digest.hexdigest()
    metadata["command"] = {k: str(v) if isinstance(v, Path) else v for k, v in metadata["command"].items()}
    if args.device.startswith("cuda"):
        metadata["gpu"] = torch.cuda.get_device_name()
        torch.cuda.reset_peak_memory_stats()
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    loading_start = perf_counter()
    frontend = MonoFrontend(sequence.K, config, args.device)
    metadata["model_parameters"] = {
        "geometry": sum(p.numel() for p in frontend.geometry.model.parameters()),
        "metric": sum(p.numel() for p in frontend.metric.model.parameters()) if frontend.metric else 0,
    }
    from huggingface_hub import try_to_load_from_cache
    metadata["model_checkpoint_files"] = {
        model: str(try_to_load_from_cache(model, "model.safetensors"))
        for model in [config.pose_model, config.metric_model] if model
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    tracker = frontend
    if not args.frontend_only:
        from .system import MonocularSystem
        first = next(iter(sequence))
        tracker = MonocularSystem(sequence.K, (first.rgb.shape[1], first.rgb.shape[0]), config,
                                  device=args.device, frontend=frontend)
        if args.load_map:
            tracker.load_map(args.load_map)
    loading_seconds = perf_counter() - loading_start
    latencies, valid_count, metric_count = [], 0, 0
    start = perf_counter()
    try:
        with (args.output / "trajectory.txt").open("w") as trajectory, \
                (args.output / "frontend_trajectory.txt").open("w") as front_trajectory, \
                (args.output / "diagnostics.jsonl").open("w") as diagnostics:
            for frame in sequence:
                frame_start = perf_counter()
                estimate = tracker.step(frame.rgb, frame.timestamp)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = perf_counter() - frame_start
                latencies.append(elapsed)
                valid_count += bool(estimate.diagnostics["valid"])
                metric_count += "scale_observation" in estimate.diagnostics
                estimate.diagnostics.update(input_index=frame.index, wall_seconds=elapsed)
                for handle, pose in [(trajectory, estimate.pose), (front_trajectory, frontend.metric_pose)]:
                    row = np.r_[frame.timestamp, pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]
                    handle.write(" ".join(f"{x:.9f}" for x in row) + "\n")
                    handle.flush()
                diagnostics.write(json.dumps(estimate.diagnostics, allow_nan=False) + "\n")
                diagnostics.flush()
                if len(latencies) % 50 == 0:
                    print(json.dumps({"frames": len(latencies), "valid": valid_count, "scale": frontend.scale_filter.scale,
                                      "fps": len(latencies) / (perf_counter() - start)}), flush=True)
        if args.save_map and not args.frontend_only:
            tracker.save_map(args.output / "map.pkl")
    finally:
        if hasattr(tracker, "shutdown"):
            tracker.shutdown()
    total = perf_counter() - start
    summary = {"frames": len(latencies), "valid_frames": valid_count, "tracking_coverage": valid_count / len(latencies),
               "metric_calls": metric_count, "model_loading_seconds": loading_seconds,
               "elapsed_seconds": total, "fps_including_io": len(latencies) / total,
               "latency_median_ms": 1000 * float(np.median(latencies)),
               "latency_p95_ms": 1000 * float(np.quantile(latencies, 0.95)),
               "scale": frontend.scale_filter.scale, "accepted_scale_observations": frontend.scale_filter.accepted,
               "rejected_scale_observations": frontend.scale_filter.rejected}
    if args.device.startswith("cuda"):
        summary["peak_gpu_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
