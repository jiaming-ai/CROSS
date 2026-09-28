"""Run image-only CROSS; evaluate separately with cross.mono.evaluate."""

import argparse
from dataclasses import asdict
import hashlib
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter, sleep

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
    parser.add_argument("--frontend", choices=["da3", "dpvo", "metric_pnp", "rotation_metric", "metric_klt", "streaming_pnp"], default="da3")
    parser.add_argument("--input-fps", type=float, help="Pose deadline rate; also pace arrivals uniformly unless --replay-timestamps")
    parser.add_argument("--sample-fps", type=float, help="Select the first RGB in each timestamp bin at this rate; keep original timestamps")
    parser.add_argument("--replay-timestamps", action="store_true", help="Pace arrivals with original RGB timestamp differences; requires the paced input worker")
    parser.add_argument("--warmup-models", action="store_true", help="Warm models using only the first RGB image; report startup separately")
    parser.add_argument("--paced-input-worker", action="store_true", help="Overlap RGB preprocessing with tracking; bounded queue, fail on overflow")
    parser.add_argument("--input-process", action="store_true", help="Run paced RGB acquisition in an independent CPU process")
    parser.add_argument("--input-buffer", type=int, default=2, help="Maximum pending input frames; overflow fails instead of dropping data")
    parser.add_argument("--mapping-process", action="store_true", help="Isolate CROSS mapping from the streaming frontend's Python process")
    parser.add_argument("--delayed-recovery", action="store_true", help="Experimental delayed reverse PnP; may cause large pose corrections")
    parser.add_argument("--teacher-lag-frames", type=int, default=0, help="Minimum source age before applying a ready metric result; late results never block tracking")
    parser.add_argument("--freeze-gc", action="store_true", help="Freeze long-lived startup objects during the run; retain collection of new objects")
    parser.add_argument("--dpvo-checkpoint", type=Path)
    parser.add_argument("--dpvo-metric-bootstrap", action="store_true")
    parser.add_argument("--mask-people", action="store_true")
    parser.add_argument("--mask-interval", type=int, default=1)
    parser.add_argument("--rotation-selection", action="store_true")
    parser.add_argument("--subpixel", action="store_true", help="Refine descriptor matches with bidirectional patch alignment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--metric-resolution", type=int, default=504)
    parser.add_argument("--anchor-interval", type=int, default=30)
    parser.add_argument("--metric-interval", type=int, default=30)
    parser.add_argument("--mapping-interval", type=int, default=5)
    parser.add_argument("--retrieval-pose", choices=["da3", "metric_pnp"], default="da3")
    parser.add_argument("--filter-mode", choices=["full", "skip_active", "adaptive"], default="full",
                        help="Inherited CROSS retrieval pose-update policy; delayed commitment remains enabled")
    parser.add_argument("--pose-model", default="depth-anything/DA3-SMALL")
    parser.add_argument("--pose-refinement", choices=["none", "xfeat"], default="none")
    parser.add_argument("--refinement-anchor-only", action="store_true")
    parser.add_argument("--metric-shape", action="store_true")
    parser.add_argument("--metric-model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--scale-mode", choices=["filtered", "direct", "initial", "relative"], default="filtered")
    parser.add_argument("--scale-recovery-observations", type=int, default=3)
    parser.add_argument("--intrinsics", nargs="+", type=float)
    parser.add_argument("--no-undistort", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Resize RGB after undistortion and adjust intrinsics for pixel-centre sampling")
    parser.add_argument("--load-map", type=Path)
    parser.add_argument("--save-map", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.input_fps is not None and (not np.isfinite(args.input_fps) or args.input_fps <= 0):
        parser.error("--input-fps must be positive")
    if args.sample_fps is not None and (not np.isfinite(args.sample_fps) or args.sample_fps <= 0):
        parser.error("--sample-fps must be finite and positive")
    if args.paced_input_worker and args.input_fps is None:
        parser.error("--paced-input-worker requires --input-fps")
    if args.replay_timestamps and not args.paced_input_worker:
        parser.error("--replay-timestamps requires --paced-input-worker")
    if args.input_process and not args.paced_input_worker:
        parser.error("--input-process requires --paced-input-worker")
    if args.input_buffer < 1:
        parser.error("--input-buffer must be positive")
    if args.teacher_lag_frames < 0 or (args.teacher_lag_frames and args.frontend != "streaming_pnp"):
        parser.error("--teacher-lag-frames must be nonnegative and requires streaming_pnp")
    if (args.mapping_process or args.delayed_recovery) and args.frontend != "streaming_pnp":
        parser.error("--mapping-process and --delayed-recovery require streaming_pnp")
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
    config = MonoConfig(frontend=args.frontend,
                        seed=args.seed,
                        dpvo_metric_bootstrap=args.dpvo_metric_bootstrap,
                        mask_people=args.mask_people,
                        mask_interval=args.mask_interval,
                        rotation_selection=args.rotation_selection,
                        subpixel=args.subpixel,
                        mapping_process=args.mapping_process, delayed_recovery=args.delayed_recovery,
                        teacher_lag_frames=args.teacher_lag_frames,
                        dpvo_checkpoint=str(args.dpvo_checkpoint) if args.dpvo_checkpoint else None,
                        pose_model=args.pose_model, metric_model=args.metric_model,
                        resolution=args.resolution, metric_resolution=args.metric_resolution,
                        anchor_interval=args.anchor_interval, mapping_interval=args.mapping_interval,
                        retrieval_pose=args.retrieval_pose,
                        filter_mode=args.filter_mode,
                        pose_refinement=args.pose_refinement,
                        refinement_anchor_only=args.refinement_anchor_only, metric_shape=args.metric_shape,
                        scale=ScaleConfig(interval=args.metric_interval, mode=args.scale_mode,
                                          recovery_observations=args.scale_recovery_observations))
    sequence = RGBSequence(args.sequence, args.stride, args.start, args.frames, args.intrinsics,
                           not args.no_undistort, args.sample_fps, args.image_size)
    if not len(sequence):
        raise ValueError("No selected images")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = os.environ.get("CROSS_SOURCE_COMMIT", "unavailable")
    metadata = {"config": asdict(config), "command": vars(args).copy(), "source_commit": commit,
                "hostname": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
                "selected_frames": len(sequence), "dataset_rgb_frames": sequence.total_frames,
                "input_selection": "first available RGB per elapsed-time bin" if args.sample_fps else "index stride",
                "arrival_schedule": "original RGB timestamp offsets" if args.replay_timestamps else
                                    ("uniform input-fps intervals" if args.input_fps else "unpaced"),
                "input_modalities": ["RGB", "timestamps", "camera_intrinsics"],
                "runtime_environment": {name: os.environ.get(name) for name in
                                        ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                         "OPENBLAS_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF"]},
                "ground_truth_used_for_inference": False, "sensor_depth_used": False,
                "calibration": sequence.K.tolist(), "distortion": sequence.distortion.tolist(),
                "input_calibration": sequence.input_K.tolist(), "input_image_size": sequence.input_size,
                "resize_image_size": sequence.resize}
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
    if args.frontend == "streaming_pnp":
        from .streaming import StreamingPnPFrontend
        frontend = StreamingPnPFrontend(sequence.K, config, args.device)
        frontend.provide_mapping_depth = not args.frontend_only
    elif args.frontend == "dpvo":
        from .dpvo_frontend import DPVOFrontend
        frontend = DPVOFrontend(sequence.K, config, args.device)
        frontend.provide_mapping_depth = not args.frontend_only
    elif args.frontend in {"metric_pnp", "rotation_metric", "metric_klt"}:
        from .pnp_frontend import MetricPnPFrontend, RotationMetricFrontend, MetricKLTFrontend
        factory = {"metric_pnp": MetricPnPFrontend, "rotation_metric": RotationMetricFrontend,
                   "metric_klt": MetricKLTFrontend}[args.frontend]
        frontend = factory(sequence.K, config, args.device)
        frontend.provide_mapping_depth = not args.frontend_only
    else:
        frontend = MonoFrontend(sequence.K, config, args.device)
    metadata["model_parameters"] = {
        "geometry": sum(p.numel() for p in frontend.geometry.model.parameters()),
        "metric": sum(p.numel() for p in frontend.metric.model.parameters()) if frontend.metric else 0,
    }
    if getattr(frontend, "refiner", None) is not None:
        metadata["model_parameters"]["xfeat"] = sum(p.numel() for p in frontend.refiner.extractor.net.parameters())
        if frontend.refiner.detector is not None:
            metadata["model_parameters"]["person_detector"] = sum(p.numel() for p in frontend.refiner.detector.parameters())
            metadata["person_detector"] = "torchvision SSDLite320 MobileNet V3 Large COCO_V1; score>=0.5; bbox padding 8+4*age px"
    from huggingface_hub import try_to_load_from_cache
    from .models import MODEL_REVISIONS
    metadata["model_checkpoint_revisions"] = {
        model: MODEL_REVISIONS.get(model) for model in [config.pose_model, config.metric_model] if model
    }
    metadata["model_checkpoint_files"] = {
        model: str(try_to_load_from_cache(model, "model.safetensors", revision=MODEL_REVISIONS.get(model)))
        for model in [config.pose_model, config.metric_model] if model
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    tracker = frontend
    if not args.frontend_only:
        if args.frontend == "streaming_pnp":
            from .streaming import StreamingMonocularSystem as MonocularSystem
        else:
            from .system import MonocularSystem
        first = next(iter(sequence))
        tracker = MonocularSystem(sequence.K, (first.rgb.shape[1], first.rgb.shape[0]), config,
                                  device=args.device, frontend=frontend)
        if args.load_map:
            tracker.load_map(args.load_map)
    warmup_seconds = 0.
    warmup_metric_calls = 0
    if args.warmup_models:
        if not hasattr(frontend, "refiner"):
            raise ValueError("Model warmup is currently supported for PnP frontends")
        warmup_start = perf_counter()
        rgb = next(iter(sequence)).rgb
        def warm_features(refiner):
            old_index, old_boxes = refiner.frame_index, refiner.boxes
            refiner.extract(rgb)
            refiner.frame_index, refiner.boxes = old_index, old_boxes
        warm_features(frontend.refiner)
        frontend.metric.predict_metric(rgb, sequence.K, rgb.shape[:2])
        warmup_metric_calls = 1
        image = frontend.geometry.prepare(rgb)
        frontend.geometry.predict([image, image])
        if not args.frontend_only:
            if hasattr(tracker, "warmup_mapping"):
                tracker.warmup_mapping(rgb)
            else:
                tracker.mapper.db.vpr_model.get_embedding(tracker.mapper.rgb_transform(rgb))
                if hasattr(tracker.mapper.pose_est, "refiner"):
                    warm_features(tracker.mapper.pose_est.refiner)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        warmup_seconds = perf_counter()-warmup_start
        metadata["warmup"] = dict(seconds=warmup_seconds, uses_only_first_rgb=True,
                                  advances_pose_or_topological_belief=False, metric_model_calls=warmup_metric_calls)
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    latencies, arrival_latencies, image_io, output_io = [], [], [], []
    valid_count, metric_count = 0, 0
    tracking_stream = None
    if args.frontend == "streaming_pnp" and args.device.startswith("cuda"):
        tracking_stream = torch.cuda.Stream(device=args.device, priority=-1)
        tracking_stream.wait_stream(torch.cuda.current_stream(args.device))
    if args.freeze_gc:
        gc.collect()
        gc.freeze()
    input_worker = None
    offsets = None
    if args.replay_timestamps:
        origin = sequence.rows[0][1][0]
        offsets = [row[1][0] - origin for row in sequence.rows]
    if args.input_process:
        from .input_process import PacedRGBProcess
        try:
            input_worker = PacedRGBProcess(sequence, args.input_fps, args.input_buffer, offsets)
        except BaseException:
            if hasattr(tracker, "shutdown"):
                tracker.shutdown()
            raise
        metadata["input_process_startup_seconds"] = input_worker.startup_seconds
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    loading_seconds = perf_counter() - loading_start
    start = perf_counter()
    gc_pauses, gc_started = [], {}
    def record_collection(phase, info):
        now = perf_counter()-start
        generation = info["generation"]
        if phase == "start":
            gc_started[generation] = now
        else:
            began = gc_started.pop(generation, now)
            gc_pauses.append(dict(generation=generation, start_seconds=began, seconds=now-began,
                                  collected=info["collected"], uncollectable=info["uncollectable"]))
    gc.callbacks.append(record_collection)
    if args.input_process:
        input_worker.start(start)
    elif args.paced_input_worker:
        from .stream_input import PacedRGBStream
        input_worker = PacedRGBStream(sequence, args.input_fps, start, args.input_buffer, offsets)
    try:
        with (args.output / "trajectory.txt").open("w") as trajectory, \
                (args.output / "frontend_trajectory.txt").open("w") as front_trajectory, \
                (args.output / "diagnostics.jsonl").open("w") as diagnostics:
            read_started = perf_counter()
            for delivered in (input_worker if input_worker is not None else sequence):
                if input_worker is not None:
                    frame, arrival, read_seconds = delivered
                    image_io.append(read_seconds)
                else:
                    frame = delivered
                    image_io.append(perf_counter()-read_started)
                    arrival = start + len(latencies) / args.input_fps if args.input_fps else None
                if arrival is not None and input_worker is None:
                    sleep(max(0., arrival-perf_counter()))
                frame_start = perf_counter()
                if tracking_stream is not None:
                    with torch.cuda.stream(tracking_stream):
                        estimate = tracker.step(frame.rgb, frame.timestamp)
                    # Synchronize only the pose stream. A device-wide barrier
                    # would silently wait for the asynchronous teacher/map.
                    tracking_stream.synchronize()
                else:
                    estimate = tracker.step(frame.rgb, frame.timestamp)
                if args.device.startswith("cuda") and tracking_stream is None:
                    torch.cuda.synchronize()
                elapsed = perf_counter() - frame_start
                last_pose_elapsed = perf_counter()-start
                if arrival is not None:
                    arrival_latencies.append(perf_counter()-arrival)
                    estimate.diagnostics["capture_to_pose_seconds"] = arrival_latencies[-1]
                latencies.append(elapsed)
                valid_count += bool(estimate.diagnostics["valid"])
                metric_count += "scale_observation" in estimate.diagnostics
                estimate.diagnostics.update(input_index=frame.index, wall_seconds=elapsed,
                                            pose_ready_elapsed_seconds=last_pose_elapsed,
                                            image_read_and_preprocessing_seconds=image_io[-1],
                                            input_timing=frame.input_timing)
                output_started = perf_counter()
                for handle, pose in [(trajectory, estimate.pose), (front_trajectory, frontend.metric_pose)]:
                    row = np.r_[frame.timestamp, pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]
                    handle.write(" ".join(f"{x:.9f}" for x in row) + "\n")
                    handle.flush()
                diagnostics.write(json.dumps(estimate.diagnostics, allow_nan=False) + "\n")
                diagnostics.flush()
                if len(latencies) % 50 == 0:
                    print(json.dumps({"frames": len(latencies), "valid": valid_count, "scale": frontend.scale_filter.scale,
                                      "fps": len(latencies) / (perf_counter() - start)}), flush=True)
                output_io.append(perf_counter()-output_started)
                read_started = perf_counter()
        if len(latencies) != len(sequence):
            raise RuntimeError(f"Input ended after {len(latencies)} poses; expected {len(sequence)} selected frames")
        emission_elapsed = perf_counter()-start
        if hasattr(tracker, "finish"):
            tracker.finish()
        if args.save_map and not args.frontend_only:
            tracker.save_map(args.output / "map.pkl")
    finally:
        try:
            if input_worker is not None:
                input_worker.close()
            if hasattr(tracker, "shutdown"):
                tracker.shutdown()
        finally:
            gc.callbacks.remove(record_collection)
            if args.freeze_gc:
                gc.unfreeze()
            (args.output / "gc_pauses.json").write_text(json.dumps(gc_pauses, indent=2) + "\n")
            if hasattr(tracker, "map_events"):
                # Preserve observed events even if input/tracking failed;
                # a completed audit still requires a successful run summary.
                (args.output / "mapping_events.json").write_text(json.dumps(tracker.map_events, indent=2) + "\n")
    total = perf_counter() - start
    summary = {"frames": len(latencies), "valid_frames": valid_count, "tracking_coverage": valid_count / len(latencies),
               "metric_calls": metric_count, "scale_observation_calls": metric_count,
               "metric_model_calls": getattr(frontend.metric, "calls", None),
               "model_loading_seconds": loading_seconds,
               "input_process_startup_seconds": metadata.get("input_process_startup_seconds", 0.),
               "warmup_seconds_in_loading": warmup_seconds, "warmup_metric_model_calls": warmup_metric_calls,
               "elapsed_seconds": total, "fps_including_io": len(latencies) / total,
               "emission_elapsed_seconds": emission_elapsed,
               "fps_until_last_pose": len(latencies) / last_pose_elapsed,
               "last_pose_elapsed_seconds": last_pose_elapsed,
               "background_drain_and_shutdown_seconds": total-emission_elapsed,
               "latency_median_ms": 1000 * float(np.median(latencies)),
               "latency_p95_ms": 1000 * float(np.quantile(latencies, 0.95)),
               "latency_p99_ms": 1000 * float(np.quantile(latencies, 0.99)),
               "scale": frontend.scale_filter.scale, "accepted_scale_observations": frontend.scale_filter.accepted,
               "rejected_scale_observations": frontend.scale_filter.rejected,
               "scale_reinitializations": frontend.scale_filter.reinitializations,
               "bootstrap_metric_calls": getattr(frontend, "bootstrap_metric_calls", 0),
               "coverage_definition": "Frontend validity flag; DPVO reports initialization, not an independent accuracy check"}
    summary["image_read_and_preprocessing_ms"] = dict(mean=1000*float(np.mean(image_io)), p95=1000*float(np.quantile(image_io, .95)))
    summary["trajectory_and_diagnostic_output_ms"] = dict(mean=1000*float(np.mean(output_io)), p95=1000*float(np.quantile(output_io, .95)))
    if input_worker is not None:
        summary["input_worker"] = dict(capacity=input_worker.capacity, maximum_queued=input_worker.maximum_queued,
                                       backend="process" if args.input_process else "thread",
                                       queue_peak_is_sampled=args.input_process,
                                       dropped_frames=0, overflow_policy="fail the run", preprocessing_begins_after_capture=True)
    if hasattr(frontend, "depth_worker"):
        summary["teacher_worker"] = frontend.depth_worker.statistics()
    if hasattr(tracker, "map_worker"):
        summary["mapping_worker"] = tracker.map_worker.statistics()
        summary["mapping_audit_complete"] = summary["mapping_worker"]["unread_results_replaced"] == 0
        (args.output / "mapping_events.json").write_text(json.dumps(tracker.map_events, indent=2) + "\n")
    if arrival_latencies:
        summary["input_fps"] = args.input_fps
        summary["capture_to_pose_p95_ms"] = 1000*float(np.quantile(arrival_latencies, .95))
        summary["capture_to_pose_p99_ms"] = 1000*float(np.quantile(arrival_latencies, .99))
        summary["pose_deadline_miss_fraction"] = float(np.mean(np.array(arrival_latencies) > 1/args.input_fps))
        if len(arrival_latencies) > 30:
            steady = np.array(arrival_latencies[30:])
            summary["after_first_30_frames"] = dict(capture_to_pose_p95_ms=1000*float(np.quantile(steady, .95)),
                                                   capture_to_pose_p99_ms=1000*float(np.quantile(steady, .99)),
                                                   pose_deadline_miss_fraction=float(np.mean(steady > 1/args.input_fps)))
    if args.device.startswith("cuda"):
        summary["peak_gpu_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
        if hasattr(tracker, "map_events") and args.mapping_process:
            child_peak = max((e.get("mapper_process_peak_gpu_allocated_gb", 0.) for e in tracker.map_events), default=0.)
            summary["mapper_process_peak_gpu_allocated_gb"] = child_peak
            summary["sum_process_peak_gpu_allocated_gb"] = summary["peak_gpu_allocated_gb"]+child_peak
    if args.frontend in {"dpvo", "rotation_metric"}:
        native_frontend = frontend if args.frontend == "dpvo" else frontend.rotation_tracker
        native_tracker = native_frontend.tracker
        metadata["model_parameters"]["dpvo"] = sum(p.numel() for p in native_tracker.network.parameters())
        if native_frontend.background_patchifier is not None:
            detector_parameters = sum(p.numel() for p in native_frontend.background_patchifier.detector.parameters())
            metadata["model_parameters"]["dpvo"] -= detector_parameters
            metadata["model_parameters"]["dpvo_person_detector"] = detector_parameters
            metadata["dpvo_patch_selection"] = "8x native candidate pool; exclude SSDLite320 MobileNet V3 Large COCO_V1 person boxes, score>=0.5, padding 8+4*age px"
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
