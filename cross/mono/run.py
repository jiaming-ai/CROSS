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


def motion_std_kwargs(values):
    if not values:
        return {}
    if len(values) not in (2, 4) or any(not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("--motion-std takes 2 or 4 nonnegative values")
    keys = ["translation_std_floor", "rotation_std_floor", "translation_std_per_meter", "rotation_std_per_radian"]
    return dict(zip(keys, values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontend-only", action="store_true")
    parser.add_argument("--frontend", choices=["da3", "dpvo", "metric_pnp", "rotation_metric", "learned_rotation_pnp", "metric_klt", "streaming_pnp", "streaming_dpvo"], default="da3")
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
    parser.add_argument("--adaptive-anchor", action="store_true",
                        help="Experimental metric refresh before tracking loss; request on fewer than 80 PnP inliers")
    parser.add_argument("--stable-teacher-cadence", action="store_true",
                        help="Experimental fixed regular request grid with bounded emergency requests; streaming_pnp only")
    parser.add_argument('--retrieve-during-loss', action='store_true',
                        help='Experimental global observations during local tracking loss; preserve failed motion uncertainty and mapping cadence')
    parser.add_argument("--trace-metric-sources", action="store_true",
                        help="Record reused teacher identities and signed scale responses; does not change inference")
    parser.add_argument('--conditional-sources', action='store_true',
                        help='Experimental shared source inference; requires streaming_pnp, metric_pnp, chart-aware and session-recovery')
    parser.add_argument('--motion-covariance-bound', choices=['axes', 'matrix'], default='axes',
                        help='Experimental full-moment correlation bound before diagonal motion input; requires conditional sources')
    parser.add_argument('--source-log-std', type=float, default=.12,
                        help='Declared per-prediction log-depth prior std for conditional inference; not empirically calibrated')
    parser.add_argument("--freeze-gc", action="store_true", help="Freeze long-lived startup objects during the run; retain collection of new objects")
    parser.add_argument("--dpvo-checkpoint", type=Path)
    parser.add_argument('--rotation-tracker', choices=['none', 'dpvo'], default='none',
                        help='Experimental native rotation with streaming_pnp; shares masks and preserves delayed source poses')
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
    parser.add_argument("--retrieval-pose", choices=["da3", "metric_pnp", "metric_two_view", "ff"], default="da3")
    parser.add_argument("--ff-backend", choices=["da3", "vggt_omega"], default="da3",
                        help="Feed-forward multi-view model for --retrieval-pose ff")
    parser.add_argument("--ff-checkpoint", default="depth-anything/DA3-LARGE-1.1",
                        help="Hugging Face ID / directory (DA3) or .pt file (VGGT-Omega)")
    parser.add_argument("--ff-resolution", type=int, default=504)
    parser.add_argument("--ff-min-covisibility", type=float, default=0.3)
    parser.add_argument("--ff-fallback-only", action="store_true",
                        help="Keep metric two-view poses; run the feed-forward model only on rejected references")
    parser.add_argument("--ff-scope", choices=["all", "map", "relocalization"], default="all",
                        help="With --ff-fallback-only: which rejected references the model sees (map: loaded-map "
                             "references; relocalization: loaded-map references before the session is joined)")
    parser.add_argument("--retrieval-matcher", choices=["mnn", "lighterglue", "superpoint_lightglue"], default="mnn",
                        help="Matcher for low-rate metric-PnP retrieval; local tracking is unchanged")
    parser.add_argument("--two-view-rotation-check", action="store_true",
                        help="Experimentally screen two-view fallback rotation with the configured pose model")
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
    parser.add_argument("--session-recovery", action="store_true",
                        help="Experimental loaded-map recovery using historical support and inherited delayed evidence")
    parser.add_argument("--chart-aware", action="store_true",
                        help="Keep disconnected pose charts separate; requires --session-recovery")
    parser.add_argument("--schmidt-map-geometry", action="store_true",
                        help="Experimental shared map uncertainty; requires conditional sources and a complete graph")
    parser.add_argument('--map-geometry-basis', choices=['epoch','factor'], default='epoch',
                        help='Experimental shared-geometry coordinates; factor preserves noise identity across competing hypotheses')
    parser.add_argument("--historical-retrieval-slots", type=int, default=0,
                        help="Reserve saved-map candidates within the same retrieval/verification budget")
    parser.add_argument("--historical-min-score", type=float,
                        help="Experimental score floor for reserved saved-map candidates only; default uses existing thresholds")
    parser.add_argument("--save-map", action="store_true")
    parser.add_argument("--motion-std", type=float, nargs="+", metavar="V",
                        help="Per-frame motion std: translation floor (m), rotation floor (rad) "
                             "[, translation per metre, rotation per radian] (streaming_dpvo)")
    parser.add_argument("--min-texture-corners", type=int, default=0,
                        help="streaming_dpvo: treat frames with fewer FAST corners as degenerate views (motion unknown)")
    parser.add_argument("--cross-config", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a CROSS SystemConfig entry, e.g. mapping.hypothesis.h0_informative_only=true")
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
    if args.adaptive_anchor and args.frontend != "streaming_pnp":
        parser.error("--adaptive-anchor requires streaming_pnp")
    if args.stable_teacher_cadence and args.frontend != "streaming_pnp":
        parser.error("--stable-teacher-cadence requires streaming_pnp")
    if args.mapping_process and args.frontend not in {"streaming_pnp", "streaming_dpvo"}:
        parser.error("--mapping-process requires a streaming frontend")
    if args.delayed_recovery and args.frontend != "streaming_pnp":
        parser.error("--delayed-recovery requires streaming_pnp")
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
                        rotation_tracker=args.rotation_tracker,
                        mask_people=args.mask_people,
                        mask_interval=args.mask_interval,
                        rotation_selection=args.rotation_selection,
                        subpixel=args.subpixel,
                        mapping_process=args.mapping_process, delayed_recovery=args.delayed_recovery,
                        teacher_lag_frames=args.teacher_lag_frames,
                        adaptive_anchor=args.adaptive_anchor,
                        stable_teacher_cadence=args.stable_teacher_cadence,
                        retrieve_during_loss=args.retrieve_during_loss,
                        trace_metric_sources=args.trace_metric_sources,
                        conditional_sources=args.conditional_sources,source_log_std=args.source_log_std,
                        motion_covariance_bound=args.motion_covariance_bound,
                        schmidt_map_geometry=args.schmidt_map_geometry,
                        map_geometry_basis=args.map_geometry_basis,
                        dpvo_checkpoint=str(args.dpvo_checkpoint) if args.dpvo_checkpoint else None,
                        pose_model=args.pose_model, metric_model=args.metric_model,
                        resolution=args.resolution, metric_resolution=args.metric_resolution,
                        anchor_interval=args.anchor_interval, mapping_interval=args.mapping_interval,
                        retrieval_pose=args.retrieval_pose,
                        ff_backend=args.ff_backend, ff_checkpoint=args.ff_checkpoint,
                        ff_resolution=args.ff_resolution, ff_min_covisibility=args.ff_min_covisibility,
                        ff_fallback_only=args.ff_fallback_only, ff_scope=args.ff_scope,
                        retrieval_matcher=args.retrieval_matcher,
                        two_view_rotation_check=args.two_view_rotation_check,
                        filter_mode=args.filter_mode,
                        session_recovery=args.session_recovery,
                        chart_aware=args.chart_aware,
                        historical_retrieval_slots=args.historical_retrieval_slots,
                        historical_min_score=args.historical_min_score,
                        pose_refinement=args.pose_refinement,
                        refinement_anchor_only=args.refinement_anchor_only, metric_shape=args.metric_shape,
                        cross_overrides=tuple(args.cross_config), min_texture_corners=args.min_texture_corners,
                        **motion_std_kwargs(args.motion_std),
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
    if args.frontend == "streaming_dpvo":
        from .streaming_dpvo import StreamingDPVOFrontend
        frontend = StreamingDPVOFrontend(sequence.K, config, args.device)
        frontend.provide_mapping_depth = not args.frontend_only
    elif args.frontend == "streaming_pnp":
        from .streaming import StreamingPnPFrontend
        frontend = StreamingPnPFrontend(sequence.K, config, args.device)
        frontend.provide_mapping_depth = not args.frontend_only
    elif args.frontend == "dpvo":
        from .dpvo_frontend import DPVOFrontend
        frontend = DPVOFrontend(sequence.K, config, args.device)
        frontend.provide_mapping_depth = not args.frontend_only
    elif args.frontend in {"metric_pnp", "rotation_metric", "learned_rotation_pnp", "metric_klt"}:
        from .pnp_frontend import MetricPnPFrontend, RotationMetricFrontend, LearnedRotationPnPFrontend, MetricKLTFrontend
        factory = {"metric_pnp": MetricPnPFrontend, "rotation_metric": RotationMetricFrontend,
                   "learned_rotation_pnp": LearnedRotationPnPFrontend, "metric_klt": MetricKLTFrontend}[args.frontend]
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
        if args.frontend in {"streaming_pnp", "streaming_dpvo"}:
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
        if not hasattr(frontend, "refiner") and args.frontend != 'streaming_dpvo':
            raise ValueError("Model warmup requires PnP or streaming_dpvo")
        warmup_start = perf_counter()
        rgb = next(iter(sequence)).rgb
        def warm_features(refiner):
            old_index, old_boxes = refiner.frame_index, refiner.boxes
            refiner.extract(rgb)
            refiner.frame_index, refiner.boxes = old_index, old_boxes
        if args.frontend == 'streaming_dpvo':
            frontend.initialize_tracker(*rgb.shape[:2])
        else:
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
    if args.frontend in {"streaming_pnp", "streaming_dpvo"} and args.device.startswith("cuda"):
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
               "coverage_definition": "Frontend validity flag; DPVO requires native initialization, background support and scale availability (explicit unit gauge in relative mode), not an independent accuracy check"}
    summary["image_read_and_preprocessing_ms"] = dict(mean=1000*float(np.mean(image_io)), p95=1000*float(np.quantile(image_io, .95)))
    summary["trajectory_and_diagnostic_output_ms"] = dict(mean=1000*float(np.mean(output_io)), p95=1000*float(np.quantile(output_io, .95)))
    if input_worker is not None:
        summary["input_worker"] = dict(capacity=input_worker.capacity, maximum_queued=input_worker.maximum_queued,
                                       backend="process" if args.input_process else "thread",
                                       queue_peak_is_sampled=args.input_process,
                                       dropped_frames=0, overflow_policy="fail the run", preprocessing_begins_after_capture=True)
    if hasattr(frontend, "depth_worker"):
        summary["teacher_worker"] = frontend.depth_worker.statistics()
    if hasattr(frontend, 'scale_events'):
        (args.output / 'scale_events.json').write_text(json.dumps(frontend.scale_events, indent=2) + '\n')
        summary['received_metric_observations'] = sum(e['scale_update_requested'] for e in frontend.scale_events)
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
    if args.frontend in {"dpvo", "rotation_metric", "streaming_dpvo"} or args.rotation_tracker == 'dpvo':
        native_frontend = (frontend if args.frontend == "dpvo" else frontend.native_frontend
                           if args.frontend == 'streaming_dpvo' else frontend.rotation_tracker)
        native_tracker = native_frontend.tracker
        metadata["model_parameters"]["dpvo"] = sum(p.numel() for p in native_tracker.network.parameters())
        if native_frontend.background_patchifier is not None:
            detector = native_frontend.background_patchifier.detector
            detector_parameters = sum(p.numel() for p in detector.parameters()) if detector is not None else 0
            metadata["model_parameters"]["dpvo"] -= detector_parameters
            metadata["model_parameters"]["dpvo_person_detector"] = detector_parameters
            metadata['dpvo_shared_person_detector'] = detector is None
            metadata["dpvo_patch_selection"] = "8x native candidate pool; exclude SSDLite320 MobileNet V3 Large COCO_V1 person boxes, score>=0.5, padding 8+4*age px"
        import importlib
        metadata['dpvo_runtime'] = dict(
            checkpoint_sha256=hashlib.sha256(Path(config.dpvo_checkpoint).read_bytes()).hexdigest(),
            actual_torch_threads=torch.get_num_threads(),
            binaries={})
        for module_name in ('cuda_ba', 'cuda_corr', 'lietorch_backends'):
            path = Path(importlib.import_module(module_name).__file__)
            metadata['dpvo_runtime']['binaries'][module_name] = dict(
                path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
