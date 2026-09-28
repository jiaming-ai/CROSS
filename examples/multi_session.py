"""
Run CROSS mapping across multiple sequences with a single System instance.

Usage:
    python examples/multi_session.py scene1.r3d scene2.r3d scene3.r3d
    python examples/multi_session.py scene1.r3d scene2.r3d --starts 900 0 --frames 500 200
    python examples/multi_session.py scene1.r3d --no-viz
"""

import argparse
import time

import numpy as np
from loguru import logger

from cross.core.config import SystemConfig, VisualizationConfig
from cross.core.system import System
from cross.core.types import Camera
from cross.dataloader.r3d_loader import R3DDataset
from cross.utils.profile import print_timing_registry
from cross.utils.memory_profile import print_memory_usage

np.set_printoptions(formatter={"float": lambda x: f"{x:0.2f}"})


def run_multi_session(
    scene_files: list[str],
    frames_per_sequence: list[int] | None = None,
    start_idxs: list[int] | None = None,
    visualize: bool = True,
    debug: bool = True,
    async_update: bool = False,
):
    """
    Process multiple sequences with a single System instance.

    Args:
        scene_files: List of .r3d file paths to process sequentially.
        frames_per_sequence: Number of frames to process per sequence.
        start_idxs: Starting frame index for each sequence.
        visualize: Enable visualization.
        debug: Enable debug mode.
        async_update: Enable async step pipeline.
    """
    if start_idxs is None:
        start_idxs = [0] * len(scene_files)
    if frames_per_sequence is None:
        frames_per_sequence = [100000] * len(scene_files)

    first_dataset = R3DDataset(scene_files[0])
    camera = Camera(
        K=first_dataset.rgb_K,
        frame_width=first_dataset.rgb_width,
        frame_height=first_dataset.rgb_height,
    )

    config = SystemConfig(
        async_update=async_update,
        visualization=VisualizationConfig(
            visualize_pointcloud=False,
            visualize_current_gmm_state=True,
            visualize_keyframe_gmms=True,
            visualize_system_data=True,
            visualize_trajectory=True,
            visualize_particles=True,
        ),
    )
    system = System(
        visualize=visualize,
        debug=debug,
        camera=camera,
        config=config,
    )

    total_frames_processed = 0
    session_stats = []

    try:
        for seq_idx, scene_file in enumerate(scene_files):
            logger.info(f"Processing sequence {seq_idx + 1}/{len(scene_files)}: {scene_file}")

            dataset = R3DDataset(scene_file)
            end_idx = min(start_idxs[seq_idx] + frames_per_sequence[seq_idx], len(dataset))
            num_frames = end_idx - start_idxs[seq_idx]
            logger.info(f"Frames {start_idxs[seq_idx]}..{end_idx} ({num_frames} frames)")

            reply = dataset.replay_data(start_idx=start_idxs[seq_idx], end_idx=end_idx)
            start_time = time.time()
            frames_in_seq = 0

            for idx, d in enumerate(reply):
                if seq_idx > 0 and idx == 0:
                    d["delta_pose"] = None  # kidnapped at sequence transition

                system.step(obs=d, data=d)
                frames_in_seq += 1
                total_frames_processed += 1

                if frames_in_seq % 50 == 0:
                    logger.info(f"  {frames_in_seq}/{num_frames} frames processed")
                if total_frames_processed % 500 == 0:
                    print_timing_registry()

            elapsed = time.time() - start_time
            fps = frames_in_seq / elapsed if elapsed > 0 else 0
            session_stats.append({"sequence": scene_file, "frames": frames_in_seq, "time": elapsed, "fps": fps})
            logger.info(f"Sequence {seq_idx + 1} done: {frames_in_seq} frames, {elapsed:.2f}s, {fps:.2f} FPS")

    finally:
        logger.info(f"Total: {total_frames_processed} frames across {len(session_stats)} sequences")
        total_time = sum(s["time"] for s in session_stats)
        if total_time > 0:
            logger.info(f"Overall: {total_time:.2f}s, {total_frames_processed / total_time:.2f} FPS")
        for i, s in enumerate(session_stats, 1):
            logger.info(f"  {i}. {s['sequence']}: {s['frames']} frames, {s['time']:.2f}s, {s['fps']:.2f} FPS")

        print_timing_registry()
        print_memory_usage(system, detailed=True)
        system.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CROSS: Multi-session mapping across multiple R3D sequences")
    parser.add_argument("scenes", nargs="+", help="R3D scene files to process sequentially")
    parser.add_argument("--starts", type=int, nargs="*", default=None,
                        help="Starting frame index per sequence (default: 0 for each)")
    parser.add_argument("--frames", type=int, nargs="*", default=None,
                        help="Max frames per sequence (default: all)")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--async", dest="async_update", action="store_true", help="Enable async step pipeline")
    args = parser.parse_args()

    # Pad starts/frames to match number of scenes
    n = len(args.scenes)
    starts = args.starts if args.starts else [0] * n
    if len(starts) < n:
        starts.extend([0] * (n - len(starts)))
    frames = args.frames if args.frames else None

    run_multi_session(
        scene_files=args.scenes,
        start_idxs=starts,
        frames_per_sequence=frames,
        visualize=not args.no_viz,
        async_update=args.async_update,
    )
