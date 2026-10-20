"""
Run CROSS mapping across multiple sequences with a single System instance (RGB-D mode, configs/modes/rgbd_wheel.yaml).
Each new sequence starts without a pose ("kidnapped"): it relocalizes in the map of the earlier ones and extends it.

Usage:
    python examples/multi_session.py data/posed/office1-1 data/posed/office1-2 data/posed/office1-3
    python examples/multi_session.py seq1 seq2 --starts 900 0 --frames 500 200
    python examples/multi_session.py seq1 seq2 --no-viz --save-map maps/office/map.pkl

Sequences: any format run.py reads (posed RGB-D folders, R3D files, ...).
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from loguru import logger

from cross.core.config import load_config
from cross.core.system import System
from cross.core.types import Camera
from cross.dataloader.auto import load_dataset
from cross.utils.memory_profile import print_memory_usage
from cross.utils.profile import print_timing_registry

np.set_printoptions(formatter={"float": lambda x: f"{x:0.2f}"})
MODE_FILE = Path(__file__).resolve().parents[1] / "configs" / "modes" / "rgbd_wheel.yaml"


def run_multi_session(
    scene_files: list[str],
    frames_per_sequence: list[int] | None = None,
    start_idxs: list[int] | None = None,
    visualize: bool = True,
    save_map: str | None = None,
):
    """
    Process multiple sequences with a single System instance.

    Args:
        scene_files: Sequence paths to process one after the other.
        frames_per_sequence: Number of frames to process per sequence.
        start_idxs: Starting frame index for each sequence.
        visualize: Enable visualization.
        save_map: Save the map of all sessions there at the end.
    """
    if start_idxs is None:
        start_idxs = [0] * len(scene_files)
    if frames_per_sequence is None:
        frames_per_sequence = [100000] * len(scene_files)

    first_dataset = load_dataset(scene_files[0])
    camera = Camera(
        K=first_dataset.rgb_K,
        frame_width=first_dataset.rgb_width,
        frame_height=first_dataset.rgb_height,
    )

    config = load_config(MODE_FILE, visualization={"visualize_pointcloud": False})
    system = System(visualize=visualize, camera=camera, config=config)

    total_frames_processed = 0
    session_stats = []

    try:
        for seq_idx, scene_file in enumerate(scene_files):
            logger.info(f"Processing sequence {seq_idx + 1}/{len(scene_files)}: {scene_file}")

            dataset = first_dataset if seq_idx == 0 else load_dataset(scene_file)
            end_idx = min(start_idxs[seq_idx] + frames_per_sequence[seq_idx], len(dataset))
            num_frames = end_idx - start_idxs[seq_idx]
            logger.info(f"Frames {start_idxs[seq_idx]}..{end_idx} ({num_frames} frames)")

            reply = dataset.replay_data(start_idx=start_idxs[seq_idx], end_idx=end_idx)
            start_time = time.time()
            frames_in_seq = 0

            for idx, d in enumerate(reply):
                if idx == 0:
                    d["delta_pose"] = None  # first frame / kidnapped at the sequence transition

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

        if save_map:
            os.makedirs(os.path.dirname(os.path.abspath(save_map)), exist_ok=True)
            system.save_map(save_map)
            logger.info(f"Map saved: {save_map}")

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
    parser = argparse.ArgumentParser(description="CROSS: Multi-session mapping across several sequences")
    parser.add_argument("scenes", nargs="+", help="sequences to process one after the other")
    parser.add_argument("--starts", type=int, nargs="*", default=None,
                        help="Starting frame index per sequence (default: 0 for each)")
    parser.add_argument("--frames", type=int, nargs="*", default=None,
                        help="Max frames per sequence (default: all)")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--save-map", default=None, help="save the map of all sessions (e.g. maps/office/map.pkl)")
    args = parser.parse_args()

    # Pad starts/frames to match number of scenes
    n = len(args.scenes)
    starts = args.starts if args.starts else [0] * n
    if len(starts) < n:
        starts.extend([0] * (n - len(starts)))
    frames = args.frames if args.frames else None
    if frames is not None and len(frames) < n:
        frames.extend([100000] * (n - len(frames)))

    run_multi_session(
        scene_files=args.scenes,
        start_idxs=starts,
        frames_per_sequence=frames,
        visualize=not args.no_viz,
        save_map=args.save_map,
    )
