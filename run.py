#!/usr/bin/env python3
"""
CROSS — Run topological mapping on an RGB-D dataset.

Usage:
    python run.py <dataset_path> [options]                  # RGB-D mode (PnP on depth)
    python run.py <stereo_sequence> --mode stereo [options] # stereo mode (feed-forward estimator, stereo scale)

Examples:
    python run.py data/r3d/lab2.r3d
    python run.py data/r3d/lab2.r3d --no-viz --frames 500
    python run.py data/rosbag/topomap_ssi_1 --loader rosbag
    python run.py data/posed/home1-1 --loader posed                       # posed RGB-D folder (e.g. OpenLORIS)
    python run.py data/kitti_raw/2011_09_30/2011_09_30_drive_0027_sync --mode stereo --config configs/outdoor.yaml
"""

import argparse
import os
import time

import numpy as np
from loguru import logger

from cross.core.config import SystemConfig, load_config
from cross.core.system import System
from cross.core.types import Camera
from cross.utils.profile import print_timing_registry

np.set_printoptions(formatter={"float": lambda x: f"{x:0.2f}"})


def load_dataset(path: str, loader: str = "auto", **kwargs):
    """Load a dataset by path, auto-detecting format or using the specified loader."""
    if loader == "auto":
        if os.path.isfile(os.path.join(path, "calib.json")) and os.path.isdir(os.path.join(path, "rgb")):
            loader = "posed"                      # posed RGB-D folder (detected by its layout, before name heuristics)
        elif path.endswith(".r3d"):
            loader = "r3d"
        elif "rosbag" in path or "topomap" in path:
            loader = "rosbag"
        elif "loris" in path or "corridor" in path or "cafe" in path:
            loader = "loris"
        elif "tum" in path:
            loader = "tum"
        else:
            loader = "r3d"

    if loader == "r3d":
        from cross.dataloader.r3d_loader import R3DDataset
        return R3DDataset(path, **kwargs)
    elif loader == "rosbag":
        from cross.dataloader.rosbag_loader import RosbagLoader
        return RosbagLoader(path, **kwargs)
    elif loader == "loris":
        from cross.dataloader.loris import OpenLorisLoader
        return OpenLorisLoader(path, **kwargs)
    elif loader == "tum":
        from cross.dataloader.tum import TUMDataset
        return TUMDataset(path, **kwargs)
    elif loader == "posed":
        from cross.dataloader.posed_rgbd import PosedRGBDLoader
        return PosedRGBDLoader(path, **kwargs)
    elif loader == "stereo":
        from cross.dataloader.stereo_loader import StereoSequenceLoader
        return StereoSequenceLoader(path, **kwargs)
    else:
        raise ValueError(f"Unknown loader: {loader}")


def main():
    parser = argparse.ArgumentParser(description="CROSS: Pose-aware topological mapping")
    parser.add_argument("dataset", help="Path to dataset (e.g., data/r3d/lab2.r3d)")
    parser.add_argument("--mode", default="rgbd", choices=["rgbd", "stereo"],
                        help="rgbd: PnP relative poses from RGB-D (default); stereo: feed-forward estimator with stereo "
                             "scale anchors (needs install.sh --stereo; layers configs/stereo.yaml)")
    parser.add_argument("--loader", default="auto", choices=["auto", "r3d", "rosbag", "loris", "tum", "posed", "stereo"],
                        help="Dataset loader type (default: auto-detect; stereo mode always uses the stereo loader)")
    parser.add_argument("--baseline", type=float, default=None,
                        help="stereo mode: rendered baseline to use for SimChange sequences with several right cameras")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--snr", type=float, default=None, help="Signal-to-noise ratio for R3D datasets")
    parser.add_argument("--async", dest="async_update", action="store_true", help="Enable async step pipeline")
    parser.add_argument("--config", nargs="*", default=[], help="YAML config file(s), merged left to right")
    args = parser.parse_args()

    # Load dataset
    loader_kwargs = {}
    if args.snr is not None:
        loader_kwargs["snr"] = args.snr

    if args.mode == "stereo":
        loader_kwargs["baseline"] = args.baseline
        dataset = load_dataset(args.dataset, loader="stereo", **loader_kwargs)
    else:
        dataset = load_dataset(args.dataset, loader=args.loader, **loader_kwargs)
    logger.info(f"Dataset: {args.dataset}, {len(dataset)} frames")

    camera = Camera(
        K=dataset.rgb_K,
        frame_width=dataset.rgb_width,
        frame_height=dataset.rgb_height,
    )

    # Build config: layer YAML files, then apply CLI overrides
    configs = list(args.config)
    if args.mode == "stereo":
        configs = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "stereo.yaml")] + configs
    config = load_config(*configs) if configs else SystemConfig()
    if args.async_update:
        config.async_update = True

    system = System(
        visualize=not args.no_viz,
        debug=True,
        camera=camera,
        config=config,
        T_right_in_left=getattr(dataset, "T_right_in_left", None),
    )

    end_idx = min(args.start + args.frames, len(dataset)) if args.frames else len(dataset)
    reply = dataset.replay_data(start_idx=args.start, end_idx=end_idx)

    t0 = time.time()
    for idx, d in enumerate(reply):
        if idx == 0:
            d["delta_pose"] = None  # first frame initialization

        system.step(obs=d, data=d)

        if idx % 100 == 0:
            logger.info(f"Step {idx}/{end_idx - args.start}")

    elapsed = time.time() - t0
    n_kfs = len(system.hypothesis_manager.nodes)
    logger.info(f"Done: {idx + 1} frames in {elapsed:.1f}s ({(idx + 1) / elapsed:.1f} FPS), {n_kfs} keyframes")
    print_timing_registry()

    system.shutdown()


if __name__ == "__main__":
    main()
