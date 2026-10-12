#!/usr/bin/env python3
"""
CROSS — Run topological mapping on an RGB-D dataset.

Usage:
    python run.py <dataset_path> [options]                        # RGB-D with the dataset's odometry (rgbd_wheel)
    python run.py <stereo_sequence> --mode stereo_wheel [options] # stereo mode (VGGT-Omega, stereo scale)
    python run.py <dataset_path> --mode mono_imu [options]        # one camera + IMU

--mode takes one of the twelve shipped modes (configs/<profile>/<mode>.yaml): stereo_{wheel,basalt,imu,dpvo},
rgbd_{wheel,basalt,dpvo}, mono_{wheel,basalt,imu,vio,dpvo} (docs/USAGE.md section 1); or a sensor mode (rgbd, stereo,
mono) with --odometry / --mono-estimator.

Examples:
    python run.py data/r3d/lab2.r3d
    python run.py data/r3d/lab2.r3d --no-viz --frames 500
    python run.py data/rosbag/topomap_ssi_1 --loader rosbag
    python run.py data/posed/home1-1 --loader posed                       # posed RGB-D folder (e.g. OpenLORIS)
    python run.py data/kitti_raw/2011_09_30/2011_09_30_drive_0027_sync --mode stereo --config configs/overlays/outdoor.yaml
    python run.py data/posed/home1-1 --loader posed --mode rgbd_dpvo --no-viz             # RGB-D, visual odometry
    python run.py data/posed/home1-1 --loader posed --mode mono_dpvo                      # RGB only
"""

import argparse
import os
import time

import numpy as np
from loguru import logger

from cross.core.config import load_config
from cross.core.types import Camera
from cross.pipeline import add_session_args, finish_session_args, session_factory, shipped_config_files
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
    parser.add_argument("--loader", default="auto", choices=["auto", "r3d", "rosbag", "loris", "tum", "posed", "stereo"],
                        help="Dataset loader type (default: auto-detect; stereo mode always uses the stereo loader)")
    parser.add_argument("--baseline", type=float, default=None,
                        help="stereo mode: rendered baseline to use for SimChange sequences with several right cameras")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--snr", type=float, default=None, help="Signal-to-noise ratio for R3D datasets")
    parser.add_argument("--async", dest="async_update", action="store_true", help="Enable async step pipeline")
    parser.add_argument("--config", nargs="*", default=[], help="YAML config file(s), merged left to right after the "
                        "mode's shipped file (configs/<--profile>/<mode>.yaml), e.g. configs/overlays/outdoor.yaml")
    parser.add_argument("--set", nargs="*", action="extend", default=[],
                        help="config overrides section.sub.key=value (YAML-parsed), after --config; e.g. "
                             "pose_est.ff.checkpoint=released (the released VGGT-Omega instead of the fine-tuned one)")
    parser.add_argument("--odom-file", default=None, help="odometry file of a prepared folder to read instead of "
                        "odom_left.txt (--odometry basalt: odom_vio.txt)")
    add_session_args(parser)   # --mode, --odometry, --profile, --mono-estimator, --dpvo-checkpoint, ...
    args = finish_session_args(parser.parse_args())
    args.mode = args.mode or "rgbd"

    # Load dataset
    loader_kwargs = {}
    if args.snr is not None:
        loader_kwargs["snr"] = args.snr
    if args.odom_file:
        loader_kwargs["odom_file"] = args.odom_file

    if args.mode == "stereo":
        loader_kwargs["baseline"] = args.baseline
        if args.odometry == "visual":
            loader_kwargs["depth_source"] = "sgbm"      # metric scale of the visual odometry
        dataset = load_dataset(args.dataset, loader="stereo", **loader_kwargs)
    else:
        dataset = load_dataset(args.dataset, loader=args.loader, **loader_kwargs)
    logger.info(f"Dataset: {args.dataset}, {len(dataset)} frames")

    camera = Camera(
        K=dataset.rgb_K,
        frame_width=dataset.rgb_width,
        frame_height=dataset.rgb_height,
    )

    # Build config: the mode's shipped configuration (configs/<--profile>/<mode>.yaml), then the user's --config files
    # and --set overrides
    config = load_config(*shipped_config_files(args), *args.config, sets=args.set)
    if args.async_update:
        config.async_update = True

    # the loaders' timestamps are used by DPVO (visual odometry, mono mode); stereo mode, mono mode and visual
    # odometry are described in cross/pipeline.py
    system = session_factory(args, camera, config, T_right_in_left=getattr(dataset, "T_right_in_left", None),
                             visualize=not args.no_viz and args.mode != "mono")()

    end_idx = min(args.start + args.frames, len(dataset)) if args.frames else len(dataset)
    reply = dataset.replay_data(start_idx=args.start, end_idx=end_idx)

    t0 = time.time()
    for idx, d in enumerate(reply):
        if idx == 0:
            d["delta_pose"] = None  # first frame initialization

        system.process(d)

        if idx % 100 == 0:
            logger.info(f"Step {idx}/{end_idx - args.start}")

    elapsed = time.time() - t0
    n_kfs = len(system.hypothesis_manager.nodes)
    logger.info(f"Done: {idx + 1} frames in {elapsed:.1f}s ({(idx + 1) / elapsed:.1f} FPS), {n_kfs} keyframes")
    print_timing_registry()

    system.release()


if __name__ == "__main__":
    main()
