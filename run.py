#!/usr/bin/env python3
"""
CROSS — map a sequence, or relocalize it in a saved map.

Usage:
    python run.py <sequence> --mode <mode> [options]

--mode takes one of the fourteen shipped modes (configs/modes/<mode>.yaml): stereo_{wheel,basalt,imu,dpvo},
rgbd_{wheel,basalt,dpvo}, rgbd_ff_{wheel,basalt}, mono_{wheel,basalt,imu,vio,dpvo} (docs/GUIDE.md section 3); or a
sensor mode (rgbd, stereo, mono) with --odometry / --mono-estimator. Without --mode: rgbd_wheel.

Examples:
    python run.py data/posed/office1-1 --mode rgbd_wheel --save-map maps/office/map.pkl      # map, save
    python run.py data/posed/office1-2 --mode rgbd_wheel --load-map maps/office/map.pkl \\
        --save-poses office1-2.txt                                                          # relocalize
    python run.py data/kitti_raw/2011_09_30/2011_09_30_drive_0027_sync --mode stereo_wheel \\
        --config configs/overlays/outdoor.yaml --no-viz --frames 500
    python run.py data/posed/office1-1 --mode mono_dpvo                                     # RGB only
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from loguru import logger

from cross.core.config import load_config
from cross.core.types import Camera
from cross.dataloader.auto import LOADERS, detect_loader, load_dataset
from cross.pipeline import add_session_args, finish_session_args, session_factory, shipped_config_files
from cross.utils.profile import print_timing_registry

np.set_printoptions(formatter={"float": lambda x: f"{x:0.2f}"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CROSS: map a sequence, or relocalize it in a saved map",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("dataset", help="sequence folder (posed RGB-D or stereo folder, KITTI raw, ...) or .r3d file")
    parser.add_argument("--loader", default="auto", choices=LOADERS,
                        help="dataset format (default: detected from the folder layout; the stereo modes always read "
                             "it as a stereo sequence)")
    parser.add_argument("--baseline", type=float, default=None,
                        help="stereo: which rendered right camera of a SimChange sequence to use")
    parser.add_argument("--no-viz", action="store_true", help="no Rerun viewer window (e.g. on a server)")
    parser.add_argument("--frames", type=int, default=None, help="process at most this many frames")
    parser.add_argument("--start", type=int, default=0, help="first frame")
    parser.add_argument("--snr", type=float, default=None,
                        help="noise of the odometry simulated from the ground truth when the sequence has no odometry "
                             "file (signal-to-noise ratio; default: noise-free)")
    parser.add_argument("--config", nargs="*", action="extend", default=[],
                        help="YAML config file(s), merged left to right after the mode's shipped file "
                             "(configs/modes/<mode>.yaml), e.g. configs/overlays/outdoor.yaml; repeatable")
    parser.add_argument("--set", nargs="*", action="extend", default=[],
                        help="config overrides section.sub.key=value (YAML-parsed), after --config; e.g. "
                             "pose_est.ff.checkpoint=released (the released VGGT-Omega instead of the fine-tuned one)")
    parser.add_argument("--odom-file", default=None, help="odometry file of a prepared folder to read instead of "
                        "odom_left.txt (--odometry basalt: odom_vio.txt)")
    parser.add_argument("--load-map", default=None, metavar="MAP",
                        help="start in a saved map (map.pkl of --save-map or scripts/map_and_reloc*.py): the sequence "
                             "relocalizes in it and extends it")
    parser.add_argument("--save-map", default=None, metavar="MAP",
                        help="save the map at the end: MAP (the graph) and the folder MAP.store (images, descriptors)")
    parser.add_argument("--save-poses", default=None, metavar="FILE",
                        help="write the camera pose of every frame in the map frame (TUM format: time x y z qx qy qz "
                             "qw), from the frame on which the session is placed in the map")
    parser.add_argument("--log-file", default=None, metavar="FILE", help="also write the debug log to FILE")
    add_session_args(parser)   # --mode, --odometry, --mono-estimator, --dpvo-checkpoint, --fast, --reproducible, ...
    return parser


def _pose_matrix(pose) -> np.ndarray:
    """4x4 matrix of a pypose SE3 pose (session.belief)."""
    from cross.utils.lie_tensor import normalize_SE3
    return normalize_SE3(pose).matrix().detach().cpu().numpy().astype(np.float64).reshape(4, 4)


def _tum_line(t: float, T: np.ndarray) -> str:
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(T[:3, :3]).as_quat()            # x y z w
    return f"{t:.6f} " + " ".join(f"{v:.6f}" for v in (*T[:3, 3], *q))


def main():
    args = finish_session_args(build_parser().parse_args())
    args.mode = args.mode or "rgbd"
    if args.log_file:
        logger.add(args.log_file, level="DEBUG", mode="w")

    # Load the sequence
    loader_kwargs = {}
    if args.snr is not None:
        loader_kwargs["snr"] = args.snr
    if args.odom_file:
        loader_kwargs["odom_file"] = args.odom_file
    loader = "stereo" if args.mode == "stereo" else args.loader
    if loader == "auto":
        loader = detect_loader(args.dataset)
    if loader == "stereo":
        loader_kwargs["baseline"] = args.baseline
        if args.mode == "stereo" and args.odometry == "visual":
            loader_kwargs["depth_source"] = "sgbm"      # metric scale of the visual odometry
        elif args.mode == "rgbd":                      # PnP on the folder's depth, else on stereo-matched depth
            loader_kwargs["depth_source"] = "gt" if (Path(args.dataset) / "depth").is_dir() else "sgbm"
    dataset = load_dataset(args.dataset, loader=loader, **loader_kwargs)
    logger.info(f"Dataset: {args.dataset} ({loader}), {len(dataset)} frames")

    camera = Camera(K=dataset.rgb_K, frame_width=dataset.rgb_width, frame_height=dataset.rgb_height)

    # the mode's shipped configuration (configs/modes/<mode>.yaml), then the user's --config files and --set overrides
    config = load_config(*shipped_config_files(args), *args.config, sets=args.set)
    system = session_factory(args, camera, config, T_right_in_left=getattr(dataset, "T_right_in_left", None),
                             visualize=not args.no_viz and args.mode != "mono")()
    if args.load_map:
        system.load_map(args.load_map)
        logger.info(f"Loaded map {args.load_map}: {len(system.hypothesis_manager.nodes)} keyframes")

    poses = open(args.save_poses, "w") if args.save_poses else None
    end_idx = min(args.start + args.frames, len(dataset)) if args.frames else len(dataset)
    t0 = time.time()
    n, placed = 0, None
    try:
        for idx, d in enumerate(dataset.replay_data(start_idx=args.start, end_idx=end_idx)):
            if idx == 0:
                d["delta_pose"] = None  # first frame: no motion
            system.process(d)
            n += 1
            if poses is not None and system.localized():
                if placed is None:
                    placed = idx
                    if args.load_map:
                        logger.info(f"Frame {idx}: the session is placed in the map")
                pose, _, _ = system.belief(_pose_matrix)
                t = d.get("timestamp")
                poses.write(_tum_line(float(idx) if t is None else float(t), pose) + "\n")
            if idx % 100 == 0:
                logger.info(f"Step {idx}/{end_idx - args.start}")
    finally:
        if poses is not None:
            poses.close()

    elapsed = time.time() - t0
    logger.info(f"Done: {n} frames in {elapsed:.1f}s ({n / max(elapsed, 1e-9):.1f} FPS), "
                f"{len(system.hypothesis_manager.nodes)} keyframes")
    if args.save_poses:
        logger.info(f"Poses: {args.save_poses}" + ("" if placed is not None else " (empty: never placed in the map)"))
    print_timing_registry()

    if args.save_map:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_map)), exist_ok=True)
        system.save_map(args.save_map)
        logger.info(f"Map saved: {args.save_map} (+ {args.save_map}.store/)")
    system.release()


if __name__ == "__main__":
    main()
