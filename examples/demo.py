"""Interactive demo for CROSS topological mapping (RGB-D mode, configs/modes/rgbd_wheel.yaml).

Usage:
    python examples/demo.py data/posed/office1-1

Commands: [s]tart index (the next run starts there without a pose, as if the robot were carried there), [g]o (run
to an end index), [p]lan a path to a keyframe, [v]isualize the planning graph, [q]uit.  Sequences: any format run.py
reads (posed RGB-D folders, R3D files, ...).
"""

import argparse
import time
from pathlib import Path

import numpy as np
from loguru import logger

from cross.core.config import load_config
from cross.core.planning_system import PlanningSystem
from cross.core.system import System
from cross.core.types import Camera
from cross.dataloader.auto import load_dataset
from cross.utils.profile import print_timing_registry

np.set_printoptions(formatter={"float": lambda x: f"{x:0.2f}"})
MODE_FILE = Path(__file__).resolve().parents[1] / "configs" / "modes" / "rgbd_wheel.yaml"


def run(dataset):
    logger.info(f"Total frames: {len(dataset)}")

    camera = Camera(
        K=dataset.rgb_K,
        frame_width=dataset.rgb_width,
        frame_height=dataset.rgb_height,
    )
    system = System(visualize=True, camera=camera, config=load_config(MODE_FILE))

    start_idx = 0
    end_idx = len(dataset)
    start_idx_user = start_idx
    kidnapped = False
    planning = None

    while True:
        cmd = input("Command [s]tart [g]o [p]lan [v]iz [q]uit: ").strip()
        if cmd == "q":
            break
        elif cmd == "s":
            start_idx_user = int(input("Start index: ")) + start_idx
            kidnapped = True
        elif cmd == "p":
            goal_kf_id = int(input("Goal keyframe id: "))
            planning = PlanningSystem(system=system)          # built from the current map
            planning.set_goal(goal_kf_id=goal_kf_id)
            result = planning.plan_to_goal()
            if result and result["sparse_path"]:
                path = result["detailed_path"] or result["sparse_path"]
                logger.info(f"Path ({len(path)} keyframes): {' -> '.join(map(str, path))}")
                planning.visualize_graph(path=path, title=f"Path to KF {goal_kf_id}")
            else:
                logger.warning("No path found!")
        elif cmd == "v":
            if planning:
                planning.visualize_graph(title="Sparse Graph")
            else:
                logger.warning("Run 'p' first to initialize planning.")
        elif cmd == "g":
            end_input = input("End index (or enter for all): ").strip()
            end_idx_user = int(end_input) if end_input.isdigit() else end_idx

            reply = dataset.replay_data(start_idx=start_idx_user, end_idx=end_idx_user)
            t0 = time.time()

            for idx, d in enumerate(reply):
                if kidnapped:
                    d["delta_pose"] = None
                    kidnapped = False

                system.step(obs=d, data=d)

                if idx % 100 == 0:
                    logger.info(f"Step {idx}")
                if idx % 500 == 0:
                    print_timing_registry()

            logger.info(f"Done in {time.time() - t0:.2f}s")

            if end_idx_user >= end_idx:
                start_idx_user = start_idx
                kidnapped = True
            else:
                start_idx_user = end_idx_user

    system.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CROSS interactive demo")
    parser.add_argument("sequence", help="sequence path (posed RGB-D folder, .r3d file, ...)")
    parser.add_argument("--snr", type=float, default=None, help="noise of simulated odometry (sequences without one)")
    args = parser.parse_args()
    run(load_dataset(args.sequence, **({"snr": args.snr} if args.snr is not None else {})))
