"""Interactive demo for CROSS topological mapping."""

import time

import numpy as np
from loguru import logger

from cross.core.system import System
from cross.core.types import Camera
from cross.core.planning_system import PlanningSystem
from cross.dataloader.r3d_loader import R3DDataset
from cross.utils.profile import print_timing_registry

np.set_printoptions(formatter={"float": lambda x: f"{x:0.2f}"})


def run(dataset: R3DDataset):
    logger.info(f"Total frames: {len(dataset)}")

    camera = Camera(
        K=dataset.rgb_K,
        frame_width=dataset.rgb_width,
        frame_height=dataset.rgb_height,
    )

    from cross.core.config import SystemConfig
    config = SystemConfig(async_update=False)
    system = System(visualize=True, debug=True, camera=camera, config=config)

    start_idx = 0
    end_idx = 10000
    start_idx_user = start_idx
    end_idx_user = start_idx_user + 100
    kidnapped = False
    planning = None

    while True:
        cmd = input("Command [l]oad [s]tart [g]o [p]lan [v]iz [q]uit: ").strip()
        if cmd == "q":
            break
        elif cmd == "l":
            seq_id = int(input("Sequence id: "))
            dataset.load_sequence(seq_id)
            logger.info(f"Loaded sequence {seq_id}")
        elif cmd == "s":
            start_idx_user = int(input("Start index: ")) + start_idx
            end_idx_user = start_idx_user + 100
            kidnapped = True
        elif cmd == "a":
            stop_at = int(input("Stop index: "))
        elif cmd == "p":
            goal_kf_id = int(input("Goal keyframe id: "))
            planning = PlanningSystem(system=system)
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
                end_idx_user = start_idx_user + 100
                kidnapped = True
            else:
                start_idx_user = end_idx_user

    system.shutdown()


if __name__ == "__main__":
    dataset = R3DDataset("data/r3d/lab2.r3d", snr=0.5)
    run(dataset)
