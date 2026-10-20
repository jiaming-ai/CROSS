"""
Save/load a map and run topological planning with CROSS (RGB-D mode, configs/modes/rgbd_wheel.yaml).

Usage:
    # Full pipeline: build map, save, load, relocalize, plan
    python examples/planner.py --map-scene data/posed/office1-1 --reloc-scene data/posed/office1-2

    # Skip mapping, load existing map and relocalize
    python examples/planner.py --skip-map --map-file maps/office/map.pkl --reloc-scene data/posed/office1-2

    # Just build and save a map
    python examples/planner.py --map-scene data/posed/office1-1 --map-file maps/office/map.pkl --skip-reloc

Sequences: any format run.py reads (posed RGB-D folders, R3D files, ...). --sem-query needs pip install -e ".[detection]".
"""

import argparse
import os
from pathlib import Path

from loguru import logger

from cross.core.config import load_config
from cross.core.system import System
from cross.core.types import Camera
from cross.dataloader.auto import load_dataset

MODE_FILE = Path(__file__).resolve().parents[1] / "configs" / "modes" / "rgbd_wheel.yaml"


def run_planner(
    map_scene: str | None = None,
    reloc_scene: str | None = None,
    map_file: str = "maps/map.pkl",
    skip_map: bool = False,
    skip_reloc: bool = False,
    reloc_start: int = 0,
    reloc_end: int | None = None,
    sem_query: str | None = None,
    visualize: bool = True,
    target_height: int = 480,
    target_width: int = 640,
):
    """
    End-to-end: build map -> save -> load -> relocalize -> plan.

    Args:
        map_scene: Path to the dataset for building the initial map.
        reloc_scene: Path to the dataset for relocalization.
        map_file: Path to save/load the map file.
        skip_map: If True, skip map building and load from map_file.
        skip_reloc: If True, skip relocalization and planning phases.
        reloc_start: Start frame for relocalization.
        reloc_end: End frame for relocalization (default: all).
        sem_query: Semantic memory query to run after mapping (optional).
        visualize: Enable visualization.
        target_height: Target image height for resizing.
        target_width: Target image width for resizing.
    """
    visualizer = None

    # Phase 1: Build and save map
    if not skip_map:
        if map_scene is None:
            raise ValueError("--map-scene is required when not using --skip-map")

        logger.info("Phase 1: Building initial map")
        dataset = load_dataset(map_scene, target_height=target_height, target_width=target_width)
        logger.info(f"Dataset: {map_scene}, {len(dataset)} frames")

        camera = Camera(K=dataset.rgb_K, frame_width=dataset.rgb_width, frame_height=dataset.rgb_height)

        if visualize:
            from cross.visualization.viz_rr import RRViz
            visualizer = RRViz(camera=camera, hypothesis_manager=None, visualize=True, visualize_pointcloud=False)

        system = System(visualize=visualize, camera=camera, visualizer=visualizer, config=load_config(MODE_FILE))

        for idx, d in enumerate(dataset.replay_data()):
            if idx == 0:
                d["delta_pose"] = None
            system.step(obs=d, data=d)

        logger.info(f"Map built: {len(system.hypothesis_manager.nodes)} keyframes")

        if sem_query:
            from cross.core.mem import SemanticMemoryManager
            logger.info(f"Semantic memory search: '{sem_query}'")
            mem = SemanticMemoryManager(system)
            res = mem.search_semantic_memory(sem_query)
            top_kfs = [r["kf_id"] for r in (res.get("results") or [])[:20]]
            logger.info(f"Top keyframes: {top_kfs}")
            os.makedirs("logs", exist_ok=True)
            mem.visualize(res, k=5, save_path=f"logs/sem_mem_{sem_query}.png", show=False)

        os.makedirs(os.path.dirname(os.path.abspath(map_file)), exist_ok=True)
        system.save_map(map_file)
        logger.info(f"Map saved to {map_file}")

        system.shutdown()
        del system

    if skip_reloc:
        logger.info("Skipping relocalization and planning (--skip-reloc).")
        return

    # Phase 2: Load map and relocalize
    if reloc_scene is None:
        raise ValueError("--reloc-scene is required when not using --skip-reloc")

    logger.info("Phase 2: Load map and relocalize")
    dataset = load_dataset(reloc_scene, target_height=target_height, target_width=target_width)
    camera = Camera(K=dataset.rgb_K, frame_width=dataset.rgb_width, frame_height=dataset.rgb_height)

    system = System(visualize=visualize, camera=camera, visualizer=visualizer, config=load_config(MODE_FILE))
    system.load_map(map_file)
    logger.info(f"Map loaded: {len(system.hypothesis_manager.nodes)} keyframes")

    actual_end = reloc_end or len(dataset)
    for idx, d in enumerate(dataset.replay_data(start_idx=reloc_start, end_idx=actual_end)):
        if idx == 0:
            d["delta_pose"] = None
        system.step(obs=d, data=d)
        if (idx + 1) % 100 == 0:
            logger.info(f"  {idx + 1}/{actual_end - reloc_start} frames")

    logger.info(f"Relocalization done: {len(system.hypothesis_manager.nodes)} keyframes")

    # Phase 3: Planning
    logger.info("Phase 3: Path planning")

    topo_map = getattr(system, "topo_map", None)
    if topo_map is None:
        logger.warning("Topo map not initialized; skipping planning.")
        system.shutdown()
        return

    if not topo_map.config.enable_incremental_proximity:
        topo_map.rebuild_graph()

    current_kf_ids = sorted(system.hypothesis_manager.nodes.keys())
    if not current_kf_ids:
        logger.warning("No keyframes available for planning.")
        system.shutdown()
        return

    target_kf_id = current_kf_ids[0]
    logger.info(f"Planning path to keyframe {target_kf_id}...")
    plan = topo_map.plan_to_goal(target_kf_id)

    if plan and plan.get("path"):
        path = plan["path"]
        logger.info(f"Path: {len(path)} keyframes, cost={plan['cost']:.2f}")
        logger.info(f"  {' -> '.join(map(str, path))}")

        waypoints = topo_map.get_next_waypoint(next_n_waypoints=10)
        if waypoints:
            logger.info(f"  Next waypoints: {waypoints['waypoints']}")
    else:
        logger.warning("No path found")

    system.shutdown()
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CROSS: Build map, save/load, relocalize, and plan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--map-scene", help="Dataset path for building the initial map")
    parser.add_argument("--reloc-scene", help="Dataset path for relocalization")
    parser.add_argument("--map-file", default="maps/map.pkl",
                        help="Path to save/load the map (default: maps/map.pkl)")
    parser.add_argument("--skip-map", action="store_true",
                        help="Skip map building; load from --map-file instead")
    parser.add_argument("--skip-reloc", action="store_true",
                        help="Skip relocalization and planning (map-only mode)")
    parser.add_argument("--reloc-start", type=int, default=0, help="Start frame for relocalization")
    parser.add_argument("--reloc-end", type=int, default=None, help="End frame for relocalization")
    parser.add_argument("--sem-query", default=None,
                        help="Run semantic memory search after mapping (e.g., 'coke')")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--height", type=int, default=480, help="Target image height (default: 480)")
    parser.add_argument("--width", type=int, default=640, help="Target image width (default: 640)")
    args = parser.parse_args()

    run_planner(
        map_scene=args.map_scene,
        reloc_scene=args.reloc_scene,
        map_file=args.map_file,
        skip_map=args.skip_map,
        skip_reloc=args.skip_reloc,
        reloc_start=args.reloc_start,
        reloc_end=args.reloc_end,
        sem_query=args.sem_query,
        visualize=not args.no_viz,
        target_height=args.height,
        target_width=args.width,
    )
