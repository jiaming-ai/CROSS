"""
Planning System - Unified interface for path planning

This module provides a self-contained planning system that wraps both SparseGraph
and Planner components, providing a clean API for all planning operations.

The PlanningSystem:
- Encapsulates sparse graph management and path planning
- Maintains reference to mapping system for data access
- Automatically stays synchronized with mapping updates
- Provides unified API for planning operations
"""

from typing import List, Optional, Dict, Any, Literal
from loguru import logger
from cross.core.sparse_graph import SparseGraph
from cross.core.planner import Planner


class PlanningSystem:
    """
    Unified path planning system that wraps the sparse planning graph and planner.

    Overview
    - Encapsulates construction and maintenance of a sparse planning graph over the
      pose graph produced by the mapping system.
    - Provides an API to set a goal, plan a path (sparse + detailed), and stream
      the next waypoints for navigation.
    - Reacts to mapping updates (new permanent keyframes, loop closures) to keep
      the planning graph in sync.

    Submodules and Responsibilities
    - SparseGraph (`cross.core.sparse_graph.SparseGraph`)
      - Permanent backbone: Maintains an ordered list of permanent keyframes
        (`permanent_kf_ids`) and builds backbone edges between consecutive permanents.
        Each backbone edge stores a composed SE3 transform, uncertainty, and cost
        computed as the sum of the underlying odometry chain between those permanents.
      - Binary lifting jump tables: Precomputes range compositions for powers of two
        strides to enable O(log L) queries across long ranges (composition of transforms,
        variances, and costs).
      - Chain shortcut edges: Creates long-range shortcut edges at phase-aligned
        indices for stride `S = 2^p`. For each stride, only indices where `idx % S == 0`
        are connected to `idx - S`, keeping the graph sparse while preserving optimal
        path costs. Both backbone and shortcuts are exposed as undirected edges for
        planning adjacency, keyed by sorted keyframe ID tuples.
      - Update lifecycle:
        - `update_on_new_permanent_keyframe()`: Extends the backbone and periodically
          calls `create_shortcut_edges()` when enough permanents exist.
        - `rebuild_after_loop_closure(affected_kf_ids)`: Rebuilds backbone/jump tables
          and regenerates shortcuts starting at the earliest affected permanent after PGO.
        - `rebuild_from_index(i)`: Internal helper used for partial rebuilds.
      - Queries:
        - `get_edge_cost(u, v)`: Returns the cost of a backbone or shortcut edge if present.
        - `get_sparse_graph_neighbors(u)`: Returns undirected neighbors in the sparse graph.

    - Planner (`cross.core.planner.Planner`)
      - Graph used for search: Combines sparse graph neighbors (backbone + shortcuts)
        with odometry neighbors when `use_temporary_keyframes=True`.
      - Edge costs: Prefer sparse graph edge costs when available; otherwise fall back
        to odometry edge costs. Odometry edges are stored directed but treated as
        traversable in either direction for planning cost queries.
      - Algorithms: `plan_to_goal(algorithm)` supports `"astar"` and `"dijkstra"`.
        - A*: Uses straight-line Euclidean distance between the translations of the
          first GMM components of the keyframe poses as the heuristic.
        - Dijkstra: Uniform-cost search without a heuristic.
      - Path detail levels:
        - Sparse only: return the sparse path as-is.
        - Backbone: expand between consecutive sparse waypoints by slicing the ordered
          list of permanent keyframes in the correct direction.
        - Odometry: expand between consecutive sparse waypoints by running a Dijkstra
          over an odometry-only, undirected adjacency to recover the full chain of
          keyframes (including temporary KFs) in either direction. This guarantees
          detailed paths correctly traverse back along odometry when needed.
      - Backtracking removal: After segment expansion, removes loops/backtracking by
        dropping segments between repeated nodes while preserving the first occurrence.
      - Path cost: Sum of translation norms along consecutive edges. If any segment is
        missing a valid edge, the cost is treated as infinity (invalid path).

    Usage Pattern
    1) Specify a goal
       - `set_goal(goal_kf_id=...)` to pin to a known keyframe ID.
       - Or `set_goal(goal_image=...)` using VPR to select a goal keyframe.
       - (Text goals are stubbed) `set_goal(goal_text=...)`.
    2) Plan to the goal
       - `plan_to_goal(algorithm="astar", sparse_only=False)` returns a dict:
         - `sparse_path`: path over backbone+shortcuts.
         - `detailed_path`: expanded path at the selected detail level
           (`"odometry"` if temporary keyframes are enabled, otherwise `"backbone"`).
         - `sparse_cost`, `detailed_cost`, `algorithm`, `detail_level`.
    3) Stream the next waypoints
       - `get_next_waypoint(next_n_waypoints=5)` returns the next IDs on the current
         detailed path, given the system's current closest permanent node. It tracks
         visited waypoints to avoid repeats and reports whether the goal is reached.

    Integration with System
    - The planning system registers itself on initialization so the mapping system can
      notify it of events:
      - `on_new_permanent_keyframe()` to update the backbone and shortcuts.
      - `on_loop_closure(affected_kf_ids)` to rebuild affected regions.
    - `rebuild_graph()` can be called after loading a map to rebuild from scratch.
    """

    def __init__(
        self,
        system,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        """
        Initialize the planning system.

        Args:
            system: The SLAM system instance to plan within
            config: Optional configuration dict with keys:
                - max_stride_power: Maximum stride as power of 2 (default: 6)
                - min_perm_for_shortcuts: Min permanent keyframes before shortcuts (default: 4)
                - update_frequency: Create shortcuts every N new permanent keyframes (default: 5)
                - use_temporary_keyframes: Include temporary keyframes in planning (default: True)
            enabled: Whether planning is enabled (default: True)
        """
        self.system = system
        self.enabled = enabled

        # Parse configuration
        if config is None:
            config = {}

        # Sparse graph configuration
        sparse_config = {
            "max_stride_power": config.get("max_stride_power", 6),
            "min_perm_for_shortcuts": config.get("min_perm_for_shortcuts", 4),
            "update_frequency": config.get("update_frequency", 5),
        }

        # Planner configuration
        use_temporary_keyframes = config.get("use_temporary_keyframes", False)

        # Initialize components
        if self.enabled:
            logger.info("Initializing planning system...")

            # Create sparse graph
            self.sparse_graph = SparseGraph(
                system=system,
                max_stride_power=sparse_config["max_stride_power"],
                min_perm_for_shortcuts=sparse_config["min_perm_for_shortcuts"],
                update_frequency=sparse_config["update_frequency"],
                device=system.device,
            )

            # Create planner
            self.planner = Planner(
                system=system,
                sparse_graph=self.sparse_graph,
                use_temporary_keyframes=use_temporary_keyframes,
            )

            # Register with system for update notifications
            system.planning_system = self

            # Check if system already has keyframes (e.g., after loading a map)
            # and build the sparse graph on demand
            perm_kfs = [kf for kf in system.hypothesis_manager.nodes.values() if not kf.temporary]
            if len(perm_kfs) >= 2:
                logger.info(f"Found {len(perm_kfs)} existing permanent keyframes, building sparse graph...")
                self.rebuild_graph()

            logger.info(f"Planning system initialized (max_stride={2**sparse_config['max_stride_power']}, "
                       f"use_temp_kfs={use_temporary_keyframes})")
        else:
            self.sparse_graph = None
            self.planner = None
            logger.info("Planning system disabled")

    # ============ Notification Handlers (called by System) ============

    def on_new_permanent_keyframe(self, kf_id: Optional[int] = None):
        """
        Called when a new permanent keyframe is added to the system.
        Updates sparse graph backbone and shortcuts.

        Args:
            kf_id: ID of the newly added permanent keyframe (unused, for backward compatibility)
        """
        if not self.enabled or self.sparse_graph is None:
            return

        self.sparse_graph.update_on_new_permanent_keyframe()

    def on_loop_closure(self, affected_kf_ids: set):
        """
        Called when loop closure occurs and poses are updated.
        Rebuilds sparse graph for affected regions.

        Args:
            affected_kf_ids: Set of keyframe IDs whose poses were updated
        """
        if not self.enabled or self.sparse_graph is None:
            return

        self.sparse_graph.rebuild_after_loop_closure(affected_kf_ids)

    def rebuild_graph(self):
        """
        Rebuild the entire sparse graph from scratch.
        Useful after loading a map or major changes.
        """
        if not self.enabled or self.sparse_graph is None:
            return

        logger.info("Rebuilding sparse graph from scratch...")

        # Get all permanent keyframes
        perm_kfs = sorted(
            [kf for kf in self.system.hypothesis_manager.nodes.values() if not kf.temporary],
            key=lambda k: k.id
        )

        if len(perm_kfs) < 2:
            logger.warning("Not enough permanent keyframes to rebuild graph")
            return

        # Rebuild backbone and shortcuts by processing each permanent keyframe
        for kf in perm_kfs:
            self.sparse_graph.update_on_new_permanent_keyframe()

        # Get stats
        stats = self.sparse_graph.get_stats()
        logger.info(f"Rebuilt sparse graph: {stats['shortcut_edges']} shortcuts, "
                   f"max_stride={stats['max_stride']}")

    # ============ Planning API ============

    def set_goal(
        self,
        goal_kf_id: Optional[int] = None,
        goal_text: Optional[str] = None,
        goal_image: Optional[Any] = None,
        plan_to_goal: bool = False,
    ):
        """
        Set the goal for planning.

        Args:
            goal_kf_id: Goal keyframe ID (direct specification)
            goal_text: Natural language goal description (future feature)
            goal_image: Goal image for visual place recognition
            plan_to_goal: Whether to immediately plan to the goal
        """
        if not self.enabled or self.planner is None:
            logger.warning("Planning system is disabled")
            return

        self.planner.set_goal(
            goal_kf_id=goal_kf_id,
            goal_text=goal_text,
            goal_image=goal_image,
            plan_to_goal=plan_to_goal,
        )

    def plan_to_goal(
        self,
        algorithm: Literal["dijkstra", "astar"] = "astar",
        sparse_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Plan a path to the goal keyframe.

        Args:
            algorithm: Planning algorithm - "dijkstra" or "astar" (default)
            sparse_only: If True, return only sparse path without expansion

        Returns:
            dict with keys:
                - sparse_path: List[int] - Sparse path using shortcuts
                - detailed_path: List[int] - Detailed path for navigation
                - sparse_cost: float - Cost of sparse path
                - detailed_cost: float - Cost of detailed path
                - algorithm: str - Algorithm used
                - detail_level: str - Detail level used
            Returns None if no path found or planning disabled
        """
        if not self.enabled or self.planner is None:
            logger.warning("Planning system is disabled")
            return None

        return self.planner.plan_to_goal(
            algorithm=algorithm,
            sparse_only=sparse_only,
        )

    def get_next_waypoint(self, next_n_waypoints: int = 5) -> Optional[Dict[str, Any]]:
        """
        Get the next waypoint(s) along the current plan.

        Args:
            next_n_waypoints: Number of waypoints to return (default: 5)

        Returns:
            dict with keys:
                - waypoints: List[int] - Next keyframe IDs
                - current_position: int - Current position in plan
                - remaining_waypoints: int - Waypoints remaining to goal
                - reached_goal: bool - Whether goal has been reached
            Returns None if no plan exists or planning disabled
        """
        if not self.enabled or self.planner is None:
            logger.warning("Planning system is disabled")
            return None

        return self.planner.get_next_waypoint(next_n_waypoints)

    # ============ Visualization API ============

    def visualize_graph(
        self,
        path: Optional[List[int]] = None,
        title: str = "Sparse Graph",
        save_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Visualize the sparse graph with optional path overlay.

        Args:
            path: Optional list of keyframe IDs to highlight as path
            title: Plot title
            save_path: Optional path to save visualization
            backend: Visualization backend - "plotly" (interactive) or "matplotlib"
            **kwargs: Additional arguments passed to visualization function
        """
        if not self.enabled or self.sparse_graph is None:
            logger.warning("Planning system is disabled, cannot visualize")
            return

        from cross.visualization.viz_graph import (
            visualize_sparse_graph_matplotlib,
        )

        visualize_sparse_graph_matplotlib(
            system=self.system,
            path=path,
            title=title,
            save_path=save_path,
            **kwargs,
        )

    # ============ Utility API ============

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the sparse graph.

        Returns:
            Dictionary with graph statistics:
                - permanent_keyframes: int
                - backbone_edges: int
                - shortcut_edges: int
                - max_stride: int
                - enabled: bool
        """
        if not self.enabled or self.sparse_graph is None:
            return {"enabled": False}

        stats = self.sparse_graph.get_stats()
        stats["enabled"] = True
        return stats

    def get_edge_cost(self, from_kf_id: int, to_kf_id: int) -> float:
        """
        Get the cost of a sparse graph edge between two keyframes.

        Args:
            from_kf_id: Source keyframe ID
            to_kf_id: Target keyframe ID

        Returns:
            Edge cost (translation norm), or infinity if no edge exists
        """
        if not self.enabled or self.sparse_graph is None:
            return float('inf')

        return self.sparse_graph.get_edge_cost(from_kf_id, to_kf_id)

    def get_neighbors(self, kf_id: int) -> set:
        """
        Get all sparse graph neighbors of a keyframe.

        Args:
            kf_id: Keyframe ID

        Returns:
            Set of neighbor keyframe IDs (empty if planning disabled)
        """
        if not self.enabled or self.sparse_graph is None:
            return set()

        return self.sparse_graph.get_sparse_graph_neighbors(kf_id)
