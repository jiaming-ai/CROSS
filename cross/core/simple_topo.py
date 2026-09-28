import heapq
from dataclasses import dataclass
from typing import Dict, Tuple, Set, List, Optional, Any

import torch
from loguru import logger

from cross.core.types import Edge, EdgeType
from cross.utils.spatial_hash import SpatialHashGrid3D
import pypose as pp

@dataclass
class SimpleTopoConfig:
    """Configuration for SimpleTopo behavior."""
    proximity_distance_thresh: float = 0.5
    proximity_std_trans: float = 0.05
    proximity_std_rot: float = 0.1
    use_proximity_grid: bool = False
    enable_incremental_proximity: bool = False


class SimpleTopo:
    """
    Simple topological graph over keyframes using odometry and proximity edges.

    Responsibilities
    - Maintain proximity edges between nearby permanent keyframes.
      - Optional incremental updates when a new keyframe is added.
      - Refresh proximity edges for nodes affected by PGO.
      - Full rebuild from scratch on demand.
    - Provide basic planning (Dijkstra / A*) over odometry + proximity edges.

    Notes
    - Proximity edges are undirected and never duplicate existing odometry edges.
    - Only permanent keyframes (temporary=False) are considered for proximity edges.
    - All structural reads/writes are guarded by the HypothesisManager.graph_lock.
    """

    def __init__(self, system, config: Optional[SimpleTopoConfig] = None):
        # Avoid hard import to prevent circular dependency with system.py
        assert hasattr(system, "hypothesis_manager"), "SimpleTopo expects a System with hypothesis_manager"

        self._system = system
        self._hm = system.hypothesis_manager
        self.config = config or SimpleTopoConfig()

        # Proximity graph storage (undirected, sorted node-id keys)
        self.proximity_edges: Dict[Tuple[int, int], Edge] = {}
        self.proximity_adjacency: Dict[int, Set[int]] = {}

        # Lightweight planning state
        self.current_plan: Optional[Dict[str, Any]] = None
        self.goal_kf_id: Optional[int] = None
        self.visited_waypoints: Set[int] = set()

    # ===================== Proximity edge maintenance ===================== #

    def _remove_proximity_edge(self, key: Tuple[int, int]):
        """Remove a stored proximity edge and adjacency entries."""
        if key in self.proximity_edges:
            del self.proximity_edges[key]
        a, b = key
        if a in self.proximity_adjacency:
            self.proximity_adjacency[a].discard(b)
            if not self.proximity_adjacency[a]:
                del self.proximity_adjacency[a]
        if b in self.proximity_adjacency:
            self.proximity_adjacency[b].discard(a)
            if not self.proximity_adjacency[b]:
                del self.proximity_adjacency[b]

    def _add_proximity_edge(self, id1: int, id2: int):
        """Add an undirected proximity edge if one does not already exist."""
        if id1 == id2:
            return

        key = tuple(sorted((id1, id2)))

        # Avoid duplicates
        if key in self.proximity_edges:
            return

        # Avoid parallel to existing odometry edges
        if (id1, id2) in self._hm.odom_edges or (id2, id1) in self._hm.odom_edges:
            return

        if id1 not in self._hm.nodes or id2 not in self._hm.nodes:
            return

        kf1 = self._hm.nodes[id1]
        kf2 = self._hm.nodes[id2]
        pose1 = kf1.pose_mu[0]
        pose2 = kf2.pose_mu[0]
        rel_mean = pose1.Inv() @ pose2

        device = pose1.tensor().device

        edge = Edge(
            mean=rel_mean,
            std=pp.se3(torch.tensor([0.05, 0.05, 0.05, 0.1, 0.1, 0.1], device=device)),
            type=EdgeType.PROXIMITY,
        )
        self.proximity_edges[key] = edge
        self.proximity_adjacency.setdefault(id1, set()).add(id2)
        self.proximity_adjacency.setdefault(id2, set()).add(id1)

    def handle_new_node(self, node_id: int):
        """
        Optionally add proximity edges from a newly added node to nearby permanent nodes.

        No-op if incremental proximity is disabled.
        """
        if not self.config.enable_incremental_proximity:
            return

        with self._hm.graph_lock:
            if node_id not in self._hm.nodes:
                return
            kf_new = self._hm.nodes[node_id]
            if kf_new.temporary:
                return

            pos_new = kf_new.pose_mu[0].tensor()[:3]

            # Scan all permanent nodes and connect those within distance threshold.
            for other in self._hm.nodes.values():
                if other.id == node_id or other.temporary:
                    continue
                pos_other = other.pose_mu[0].tensor()[:3]
                # Use x/z distance for proximity threshold to match existing logic.
                dist_xz = torch.norm(
                    pos_new[[0, 2]] - pos_other[[0, 2]]
                ).item()
                if dist_xz < self.config.proximity_distance_thresh:
                    self._add_proximity_edge(node_id, other.id)

    def update_after_pgo(self, affected_ids: Set[int]):
        """
        Refresh proximity edges for nodes whose poses changed after PGO.

        This mirrors the previous `_update_proximity_edges_after_pgo` logic but is
        encapsulated here. Assumes affected_ids is a set of keyframe IDs.
        """
        if not affected_ids:
            return

        with self._hm.graph_lock:
            # Only consider permanent nodes to avoid churn from temporary cleanup.
            positions = {
                kf.id: kf.pose_mu[0].tensor()[:3]
                for kf in self._hm.nodes.values()
                if not kf.temporary
            }

            if not positions:
                self.proximity_edges.clear()
                self.proximity_adjacency.clear()
                return

            affected_ids = set(affected_ids)

            # Drop existing proximity edges incident to affected nodes
            to_remove = [
                key
                for key in list(self.proximity_edges.keys())
                if key[0] in affected_ids or key[1] in affected_ids
            ]
            for key in to_remove:
                self._remove_proximity_edge(key)

            if self.config.use_proximity_grid:
                grid = SpatialHashGrid3D.from_positions(
                    positions, self.config.proximity_distance_thresh
                )
                for node_id in affected_ids:
                    if node_id not in positions:
                        continue
                    pos = positions[node_id]
                    for neighbor_id, neighbor_pos in grid.query(pos):
                        if neighbor_id == node_id:
                            continue
                        # Only consider x and z coordinates
                        if (
                            pos[:, [0, 2]] - neighbor_pos[:, [0, 2]]
                        ).norm().item() >= self.config.proximity_distance_thresh:
                            continue
                        self._add_proximity_edge(node_id, neighbor_id)
            else:
                node_ids = list(positions.keys())
                id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
                pos_tensor = torch.stack(
                    [positions[nid] for nid in node_ids], dim=0
                )
                for node_id in affected_ids:
                    idx = id_to_idx.get(node_id)
                    if idx is None:
                        continue
                    # Only consider x and z coordinates
                    dists = torch.norm(
                        pos_tensor[:, [0, 2]] - pos_tensor[idx, [0, 2]], dim=1
                    )
                    close_indices = (
                        torch.nonzero(
                            dists < self.config.proximity_distance_thresh,
                            as_tuple=False,
                        )
                        .flatten()
                        .tolist()
                    )
                    for j in close_indices:
                        neighbor_id = node_ids[j]
                        self._add_proximity_edge(node_id, neighbor_id)

    def rebuild_graph(self):
        """
        Rebuild all proximity edges from scratch using current node poses.

        This ignores any existing proximity edges and re-creates them by checking
        all pairs of permanent nodes for spatial proximity.
        """
        with self._hm.graph_lock:
            self.proximity_edges.clear()
            self.proximity_adjacency.clear()

            # Collect permanent nodes
            perm_nodes = [
                kf for kf in self._hm.nodes.values() if not kf.temporary
            ]
            if not perm_nodes:
                return

            # Extract node IDs and positions
            node_ids = [kf.id for kf in perm_nodes]
            positions = torch.stack(
                [kf.pose_mu[0].tensor()[:3] for kf in perm_nodes], dim=0
            )  # (N, 3)

            # Compute pairwise distances on x and z coordinates only
            pos_xz = positions[:, [0, 2]]  # (N, 2)

            # Compute pairwise distance matrix
            # Broadcasting: (N, 1, 2) - (1, N, 2) = (N, N, 2)
            diffs = pos_xz.unsqueeze(1) - pos_xz.unsqueeze(0)  # (N, N, 2)
            dists = torch.norm(diffs, dim=2)  # (N, N)

            # Find pairs below threshold (upper triangle only to avoid duplicates)
            mask = dists < self.config.proximity_distance_thresh
            mask = torch.triu(mask, diagonal=1)  # Zero out diagonal and lower triangle

            # Get indices of pairs to connect
            pairs = torch.nonzero(mask, as_tuple=False)  # (M, 2)

            # Add proximity edges
            for pair in pairs:
                i, j = pair.tolist()
                self._add_proximity_edge(node_ids[i], node_ids[j])

    # ===================== Planning over odom + proximity ===================== #

    def _get_neighbors(self, kf_id: int) -> List[int]:
        """Get neighbors of a keyframe via odometry and proximity edges."""
        neighbors: Set[int] = set()

        # Proximity neighbors (undirected)
        neighbors.update(self.proximity_adjacency.get(kf_id, set()))

        # Odometry neighbors (treat as undirected for planning)
        for (id1, id2) in self._hm.odom_edges.keys():
            if id1 == kf_id:
                neighbors.add(id2)
            elif id2 == kf_id:
                neighbors.add(id1)

        return list(neighbors)

    def _get_edge_cost(self, from_kf_id: int, to_kf_id: int) -> float:
        """Get cost of an edge, using proximity first, then odometry."""
        # Proximity edge (undirected)
        prox_key = tuple(sorted((from_kf_id, to_kf_id)))
        prox_edge = self.proximity_edges.get(prox_key)
        if prox_edge is not None:
            return prox_edge.cost

        # Odometry edge (directed, but we treat as bidirectional if either direction exists)
        odom_key_fwd = (from_kf_id, to_kf_id)
        odom_key_bwd = (to_kf_id, from_kf_id)

        if odom_key_fwd in self._hm.odom_edges:
            return self._hm.odom_edges[odom_key_fwd].cost
        if odom_key_bwd in self._hm.odom_edges:
            return self._hm.odom_edges[odom_key_bwd].cost

        return float("inf")

    def _compute_heuristic(self, from_kf_id: int, to_kf_id: int) -> float:
        """Straight-line distance between two keyframes (translation norm)."""
        from_kf = self._hm.nodes.get(from_kf_id)
        to_kf = self._hm.nodes.get(to_kf_id)
        if from_kf is None or to_kf is None:
            return 0.0

        from_pos = from_kf.pose_mu[0].translation()
        to_pos = to_kf.pose_mu[0].translation()
        return torch.norm(to_pos - from_pos).item()

    def _dijkstra_search(self, start_id: int, goal_id: int) -> Optional[List[int]]:
        """Dijkstra search over odometry + proximity edges."""
        if start_id == goal_id:
            return [start_id]

        pq: List[Tuple[float, int]] = [(0.0, start_id)]
        costs: Dict[int, float] = {start_id: 0.0}
        parents: Dict[int, Optional[int]] = {start_id: None}
        visited: Set[int] = set()

        while pq:
            current_cost, current_id = heapq.heappop(pq)
            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id == goal_id:
                break

            for neighbor_id in self._get_neighbors(current_id):
                if neighbor_id in visited:
                    continue

                edge_cost = self._get_edge_cost(current_id, neighbor_id)
                if edge_cost == float("inf"):
                    continue

                new_cost = current_cost + edge_cost
                if neighbor_id not in costs or new_cost < costs[neighbor_id]:
                    costs[neighbor_id] = new_cost
                    parents[neighbor_id] = current_id
                    heapq.heappush(pq, (new_cost, neighbor_id))

        if goal_id not in parents:
            return None

        # Reconstruct path
        path: List[int] = []
        curr = goal_id
        while curr is not None:
            path.append(curr)
            curr = parents[curr]
        path.reverse()
        return path

    def _astar_search(self, start_id: int, goal_id: int) -> Optional[List[int]]:
        """A* search with straight-line heuristic."""
        if start_id == goal_id:
            return [start_id]

        initial_h = self._compute_heuristic(start_id, goal_id)
        pq: List[Tuple[float, float, int]] = [(initial_h, 0.0, start_id)]
        g_scores: Dict[int, float] = {start_id: 0.0}
        parents: Dict[int, Optional[int]] = {start_id: None}
        visited: Set[int] = set()

        while pq:
            f_score, g_score, current_id = heapq.heappop(pq)
            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id == goal_id:
                break

            for neighbor_id in self._get_neighbors(current_id):
                if neighbor_id in visited:
                    continue

                edge_cost = self._get_edge_cost(current_id, neighbor_id)
                if edge_cost == float("inf"):
                    continue

                new_g_score = g_score + edge_cost
                if neighbor_id not in g_scores or new_g_score < g_scores[neighbor_id]:
                    g_scores[neighbor_id] = new_g_score
                    parents[neighbor_id] = current_id

                    h_score = self._compute_heuristic(neighbor_id, goal_id)
                    f_score = new_g_score + h_score
                    heapq.heappush(pq, (f_score, new_g_score, neighbor_id))

        if goal_id not in parents:
            return None

        # Reconstruct path
        path: List[int] = []
        curr = goal_id
        while curr is not None:
            path.append(curr)
            curr = parents[curr]
        path.reverse()
        return path

    def plan(self, start_id: int, goal_id: int, algorithm: str = "astar") -> Optional[List[int]]:
        """
        Plan a path between two nodes using odometry + proximity edges.

        Args:
            start_id: Current node ID.
            goal_id: Target node ID.
            algorithm: "astar" (default) or "dijkstra".

        Returns:
            List of node IDs from start to goal, or None if unreachable.
        """
        with self._hm.graph_lock:
            if start_id not in self._hm.nodes or goal_id not in self._hm.nodes:
                logger.warning(f"Plan requested between non-existent nodes: {start_id}, {goal_id}")
                return None

            if algorithm.lower() == "dijkstra":
                return self._dijkstra_search(start_id, goal_id)
            else:
                return self._astar_search(start_id, goal_id)

    # ===================== Simple planning helpers ===================== #

    def _compute_path_cost(self, path: List[int]) -> float:
        """Sum edge costs along a path; returns inf if any edge is missing."""
        if not path or len(path) == 1:
            return 0.0

        cost = 0.0
        with self._hm.graph_lock:
            for i in range(len(path) - 1):
                edge_cost = self._get_edge_cost(path[i], path[i + 1])
                if edge_cost == float("inf"):
                    return float("inf")
                cost += edge_cost
        return cost

    def plan_to_goal(self, target_kf_id: int, algorithm: str = "astar") -> Optional[Dict[str, Any]]:
        """
        Plan from the current closest permanent node to a target node.

        Mirrors the PlanningSystem usage pattern: fetch current node from System,
        then run graph search over odometry + proximity edges.
        """
        if target_kf_id is None:
            logger.warning("No target_kf_id provided for plan_to_goal")
            return None

        state = self._system.get_current_kf()
        if state is None or state.get("closest_perm_node") is None:
            logger.warning("Current closest permanent node is not available for planning")
            return None

        start_kf = state["closest_perm_node"]
        path = self.plan(start_kf.id, target_kf_id, algorithm=algorithm)
        if path is None:
            logger.warning(f"[{algorithm.upper()}] No path found from {start_kf.id} to {target_kf_id}")
            self.current_plan = None
            self.goal_kf_id = None
            return None

        cost = self._compute_path_cost(path)
        self.goal_kf_id = target_kf_id
        self.visited_waypoints = set()
        self.current_plan = {
            "path": path,
            "cost": cost,
            "algorithm": algorithm,
            "start_id": start_kf.id,
            "goal_id": target_kf_id,
        }
        return self.current_plan

    def get_next_waypoint(self, next_n_waypoints: int = 5) -> Optional[Dict[str, Any]]:
        """
        Get the next waypoint(s) along the current topo plan.

        Returns waypoints and their poses in the current local frame, similar to
        Planner.get_next_waypoint in the planning system.
        """
        if self.current_plan is None or not self.current_plan.get("path"):
            return None

        state = self._system.get_current_kf()
        if state is None or state.get("closest_perm_node") is None:
            return None

        current_kf = state["closest_perm_node"]
        current_kf_id = current_kf.id

        self.visited_waypoints.add(current_kf_id)
        path = self.current_plan["path"]

        # Find current position in the path
        try:
            current_idx = path.index(current_kf_id)
        except ValueError:
            min_dist = float("inf")
            closest_idx = 0
            for i, waypoint_id in enumerate(path):
                if waypoint_id in self.visited_waypoints:
                    continue
                dist = self._compute_heuristic(current_kf_id, waypoint_id)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            current_idx = max(closest_idx - 1, -1)

        if current_kf_id == self.goal_kf_id:
            return {
                "waypoints": [],
                "current_position": len(path) - 1,
                "remaining_waypoints": 0,
                "reached_goal": True,
                "waypoint_poses": [],
            }

        waypoints: List[int] = []
        waypoint_poses = []
        with self._hm.graph_lock:
            for i in range(current_idx + 1, len(path)):
                waypoint_id = path[i]
                if waypoint_id not in self.visited_waypoints or waypoint_id == self.goal_kf_id:
                    waypoints.append(waypoint_id)
                    if waypoint_id in self._hm.nodes:
                        waypoint_poses.append(self._hm.nodes[waypoint_id].pose_mu[0])
                    if len(waypoints) >= next_n_waypoints:
                        break

        waypoint_poses_np = []
        if waypoint_poses:
            current_pose = self._hm.dist[0][0].unsqueeze(0)
            waypoint_poses_tensor = torch.stack(waypoint_poses)
            waypoint_poses_local = current_pose.Inv() @ waypoint_poses_tensor
            waypoint_poses_np = waypoint_poses_local.cpu().numpy()

        remaining_count = 0
        for i in range(current_idx + 1, len(path)):
            if path[i] not in self.visited_waypoints or path[i] == self.goal_kf_id:
                remaining_count += 1

        return {
            "waypoints": waypoints,
            "current_position": current_idx,
            "remaining_waypoints": remaining_count,
            "reached_goal": len(waypoints) == 0 or (len(waypoints) == 1 and waypoints[0] == self.goal_kf_id),
            "waypoint_poses": waypoint_poses_np,
        }
