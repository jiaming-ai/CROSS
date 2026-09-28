import numpy as np
import torch
from typing import List, Literal
from loguru import logger


class Planner:
    """Path planner for navigation using odometry and shortcut edges.

    Uses Dijkstra's algorithm on a graph composed of:
    - Odometry edges (sequential, between consecutive keyframes)
    - Backbone edges (direct edges between consecutive permanent keyframes)
    - Chain shortcut edges (long-range connections via binary lifting)
    """

    def __init__(self, system, sparse_graph=None, use_temporary_keyframes: bool = True):
        """
        Initialize the planner.

        Args:
            system: The SLAM system instance
            sparse_graph: The SparseGraph instance (optional, for new architecture)
            use_temporary_keyframes: If True, include temporary keyframes and odometry edges
                                     in planning. If False, only use permanent keyframes via
                                     sparse graph (backbone + shortcuts). Default is True.
        """
        self.system = system
        self.use_temporary_keyframes = use_temporary_keyframes
        # Cache for hypothesis manager and sparse graph
        self.hypothesis_manager = system.hypothesis_manager
        self.sparse_graph = sparse_graph


        self.current_plan = None
        self.goal_kf_id = None
        self.visited_waypoints = set()  # Track visited waypoints in current plan

    def set_goal(
        self,
        goal_kf_id: int = None,
        goal_text: str = None,
        goal_image: np.ndarray = None,
        plan_to_goal: bool = True,
    ):
        """
        Set the goal kf_id.

        Currently the system support setting the goal by either image or text.
        - Image: use the image to retrieve the best matching keyframe as the goal.
        - Text: Use natural language to retrieve the best matching keyframe as the goal.
        It first interprets the text using LLM, then based on the intent:
            - find object: use detection model to search for object in the images, then use
            the kf with highest confidence as the goal.

        Args:
            goal_kf_id: The goal kf_id.
            goal_text: The goal text.
            goal_image: The goal image.
        """
        if goal_kf_id is not None:
            self.goal_kf_id = goal_kf_id
            return
        if goal_text is not None:
            self._set_goal_by_text(goal_text)
        elif goal_image is not None:
            self._set_goal_by_image(goal_image)
        else:
            raise ValueError("Either goal_kf_id or goal_image or goal_text must be set")
        
        if plan_to_goal:
            self.plan_to_goal()

    def get_next_waypoint(self, next_n_waypoints: int = 5):
        """
        Get the next waypoint.
        First get current kf, then compute in which segment in the plan it is,
        then return the next n_waypoints waypoints in the plan.
        Also remember which node has been visited, so we don't visit it again.

        Args:
            next_n_waypoints: Number of waypoints to return (default: 5)

        Returns:
            dict: {
                "waypoints": List[int] - List of next keyframe IDs (up to next_n_waypoints),
                "current_position": int - Current position index in the plan,
                "remaining_waypoints": int - Number of waypoints remaining to goal,
                "reached_goal": bool - True if we've reached the goal
            }
            Returns None if no plan exists
        """
        if self.current_plan is None or self.current_plan["detailed_path"] is None:
            return None

        # Get current keyframe
        state = self.system.get_current_kf()
        if state is None or state.get("closest_perm_node") is None:
            return None

        current_kf = state["closest_perm_node"]
        current_kf_id = current_kf.id

        # Mark current position as visited
        self.visited_waypoints.add(current_kf_id)

        # Use detailed path for navigation
        path = self.current_plan["detailed_path"]

        # Find current position in the path
        try:
            current_idx = path.index(current_kf_id)
        except ValueError:
            # Current keyframe not in path, find closest unvisited waypoint
            # This can happen if we've deviated from the path
            min_dist = float('inf')
            closest_idx = 0

            for i, waypoint_id in enumerate(path):
                if waypoint_id not in self.visited_waypoints:
                    # Compute distance to this waypoint
                    dist = self._compute_heuristic(current_kf_id, waypoint_id)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i

            current_idx = closest_idx - 1  # Start from one before the closest

        # Check if we've reached the goal
        if current_kf_id == self.goal_kf_id:
            return {
                "waypoints": [],
                "current_position": len(path) - 1,
                "remaining_waypoints": 0,
                "reached_goal": True,
                "waypoint_poses": []
            }

        # Get next unvisited waypoints
        waypoints = []
        waypoint_poses = []
        for i in range(current_idx + 1, len(path)):
            waypoint_id = path[i]
            if waypoint_id not in self.visited_waypoints or waypoint_id == self.goal_kf_id:
                waypoints.append(waypoint_id)
                waypoint_poses.append(self.hypothesis_manager.nodes[waypoint_id].pose_mu[0])
                if len(waypoints) >= next_n_waypoints:
                    break
        # transform to local frame
        # current_pose = self.hypothesis_manager.nodes[current_kf_id].pose_mu[0].unsqueeze(0)
        current_pose = self.hypothesis_manager.dist[0][0].unsqueeze(0)
        waypoint_poses_tensor = torch.stack(waypoint_poses)
        waypoint_poses_local = current_pose.Inv() @ waypoint_poses_tensor
        
        # Count remaining waypoints (including those beyond next_n_waypoints)
        remaining_count = 0
        for i in range(current_idx + 1, len(path)):
            if path[i] not in self.visited_waypoints or path[i] == self.goal_kf_id:
                remaining_count += 1

        return {
            "waypoints": waypoints,
            "current_position": current_idx,
            "remaining_waypoints": remaining_count,
            "reached_goal": len(waypoints) == 0 or (len(waypoints) == 1 and waypoints[0] == self.goal_kf_id),
            "waypoint_poses": waypoint_poses_local.cpu().numpy()
        }

    def _set_goal_by_image(self, goal_image: np.ndarray):
        """
        Set the goal by image using visual place recognition.

        Uses the keyframe database to find the best matching keyframe
        for the given goal image.

        Args:
            goal_image: RGB image to use as goal (numpy array, shape H x W x 3)
        """
        # Query the database for the best matching keyframe
        db = self.system.keyframe_db
        results = db.query(goal_image)

        # Check if any matches were found
        if not results["keyframes"]:
            raise ValueError("No matching keyframes found for the goal image")

        # Get the best match (first result, highest score)
        best_keyframe = results["keyframes"][0]
        best_score = results["scores"][0]

        # Set the goal to the best matching keyframe
        self.goal_kf_id = best_keyframe.id

        logger.info(f"Goal set to keyframe {self.goal_kf_id} (VPR score: {best_score:.3f})")

    def _set_goal_by_text(self, goal_text: str):
        """
        Set the goal by text.
        """
        pass

    def _get_edge_cost(self, from_kf_id: int, to_kf_id: int) -> float:
        """
        Get the cost of an edge between two keyframes.

        Checks in order:
        1. Sparse graph edges (backbone + shortcuts) - preferred for long-range
        2. Odometry edges (only if use_temporary_keyframes=True)

        Args:
            from_kf_id: Source keyframe ID
            to_kf_id: Target keyframe ID

        Returns:
            Edge cost (translation norm), or infinity if no edge exists
        """
        # Check for sparse graph edge (backbone or shortcut)
        sparse_cost = self.sparse_graph.get_edge_cost(from_kf_id, to_kf_id)
        if sparse_cost != float('inf'):
            return sparse_cost

        # Check for proximity edge (undirected)
        prox_key = tuple(sorted((from_kf_id, to_kf_id)))
        prox_edge = self.hypothesis_manager.proximity_edges.get(prox_key)
        if prox_edge is not None:
            return prox_edge.cost

        # Check for odometry edge only if temporary keyframes are enabled
        if self.use_temporary_keyframes:
            # Check for odometry edge (directed, includes temporary keyframes)
            odom_key_fwd = (from_kf_id, to_kf_id)
            odom_key_bwd = (to_kf_id, from_kf_id)

            if odom_key_fwd in self.hypothesis_manager.odom_edges:
                edge = self.hypothesis_manager.odom_edges[odom_key_fwd]
                return edge.cost
            elif odom_key_bwd in self.hypothesis_manager.odom_edges:
                edge = self.hypothesis_manager.odom_edges[odom_key_bwd]
                return edge.cost

        return float('inf')

    def _get_neighbors(self, kf_id: int) -> List[int]:
        """
        Get all neighbors of a keyframe in the planning graph.

        Includes:
        - Sparse graph neighbors (backbone + shortcuts between permanent keyframes)
        - Odometry neighbors (only if use_temporary_keyframes=True)

        Args:
            kf_id: Keyframe ID

        Returns:
            List of neighbor keyframe IDs
        """
        neighbors = set()

        # Add sparse graph neighbors (backbone + shortcuts)
        sparse_neighbors = self.sparse_graph.get_sparse_graph_neighbors(kf_id)
        neighbors.update(sparse_neighbors)

        # Add proximity neighbors (undirected)
        neighbors.update(self.hypothesis_manager.proximity_adjacency.get(kf_id, set()))

        # Add odometry neighbors only if temporary keyframes are enabled
        if self.use_temporary_keyframes:
            # Add odometry neighbors (both directions, includes temporary keyframes)
            for (id1, id2) in self.hypothesis_manager.odom_edges.keys():
                if id1 == kf_id:
                    neighbors.add(id2)
                elif id2 == kf_id:
                    neighbors.add(id1)

        return list(neighbors)

    def _dijkstra_search(self, current_kf_id: int, goal_kf_id: int):
        """
        Search the shortest path in the graph using Dijkstra's algorithm.

        The graph includes:
        - Odometry edges (between consecutive keyframes)
        - Chain shortcut edges (long-range connections)

        Args:
            current_kf_id: Starting keyframe ID
            goal_kf_id: Goal keyframe ID

        Returns:
            List[int]: Sequence of keyframe IDs from current to goal, or None if no path
        """
        import heapq

        if current_kf_id == goal_kf_id:
            return [current_kf_id]

        # Dijkstra's algorithm
        # Priority queue: (cost, kf_id)
        pq = [(0.0, current_kf_id)]
        # Best costs found so far
        costs = {current_kf_id: 0.0}
        # Parent pointers for path reconstruction
        parents = {current_kf_id: None}
        # Visited set
        visited = set()

        while pq:
            current_cost, current_id = heapq.heappop(pq)

            # Skip if already visited
            if current_id in visited:
                continue

            visited.add(current_id)

            # Found goal
            if current_id == goal_kf_id:
                break

            # Explore neighbors
            neighbors = self._get_neighbors(current_id)

            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue

                # Get edge cost
                edge_cost = self._get_edge_cost(current_id, neighbor_id)

                if edge_cost == float('inf'):
                    continue

                new_cost = current_cost + edge_cost

                # Update if found better path
                if neighbor_id not in costs or new_cost < costs[neighbor_id]:
                    costs[neighbor_id] = new_cost
                    parents[neighbor_id] = current_id
                    heapq.heappush(pq, (new_cost, neighbor_id))

        # Check if goal was reached
        if goal_kf_id not in parents:
            return None

        # Reconstruct path
        path = []
        current = goal_kf_id
        while current is not None:
            path.append(current)
            current = parents[current]

        path.reverse()
        return path

    def _compute_heuristic(self, from_kf_id: int, to_kf_id: int) -> float:
        """
        Compute the straight-line distance heuristic between two keyframes.
        Uses the Euclidean distance between pose positions (translation part).

        Args:
            from_kf_id: Source keyframe ID
            to_kf_id: Target keyframe ID

        Returns:
            Straight-line distance between keyframe poses
        """
        # Get keyframes
        from_kf = self.hypothesis_manager.nodes.get(from_kf_id)
        to_kf = self.hypothesis_manager.nodes.get(to_kf_id)

        if from_kf is None or to_kf is None:
            return 0.0

        # Extract translation from pose_mu (first component of GMM)
        # pose_mu is (K, 7) where K is number of components
        # We use the first component (index 0) as the best estimate
        from_pos = from_kf.pose_mu[0].translation()  # (3,) tensor
        to_pos = to_kf.pose_mu[0].translation()  # (3,) tensor

        # Compute Euclidean distance
        distance = torch.norm(to_pos - from_pos).item()
        return distance

    def _astar_search(self, current_kf_id: int, goal_kf_id: int):
        """
        Search the shortest path in the graph using A* algorithm.
        Uses straight-line distance to goal as heuristic.

        The graph includes:
        - Odometry edges (between consecutive keyframes)
        - Backbone edges (between consecutive permanent keyframes)
        - Chain shortcut edges (long-range connections)

        Args:
            current_kf_id: Starting keyframe ID
            goal_kf_id: Goal keyframe ID

        Returns:
            List[int]: Sequence of keyframe IDs from current to goal, or None if no path
        """
        import heapq

        if current_kf_id == goal_kf_id:
            return [current_kf_id]

        # A* algorithm
        # Priority queue: (f_score, g_score, kf_id)
        # f_score = g_score + heuristic (for priority)
        # g_score = actual cost from start (for comparison)
        initial_h = self._compute_heuristic(current_kf_id, goal_kf_id)
        pq = [(initial_h, 0.0, current_kf_id)]

        # Best g_scores (actual costs) found so far
        g_scores = {current_kf_id: 0.0}
        # Parent pointers for path reconstruction
        parents = {current_kf_id: None}
        # Visited set
        visited = set()

        while pq:
            f_score, g_score, current_id = heapq.heappop(pq)

            # Skip if already visited
            if current_id in visited:
                continue

            visited.add(current_id)

            # Found goal
            if current_id == goal_kf_id:
                break

            # Explore neighbors
            neighbors = self._get_neighbors(current_id)

            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue

                # Get edge cost
                edge_cost = self._get_edge_cost(current_id, neighbor_id)

                if edge_cost == float('inf'):
                    continue

                # Compute new g_score (actual cost from start)
                new_g_score = g_score + edge_cost

                # Update if found better path
                if neighbor_id not in g_scores or new_g_score < g_scores[neighbor_id]:
                    g_scores[neighbor_id] = new_g_score
                    parents[neighbor_id] = current_id

                    # Compute f_score = g_score + heuristic
                    h_score = self._compute_heuristic(neighbor_id, goal_kf_id)
                    f_score = new_g_score + h_score

                    heapq.heappush(pq, (f_score, new_g_score, neighbor_id))

        # Check if goal was reached
        if goal_kf_id not in parents:
            return None

        # Reconstruct path
        path = []
        current = goal_kf_id
        while current is not None:
            path.append(current)
            current = parents[current]

        path.reverse()
        return path

    def _expand_path_segment(
        self,
        from_kf_id: int,
        to_kf_id: int,
        detail_level: Literal["odometry", "backbone"]
    ) -> List[int]:
        """
        Expand a path segment between two keyframes using odometry or backbone edges.

        Args:
            from_kf_id: Starting keyframe ID
            to_kf_id: Ending keyframe ID
            detail_level: "odometry" for full detail (includes temporary keyframes),
                         "backbone" for permanent keyframes only

        Returns:
            List[int]: Sequence of keyframe IDs from from_kf_id to to_kf_id (inclusive)
        """
        if from_kf_id == to_kf_id:
            return [from_kf_id]

        # For odometry detail level, follow the odometry chain (both directions)
        if detail_level == "odometry":
            # Build lightweight adjacency for odometry edges (undirected)
            # Note: odometry edges are stored as directed (id1 -> id2) but represent
            # a bidirectional traversable chain for navigation purposes.
            adjacency = {}
            for (id1, id2) in self.hypothesis_manager.odom_edges.keys():
                adjacency.setdefault(id1, set()).add(id2)
                adjacency.setdefault(id2, set()).add(id1)

            # Dijkstra over odometry-only edges to support both forward and backward traversal
            import heapq
            pq = [(0.0, from_kf_id)]  # (cost, node)
            costs = {from_kf_id: 0.0}
            parents = {from_kf_id: None}

            def odom_edge_cost(a: int, b: int) -> float:
                # Prefer stored forward edge cost; fallback to reverse
                e = self.hypothesis_manager.odom_edges.get((a, b))
                if e is not None:
                    return e.cost
                e = self.hypothesis_manager.odom_edges.get((b, a))
                if e is not None:
                    return e.cost
                return float('inf')

            visited = set()
            while pq:
                g_cost, nid = heapq.heappop(pq)
                if nid in visited:
                    continue
                visited.add(nid)
                if nid == to_kf_id:
                    break
                for nb in adjacency.get(nid, []):
                    if nb in visited:
                        continue
                    ec = odom_edge_cost(nid, nb)
                    if ec == float('inf'):
                        continue
                    ng = g_cost + ec
                    if nb not in costs or ng < costs[nb]:
                        costs[nb] = ng
                        parents[nb] = nid
                        heapq.heappush(pq, (ng, nb))

            if to_kf_id not in parents:
                # Disconnected in odometry graph — fall back to direct pair
                return [from_kf_id, to_kf_id]

            # Reconstruct path
            path = []
            cur = to_kf_id
            while cur is not None:
                path.append(cur)
                cur = parents[cur]
            path.reverse()
            return path

        # For backbone detail level, use permanent keyframes only
        else:  # detail_level == "backbone"
            # Get permanent keyframe IDs in order
            perm_kf_ids = self.sparse_graph.permanent_kf_ids

            if from_kf_id not in perm_kf_ids or to_kf_id not in perm_kf_ids:
                # One or both are not permanent, return direct edge
                return [from_kf_id, to_kf_id]

            from_idx = perm_kf_ids.index(from_kf_id)
            to_idx = perm_kf_ids.index(to_kf_id)

            if from_idx < to_idx:
                # Forward direction
                return perm_kf_ids[from_idx:to_idx + 1]
            elif from_idx > to_idx:
                # Backward direction
                return perm_kf_ids[to_idx:from_idx + 1][::-1]
            else:
                return [from_kf_id]

    def _compute_path_cost(self, path: List[int]) -> float:
        """
        Compute the total cost of a path by summing edge costs.

        Args:
            path: Sequence of keyframe IDs

        Returns:
            Total path cost (sum of translation norms)
        """
        if len(path) <= 1:
            return 0.0

        total_cost = 0.0
        for i in range(len(path) - 1):
            from_id = path[i]
            to_id = path[i + 1]
            cost = self._get_edge_cost(from_id, to_id)
            if cost == float('inf'):
                # Invalid segment — treat whole path as invalid
                return float('inf')
            total_cost += cost

        return total_cost

    def _remove_backtracking(self, path: List[int]) -> List[int]:
        """
        Remove backtracking from a path by detecting when we revisit earlier nodes.

        When the path goes [a, b, c, d, c, e], we can shortcut to [a, b, c, e]
        since we visit 'c' twice.

        Args:
            path: Sequence of keyframe IDs possibly containing backtracking

        Returns:
            Path with backtracking removed
        """
        if len(path) <= 2:
            return path

        # Keep track of the last occurrence of each node
        optimized_path = []
        node_positions = {}  # Maps node_id -> index in optimized_path

        for node_id in path:
            if node_id in node_positions:
                # We've seen this node before - this is backtracking
                # Remove everything after the first occurrence
                last_pos = node_positions[node_id]
                optimized_path = optimized_path[:last_pos + 1]

                # Clear position tracking for removed nodes
                nodes_to_remove = [nid for nid, pos in node_positions.items() if pos > last_pos]
                for nid in nodes_to_remove:
                    del node_positions[nid]
            else:
                # New node, add it to the path
                node_positions[node_id] = len(optimized_path)
                optimized_path.append(node_id)

        return optimized_path

    def plan_to_goal(
        self,
        algorithm: Literal["dijkstra", "astar"] = "astar",
        sparse_only: bool = False,
    ):
        """
        Plan the trajectory to the target location.

        Args:
            algorithm: Planning algorithm to use - "dijkstra" or "astar" (default)
                      - "dijkstra": Classic Dijkstra's algorithm (no heuristic)
                      - "astar": A* with straight-line distance heuristic (prefers shortcuts)

        Returns:
            dict: {
                "sparse_path": List[int] or None - Sparse path using shortcuts
                "detailed_path": List[int] or None - Detailed path for navigation
                "sparse_cost": float - Cost of sparse path
                "detailed_cost": float - Cost of detailed path
                "algorithm": str - Algorithm used
                "detail_level": str - Detail level used
            }
        """
        if sparse_only:
            detail_level = "sparse"
        else:
            detail_level = "odometry" if self.use_temporary_keyframes else "backbone"

        assert self.goal_kf_id is not None, "Goal is not set"
        state = self.system.get_current_kf()
        assert state is not None and state.get("closest_perm_node") is not None, "Closest node is not set"
        current_kf = state["closest_perm_node"]

        # Select algorithm to get sparse path
        if algorithm == "dijkstra":
            sparse_path = self._dijkstra_search(current_kf.id, self.goal_kf_id)
        elif algorithm == "astar":
            sparse_path = self._astar_search(current_kf.id, self.goal_kf_id)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'dijkstra' or 'astar'")

        if sparse_path is None:
            logger.warning(f"[{algorithm.upper()}] No path found from keyframe {current_kf.id} to goal {self.goal_kf_id}")
            return {
                "sparse_path": None,
                "detailed_path": None,
                "sparse_cost": float('inf'),
                "detailed_cost": float('inf'),
                "algorithm": algorithm,
                "detail_level": detail_level,
            }

        # Compute sparse path cost
        sparse_cost = self._compute_path_cost(sparse_path)

        # Store the trajectory (for backward compatibility)
        self.trajectory = sparse_path

        # Generate detailed path if requested
        detailed_path = None
        detailed_cost = sparse_cost

        if detail_level == "sparse":
            detailed_path = sparse_path
            detailed_cost = sparse_cost
        elif detail_level in ["odometry", "backbone"]:
            # Expand each segment in the sparse path
            detailed_path = []
            for i in range(len(sparse_path) - 1):
                from_id = sparse_path[i]
                to_id = sparse_path[i + 1]
                segment = self._expand_path_segment(from_id, to_id, detail_level)

                # Add segment, avoiding duplicates at boundaries
                if i == 0:
                    detailed_path.extend(segment)
                else:
                    detailed_path.extend(segment[1:])  # Skip first node (already added)

            # Remove backtracking from the detailed path
            # This handles cases where A* shortcuts cause backward movement
            detailed_path_before = len(detailed_path)
            detailed_path = self._remove_backtracking(detailed_path)

            # Compute detailed path cost
            detailed_cost = self._compute_path_cost(detailed_path)

            if detailed_path_before != len(detailed_path):
                logger.debug(f"[{algorithm.upper()}] Removed backtracking: {detailed_path_before} -> {len(detailed_path)} keyframes")
        else:
            raise ValueError(f"Unknown detail_level: {detail_level}. Use 'sparse', 'odometry', or 'backbone'")

        logger.info(f"[{algorithm.upper()}] Sparse path: {len(sparse_path)} keyframes, cost: {sparse_cost:.2f}")
        if detail_level != "sparse":
            logger.info(f"[{algorithm.upper()}] Detailed path ({detail_level}): {len(detailed_path)} keyframes, cost: {detailed_cost:.2f}")

        # Reset visited waypoints for new plan
        self.visited_waypoints = set()

        self.current_plan = {
            "sparse_path": sparse_path,
            "detailed_path": detailed_path,
            "sparse_cost": sparse_cost,
            "detailed_cost": detailed_cost,
            "algorithm": algorithm,
            "detail_level": detail_level,
        }
        return self.current_plan
