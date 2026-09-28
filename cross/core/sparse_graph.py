"""
Sparse Graph Management with Binary Lifting for Chain Shortcuts

This module provides hierarchical shortcut edge management for efficient path planning
in SLAM systems. It uses binary lifting to create O(log N) shortcuts between permanent
keyframes while preserving optimal path costs.

Key Features:
- Binary lifting jump tables for O(log L) range queries
- Phase-locking to maintain sparse graph structure
- Automatic backbone updates when new keyframes are added
- Loop closure support with incremental rebuilding
"""

import torch
import pypose as pp
from typing import Dict, List, Tuple, Set, Optional
from loguru import logger
from cross.core.types import Edge, EdgeType
from cross.utils.profile import timeit


class SparseGraph:
    """
    Manages chain shortcut edges between permanent keyframes using binary lifting.

    The graph maintains:
    1. Backbone: Sequential segments between consecutive permanent keyframes
    2. Jump tables: Binary lifting structure for O(log N) range composition
    3. Shortcuts: Sparse edges at phase-aligned indices for planning

    Complexity:
    - Space: O(N log N) for N permanent keyframes
    - Update: O(ΔN log N) per batch of ΔN new keyframes
    - Query: O(log L) for distance L
    """

    def __init__(
        self,
        system,
        max_stride_power: int = 6,
        min_perm_for_shortcuts: int = 4,
        update_frequency: int = 5,
        device: str = "cuda",
    ):
        """
        Initialize the sparse graph manager.

        Args:
            system: Reference to System for accessing hypothesis_manager and nodes/edges
            max_stride_power: Maximum stride as power of 2 (default: 6 → 64 keyframes)
            min_perm_for_shortcuts: Minimum permanent keyframes before creating shortcuts
            update_frequency: Create shortcuts every N new permanent keyframes
            device: Device for tensor operations
        """
        self.system = system
        self.hypothesis_manager = system.hypothesis_manager
        self.device = device

        # Configuration
        self.max_stride_power = max_stride_power
        self.min_perm_for_shortcuts = min_perm_for_shortcuts
        self.update_frequency = update_frequency

        # Permanent keyframe backbone
        self.permanent_kf_ids: List[int] = []

        # Backbone segments between consecutive permanents: i → i+1
        # These are stored internally for building jump tables
        self.backbone_T: List[pp.LieTensor] = []  # SE3 transforms
        self.backbone_S: List[pp.LieTensor] = []  # se3 diagonal stds
        self.backbone_cost: List[float] = []      # Edge costs

        # Binary lifting jump tables (indexed by perm_idx, power-of-2 level)
        # nxt[i][p] = index after jumping 2^p steps from i (or -1 if out of bounds)
        self.jump_nxt: List[List[int]] = []
        self.jump_T: List[List[pp.LieTensor]] = []
        self.jump_S: List[List[pp.LieTensor]] = []
        self.jump_cost: List[List[float]] = []

        # Phase-locking state: last processed index for each stride
        self.last_processed_idx: Dict[int, int] = {}

        # Backbone edges (direct edges between consecutive permanent keyframes)
        # Keyed by (min(kf_id_a, kf_id_b), max(kf_id_a, kf_id_b)) for undirected graph
        self.backbone_edges: Dict[Tuple[int, int], Edge] = {}

        # Shortcut edges (long-range edges at phase-aligned indices)
        # Keyed by (min(kf_id_a, kf_id_b), max(kf_id_a, kf_id_b)) for undirected graph
        self.shortcut_edges: Dict[Tuple[int, int], Edge] = {}

        # Planning adjacency includes both backbone and shortcut neighbors (undirected)
        self.planning_adjacency: Dict[int, Set[int]] = {}

    def update_on_new_permanent_keyframe(self):
        """
        Called when a new permanent keyframe is added to the system.
        Updates backbone and periodically triggers shortcut creation.

        Args:
            keyframe_id: ID of the newly added permanent keyframe
        """
        self._update_permanent_backbone()

        # Periodically update shortcuts
        if (len(self.permanent_kf_ids) % self.update_frequency == 0 and
            len(self.permanent_kf_ids) >= self.min_perm_for_shortcuts):
            self.create_shortcut_edges()

    def _update_permanent_backbone(self):
        """
        Update the permanent keyframe backbone with new permanent keyframes.
        Builds backbone segments and extends jump tables.
        """
        # Get all permanent keyframes sorted by ID
        all_perm_kfs = sorted(
            [kf for kf in self.hypothesis_manager.nodes.values() if not kf.temporary],
            key=lambda k: k.id
        )

        if len(all_perm_kfs) < 2:
            return

        # Update permanent_kf_ids list
        old_count = len(self.permanent_kf_ids)
        self.permanent_kf_ids = [kf.id for kf in all_perm_kfs]
        new_count = len(self.permanent_kf_ids)

        if new_count == old_count:
            return  # No new permanent keyframes

        # Build backbone segments for newly added permanents
        for i in range(max(0, old_count - 1), new_count - 1):
            kf_a_id = self.permanent_kf_ids[i]
            kf_b_id = self.permanent_kf_ids[i + 1]

            # Compose odometry chain from kf_a to kf_b
            result = self._compose_odom_chain(kf_a_id, kf_b_id)

            # If path is incomplete, stop here (subsequent paths likely incomplete too)
            if result is None:
                logger.debug(f"Stopping backbone update at index {i}: path from {kf_a_id} to {kf_b_id} not ready")
                # Truncate permanent_kf_ids to only include keyframes with complete paths
                self.permanent_kf_ids = self.permanent_kf_ids[:i+1]
                break

            T_composed, S_composed, cost_sum = result

            # Ensure we have space in backbone lists
            while len(self.backbone_T) <= i:
                self.backbone_T.append(None)
                self.backbone_S.append(None)
                self.backbone_cost.append(0.0)

            self.backbone_T[i] = T_composed
            self.backbone_S[i] = S_composed
            self.backbone_cost[i] = cost_sum

            # Create backbone edge object for planner access
            edge_key = (min(kf_a_id, kf_b_id), max(kf_a_id, kf_b_id))
            backbone_edge = Edge(
                mean=T_composed,
                std=S_composed,
                type=EdgeType.BACKBONE,
                cost=cost_sum,  # Pre-computed cost
            )
            self.backbone_edges[edge_key] = backbone_edge

            # Update planning adjacency for backbone edges
            self.planning_adjacency.setdefault(kf_a_id, set()).add(kf_b_id)
            self.planning_adjacency.setdefault(kf_b_id, set()).add(kf_a_id)

        # Extend jump tables for new permanent keyframes
        self._extend_jump_tables()

    def _compose_odom_chain(
        self,
        from_kf_id: int,
        to_kf_id: int
    ) -> Optional[Tuple[pp.LieTensor, pp.LieTensor, float]]:
        """
        Compose odometry edges from from_kf_id to to_kf_id.
        Follows the odometry chain through temporary keyframes if necessary.

        Args:
            from_kf_id: Starting keyframe ID
            to_kf_id: Ending keyframe ID

        Returns:
            Tuple of (T_composed, S_composed, cost_sum) if path exists, None otherwise
            T_composed: SE3 transform from from_kf to to_kf
            S_composed: se3 diagonal std (composed variances)
            cost_sum: Sum of edge costs (translation norms)
        """
        # Find the path through odometry edges (following sequential chain)
        current_id = from_kf_id
        T_composed = pp.identity_SE3(1, device=self.device).squeeze(0)
        S_squared_sum = torch.zeros(6, device=self.device)
        cost_sum = 0.0

        visited = set()
        max_iterations = len(self.hypothesis_manager.nodes) + 1  # Safety limit

        while current_id != to_kf_id:
            if current_id in visited or len(visited) >= max_iterations:
                logger.debug(f"No valid odometry path from {from_kf_id} to {to_kf_id} (cycle or too long)")
                return None

            visited.add(current_id)

            # Find next odometry edge (forward direction: id1 -> id2)
            next_id = None
            next_edge = None

            for (id1, id2), edge in self.hypothesis_manager.odom_edges.items():
                if id1 == current_id:
                    next_id = id2
                    next_edge = edge
                    break

            if next_id is None:
                # Path not yet created - this is normal during incremental updates
                logger.debug(f"Odometry path incomplete from {from_kf_id} to {to_kf_id} (stopped at {current_id})")
                return None

            # Compose transform
            T_composed = T_composed @ next_edge.mean
            # Add variances in tangent space
            S_squared_sum += next_edge.std.tensor() ** 2
            # Sum costs (translation norm)
            cost_sum += torch.norm(next_edge.mean.tensor()[:3]).item()

            current_id = next_id

        S_composed = pp.se3(S_squared_sum ** 0.5)
        return T_composed, S_composed, cost_sum

    def _extend_jump_tables(self):
        """
        Extend binary lifting jump tables to cover all permanent keyframes.

        For each permanent keyframe index i and level p:
        - jump_nxt[i][p] = i + 2^p (or -1 if out of bounds)
        - jump_T/S/cost[i][p] = composition of two level-(p-1) jumps
        """
        N = len(self.permanent_kf_ids)
        if N < 2:
            return

        # Ensure jump_nxt has enough rows
        while len(self.jump_nxt) < N:
            self.jump_nxt.append([])
            self.jump_T.append([])
            self.jump_S.append([])
            self.jump_cost.append([])

        # For each permanent keyframe index
        for i in range(N):
            # Level 0: single hop (i → i+1)
            if len(self.jump_nxt[i]) == 0 and i < N - 1:
                self.jump_nxt[i].append(i + 1)
                self.jump_T[i].append(self.backbone_T[i])
                self.jump_S[i].append(self.backbone_S[i])
                self.jump_cost[i].append(self.backbone_cost[i])

            # Higher levels: double the jump distance
            p = 1
            while p <= self.max_stride_power:
                if len(self.jump_nxt[i]) <= p:
                    # Need to compute level p
                    if len(self.jump_nxt[i]) < p:
                        # Fill gaps with -1 (out of bounds)
                        while len(self.jump_nxt[i]) < p:
                            self.jump_nxt[i].append(-1)
                            self.jump_T[i].append(pp.identity_SE3(1, device=self.device).squeeze(0))
                            self.jump_S[i].append(pp.identity_se3(1, device=self.device).squeeze(0))
                            self.jump_cost[i].append(0.0)

                    # Compute level p by composing two level p-1 jumps
                    prev_level = p - 1
                    if (prev_level >= len(self.jump_nxt[i]) or
                        self.jump_nxt[i][prev_level] == -1):
                        self.jump_nxt[i].append(-1)
                        self.jump_T[i].append(pp.identity_SE3(1, device=self.device).squeeze(0))
                        self.jump_S[i].append(pp.identity_se3(1, device=self.device).squeeze(0))
                        self.jump_cost[i].append(0.0)
                    else:
                        mid = self.jump_nxt[i][prev_level]
                        if (mid == -1 or mid >= len(self.jump_nxt) or
                            prev_level >= len(self.jump_nxt[mid]) or
                            self.jump_nxt[mid][prev_level] == -1):
                            self.jump_nxt[i].append(-1)
                            self.jump_T[i].append(pp.identity_SE3(1, device=self.device).squeeze(0))
                            self.jump_S[i].append(pp.identity_se3(1, device=self.device).squeeze(0))
                            self.jump_cost[i].append(0.0)
                        else:
                            # Compose two half-jumps
                            nxt = self.jump_nxt[mid][prev_level]
                            T = self.jump_T[i][prev_level] @ self.jump_T[mid][prev_level]
                            S_squared = (self.jump_S[i][prev_level].tensor() ** 2 +
                                       self.jump_S[mid][prev_level].tensor() ** 2)
                            S = pp.se3(S_squared ** 0.5)
                            cost = self.jump_cost[i][prev_level] + self.jump_cost[mid][prev_level]

                            self.jump_nxt[i].append(nxt)
                            self.jump_T[i].append(T)
                            self.jump_S[i].append(S)
                            self.jump_cost[i].append(cost)
                p += 1

    def query_range(
        self,
        head_idx: int,
        tail_idx: int
    ) -> Tuple[pp.LieTensor, pp.LieTensor, float]:
        """
        Query composed transform, std, and cost for range [head_idx, tail_idx].
        Uses binary lifting for O(log distance) composition.

        Args:
            head_idx: Starting permanent keyframe index
            tail_idx: Ending permanent keyframe index

        Returns:
            T_range: SE3 transform from head to tail
            S_range: se3 diagonal std (composed)
            cost_range: Sum of costs along the path
        """
        if head_idx >= tail_idx or head_idx < 0 or tail_idx >= len(self.permanent_kf_ids):
            return (pp.identity_SE3(1, device=self.device).squeeze(0),
                    pp.identity_se3(1, device=self.device).squeeze(0),
                    0.0)

        current_idx = head_idx
        T_range = pp.identity_SE3(1, device=self.device).squeeze(0)
        S_squared_sum = torch.zeros(6, device=self.device)
        cost_range = 0.0

        # Binary decomposition of distance
        p = self.max_stride_power
        while p >= 0 and current_idx < tail_idx:
            jump_size = 2 ** p
            if current_idx + jump_size <= tail_idx:
                # Take this jump
                if (p < len(self.jump_nxt[current_idx]) and
                    self.jump_nxt[current_idx][p] != -1):
                    T_range = T_range @ self.jump_T[current_idx][p]
                    S_squared_sum += self.jump_S[current_idx][p].tensor() ** 2
                    cost_range += self.jump_cost[current_idx][p]
                    current_idx = self.jump_nxt[current_idx][p]
                else:
                    p -= 1
            else:
                p -= 1

        S_range = pp.se3(S_squared_sum ** 0.5)
        return T_range, S_range, cost_range

    @timeit
    def create_shortcut_edges(self):
        """
        Create chain shortcut edges between permanent keyframes.

        Uses phase-locking: for stride S = 2^p, only create shortcuts at indices
        where idx % S == 0. This ensures:
        - Sparse graph: O(N log N) total shortcuts
        - Optimal paths preserved: costs match odometry sums
        - Efficient updates: process only new phase-aligned indices
        """
        N = len(self.permanent_kf_ids)

        # Early out if not enough permanent keyframes
        if N < self.min_perm_for_shortcuts:
            return

        # Ensure jump tables are up to date
        self._extend_jump_tables()

        shortcuts_added = 0

        # For each stride S = 2^p
        for p in range(1, self.max_stride_power + 1):
            S = 2 ** p

            # Get last processed index for this stride
            start_idx = self.last_processed_idx.get(S, S - 1) + 1

            # Advance to phase-aligned index: idx % S == 0
            idx = ((start_idx + S - 1) // S) * S

            # Loop through phase-aligned indices
            while idx < N:
                head_idx = idx - S
                if head_idx < 0:
                    idx += S
                    continue

                # Query range [head_idx, idx] via jump table
                T_range, S_range, cost_range = self.query_range(head_idx, idx)

                # Get actual keyframe IDs
                u = self.permanent_kf_ids[head_idx]
                v = self.permanent_kf_ids[idx]

                # Create shortcut edge (undirected, use sorted key)
                edge_key = (min(u, v), max(u, v))

                # Check if edge exists and if new cost is better
                existing_edge = self.shortcut_edges.get(edge_key)
                if existing_edge is None or cost_range < existing_edge.cost:
                    # Create or update shortcut edge
                    shortcut_edge = Edge(
                        mean=T_range,
                        std=S_range,
                        type=EdgeType.CHAIN,
                        cost=cost_range,  # Pre-computed cost
                    )
                    self.shortcut_edges[edge_key] = shortcut_edge

                    # Update planning adjacency (undirected)
                    self.planning_adjacency.setdefault(u, set()).add(v)
                    self.planning_adjacency.setdefault(v, set()).add(u)

                    shortcuts_added += 1
                    logger.debug(f"Added chain shortcut {u} <-> {v} (stride={S}, cost={cost_range:.2f})")

                idx += S

            # Update last processed index for this stride
            self.last_processed_idx[S] = N - 1

        if shortcuts_added > 0:
            logger.info(f"Created {shortcuts_added} chain shortcut edges")

    def rebuild_from_index(self, start_idx: int):
        """
        Rebuild backbone and jump tables starting from a given index.
        Used after loop closure when poses have been updated.

        Args:
            start_idx: Index in permanent_kf_ids to start rebuilding from
        """
        N = len(self.permanent_kf_ids)

        if start_idx < 0 or start_idx >= N - 1:
            return

        # Rebuild backbone segments from start_idx onwards
        for i in range(start_idx, N - 1):
            kf_a_id = self.permanent_kf_ids[i]
            kf_b_id = self.permanent_kf_ids[i + 1]

            # Compose odometry chain from kf_a to kf_b
            result = self._compose_odom_chain(kf_a_id, kf_b_id)

            # Skip if path is incomplete (shouldn't happen during rebuild, but be safe)
            if result is None:
                logger.warning(f"Cannot rebuild backbone at index {i}: path from {kf_a_id} to {kf_b_id} incomplete")
                continue

            T_composed, S_composed, cost_sum = result
            self.backbone_T[i] = T_composed
            self.backbone_S[i] = S_composed
            self.backbone_cost[i] = cost_sum

            # Update backbone edge object
            edge_key = (min(kf_a_id, kf_b_id), max(kf_a_id, kf_b_id))
            backbone_edge = Edge(
                mean=T_composed,
                std=S_composed,
                type=EdgeType.BACKBONE,
                cost=cost_sum,  # Pre-computed cost
            )
            self.backbone_edges[edge_key] = backbone_edge

            # Update planning adjacency for backbone edges
            self.planning_adjacency.setdefault(kf_a_id, set()).add(kf_b_id)
            self.planning_adjacency.setdefault(kf_b_id, set()).add(kf_a_id)

        # Rebuild jump tables from start_idx onwards
        for i in range(start_idx, N):
            # Clear existing jump table entries for this index
            self.jump_nxt[i] = []
            self.jump_T[i] = []
            self.jump_S[i] = []
            self.jump_cost[i] = []

        # Rebuild all jump tables
        self._extend_jump_tables()

        # Reset last_processed_idx for all strides to trigger shortcut rebuild
        for S in list(self.last_processed_idx.keys()):
            if self.last_processed_idx[S] >= start_idx:
                self.last_processed_idx[S] = max(0, start_idx - 1)

        logger.info(f"Rebuilt backbone and jump tables from index {start_idx}")

    def rebuild_after_loop_closure(self, affected_kf_ids: Set[int]):
        """
        Rebuild shortcuts after loop closure based on affected keyframes.

        Args:
            affected_kf_ids: Set of keyframe IDs whose poses were updated
        """
        # Find earliest affected permanent keyframe
        permanent_affected = [
            kf_id for kf_id in self.permanent_kf_ids
            if kf_id in affected_kf_ids
        ]

        if not permanent_affected:
            return

        earliest_affected_id = min(permanent_affected)

        # Find its index in permanent list
        if earliest_affected_id in self.permanent_kf_ids:
            earliest_idx = self.permanent_kf_ids.index(earliest_affected_id)
            logger.info(f"Rebuilding sparse graph from index {earliest_idx} after loop closure")

            # Rebuild backbone and jump tables
            self.rebuild_from_index(earliest_idx)

            # Rebuild shortcuts for affected region
            self.create_shortcut_edges()

    def get_edge_cost(
        self,
        from_kf_id: int,
        to_kf_id: int,
    ) -> float:
        """
        Get the cost of a sparse graph edge (backbone or shortcut) between two keyframes.

        Args:
            from_kf_id: Source keyframe ID
            to_kf_id: Target keyframe ID

        Returns:
            Edge cost (translation norm), or infinity if no edge exists
        """
        edge_key = (min(from_kf_id, to_kf_id), max(from_kf_id, to_kf_id))

        # Check shortcut edges first (may be cheaper than backbone)
        if edge_key in self.shortcut_edges:
            return self.shortcut_edges[edge_key].cost

        # Check backbone edges
        if edge_key in self.backbone_edges:
            return self.backbone_edges[edge_key].cost

        return float('inf')

    def get_sparse_graph_neighbors(
        self,
        kf_id: int,
    ) -> Set[int]:
        """
        Get all sparse graph neighbors (backbone + shortcut) of a keyframe.

        Args:
            kf_id: Keyframe ID

        Returns:
            Set of neighbor keyframe IDs connected by backbone or shortcut edges
        """
        if kf_id in self.planning_adjacency:
            return self.planning_adjacency[kf_id].copy()

        return set()

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the sparse graph.

        Returns:
            Dictionary with graph statistics
        """
        return {
            "permanent_keyframes": len(self.permanent_kf_ids),
            "backbone_edges": len(self.backbone_edges),
            "shortcut_edges": len(self.shortcut_edges),
            "jump_table_rows": len(self.jump_nxt),
            "max_stride": 2 ** self.max_stride_power,
        }
