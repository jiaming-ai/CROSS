import math
from cross.visualization.viz_graph import visualize_pose_graph
from cross.utils.profile import timeit
import pypose as pp
from typing import Tuple, List, Dict, Set, Optional, Any
import torch
from dataclasses import dataclass, field
from loguru import logger
import collections
import threading

from cross.core.types import Keyframe, Edge, VisualEdge, EdgeType
from cross.utils.lie_tensor import project_SE3
from cross.core.pgo import (
    PoseGraph,
)
from cross.core.config import HypothesisConfig
@dataclass
class Hypothesis:
    """
    Represents a single, self-consistent hypothesis (a component in the GMM) of the world,
    containing a graph of keyframes and their relative pose constraints.
    """
    component_id: int
    start_idx: int  # The keyframe ID where this hypothesis was initiated.

    # visual edges can be spurious, so we store in hypothesis
    visual_edges: Dict[Tuple[int, int], List[VisualEdge]] = field(default_factory=dict)
    # adjacency: A simple graph representation for efficient traversal (e.g., BFS/DFS).
    # Using Set for O(1) membership checking and automatic duplicate prevention
    visual_adjacency: Dict[int, Set[int]] = field(default_factory=dict)


    def add_visual_edge(
        self,
        id1: int,
        id2: int,
        factor: VisualEdge,
    ):
        """
        Adds a pre-constructed EdgeFactor to the hypothesis graph.

        Args:
            id1 (int): Source keyframe ID.
            id2 (int): Destination keyframe ID.
            factor (VisualEdgeFactor): The edge factor to add.
        """
        # --- Add Edge to Storage ---
        edge_key = (id1, id2)
        self.visual_edges.setdefault(edge_key, []).append(factor)

        # --- Update Adjacency List for Graph Traversal ---
        # Sets automatically prevent duplicates
        self.visual_adjacency.setdefault(id1, set()).add(id2)
        self.visual_adjacency.setdefault(id2, set()).add(id1)

class HypothesisManager:
    """
    Manages all active hypotheses (GMM components) and the global graph of keyframes.

    Overview
    - Tracks a K-component GMM belief over SE(3): `self.dist = (mu[K], sigma[K], weights[K])`.
      Component 0 is the reference world. Each component index can be Unrealized-Tracking (no branch) or
      Realized-Tracking (has a `Hypothesis` branch). Realization controls edge storage and LC eligibility;
      belief tracking is identical for both.
    - Pose-update gating: `gmm_filtering` accepts an optional boolean mask to skip retrieval-based pose
      updates for selected components (e.g., component 0 when odometry is strong). When skipped, a component’s
      posterior pose/covariance reverts to the true prior (without additional process noise). Newborn components
      are always initialized from proposals regardless of the mask. Weights, evidence, and TTL are still updated
      for all components so loop-closure detection and hypothesis lifecycle continue unaffected.
    - Maintains `nodes` (keyframes), global odometry edges, and per-hypothesis visual edges/adjacency.
      Handles proposal alignment, births, realization, LC/PGO, and cleanup.
    - Per-component metadata: `ttl` (expiry/extension bookkeeping), `last_seen_step` (eviction priority), and
      `newborn` (one-step seeding of prior to first proposal). Component lifecycle governed by
      `realize_*`, `detect_*`, and `ttl_*` thresholds; eviction by staleness when capacity reached.

    Stages
    - Free/Untracked: Slot is available (no branch, no active TTL, weight ≈ 0). Assertions: `realized[i] == False`,
      `ttl[i] == 0`, and `weights[i] ≤ tracking_active_threshold`.
    - Tracking: Components with `weights[i] > tracking_active_threshold and ttl[i] > 0` (belief actively updated). Unrealized and
      realized candidates share the same motion/filtering/TTL logic; the only difference is whether a branch exists.
      - Unrealized (no branch yet): `realized[i] == False`; edges are not stored until realized.
      - Realized: `realized[i] == True`; edges stored; eligible for LC detection/PGO.
    - LC Detected (True Positive): Transient event when evidence passes detection thresholds, immediately triggering
      PGO/merge and returning the component to tracking.

    Observability and comp 0 keep-alive
    - Component 0 represents the current world trajectory and is treated specially to maintain
      observability when retrieval yields no matching proposal:
      - During proposal alignment, if comp 0 is unmatched, keep its pose and apply only a mild
        uncertainty inflation ("no measurement") and assign a moderate keep-alive proposal score.
        This prevents comp 0 from being starved by transient retrieval failures while not dominating
        genuine loop-closure proposals.
      - In filtering, apply a small weight floor to comp 0 before normalization to keep it above the
        active distribution threshold used by downstream consumers. TTL logic still excludes comp 0
        from decay/removal.

    Transitions (with conditions and parameters)
    - Birth: Free → Tracking (Unrealized)
      - Trigger: Unmatched proposal assigned to a slot in `align_proposal_prior`.
      - Params: eviction by oldest `last_seen_step` when all slots occupied; matching floor
        `tracking_floor_weight` for weights (when `ttl > 0`), newborn seeding enabled.
      - Actions: `newborn=True`, `ttl=death_ttl_base`, `last_seen_step=step_counter`.

    - Realize: Tracking (Unrealized) → Tracking (Realized)
      - Trigger: Evidence exceeds realization thresholds (batch check in `add_node()`).
      - Params: `realize_sum_thresh`, `realize_hitrate_thresh` over LLR evidence (`last_sum_pos`, `last_hit_rate`).
      - Actions: `create_hypothesis_branch()` on next `add_node`; initialize realized TTL.

    - Detect LC: Tracking (Realized) → LC Detected
      - Inter-hypothesis LC (k > 0): Overlap-only evidence exceeds detection thresholds in `detect_loop_closure()`.
      - Self-LC for comp 0 (fallback): After alignment, if the aligned cluster confidence for component 0
        exceeds `self_lc_conf_thresh` and at least one matched permanent keyframe mapped to comp 0 satisfies
        both an ID-gap ≥ `self_lc_min_kf_id_gap` relative to the latest KF and a cooldown window
        `step_counter - last_pgo_step ≥ self_lc_cooldown_steps`, trigger PGO within hypothesis 0.
      - Signals (windowed over recent frames):
        - Relative overlap-only history `log_c_hist = log_c - log_c[0]` (no confidence mixed in).
        - Relative confidence history `log_conf_hist = log(conf_k/conf_0)` used as a soft guard with a positive bias margin.
      - Decision (k > 0):
        - Windowed metrics: `sum_overlap = Σ_t ReLU(log_c_rel_t)`, `hit_overlap = mean_t(log_c_rel_t > 0)`.
        - Confidence guard (soft): `conf_hit_rate = mean_t(log_conf_rel_t + margin > 0)`.
        - Trigger if `sum_overlap ≥ detect_overlap_sum_thresh` and `hit_overlap ≥ detect_overlap_hitrate_thresh` and `conf_hit_rate ≥ detect_conf_hitrate_thresh`.
      - Actions: Pose-graph construction and PGO; merge hypotheses; update poses; return to Tracking.

    - Death: Tracking → Free
      - Unified TTL band – if `normed ≥ ttl_norm_thresh` or `hit_rate ≥ ttl_hitrate_thresh`, TTL is extended;
        otherwise it is decremented. When `ttl == 0`, the slot is cleared (μ/σ to identity, weight to 0,
        `newborn=False`; realized branches are removed during this cleanup).

    Matching/alignment and edges
    - `align_proposal_prior` matches proposals to active components (weights > `tracking_active_threshold`), updates
      `last_seen_step`, births new unrealized components, and evicts the stalest unrealized candidate when all slots
      are occupied. Returns `edge_mapping` consumed by `System._add_new_kf`; edges are stored only for realized
      targets (unrealized ones are intentionally skipped). Newborn seeding sets posterior μ/Σ to the first proposal
      once to establish a meaningful prior; thereafter motion + fusion maintain it. In gmm_filtering, newborn weights
      are mixed via a configurable coefficient `newborn_mix_coeff` that allocates a small share of mass to the set of
      brand-new newborns using proposal confidence as distribution (optionally powered by `newborn_conf_power`). After
      TTL cleanup, apply a small post-TTL weight floor to components with `ttl > 0` to keep slots matchable across frames.

    Tuning Parameters (key)
    - Tracking gate: `tracking_active_threshold` (weights above this are considered active/tracking).
    - Realization (tracking → realized): `realize_sum_thresh`, `realize_hitrate_thresh`.
    - Loop-closure detection (tracking → LC): `detect_overlap_sum_thresh`, `detect_overlap_hitrate_thresh`, `detect_conf_hitrate_thresh`.
    - Self-LC detection (comp 0): `self_lc_conf_thresh`, `self_lc_cooldown_steps`, `self_lc_min_kf_id_gap`.
    - TTL band (tracking → free): `death_ttl_base`, `death_ttl_gain`, `ttl_norm_thresh`, `ttl_hitrate_thresh`.
    - Newborn mixing: `newborn_mix_coeff` (share for this-step newborns), `newborn_conf_power` (confidence shaping).
    - Eviction: When all slots occupied, evict stalest unrealized component by smallest `last_seen_step`.
    - Matching floor for unrealized weights: `tracking_floor_weight` (applied post-TTL when `ttl > 0`).

    Persistence
    - `save_state`/`load_state` save/restore graph structure (temporary keyframes, odometry edges, hypothesis 0 only).
    - Tracking state (lifecycle metadata, evidence) is NOT persisted and is reset via `reset_tracking_state()`
      when loading a map, initializing the system, or handling kidnapped events.
    """
    def __init__(self, system, n_components: int, config: Optional[HypothesisConfig] = None):
        cfg = config or HypothesisConfig()

        # ========== Core Data Structures ==========
        self.dist: Tuple[pp.LieTensor, pp.LieTensor, torch.Tensor] = None
        self.nodes: Dict[int, Keyframe] = {}  # The first hypothesis (component 0) is the base reality
        self.odom_edges: Dict[Tuple[int, int], Edge] = {}  # odom edges are always from previous kf to current kf
        self.hypotheses: Dict[int, Hypothesis] = {}
        self.system = system
        self.device = "cuda"
        self.disappearance_counts = collections.defaultdict(int)
        # Graph-lock to guard structural reads/writes across threads (nodes/edges/adjacency)
        # Use re-entrant lock since some operations call other locked methods.
        self.graph_lock = threading.RLock()

        # ========== Component Lifecycle Metadata ==========
        self.n_components = n_components

        # ========== Evidence Tracking (LLR & Windowing) ==========
        self.llr_hist_length = cfg.llr_hist_length
        self.llr_bias = cfg.llr_bias

        # ========== Active Distribution ==========
        self.active_dist_threshold = cfg.active_dist_threshold

        # ========== Birth: Free → Tracking (Unrealized) ==========
        self.alignment_threshold = cfg.alignment_threshold
        self.tracking_active_threshold = cfg.tracking_active_threshold
        self.tracking_floor_weight = cfg.tracking_floor_weight

        # ========== Realize: Tracking (Unrealized) → Tracking (Realized) ==========
        self.realize_sum_thresh = cfg.realize_sum_thresh
        self.realize_hitrate_thresh = cfg.realize_hitrate_thresh

        # ========== Death (Tracking → Free) ==========
        self.death_ttl_base = cfg.death_ttl_base
        self.death_ttl_gain = cfg.death_ttl_gain
        self.ttl_sum_thresh = cfg.ttl_sum_thresh
        self.ttl_hitrate_thresh = cfg.ttl_hitrate_thresh
        self.death_ttl_max = cfg.death_ttl_max

        # ========== Newborn Weight Mixing ==========
        self.newborn_mix_coeff = cfg.newborn_mix_coeff
        self.newborn_conf_power = cfg.newborn_conf_power

        # ========== LC Detection (Overlap-only with confidence guard) ==========
        self.detect_overlap_sum_thresh = cfg.detect_overlap_sum_thresh
        self.detect_overlap_hitrate_thresh = cfg.detect_overlap_hitrate_thresh
        self.detect_overlap_rel_margin = cfg.detect_overlap_rel_margin
        self.detect_conf_rel_margin = cfg.detect_conf_rel_margin
        self.detect_conf_hitrate_thresh = cfg.detect_conf_hitrate_thresh

        # ========== Self LC Detection (comp 0, aligned confidence) ==========
        self.self_lc_conf_thresh = cfg.self_lc_conf_thresh
        self.self_lc_cooldown_steps = cfg.self_lc_cooldown_steps
        self.self_lc_min_kf_id_gap = cfg.self_lc_min_kf_id_gap

        # no pgo for lc (used for testing only)
        self.no_pgo_for_lc = cfg.no_pgo_for_lc

        # ========== comp 0 keep-alive (observability) ==========
        self.comp0_keepalive_score = cfg.comp0_keepalive_score
        self.comp0_sigma_inflation_factor = cfg.comp0_sigma_inflation_factor
        self.comp0_weight_floor = cfg.comp0_weight_floor

        # Initialize all tracking state metadata to defaults
        self.reset_tracking_state()

        # Ensure hypothesis 0 exists
        self.create_hypothesis_branch(0, 0)

        self.visualize_pose_graph = cfg.visualize_pose_graph

    @property
    def proximity_edges(self) -> Dict[Tuple[int, int], Edge]:
        """
        Expose proximity edges maintained by System.topo_map.

        Kept for compatibility with existing planner code.
        """
        topo_map = getattr(self.system, "topo_map", None)
        return topo_map.proximity_edges if topo_map is not None else {}

    @property
    def proximity_adjacency(self) -> Dict[int, Set[int]]:
        """
        Expose proximity adjacency maintained by System.topo_map.

        Kept for compatibility with existing planner code.
        """
        topo_map = getattr(self.system, "topo_map", None)
        return topo_map.proximity_adjacency if topo_map is not None else {}

    def get_active_dist(self):
        """
        Returns the active distribution.
        The first component is always active (never masked out), even if its
        instantaneous tracking weight dips below the general active threshold.
        """
        non_active_components = torch.logical_or(self.dist[2] < self.active_dist_threshold, ~self.realized)
        # Never mask out comp 0 in the returned active distribution
        if non_active_components.numel() > 0:
            non_active_components[0] = False

        current_mu = self.dist[0].clone()
        current_sigma = self.dist[1].clone()
        current_weights = self.dist[2].clone()
        current_mu[non_active_components] = pp.identity_SE3(1, device=current_mu.device)
        current_sigma[non_active_components] = pp.identity_se3(1, device=current_mu.device)
        current_weights[non_active_components] = 0.0

        return current_mu, current_sigma, current_weights

    def reset_tracking_state(self):
        """
        Reset all tracking state metadata to defaults.

        Called when:
        1. Initializing the system (__init__)
        2. Loading a map (load_state)
        3. Robot is kidnapped (system detects no odometry)

        This resets:
        - Lifecycle metadata (ttl, realized, last_seen_step, newborn, step_counter)
        - Evidence tracking (llr_hist, last_sum_pos, last_hit_rate, log_c_hist, log_conf_hist)

        Does NOT reset:
        - Graph structure (nodes, odom_edges, hypotheses)
        - Distribution (dist) - remains None until initialized
        """
        # Reset lifecycle metadata
        self.ttl = torch.zeros(self.n_components, dtype=torch.long, device=self.device)
        self.last_seen_step = torch.zeros(self.n_components, dtype=torch.long, device=self.device)
        self.newborn = torch.zeros(self.n_components, dtype=torch.bool, device=self.device)
        self.step_counter = 0
        self.realized = torch.zeros(self.n_components, dtype=torch.bool, device=self.device)

        # Reset evidence tracking
        self.llr_hist = torch.zeros(self.n_components, self.llr_hist_length, device=self.device)
        self.llr_hist_ptr = 0
        self.last_sum_pos = torch.zeros(self.n_components, device=self.device)
        self.last_hit_rate = torch.zeros(self.n_components, device=self.device)
        self.log_c_hist = torch.zeros(self.n_components, self.llr_hist_length, device=self.device)
        self.log_conf_hist = torch.zeros(self.n_components, self.llr_hist_length, device=self.device)

        # Mark existing hypotheses as realized and give them base TTL
        for comp_id in self.hypotheses.keys():
            if comp_id < self.realized.numel():
                self.realized[comp_id] = True
                self.ttl[comp_id] = self.death_ttl_base

        logger.debug("Reset tracking state metadata to defaults")

    def add_node(self, keyframe: Keyframe):
        """Registers a new keyframe in the system."""
        assert 0 in self.hypotheses, "Hypothesis 0 must exist"
        # Structural insert guarded by graph lock
        with self.graph_lock:
            if keyframe.id not in self.nodes:
                self.nodes[keyframe.id] = keyframe
                # Optionally update proximity edges incrementally for new permanent keyframes
                if self.system.topo_map is not None:
                    self.system.topo_map.handle_new_node(keyframe.id)

        # Batch check all unrealized components (excluding comp 0) for realization
        unrealized_mask = ~self.realized
        unrealized_mask[0] = False  # Exclude comp 0 (already realized)

        if self.last_sum_pos is None:
            # not initialized yet, skip
            return
        # Vectorized realization check
        to_realize_mask = unrealized_mask & \
                          (self.last_sum_pos >= self.realize_sum_thresh) & \
                          (self.last_hit_rate >= self.realize_hitrate_thresh)

        to_realize_indices = torch.where(to_realize_mask)[0].tolist()
        for comp_id in to_realize_indices:
            self.create_hypothesis_branch(comp_id, keyframe.id)
            
    def create_hypothesis_branch(self, new_comp_id: int, start_idx: int) -> int:
        """
        Creates a new hypothesis by branching from an existing one.
        The new hypothesis inherits the graph structure of its parent up to the branching point.

        Args:
            new_comp_id (int): The component ID of the new hypothesis.
            start_idx (int): The keyframe ID where the divergence occurs.

        Returns:
            int: The component ID of the newly created hypothesis.
        """
        # we can later solve this by first extract two subgraphs for parent and the new hypothesis, and then
        # merge them together by linking using the odom edges

        logger.debug(f"Creating new hypothesis branch {new_comp_id} at KF {start_idx}.")
        new_h = Hypothesis(component_id=new_comp_id, start_idx=start_idx)

        self.hypotheses[new_comp_id] = new_h
        self.realized[new_comp_id] = True
        # Initialize TTL for realized components
        self.ttl[new_comp_id] = max(int(self.ttl[new_comp_id].item()), self.death_ttl_base)

        return new_comp_id


    def add_edge(
        self,
        id1: int,
        id2: int,
        rel_pose_mean: pp.LieTensor,
        rel_pose_std: pp.LieTensor,
        type: EdgeType,
        from_comp_id: int = 0,
        to_comp_id: int = 0,
    ):
        """
        Adds a measurement (edge) to the relevant hypothesis graphs.

        Args:
            id1 (int): Source keyframe ID.
            id2 (int): Destination keyframe ID.
            rel_pose_mean (pp.LieTensor): Relative pose measurement.
            rel_pose_std (pp.LieTensor): Measurement std.
            type (EdgeType): Type of edge (EdgeType.VISUAL, EdgeType.ODOMETRY, etc.).
            from_comp_id (int): Source component ID (for visual/inter-hypothesis edges).
            to_comp_id (int): Target component ID (for visual/inter-hypothesis edges).
        """
        if id1 not in self.nodes or id2 not in self.nodes:
            logger.warning(f"Attempted to add edge between non-existent nodes: {id1}, {id2}")
            return

        # --- Logic for Odometry Edges ---
        # Odometry connects consecutive keyframes. It is added to the global graph,
        # as it represents the continuous motion of the robot in each possible "reality".
        if type == EdgeType.ODOMETRY:
            factor = Edge(rel_pose_mean, rel_pose_std, type) # comp_ids are not used for odom edges
            # Structural mutation under lock
            with self.graph_lock:
                self.odom_edges[(id1, id2)] = factor # it's directed edge, from id1 to id2

        # --- Logic for Visual Edges 
        elif type == EdgeType.VISUAL:
            # if to_comp_id is not in the hypotheses, skip
            if to_comp_id not in self.hypotheses:
                logger.debug(f"Visual edge to non-existent hypothesis {to_comp_id} skipped.")
                return
            factor = VisualEdge(
                rel_pose_mean, rel_pose_std, type, from_comp_id, to_comp_id
            )
            # NOTE: the visual edge is only added to the target hypothesis,
            with self.graph_lock:
                self.hypotheses[to_comp_id].add_visual_edge(id1, id2, factor)

    @timeit
    def remove_temporary_keyframe(self, last_k: int = 30):
        """Remove temporary keyframes from the hypothesis manager.
        It iterate from n-k to n (last kf), and check if they are temporary.
        1. add a odom edge between k-1 and k+1 kf (if it's not the last)
        2. remove the temporary kf
        Args:
            last_k (int): The number of last keyframes to keep.
        """
        if not self.nodes or len(self.nodes) < 20:
            return
            
        # Get all keyframe IDs sorted by ID (which should be chronological order)
        sorted_kf_ids = sorted(self.nodes.keys())
        
        # Identify temporary keyframes to remove
        temp_kfs_to_remove = [
            kf_id for kf_id in sorted_kf_ids if self.nodes[kf_id].temporary
        ][:-last_k]
        
        if not temp_kfs_to_remove:
            logger.debug("No temporary keyframes to remove")
            return
            
        logger.debug(f"Removing {len(temp_kfs_to_remove)} temporary keyframes: {temp_kfs_to_remove}")

        # Structural remove guarded by graph lock
        with self.graph_lock:
            # Process each temporary keyframe for removal
            for temp_kf_id in temp_kfs_to_remove:
                # Find predecessor and successor in odometry chain
                predecessor_id = None
                successor_id = None
            
            # Find predecessor (keyframe that has odometry edge TO this temp keyframe)
            for (id1, id2), _ in self.odom_edges.items():
                if id2 == temp_kf_id:
                    predecessor_id = id1
                    break
                    
            # Find successor (keyframe that this temp keyframe has odometry edge TO)
            for (id1, id2), _ in self.odom_edges.items():
                if id1 == temp_kf_id:
                    successor_id = id2
                    break
            
            # If we have both predecessor and successor, create a bridging odometry edge
            if predecessor_id is not None and successor_id is not None:
                # Get the two odometry edges to combine
                pred_edge = self.odom_edges.get((predecessor_id, temp_kf_id))
                succ_edge = self.odom_edges.get((temp_kf_id, successor_id))
                
                if pred_edge and succ_edge:
                    # Compose the relative poses: T_pred_to_succ = T_pred_to_temp @ T_temp_to_succ
                    combined_pose = pred_edge.mean @ succ_edge.mean
                    
                    # Combine the uncertainties (add variances in tangent space)
                    combined_std_squared = pred_edge.std.tensor()**2 + succ_edge.std.tensor()**2
                    combined_std = pp.se3(combined_std_squared**0.5)
                    
                    # Create new bridging odometry edge
                    bridging_factor = Edge(combined_pose, combined_std, EdgeType.ODOMETRY)
                    self.odom_edges[(predecessor_id, successor_id)] = bridging_factor
                    
                    # Note: No need to maintain adjacency list since odometry edges are sequential
                    
                    logger.debug(f"Created bridging odometry edge from KF {predecessor_id} to KF {successor_id}")
            
            # Remove odometry edges involving the temporary keyframe
            edges_to_remove = []
            for edge_key in list(self.odom_edges.keys()):
                id1, id2 = edge_key
                if id1 == temp_kf_id or id2 == temp_kf_id:
                    edges_to_remove.append(edge_key)
            
            for edge_key in edges_to_remove:
                del self.odom_edges[edge_key]
                logger.debug(f"Removed odometry edge {edge_key}")
            
            # Note: No odometry adjacency list to update since odometry edges are sequential
            
            # Remove visual edges involving the temporary keyframe from all hypotheses
            for hypothesis in self.hypotheses.values():
                visual_edges_to_remove = []
                for edge_key in hypothesis.visual_edges.keys():
                    id1, id2 = edge_key
                    if id1 == temp_kf_id or id2 == temp_kf_id:
                        visual_edges_to_remove.append(edge_key)
                
                for edge_key in visual_edges_to_remove:
                    del hypothesis.visual_edges[edge_key]
                    logger.debug(f"Removed visual edge {edge_key} from hypothesis {hypothesis.component_id}")
                
                # Update visual adjacency lists
                if temp_kf_id in hypothesis.visual_adjacency:
                    # Remove temp_kf_id from its neighbors' adjacency lists
                    for neighbor_id in hypothesis.visual_adjacency[temp_kf_id]:
                        if neighbor_id in hypothesis.visual_adjacency and temp_kf_id in hypothesis.visual_adjacency[neighbor_id]:
                            hypothesis.visual_adjacency[neighbor_id].remove(temp_kf_id)
                    # Remove temp_kf_id's own adjacency list
                    del hypothesis.visual_adjacency[temp_kf_id]
            
            # Finally, remove the temporary keyframe from nodes
            del self.nodes[temp_kf_id]
            logger.debug(f"Removed temporary keyframe {temp_kf_id}")
        
        logger.debug(f"Successfully removed {len(temp_kfs_to_remove)} temporary keyframes")

    def remove_hypothesis(self, component_id: int):
        """
        Removes a hypothesis and cleans up its component data from all keyframes.

        Args:
            component_id (int): The ID of the hypothesis to remove.
        """

        with self.graph_lock:
            h_to_remove = self.hypotheses[component_id]
            logger.debug(f"Removing hypothesis {component_id}, which started at KF {h_to_remove.start_idx}.")

            # --- Remove Component from Keyframes ---
            for kf in self.nodes.values():
                if kf.id >= h_to_remove.start_idx:
                    kf.pose_mu[component_id] = pp.identity_SE3(1, device=kf.pose_mu.device)
                    kf.pose_std[component_id] = pp.identity_se3(1, device=kf.pose_mu.device)
                    kf.pose_weights[component_id] = 0.0

            # --- Delete the Hypothesis ---
            del self.hypotheses[component_id]

        # remove from self.dist (tracking dist) and renormalize
        mu, sigma, weights = self.dist
        mu[component_id] = pp.identity_SE3(1, device=mu.device)
        sigma[component_id] = pp.identity_se3(1, device=sigma.device)
        weights[component_id] = 0.0
        weights = weights / weights.sum()
        self.dist = (mu, sigma, weights)


    def reset_tracking_dist(self):
        """
        Reset the tracking distribution to the identity so that it has
        identity pose and std, and all zero weights
        """
        B = self.dist[0].shape[0]
        device = self.dist[0].device
        mu = pp.identity_SE3(B, device=device)
        sigma = pp.identity_se3(B, device=device)
        weights = torch.zeros(B, device=device)
        self.dist = (mu, sigma, weights)


    def motion_update(
        self, 
        delta_pose: pp.LieTensor,
        delta_std: pp.LieTensor,
    ):
        """Motion update the current state GMM
        Args:
            delta_pose: (7,)
            delta_std: (6,)
            pose_update_mask: (K,) bool mask; True means apply retrieval update; False means revert to prior
        The motion update won't change the number of components, since it's continuous.
        If pose is not filtered, we'll not update the std
        """
        last_gmm_mu, last_gmm_sigma, last_gmm_weights = self.dist
        # Update all active components by weight threshold so priors evolve with motion
        active_mask = last_gmm_weights > self.tracking_active_threshold
        last_gmm_mu[active_mask] = last_gmm_mu[active_mask] @ delta_pose.unsqueeze(0)
        last_gmm_sigma[active_mask] = last_gmm_sigma[active_mask] + delta_std.unsqueeze(0)

        self.dist = (last_gmm_mu, last_gmm_sigma, last_gmm_weights)

    @timeit
    def align_proposal_prior(self, proposal_hypotheses: List[Dict]):
        """
        Aligns new proposals with the current GMM state, handles hypothesis birth/death,
        and prepares the GMMs for the filtering step.

        Args:
            proposal_hypotheses: A list of dicts, 
            {
                'pose': pp.LieTensor,
                'std': pp.LieTensor,
                'score': float,
                'source_indices': List[Tuple[int, int]], # M, 2
            }
        """
        # --- Step 1: Initialization and Projection ---
        current_mu, current_std, current_weights = self.dist

        # Advance internal step counter for recency tracking
        self.step_counter += 1

        # Active set by weight: include any component with nontrivial mass
        active_comps_mask = (current_weights > self.tracking_active_threshold).to(current_mu.tensor().device)
        # Always consider comp 0 active for alignment/matching to maintain observability
        if active_comps_mask.numel() > 0:
            active_comps_mask[0] = True
        active_comp_indices = torch.where(active_comps_mask)[0]
        
        # Key: kf_id (from_node_id), Value: List of (from_comp_id, to_comp_id)
        edge_mapping: Dict[int, Tuple[int, int]] = {}

        proposal_mu = torch.stack([h['pose'] for h in proposal_hypotheses])
        
        # Project poses to tangent space to calculate meaningful distances.
        current_mu_proj = project_SE3(current_mu[active_comps_mask])
        proposal_mu_proj = project_SE3(proposal_mu)

        # --- Step 2: Greedy Best-First Matching ---
        # Calculate the pairwise distance between every active component and every proposal.
        dist_matrix = torch.cdist(current_mu_proj, proposal_mu_proj)

        # Prepare tensors for the new, aligned GMM. Default to low-confidence values.
        num_components = current_mu.shape[0]
        aligned_mu = pp.identity_SE3(num_components, device=current_mu.device)
        aligned_sigma = pp.identity_se3(num_components, device=current_mu.device) # High uncertainty
        aligned_weights = torch.zeros(num_components, device=current_mu.device)

        aligned_confidence = torch.zeros(num_components, device=current_mu.device)

        matched_proposals = set()
        matched_components = set()

        num_matches = min(len(active_comp_indices), len(proposal_hypotheses))
        for _ in range(num_matches):
            # Find the best possible match (smallest distance) in the matrix.
            min_val = dist_matrix.min()
            if min_val > self.alignment_threshold:
                break # No more good matches left.
            
            # Get the indices of this best match.
            res = torch.where(dist_matrix == min_val)
            # This gives the index within the *active* components and proposals
            active_comp_idx_in_subset, proposal_idx = res[0][0].item(), res[1][0].item()
            # Get the true component index in the full GMM tensor
            true_comp_idx = active_comp_indices[active_comp_idx_in_subset].item()
            
            # --- A match is found: update the aligned GMM ---
            proposal = proposal_hypotheses[proposal_idx]
            aligned_mu[true_comp_idx] = proposal['pose']
            aligned_sigma[true_comp_idx] = proposal['std']
            aligned_weights[true_comp_idx] = proposal['score']
            aligned_confidence[true_comp_idx] = proposal['score']

            # mark component as recently seen
            if true_comp_idx < len(self.last_seen_step):
                self.last_seen_step[true_comp_idx] = self.step_counter
            
            # For every source that formed this proposal, map it to the matched component.
            for kf_id, source_comp_id in proposal['source_indices']:
                edge_mapping[kf_id] = (source_comp_id, true_comp_idx)
                
            # Mark as matched so they are not used again.
            matched_proposals.add(proposal_idx)
            matched_components.add(true_comp_idx)
            self.disappearance_counts[true_comp_idx] = 0 # Reset disappearance counter

            # Invalidate this row and column in the distance matrix.
            dist_matrix[active_comp_idx_in_subset, :] = float('inf')
            dist_matrix[:, proposal_idx] = float('inf')

        # --- Step 3: Handle Unmatched Components and Proposals (Birth/Death) ---

        # Handle components that were not matched (potential disappearance).
        for comp_idx in active_comp_indices.tolist():
            if comp_idx not in matched_components:
                # Keep-alive for unmatched components.
                # - comp 0: mild std inflation, moderate score to maintain observability.
                # - others: retain pose, inflate uncertainty, tiny score.
                if comp_idx == 0:
                    logger.debug("Component 0 not matched, applying keep-alive (mild inflation, moderate score).")
                    aligned_mu[comp_idx] = current_mu[comp_idx]
                    aligned_sigma[comp_idx] = current_std[comp_idx] * self.comp0_sigma_inflation_factor
                    aligned_weights[comp_idx] = self.comp0_keepalive_score
                else:
                    logger.debug(f"Component {comp_idx} not matched, reducing weight.")
                    aligned_mu[comp_idx] = current_mu[comp_idx]
                    aligned_sigma[comp_idx] = current_std[comp_idx] * 2
                    aligned_weights[comp_idx] = 1e-6  # tiny keep-alive proposal score


        # Handle proposals that were not matched (new hypothesis birth with capacity/eviction)
        unmatched_proposals_indices = set(range(len(proposal_hypotheses))) - matched_proposals
        for proposal_idx in unmatched_proposals_indices:
            # Find the first empty slot in the GMM.
            available_slots = torch.where(aligned_weights == 0)[0]
            if len(available_slots) == 0:
                victim_idx = self._select_unrealized_eviction_candidate(current_weights)
                if victim_idx is None:
                    logger.debug("No free slots and no unrealized to evict. Proposal ignored.")
                    break
                self.realized[victim_idx] = False
                self.ttl[victim_idx] = 0
                self.newborn[victim_idx] = False
                available_slots = torch.tensor([victim_idx], device=available_slots.device)

            new_comp_idx = available_slots[0].item()
            proposal = proposal_hypotheses[proposal_idx]
            
            aligned_mu[new_comp_idx] = proposal['pose']
            aligned_sigma[new_comp_idx] = proposal['std']
            aligned_weights[new_comp_idx] = proposal['score']
            aligned_confidence[new_comp_idx] = proposal['score']

            # --- Build the mapping for the new hypothesis ---
            for kf_id, source_comp_id in proposal['source_indices']:
                edge_mapping[kf_id] = (source_comp_id, new_comp_idx)
            # Initialize newborn/unrealized metadata
            self.ttl[new_comp_idx] = self.death_ttl_base
            self.last_seen_step[new_comp_idx] = self.step_counter
            self.newborn[new_comp_idx] = True
            if new_comp_idx < self.realized.numel():
                self.realized[new_comp_idx] = False
            
            logger.debug(f"New pending hypothesis {new_comp_idx} detected. Tracking weights: {self.dist[2]}, Existing: {self.hypotheses.keys()}")
            
        # Renormalize weights to sum to 1.
        total_weight = aligned_weights.sum()
        if total_weight > 1e-6:
            aligned_weights /= total_weight
        
        return aligned_mu, aligned_sigma, aligned_weights, aligned_confidence, edge_mapping

    def _select_unrealized_eviction_candidate(self, current_weights: torch.Tensor) -> Optional[int]:
        """Select an unrealized active component to evict when all slots are full.

        Prefers the stalest (smallest last_seen_step) among unrealized actives (weight > threshold).
        Never evicts comp 0 or realized components.
        Returns comp index or None if no unrealized candidates exist.

        Note: This is only called when all slots are occupied and we need to make room for
        a new proposal. The artificial max_unrealized cap has been removed - TTL and evidence
        thresholds naturally control component lifecycle.
        """
        if self.n_components == 0:
            return None
        active = current_weights > self.tracking_active_threshold
        candidates = [
            i for i in range(1, self.n_components)
            if not bool(self.realized[i].item()) and bool(active[i].item())
        ]
        if len(candidates) == 0:
            return None
        # Always evict the stalest unrealized component when all slots are full
        victim_idx = min(candidates, key=lambda i: int(self.last_seen_step[i].item()))
        return int(victim_idx)

    @timeit
    def gmm_filtering(
        self,
        proposal_mu: pp.LieTensor,      # C1x7
        proposal_std_diag: pp.LieTensor,   # C1x6
        proposal_weights: torch.Tensor, # C1
        proposal_confidence: torch.Tensor, # C1
        pose_update_mask: Optional[torch.Tensor] = None, # K bool mask; True means apply retrieval update
    ):
        """
        Computes the final distribution p = alpha * (p_proposal * p_prior) + (1 - alpha) * p_proposal.
        """
        # Clamp confidences for numerical stability in LLR downstream
        proposal_confidence_clamped = torch.clamp(proposal_confidence, min=0.1, max=10)

        #################
        # First, we fuse the proposal with the prior for all tracking components
        #################
        # NOTE: we do the fusion in the prior tangent, which is more stable
        alpha = 0.9  # Fusion parameter. 0.5 gives equal weight to product and proposal.
        prior_mu, prior_std_diag, prior_weights = self.dist

        # Preserve the true prior variance without process noise for optional gating
        prior_var_diag_noQ = prior_std_diag**2

        currently_tracking = prior_weights > self.tracking_active_threshold

        prior_var_diag = prior_std_diag**2
        proposal_var_diag = proposal_std_diag**2

        # add process noise to the prior
        # NOTE: this is required, otherwise the prior std will keep shrinking
        # and kalman gain will goes to zero.
        # TODO: use the motion std?
        proc_std = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], 
                        device=prior_var_diag.device, dtype=prior_var_diag.dtype)
        Q = (proc_std**2).view(1,6)

        prior_var_diag = prior_var_diag + Q

        # --- Step 1: Compute the product GMM p_prod = p1 * p2 ---
        # Do the Gaussian product in a COMMON tangent (use the prior's tangent).
        # Let r be the residual log in the prior tangent: prior^{-1} * proposal.
        # Then δ = Σ (Σ1^{-1} r), with Σ = (Σ1^{-1} + Σ2^{-1})^{-1}.
        # No Adjoint needed since sigmas are already local/compatible.

        eps = 1e-9
        inv_var1_diag = 1.0 / proposal_var_diag.clamp_min(eps)  # proposal info (diag) in prior tangent
        inv_var2_diag = 1.0 / prior_var_diag.clamp_min(eps)     # prior info   (diag) in prior tangent
        prod_inv_var_diag = inv_var1_diag + inv_var2_diag
        prod_var_diag = 1.0 / prod_inv_var_diag                 # posterior std (diag) in prior tangent

        # Residual in prior tangent
        r = (prior_mu.Inv() @ proposal_mu).Log().tensor()                    # 6D twist in prior tangent

        # Posterior mean offset in prior tangent (δ)
        prod_log_mu = prod_var_diag * (inv_var1_diag * r)       # δ = Σ * (Σ1^{-1} r)

        # Map back to the group: μ_prod = μ_prior ∘ Exp(δ)
        prod_mu = prior_mu @ pp.se3(prod_log_mu).Exp()

        # Gaussian-overlap factor  c_k = N(r_k ; 0, S_k),  with S_k = Σ1_k + Σ2_k  (all diag)
        S_diag = (proposal_var_diag + prior_var_diag).clamp_min(eps)   # (C,6)
        inv_S_diag = 1.0 / S_diag
        maha = (r * r * inv_S_diag).sum(dim=-1)                            # (C,)
        log_det_S = torch.log(S_diag).sum(dim=-1)                          # (C,)
        log_c = -0.5 * (maha + (6.0 * math.log(2.0 * math.pi) + log_det_S))

        #################
        # Evidence tracking with LLR + bias
        #################
        eps_c = 1e-12
        llr_bias = self.llr_bias
        conf_ratio = (proposal_confidence_clamped + eps_c) / (proposal_confidence_clamped[0] + eps_c)
        conf_ratio = torch.clamp(conf_ratio, max=1.0) # low confidence penalize hypothesis, while high confidence neutral to avoid false positives
        llr = (log_c - log_c[0]) + torch.log(conf_ratio) + llr_bias

        # keep positive support
        pos = torch.relu(llr)

        # GATE: Only accumulate evidence for actively tracked components
        # Inactive components (weight ≈ 0) should not accumulate spurious evidence
        # from identity-pose overlap with identity proposals
        active_tracking_mask = prior_weights > self.tracking_active_threshold
        pos = torch.where(active_tracking_mask, pos, torch.zeros_like(pos))

        # Debug history: log individual LLR components (for tuning/visualization)
        log_c_rel = (log_c - log_c[0])  # relative log-likelihood
        log_conf = torch.log(conf_ratio)  # log confidence ratio
        self.log_c_hist[:, self.llr_hist_ptr] = torch.where(active_tracking_mask, log_c_rel, torch.zeros_like(log_c_rel))
        self.log_conf_hist[:, self.llr_hist_ptr] = torch.where(active_tracking_mask, log_conf, torch.zeros_like(log_conf))

        self.llr_hist[:, self.llr_hist_ptr] = pos
        self.llr_hist_ptr = (self.llr_hist_ptr + 1) % self.llr_hist_length

        # Summarize recent support
        sum_pos   = self.llr_hist.sum(dim=1)                       # total positive support in window
        hit_rate  = (self.llr_hist > 0).float().mean(dim=1)

        # Store evidence for downstream decisions
        self.last_sum_pos = sum_pos
        self.last_hit_rate = hit_rate

        #################
        # newborn seeding
        #################

        # Newborn seeding (once): set posterior μ/Σ to proposal for newborn
        newborn_mask = self.newborn.clone()
        if newborn_mask.any():
            prod_mu[newborn_mask] = proposal_mu[newborn_mask]
            prod_var_diag[newborn_mask] = proposal_var_diag[newborn_mask]

        #################
        # Weight update (state tracking only)
        #################
        prod_weights_unnorm = proposal_weights * prior_weights * alpha * log_c.exp()
        denom = prod_weights_unnorm[currently_tracking].sum().clamp_min(1e-12)
        prod_weights = prod_weights_unnorm / denom

        #################
        # Newborn weight mixing 
        #################
        if newborn_mask.any():
            # Effective mixing coefficient: 
            existing_mask = ~newborn_mask
            existing_sum = prod_weights[existing_mask].sum()

            B_eff = self.newborn_mix_coeff
            # Scale existing weights to sum to (1 - B_eff)
            prod_weights[existing_mask] = (1.0 - B_eff) * (prod_weights[existing_mask] / existing_sum.clamp_min(1e-12))

            # Distribute B_eff among newborns by confidence power
            # TODO: is this just softmax?
            s = proposal_confidence_clamped[newborn_mask]
            prod_weights[newborn_mask] = B_eff * (s / s.sum())

            # Renormalize for numerical stability
            prod_weights = prod_weights / prod_weights.sum().clamp_min(1e-12)

        #################
        # TTL + cleanup (unified, vectorized)
        #################
        # Active mask based on current weights
        active_mask = prod_weights > self.tracking_active_threshold

        # Work on components 1..K-1 (exclude comp 0 from TTL logic)
        if prod_weights.numel() > 1:
            idx_slice = slice(1, None)

            ttl_current = self.ttl[idx_slice].clone()
            sum_pos_sub = sum_pos[idx_slice]
            hit_rate_sub = hit_rate[idx_slice]
            active_sub = active_mask[idx_slice]

            # Boost condition: sufficient normalized evidence or hit rate
            boost_sub = (sum_pos_sub >= self.ttl_sum_thresh) | (hit_rate_sub >= self.ttl_hitrate_thresh)

            # Target TTL as integer: base + gain * normed, with base as minimum
            ttl_target_float = self.death_ttl_base + self.death_ttl_gain * sum_pos_sub
            ttl_target = torch.clamp(ttl_target_float, min=float(self.death_ttl_base), max=self.death_ttl_max).to(dtype=torch.long)

            # Unified update: if active and boosted => extend to at least target; else decrement
            extend_vals = torch.maximum(ttl_current, ttl_target)
            decayed_vals = torch.clamp(ttl_current - 1, min=0)
            self.ttl[idx_slice] = torch.where(active_sub & boost_sub, extend_vals, decayed_vals)

            # Cleanup components whose TTL reached zero (batched)
            dead_mask_sub = self.ttl[idx_slice] == 0
            if torch.any(dead_mask_sub):
                # Build full-length mask and list of dead indices (1..K-1)
                dead_full_mask = torch.zeros_like(prod_weights, dtype=torch.bool)
                dead_full_mask[idx_slice] = dead_mask_sub
                dead_indices = torch.nonzero(dead_full_mask, as_tuple=False).squeeze(-1)

                # Reset mixture parameters for dead components
                n_dead = int(dead_indices.numel())
                if n_dead > 0:
                    prod_weights[dead_full_mask] = 0.0
                    prod_mu[dead_full_mask] = pp.identity_SE3(n_dead, device=prod_mu.device)
                    prod_var_diag[dead_full_mask] = pp.identity_se3(n_dead, device=prod_var_diag.device)
                    self.realized[dead_full_mask] = False
                    self.ttl[dead_full_mask] = 0

                    # Remove realized hypothesis branches for dead comps (data-structure loop)
                    for comp_idx in dead_indices.tolist():
                        if comp_idx in self.hypotheses:
                            self.remove_hypothesis(comp_idx)
                        active_mask[comp_idx] = False

            # Renormalize after TTL removals
            prod_weights = prod_weights / prod_weights.sum().clamp_min(1e-12)

        #################
        # Post-TTL weight floor for still-tracking components
        #################
        still_tracking = self.ttl > 0
        if torch.any(still_tracking):
            floor_vals = torch.full_like(prod_weights[still_tracking], self.tracking_floor_weight)
            prod_weights[still_tracking] = torch.maximum(prod_weights[still_tracking], floor_vals)
            # Renormalize to maintain a valid mixture
            prod_weights = prod_weights / prod_weights.sum().clamp_min(1e-12)

        # Apply pose-update gating: optionally skip retrieval-based pose update for selected components.
        # If a component is skipped, revert its pose/covariance to the true prior (without added process noise).
        if pose_update_mask is not None:
            if pose_update_mask.dtype != torch.bool:
                pose_update_mask = pose_update_mask.to(dtype=torch.bool)
            # Always update newborns regardless of mask
            final_update_mask = pose_update_mask.clone()
            if final_update_mask.numel() != prior_var_diag_noQ.shape[0]:
                # Ensure shape aligns with number of components
                final_update_mask = torch.ones(prior_var_diag_noQ.shape[0], dtype=torch.bool, device=prior_var_diag_noQ.device)
            final_update_mask = torch.logical_or(final_update_mask, newborn_mask)

            # Revert selected components to prior without Q
            if (~final_update_mask).any():
                revert_mask = ~final_update_mask
                prod_mu[revert_mask] = prior_mu[revert_mask]

        # Comp 0 specific weight floor to maintain observability in downstream consumers.
        # This keeps comp 0 above the active distribution threshold without dominating others.
        if prod_weights.numel() > 0:
            comp0_floor = torch.tensor(self.comp0_weight_floor, device=prod_weights.device, dtype=prod_weights.dtype)
            prod_weights[0] = torch.maximum(prod_weights[0], comp0_floor)
            prod_weights = prod_weights / prod_weights.sum().clamp_min(1e-12)

        # Clear newborn flags that were applied in this step
        if newborn_mask.any():
            self.newborn[newborn_mask] = False

        prod_std_diag = pp.se3(prod_var_diag**0.5)

        if torch.isnan(prod_weights).any():
            logger.warning(f"prod_weights is nan: {prod_weights}")
        self.dist = (prod_mu, prod_std_diag, prod_weights)
        
    def detect_loop_closure(self, ret):
        """
        Detect loop closure using overlap-only evidence with a confidence guard.

        Signals (existing histories populated in gmm_filtering):
        - `log_c_hist` (K × L): relative overlap-only log-likelihood vs comp 0 per frame
          (no confidence mixed in). We require strong positive support and a high
          hit-rate against comp 0.
        - `log_conf_hist` (K × L): relative confidence log-ratio vs comp 0 per frame.
          A positive bias margin is applied so that slightly lower-than-H0 confidence
          does not suppress loop-closure.

        Decision (realized + currently active comps only, exclude 0):
        - Overlap metrics per comp: sum_overlap = Σ_t ReLU(log_c_rel_t); hit_overlap = mean_t(log_c_rel_t > 0).
        - Confidence guard (soft): conf_hit_rate = mean_t(log_conf_rel_t + margin > 0).
        - Select comp with max sum_overlap; trigger if sum_overlap ≥ detect_overlap_sum_thresh and
          hit_overlap ≥ detect_overlap_hitrate_thresh and conf_hit_rate ≥ detect_conf_hitrate_thresh.

        Returns:
            dict: { 'loop_closure': bool, 'loop_closure_hypo_id': Optional[int] }
        """

        # 1) Inter-hypothesis LC detection (exclude comp 0)
        # only consider active and realized components
        active_mask = torch.logical_and(self.dist[2] > self.active_dist_threshold, self.realized)[1:]
        if active_mask.any():
            realized_ids = torch.nonzero(active_mask, as_tuple=False).squeeze(-1) + 1 # +1 because we exclude comp 0

            # Stack and index using realized IDs
            log_c_rel_realized = self.log_c_hist[realized_ids, :] # (C, L)
            log_conf_rel_realized = self.log_conf_hist[realized_ids, :] # (C, L)

            log_c_pos_sum = torch.relu(log_c_rel_realized + self.detect_overlap_rel_margin).sum(dim=1)
            log_c_pos_hit_rate = (log_c_rel_realized + self.detect_overlap_rel_margin > 0).float().mean(dim=1)

            log_conf_hit_rate = (log_conf_rel_realized + self.detect_conf_rel_margin > 0).float().mean(dim=1) # bias to allow lower than H0

            # Reject LC if the candidate is too close to the current pose, i.e. likely to be a false positive
            current_pose = self.dist[0][0][:3].unsqueeze(0) # (1, 3)
            candidate_poses = self.dist[0][realized_ids][:, :3] # (C, 3)
            distances = torch.norm(candidate_poses - current_pose, dim=1) # (C,)

            close_mask = distances < 3 # 

            detected_mask = (log_c_pos_sum >= self.detect_overlap_sum_thresh) \
                & (log_c_pos_hit_rate >= self.detect_overlap_hitrate_thresh) \
                & (log_conf_hit_rate >= self.detect_conf_hitrate_thresh) \
                & (~close_mask)

            if detected_mask.any():
                # Map argmax within the masked array back to original realized indices
                masked_scores = log_c_pos_sum[detected_mask]
                best_within_mask = int(torch.argmax(masked_scores).item())
                mask_indices = torch.nonzero(detected_mask, as_tuple=False).squeeze(-1)
                best_idx = int(mask_indices[best_within_mask].item())

                comp_id = int(realized_ids[best_idx].item())
                logger.debug(
                    f"LC candidate by overlap-only: comp {comp_id}, sum_overlap={log_c_pos_sum[best_idx].item():.3f}, hit_overlap={log_c_pos_hit_rate[best_idx].item():.3f}, conf_hit_rate={log_conf_hit_rate[best_idx].item():.3f}"
                )
                return { 'loop_closure': True, 'loop_closure_hypo_id': comp_id }

        return { 'loop_closure': False, 'loop_closure_hypo_id': None }
    
    def handle_loop_closure(
        self, hypo_id: int, target_node_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Handle loop closure.
        1. Construct a pose graph with current active (0) hypo and the other hypo for loop closure.
        2. perform PGO
        3. Update the current active (0) hypo with the new poses
        4. Remove the other hypo and temporary nodes and edges
        
        Args:
            hypo_id: The hypothesis ID to merge with hypothesis 0
            target_node_id: The central node to expand from (defaults to latest keyframe)
        
        Returns:
            Dict[str, Any]: Information required for visualization and logging. 
        """
        result = {
            "success": False,
            "cost": None,
            "nodes": [],
            "edges": [],
            "optimized_poses": {},
            "optim_nodes_ids": set(),
            "fixed_nodes_ids": set(),
            "hypothesis_id": 0,
            "other_hypothesis_id": hypo_id,
            "target_node_id": target_node_id,
            "message": None,
        }
        assert hypo_id in self.hypotheses, f"Hypothesis {hypo_id} does not exist"

        if self.no_pgo_for_lc:
            # instead of merging, we just change the hypothesis 0 to the new hypothesis
            # this is only for test
            self.change_hypo_to_first(hypo_id)
            return result

        # Use the latest keyframe as target if not specified
        if target_node_id is None:
            target_node_id = max(self.nodes.keys())
        result["target_node_id"] = target_node_id

        logger.info(f"Handling loop closure: merging hypothesis {hypo_id} with hypothesis 0")

        # Step 1: Construct the pose graph for loop closure (under graph lock)
        with self.graph_lock:
            pg = PoseGraph(
                self,
                depth=1000,
                k_hop=2,
                device=self.device,
            )
            pg.construct_for_loop_closure(
                target_node_id=target_node_id,
                other_hypothesis_id=hypo_id,
            )

        if len(pg.vertices) < 10 or len(pg.edges) < 10:
            message = "Too few vertices or edges for loop closure, skipping"
            logger.warning(message)
            result["message"] = message
            return result

        # Step 2: Perform PGO
        # We'll fix the earliest keyframe in hypothesis 0 and optimize the rest
        # Separate original keyframes from temporary vertices
        original_kf_ids = [v.id for v in pg.vertices if v.id in self.nodes]
        temp_vertex_ids = [v.id for v in pg.vertices if v.id not in self.nodes]

        # Fix the earliest keyframe from hypothesis 0
        fixed_node_id = min(original_kf_ids)
        optim_node_ids = set(original_kf_ids + temp_vertex_ids) - {fixed_node_id}

        if self.visualize_pose_graph:
            visualize_pose_graph(
                pg,
                title=f"Loop Closure: Hypo 0 + Hypo {hypo_id}",
                save_path="logs/loop_closure_graph.png",
                last_k_nodes=10,
                show_interactive=False,
            )

        pg.solve(
            optim_node_ids=optim_node_ids,
            fixed_node_ids={fixed_node_id},
        )

        logger.info(f"Loop closure PGO completed with cost: {pg.optimization_cost}")

        # Step 3: Apply updates via unified method (also used by async engine)
        self.apply_pgo_result({
            "success": True,
            "pose_graph": pg,
            "optimized_poses": pg.optimized_poses,
            "other_hypothesis_id": hypo_id,
            "target_node_id": target_node_id,
        })

        result.update(
            {
                "success": True,
                "cost": pg.optimization_cost,
                "pose_graph": pg,
                "optimized_poses": pg.optimized_poses,
                "optim_nodes_ids": optim_node_ids,
                "fixed_nodes_ids": {fixed_node_id},
                "message": None,
            }
        )

        return result

    def apply_pgo_result(self, pgo_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a PGO result to update node poses and optionally merge hypotheses.

        Args:
            pgo_result: Dict containing at least:
                - optimized_poses: Dict[int, pp.LieTensor]
                - other_hypothesis_id: int
                - pose_graph: PoseGraph (optional)
        Returns:
            Dict summarizing the application with success and cost.
        """
        optimized_poses: Dict[int, pp.LieTensor] = pgo_result.get("optimized_poses", {})
        other_hypo = int(pgo_result.get("other_hypothesis_id", 0))
        pg = pgo_result.get("pose_graph")
        affected_ids = set(optimized_poses.keys())

        # Mutate shared graph under lock
        with self.graph_lock:
            for node_id, optimized_pose in optimized_poses.items():
                if node_id in self.nodes:
                    kf = self.nodes[node_id]
                    kf.pose_mu[0] = optimized_pose
                    # Reduce uncertainty after optimization
                    kf.pose_std[0] = kf.pose_std[0] * 0.5
                    kf.last_pgo_step = int(self.step_counter)
                    logger.debug(f"Applied PGO update to KF {node_id}")

            if other_hypo != 0 and other_hypo in self.hypotheses:
                self.merge_hypotheses(other_hypo)

            if affected_ids:
                if self.system.topo_map is not None:
                    self.system.topo_map.update_after_pgo(affected_ids)

            # Align tracking dist to latest KF in comp 0
            if self.system.last_added_kf_id is not None and self.system.last_added_kf_id in self.nodes:
                last_kf_id = self.system.last_added_kf_id
                self.dist[0][0] = self.nodes[last_kf_id].pose_mu[0]
                self.dist[1][0] = self.nodes[last_kf_id].pose_std[0]
                self.dist[2][0] = 1

        ret = {
            "success": True,
            "pose_graph": pg,
            "optimized_poses": optimized_poses,
            "other_hypothesis_id": other_hypo,
        }
        if pg is not None:
            ret["cost"] = pg.optimization_cost
        return ret

    def rebuild_topology_graph(self):
        """
        Rebuild the SimpleTopo proximity graph from scratch.

        This clears existing proximity edges and recomputes them using current
        node poses. Useful when incremental proximity is disabled.
        """
        if self.system.topo_map is not None:
            self.system.topo_map.rebuild_graph()

    def merge_hypotheses(self, comp_idx: int):
        """
        Merges the hypothesis after loop closure
        """
        logger.debug(f"Merging hypothesis {comp_idx} after loop closure")
        with self.graph_lock:
            # copy edges and adjs
            # cache start index for keyframe cleanup before deleting hypothesis object
            start_idx = self.hypotheses[comp_idx].start_idx if comp_idx in self.hypotheses else None
            # first update the edge comp_ids to 0
            for edge_key, edge_factors in self.hypotheses[comp_idx].visual_edges.items():
                for edge_factor in edge_factors:
                    # ignore edges from other hypothesis than 0 and comp_idx
                    if (edge_factor.from_comp_id == 0 and edge_factor.to_comp_id == comp_idx) or \
                        (edge_factor.from_comp_id == comp_idx and edge_factor.to_comp_id == 0):

                        # change the comp_ids to 0
                        edge_factor.from_comp_id = 0
                        edge_factor.to_comp_id = 0

                        # add modified edge factor
                        self.hypotheses[0].visual_edges.setdefault(edge_key, []).append(edge_factor)
                        
                        # Sets automatically prevent duplicates
                        self.hypotheses[0].visual_adjacency.setdefault(edge_key[0], set()).add(edge_key[1])
                        self.hypotheses[0].visual_adjacency.setdefault(edge_key[1], set()).add(edge_key[0])

        self.remove_hypothesis(comp_idx)


        # Reset lifecycle metadata for the removed component
        self.realized[comp_idx] = False
        self.ttl[comp_idx] = 0
        self.newborn[comp_idx] = False

        # Reset evidence history if present
        self.llr_hist[comp_idx, :] = 0.0
        self.last_sum_pos[comp_idx] = 0.0
        self.last_hit_rate[comp_idx] = 0.0

    def change_hypo_to_first(self, comp_idx: int):
        """
        Changes the hypothesis with the highest weight to the first component, and remove it
        """
        logger.debug(f"Changing hypothesis {comp_idx} to first component")
        hypothesis_exist = comp_idx in self.hypotheses

        if not hypothesis_exist:
            logger.debug(f"Hypothesis {comp_idx} does not exist, skipping")
            return

        # copy edges and adjs
        # TODO: this is not correct, as the edge comp_ids are not updated
        self.hypotheses[0].visual_edges.update(self.hypotheses[comp_idx].visual_edges)
        
        # Merge adjacency sets - set union automatically handles duplicates
        for node_id, neighbors in self.hypotheses[comp_idx].visual_adjacency.items():
            if node_id not in self.hypotheses[0].visual_adjacency:
                self.hypotheses[0].visual_adjacency[node_id] = neighbors.copy()
            else:
                self.hypotheses[0].visual_adjacency[node_id] |= neighbors

        # update the poses in the keyframes
        # only hypothesis existing, new components is added
        for kf in self.nodes.values():
            if kf.id >= self.hypotheses[comp_idx].start_idx:
                kf.pose_mu[0] = kf.pose_mu[comp_idx]
                kf.pose_std[0] = kf.pose_std[comp_idx]
                kf.pose_weights[0] = kf.pose_weights[comp_idx]
                kf.pose_mu[comp_idx] = pp.identity_SE3(1, device=kf.pose_mu.device)
                kf.pose_std[comp_idx] = pp.identity_se3(1, device=kf.pose_mu.device)
                kf.pose_weights[comp_idx] = 0.0
        
        # remove the hypothesis
        del self.hypotheses[comp_idx]


        mu, sigma, weights = self.dist
        mu[0] = mu[comp_idx]
        sigma[0] = sigma[comp_idx]
        weights[0] = weights[comp_idx]
        mu[comp_idx] = pp.identity_SE3(1, device=mu.device)
        sigma[comp_idx] = pp.identity_se3(1, device=sigma.device)
        weights[comp_idx] = 0.0
        weights = weights / weights.sum()
        self.dist = (mu, sigma, weights)
    
    def save_state(self):
        """Save the hypothesis manager state for map persistence.
        Only saves hypothesis 0 (ground truth) and temporary keyframes.
        Warns if multiple realized hypotheses exist at save time.

        Returns:
            dict: Hypothesis manager state including temp keyframes, edges, and hypothesis 0
        """
        # --- Check for unresolved ambiguity ---
        realized_hypos = [comp_id for comp_id in self.hypotheses.keys() if self.realized[comp_id]]
        if len(realized_hypos) > 1:
            logger.warning(
                f"Multiple realized hypotheses exist at save time: {realized_hypos}. "
                f"Only hypothesis 0 will be saved. Other hypotheses represent unresolved ambiguity "
                f"that may not be relevant after loading the map in a new session."
            )

        # --- 1. Save only temporary keyframes (not in database) ---
        temp_keyframes = []
        for kf_id, kf in self.nodes.items():
            if kf.temporary:
                temp_keyframes.append({
                    "id": kf.id,
                    "pose_mu": kf.pose_mu.cpu() if kf.pose_mu is not None else None,
                    "pose_std": kf.pose_std.cpu() if kf.pose_std is not None else None,
                    "pose_weights": kf.pose_weights.cpu() if kf.pose_weights is not None else None,
                    "timestamp": kf.timestamp,
                    "temporary": kf.temporary,
                    "atlas_id": kf.atlas.id if kf.atlas is not None else None,
                    "last_pgo_step": getattr(kf, "last_pgo_step", -1),
                })

        # --- 2. Save odometry edges ---
        odom_edges = {}
        for edge_key, edge in self.odom_edges.items():
            odom_edges[edge_key] = {
                "mean": edge.mean.cpu(),
                "std": edge.std.cpu(),
                "type": edge.type.name,
            }

        # --- 3. Save only hypothesis 0 (ground truth) ---
        hypotheses_data = {}
        if 0 in self.hypotheses:
            hypothesis = self.hypotheses[0]
            visual_edges = {}
            for edge_key, edge_list in hypothesis.visual_edges.items():
                visual_edges[edge_key] = [
                    {
                        "mean": edge.mean.cpu(),
                        "std": edge.std.cpu(),
                        "type": edge.type.name,
                        "from_comp_id": edge.from_comp_id,
                        "to_comp_id": edge.to_comp_id,
                    }
                    for edge in edge_list
                ]

            hypotheses_data[0] = {
                "component_id": hypothesis.component_id,
                "start_idx": hypothesis.start_idx,
                "visual_edges": visual_edges,
                "visual_adjacency": {k: list(v) for k, v in hypothesis.visual_adjacency.items()},
            }

        return {
            "temp_keyframes": temp_keyframes,
            "odom_edges": odom_edges,
            "hypotheses_data": hypotheses_data,
        }
    
    def load_state(self, hypo_data: dict, db, storage_device: str, device: str, existing_keyframes: dict):
        """Load the hypothesis manager state from saved data.
        Only restores hypothesis 0 (ground truth) and resets all tracking state for a new session.

        Args:
            hypo_data: Dictionary containing saved hypothesis manager state
            db: Database instance (to access atlases)
            storage_device: Device to store tensors
            device: Device for computation
            existing_keyframes: Dictionary mapping keyframe ID to Keyframe objects from database
        """
        # --- 1. Restore temporary keyframes ---
        all_keyframes_map = existing_keyframes.copy()

        for kf_data in hypo_data["temp_keyframes"]:
            atlas = db.get_atlas(kf_data["atlas_id"]) if kf_data["atlas_id"] is not None else None

            kf = Keyframe(
                pose_mu=kf_data["pose_mu"].to(storage_device) if kf_data["pose_mu"] is not None else None,
                pose_std=kf_data["pose_std"].to(storage_device) if kf_data["pose_std"] is not None else None,
                pose_weights=kf_data["pose_weights"].to(storage_device) if kf_data["pose_weights"] is not None else None,
                atlas=atlas,
                timestamp=kf_data["timestamp"],
                temporary=kf_data["temporary"],
                last_pgo_step=kf_data["last_pgo_step"],
            )

            # Manually set the ID to match the saved one
            kf.id = kf_data["id"]

            all_keyframes_map[kf.id] = kf

        # --- 2. Restore nodes (both from database and temporary) ---
        self.nodes.clear()
        for kf_id, kf in all_keyframes_map.items():
            self.nodes[kf_id] = kf

        # --- 3. Restore odometry edges ---
        self.odom_edges.clear()
        for edge_key, edge_data in hypo_data["odom_edges"].items():
            edge = Edge(
                mean=edge_data["mean"].to(device),
                std=edge_data["std"].to(device),
                type=EdgeType[edge_data["type"]],
            )
            self.odom_edges[edge_key] = edge

        # --- 4. Restore only hypothesis 0 (ground truth) ---
        self.hypotheses.clear()
        if "0" in hypo_data["hypotheses_data"] or 0 in hypo_data["hypotheses_data"]:
            # Handle both string and int keys (pickle may serialize differently)
            hypo_key = "0" if "0" in hypo_data["hypotheses_data"] else 0
            hypo_data_item = hypo_data["hypotheses_data"][hypo_key]

            hypothesis = Hypothesis(
                component_id=0,  # Always restore as hypothesis 0
                start_idx=hypo_data_item["start_idx"],
            )

            # Restore visual edges
            for edge_key, edge_list_data in hypo_data_item["visual_edges"].items():
                for edge_data in edge_list_data:
                    edge = VisualEdge(
                        mean=edge_data["mean"].to(device),
                        std=edge_data["std"].to(device),
                        type=EdgeType[edge_data["type"]],
                        from_comp_id=edge_data["from_comp_id"],
                        to_comp_id=edge_data["to_comp_id"],
                    )
                    hypothesis.visual_edges.setdefault(edge_key, []).append(edge)

            # Restore visual adjacency
            for node_id, neighbors in hypo_data_item["visual_adjacency"].items():
                hypothesis.visual_adjacency[node_id] = set(neighbors)

            self.hypotheses[0] = hypothesis

        # --- 5. Reset all tracking state metadata for new session ---
        # Don't restore old tracking state - start fresh
        self.reset_tracking_state()

        # Ensure hypothesis 0 exists
        if 0 not in self.hypotheses:
            self.create_hypothesis_branch(0, 0)
