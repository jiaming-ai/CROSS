import bisect
import math
from cross.visualization.viz_graph import visualize_pose_graph
from cross.utils.profile import timeit
import pypose as pp
from typing import Tuple, List, Dict, Set, Optional, Any
import torch
import dataclasses
from dataclasses import dataclass, field
from loguru import logger
import collections
import threading
import numpy as np

from cross.core.types import Keyframe, Edge, VisualEdge, EdgeType, unpacked
from cross.utils.lie_tensor import project_SE3, normalize_se3
from cross.utils.lie_tensor import project_SE3, normalize_SE3
from cross.core.pgo import (
    PoseGraph,
    as_se3,
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
        self.cfg = cfg
        self._adopt_counter = (None, 0)

        # ========== Core Data Structures ==========
        self.dist: Tuple[pp.LieTensor, pp.LieTensor, torch.Tensor] = None
        self.nodes: Dict[int, Keyframe] = {}  # The first hypothesis (component 0) is the base reality
        self.odom_edges: Dict[Tuple[int, int], Edge] = {}  # odom edges are always from previous kf to current kf
        self.hypotheses: Dict[int, Hypothesis] = {}
        self.system = system
        self.device = getattr(system, "state_device", None) or getattr(system, "device", "cuda")
        self.disappearance_counts = collections.defaultdict(int)
        # Graph-lock to guard structural reads/writes across threads (nodes/edges/adjacency)
        # Use re-entrant lock since some operations call other locked methods.
        self.graph_lock = threading.RLock()
        # incremented whenever keyframe poses of hypothesis 0 move (PGO, merges): position indexes rebuild on change
        self.pose_epoch = 0
        # incremented whenever the structure of hypothesis 0 is replaced (a merge, a promoted hypothesis): a background
        # optimisation (cross.core.async_pgo) computed on an earlier structure is discarded
        self.graph_epoch = 0

        # ========== Component Lifecycle Metadata ==========
        self.n_components = n_components
        self.chart_aware = cfg.chart_aware
        self.schmidt_map_geometry = cfg.schmidt_map_geometry
        self.map_geometry_basis = cfg.map_geometry_basis
        if self.map_geometry_basis not in {'epoch','factor'}:
            raise ValueError('Unknown shared map geometry basis')
        if self.map_geometry_basis == 'factor' and not self.schmidt_map_geometry:
            raise ValueError('Persistent factor geometry requires Schmidt map geometry')
        if self.schmidt_map_geometry and (not cfg.conditional_sources or not self.chart_aware):
            raise ValueError('Schmidt map geometry requires conditional sources and coordinate charts')
        if self.chart_aware and not cfg.session_recovery:
            raise ValueError("Chart-aware mapping requires historical-support recovery")
        if self.chart_aware and cfg.no_pgo_for_lc:
            raise ValueError("Chart-aware commitment requires graph optimization")
        self.component_charts = torch.zeros(n_components, dtype=torch.long, device=self.device)
        self.component_generations = [0] * n_components
        self.next_chart_id = 0
        # Experimental conditional pose/source filter. Activated explicitly by
        # a caller supplying conditional geometry and source-aware factors;
        # the existing RGB-D/monocular adapters keep the legacy path.
        self.source_states = None
        self.saved_source_belief = None
        self.last_conditional_audit = []

        # ========== Evidence Tracking (LLR & Windowing) ==========
        self.llr_hist_length = cfg.llr_hist_length
        self.llr_bias = cfg.llr_bias
        self.filter_process_std = cfg.filter_process_std
        self.motion_std_accumulation = cfg.motion_std_accumulation
        self.h0_informative_only = cfg.h0_informative_only
        self.reset_evidence_on_slot_reuse = cfg.reset_evidence_on_slot_reuse
        self.unmatched_evidence = cfg.unmatched_evidence
        self.unmatched_miss_margin = cfg.unmatched_miss_margin
        self.comp0_informative = None

        # ========== Active Distribution ==========
        self.active_dist_threshold = cfg.active_dist_threshold

        # ========== Birth: Free → Tracking (Unrealized) ==========
        self.alignment_threshold = cfg.alignment_threshold
        self.tracking_active_threshold = cfg.tracking_active_threshold
        self.tracking_floor_weight = cfg.tracking_floor_weight

        # ========== Realize: Tracking (Unrealized) → Tracking (Realized) ==========
        self.realize_sum_thresh = cfg.realize_sum_thresh
        self.realize_hitrate_thresh = cfg.realize_hitrate_thresh
        self.realize_min_frames = cfg.realize_min_frames

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
        from cross.core.reference_support import ReferenceSupport
        self.reference_support = ReferenceSupport(n_components, self.llr_hist_length,
                                                  self.detect_overlap_hitrate_thresh, cfg.session_recovery,
                                                  track_after_anchor=self.chart_aware)
        self.detect_min_frames = cfg.detect_min_frames
        self.detect_llr_cap = cfg.detect_llr_cap
        self.detect_min_weight = cfg.detect_min_weight
        self.reloc_unique_evidence = cfg.reloc_unique_evidence
        self.reloc_unique_min_dist = cfg.reloc_unique_min_dist
        self.reloc_min_frames = cfg.reloc_min_frames
        self.h0_innovation_gate = cfg.h0_innovation_gate
        self.reloc_association_evidence = cfg.reloc_association_evidence
        self.reloc_detection_prob = cfg.reloc_detection_prob
        self.reloc_consistency_nats = cfg.reloc_consistency_nats
        self.reloc_min_verified_frames = cfg.reloc_min_verified_frames
        self.reloc_realize_nats = cfg.reloc_realize_nats
        if self.reloc_association_evidence and not (cfg.session_recovery and cfg.chart_aware):
            raise ValueError("Association evidence requires session_recovery with chart_aware")
        self.detect_reject_cooldown_steps = cfg.detect_reject_cooldown_steps
        self.strong_pass_frames = int(getattr(cfg, "strong_pass_frames", 0))
        self.strong_pass_min_refs = int(getattr(cfg, "strong_pass_min_refs", 2))
        self.strong_pass_min_covis = float(getattr(cfg, "strong_pass_min_covis", 0.3))
        self.verify_outlier_sigma = cfg.verify_outlier_sigma
        self.verify_max_outlier_frac = cfg.verify_max_outlier_frac

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

    def odom_adjacency(self) -> Dict[int, Set[int]]:
        """Undirected adjacency of the odometry edges (planners), rebuilt when the odometry edges change."""
        key = (getattr(self, "odom_edges_version", 0), len(self.odom_edges))
        cache = getattr(self, "_odom_adj_cache", None)
        if cache is None or cache[0] != key:
            adj: Dict[int, Set[int]] = {}
            for (a, b) in list(self.odom_edges.keys()):
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
            self._odom_adj_cache = (key, adj)
        return self._odom_adj_cache[1]

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

    def latest_kf_id_before(self, step: int):
        """Id of the newest permanent keyframe created before processing step `step` (None if there is none)."""
        # NOTE: keyframes record their creation step in `step_created`; the previous fallback compared keyframe *ids*
        # with step counts, which made RetrievalConfig.recent_window_steps inert after the first ~200 steps.
        ids = [kf.id for kf in self.nodes.values() if not kf.temporary and getattr(kf, "step_created", kf.id) < step]
        return max(ids) if ids else None

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

    def start_tracking_chart(self):
        """A restarted trajectory has an independent gauge until commitment."""
        if not self.chart_aware:
            return
        for component in list(self.hypotheses):
            if component != 0:
                self.remove_hypothesis(component)
        self.component_charts.fill_(-1)
        self.component_charts[0] = self.next_chart_id
        self.next_chart_id += 1

    def get_active_charts(self):
        if not self.chart_aware:
            return None
        charts = self.component_charts.clone()
        inactive = (self.dist[2] < self.active_dist_threshold) | ~self.realized
        inactive[0] = False
        charts[inactive] = -1
        return charts

    def reference_audit(self, component):
        cross_chart = (int(self.component_charts[component]) != int(self.component_charts[0])) if self.chart_aware else None
        return self.reference_support.audit(component, cross_chart=cross_chart,
                                            min_hits=self.reloc_min_verified_frames if self.reloc_association_evidence else None)

    def association_components(self):
        """Components scored by loaded-map association evidence (unjoined session, other chart than h0)."""
        mask = torch.zeros(self.n_components, dtype=torch.bool, device=self.device)
        if self.reloc_association_evidence and self.reference_support.unanchored:
            mask = self.component_charts.to(self.device) != self.component_charts[0].to(self.device)
            mask[0] = False
        return mask

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
        self.reference_support.clear_tracking()
        self.source_states = None
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
        # relocalization evidence: log-likelihood against the best other place (see reloc_unique_evidence)
        self.log_u_hist = torch.zeros(self.n_components, self.llr_hist_length, device=self.device)
        self.log_conf_hist = torch.zeros(self.n_components, self.llr_hist_length, device=self.device)
        # which history slots hold real evidence (component active and past its birth frame)
        self.hist_valid = torch.zeros(self.n_components, self.llr_hist_length, dtype=torch.bool, device=self.device)
        # which valid history slots came from a strong pass (strong_pass_frames)
        self.strong_hist = torch.zeros(self.n_components, self.llr_hist_length, dtype=torch.bool, device=self.device)
        self._pending_strong = torch.zeros(self.n_components, dtype=torch.bool, device=self.device)
        self._lc_reject_until: Dict[int, int] = {}

        # Mark existing hypotheses as realized and give them base TTL
        for comp_id in self.hypotheses.keys():
            if comp_id < self.realized.numel():
                self.realized[comp_id] = True
                self.ttl[comp_id] = self.death_ttl_base

        logger.debug("Reset tracking state metadata to defaults")

    def add_node(self, keyframe: Keyframe):
        """Registers a new keyframe in the system."""
        assert 0 in self.hypotheses, "Hypothesis 0 must exist"
        if self.source_states is not None:
            from cross.core.conditional_pose import ConditionalPose
            keyframe.conditional_poses = [
                ConditionalPose.from_state(state) if state is not None and keyframe.pose_weights[i] > 0 else None
                for i,state in enumerate(self.source_states)]
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
        n_valid = self.hist_valid.sum(dim=1)
        hit_valid = ((self.llr_hist > 0) & self.hist_valid).float().sum(dim=1) / n_valid.clamp(min=1)
        to_realize_mask = unrealized_mask & \
                          (self.last_sum_pos >= self.realize_sum_thresh) & \
                          (hit_valid >= self.realize_hitrate_thresh) & \
                          (self._effective_frames(n_valid) >= self.realize_min_frames)
        association = self.association_components()
        if bool(association.any()):
            # sequential association evidence: signed sum over the valid window and enough verified frames
            signed = (self.log_c_hist * self.hist_valid).sum(dim=1)
            verified_frames = torch.tensor([self.reference_support.audit(k)["supported_frames"]
                                            for k in range(self.n_components)], device=signed.device)
            to_realize_mask = torch.where(association,
                                          unrealized_mask & (signed >= self.reloc_realize_nats)
                                          & (verified_frames >= min(2, self.reloc_min_verified_frames)),
                                          to_realize_mask)

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
        if self.chart_aware and new_comp_id != 0 and start_idx in self.nodes:
            # add_node realizes after the keyframe snapshot was taken. Store
            # the newly realized pose instead of an inactive identity placeholder.
            kf = self.nodes[start_idx]
            kf.pose_mu[new_comp_id] = self.dist[0][new_comp_id]
            kf.pose_std[new_comp_id] = self.dist[1][new_comp_id]
            kf.pose_weights[new_comp_id] = self.dist[2][new_comp_id]
            kf.pose_charts[new_comp_id] = self.component_charts[new_comp_id]
            if self.source_states is not None:
                from cross.core.conditional_pose import ConditionalPose
                kf.conditional_poses[new_comp_id] = ConditionalPose.from_state(self.source_states[new_comp_id])
        # Initialize TTL for realized components
        self.ttl[new_comp_id] = max(int(self.ttl[new_comp_id].item()), self.death_ttl_base)

        return new_comp_id


    def graph_cost_now(self) -> tuple:
        """Diagnostics: cost of the hypothesis-0 graph at the current poses (no optimisation), its factor counts and
        the five most expensive factors."""
        pg = PoseGraph(self, depth=1000, k_hop=2, device=self.device, noise_fn=self.pgo_noise_fn(), skip_fn=self.pgo_skip_fn())
        pg.eval_only = True
        with self.graph_lock:
            pg.construct_for_loop_closure(target_node_id=max(self.nodes.keys()), other_hypothesis_id=0)
        ids = [v.id for v in pg.vertices if v.id in self.nodes]
        pg.solve(optim_node_ids=set(ids) - {min(ids)}, fixed_node_ids={min(ids)})
        return pg.initial_cost, getattr(pg, "n_factors", None), getattr(pg, "factor_costs", None)

    def pgo_noise_fn(self):
        """Calibrated noise model of the verified loop closure for the pose-graph optimisation (None: stored stds)."""
        v = getattr(self.system, "_lc_verifier", None)
        if v is None:
            return None

        def noise(f):
            cov = v.noise.factor_cov_gtsam(f)          # anisotropic visual factor: full covariance (gtsam order)
            return cov if cov is not None else v.noise.factor_sigmas_pypose(f)
        # for visual and odometry factors the result depends only on the factor (measurement, noise metadata) and the
        # calibrated model's parameters: the pose graph may cache it per factor under this key
        noise.cache_types = (EdgeType.VISUAL, EdgeType.ODOMETRY)
        noise.cache_key = lambda: (id(v.noise), dataclasses.astuple(v.noise.cfg))
        return noise

    def pgo_skip_fn(self):
        """Information criterion of the verified loop closure: visual measurements that the odometry chain already
        explained better at the time of the observation do not constrain the graph (None: keep every factor)."""
        v = getattr(self.system, "_lc_verifier", None)
        if v is None:
            return None
        return lambda f: f.type == EdgeType.VISUAL and getattr(f, "informative", True) is False

    def add_edge(
        self,
        id1: int,
        id2: int,
        rel_pose_mean: pp.LieTensor,
        rel_pose_std: pp.LieTensor,
        type: EdgeType,
        from_comp_id: int = 0,
        to_comp_id: int = 0,
        conditional_pose=None,
        meta: Optional[Dict[str, Any]] = None,
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
        if self.source_states is not None and conditional_pose is None:
            raise ValueError('Conditional mapping requires a source model for each raw graph edge')

        # --- Logic for Odometry Edges ---
        # Odometry connects consecutive keyframes. It is added to the global graph,
        # as it represents the continuous motion of the robot in each possible "reality".
        if type == EdgeType.ODOMETRY:
            factor = Edge(rel_pose_mean, rel_pose_std, type) # comp_ids are not used for odom edges
            factor.conditional_pose = conditional_pose
            for k, v in (meta or {}).items():
                setattr(factor, k, v)
            # Structural mutation under lock
            with self.graph_lock:
                self.odom_edges[(id1, id2)] = factor # it's directed edge, from id1 to id2
                self.odom_edges_version = getattr(self, "odom_edges_version", 0) + 1

        # --- Logic for Visual Edges 
        elif type == EdgeType.VISUAL:
            # if to_comp_id is not in the hypotheses, skip
            if to_comp_id not in self.hypotheses:
                logger.debug(f"Visual edge to non-existent hypothesis {to_comp_id} skipped.")
                return
            factor = VisualEdge(
                rel_pose_mean, rel_pose_std, type, from_comp_id, to_comp_id
            )
            factor.conditional_pose = conditional_pose
            for k, v in (meta or {}).items():
                setattr(factor, k, v)
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
        # Raw-factor coordinates must keep their measurement lineage. Retain
        # temporary nodes until noise-preserving composed edges are available.
        if self.schmidt_map_geometry and self.map_geometry_basis == 'factor':
            return
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
                    if self.source_states is not None:
                        from cross.core.conditional_pose import compose
                        pose,model = compose(pred_edge.mean.matrix().double().cpu().numpy(),pred_edge.conditional_pose,
                            succ_edge.mean.matrix().double().cpu().numpy(),succ_edge.conditional_pose,self.source_states[0])
                        bridging_factor.mean = pp.from_matrix(torch.as_tensor(pose,device=self.device,dtype=combined_pose.dtype),pp.SE3_type)
                        bridging_factor.std = pp.se3(torch.as_tensor(np.sqrt(model.geometry_covariance.diagonal().clip(0)),
                                                                  device=self.device,dtype=combined_std.dtype))
                        bridging_factor.conditional_pose = model
                    bridging_factor.n_frames = (getattr(pred_edge, "n_frames", None) or 1) + (getattr(succ_edge, "n_frames", None) or 1)
                    # odometry-fault inflation of the odometry scale guard (lc_verify): the larger of the two
                    fault = max(float(getattr(pred_edge, "odom_fault", 0.0) or 0.0), float(getattr(succ_edge, "odom_fault", 0.0) or 0.0))
                    if fault > 0:
                        bridging_factor.odom_fault = fault
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
            self.odom_edges_version = getattr(self, "odom_edges_version", 0) + 1
            
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
                    if kf.pose_charts is not None:
                        kf.pose_charts[component_id] = -1
                    if kf.conditional_poses is not None:
                        kf.conditional_poses[component_id] = None

            # --- Delete the Hypothesis ---
            del self.hypotheses[component_id]

        # remove from self.dist (tracking dist) and renormalize
        mu, sigma, weights = self.dist
        mu[component_id] = pp.identity_SE3(1, device=mu.device)
        sigma[component_id] = pp.identity_se3(1, device=sigma.device)
        weights[component_id] = 0.0
        weights = weights / weights.sum()
        self.dist = (mu, sigma, weights)
        self._reset_component_evidence(component_id)

    def _reset_component_evidence(self, component_id):
        """A bounded slot is storage, not the identity of a place hypothesis."""
        if self.reset_evidence_on_slot_reuse:
            self.llr_hist[component_id, :] = 0
            self.log_c_hist[component_id, :] = 0
            self.log_conf_hist[component_id, :] = 0
            self.hist_valid[component_id, :] = False
            self.strong_hist[component_id, :] = False
            self.last_sum_pos[component_id] = 0
            self.last_hit_rate[component_id] = 0
        self.reference_support.clear_component(component_id)
        self.component_charts[component_id] = -1
        self.component_generations[component_id] += 1
        if self.source_states is not None:
            self.source_states[component_id] = None

    def _effective_frames(self, n_valid: torch.Tensor, rows=None) -> torch.Tensor:
        """Valid evidence frames, a strong pass counting strong_pass_frames (rows: the components of n_valid)."""
        if self.strong_pass_frames <= 1:
            return n_valid
        strong = (self.strong_hist & self.hist_valid) if rows is None else (self.strong_hist[rows] & self.hist_valid[rows])
        return n_valid + (self.strong_pass_frames - 1) * strong.sum(dim=1)

    def _is_strong(self, proposal: Dict) -> bool:
        return self.strong_pass_frames > 1 and int(proposal.get("strong_refs", 0)) >= self.strong_pass_min_refs

    def initialize_source_filter(self):
        """Start the conditional filter after initializing the tracking poses.

        The current covariance is interpreted as conditional geometry noise.
        This must not be called on a scale-containing marginal covariance.
        Source priors are introduced by named factors once. The input adapter,
        node messages and PGO transport must all support this representation
        before it can be used as an end-to-end monocular mode.
        """
        from cross.core.conditional import SourceState
        if self.dist is None or self.source_states is not None:
            raise ValueError("Initialize source filtering once, after the pose distribution")
        self.source_states = [SourceState(torch.diag(s.tensor().square()).double().cpu().numpy())
                              for s in self.dist[1]]
        if self.saved_source_belief is not None:
            for state in self.source_states:
                state.keys = self.saved_source_belief.keys
                state.mean = self.saved_source_belief.mean.copy()
                state.covariance = self.saved_source_belief.covariance.copy()
                state.prior_variances = self.saved_source_belief.prior_variances.copy()
                state.jacobian = np.zeros((6,len(state.keys)))
                state.seen_factors = self.saved_source_belief.seen_factors


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
        self.source_states = None


    def motion_update(
        self, 
        delta_pose: pp.LieTensor,
        delta_std: pp.LieTensor,
        source_factor=None,
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
        if self.source_states is not None:
            if source_factor is None:
                raise ValueError("Conditional filtering needs the motion factor's source Jacobians")
            from cross.utils.lie_tensor import SE3_Adj
            from cross.core.conditional import SourceState, transport_covariance
            from cross.core.conditional_pose import normalize_mean
            active = torch.where(last_gmm_weights > self.tracking_active_threshold)[0].tolist()
            pending = []
            for component in active:
                state = self.source_states[component]
                if state is None:
                    raise ValueError("Active hypothesis has no conditional source state")
                state, motion_J, offset = state.expand(source_factor)
                twist = torch.as_tensor(offset,device=delta_pose.device,dtype=delta_pose.dtype)
                corrected_delta = normalize_mean(delta_pose @ pp.se3(twist).Exp())
                A = SE3_Adj(corrected_delta.Inv()).double().cpu().numpy()
                covariance = transport_covariance(state.geometry_covariance,A)
                # Keep the inherited std-sum policy for residual geometric
                # errors, after transporting to the new local tangent.
                std = torch.as_tensor(covariance.diagonal().copy()).clamp_min(0).sqrt().numpy()
                noise = delta_std.tensor().double().cpu().numpy()
                covariance += np.diag((std+noise)**2-std**2)
                posterior = SourceState(covariance,state.keys,state.mean,state.covariance,
                    A@state.jacobian+motion_J,state.prior_variances,state.seen_factors)
                marginal = torch.as_tensor(posterior.marginal_covariance().diagonal().copy(),
                                            device=last_gmm_sigma.device,dtype=last_gmm_sigma.dtype)
                pending.append((component,posterior,normalize_mean(last_gmm_mu[component] @ corrected_delta),
                                pp.se3(marginal.clamp_min(0).sqrt())))
            # Validate all components before mutating any live pose/bias state.
            for component,posterior,mean,std in pending:
                self.source_states[component] = posterior
                last_gmm_mu[component],last_gmm_sigma[component] = mean,std
            self.dist = (last_gmm_mu,last_gmm_sigma,last_gmm_weights)
            return
        if source_factor is not None:
            raise ValueError("A source-aware motion factor needs initialize_source_filter()")
        # Update all active components by weight threshold so priors evolve with motion
        active_mask = last_gmm_weights > self.tracking_active_threshold
        last_gmm_mu[active_mask] = normalize_SE3(last_gmm_mu[active_mask] @ delta_pose.unsqueeze(0))
        if self.motion_std_accumulation == "variance":
            last_gmm_sigma[active_mask] = pp.se3((last_gmm_sigma[active_mask].tensor() ** 2 + delta_std.tensor().unsqueeze(0) ** 2) ** 0.5)
        else:
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
        self._pending_strong = torch.zeros(self.n_components, dtype=torch.bool, device=self.device)

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
        self.comp0_informative = None

        proposal_mu = torch.stack([h['pose'] for h in proposal_hypotheses])
        
        # Project poses to the place coordinates of the proposal clustering to calculate meaningful distances.
        projection = getattr(self.system, "place_projection", None)
        current_mu_proj = project_SE3(current_mu[active_comps_mask], projection=projection)
        proposal_mu_proj = project_SE3(proposal_mu, projection=projection)

        # --- Step 2: Greedy Best-First Matching ---
        # Calculate the pairwise distance between every active component and every proposal.
        dist_matrix = torch.cdist(current_mu_proj, proposal_mu_proj)
        pcfg = getattr(getattr(self.system, "config", None), "mapping", None)
        gate = pcfg.projection.vertical_gate if pcfg is not None else 0.0
        if gate > 0:
            # a proposal more than the gate away vertically is another place (or a scale blow-up), not this hypothesis
            dv = (self.system.vertical_offsets(current_mu[active_comps_mask])[:, None]
                  - self.system.vertical_offsets(proposal_mu)[None, :])
            dist_matrix.masked_fill_(dv.abs().to(dist_matrix.device) > gate, float("inf"))
        if self.chart_aware:
            proposal_charts = torch.tensor([h['chart_id'] for h in proposal_hypotheses],
                                           device=self.device, dtype=torch.long)
            compatible = self.component_charts[active_comp_indices, None] == proposal_charts[None, :]
            dist_matrix.masked_fill_(~compatible, float('inf'))
        self.last_alignment_audit = [dict(
            sources=[dict(keyframe_id=int(k), source_component=int(c),
                          loaded=k in getattr(self.system, "loaded_node_ids", ()))
                     for k, c in proposal["source_indices"]],
            pose=proposal["pose"].tensor().detach().cpu().tolist(),
            score=float(proposal["score"]), nearest_prior_distance=float(dist_matrix[:, i].min()),
            component=None, action="capacity_rejected")
            for i, proposal in enumerate(proposal_hypotheses)]
        if self.chart_aware:
            for audit, proposal in zip(self.last_alignment_audit, proposal_hypotheses):
                audit['chart_id'] = proposal['chart_id']
                if not math.isfinite(audit['nearest_prior_distance']):
                    audit['nearest_prior_distance'] = None
        # verified loop closure: a proposal that failed the prior-consistency test against hypothesis 0 must not be
        # matched to (fused into) hypothesis 0; it may still feed or spawn another hypothesis
        h0_flags = [h.get("h0_ok") for h in proposal_hypotheses]
        row0 = int((active_comp_indices == 0).nonzero()[0].item()) if (active_comp_indices == 0).any() else None
        if self.h0_innovation_gate > 0 and row0 is not None:
            # chi-square gate against hypothesis 0 for proposals the verified loop closure could not test
            prior_var = current_std[0].tensor() ** 2 + float(self.filter_process_std) ** 2
            for j, h in enumerate(proposal_hypotheses):
                if h0_flags[j] is not None or not math.isfinite(float(dist_matrix[row0, j])):
                    continue
                residual = (current_mu[0].Inv() @ h['pose']).Log().tensor()
                maha = float((residual ** 2 / (prior_var + h['std'].tensor() ** 2).clamp_min(1e-12)).sum())
                if maha > self.h0_innovation_gate:
                    h0_flags[j] = False
        dropped = set()
        if row0 is not None:
            for j, f in enumerate(h0_flags):
                if f is False:
                    if float(dist_matrix[row0, j]) < self.alignment_threshold:
                        # inconsistent with hypothesis 0 yet within its alignment radius: an outlier measurement of
                        # the same place, not a new place -- neither fused nor born (it would spawn a hypothesis
                        # seeded by the outlier that can take over the belief)
                        dropped.add(j)
                        dist_matrix[:, j] = float("inf")
                    else:
                        dist_matrix[row0, j] = float("inf")
        if dropped:
            v = getattr(self.system, "_lc_verifier", None)
            if v is not None:
                v.stats["outliers_dropped"] = v.stats.get("outliers_dropped", 0) + len(dropped)
            logger.debug(f"proposals {sorted(dropped)} dropped as outliers of hypothesis 0")

        # Prepare tensors for the new, aligned GMM. Default to low-confidence values.
        num_components = current_mu.shape[0]
        aligned_mu = pp.identity_SE3(num_components, device=current_mu.device)
        aligned_sigma = pp.identity_se3(num_components, device=current_mu.device) # High uncertainty
        aligned_weights = torch.zeros(num_components, device=current_mu.device)

        aligned_confidence = torch.zeros(num_components, device=current_mu.device)
        self.aligned_conditional_models = {} if self.source_states is not None else None

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
            self.last_alignment_audit[proposal_idx].update(component=true_comp_idx, action="matched")
            
            # --- A match is found: update the aligned GMM ---
            proposal = proposal_hypotheses[proposal_idx]
            self._pending_strong[true_comp_idx] = self._is_strong(proposal)
            if true_comp_idx == 0:
                self.comp0_informative = proposal.get('informative', True)
            aligned_mu[true_comp_idx] = proposal['pose']
            aligned_sigma[true_comp_idx] = proposal['std']
            aligned_weights[true_comp_idx] = proposal['score']
            aligned_confidence[true_comp_idx] = proposal['score']
            if self.aligned_conditional_models is not None:
                self.aligned_conditional_models[true_comp_idx] = proposal['conditional_pose']

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
        unmatched_proposals_indices = set(range(len(proposal_hypotheses))) - matched_proposals - dropped
        for proposal_idx in unmatched_proposals_indices:
            if h0_flags[proposal_idx] is True:
                # consistent with hypothesis 0 along the odometry chain but beyond the alignment radius (drift):
                # a loop closure of hypothesis 0, not a new hypothesis.  Its measurements become hypothesis-0 edges
                # and the pose-graph optimisation moves the belief; no component is born for it.
                for kf_id, source_comp_id in proposal_hypotheses[proposal_idx]['source_indices']:
                    if source_comp_id == 0:
                        edge_mapping[kf_id] = (0, 0)
                logger.debug(f"proposal {proposal_idx} is a verified loop closure of hypothesis 0 (no birth)")
                continue
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
            self._reset_component_evidence(new_comp_idx)
            self.last_alignment_audit[proposal_idx].update(component=new_comp_idx, action="born")
            proposal = proposal_hypotheses[proposal_idx]
            self._pending_strong[new_comp_idx] = self._is_strong(proposal)
            
            aligned_mu[new_comp_idx] = proposal['pose']
            self.component_charts[new_comp_idx] = proposal['chart_id'] if self.chart_aware else 0
            aligned_sigma[new_comp_idx] = proposal['std']
            aligned_weights[new_comp_idx] = proposal['score']
            aligned_confidence[new_comp_idx] = proposal['score']
            if self.aligned_conditional_models is not None:
                self.aligned_conditional_models[new_comp_idx] = proposal['conditional_pose']

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
        source_factors=None,
        source_covariances=None,
        variance_update_mask: Optional[torch.Tensor] = None,  # K bool: components whose skipped update keeps the fused variance
    ):
        """
        Computes the final distribution p = alpha * (p_proposal * p_prior) + (1 - alpha) * p_proposal.

        Experimental source_factors is a dict from aligned component ID to a
        conditional SourceFactor. Proposal stds then describe conditional
        geometry only. Source priors and cross-covariances stay in source_states;
        None/missing means no new geometric factor for that component.
        """
        if (source_factors is None) != (self.source_states is None):
            raise ValueError("Conditional states and source-aware observation factors must be supplied together")
        if source_covariances is not None and source_factors is None:
            raise ValueError('Conditional covariances require named source factors')
        if source_factors is not None:
            if any(not isinstance(i,int) or i < 0 or i >= self.n_components for i in source_factors):
                raise ValueError("Source factors must name valid aligned component IDs")
            if any(factor is not None and factor.factor_id is None for factor in source_factors.values()):
                raise ValueError("Conditional observation factors need stable geometric identities")
            duplicates = [factor.factor_id is not None and self.source_states[i] is not None
                          and factor.factor_id in self.source_states[i].seen_factors
                          for i,factor in source_factors.items() if factor is not None and not self.newborn[i]]
            if duplicates and any(duplicates):
                if not all(duplicates) or any(bool(self.newborn[i]) for i in source_factors):
                    raise ValueError("Remove reused factors before constructing a mixed fresh/replayed observation")
                # Exact delivery replay must not alter pose, bias, weights,
                # evidence-window position or TTL. It is not another frame of
                # independent support for delayed commitment.
                self.last_conditional_audit = [dict(component=i,duplicate=True) for i in source_factors]
                return
        # candidate poses are compositions of stored keyframe poses and relative poses: keep them on SE(3)
        proposal_mu = normalize_SE3(proposal_mu)
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
        proc_std = torch.full((6,), float(self.filter_process_std),
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
        logger.debug(f"filter comp0: prior std {[round(float(x), 4) for x in prior_var_diag[0].sqrt()]} proposal std "
                     f"{[round(float(x), 4) for x in proposal_var_diag[0].sqrt()]} residual {[round(float(x), 4) for x in r[0]]} "
                     f"delta {[round(float(x), 4) for x in prod_log_mu[0]]}")

        # Gaussian-overlap factor  c_k = N(r_k ; 0, S_k),  with S_k = Σ1_k + Σ2_k  (all diag)
        S_diag = (proposal_var_diag + prior_var_diag).clamp_min(eps)   # (C,6)
        inv_S_diag = 1.0 / S_diag
        maha = (r * r * inv_S_diag).sum(dim=-1)                            # (C,)
        log_det_S = torch.log(S_diag).sum(dim=-1)                          # (C,)
        log_c = -0.5 * (maha + (6.0 * math.log(2.0 * math.pi) + log_det_S))
        if self.unmatched_evidence == "miss":
            # a component that no proposal was aligned to (keep-alive: its own mean, confidence 0) explains none of the
            # observations.  Scoring that self-match with zero residual gave a confident component (tight odometry) a
            # density no real measurement can reach, so a kidnapped hypothesis 0 out-scored every map hypothesis and no
            # relocalization session ever converged.  Its evidence is that of the weakest supported component minus a
            # margin: a supported hypothesis always beats an unsupported one, and aliased support must still persist over
            # the realisation window and pass the merge verification.
            unmatched = proposal_confidence <= 0
            supported = (~unmatched) & (prior_weights > self.tracking_active_threshold)
            # hypothesis 0 is exempt while it is localized: in a mapping session, or once a relocalization session is
            # anchored to the map by a verified map edge, it is supported by its own odometry chain even when the only
            # proposals of an observation are aliased places (they spawn hypotheses of their own)
            if not self._h0_unlocalized() and unmatched.numel() > 0:
                unmatched = unmatched.clone()
                unmatched[0] = False
            if bool(unmatched.any()) and bool(supported.any()):
                floor = log_c[supported].min() - float(self.unmatched_miss_margin)
                log_c = torch.where(unmatched, torch.minimum(log_c, floor), log_c)

        pending_sources = None
        conditional_newborns = {}
        conditional_evidence_mask = None
        if source_factors is not None:
            from cross.core.conditional import transport_covariance
            from cross.core.conditional_pose import ConditionalPose, right_jacobian, normalize_mean, residual_product, retract_frozen_response
            pending_sources = list(self.source_states)
            conditional_evidence_mask = torch.zeros_like(currently_tracking)
            self.last_conditional_audit = []
            for component in range(self.n_components):
                factor = source_factors.get(component)
                state = self.source_states[component]
                if factor is None:
                    if state is not None:
                        prod_mu[component] = prior_mu[component]
                        prod_var_diag[component] = prior_var_diag_noQ[component]
                    continue
                R = np.diag(proposal_std_diag[component].tensor().square().double().cpu().numpy().clip(1e-9))
                if source_covariances is not None:
                    R = np.asarray(source_covariances[component], dtype=np.float64)
                if bool(self.newborn[component]):
                    # The newborn has no pose prior in its new chart. Reuse
                    # the committed bias belief once; seed x conditionally.
                    state, offset = self.source_states[0].seed(factor,R)
                    mean = normalize_mean(proposal_mu[component] @ pp.se3(torch.as_tensor(offset,device=self.device,dtype=prior_mu.dtype)).Exp())
                    T = right_jacobian(offset)
                    state.geometry_covariance = transport_covariance(state.geometry_covariance,T)
                    state.jacobian = retract_frozen_response(state.keys,state.jacobian,offset)
                    variance = torch.as_tensor(state.marginal_covariance().diagonal().copy(),device=self.device,dtype=prod_var_diag.dtype)
                    conditional_newborns[component] = (mean,variance)
                    audit = dict(component=component,newborn=True,source_count=len(state.keys))
                else:
                    if state is None:
                        raise ValueError("A source factor has no live hypothesis prior")
                    # Evaluate H(x|b) at this branch's bias mean, then express
                    # its right-tangent noise/response in the prior log chart.
                    observation,model,state = ConditionalPose(R,factor).at(
                        proposal_mu[component].matrix().double().cpu().numpy(),state)
                    observed = pp.from_matrix(torch.as_tensor(observation,device=self.device,dtype=prior_mu.dtype),pp.SE3_type)
                    residual = (prior_mu[component].Inv()@observed).Log().tensor().double().cpu().numpy()
                    source_before = state.jacobian.copy()
                    result = residual_product(state,residual,model.geometry_covariance,model.factor,
                        np.diag(Q[0].double().cpu().numpy()),
                        frozen_keys=tuple(k for k in state.keys if k.startswith('geometry:'))
                                    if self.schmidt_map_geometry else ())
                    state = result.state
                    mean = normalize_mean(prior_mu[component] @ pp.se3(torch.as_tensor(result.pose_offset,device=self.device,dtype=prior_mu.dtype)).Exp())
                    T = right_jacobian(result.pose_offset)
                    state.geometry_covariance = transport_covariance(state.geometry_covariance,T)
                    state.jacobian = retract_frozen_response(state.keys,state.jacobian,result.pose_offset,source_before)
                    variance = torch.as_tensor(state.marginal_covariance().diagonal().copy(),device=self.device,dtype=prod_var_diag.dtype)
                    log_c[component] = result.log_overlap
                    conditional_evidence_mask[component] = True
                    audit = dict(component=component,newborn=False,source_count=len(state.keys),
                                 source_keys=list(state.keys),
                                 log_overlap=result.log_overlap,
                                 bias_mean=state.mean.tolist(),bias_std=np.sqrt(state.covariance.diagonal().clip(0)).tolist(),
                                 innovation_diagonal=result.innovation_covariance.diagonal().tolist())
                pending_sources[component] = state
                prod_mu[component],prod_var_diag[component] = mean,variance
                self.last_conditional_audit.append(audit)

        #################
        # Evidence tracking with LLR + bias
        #################
        eps_c = 1e-12
        llr_bias = self.llr_bias
        conf_ratio = (proposal_confidence_clamped + eps_c) / (proposal_confidence_clamped[0] + eps_c)
        conf_ratio = torch.clamp(conf_ratio, max=1.0) # low confidence penalize hypothesis, while high confidence neutral to avoid false positives
        llr = (log_c - log_c[0]) + torch.log(conf_ratio) + llr_bias
        association = self.association_components()
        if bool(association.any()):
            verified = torch.tensor([bool(self.reference_support.current(k)) for k in range(self.n_components)],
                                    device=llr.device)
            miss = math.log(max(1.0 - self.reloc_detection_prob, 1e-6))
            assoc_llr = torch.where(verified, self.reloc_consistency_nats - 0.5 * maha,
                                    torch.full_like(llr, miss))
            llr = torch.where(association, assoc_llr, llr)

        # keep positive support
        pos = torch.relu(llr)

        # GATE: Only accumulate evidence for actively tracked components
        # Inactive components (weight ≈ 0) should not accumulate spurious evidence
        # from identity-pose overlap with identity proposals
        # A newborn that evicts an active slot has no prior of its own yet.
        # Comparing it to the evicted identity would be spurious evidence.
        active_tracking_mask = (prior_weights > self.tracking_active_threshold) & ~self.newborn
        if conditional_evidence_mask is not None:
            active_tracking_mask &= conditional_evidence_mask
        pos = torch.where(active_tracking_mask, pos, torch.zeros_like(pos))

        # Debug history: log individual LLR components (for tuning/visualization)
        log_c_rel = (log_c - log_c[0])  # relative log-likelihood
        log_conf = torch.log(conf_ratio)  # log confidence ratio
        if bool(association.any()):
            log_c_rel = torch.where(association, llr, log_c_rel)
            # a missed map detection is not low geometric confidence
            log_conf = torch.where(association & ~verified, torch.zeros_like(log_conf), log_conf)
        self.log_c_hist[:, self.llr_hist_ptr] = torch.where(active_tracking_mask, log_c_rel, torch.zeros_like(log_c_rel))
        if self.reloc_unique_evidence:
            # evidence of each component against the best competing place: the active components farther than
            # reloc_unique_min_dist (hypothesis 0 included; unsupported ones carry their miss score)
            pos_t = prior_mu.tensor()[:, :3]
            far = (torch.cdist(pos_t, pos_t) > self.reloc_unique_min_dist) & active_tracking_mask.unsqueeze(0)
            far.fill_diagonal_(False)
            alt = torch.where(far, log_c.unsqueeze(0).expand_as(far), torch.full_like(far, float("-inf"), dtype=log_c.dtype)).max(dim=1).values
            alt = torch.where(torch.isfinite(alt), alt, log_c[0].expand_as(alt))
            log_u_rel = log_c - alt
            self.log_u_hist[:, self.llr_hist_ptr] = torch.where(active_tracking_mask, log_u_rel, torch.zeros_like(log_u_rel))
        self.log_conf_hist[:, self.llr_hist_ptr] = torch.where(active_tracking_mask, log_conf, torch.zeros_like(log_conf))
        # a newborn is seeded with its proposal (zero residual): its birth frame is not evidence
        self.hist_valid[:, self.llr_hist_ptr] = active_tracking_mask & ~self.newborn
        self.strong_hist[:, self.llr_hist_ptr] = self._pending_strong.to(self.strong_hist.device) & self.hist_valid[:, self.llr_hist_ptr]
        self._pending_strong = torch.zeros_like(self._pending_strong)

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
            for component,(mean,variance) in conditional_newborns.items():
                prod_mu[component],prod_var_diag[component] = mean,variance

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
                    self.hist_valid[dead_full_mask] = False
                    self.strong_hist[dead_full_mask] = False
                    self.log_c_hist[dead_full_mask] = 0.0
                    self.log_u_hist[dead_full_mask] = 0.0
                    self.log_conf_hist[dead_full_mask] = 0.0

                    # Remove realized hypothesis branches for dead comps (data-structure loop)
                    for comp_idx in dead_indices.tolist():
                        if comp_idx in self.hypotheses:
                            self.remove_hypothesis(comp_idx)
                        else:
                            self._reset_component_evidence(comp_idx)
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
                var_revert = revert_mask
                if variance_update_mask is not None and source_factors is None:
                    var_revert = revert_mask & ~variance_update_mask.to(device=revert_mask.device, dtype=torch.bool)
                prod_var_diag[var_revert] = prior_var_diag_noQ[var_revert]
                if pending_sources is not None:
                    for component in torch.where(revert_mask)[0].tolist():
                        retained = self.source_states[component]
                        if retained is not None:
                            retained = retained.copy()
                            # The factor was already used for place evidence,
                            # even though pose/bias updates were explicitly gated.
                            retained.seen_factors = pending_sources[component].seen_factors
                        pending_sources[component] = retained

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
        prod_mu = normalize_SE3(prod_mu)

        if torch.isnan(prod_weights).any():
            logger.warning(f"prod_weights is nan: {prod_weights}")
        if source_factors is None:
            prod_mu = normalize_se3(prod_mu)
        self.dist = (prod_mu, prod_std_diag, prod_weights)
        if pending_sources is not None:
            for component,state in enumerate(pending_sources):
                if component == 0 or bool(self.ttl[component] > 0):
                    self.source_states[component] = state
                else:
                    self.source_states[component] = None
        
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

        # Keep the actual decision evidence available for monocular audits.
        # These values do not change the inherited commitment policy.
        self.last_loop_audit = {"realized": self.realized.detach().cpu().tolist(),
                                "weights": self.dist[2].detach().cpu().tolist(),
                                "realization_positive_llr_sum": self.last_sum_pos.detach().cpu().tolist(),
                                "realization_positive_llr_hit_rate": self.last_hit_rate.detach().cpu().tolist(),
                                "realization_thresholds": dict(sum=self.realize_sum_thresh,
                                                               hit_rate=self.realize_hitrate_thresh),
                                "relative_overlap_history": self.log_c_hist.detach().cpu().tolist(),
                                "relative_confidence_history": self.log_conf_hist.detach().cpu().tolist(),
                                "historical_support": [self.reference_audit(i)
                                                       for i in range(self.n_components)],
                                "candidates": []}
        if self.chart_aware:
            self.last_loop_audit["component_charts"] = self.component_charts.tolist()
        # 1) Inter-hypothesis LC detection (exclude comp 0)
        # only consider active and realized components
        active_mask = torch.logical_and(self.dist[2] > self.active_dist_threshold, self.realized)[1:]
        # relocalization (hypothesis 0 not anchored to the stored map): evidence against the best other place
        reloc = self.reloc_unique_evidence and self._h0_unlocalized()
        ev_hist = self.log_u_hist if reloc else self.log_c_hist
        min_frames = max(self.detect_min_frames, self.reloc_min_frames) if reloc else self.detect_min_frames
        # diagnostics: the dominant component is not hypothesis 0 -> why is it not merged?  (every observation step)
        kd = int(torch.argmax(self.dist[2]).item())
        if kd != 0 and float(self.dist[2][kd]) >= 0.5:
            valid_d = self.hist_valid[kd]; nv = int(valid_d.sum())
            net = float((ev_hist[kd].clamp(-self.detect_llr_cap, self.detect_llr_cap) * valid_d).sum())
            hit = float(((ev_hist[kd] + self.detect_overlap_rel_margin > 0) & valid_d).float().sum() / max(nv, 1))
            chit = float(((self.log_conf_hist[kd] + self.detect_conf_rel_margin > 0) & valid_d).float().sum() / max(nv, 1))
            dist_d = float(torch.norm(self.dist[0][kd].tensor()[:3] - self.dist[0][0].tensor()[:3]))
            logger.debug(f"LC gate: comp {kd} w={float(self.dist[2][kd]):.2f} realized={bool(self.realized[kd])} ttl={int(self.ttl[kd])} "
                         f"n_valid={nv} net_llr={net:.2f} hit={hit:.2f} conf_hit={chit:.2f} dist={dist_d:.1f} "
                         f"sum_pos={float(self.last_sum_pos[kd]) if self.last_sum_pos is not None else -1:.2f} "
                         f"hit_rate={float(self.last_hit_rate[kd]) if self.last_hit_rate is not None else -1:.2f} "
                         f"cooldown={self._lc_reject_until.get(kd, -1) >= int(self.step_counter)} reloc={reloc}")
        if active_mask.any():
            realized_ids = torch.nonzero(active_mask, as_tuple=False).squeeze(-1) + 1 # +1 because we exclude comp 0

            # Stack and index using realized IDs
            log_c_rel_realized = ev_hist[realized_ids, :] # (C, L)
            log_conf_rel_realized = self.log_conf_hist[realized_ids, :] # (C, L)

            valid = self.hist_valid[realized_ids, :]                                  # (C, L) real evidence only
            n_valid = valid.sum(dim=1)
            rel = log_c_rel_realized + self.detect_overlap_rel_margin
            # net LLR over the valid frames (clamped per frame): negative frames count against the candidate
            log_c_pos_sum = (log_c_rel_realized.clamp(-self.detect_llr_cap, self.detect_llr_cap) * valid).sum(dim=1)
            log_c_pos_hit_rate = ((rel > 0) & valid).float().sum(dim=1) / n_valid.clamp(min=1)
            log_conf_hit_rate = ((log_conf_rel_realized + self.detect_conf_rel_margin > 0) & valid).float().sum(dim=1) / n_valid.clamp(min=1)
            enough = self._effective_frames(n_valid, realized_ids) >= min_frames
            # the belief must actually have moved to the candidate, and a candidate whose merge was just rejected
            # (geometric verification) is ignored for a while
            weights = self.dist[2][realized_ids]
            heavy = weights >= self.detect_min_weight
            not_rejected = torch.tensor([self._lc_reject_until.get(int(c), -1) < int(self.step_counter) for c in realized_ids.tolist()],
                                        device=weights.device, dtype=torch.bool)

            # Reject LC if the candidate is too close to the current pose, i.e. likely to be a false positive
            current_pose = self.dist[0][0][:3].unsqueeze(0) # (1, 3)
            candidate_poses = self.dist[0][realized_ids][:, :3] # (C, 3)
            distances = torch.norm(candidate_poses - current_pose, dim=1) # (C,)

            close_mask = distances < 3 # 
            reference_audits = [self.reference_audit(i) for i in realized_ids.tolist()]
            cross_chart = (self.component_charts[realized_ids] != self.component_charts[0]) if self.chart_aware else torch.tensor(
                [a["unanchored_reference_candidate"] for a in reference_audits], device=distances.device, dtype=torch.bool)
            reference_supported = torch.tensor([a["eligible"] for a in reference_audits],
                                               device=distances.device, dtype=torch.bool)
            separation_gate = torch.where(cross_chart, reference_supported, ~close_mask)

            detected_mask = (log_c_pos_sum >= self.detect_overlap_sum_thresh) \
                & (log_c_pos_hit_rate >= self.detect_overlap_hitrate_thresh) \
                & (log_conf_hit_rate >= self.detect_conf_hitrate_thresh) \
                & separation_gate & enough & heavy & not_rejected

            for j, component in enumerate(realized_ids.tolist()):
                self.last_loop_audit["candidates"].append({
                    "component": component,
                    "overlap_sum": float(log_c_pos_sum[j]),
                    "overlap_hit_rate": float(log_c_pos_hit_rate[j]),
                    "confidence_hit_rate": float(log_conf_hit_rate[j]),
                    "distance_m": None if self.chart_aware and bool(cross_chart[j]) else float(distances[j]),
                    "passes_overlap_sum": bool(log_c_pos_sum[j] >= self.detect_overlap_sum_thresh),
                    "passes_overlap_hit_rate": bool(log_c_pos_hit_rate[j] >= self.detect_overlap_hitrate_thresh),
                    "passes_confidence": bool(log_conf_hit_rate[j] >= self.detect_conf_hitrate_thresh),
                    "passes_distance": None if self.chart_aware and bool(cross_chart[j]) else bool(~close_mask[j]),
                    "reference_support": reference_audits[j],
                    "separation_rule": "historical_support" if bool(cross_chart[j]) else "legacy_three_meters",
                    "passes_separation_gate": bool(separation_gate[j]),
                    "valid_frames": int(n_valid[j]),
                    "weight": float(weights[j]),
                    "detected": bool(detected_mask[j]),
                })

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
        self, hypo_id: int, target_node_id: Optional[int] = None, apply: bool = True, global_opt: bool = False,
        window_ref: Optional[int] = None,
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
            window_ref: the oldest keyframe of the loop edges that trigger this optimisation; with
                mapping.loop_closure.pgo_window_min_nodes set, a session graph of that size is optimised only from
                window_ref (minus pgo_window_margin keyframes) on, older keyframes that share a factor with it fixed
        
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

        if self.no_pgo_for_lc and hypo_id != 0:
            # instead of merging, we just change the hypothesis 0 to the new hypothesis
            # this is only for test
            self.change_hypo_to_first(hypo_id)
            return result

        # Use the latest keyframe as target if not specified
        if target_node_id is None:
            target_node_id = max(self.nodes.keys())
        result["target_node_id"] = target_node_id
        result["window"] = None

        if hypo_id == 0:
            logger.info("Handling intra-hypothesis loop closure: optimising the graph of hypothesis 0")
        else:
            logger.info(f"Handling loop closure: merging hypothesis {hypo_id} with hypothesis 0")

        # Step 1: Construct the pose graph for loop closure (under graph lock)
        # global: every keyframe of every session; a session without a loaded map: every keyframe of the session
        lc_cfg = getattr(getattr(getattr(self.system, "config", None), "mapping", None), "loop_closure", None)
        full = global_opt or (bool(getattr(lc_cfg, "full_session_pgo", False))
                              and int(getattr(self.system, "_session_start_kf_id", 0)) == 0)
        window, boundary = None, set()
        min_nodes = int(getattr(lc_cfg, "pgo_window_min_nodes", 0) or 0)
        if (window_ref is not None and hypo_id == 0 and full and not global_opt and min_nodes > 0
                and len(self.nodes) >= min_nodes and not self.chart_aware and self.source_states is None):
            ids = sorted(self.nodes.keys())
            i = bisect.bisect_left(ids, int(window_ref)) - int(getattr(lc_cfg, "pgo_window_margin", 0) or 0)
            if i > 0:
                window = ids[i]
        with self.graph_lock:
            pg = PoseGraph(
                self,
                depth=(10 ** 7 if full else 1000),
                k_hop=2,
                device=self.device,
                noise_fn=self.pgo_noise_fn(),
                skip_fn=self.pgo_skip_fn(),
            )
            try:
                if window is not None:
                    boundary = pg.construct_window(window)
                else:
                    pg.construct_for_loop_closure(
                        target_node_id=target_node_id,
                        other_hypothesis_id=hypo_id,
                    )
            except ValueError as ex:
                # an inconsistent proposal (a chart-aware graph that does not reach the proposed reference chart, seen
                # once in a ROVER relocalization of the mono mode) skips this loop closure instead of ending the session
                message = f"Loop closure graph not built ({ex}), skipping"
                logger.warning(message)
                result["message"] = message
                return result

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

        # Fix the earliest keyframe from hypothesis 0; in a relocalization session (map loaded from a
        # previous session) all map keyframes stay fixed: the merge aligns the new session to the map
        session_start = getattr(self.system, "_session_start_kf_id", 0)
        if window is not None:
            # windowed optimisation: the keyframes before the window that share a factor with it stay where they are
            fixed_ids = set(boundary) if boundary else {min(original_kf_ids)}
        elif global_opt:
            # joint optimisation of the merged map (all sessions): only the very first keyframe is fixed, so the
            # cross-session edges reconcile the sessions with each other instead of pinning every earlier session
            fixed_ids = {min(original_kf_ids)}
        elif session_start > 0 and any(k < session_start for k in original_kf_ids):
            fixed_ids = {k for k in original_kf_ids if k < session_start}
        else:
            fixed_ids = {min(original_kf_ids)}
        if self.chart_aware:
            # the reference-chart anchor selected by the chart join keeps the output coordinates
            fixed_ids = ({k for k in fixed_ids if k != min(original_kf_ids)} | {pg.preferred_fixed_node}) \
                if session_start > 0 and not global_opt else {pg.preferred_fixed_node}
        # GNSS factors (cross.geo): in a mapping session they determine the map's position and heading, so the first
        # keyframe gets a soft prior (its tilt only) instead of the hard fix; in a relocalization session the map stays
        # fixed and the factors pull the session's keyframes
        geo = getattr(self.system, "_geo", None)
        if geo is not None and not self.chart_aware and self.source_states is None:
            pg.unary_position_factors = geo.pgo_factors(self.nodes, set(original_kf_ids))
            if pg.unary_position_factors:
                pg.unary_robust_c = geo.robust_c
                if session_start == 0 and fixed_ids == {min(original_kf_ids)}:
                    k0 = min(original_kf_ids)
                    R0 = self.nodes[k0].pose_mu[0].matrix().detach().cpu().numpy().astype(np.float64)[:3, :3]
                    pg.soft_priors = {k0: geo.soft_gauge_cov(R0)}
                    fixed_ids = set()
        fixed_node_id = pg.preferred_fixed_node if self.chart_aware else (min(fixed_ids) if fixed_ids else None)
        optim_node_ids = set(original_kf_ids + temp_vertex_ids) - fixed_ids

        if self.visualize_pose_graph:
            visualize_pose_graph(
                pg,
                title=f"Loop Closure: Hypo 0 + Hypo {hypo_id}",
                save_path="logs/loop_closure_graph.png",
                last_k_nodes=10,
                show_interactive=False,
            )

        try:
            pg.solve(
                optim_node_ids=optim_node_ids,
                fixed_node_ids=fixed_ids,
            )
        except RuntimeError as ex:      # e.g. a graph piece without prior (indeterminant system): keep the map as is
            logger.warning(f"Loop closure PGO failed ({type(ex).__name__}: {str(ex).splitlines()[0]}); graph left unchanged")
            return {"success": False, "message": f"PGO failed: {ex}"}

        logger.info(f"Loop closure PGO completed with cost: {pg.optimization_cost}")

        # --- geometric verification: the candidate's visual edges must fit the optimised graph ---
        if hypo_id != 0:
            verifier = getattr(self.system, "_lc_verifier", None)
            if verifier is not None:
                frac, n_edges = verifier.merged_edge_outlier_fraction(pg)      # calibrated posterior chi^2 test
            else:
                frac, n_edges = self._candidate_edge_outlier_fraction(pg)
            if n_edges > 0 and frac > self.verify_max_outlier_frac:
                message = (f"Loop closure with hypothesis {hypo_id} rejected: {frac:.0%} of its {n_edges} visual edges "
                           f"remain outliers after the optimisation")
                logger.warning(message)
                self._lc_reject_until[int(hypo_id)] = int(self.step_counter) + self.detect_reject_cooldown_steps
                result["message"] = message
                return result
        # --- diagnostics (merges only): how far did the session keyframes move, and where are their twins ---
        # (per-edge pypose residuals on the device: seconds for a 500-keyframe graph, so never for hypothesis 0)
        try:
            if hypo_id == 0:
                raise StopIteration
            start_idx = self.hypotheses[hypo_id].start_idx
            n_map_v = sum(1 for v in pg.vertices if v.id in self.nodes and v.id < start_idx)
            n_sess_v = sum(1 for v in pg.vertices if v.id in self.nodes and v.id >= start_idx)
            types = {}
            for (a, b, fs) in pg.edges:
                for f in fs:
                    types[f.type.name] = types.get(f.type.name, 0) + 1
            twins = {v.original_kf_id: v for v in pg.vertices if v.id not in self.nodes}
            lines = []
            for v in pg.vertices:
                if v.id in self.nodes and v.id >= start_idx:
                    init_p = v.pose.tensor()[:3].tolist()
                    twin_p = twins[v.id].pose.tensor()[:3].tolist() if v.id in twins else None
                    opt_p = pg.optimized_poses[v.id].tensor()[:3].tolist() if v.id in pg.optimized_poses else None
                    lines.append(f"kf {v.id}: init {[round(x, 1) for x in init_p]} twin {[round(x, 1) for x in twin_p] if twin_p else None} opt {[round(x, 1) for x in opt_p] if opt_p else None}")
            # residuals of edges touching the session at the *initial* poses
            import pypose as _pp
            vm = pg.vertex_map
            sess_ids = {v.id for v in pg.vertices if (v.id in self.nodes and v.id >= start_idx) or v.id not in self.nodes}
            res_lines = []
            for (a, b, fs) in pg.edges:
                if a in sess_ids or b in sess_ids:
                    if a not in vm or b not in vm:
                        continue
                    pred = vm[a].pose.Inv() @ vm[b].pose
                    for f in fs:
                        r = (f.mean.Inv() @ pred).Log().tensor()
                        res_lines.append((float(r[:3].norm()), f"{f.type.name} {a}->{b} |t_res|={float(r[:3].norm()):.1f} |r_res|={float(r[3:].norm()):.2f} std={[round(x, 3) for x in f.std.tensor()[:3].tolist()]}"))
            res_lines.sort(key=lambda x: -x[0])
            lines.append("worst initial residuals: " + " ; ".join(t for _, t in res_lines[:6]))
            logger.debug(f"LC-PGO diag: fixed={len(fixed_ids)} map_vertices={n_map_v} session_vertices={n_sess_v} twins={len(twins)} edges={types}\n" + "\n".join(lines[-4:]))
        except StopIteration:
            pass
        except Exception as e:  # diagnostics must never break the pipeline
            logger.debug(f"LC-PGO diag failed: {e}")

        # Step 3: Apply updates via unified method (also used by async engine); apply=False returns the solution without
        # writing it back (the verified loop closure tests it first and discards it when new edges turn out outliers)
        if apply:
            applied = self.apply_pgo_result({
                "success": True,
                "pose_graph": pg,
                "optimized_poses": pg.optimized_poses,
                "other_hypothesis_id": hypo_id,
                "target_node_id": target_node_id,
            })
            if not applied['success']:
                result['message'] = applied.get('message')
                return result

        result.update(
            {
                "success": True,
                "cost": pg.optimization_cost,
                "pose_graph": pg,
                "optimized_poses": pg.optimized_poses,
                "optim_nodes_ids": optim_node_ids,
                "fixed_nodes_ids": fixed_ids,
                "window": window,
                "message": None,
            }
        )

        return result

    def _candidate_edge_outlier_fraction(self, pg) -> Tuple[float, int]:
        """Fraction of the merged hypothesis' visual edges (edges touching a twin vertex) whose Mahalanobis residual
        at the optimised poses exceeds `verify_outlier_sigma`."""
        vm = pg.vertex_map
        n_out, n = 0, 0
        for (a, b, factors) in pg.edges:
            if a in self.nodes and b in self.nodes:
                continue                      # hypothesis-0 edge
            if a not in vm or b not in vm:
                continue
            pa = pg.optimized_poses.get(a, vm[a].pose)
            pb = pg.optimized_poses.get(b, vm[b].pose)
            pred = pa.Inv() @ pb
            for f in factors:
                if f.type != EdgeType.VISUAL:
                    continue
                r = (f.mean.Inv() @ pred).Log().tensor()
                std = f.std.tensor().clamp(min=1e-3)
                if float(torch.norm(r / std)) > self.verify_outlier_sigma * math.sqrt(6.0):
                    n_out += 1
                n += 1
        return (n_out / n if n else 0.0), n

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
        self.pose_epoch += 1
        if self.source_states is not None:
            from cross.core.conditional_pgo import apply_result
            with self.graph_lock:
                return apply_result(self,pgo_result.get('pose_graph'),pgo_result.get('optimized_poses',{}),
                                    int(pgo_result.get('other_hypothesis_id',0)))
        optimized_poses: Dict[int, pp.LieTensor] = pgo_result.get("optimized_poses", {})
        other_hypo = int(pgo_result.get("other_hypothesis_id", 0))
        pg = pgo_result.get("pose_graph")
        affected_ids = set(optimized_poses.keys())
        anchored_reference = other_hypo != 0 and self.reference_audit(other_hypo)["unanchored_reference_candidate"]

        # Mutate shared graph under lock
        with self.graph_lock:
            if self.chart_aware:
                if pg is None or pg.output_chart is None:
                    return dict(success=False, message="PGO result lacks coordinate provenance")
                for component in {0, other_hypo}:
                    if (pg.source_component_charts[component] != int(self.component_charts[component]) or
                            pg.source_component_generations[component] != self.component_generations[component]):
                        return dict(success=False, message="PGO result refers to a retired chart or hypothesis")
                # Temporary optimizer vertices can collide with real keyframe
                # IDs allocated while an asynchronous solve was in flight.
                # The snapshot's component provenance, not current membership
                # in self.nodes, determines which solved poses are originals.
                optimized_poses = {i: p for i, p in optimized_poses.items() if i in pg.source_node_poses}
                affected_ids = set(optimized_poses)
                affected_ids |= self._transport_merged_charts(pg, optimized_poses)
            self._write_poses(optimized_poses, pg.output_chart if self.chart_aware else None)

            if other_hypo != 0 and other_hypo in self.hypotheses:
                self.merge_hypotheses(other_hypo)
                if anchored_reference:
                    self.reference_support.mark_anchored()

            if affected_ids:
                if self.system.topo_map is not None:
                    self.system.topo_map.update_after_pgo(affected_ids)

            # Align tracking dist to latest KF in comp 0
            if self.system.last_added_kf_id is not None and self.system.last_added_kf_id in self.nodes:
                last_kf_id = self.system.last_added_kf_id
                self.dist[0][0] = self.nodes[last_kf_id].pose_mu[0]
                self.dist[1][0] = self.nodes[last_kf_id].pose_std[0]
                self.dist[2][0] = 1
                if self.chart_aware:
                    self.component_charts[0] = self.nodes[last_kf_id].pose_charts[0]

        ret = {
            "success": True,
            "pose_graph": pg,
            "optimized_poses": optimized_poses,
            "other_hypothesis_id": other_hypo,
        }
        if pg is not None:
            ret["cost"] = pg.optimization_cost
        return ret

    def _write_poses(self, optimized_poses: Dict[int, pp.LieTensor], output_chart=None) -> None:
        """Write optimised hypothesis-0 poses into the keyframes (those still in the graph)."""
        step = int(self.step_counter)
        # plain tensor assignment (same values; pypose's dispatch costs ~90 us per keyframe)
        with torch._C.DisableTorchFunctionSubclass():
            for node_id, optimized_pose in optimized_poses.items():
                if node_id in self.nodes:
                    kf = self.nodes[node_id]
                    kf.pose_mu[0] = optimized_pose
                    if self.chart_aware:
                        kf.pose_charts[0] = output_chart
                    # The keyframe's std is left as it is: the optimisation does not compute marginals, and halving
                    # it at every optimisation (the previous behaviour) underflowed to exactly zero after ~100
                    # optimisations (HSSD house: 1037 of 1045 keyframes at std 0), after which the belief fusion
                    # produced garbage poses that no later optimisation could repair
                    kf.last_pgo_step = step

    def apply_stale_pgo_result(self, ids, opt, fork, max_id: int, fresh_poses: bool) -> Dict[str, Any]:
        """Apply an optimisation of hypothesis 0 that was computed on the state at an earlier step (a background job,
        cross.core.async_pgo) to the present state.

        ids / opt / fork: the optimised keyframes, their optimised poses and their poses at the fork ((n, 7) float32).
        max_id: the latest keyframe at the fork.  fresh_poses: no other optimisation moved any keyframe since the fork.
        - A keyframe still at its fork pose takes the optimised pose; one that moved meanwhile gets the correction
          `opt fork^-1` left-multiplied to its present pose.
        - Keyframes added after the fork (id > max_id) and the tracked pose take the correction of keyframe max_id
          (they hang off it through the odometry chain): the tail moves rigidly with it.
        Returns {"success", "n_tail", "affected", "correction_t"}."""
        self.pose_epoch += 1
        ids = [int(i) for i in ids]
        opt = np.asarray(opt, dtype=np.float32)
        fork = np.asarray(fork, dtype=np.float32)
        corr = (pp.SE3(torch.from_numpy(opt.astype(np.float64))) @ pp.SE3(torch.from_numpy(fork.astype(np.float64))).Inv()).tensor()

        def moved_to(C: torch.Tensor, kf) -> torch.Tensor:
            """The keyframe's present pose with the world-frame correction C applied (float32 row)."""
            cur = torch.from_numpy(kf.plain_row("pose_mu", 0).detach().cpu().numpy().astype(np.float64))
            return (pp.SE3(C) @ pp.SE3(cur)).tensor().to(torch.float32).clone()

        with self.graph_lock:
            affected = set()
            for i, nid in enumerate(ids):
                kf = self.nodes.get(nid)
                if kf is None:
                    continue
                if not fresh_poses and not np.array_equal(kf.plain_row("pose_mu", 0).detach().cpu().numpy(), fork[i]):
                    row = moved_to(corr[i], kf)
                else:
                    row = torch.from_numpy(opt[i]).clone()
                with torch._C.DisableTorchFunctionSubclass():
                    kf.pose_mu[0] = as_se3(row)
                    kf.last_pgo_step = int(self.step_counter)
                affected.add(nid)
            C = corr[ids.index(max_id)] if max_id in ids else None
            n_tail = 0
            if C is not None:
                for nid in reversed(self.nodes):
                    if nid <= max_id:
                        break
                    kf = self.nodes[nid]
                    row = moved_to(C, kf)
                    with torch._C.DisableTorchFunctionSubclass():
                        kf.pose_mu[0] = as_se3(row)
                    affected.add(nid)
                    n_tail += 1
                if self.dist is not None:
                    d0 = self.dist[0]
                    d0[0] = pp.SE3(C.to(device=d0.device, dtype=d0.dtype)) @ d0[0]
            if affected and self.system.topo_map is not None:
                self.system.topo_map.update_after_pgo(affected)
        return {"success": True, "n_tail": n_tail, "affected": len(affected),
                "correction_t": float(C[:3].norm()) if C is not None else 0.0}

    def _transport_merged_charts(self, pg, optimized_poses):
        """Transport unsolved poses as well as solved nodes after commitment.

        PGO supplies local corrections at its vertices. For the rest of a
        merged chart, use the most recent solved pose as its rigid anchor.
        Left multiplication preserves right-tangent stds and relative edges;
        this is an SE(3) chart join, not a scale or Sim(3) update.
        """
        transforms = {}
        for node_id in sorted(pg.source_node_poses):
            chart = pg.source_node_charts[node_id]
            if chart == pg.output_chart or node_id not in optimized_poses:
                continue
            transforms[chart] = optimized_poses[node_id] @ pg.source_node_poses[node_id].Inv()
        affected = set()
        for chart, transform in transforms.items():
            for kf in self.nodes.values():
                mask = kf.pose_charts == chart
                if bool(mask.any()):
                    kf.pose_mu[mask] = transform @ kf.pose_mu[mask]
                    kf.pose_charts[mask] = pg.output_chart
                    affected.add(kf.id)
            mask = self.component_charts == chart
            self.dist[0][mask] = transform @ self.dist[0][mask]
            self.component_charts[mask] = pg.output_chart
        return affected

    def rebuild_topology_graph(self):
        """
        Rebuild the SimpleTopo proximity graph from scratch.

        This clears existing proximity edges and recomputes them using current
        node poses. Useful when incremental proximity is disabled.
        """
        if self.system.topo_map is not None:
            self.system.topo_map.rebuild_graph()

    def _h0_unlocalized(self) -> bool:
        """Relocalization session whose hypothesis 0 is not (or no longer) anchored to the stored map."""
        sysm = self.system
        v = getattr(sysm, "_lc_verifier", None)
        return getattr(sysm, "_session_start_kf_id", 0) > 0 and (v is None or v.anchor is None)

    def _reset_session_anchor(self):
        """Hypothesis 0 is being replaced (merge / adoption): its link to the stored map is void (and is checked again
        on the new hypothesis 0's edges, System.session_localized)."""
        if hasattr(self.system, "_session_localized"):
            self.system._session_localized = False
        v = getattr(self.system, "_lc_verifier", None)
        if v is not None:
            v.anchor = None
        for name in ("_anchor_pending", "_contra_pending"):
            pend = getattr(self.system, name, None)
            if pend is not None:
                pend.clear()

    def merge_hypotheses(self, comp_idx: int, conditional_transport_done=False):
        """
        Merges the hypothesis after loop closure
        """
        self.pose_epoch += 1
        self.graph_epoch += 1
        self._reset_session_anchor()
        if self.source_states is not None and not conditional_transport_done:
            raise NotImplementedError("Conditional pose/source graph transport is required before merging hypotheses")
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
                        (edge_factor.from_comp_id == comp_idx and edge_factor.to_comp_id == 0) or \
                        (self.chart_aware and edge_factor.from_comp_id == edge_factor.to_comp_id == comp_idx):

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
        self.hist_valid[comp_idx, :] = False
        self.strong_hist[comp_idx, :] = False
        self.log_c_hist[comp_idx, :] = 0.0
        self.log_u_hist[comp_idx, :] = 0.0
        self.log_conf_hist[comp_idx, :] = 0.0
        self.last_sum_pos[comp_idx] = 0.0
        self.last_hit_rate[comp_idx] = 0.0

    def change_hypo_to_first(self, comp_idx: int):
        """
        Adopt hypothesis `comp_idx` as hypothesis 0: its keyframe poses (from its start index on), its
        visual edges and its mixture component replace those of hypothesis 0, and the slot is freed.
        """
        self.pose_epoch += 1
        self.graph_epoch += 1
        if self.source_states is not None:
            raise NotImplementedError("Conditional pose/source graph transport is required before promoting a hypothesis")
        logger.debug(f"Changing hypothesis {comp_idx} to first component")
        logger.info(f"Adopting hypothesis {comp_idx} as hypothesis 0")
        hypothesis_exist = comp_idx in self.hypotheses
        if not hypothesis_exist:
            logger.debug(f"Hypothesis {comp_idx} does not exist, skipping")
            return
        with self.graph_lock:
            for edge_key, edge_factors in self.hypotheses[comp_idx].visual_edges.items():
                for edge_factor in edge_factors:
                    if edge_factor.from_comp_id == comp_idx:
                        edge_factor.from_comp_id = 0
                    if edge_factor.to_comp_id == comp_idx:
                        edge_factor.to_comp_id = 0
                    if edge_factor.from_comp_id == 0 and edge_factor.to_comp_id == 0:
                        bucket = self.hypotheses[0].visual_edges.setdefault(edge_key, [])
                        if not any(e is edge_factor for e in bucket):
                            bucket.append(edge_factor)
                        self.hypotheses[0].visual_adjacency.setdefault(edge_key[0], set()).add(edge_key[1])
                        self.hypotheses[0].visual_adjacency.setdefault(edge_key[1], set()).add(edge_key[0])
            start_idx = self.hypotheses[comp_idx].start_idx
            for kf in self.nodes.values():
                if kf.id >= start_idx:
                    kf.pose_mu[0] = kf.pose_mu[comp_idx]
                    kf.pose_std[0] = kf.pose_std[comp_idx]
                    kf.pose_weights[0] = kf.pose_weights[0] + kf.pose_weights[comp_idx]
                    kf.pose_mu[comp_idx] = pp.identity_SE3(1, device=kf.pose_mu.device)
                    kf.pose_std[comp_idx] = pp.identity_se3(1, device=kf.pose_mu.device)
                    kf.pose_weights[comp_idx] = 0.0
            del self.hypotheses[comp_idx]
        mu, sigma, weights = self.dist
        mu[0] = mu[comp_idx]
        sigma[0] = sigma[comp_idx]
        weights[0] = weights[0] + weights[comp_idx]
        mu[comp_idx] = pp.identity_SE3(1, device=mu.device)
        sigma[comp_idx] = pp.identity_se3(1, device=sigma.device)
        weights[comp_idx] = 0.0
        weights = weights / weights.sum()
        self.dist = (mu, sigma, weights)
        self.realized[comp_idx] = False
        self.ttl[comp_idx] = 0
        self.newborn[comp_idx] = False
        self.llr_hist[comp_idx, :] = 0.0
        self.hist_valid[comp_idx, :] = False
        self.strong_hist[comp_idx, :] = False
        self.log_c_hist[comp_idx, :] = 0.0
        self.log_u_hist[comp_idx, :] = 0.0
        self.log_conf_hist[comp_idx, :] = 0.0
        self.last_sum_pos[comp_idx] = 0.0
        self.last_hit_rate[comp_idx] = 0.0
        self._adopt_counter = (None, 0)

    def maybe_adopt_dominant_hypothesis(self, force: bool = False) -> bool:
        """Adopt a realized hypothesis that has held the belief while hypothesis 0 died (see config
        adopt_*).  With `force`, adopt immediately whenever the dominant component is not 0 (used before
        the map is saved)."""
        weights = self.dist[2]
        if weights.numel() < 2:
            return False
        if not getattr(self.cfg, "adopt_in_mapping_session", True) and int(getattr(self.system, "_session_start_kf_id", 0)) == 0:
            self._adopt_counter = (None, 0)     # mapping session: only the verified merge may replace hypothesis 0
            return False
        k = int(torch.argmax(weights).item())
        cfg_steps = getattr(self.cfg, "adopt_dominant_steps", 20)
        w0_max = getattr(self.cfg, "adopt_w0_max", 0.05)
        wk_min = getattr(self.cfg, "adopt_wk_min", 0.9)
        dominant = k != 0 and k in self.hypotheses and bool(self.realized[k]) and float(weights[k]) > wk_min \
            and float(weights[0]) < w0_max
        if not dominant:
            self._adopt_counter = (None, 0)
            return False
        comp, count = getattr(self, "_adopt_counter", (None, 0))
        count = count + 1 if comp == k else 1
        self._adopt_counter = (k, count)
        dist = float(torch.norm(self.dist[0][k].tensor()[:3] - self.dist[0][0].tensor()[:3]))
        if dist < getattr(self.cfg, "adopt_close_dist", 3.0):
            cfg_steps = min(cfg_steps, getattr(self.cfg, "adopt_close_steps", 5))
        if force or count >= cfg_steps:
            logger.info(f"Adopting dominant hypothesis {k} as hypothesis 0 ({dist:.1f} m away, after {count} steps)")
            self._reset_session_anchor()
            self.change_hypo_to_first(k)
            return True
        return False

    def save_state(self, columns: bool = False):
        """Save the hypothesis manager state for map persistence.
        Only saves hypothesis 0 (ground truth) and temporary keyframes.
        Warns if multiple realized hypotheses exist at save time.

        `columns`: the records already encoded as map columns (cross.db.store format v2; System.save_map), built from
        the graph objects in bulk (cross.core.bulk_load) where they allow it: the same columns, without a tensor per
        field.

        Returns:
            dict: Hypothesis manager state including temp keyframes, edges, and hypothesis 0
        """
        if columns:
            return self._save_state_columns()
        from cross.core.conditional_pose import records
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
                    "pose_charts": kf.pose_charts.cpu() if kf.pose_charts is not None else None,
                    "metric_source": kf.metric_source,
                    "conditional_poses": records(kf.conditional_poses),
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
                "conditional_pose": edge.conditional_pose.record() if edge.conditional_pose is not None else None,
                "n_frames": getattr(edge, "n_frames", None),
            }
            if getattr(edge, "odom_fault", None):
                odom_edges[edge_key]["odom_fault"] = float(edge.odom_fault)

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
                        "conditional_pose": edge.conditional_pose.record() if edge.conditional_pose is not None else None,
                        "conf": getattr(edge, "conf", None),
                        "noise_scale": getattr(edge, "noise_scale", None),
                        "noise_scale_along": getattr(edge, "noise_scale_along", None),
                        "noise_scale_rot": getattr(edge, "noise_scale_rot", None),
                        "informative": getattr(edge, "informative", None),
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
            "source_belief": (self.source_states[0].record() if self.source_states is not None else
                              self.saved_source_belief.record() if self.saved_source_belief is not None else None),
            "temp_keyframes": temp_keyframes,
            "odom_edges": odom_edges,
            "hypotheses_data": hypotheses_data,
        }

    def _save_state_columns(self):
        """save_state() encoded as cross.db.store._encode_hypo encodes it, built in bulk; a part the bulk encoders
        cannot reproduce exactly is encoded from its records (save_state's)."""
        from cross.core import bulk_load
        from cross.db import store
        ref = None

        def records():                               # the record path, built once if a part needs it
            nonlocal ref
            if ref is None:
                ref = self.save_state()
            return ref
        temps = [kf for kf in self.nodes.values() if kf.temporary]
        enc_t = bulk_load.encode_keyframes(temps, image_fields=False)
        if enc_t is None:
            enc_t = store.encode_records(records()["temp_keyframes"])
        enc_o = bulk_load.encode_edges(list(self.odom_edges.values()), visual=False)
        enc_o = ({"__dor__": True, "keys": store._encode_keys(list(self.odom_edges.keys())), "recs": enc_o}
                 if enc_o is not None else store.encode_dict_of_records(records()["odom_edges"]))
        hypotheses_data = {}
        if 0 in self.hypotheses:
            h = self.hypotheses[0]
            flat = [e for l in h.visual_edges.values() for e in l]
            enc_v = bulk_load.encode_edges(flat, visual=True)
            if enc_v is not None:
                enc_v = {"__dol__": True, "keys": store._encode_keys(list(h.visual_edges.keys())),
                         "counts": np.array([len(l) for l in h.visual_edges.values()], dtype=np.int64),
                         "records": True, "items": enc_v}
            else:
                enc_v = store.encode_dict_of_lists(records()["hypotheses_data"][0]["visual_edges"], records=True)
            hypotheses_data[0] = {
                "component_id": h.component_id,
                "start_idx": h.start_idx,
                "visual_edges": enc_v,
                "visual_adjacency": store.encode_dict_of_lists({k: list(v) for k, v in h.visual_adjacency.items()},
                                                               records=False),
            }
        return {
            "source_belief": (self.source_states[0].record() if self.source_states is not None else
                              self.saved_source_belief.record() if self.saved_source_belief is not None else None),
            "temp_keyframes": enc_t,
            "odom_edges": enc_o,
            "hypotheses_data": hypotheses_data,
            "__columns__": True,
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
        from cross.core.conditional import SourceState
        from cross.core.conditional_pose import ConditionalPose, restore
        from cross.db.store import to_device      # .to() skipped on the same device (large maps: 2 LieTensors per edge)
        # --- 1. Restore temporary keyframes ---
        all_keyframes_map = existing_keyframes.copy()

        # a v2 map read for System.load_map: graph objects built from the columns in bulk (cross.core.bulk_load),
        # with the attributes the record-by-record restore below gives them
        from cross.core import bulk_load
        cpu = all(d is None or torch.device(d).type == "cpu" for d in (storage_device, device))
        temps = hypo_data["temp_keyframes"]
        bulk_temps = None
        if cpu and getattr(temps, "enc", None) is not None:
            bulk_temps = bulk_load.keyframes(temps.enc, lambda a: db.get_atlas(a) if a is not None else None,
                                             normalize_mu=True)   # maps saved before the renormalization fix carry |q| < 1
        if bulk_temps is not None:
            all_keyframes_map.update((kf.id, kf) for kf in bulk_temps)

        for kf_data in (temps if bulk_temps is None else ()):
            atlas = db.get_atlas(kf_data["atlas_id"]) if kf_data["atlas_id"] is not None else None

            kf = Keyframe(
                pose_mu=to_device(normalize_SE3(unpacked(kf_data["pose_mu"])), storage_device) if kf_data["pose_mu"] is not None else None,   # maps saved before the renormalization fix carry |q| < 1
                pose_std=to_device(kf_data["pose_std"], storage_device),
                pose_weights=to_device(kf_data["pose_weights"], storage_device),
                atlas=atlas,
                timestamp=kf_data["timestamp"],
                temporary=kf_data["temporary"],
                last_pgo_step=kf_data["last_pgo_step"],
                pose_charts=to_device(kf_data.get("pose_charts"), storage_device),
                metric_source=kf_data.get("metric_source"),
                conditional_poses=restore(kf_data.get("conditional_poses")),
            )

            # Manually set the ID to match the saved one
            kf.id = kf_data["id"]

            all_keyframes_map[kf.id] = kf

        # --- 2. Restore nodes (both from database and temporary) ---
        self.nodes.clear()
        self.nodes.update(all_keyframes_map)

        # --- 3. Restore odometry edges ---
        self.odom_edges.clear()
        self.odom_edges_version = getattr(self, "odom_edges_version", 0) + 1
        odom = hypo_data["odom_edges"]
        bulk_odom = None
        if cpu and getattr(odom, "enc", None) is not None:
            bulk_odom = bulk_load.edges(odom.enc["recs"], visual=False) if odom.enc["recs"].get("__records__") else None
        if bulk_odom is not None:
            self.odom_edges.update(zip(bulk_load.keys(odom.enc["keys"]), bulk_odom))
        for edge_key, edge_data in (odom.items() if bulk_odom is None else ()):
            edge = Edge(
                mean=to_device(edge_data["mean"], device),
                std=to_device(edge_data["std"], device),
                type=EdgeType[edge_data["type"]],
            )
            edge.n_frames = edge_data.get("n_frames")
            if edge_data.get("odom_fault"):
                edge.odom_fault = float(edge_data["odom_fault"])
            self.odom_edges[edge_key] = edge
            if edge_data.get('conditional_pose') is not None:
                edge.conditional_pose = ConditionalPose.from_record(edge_data['conditional_pose'])

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
            ve = hypo_data_item["visual_edges"]
            bulk_vis = None
            if cpu and getattr(ve, "enc", None) is not None and ve.enc["records"] and ve.enc["items"].get("__records__"):
                bulk_vis = bulk_load.edges(ve.enc["items"], visual=True)
            if bulk_vis is not None:
                hypothesis.visual_edges.update((k, l) for k, l in zip(bulk_load.keys(ve.enc["keys"]),
                                                                      bulk_load.grouped(bulk_vis, ve.enc["counts"])) if l)
            for edge_key, edge_list_data in (ve.items() if bulk_vis is None else ()):
                for edge_data in edge_list_data:
                    edge = VisualEdge(
                        mean=to_device(edge_data["mean"], device),
                        std=to_device(edge_data["std"], device),
                        type=EdgeType[edge_data["type"]],
                        from_comp_id=edge_data["from_comp_id"],
                        to_comp_id=edge_data["to_comp_id"],
                    )
                    edge.conf = edge_data.get("conf")
                    edge.noise_scale = edge_data.get("noise_scale")
                    edge.noise_scale_along = edge_data.get("noise_scale_along")
                    edge.noise_scale_rot = edge_data.get("noise_scale_rot")
                    edge.informative = edge_data.get("informative")
                    hypothesis.visual_edges.setdefault(edge_key, []).append(edge)
                    if edge_data.get('conditional_pose') is not None:
                        edge.conditional_pose = ConditionalPose.from_record(edge_data['conditional_pose'])

            # Restore visual adjacency
            va = hypo_data_item["visual_adjacency"]
            enc = getattr(va, "enc", None)
            if enc is not None and not enc["records"] and enc["items"]["k"] == "int":
                hypothesis.visual_adjacency.update(zip(bulk_load.keys(enc["keys"]), map(set, bulk_load.grouped(
                    enc["items"]["a"].tolist(), enc["counts"]))))
            else:
                for node_id, neighbors in va.items():
                    hypothesis.visual_adjacency[node_id] = set(neighbors)

            self.hypotheses[0] = hypothesis

        # --- 5. Reset all tracking state metadata for new session ---
        if self.chart_aware:
            from cross.core.charts import restore_node_charts
            self.next_chart_id = restore_node_charts(self.nodes, self.odom_edges,
                                                     self.hypotheses[0].visual_edges if 0 in self.hypotheses else {})
        # Don't restore old tracking state - start fresh
        self.reset_tracking_state()
        self.saved_source_belief = (SourceState.from_record(hypo_data['source_belief'])
                                   if hypo_data.get('source_belief') is not None else None)
        if self.saved_source_belief is not None:
            for kf in self.nodes.values():
                if kf.conditional_poses is None or kf.conditional_poses[0] is None:
                    raise ValueError('Conditional map is missing a committed node message')
                # Check source definitions without adding a second bias prior.
                expanded,_,_ = self.saved_source_belief.expand(kf.conditional_poses[0].factor)
                if expanded.keys != self.saved_source_belief.keys:
                    raise ValueError('Conditional node names a source absent from the saved belief')
                for component in range(1,len(kf.conditional_poses)):
                    kf.conditional_poses[component] = None

        # Ensure hypothesis 0 exists
        if 0 not in self.hypotheses:
            self.create_hypothesis_branch(0, 0)
