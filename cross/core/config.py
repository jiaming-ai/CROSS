"""
Typed configuration system for CROSS.

All defaults live here as dataclass defaults. YAML files override them via
``load_config()``, which deep-merges one or more YAML files and then applies
optional keyword overrides.

Usage::

    from cross.core.config import SystemConfig, load_config

    # Pure defaults
    cfg = SystemConfig()

    # From YAML
    cfg = load_config("configs/default.yaml")

    # Layered: base + experiment + CLI overrides
    cfg = load_config("configs/default.yaml", "configs/experiments/indoor_vio.yaml",
                      tracking={"filter_mode": "adaptive"})
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml

# ---------------------------------------------------------------------------
# Enums for string-valued choices
# ---------------------------------------------------------------------------

class FilterMode(str, Enum):
    FULL = "full"
    SKIP_ACTIVE = "skip_active"
    ADAPTIVE = "adaptive"


class PoseEstType(str, Enum):
    PNP = "pnp"
    VGGT = "vggt"
    FF = "ff"  # stereo mode: feed-forward multi-view model with stereo scale anchors (cross/cv/pose_est_ff.py)
    # mono mode: the relative-pose estimator is injected by the monocular pipeline (System(pose_estimator=...))
    DA3 = "da3"
    METRIC_PNP = "metric_pnp"


class FFBackend(str, Enum):
    VGGT_OMEGA = "vggt_omega"
    DA3 = "da3"



class KPDetectorType(str, Enum):
    XFEAT = "xfeat"
    DISK = "disk"


class KPMatcherType(str, Enum):
    LIGHTGLUE = "lightglue"


class VPRModelType(str, Enum):
    BOQ = "boq"


class RobustKernelType(str, Enum):
    HUBER = "huber"
    CAUCHY = "cauchy"
    TUKEY = "tukey"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveFilterConfig:
    trans_thresh: float = 0.10  # meters
    rot_thresh: float = 0.10   # radians


@dataclass
class TrackingConfig:
    use_VO: bool = False
    use_odometry: bool = True
    new_kf_after_n_unsuccessful_steps: int = 5
    odom_std_per_meter: float = 0.5
    odom_std_per_radian: float = 0.5
    odom_min_std_translation: float = 0.1
    odom_min_std_rotation: float = 0.1
    # take the four odometry constants above from the calibrated noise model of the verified loop closure
    # (NoiseModelConfig.odom_k_t / odom_k_r / odom_floor_t / odom_floor_r, fitted without ground truth by
    # scripts/lc/calibrate_noise.py) instead of the hand-set defaults, so that the belief's motion uncertainty describes
    # the odometry actually used.  The defaults (0.5 per metre / radian, floors 0.1 m / 0.1 rad per step) are 5-50x wider
    # than a typical wheel / VIO odometry and hand every visual measurement a large gain.  Needs loop_closure.mode verified.
    odom_std_from_noise_model: bool = False
    filter_mode: FilterMode = FilterMode.FULL
    adaptive_filter: AdaptiveFilterConfig = field(default_factory=AdaptiveFilterConfig)


@dataclass
class RetrievalConfig:
    vpr_model_type: VPRModelType = VPRModelType.BOQ
    top_k: int = 10
    own_session_slots: int = 2   # retrieval slots for keyframes of the current session; the rest go to map keyframes
    # mapping (no map loaded): reserve the same number of slots for recent keyframes and give the rest to
    # keyframes older than recent_window_steps, so that loop-closure candidates are not crowded out by the
    # keyframes just behind the robot (0 disables the split)
    # 0 = disabled (default).  All reported results were produced with this split inactive: its keyframe-age test
    # compared keyframe ids with step counts and never triggered after ~200 steps.  Enabling it (e.g. 150) gives the
    # keyframes just behind the robot only `own_session_slots` slots, which weakens local tracking and, in scenes with
    # repetitive structure (Lone Monk arcades), lets aliased old keyframes spawn false loop closures.
    recent_window_steps: int = 0
    # when the split is enabled: keyframes older than the window are admitted only with a VPR score of at least
    # recent_split_min_score and at least recent_split_rel_score times the best score of the query, and the recent
    # keyframes keep at least recent_split_recent_slots slots (rank-only admission of old keyframes fed aliased
    # places to the pose estimator and starved local tracking)
    recent_split_min_score: float = 0.45
    recent_split_rel_score: float = 0.85
    recent_split_recent_slots: int = 4
    map_score_threshold: float = 0.0   # VPR score threshold for map keyframes (rank-based retrieval; geometry verifies)
    # pose-guided retrieval (relocalization sessions): once the session is anchored to the map (a verified map edge),
    # up to pose_guided_k map keyframes whose viewed region overlaps the current one are added in front of the VPR
    # results.  The viewed region of a camera is the point pose_guided_depth metres along its optical axis; overlap =
    # those points within pose_guided_radius and optical axes within pose_guided_max_angle_deg.  Appearance retrieval
    # across sessions under strong appearance change is weak, the tracked pose is not.
    # keyframe images stored as uint8 and depth as fp16 instead of float32: 4x less memory per keyframe and per saved map
    # (images come from 8-bit sensors, so the round trip is lossless; fp16 depth resolution is < 0.1 % of the range);
    # every consumer converts with cross.db.db.as_float_image
    store_images_uint8: bool = True
    pose_guided_k: int = 0
    pose_guided_depth: float = 2.0
    pose_guided_radius: float = 1.0
    pose_guided_max_angle_deg: float = 60.0
    vpr_score_threshold_high: float = 0.3
    vpr_score_threshold_low: float = 0.3
    initial_buffer_size: int = 1000
    historical_slots: int = 0  # reserve within top_k after loading a map; same score thresholds
    historical_min_score: float | None = None  # opt-in exploration floor, saved-map candidates only


@dataclass
class NoiseModelConfig:
    """Calibrated measurement noise used by the verified loop closure and the pose-graph optimisation.
    sigma_visual = a + b * |t| (relative pose of a visual edge), sigma_odom = k * motion / sqrt(n) + floor (n integrated
    readings), sigma_pair (two references registered in one feed-forward pass) and sigma_map (relative pose of two
    stored map keyframes) are linear in the distance as well.  Rotations in radians, translations in metres.
    Defaults are conservative values from the simulator (indoor and outdoor scenes); scripts/lc/calibrate_noise.py
    estimates all of them without ground truth from about a minute of the robot's own data (`noise_file`)."""
    visual_scale: float = 1.0      # metric scale of the estimator's translations relative to the odometry (measured / true); calibrated, translations are divided by it
    visual_t_a: float = 0.03
    visual_t_b: float = 0.02
    visual_r_a: float = 0.003
    visual_r_b: float = 0.001
    odom_k_t: float = 0.06
    odom_k_r: float = 0.07
    odom_floor_t: float = 0.002
    odom_floor_r: float = 0.001
    pair_t_a: float = 0.05
    pair_t_b: float = 0.025
    pair_r_a: float = 0.003
    pair_r_b: float = 0.0015
    map_t_a: float = 0.05
    map_t_b: float = 0.03
    map_r_a: float = 0.02
    map_r_b: float = 0.001
    # odometry covariance inflation used by the prior gate only (uncertainty of the odometry noise model itself)
    gate_inflation: float = 2.0


@dataclass
class LoopClosureConfig:
    async_: bool = False
    queue_size: int = 1
    # "verified": consistency-tested loop closure of hypothesis 0 (cross/core/lc_verify.py) with calibrated noise in
    #             the pose-graph optimisation; one decision parameter (`confidence`, chi-square level, 6 dof).
    # "heuristic": the intra-hypothesis PGO of 2026-09-08 (intra_* parameters below) with the system's own stds.
    mode: str = "verified"
    confidence: float = 0.999
    noise: NoiseModelConfig = field(default_factory=NoiseModelConfig)
    noise_file: Optional[str] = None      # YAML written by scripts/lc/calibrate_noise.py (overrides `noise`)
    use_inpass: bool = True               # ablations of the three tests
    use_prior: bool = True
    use_posterior: bool = True
    # what the posterior test does with new edges that remain outliers after the optimisation: "flag" (statistics
    # only; the robust optimisation keeps every prior-consistent edge, so a poor early solution is overturned by later
    # evidence) or "remove" (quarantine them and re-optimise; first-come lock-in observed on Lone Monk seed 2)
    posterior_action: str = "remove"   # a new edge that remains an outlier after the robust optimisation is quarantined and the graph re-optimised
    # degrees of freedom of the consistency tests: "translation" (3 dof: the marginal test on the translation
    # residual, whose covariance includes the rotation-induced position uncertainty; wrong-place references are a
    # translation phenomenon and the rotation noise of the estimator is the least well calibrated quantity) or
    # "full" (6 dof)
    test_dof: str = "translation"
    # online adaptation of the visual noise scale: the normalised innovation of measurements to keyframes a few
    # steps back (same statistic as the calibration) is tracked in a sliding window; under appearance change the
    # estimator gets noisier and the model is inflated accordingly (never deflated below the calibration)
    adaptive_scale: bool = True
    adaptive_window: int = 150
    # minimum odometry-chain length (m) of a span used for the online metric-scale ratio (measured / odometry
    # translation over chains of <= 5 edges); 1 m for driving / indoor scenes, less for slow platforms
    scale_min_span_m: float = 1.0
    # Intra-hypothesis loop closure.  The multi-hypothesis detector (HypothesisConfig.detect_*) only fires when a
    # *second* realized hypothesis out-scores hypothesis 0, i.e. when the revisit is inconsistent with the tracked
    # belief by more than the measurement noise (and its candidate is > 3 m away).  A revisit with sub-metre drift
    # is absorbed by hypothesis 0 as an ordinary measurement update: visual edges to the old keyframes are stored
    # but no pose-graph optimisation runs and the accumulated drift stays in the map.  With this option, hypothesis 0
    # is optimised (earliest keyframe fixed; map keyframes fixed in a relocalization session) as soon as enough
    # accepted visual edges to keyframes of the current session older than RetrievalConfig.recent_window_steps
    # have been added.
    # relocalization sessions: a new edge to a stored-map keyframe whose residual against the session graph exceeds the
    # calibrated noise triggers the optimisation of the session against the fixed map (like a loop candidate)
    map_edges_trigger: bool = False
    # write an optimisation back only after its new edges passed the posterior test (else quarantine them and re-solve
    # from the unmodified graph); False: apply first, test, re-solve from the applied solution (original behaviour)
    test_before_apply: bool = False
    # temporal corroboration of intra-session loop candidates: a loop edge constrains the graph (and may trigger an
    # optimisation) only when another loop edge within corroborate_window observations implies the same correction of
    # the current pose (tolerances below); 0 disables.  Once odometry drift is large the prior gate is wide and single
    # aliased revisits pass it (seen on 1 km surveys: map error 12-16 m against 2.5 m for odometry alone)
    corroborate_window: int = 0
    corroborate_tol_t: float = 0.5
    corroborate_tol_r_deg: float = 5.0
    # the optimisation run when a map is saved covers every keyframe of every session with only the first keyframe fixed
    # (False: the neighbourhood of the latest keyframe, earlier sessions fixed), so a merged multi-session map is made
    # consistent before the next session registers to it
    global_final_opt: bool = False
    # relocalization sessions: a map edge of hypothesis 0 anchors the session to the map (prior test of the following
    # map references) when it passes the prior test through an earlier, still unanchored map edge of hypothesis 0 from
    # another observation (within anchor_corroborate_window steps) to another map keyframe.  Without it a session
    # anchors only on a map edge corroborated inside one forward pass (>= 2 mutually consistent map references), which
    # on low-texture sites rarely happens, and an unanchored hypothesis 0 is replaced by every aliased place.  0: off.
    anchor_corroborate_window: int = 0
    # a session anchor contradicted by a consensus is dropped: when a map measurement rejected by the prior test agrees
    # (through the odometry chain) with at least anchor_contradict_min earlier rejected map measurements of distinct
    # observations and map keyframes within anchor_contradict_window steps.  An anchored hypothesis 0 is exempt from
    # the miss evidence, so a wrong anchor (a wrong merge, or a drifted part of an earlier session) otherwise locks the
    # session out of the map: the prior test rejects the true measurements and no hypothesis can out-score it.  0: off.
    anchor_contradict_min: int = 0
    anchor_contradict_window: int = 20
    # the map measurements that corroborate an anchor or contradict it must come from session keyframes at least this far
    # apart (m): consecutive frames see the same aliased place and agree with each other by construction (seen: a
    # correct anchor was dropped on two measurements one step apart).  0: any other observation counts.
    anchor_min_separation: float = 0.0
    # a rejected measurement can contradict the anchor only when the pose it implies for the current keyframe is at least
    # this far (m) from the belief: a sub-metre disagreement is measurement bias, not a wrong anchor (the lock-outs seen
    # were 2-60 m off).  0: no minimum.
    anchor_contradict_min_offset: float = 0.0
    intra_enabled: bool = True
    intra_min_edges: int = 3          # accepted long-range visual edges ...
    intra_window_steps: int = 15      # ... within this many processed steps
    intra_min_loop_steps: int = 300   # a counted edge must reach a keyframe created at least this many steps ago
    intra_min_conf: float = 0.3       # minimum pose-estimator confidence of a counted edge
    # the graph is only optimised when the long-range measurements disagree with the poses the graph implies
    # (residual of the measured relative pose against the current keyframe poses); a consistent revisit is left alone
    intra_min_residual_t: float = 0.10     # metres
    intra_min_residual_r_deg: float = 1.0  # degrees
    intra_cooldown_steps: int = 100   # minimum number of steps between two intra-hypothesis optimisations
    # the residual must also be significant against the edge uncertainty (Mahalanobis, over the 6 dof)
    intra_min_residual_sigma: float = 2.0
    # no intra-hypothesis optimisation for this many steps after a hypothesis merge: the merge already optimised
    # the graph with the (much stiffer) twin constraints of the tracked hypothesis, and re-solving the plain graph,
    # whose visual edges are weak relative to odometry, drags the loop back towards the odometry chain
    intra_after_merge_cooldown_steps: int = 300


@dataclass
class LocalSmoothingConfig:
    enabled: bool = False
    window_kfs: int = 30
    period_steps: int = 10
    k_hop: int = 1


@dataclass
class ClusterStdConfig:
    use_conf_weight: bool = True
    min_std_translation: float = 0.01  # meters
    min_std_rotation: float = 0.01     # radians
    # the std of a multi-member cluster is at least the smallest member std: the references of one forward pass share
    # the pass's gauge and scale errors, so their agreement (dispersion) says little about the error of the cluster;
    # False keeps the original dispersion-only std (floored at min_std_*), which lets correlated, biased measurements
    # dominate the belief
    floor_by_member_std: bool = False


@dataclass
class HypothesisConfig:
    """All tuning parameters for HypothesisManager."""

    conditional_sources: bool = False
    schmidt_map_geometry: bool = False  # experimental shared map uncertainty
    map_geometry_basis: str = 'epoch'  # 'factor' retains raw factor identities across solves
    # --- Observation update ---
    # process noise (std, metres / radians per axis) added to the prior of every component before it is fused with
    # an observation.  0.05 is the original CROSS value: it keeps the Kalman gain away from zero but, with good
    # odometry, lets every (possibly biased) measurement pull the belief by a large fraction of its residual.
    filter_process_std: float = 0.05
    # how the odometry std of a step is added to the belief std in the motion update: "linear" (original: sigma +=
    # sigma_step, i.e. n steps give n sigma_step) or "variance" (sigma^2 += sigma_step^2, independent increments: sqrt(n)
    # sigma_step).  The linear rule inflates the belief between observations (at 10 Hz with an observation every few
    # frames it settles at ~6 cm / 0.7 deg for mm-level odometry), which hands every visual measurement a 20-30 % gain and
    # lets the small biases of the feed-forward estimator accumulate into metres of drift.
    motion_std_accumulation: str = "linear"
    # hypothesis 0's pose (and covariance) is updated only by informative proposals (verified-LC loop candidates, map
    # references of a relocalization session); measurements to the keyframes just behind the robot only weigh the
    # hypotheses and become graph edges (their keyframe poses came from the same belief, so fusing them re-applies the
    # estimator's bias at every observation).  Needs mapping.loop_closure.mode = verified (loop flags).  On by default
    # (2026-09-30): original CROSS relocalization success +0.3 on the hard HSSD variants, +0.2 on OpenLORIS home,
    # TUM 0.79 -> 0.96; CROSS-stereo mapping ATE OpenLORIS office 0.117 -> 0.052 m, cafe 0.52 -> 0.17 m, KITTI 3/3 better,
    # relocalization unchanged.  Known weakness: abrupt on-the-spot turns with noisy odometry (Lone Monk, 23 deg per
    # frame): the lagging hypothesis 0 is out-scored and replaced repeatedly.
    h0_informative_only: bool = True
    # measurements to keyframes of previous sessions (the loaded map) count as informative pose-graph constraints (they are
    # independent of the session's odometry drift); for the pose update of hypothesis 0: "always", or "belief" = only while hypothesis 0 is less certain
    # than the measurement (translation), "off" / False = the loop-candidate test alone
    map_refs_informative: object = False
    # evidence of a component without an aligned proposal: "self" scores its keep-alive self-match (zero residual, i.e.
    # the component's own peak density; original behaviour); "miss" scores it below the weakest supported component
    # (by unmatched_miss_margin nats)
    unmatched_evidence: str = "self"
    unmatched_miss_margin: float = 2.0

    # --- Evidence tracking (LLR) ---
    llr_hist_length: int = 8
    llr_bias: float = 0.1

    # --- Active distribution ---
    active_dist_threshold: float = 1e-3

    # --- Birth: Free → Tracking (Unrealized) ---
    alignment_threshold: float = 1.0
    # adoption of a dominant non-zero hypothesis: when hypothesis 0 has effectively died (weight < adopt_w0_max)
    # and one realized hypothesis holds the belief (weight > adopt_wk_min) for adopt_dominant_steps consecutive
    # observation steps without a loop closure, that hypothesis becomes hypothesis 0 (otherwise a map built
    # on a loop-free trajectory can end with its belief in a component that is not persisted / not used by PGO)
    adopt_dominant_steps: int = 20
    # a dominant hypothesis within adopt_close_dist of hypothesis 0 is the same place with a better pose: the
    # loop-closure detector refuses to merge it (proximity veto), so it is adopted after adopt_close_steps steps
    # instead of adopt_dominant_steps (a swap between two poses of the same place cannot produce a gross error)
    adopt_close_steps: int = 5
    adopt_close_dist: float = 3.0
    adopt_w0_max: float = 0.05
    adopt_wk_min: float = 0.9
    tracking_active_threshold: float = 1e-12
    tracking_floor_weight: float = 1e-8

    # --- Realize: Unrealized → Realized ---
    realize_sum_thresh: float = 0.3
    realize_hitrate_thresh: float = 0.4
    # the hit rate is taken over the frames the component actually observed (hist_valid), not over the fixed
    # 8-slot window: a candidate born a few frames ago needed >= 4 positive slots (~12 steps at the observation
    # cadence) before it could be realized, and only realized candidates can be merged or adopted
    realize_min_frames: int = 2

    # --- Death: Tracking → Free ---
    death_ttl_base: int = 4
    death_ttl_gain: float = 4.0
    ttl_sum_thresh: float = 0.25
    ttl_hitrate_thresh: float = 0.4
    death_ttl_max: int = 20

    # --- Newborn weight mixing ---
    newborn_mix_coeff: float = 0.1
    newborn_conf_power: float = 1.0

    # --- LC detection (overlap-only with confidence guard) ---
    detect_overlap_sum_thresh: float = 2.0
    detect_overlap_hitrate_thresh: float = 0.5
    detect_overlap_rel_margin: float = 1.0
    detect_conf_rel_margin: float = 1.0
    detect_conf_hitrate_thresh: float = 0.5
    session_recovery: bool = False  # experimental historical support in place of chart-distance guard
    chart_aware: bool = False  # keep disconnected coordinate frames separate
    # clear the evidence history (LLR, overlap, confidence, validity) of a component slot when it is freed or reused,
    # so a new hypothesis does not inherit the evidence of the place that held the slot before (from CROSS-mono)
    reset_evidence_on_slot_reuse: bool = False
    # re-run loop-closure detection in the same step when adding the keyframe realized a new branch, instead of
    # waiting for the next observation (from CROSS-mono)
    lc_recheck_after_realization: bool = False
    # Only history entries recorded while the component was active and past its birth frame count as evidence
    # (before this fix, empty history slots of a just-realized hypothesis passed the tests above through the margins,
    # so a hypothesis was merged within one or two observations of its birth -- the mechanism behind the false loop
    # closures seen in the replay traces).  A merge also needs the belief to have moved to the candidate.
    detect_min_frames: int = 3            # valid evidence frames required in the history window
    # per-frame evidence is the net log-likelihood ratio of the candidate against hypothesis 0, clamped to
    # +-detect_llr_cap nats; frames in which hypothesis 0 explains the observations better count *against* the
    # candidate (the one-sided sum used before ignored them), and a lost hypothesis 0 makes every frame +cap
    detect_llr_cap: float = 5.0
    detect_min_weight: float = 0.5        # minimum current weight of the candidate hypothesis
    # relocalization (hypothesis 0 not yet anchored to the stored map): the per-frame evidence of a candidate is its
    # log-likelihood against the best *other* place (any active component farther than reloc_unique_min_dist, hypothesis 0
    # included), not against the unsupported hypothesis 0 alone.  Against a lost hypothesis 0 every supported frame is
    # +cap, so an aliased place supported in 3 of 8 frames was merged even while other places were supported just as
    # often; with the unique evidence only frames in which the candidate explains the observations better than every
    # competing place count.  reloc_min_frames (0: detect_min_frames) is the evidence frames required in that state.
    reloc_unique_evidence: bool = False
    reloc_unique_min_dist: float = 3.0
    reloc_min_frames: int = 0
    # Loaded-map recovery with sparse verification (monocular): while the session is not yet joined to the map, a
    # component in another chart than hypothesis 0 is scored by map-association evidence instead of by its Gaussian
    # overlap relative to hypothesis 0 (whose own-session observations say nothing about map association). A frame
    # with a verified loaded-map edge contributes reloc_consistency_nats - 0.5 * Mahalanobis(residual); a frame
    # without one contributes log(1 - reloc_detection_prob) (a missed detection). The sequential sum over the
    # evidence window then realizes and commits the branch; at least reloc_min_verified_frames verified frames from
    # two distinct map keyframes are required. Requires session_recovery with chart_aware.
    # Innovation gate for hypothesis 0: a proposal whose Mahalanobis residual against hypothesis 0 (prior covariance plus
    # filter process noise plus proposal covariance, 6 dof) exceeds this chi-square value is not fused into hypothesis 0;
    # it may still feed or seed another hypothesis, which delayed commitment resolves (0 disables). Applies where the
    # verified loop closure has no odometry-chain prediction, e.g. saved-map references after the map join.
    h0_innovation_gate: float = 0.0
    reloc_association_evidence: bool = False
    reloc_detection_prob: float = 0.3
    reloc_consistency_nats: float = 3.0
    reloc_min_verified_frames: int = 3
    reloc_realize_nats: float = 2.0
    detect_reject_cooldown_steps: int = 30   # steps a candidate is ignored after a rejected merge
    # geometric verification of a merge: fraction of the candidate's visual edges that remain outliers
    # (Mahalanobis norm > verify_outlier_sigma) after the loop-closure optimisation
    verify_outlier_sigma: float = 4.0
    verify_max_outlier_frac: float = 0.5

    # --- Self LC detection (comp 0) ---
    self_lc_conf_thresh: float = 0.55
    self_lc_cooldown_steps: int = 300
    self_lc_min_kf_id_gap: int = 30

    # --- Misc ---
    no_pgo_for_lc: bool = False

    # --- Comp 0 keep-alive (observability) ---
    comp0_keepalive_score: float = 1e-2
    comp0_sigma_inflation_factor: float = 1.2
    comp0_weight_floor: float = 1e-2

    # --- Visualization ---
    visualize_pose_graph: bool = False


@dataclass
class TopoConfig:
    """Configuration for SimpleTopo (proximity graph for planning)."""
    proximity_distance_thresh: float = 0.5
    proximity_std_trans: float = 0.05
    proximity_std_rot: float = 0.1
    use_proximity_grid: bool = False
    enable_incremental_proximity: bool = False
    # the proximity graph serves the planner only; its refresh after every pose-graph optimisation is O(N^2) with a GPU
    # sync per keyframe (~10 s per optimisation at 2000 keyframes) and assumes a planar (x, z) layout
    enabled: bool = True


@dataclass
class MappingConfig:
    kf_gmm_n_components: int = 5
    kf_retrieval_threshold_new_kf: float = 0.75
    kf_match_threshold_new_kf: int = 50
    new_component_weight_threshold: float = 0.2
    cluster_eps: float = 1.0             # DBSCAN radius (x, z, yaw) for proposal clustering
    loop_closure: LoopClosureConfig = field(default_factory=LoopClosureConfig)
    local_smoothing: LocalSmoothingConfig = field(default_factory=LocalSmoothingConfig)
    cluster_std: ClusterStdConfig = field(default_factory=ClusterStdConfig)
    hypothesis: HypothesisConfig = field(default_factory=HypothesisConfig)
    topo: TopoConfig = field(default_factory=TopoConfig)


@dataclass
class PGOConfig:
    std_reduction_factor: float = 0.3
    visual_robust_enabled: bool = True
    visual_robust_type: RobustKernelType = RobustKernelType.HUBER
    visual_robust_delta: float = 1.0


@dataclass
class KPDetectorConfig:
    type: KPDetectorType = KPDetectorType.XFEAT
    n_keypoints: int = 300
    detection_threshold: float = 0.1


@dataclass
class KPMatcherConfig:
    type: KPMatcherType = KPMatcherType.LIGHTGLUE
    min_conf: float = 0.7
    allow_batch_inference: bool = True


@dataclass
class FeedForwardConfig:
    """Stereo mode: feed-forward (learned multi-view) relative pose estimation with stereo scale anchors
    (pose_est.type = ff; needs the optional stereo dependencies, see install.sh --stereo).  Observation gating and
    the calibrated measurement noise are estimator-generic options of PoseEstConfig."""
    backend: FFBackend = FFBackend.VGGT_OMEGA
    checkpoint: str = "models/VGGT-Omega/vggt_omega_1b_512.pt"
    image_resolution: int = 512          # longest side fed to the model (multiple of patch size)
    da3_process_res: int = 504
    half_precision_weights: bool = True  # keep the transformer weights in bf16 (halves memory, ~1cm difference)
    max_refs: int = 6                    # at most this many retrieved references per forward pass
    n_ref_anchors: int = 2               # stored right images of the best references used as extra anchors
    use_curr_anchor: bool = True         # include the current right image (ablation switch)
    store_right_images: bool = False     # keep right images of keyframes even when n_ref_anchors == 0
    use_odom_anchor: bool = False        # previous frame + odometry as an additional metric anchor
    odom_anchor_min_translation: float = 0.15
    odom_anchor_weight: float = 0.5
    scale_method: str = "adaptive"       # adaptive | huber_log | median | mean | norm_ls
    # map-consistency anchors: pairs of retrieved keyframes whose metric relative pose is known from the map act as
    # long-baseline scale anchors in the same forward pass (the stereo pair alone has a baseline that is tiny
    # relative to large scenes, which biases the recovered scale by several percent)
    use_map_anchors: bool = False
    map_anchor_weight: float = 1.0
    map_anchor_min_dist: float = 0.5     # metres between the two keyframes
    map_anchor_max_dist: float = 60.0
    map_anchor_max_pairs: int = 8
    anchor_weight_by_baseline: bool = True   # weight anchors by predicted baseline length (precision of the ratio)
    anchor_max_rot_err_deg: float = 20.0
    anchor_min_dir_cos: float = 0.5
    covis_grid: int = 48
    covis_depth_tol: float = 0.15
    covis_symmetric: bool = False
    min_covis: float = 0.15              # validity threshold on the covisibility confidence
    max_rel_distance: float = 40.0       # reject relative poses further than this (m)
    kf_conf_threshold_new_kf: float = 0.35  # covis below this -> current view is novel -> permanent keyframe
    scale_std_inflation: bool = True     # inflate translation std by |t| * relative scale std
    # base measurement std [tx, ty, tz, rx, ry, rz] of the feed-forward estimator (None: the PnP default
    # [0.2, 0.2, 0.3, 0.2, 0.2, 0.2]); it is divided by 4 * covisibility * retrieval score per reference
    base_measurement_std: Optional[List[float]] = None


@dataclass
class PoseEstConfig:
    type: PoseEstType = PoseEstType.PNP
    kp_detector: KPDetectorConfig = field(default_factory=KPDetectorConfig)
    kp_matcher: KPMatcherConfig = field(default_factory=KPMatcherConfig)
    kf_match_threshold: int = 10
    inlier_count_threshold: int = 10
    max_depth: float = 30.0
    # measurement std of the observation update from the calibrated noise model of the verified loop closure
    # (sigma = visual_*_a + visual_*_b |t|, times the online noise scale of the reference type; see
    # scripts/lc/calibrate_noise.py) instead of the heuristic 0.2 / (4 inlier ratio retrieval score).  Needs
    # mapping.loop_closure.mode = verified.
    meas_std_from_noise_model: bool = False
    # observation cadence: retrieval + relative pose estimation are skipped until the robot moved at least
    # obs_min_translation (m) or turned obs_min_rotation (rad) since the last observation, or obs_max_interval_steps
    # frames elapsed; the motion model carries the belief in between (odometry is cheap, the observation is not).
    # 0 / 0 / 1 = observe every frame (original behaviour).  Every frame is observed for obs_warmup_steps frames after
    # (re)initialisation so that a relocalization starts quickly.  Runtime option (off by default): with 0.3 m / 0.15 rad /
    # 3 frames, mapping and relocalization run 1.4-1.8x faster (HSSD house 4.1 -> 5.8 FPS, accuracy unchanged), but on
    # the real OpenLORIS home sessions relocalization success dropped 0.83 -> 0.70 and the map ATE rose 0.13 -> 0.20 m.
    obs_min_translation: float = 0.0
    obs_min_rotation: float = 0.0
    obs_max_interval_steps: int = 1
    obs_warmup_steps: int = 10
    # adaptive cadence (off with 0): while one hypothesis holds at least obs_confident_weight of the belief and the last
    # observation was verified against existing keyframes without adding a permanent one (the place is mapped and the
    # robot is localized in it), observe only after these larger intervals.  Mapping of new places is unaffected.
    obs_confident_max_interval_steps: int = 0
    obs_confident_min_translation: float = 0.6
    obs_confident_min_rotation: float = 0.3
    obs_confident_weight: float = 0.9
    ff: FeedForwardConfig = field(default_factory=FeedForwardConfig)


@dataclass
class DepthPredConfig:
    use_depth_pred: bool = False


@dataclass
class VisualizationConfig:
    visualize_pointcloud: bool = False
    visualize_current_gmm_state: bool = True
    visualize_keyframe_gmms: bool = True
    visualize_system_data: bool = True
    visualize_trajectory: bool = True
    visualize_camera: bool = True
    visualize_pinhole_camera: bool = False
    visualize_keyframe_gmm_trajectory: bool = True
    visualize_hypotheses: bool = True
    visualize_odom_trajectory: bool = True


@dataclass
class SystemConfig:
    """Root configuration for the CROSS system."""
    async_update: bool = False
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    pgo: PGOConfig = field(default_factory=PGOConfig)
    pose_est: PoseEstConfig = field(default_factory=PoseEstConfig)
    depth_pred: DepthPredConfig = field(default_factory=DepthPredConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


# ---------------------------------------------------------------------------
# Helpers: dict ↔ dataclass conversion
# ---------------------------------------------------------------------------

T = TypeVar("T")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_type(field_type):
    """Resolve the actual type, handling Optional and string annotations."""
    # Handle Optional[X] -> X
    origin = getattr(field_type, "__origin__", None)
    if origin is Union:
        args = [a for a in field_type.__args__ if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return field_type


def _from_dict(cls: Type[T], data: dict) -> T:
    """Recursively convert a plain dict into a dataclass instance of *cls*.

    Handles nested dataclasses and Enum fields automatically.
    Unknown keys in *data* are silently ignored so that forward-compatible
    YAML files don't break older code.
    """
    if not isinstance(data, dict):
        return data  # type: ignore[return-value]

    field_map = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: Dict[str, Any] = {}
    for name, fld in field_map.items():
        # Handle YAML key mapping: async_ <-> async
        yaml_key = name.rstrip("_") if name.endswith("_") else name
        if yaml_key in data:
            raw = data[yaml_key]
        elif name in data:
            raw = data[name]
        else:
            continue  # use default

        ftype = _resolve_type(fld.type) if not isinstance(fld.type, str) else fld.type

        # Resolve string annotations (forward references)
        if isinstance(ftype, str):
            ftype = eval(ftype)  # noqa: S307 – safe, only our own type names

        if dataclasses.is_dataclass(ftype) and isinstance(raw, dict):
            kwargs[name] = _from_dict(ftype, raw)
        elif isinstance(ftype, type) and issubclass(ftype, Enum):
            kwargs[name] = ftype(raw) if not isinstance(raw, ftype) else raw
        else:
            kwargs[name] = raw

    return cls(**kwargs)


def _to_dict(obj) -> dict:
    """Convert a dataclass instance to a plain dict (recursive, enum → value)."""
    if not dataclasses.is_dataclass(obj):
        return obj
    result = {}
    for fld in dataclasses.fields(obj):
        value = getattr(obj, fld.name)
        key = fld.name.rstrip("_") if fld.name.endswith("_") else fld.name
        if dataclasses.is_dataclass(value):
            result[key] = _to_dict(value)
        elif isinstance(value, Enum):
            result[key] = value.value
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(*yaml_paths: Union[str, Path], **overrides) -> SystemConfig:
    """Load and deep-merge one or more YAML files into a :class:`SystemConfig`.

    Parameters
    ----------
    *yaml_paths
        Zero or more paths to YAML config files.  Files are merged left to
        right (later files override earlier ones).
    **overrides
        Top-level section overrides as dicts, e.g.
        ``tracking={"filter_mode": "adaptive"}``.  Merged last (highest
        priority).

    Returns
    -------
    SystemConfig
        Fully resolved, typed configuration object.

    Examples
    --------
    >>> cfg = load_config("configs/default.yaml")
    >>> cfg = load_config("configs/default.yaml", "configs/exp/indoor.yaml",
    ...                   tracking={"filter_mode": "adaptive"})
    """
    merged: dict = {}

    for path in yaml_paths:
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)

    # Apply keyword overrides (each key is a top-level section name or scalar)
    if overrides:
        merged = _deep_merge(merged, overrides)

    return _from_dict(SystemConfig, merged) if merged else SystemConfig()


def config_to_yaml(cfg: SystemConfig) -> str:
    """Serialize a :class:`SystemConfig` to a YAML string."""
    return yaml.dump(_to_dict(cfg), default_flow_style=False, sort_keys=False)


def config_to_dict(cfg: SystemConfig) -> dict:
    """Serialize a :class:`SystemConfig` to a plain dict."""
    return _to_dict(cfg)
