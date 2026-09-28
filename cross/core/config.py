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
from typing import Any, Dict, Optional, Type, TypeVar, Union

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
    filter_mode: FilterMode = FilterMode.FULL
    adaptive_filter: AdaptiveFilterConfig = field(default_factory=AdaptiveFilterConfig)


@dataclass
class RetrievalConfig:
    vpr_model_type: VPRModelType = VPRModelType.BOQ
    top_k: int = 10
    vpr_score_threshold_high: float = 0.3
    vpr_score_threshold_low: float = 0.3
    initial_buffer_size: int = 1000


@dataclass
class LoopClosureConfig:
    async_: bool = False
    queue_size: int = 1


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


@dataclass
class HypothesisConfig:
    """All tuning parameters for HypothesisManager."""

    # --- Evidence tracking (LLR) ---
    llr_hist_length: int = 8
    llr_bias: float = 0.1

    # --- Active distribution ---
    active_dist_threshold: float = 1e-3

    # --- Birth: Free → Tracking (Unrealized) ---
    alignment_threshold: float = 1.0
    tracking_active_threshold: float = 1e-12
    tracking_floor_weight: float = 1e-8

    # --- Realize: Unrealized → Realized ---
    realize_sum_thresh: float = 0.3
    realize_hitrate_thresh: float = 0.4

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


@dataclass
class MappingConfig:
    kf_gmm_n_components: int = 5
    kf_retrieval_threshold_new_kf: float = 0.75
    kf_match_threshold_new_kf: int = 50
    new_component_weight_threshold: float = 0.2
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
class PoseEstConfig:
    type: PoseEstType = PoseEstType.PNP
    kp_detector: KPDetectorConfig = field(default_factory=KPDetectorConfig)
    kp_matcher: KPMatcherConfig = field(default_factory=KPMatcherConfig)
    kf_match_threshold: int = 10
    inlier_count_threshold: int = 10
    max_depth: float = 30.0


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
