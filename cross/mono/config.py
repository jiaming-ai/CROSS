from dataclasses import dataclass, field


@dataclass
class ScaleConfig:
    interval: int = 30
    mode: str = "filtered"  # filtered, direct, initial, relative
    min_pixels: int = 128
    min_tiles: int = 8
    min_inlier_fraction: float = 0.55
    max_log_mad: float = 0.35
    observation_std_floor: float = 0.12
    process_std_per_frame: float = 0.006
    posterior_std_floor: float = 0.08
    innovation_gate: float = 3.5
    recovery_observations: int = 3  # zero reproduces permanent innovation rejection

    def __post_init__(self):
        if self.recovery_observations < 0 or self.recovery_observations == 1:
            raise ValueError("Scale recovery needs at least two observations, or zero to disable")


@dataclass
class MonoConfig:
    frontend: str = "da3"
    dpvo_checkpoint: str | None = None
    dpvo_metric_bootstrap: bool = False
    mask_people: bool = False
    mask_interval: int = 1
    rotation_selection: bool = False
    subpixel: bool = False
    mapping_process: bool = False
    delayed_recovery: bool = False
    teacher_lag_frames: int = 0
    adaptive_anchor: bool = False
    stable_teacher_cadence: bool = False
    trace_metric_sources: bool = False
    conditional_sources: bool = False
    source_log_std: float = .12
    seed: int = 0
    pose_model: str = "depth-anything/DA3-SMALL"
    metric_model: str = "depth-anything/DA3METRIC-LARGE"
    resolution: int = 336
    metric_resolution: int = 504
    anchor_interval: int = 30
    recent_frames: int = 2
    pose_refinement: str = "none"  # none or xfeat
    refinement_anchor_only: bool = False
    metric_shape: bool = False
    mapping_interval: int = 5
    retrieval_pose: str = "da3"
    retrieval_matcher: str = "mnn"
    filter_mode: str = "full"
    session_recovery: bool = False
    chart_aware: bool = False
    historical_retrieval_slots: int = 0
    historical_min_score: float | None = None
    translation_std_floor: float = 0.005
    rotation_std_floor: float = 0.01
    max_relative_rotation: float = 1.2
    scale: ScaleConfig = field(default_factory=ScaleConfig)

    def __post_init__(self):
        if self.stable_teacher_cadence and self.frontend != 'streaming_pnp':
            raise ValueError('Stable teacher cadence requires streaming_pnp')
        if self.conditional_sources:
            if self.frontend != 'streaming_pnp' or self.retrieval_pose not in {'metric_pnp', 'metric_two_view'} or not self.chart_aware:
                raise ValueError('Conditional sources require streaming_pnp, metric retrieval and chart-aware mapping')
            self.trace_metric_sources = True
        if not 0 <= self.source_log_std <= 1:
            raise ValueError('Source log standard deviation must be finite and within [0,1]')
        if self.trace_metric_sources and self.frontend != "streaming_pnp":
            raise ValueError("Metric source tracing requires streaming_pnp")
        if self.chart_aware and not self.session_recovery:
            raise ValueError("Chart-aware mapping requires session_recovery")
        if self.frontend not in {"da3", "dpvo", "metric_pnp", "rotation_metric", "metric_klt", "streaming_pnp"}:
            raise ValueError("Unknown monocular frontend")
        if min(self.resolution, self.metric_resolution) < 56:
            raise ValueError("Model resolutions must be at least 56 pixels")
        if min(self.anchor_interval, self.mapping_interval, self.scale.interval, self.mask_interval) < 1:
            raise ValueError("Intervals must be positive")
        if self.recent_frames < 1:
            raise ValueError("At least one recent frame is required")
        if self.teacher_lag_frames < 0:
            raise ValueError("Teacher delivery lag cannot be negative")
        if self.historical_retrieval_slots < 0:
            raise ValueError("Historical retrieval slots cannot be negative")
        if self.historical_min_score is not None:
            if not 0 <= self.historical_min_score <= 1 or not self.historical_retrieval_slots:
                raise ValueError("Historical minimum score needs reserved slots and must be within [0,1]")
        if self.retrieval_matcher not in {"mnn", "lighterglue", "superpoint_lightglue"}:
            raise ValueError("Unknown retrieval matcher")
        if self.retrieval_matcher != "mnn" and self.retrieval_pose not in {"metric_pnp", "metric_two_view"}:
            raise ValueError("Learned retrieval matchers require metric retrieval")
        if self.retrieval_pose == "metric_two_view" and self.retrieval_matcher != "superpoint_lightglue":
            raise ValueError("Experimental two-view retrieval requires superpoint_lightglue")
        if self.pose_refinement not in {"none", "xfeat"}:
            raise ValueError("pose_refinement must be none or xfeat")
        if self.retrieval_pose not in {"da3", "metric_pnp", "metric_two_view"}:
            raise ValueError("Unknown retrieval pose estimator")
        if self.filter_mode not in {"full", "skip_active", "adaptive"}:
            raise ValueError("Unknown CROSS retrieval filter mode")
        if self.scale.mode not in {"filtered", "direct", "initial", "relative"}:
            raise ValueError(f"Unknown scale mode: {self.scale.mode}")
