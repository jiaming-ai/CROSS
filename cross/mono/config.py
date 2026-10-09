from dataclasses import dataclass, field

from cross.imu.scale_filter import ImuConfig


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
    # dpvo frontend: take metric depth from the input frames (sensor depth of an RGB-D camera, or stereo depth) for the
    # scale observations and the mapping depth, instead of a learned metric depth model (--odometry visual in the
    # RGB-D and stereo modes)
    depth_input: bool = False
    dpvo_metric_bootstrap: bool = False
    mask_people: bool = False
    mask_interval: int = 1
    rotation_selection: bool = False
    subpixel: bool = False
    mapping_process: bool = False
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
    # retrieval_pose == "ff": feed-forward multi-view relative poses (backend da3 | vggt_omega)
    ff_backend: str = "da3"
    ff_checkpoint: str = "depth-anything/DA3-LARGE-1.1"
    ff_resolution: int = 504
    ff_min_covisibility: float = 0.3
    ff_fallback_only: bool = False  # run the feed-forward model only on references the two-view matcher rejects
    ff_scope: str = "all"  # with ff_fallback_only: all | map (loaded-map references) | relocalization (map, before the join)
    session_recovery: bool = False
    chart_aware: bool = False
    historical_retrieval_slots: int = 0
    historical_min_score: float | None = None
    translation_std_floor: float = 0.005
    rotation_std_floor: float = 0.01
    # streaming_dpvo only: per-frame motion std proportional to the frame's own translation / rotation
    translation_std_per_meter: float = 0.0
    # streaming_dpvo only: frames with fewer FAST corners (320x240, threshold 20) are degenerate views (e.g. a blank
    # wall at close range); their DPVO motion is discarded as unknown (0 disables)
    min_texture_corners: int = 0
    degenerate_mode: str = "hold"
    # dpvo frontend: thumbnail correlation below this between consecutive frames triggers a feed-forward overlap check;
    # the jump is bridged by the learned relative pose (or marked unknown motion) and DPVO restarts (0 disables)
    discontinuity_ncc: float = 0.0  # hold: discard the motion; inflate: keep DPVO motion but report it as invalid
    rotation_std_per_radian: float = 0.0
    max_relative_rotation: float = 1.2
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    # dpvo frontend: metric scale from the IMU (cross.imu.scale_filter; frames carry `imu`), learned depth optional
    imu: ImuConfig = field(default_factory=ImuConfig)
    # 'dotted.path=value' overrides of the CROSS SystemConfig, applied after the monocular defaults
    cross_overrides: tuple = ()

    def __post_init__(self):
        if self.frontend == 'streaming_dpvo':
            if not self.dpvo_checkpoint or self.scale.mode == 'relative':
                raise ValueError('Streaming metric DPVO requires a checkpoint and a metric scale mode')
            if self.dpvo_metric_bootstrap:
                raise ValueError('Metric depth bootstrap is not supported by streaming_dpvo')
        if self.imu.enabled and (self.frontend != "dpvo" or self.depth_input or self.scale.mode == "relative"):
            raise ValueError("IMU scale is supported by the synchronous dpvo frontend without input depth")
        if self.depth_input and (self.frontend != "dpvo" or self.scale.mode == "relative"):
            raise ValueError("Input depth is supported by the synchronous dpvo frontend with a metric scale mode")
        if self.chart_aware and not self.session_recovery:
            raise ValueError("Chart-aware mapping requires session_recovery")
        if self.frontend not in {"da3", "dpvo", "metric_pnp", "rotation_metric", "learned_rotation_pnp", "metric_klt", "streaming_dpvo"}:
            raise ValueError("Unknown monocular frontend")
        if min(self.resolution, self.metric_resolution) < 56:
            raise ValueError("Model resolutions must be at least 56 pixels")
        if min(self.anchor_interval, self.mapping_interval, self.scale.interval, self.mask_interval) < 1:
            raise ValueError("Intervals must be positive")
        if self.recent_frames < 1:
            raise ValueError("At least one recent frame is required")
        if self.historical_retrieval_slots < 0:
            raise ValueError("Historical retrieval slots cannot be negative")
        if self.historical_min_score is not None:
            if not 0 <= self.historical_min_score <= 1 or not self.historical_retrieval_slots:
                raise ValueError("Historical minimum score needs reserved slots and must be within [0,1]")
        if self.pose_refinement not in {"none", "xfeat"}:
            raise ValueError("pose_refinement must be none or xfeat")
        if self.retrieval_pose not in {"da3", "metric_pnp", "ff"}:
            raise ValueError("Unknown retrieval pose estimator")
        if self.retrieval_pose == "ff" and (self.ff_backend not in {"da3", "vggt_omega"}
                                            or not 0 <= self.ff_min_covisibility <= 1):
            raise ValueError("Feed-forward retrieval needs backend da3 or vggt_omega and a covisibility in [0,1]")
        if self.scale.mode not in {"filtered", "direct", "initial", "relative"}:
            raise ValueError(f"Unknown scale mode: {self.scale.mode}")
