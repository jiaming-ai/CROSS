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
    translation_std_floor: float = 0.005
    rotation_std_floor: float = 0.01
    max_relative_rotation: float = 1.2
    scale: ScaleConfig = field(default_factory=ScaleConfig)

    def __post_init__(self):
        if self.frontend not in {"da3", "dpvo", "metric_pnp", "rotation_metric"}:
            raise ValueError("frontend must be da3, dpvo, metric_pnp or rotation_metric")
        if min(self.resolution, self.metric_resolution) < 56:
            raise ValueError("Model resolutions must be at least 56 pixels")
        if min(self.anchor_interval, self.mapping_interval, self.scale.interval, self.mask_interval) < 1:
            raise ValueError("Intervals must be positive")
        if self.recent_frames < 1:
            raise ValueError("At least one recent frame is required")
        if self.pose_refinement not in {"none", "xfeat"}:
            raise ValueError("pose_refinement must be none or xfeat")
        if self.retrieval_pose not in {"da3", "metric_pnp"}:
            raise ValueError("retrieval_pose must be da3 or metric_pnp")
        if self.scale.mode not in {"filtered", "direct", "initial", "relative"}:
            raise ValueError(f"Unknown scale mode: {self.scale.mode}")
