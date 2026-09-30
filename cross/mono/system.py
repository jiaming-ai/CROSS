"""Image-only entry point into the original CROSS topological mapper."""

from time import perf_counter

import cv2
import numpy as np

from cross.pipeline import Pipeline

from .config import MonoConfig
from .frontend import MonoFrontend


def apply_overrides(cfg, overrides):
    """Set 'dotted.path=value' entries on a nested config; values are parsed as YAML scalars."""
    import yaml
    for item in overrides:
        path, _, text = item.partition("=")
        if not _:
            raise ValueError(f"Override needs key=value: {item}")
        *parents, name = path.strip().split(".")
        target = cfg
        for part in parents:
            target = getattr(target, part)
        if not hasattr(target, name):
            raise ValueError(f"Unknown CROSS config key: {path}")
        setattr(target, name, yaml.safe_load(text))


def attach_pair_motion(frontend, pose_estimator, config, K):
    """Give a DPVO frontend the feed-forward overlap check it uses across view discontinuities."""
    if config.discontinuity_ncc <= 0 or not hasattr(frontend, "pair_motion"):
        return
    feed_forward = getattr(pose_estimator, "feed_forward", pose_estimator)
    if not hasattr(feed_forward, "model") or getattr(frontend, "metric", None) is None:
        raise ValueError("Discontinuity bridging needs --retrieval-pose ff and a metric depth model")
    from .ff_retrieval import pair_motion
    frontend.pair_motion = lambda previous, current: pair_motion(
        feed_forward, previous, current, frontend.metric.predict_metric(previous, K, previous.shape[:2]))


class MonocularSystem(Pipeline):
    """The mono mode: a monocular frontend (DPVO or another motion source) and the CROSS back end with learned metric
    depth and an injected monocular relative-pose estimator.

    pose_estimator: reuse the estimator of an earlier session (its models); mono_defaults: apply the monocular back-end
    defaults (filter mode, visual-odometry motion noise, top_k 3) on top of system_config (default: only when no
    system_config is given); external_odometry: the frontend passes on dataset odometry, keep the back end's default
    odometry noise model."""

    def __init__(self, K, image_size, config=None, system_config=None, device="cuda", frontend=None,
                 pose_estimator=None, mono_defaults=None, external_odometry=False):
        if frontend is None and getattr(config, "frontend", None) in {"streaming_pnp", "streaming_dpvo"}:
            raise ValueError("Use StreamingMonocularSystem for a streaming frontend")
        from cross.core.config import FilterMode, PoseEstType, SystemConfig
        from cross.core.system import System
        from cross.core.types import Camera
        from .retrieval import DA3RelativePose, MetricRelativePose
        from .two_view_retrieval import MetricTwoViewRelativePose

        self.config = config or MonoConfig()
        if frontend is not None:
            self.frontend = frontend
        elif self.config.frontend == "dpvo":
            from .dpvo_frontend import DPVOFrontend
            self.frontend = DPVOFrontend(K, self.config, device)
        elif self.config.frontend in {"metric_pnp", "rotation_metric", "learned_rotation_pnp", "metric_klt"}:
            from .pnp_frontend import MetricPnPFrontend, RotationMetricFrontend, LearnedRotationPnPFrontend, MetricKLTFrontend
            factory = {"metric_pnp": MetricPnPFrontend, "rotation_metric": RotationMetricFrontend,
                       "learned_rotation_pnp": LearnedRotationPnPFrontend, "metric_klt": MetricKLTFrontend}[self.config.frontend]
            self.frontend = factory(K, self.config, device)
        else:
            self.frontend = MonoFrontend(K, self.config, device)
        cfg = system_config or SystemConfig()
        if cfg.async_update:
            raise ValueError("MonocularSystem currently requires synchronous mapping updates")
        cfg.tracking.use_odometry = True  # internal visual increments, never input odometry
        cfg.tracking.use_VO = False
        cfg.depth_pred.use_depth_pred = False
        cfg.pose_est.type = PoseEstType.DA3 if self.config.retrieval_pose == "da3" else PoseEstType.METRIC_PNP
        # CROSS-mono ran with these two core behaviours always on (flag-gated in the merged core)
        cfg.mapping.hypothesis.reset_evidence_on_slot_reuse = True
        cfg.mapping.hypothesis.lc_recheck_after_realization = True
        # and with hypothesis 0 updated by every observation (the base default became informative-only; override with
        # --cross-config mapping.hypothesis.h0_informative_only=true)
        cfg.mapping.hypothesis.h0_informative_only = False
        if self.config.session_recovery:
            cfg.mapping.hypothesis.session_recovery = True
        cfg.mapping.hypothesis.chart_aware = self.config.chart_aware
        cfg.mapping.hypothesis.conditional_sources = self.config.conditional_sources
        cfg.mapping.hypothesis.schmidt_map_geometry = self.config.schmidt_map_geometry
        cfg.mapping.hypothesis.map_geometry_basis = self.config.map_geometry_basis
        if self.config.conditional_sources and (cfg.mapping.loop_closure.async_ or cfg.mapping.hypothesis.no_pgo_for_lc):
            raise ValueError('Conditional sources require synchronous graph optimization in the mapping worker')
        if mono_defaults is None:
            mono_defaults = system_config is None
        if mono_defaults:
            cfg.tracking.filter_mode = FilterMode(self.config.filter_mode)
            cfg.tracking.odom_min_std_translation = 0.005
            cfg.tracking.odom_min_std_rotation = 0.01
            cfg.tracking.odom_std_per_meter = 0.1
            cfg.tracking.odom_std_per_radian = 0.1
            cfg.retrieval.top_k = 3
        overrides = self.config.cross_overrides
        if external_odometry:
            # dataset odometry: the back end's default odometry noise (as in the rgbd / stereo modes), not the DPVO one
            defaults = SystemConfig()
            cfg.tracking.odom_std_per_meter = defaults.tracking.odom_std_per_meter
            cfg.tracking.odom_std_per_radian = defaults.tracking.odom_std_per_radian
            cfg.tracking.odom_min_std_translation = defaults.tracking.odom_min_std_translation
            cfg.tracking.odom_min_std_rotation = defaults.tracking.odom_min_std_rotation
            overrides = tuple(o for o in overrides if not o.startswith(("mapping.loop_closure.noise.odom", "tracking.")))
        apply_overrides(cfg, overrides)
        cfg.retrieval.historical_slots = self.config.historical_retrieval_slots
        cfg.retrieval.historical_min_score = self.config.historical_min_score
        if cfg.retrieval.historical_slots > cfg.retrieval.top_k:
            raise ValueError("Historical retrieval slots cannot exceed the total retrieval budget")
        camera = Camera(np.array(K).copy(), *image_size)
        # System rescales camera.K in place to its stored-image resolution.
        metric_adapter = MetricTwoViewRelativePose if self.config.retrieval_pose == "metric_two_view" else MetricRelativePose
        if pose_estimator is not None:
            pass
        elif self.config.retrieval_pose == "ff":
            from .ff_retrieval import FallbackFeedForwardRelativePose, FeedForwardRelativePose
            pose_estimator = FeedForwardRelativePose(self.config.ff_backend, self.config.ff_checkpoint, device,
                                                     self.config.ff_resolution, self.config.ff_min_covisibility)
            if self.config.ff_fallback_only:
                pose_estimator = FallbackFeedForwardRelativePose(
                    MetricTwoViewRelativePose(camera.K, device, self.config.mask_people, "superpoint_lightglue"),
                    pose_estimator, scope=self.config.ff_scope)
        elif self.config.retrieval_pose == "da3":
            pose_estimator = DA3RelativePose(self.frontend.geometry, device)
        else:
            pose_estimator = metric_adapter(camera.K, device, self.config.mask_people, self.config.retrieval_matcher,
                                            self.config.conditional_sources, self.config.source_log_std,
                                            **(dict(rotation_geometry=self.frontend.geometry)
                                               if self.config.two_view_rotation_check else {}))
        mapper = System(device=device, visualize=False, camera=camera, config=cfg, pose_estimator=pose_estimator)
        attach_pair_motion(self.frontend, pose_estimator, self.config, K)
        super().__init__(mapper, self.frontend, self.config.mapping_interval, mode="mono",
                         odometry="external" if external_odometry else "visual",
                         depth_model=getattr(self.frontend, "metric", None), K=np.array(K, dtype=np.float64))

    def step(self, rgb, timestamp):
        estimate, map_now = self.advance({"rgb": rgb, "timestamp": timestamp})
        estimate.diagnostics.update(
            mapping_seconds=self.last_mapping_seconds,
            mapping_update=map_now,
            permanent_keyframes=self.mapper.db.get_size(),
            graph_nodes=len(self.mapper.hypothesis_manager.nodes),
            hypotheses=len(self.mapper.hypothesis_manager.hypotheses),
        )
        if map_now:
            estimate.diagnostics["mapping_event"] = getattr(self.mapper, "last_step_diagnostics", {}).copy()
        self.last_estimate = estimate
        return estimate

    def save_map(self, path):
        self.mapper.save_map(str(path))

    def load_map(self, path):
        if self.initialized:
            raise RuntimeError("Load a map before processing images of a new session")
        self.mapper.load_map(str(path))

    def shutdown(self):
        self.mapper.shutdown()
