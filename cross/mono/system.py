"""Image-only entry point into the original CROSS topological mapper."""

from time import perf_counter

import cv2
import numpy as np

from .config import MonoConfig
from .frontend import MonoFrontend
from .geometry import inverse


class MonocularSystem:
    def __init__(self, K, image_size, config=None, system_config=None, device="cuda", frontend=None):
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
        if self.config.session_recovery:
            cfg.mapping.hypothesis.session_recovery = True
        cfg.mapping.hypothesis.chart_aware = self.config.chart_aware
        cfg.mapping.hypothesis.conditional_sources = self.config.conditional_sources
        cfg.mapping.hypothesis.schmidt_map_geometry = self.config.schmidt_map_geometry
        cfg.mapping.hypothesis.map_geometry_basis = self.config.map_geometry_basis
        if self.config.conditional_sources and (cfg.mapping.loop_closure.async_ or cfg.mapping.hypothesis.no_pgo_for_lc):
            raise ValueError('Conditional sources require synchronous graph optimization in the mapping worker')
        if system_config is None:
            cfg.tracking.filter_mode = FilterMode(self.config.filter_mode)
            cfg.tracking.odom_min_std_translation = 0.005
            cfg.tracking.odom_min_std_rotation = 0.01
            cfg.tracking.odom_std_per_meter = 0.1
            cfg.tracking.odom_std_per_radian = 0.1
            cfg.retrieval.top_k = 3
        cfg.retrieval.historical_slots = self.config.historical_retrieval_slots
        cfg.retrieval.historical_min_score = self.config.historical_min_score
        if cfg.retrieval.historical_slots > cfg.retrieval.top_k:
            raise ValueError("Historical retrieval slots cannot exceed the total retrieval budget")
        camera = Camera(np.array(K).copy(), *image_size)
        # System rescales camera.K in place to its stored-image resolution.
        metric_adapter = MetricTwoViewRelativePose if self.config.retrieval_pose == "metric_two_view" else MetricRelativePose
        pose_estimator = (DA3RelativePose(self.frontend.geometry, device) if self.config.retrieval_pose == "da3"
                          else metric_adapter(camera.K, device, self.config.mask_people, self.config.retrieval_matcher,
                                                  self.config.conditional_sources,self.config.source_log_std,
                                                  **(dict(rotation_geometry=self.frontend.geometry)
                                                     if self.config.two_view_rotation_check else {})))
        self.mapper = System(device=device, visualize=False, camera=camera, config=cfg, pose_estimator=pose_estimator)
        self.map_alignment = np.eye(4)
        self.initialized = False
        self.last_estimate = None

    def step(self, rgb, timestamp):
        estimate = self.frontend.step(rgb, timestamp)
        start = perf_counter()
        valid = estimate.diagnostics["valid"]
        map_now = valid and (not self.initialized or (self.frontend.index - 1) % self.config.mapping_interval == 0)
        # Every input increment is accumulated; image retrieval/filtering runs
        # less often. No skipped-frame odometry and no future images are used.
        self.mapper.step({
            "rgb": rgb if map_now else None,
            "depth": cv2.resize(estimate.depth, (rgb.shape[1], rgb.shape[0])) if map_now else None,
            "conf": None,
            "delta_pose": estimate.delta_pose,
            "motion_covariance": estimate.motion_covariance,
            "timestamp": timestamp,
        })
        if map_now:
            mapped = self.mapper.get_current_pose().matrix().detach().cpu().numpy()
            self.map_alignment = mapped @ inverse(estimate.pose)
            self.initialized = True
        estimate.diagnostics["frontend_pose"] = estimate.pose.tolist()
        estimate.pose = self.map_alignment @ estimate.pose
        estimate.diagnostics.update(
            mapping_seconds=perf_counter() - start,
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
