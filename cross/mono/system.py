"""Image-only entry point into the original CROSS topological mapper."""

from time import perf_counter

import cv2
import numpy as np

from .config import MonoConfig
from .frontend import MonoFrontend
from .geometry import inverse


class MonocularSystem:
    def __init__(self, K, image_size, config=None, system_config=None, device="cuda", frontend=None):
        from cross.core.config import FilterMode, PoseEstType, SystemConfig
        from cross.core.system import System
        from cross.core.types import Camera
        from .retrieval import DA3RelativePose

        self.config = config or MonoConfig()
        if frontend is not None:
            self.frontend = frontend
        elif self.config.frontend == "dpvo":
            from .dpvo_frontend import DPVOFrontend
            self.frontend = DPVOFrontend(K, self.config, device)
        elif self.config.frontend == "metric_pnp":
            from .pnp_frontend import MetricPnPFrontend
            self.frontend = MetricPnPFrontend(K, self.config, device)
        else:
            self.frontend = MonoFrontend(K, self.config, device)
        cfg = system_config or SystemConfig()
        if cfg.async_update:
            raise ValueError("MonocularSystem currently requires synchronous mapping updates")
        cfg.tracking.use_odometry = True  # internal visual increments, never input odometry
        cfg.tracking.use_VO = False
        cfg.depth_pred.use_depth_pred = False
        cfg.pose_est.type = PoseEstType.DA3
        if system_config is None:
            cfg.tracking.filter_mode = FilterMode.SKIP_ACTIVE
            cfg.tracking.odom_min_std_translation = 0.005
            cfg.tracking.odom_min_std_rotation = 0.01
            cfg.tracking.odom_std_per_meter = 0.1
            cfg.tracking.odom_std_per_radian = 0.1
            cfg.retrieval.top_k = 3
        self.mapper = System(device=device, visualize=False, camera=Camera(np.array(K).copy(), *image_size),
                             config=cfg, pose_estimator=DA3RelativePose(self.frontend.geometry, device))
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
