"""Fast metric-prior PnP baseline for low-parallax monocular motion.

Periodic image-only metric depths define short-lived local anchors. XFeat
matches and RANSAC estimate every-frame motion using known calibration.
Depths are uncertain learned priors, not sensor observations.
"""

from time import perf_counter

import numpy as np

from .config import MonoConfig
from .frontend import MonoEstimate
from .geometry import inverse, scale_translation_covariance
from .models import DA3Geometry, DA3MetricDepth
from .refinement import XFeatRefiner
from .scale import LogScaleFilter


class MetricPnPFrontend:
    def __init__(self, K, config=None, device="cuda"):
        self.config = config or MonoConfig(frontend="metric_pnp")
        if self.config.scale.mode != "filtered":
            raise ValueError("Metric PnP uses metric anchor depths directly; scalar VO ablations do not apply")
        self.K = np.asarray(K).copy()
        self.geometry = DA3Geometry(self.config.pose_model, device, self.config.resolution)
        self.metric = DA3MetricDepth(self.config.metric_model, device, self.config.metric_resolution)
        self.refiner = XFeatRefiner(K, device)
        self.scale_filter = LogScaleFilter(self.config.scale)
        self.scale_filter.initialized = True
        self.scale_filter.variance = self.config.scale.observation_std_floor**2
        self.metric_pose = np.eye(4)
        self.anchor_pose = np.eye(4)
        self.anchor_features = self.anchor_depth = None
        self.anchor_index = 0
        self.index = 0
        self.last_timestamp = None
        self.provide_mapping_depth = True

    def step(self, rgb, timestamp):
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("Frame timestamps must increase")
        start = perf_counter()
        features = self.refiner.extract(rgb)
        depth = None
        metric_seconds = 0.0

        def current_depth():
            nonlocal depth, metric_seconds
            if depth is None:
                tick = perf_counter()
                depth = self.metric.predict_metric(rgb, self.K, rgb.shape[:2])
                metric_seconds += perf_counter() - tick
            return depth

        previous = self.metric_pose.copy()
        count, error = 0, 0.0
        valid = True
        if self.anchor_features is not None:
            result = self.refiner.estimate(self.anchor_features, features, self.anchor_depth)
            if result is None:
                # Fresh image geometry may recover overlap when the old
                # anchor's predicted depth was poor. The direction is explicit.
                reverse = self.refiner.estimate(features, self.anchor_features, current_depth())
                if reverse is not None:
                    result = (inverse(reverse[0]), reverse[1], reverse[2])
            if result is not None:
                relative, count, error = result
                self.metric_pose = self.anchor_pose @ relative
            else:
                valid = False
        renew = self.anchor_features is None or self.index - self.anchor_index >= self.config.scale.interval or not valid
        if renew:
            self.anchor_depth = current_depth()
            self.anchor_features = features
            self.anchor_pose = self.metric_pose.copy()
            self.anchor_index = self.index
        if self.provide_mapping_depth and self.index % self.config.mapping_interval == 0:
            current_depth()
        delta = inverse(previous) @ self.metric_pose
        covariance = np.diag([0.005**2] * 3 + [0.01**2] * 3)
        covariance[:3, :3] += scale_translation_covariance(delta[:3, 3], self.scale_filter.uncertainty_variance)
        if not valid:
            covariance += np.eye(6)
        diagnostics = {
            "frame": self.index, "valid": valid, "initializing": False, "pose_source": "metric_pnp",
            "pnp_inliers": count, "pnp_reprojection_median_px": error, "anchor_renewed": renew,
            "scale": 1.0, "log_scale_std": float(np.sqrt(self.scale_filter.uncertainty_variance)),
            "metric_initialized": True, "metric_seconds": metric_seconds,
            "frontend_seconds": perf_counter() - start - metric_seconds, "total_seconds": perf_counter() - start,
        }
        self.last_timestamp = timestamp
        self.index += 1
        return MonoEstimate(timestamp, self.metric_pose.copy(), delta, covariance,
                            depth if depth is not None else np.ones(rgb.shape[:2], np.float32), diagnostics)
