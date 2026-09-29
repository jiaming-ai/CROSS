"""Fast metric-prior PnP baseline for low-parallax monocular motion.

Periodic image-only metric depths define short-lived local anchors. XFeat
matches and RANSAC estimate every-frame motion using known calibration.
Depths are uncertain learned priors, not sensor observations.
"""

from time import perf_counter

import cv2
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
        self.refiner = XFeatRefiner(K, device, mask_people=self.config.mask_people, mask_interval=self.config.mask_interval,
                                    rotation_selection=self.config.rotation_selection, subpixel=self.config.subpixel)
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
        self.rotation_prior = None

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
            relative_rotation = None if self.rotation_prior is None else self.rotation_prior.T @ self.anchor_pose[:3, :3]
            result = self.refiner.estimate(self.anchor_features, features, self.anchor_depth, rotation=relative_rotation)
            if result is None:
                # Fresh image geometry may recover overlap when the old
                # anchor's predicted depth was poor. The direction is explicit.
                reverse = self.refiner.estimate(features, self.anchor_features, current_depth(),
                                               rotation=None if relative_rotation is None else relative_rotation.T)
                if reverse is not None:
                    result = (inverse(reverse[0]), reverse[1], reverse[2])
            if result is not None:
                relative, count, error = result
                self.metric_pose = self.anchor_pose @ relative
            else:
                valid = False
                if self.rotation_prior is not None:
                    self.metric_pose[:3, :3] = self.rotation_prior
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
            "correspondences": self.refiner.last_correspondences,
            "masked_keypoints": features.get("masked_keypoints", 0), "person_boxes": features.get("person_boxes", 0),
            "rotation_only_selected": self.refiner.last_rotation_only,
            "scale": 1.0, "log_scale_std": float(np.sqrt(self.scale_filter.uncertainty_variance)),
            "metric_initialized": True, "metric_seconds": metric_seconds,
            "frontend_seconds": perf_counter() - start - metric_seconds, "total_seconds": perf_counter() - start,
        }
        self.last_timestamp = timestamp
        self.index += 1
        return MonoEstimate(timestamp, self.metric_pose.copy(), delta, covariance,
                            depth if depth is not None else np.ones(rgb.shape[:2], np.float32), diagnostics)


class RotationMetricFrontend(MetricPnPFrontend):
    """DPVO rotation plus robust translation from periodic metric anchors.

    This ablates scalar-only correction: learned depth shape is needed when
    monocular patch inverse depth is not constrained by enough static parallax.
    """
    def __init__(self, K, config=None, device="cuda"):
        from dataclasses import replace
        from .dpvo_frontend import DPVOFrontend
        super().__init__(K, config, device)
        config = replace(self.config, frontend="dpvo", dpvo_metric_bootstrap=False,
                         scale=replace(self.config.scale, mode="relative"))
        self.rotation_tracker = DPVOFrontend(K, config, device, geometry_model=self.geometry)
        self.rotation_tracker.provide_mapping_depth = False
        self.rotation_alignment = None

    def step(self, rgb, timestamp):
        start = perf_counter()
        rotation_estimate = self.rotation_tracker.step(rgb, timestamp)
        rotation = rotation_estimate.pose[:3, :3]
        initialized = bool(rotation_estimate.diagnostics["valid"] and np.isfinite(rotation).all())
        self.rotation_prior = self.rotation_alignment @ rotation if initialized and self.rotation_alignment is not None else None
        estimate = super().step(rgb, timestamp)
        if initialized and self.rotation_alignment is None and estimate.diagnostics["valid"]:
            self.rotation_alignment = estimate.pose[:3, :3] @ rotation.T
        estimate.diagnostics.update(pose_source="rotation_metric", rotation_initialized=initialized,
                                    total_seconds=perf_counter() - start)
        return estimate


class LearnedRotationPnPFrontend(MetricPnPFrontend):
    """Experimental compact learned rotation with metric PnP translation.

    DA3 sees only the current RGB and the metric anchor's RGB. Its translation
    and depth predictions are not used. Translation still requires the existing
    calibrated correspondence consensus against uncertain metric depth. This
    is a frontend baseline, not an independent likelihood or a mapping change.
    """

    def __init__(self, K, config=None, device="cuda"):
        super().__init__(K, config, device)
        self.rotation_anchor_image = None

    def step(self, rgb, timestamp):
        from scipy.spatial.transform import Rotation
        from .geometry import relative_pose

        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("Frame timestamps must increase")
        start = perf_counter()
        image = self.geometry.prepare(rgb)
        anchor_index = self.anchor_index
        self.rotation_prior = None
        reason = "bootstrap"
        if self.rotation_anchor_image is not None:
            prediction = self.geometry.predict([self.rotation_anchor_image, image])
            relative = relative_pose(prediction.extrinsics[0], prediction.extrinsics[1])
            if np.isfinite(relative[:3, :3]).all():
                rotation = self.anchor_pose[:3, :3] @ relative[:3, :3]
                step_angle = Rotation.from_matrix(self.metric_pose[:3, :3].T @ rotation).magnitude()
                if step_angle <= self.config.max_relative_rotation:
                    self.rotation_prior = rotation
                    reason = "accepted"
                else:
                    reason = "implausible_rotation"
            else:
                reason = "nonfinite_rotation"
        geometry_seconds = perf_counter()-start
        estimate = super().step(rgb, timestamp)
        # Renewal is decided by the metric frontend. Retain exactly its image,
        # including a held-translation anchor whose rotation could be recovered.
        if estimate.diagnostics["anchor_renewed"]:
            self.rotation_anchor_image = image
        elapsed = perf_counter()-start
        estimate.diagnostics.update(pose_source="learned_rotation_pnp",
            learned_rotation_anchor_frame=anchor_index,
            learned_rotation_used=self.rotation_prior is not None,
            learned_rotation_reason=reason, geometry_seconds=geometry_seconds,
            frontend_seconds=elapsed-estimate.diagnostics["metric_seconds"], total_seconds=elapsed)
        return estimate


class MetricKLTFrontend(MetricPnPFrontend):
    """Track persistent image points between frames against metric anchors.

    Learned descriptors seed/reacquire tracks; image-space tracking preserves
    subpixel identity between adjacent frames. It does not filter output poses.
    """
    def __init__(self, K, config=None, device="cuda"):
        from .tracks import MetricAnchorTracks
        super().__init__(K, config, device)
        self.tracks = MetricAnchorTracks(K)

    def step(self, rgb, timestamp):
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("Frame timestamps must increase")
        start = perf_counter()
        features = self.refiner.extract(rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        depth, metric_seconds = None, 0.

        def current_depth():
            nonlocal depth, metric_seconds
            if depth is None:
                tick = perf_counter()
                depth = self.metric.predict_metric(rgb, self.K, rgb.shape[:2])
                metric_seconds += perf_counter() - tick
            return depth

        previous = self.metric_pose.copy()
        valid, count, error, reacquired = True, 0, 0., False
        if self.anchor_features is not None:
            result = self.tracks.track(gray, features.get("exclusion_boxes", ()))
            if result is None:
                result = self.refiner.estimate(self.anchor_features, features, self.anchor_depth)
                if result is None:
                    reverse = self.refiner.estimate(features, self.anchor_features, current_depth())
                    if reverse is not None:
                        result = (inverse(reverse[0]), reverse[1], reverse[2])
                reacquired = result is not None
            if result is not None:
                relative, count, error = result
                self.metric_pose = self.anchor_pose @ relative
            else:
                valid = False
        renew = (self.anchor_features is None or self.index-self.anchor_index >= self.config.scale.interval
                 or not valid or reacquired)
        if renew:
            self.anchor_depth = current_depth()
            self.anchor_features, self.anchor_pose, self.anchor_index = features, self.metric_pose.copy(), self.index
            self.tracks.reset(gray, features["keypoints"].detach().cpu().numpy(), self.anchor_depth)
        if self.provide_mapping_depth and self.index % self.config.mapping_interval == 0:
            current_depth()
        delta = inverse(previous) @ self.metric_pose
        covariance = np.diag([self.config.translation_std_floor**2]*3 + [self.config.rotation_std_floor**2]*3)
        covariance[:3, :3] += scale_translation_covariance(delta[:3, 3], self.scale_filter.uncertainty_variance)
        if not valid:
            covariance += np.eye(6)
        diagnostics = dict(frame=self.index, valid=valid, initializing=False, pose_source="metric_klt",
                           pnp_inliers=count, pnp_reprojection_median_px=error, anchor_renewed=renew,
                           correspondences=self.tracks.last_candidates, descriptor_reacquisition=reacquired,
                           masked_keypoints=features.get("masked_keypoints", 0), person_boxes=features.get("person_boxes", 0),
                           scale=1., log_scale_std=float(np.sqrt(self.scale_filter.uncertainty_variance)),
                           metric_initialized=True, metric_seconds=metric_seconds,
                           frontend_seconds=perf_counter()-start-metric_seconds, total_seconds=perf_counter()-start)
        self.last_timestamp = timestamp
        self.index += 1
        return MonoEstimate(timestamp, self.metric_pose.copy(), delta, covariance,
                            depth if depth is not None else np.ones(rgb.shape[:2], np.float32), diagnostics)
