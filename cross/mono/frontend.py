"""Causal compact-memory visual odometry with periodic learned metric scale."""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from .config import MonoConfig, ScaleConfig
from .geometry import inverse, mean_pose, relative_pose, scale_translation_covariance
from .scale import LogScaleFilter, observe_scale


@dataclass
class MemoryFrame:
    index: int
    image: object
    pose: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    features: object = None


@dataclass
class MonoEstimate:
    timestamp: float
    pose: np.ndarray
    delta_pose: np.ndarray
    motion_covariance: np.ndarray
    depth: np.ndarray
    diagnostics: dict


class MonoFrontend:
    """Only RGB, timestamp and intrinsics enter this API.

    Poses and stored overlap depths are maintained in one arbitrary VO gauge.
    A separate log-scale state converts *increments* to metres. Metric updates
    never silently rescale old, committed keyframes or induce a pose jump.
    Learned metric priors can still have persistent domain bias.
    """

    def __init__(self, K, config=None, device="cuda", geometry_model=None, metric_model=None):
        self.config = config or MonoConfig()
        self.K = np.asarray(K, dtype=np.float64).copy()
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all() or min(self.K[0, 0], self.K[1, 1]) <= 0:
            raise ValueError("A finite calibrated pinhole camera matrix is required")
        if geometry_model is None:
            from .models import DA3Geometry
            geometry_model = DA3Geometry(self.config.pose_model, device, self.config.resolution)
        if metric_model is None and self.config.scale.mode != "relative":
            from .models import DA3MetricDepth
            metric_model = DA3MetricDepth(self.config.metric_model, device, self.config.metric_resolution)
        self.geometry = geometry_model
        self.metric = metric_model
        self.refiner = None
        if self.config.pose_refinement == "xfeat":
            from .refinement import XFeatRefiner
            self.refiner = XFeatRefiner(self.K, device)
        self.reset()

    def reset(self):
        self.index = 0
        self.anchor = None
        self.recent = []
        self.unit_pose = np.eye(4)
        self.metric_pose = np.eye(4)
        self.scale_filter = LogScaleFilter(self.config.scale)
        self.last_timestamp = None
        self.last_metric_index = -self.config.scale.interval

    def _references(self):
        candidates = ([self.anchor] if self.anchor is not None else []) + self.recent
        return list({frame.index: frame for frame in candidates}.values())

    def step(self, rgb, timestamp):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected HxWx3 uint8 RGB input")
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("Frame timestamps must be finite and strictly increasing")
        start = perf_counter()
        self.scale_filter.predict()
        image = self.geometry.prepare(rgb)
        refs = self._references()
        prediction = self.geometry.predict([frame.image for frame in refs] + [image])
        frontend_seconds = perf_counter() - start
        current_depth = prediction.depth[-1]
        current_confidence = prediction.confidence[-1]
        diagnostics = {"frame": self.index, "valid": True, "frontend_seconds": frontend_seconds}
        current_features = self.refiner.extract(rgb) if self.refiner else None
        proposals, weights, overlap_scales = [], [], []
        overlap_config = ScaleConfig(observation_std_floor=0.02, max_log_mad=0.55)
        for i, ref in enumerate(refs):
            observation = observe_scale(ref.depth, prediction.depth[i], prediction.confidence[i], overlap_config)
            if not observation.accepted:
                continue
            scale = np.exp(observation.log_scale)
            relative = relative_pose(prediction.extrinsics[i], prediction.extrinsics[-1], scale)
            proposals.append(ref.pose @ relative)
            weights.append(1.0 / max(observation.log_mad, 0.05))
            overlap_scales.append(scale)
        if refs and not proposals:
            # No invented motion on a failed observation. Coverage records failure.
            diagnostics["valid"] = False
            diagnostics["failure"] = "no_consistent_overlap"
            next_unit_pose = self.unit_pose.copy()
            unit_depth = current_depth.copy()
            dispersion = np.ones(3)
        elif proposals:
            next_unit_pose = mean_pose(proposals, weights)
            unit_depth = current_depth * np.average(overlap_scales, weights=weights)
            dispersion = np.std(np.asarray(proposals)[:, :3, 3], axis=0)
        else:
            next_unit_pose = np.eye(4)
            unit_depth = current_depth.copy()
            dispersion = np.zeros(3)
        if self.refiner and refs:
            refined, refined_weights = [], []
            for ref in refs:
                result = self.refiner.estimate(ref.features, current_features, ref.depth)
                if result is not None:
                    relative, inliers, error = result
                    refined.append(ref.pose @ relative)
                    refined_weights.append(inliers / max(error, 0.5))
                    if self.config.refinement_anchor_only:
                        break
            diagnostics["refined_references"] = len(refined)
            if refined:
                next_unit_pose = mean_pose(refined, refined_weights)
                dispersion = np.std(np.asarray(refined)[:, :3, 3], axis=0)
                diagnostics["pose_source"] = "xfeat_pnp"
            else:
                diagnostics["pose_source"] = "da3_fallback"
        delta_unit = inverse(self.unit_pose) @ next_unit_pose
        rotation = Rotation.from_matrix(delta_unit[:3, :3]).magnitude()
        if rotation > self.config.max_relative_rotation:
            diagnostics.update(valid=False, failure="implausible_rotation")
        metric_seconds = 0.0
        due = self.index - self.last_metric_index >= self.config.scale.interval
        if self.config.scale.mode == "initial" and self.scale_filter.initialized:
            due = False
        if due and self.metric is not None and diagnostics["valid"]:
            metric_start = perf_counter()
            metric_depth = self.metric.predict_metric(rgb, self.K, unit_depth.shape)
            observation = observe_scale(metric_depth, unit_depth, current_confidence, self.config.scale)
            self.scale_filter.update(observation)
            if self.config.metric_shape and observation.accepted:
                # Preserve the arbitrary VO gauge while improving anchor shape.
                # The scalar metric observation remains separately uncertain.
                valid_depth = np.isfinite(metric_depth) & (metric_depth > 0)
                unit_depth = np.where(valid_depth, metric_depth / np.exp(observation.log_scale), unit_depth)
            diagnostics["scale_observation"] = {
                key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                for key, value in asdict(observation).items()
            }
            self.last_metric_index = self.index
            metric_seconds = perf_counter() - metric_start
        delta_metric = delta_unit.copy()
        delta_metric[:3, 3] *= self.scale_filter.scale
        covariance = np.diag([self.config.translation_std_floor**2] * 3 + [self.config.rotation_std_floor**2] * 3)
        covariance[:3, :3] += scale_translation_covariance(delta_metric[:3, 3], self.scale_filter.variance)
        local_dispersion = inverse(self.unit_pose)[:3, :3] @ np.diag(dispersion**2) @ self.unit_pose[:3, :3]
        covariance[:3, :3] += local_dispersion * self.scale_filter.scale**2
        if diagnostics["valid"]:
            self.metric_pose = self.metric_pose @ delta_metric
            self.unit_pose = next_unit_pose
            frame = MemoryFrame(self.index, image, next_unit_pose, unit_depth, current_confidence, current_features)
            if self.anchor is None or self.index - self.anchor.index >= self.config.anchor_interval:
                self.anchor = frame
            self.recent = (self.recent + [frame])[-self.config.recent_frames:]
        else:
            delta_metric = np.eye(4)
            covariance += np.eye(6)
        self.last_timestamp = timestamp
        self.index += 1
        diagnostics.update(
            scale=self.scale_filter.scale,
            log_scale_std=float(np.sqrt(self.scale_filter.variance)),
            metric_initialized=self.scale_filter.initialized,
            metric_seconds=metric_seconds,
            accepted_metric_observations=self.scale_filter.accepted,
            rejected_metric_observations=self.scale_filter.rejected,
            overlap_references=len(proposals),
            unit_delta_translation=delta_unit[:3, 3].tolist(),
            total_seconds=perf_counter() - start,
        )
        return MonoEstimate(timestamp, self.metric_pose.copy(), delta_metric, covariance,
                            unit_depth * self.scale_filter.scale, diagnostics)
