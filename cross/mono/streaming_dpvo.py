"""Relative visual motion with bounded, delayed learned metric-scale work.

The native tracker sees RGB only. Mature sparse depths and their source image
are frozen together for the teacher; the teacher never overwrites native BA.
The existing streaming mapper consumes timestamped metric-depth snapshots.
This scalar bridge does not implement conditional pose/source inference.
"""

from dataclasses import asdict, dataclass, replace
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .config import MonoConfig
from .dpvo_frontend import DPVOFrontend
from .frontend import MonoEstimate
from .geometry import inverse, scale_translation_covariance
from .models import DA3Geometry, DA3MetricDepth
from .scale import LogScaleFilter, observe_sparse_scale
from .scaled_motion import ScaledTranslation
from .streaming import FrameSnapshot, adjoint
from .worker import LatestWorker


@dataclass(frozen=True)
class ScaleRequest:
    snapshot: FrameSnapshot
    pixels: np.ndarray
    unit_depth: np.ndarray
    update_scale: bool
    requested_frame: int
    gauge: int = 0            # native tracker generation whose unit depths this request compares


_FAST = None


def texture_corners(rgb):
    """FAST corners (threshold 20) on a 320-pixel-wide grayscale image: a cheap degenerate-view test."""
    import cv2
    global _FAST
    if _FAST is None:
        _FAST = cv2.FastFeatureDetector_create(20)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(cv2.resize(rgb, (320, max(1, round(h * 320 / w))), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_RGB2GRAY)
    return len(_FAST.detect(gray))


class StreamingDPVOFrontend:
    def __init__(self, K, config, device='cuda'):
        self.config, self.device = config, device
        self.K = np.asarray(K).copy()
        self.geometry = DA3Geometry(config.pose_model, device, config.resolution)
        self.metric = DA3MetricDepth(config.metric_model, device, config.metric_resolution)
        native_config = MonoConfig(frontend='dpvo', dpvo_checkpoint=config.dpvo_checkpoint,
            seed=config.seed, mask_people=config.mask_people, mask_interval=config.mask_interval,
            pose_model=config.pose_model, resolution=config.resolution,
            scale=replace(config.scale, mode='relative'))
        self.native_frontend = DPVOFrontend(K, native_config, device, geometry_model=self.geometry)
        self.native_frontend.provide_mapping_depth = False
        self.scale_filter = LogScaleFilter(config.scale)
        self.translation = ScaledTranslation()
        self.metric_pose = np.eye(4)
        # a restarted native tracker starts a new unit gauge; attach it to the last emitted metric pose
        self.gauge_origin = np.eye(4)
        self.native_origin_inverse = np.eye(4)
        self.degenerate = False
        self.world_std_prefix = np.zeros(6)
        self.index, self.last_timestamp = 0, None
        self.last_request_frame = self.last_scale_request_frame = -config.scale.interval
        self.last_request_source = self.last_received_source = -1
        self.history, self.ready_depths, self.scale_events = {}, [], []
        self.provide_mapping_depth, self.finished = True, False
        self.depth_stream = torch.cuda.Stream(device=device) if device.startswith('cuda') else None
        if self.depth_stream is not None:
            self.depth_stream.wait_stream(torch.cuda.current_stream(device))
        self.depth_worker = LatestWorker(self._predict, 'cross-dpvo-metric-scale')

    def initialize_tracker(self, height, width):
        self.native_frontend.initialize_tracker(height, width)

    def _predict(self, request):
        snapshot = request.snapshot
        with torch.inference_mode():
            if self.depth_stream is None:
                depth = self.metric.predict_metric(snapshot.rgb, self.K, snapshot.rgb.shape[:2])
            else:
                with torch.cuda.stream(self.depth_stream):
                    depth = self.metric.predict_metric(snapshot.rgb, self.K, snapshot.rgb.shape[:2])
                self.depth_stream.synchronize()
        h, w = depth.shape
        lookup = np.rint(request.pixels).astype(int)
        lookup[:, 0] = lookup[:, 0].clip(0, w - 1)
        lookup[:, 1] = lookup[:, 1].clip(0, h - 1)
        observed = observe_sparse_scale(depth[lookup[:, 1], lookup[:, 0]], request.unit_depth,
                                        request.pixels, (h, w), self.config.scale)
        return request, depth, observed

    def _receive(self, final_drain=False):
        received = []
        for (request, depth, observation), timing in self.depth_worker.poll():
            source = request.snapshot.index
            if not self.last_received_source < source < self.index:
                raise ValueError('Metric results must have unique, increasing past source frames')
            self.last_received_source = source
            same_gauge = request.gauge == getattr(self.native_frontend, 'restarts', 0)
            applied = self.scale_filter.update(observation) if request.update_scale and same_gauge else False
            record = {k: None if isinstance(v, float) and not np.isfinite(v) else v
                      for k, v in asdict(observation).items()}
            event = dict(source_frame=source, source_timestamp=request.snapshot.timestamp,
                         requested_frame=request.requested_frame, received_frame=self.index,
                         final_drain=final_drain, scale_update_requested=request.update_scale,
                         scale_update_applied=applied, observation=record, **timing)
            received.append(event)
            self.scale_events.append(event)
            if self.provide_mapping_depth:
                self.ready_depths.append(((request.snapshot, depth), timing))
        return received

    def _submit_mature(self):
        native = self.native_frontend
        tracker = native.tracker
        if tracker is None or not tracker.is_initialized:
            return False
        interval = min(self.config.scale.interval, self.config.mapping_interval) if self.provide_mapping_depth else self.config.scale.interval
        if self.index - self.last_request_frame < interval:
            return False
        slot = max(0, tracker.n - 4)
        source = native.input_index(tracker.pg.tstamps_[slot])
        if source <= self.last_request_source or source not in self.history:
            return False
        snapshot = self.history[source]
        patches = tracker.pg.patches_[slot, :, :, 1, 1].float().cpu().numpy()
        pixels = patches[:, :2].copy() * tracker.RES
        unit_depth = np.where(patches[:, 2] > 1e-8, 1. / np.maximum(patches[:, 2], 1e-8), np.nan)
        due = self.index - self.last_scale_request_frame >= self.config.scale.interval
        if self.config.scale.mode == 'initial' and self.scale_filter.initialized:
            due = False
        self.depth_worker.submit(ScaleRequest(snapshot, pixels, unit_depth, due, self.index, native.restarts))
        self.last_request_frame, self.last_request_source = self.index, source
        if due:
            self.last_scale_request_frame = self.index
        return True

    def step(self, rgb, timestamp):
        if self.finished:
            raise RuntimeError('Frontend already finished')
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError('Frame timestamps must increase')
        start = perf_counter()
        self.scale_filter.predict()
        received = self._receive()
        native = self.native_frontend.step(rgb, timestamp)
        if native.diagnostics.get('tracker_restarted'):
            # new native gauge and unit scale: hold the pose, re-estimate the scale, continue from here
            self.gauge_origin = self.metric_pose.copy()
            self.native_origin_inverse = inverse(native.pose)
            self.scale_filter = LogScaleFilter(self.config.scale)
            self.translation = ScaledTranslation()
            self.last_scale_request_frame = self.index - self.config.scale.interval
        corners = texture_corners(rgb) if self.config.min_texture_corners > 0 else None
        degenerate = corners is not None and corners < self.config.min_texture_corners
        if self.degenerate and not degenerate:
            # visual motion during a degenerate stretch is unknown: continue from the held pose in DPVO's current gauge
            scale = self.scale_filter.scale if self.scale_filter.initialized else None
            self.gauge_origin = self.metric_pose.copy()
            self.native_origin_inverse = inverse(native.pose)
            self.translation = ScaledTranslation()
            if scale is not None:
                self.translation.update(np.zeros(3), scale)
        self.degenerate = degenerate
        initialized = not native.diagnostics['initializing']
        previous = self.metric_pose.copy()
        if initialized and not degenerate:
            local_native = self.native_origin_inverse @ native.pose
            local = np.eye(4)
            local[:3, :3] = Rotation.from_matrix(local_native[:3, :3]).as_matrix()
            local[:3, 3] = self.translation.update(local_native[:3, 3],
                self.scale_filter.scale if self.scale_filter.initialized else None)
            self.metric_pose = self.gauge_origin @ local
        valid = bool(native.diagnostics['valid'] and self.scale_filter.initialized and not degenerate)
        delta = inverse(previous) @ self.metric_pose
        step_t = self.config.translation_std_floor + self.config.translation_std_per_meter * float(np.linalg.norm(delta[:3, 3]))
        step_r = self.config.rotation_std_floor + self.config.rotation_std_per_radian * float(
            Rotation.from_matrix(delta[:3, :3]).magnitude())
        covariance = np.diag([step_t**2] * 3 + [step_r**2] * 3)
        covariance[:3, :3] += scale_translation_covariance(delta[:3, 3], self.scale_filter.uncertainty_variance)
        if not valid:
            covariance += np.eye(6)
        world_covariance = adjoint(previous) @ covariance @ adjoint(previous).T
        self.world_std_prefix += np.sqrt(np.maximum(np.diag(world_covariance), 0.))
        self.history[self.index] = FrameSnapshot(self.index, timestamp, rgb.copy(), {},
            self.metric_pose.copy(), self.world_std_prefix.copy(), valid)
        self.history = {i: s for i, s in self.history.items() if i in self.native_frontend.rgb_memory}
        requested = self._submit_mature()
        diagnostics = dict(frame=self.index, valid=valid, pose_source='streaming_dpvo',
            initializing=not initialized, metric_initialized=self.scale_filter.initialized,
            position_units='metres' if self.scale_filter.initialized else 'unavailable',
            unit_translation=(self.native_origin_inverse @ native.pose)[:3, 3].tolist() if initialized else None,
            tracker_restarts=native.diagnostics.get('tracker_restarts', 0),
            texture_corners=corners, degenerate_view=degenerate,
            scale_application='anchored_startup_v1', scale=self.scale_filter.scale,
            log_scale_std=float(np.sqrt(self.scale_filter.uncertainty_variance)),
            accepted_metric_observations=self.scale_filter.accepted,
            rejected_metric_observations=self.scale_filter.rejected,
            bootstrap_metric_calls=0, metric_seconds=0., metric_result_events=received,
            teacher_requested=requested, snapshot_history_size=len(self.history),
            native_tracker_seconds=native.diagnostics['total_seconds'],
            background_patches=native.diagnostics.get('background_patches'),
            total_seconds=perf_counter() - start)
        self.index += 1
        self.last_timestamp = timestamp
        return MonoEstimate(timestamp, self.metric_pose.copy(), delta, covariance, None, diagnostics)

    def take_depths(self):
        result, self.ready_depths = self.ready_depths, []
        return result

    def finish(self):
        if not self.finished:
            self.depth_worker.close()
            self._receive(final_drain=True)
            self.finished = True

    def shutdown(self):
        self.finish()
