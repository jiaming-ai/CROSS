"""Causal monocular tracking with timestamped, bounded slow-model work.

The teacher supplies depth for the image it actually processed. Its delayed
output creates an anchor at that image's previously emitted local pose. No
old trajectory is rewritten and no future frame is consulted. The mapper
retains CROSS's global observation mixture and delayed commitment unchanged.
"""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .frontend import MonoEstimate
from .geometry import inverse, scale_translation_covariance
from .pnp_frontend import MetricPnPFrontend
from .system import MonocularSystem
from .worker import LatestWorker
from .metric_sources import MetricSource, TranslationResponse, identify_prediction, relative_scale_response


def adjoint(pose):
    rotation, (x, y, z) = pose[:3, :3], pose[:3, 3]
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    out = np.zeros((6, 6))
    out[:3, :3] = out[3:, 3:] = rotation
    out[:3, 3:] = skew @ rotation
    return out


@dataclass
class FrameSnapshot:
    index: int
    timestamp: float
    rgb: np.ndarray
    features: dict
    pose: np.ndarray
    world_std_prefix: np.ndarray
    valid: bool
    refresh_anchor: bool = False
    metric_source: MetricSource | None = None
    scale_response: TranslationResponse | None = None
    world_geometry_std_prefix: np.ndarray | None = None
    rotation_prior: np.ndarray | None = None


def snapshot_motion(previous, current, conditional=False):
    """All intermediate motion survives replacement of pending map images.

    Prefixes sum world-coordinate marginal standard deviations. Mapping to
    the previous camera uses |Ad| to conservatively allow unknown correlation;
    treating the diagonal as a full independent covariance would be unsafe.
    """
    if previous is None:
        return np.eye(4), np.zeros((6, 6))
    if current.index <= previous.index:
        raise ValueError("Mapping snapshots must increase")
    if conditional:
        if previous.world_geometry_std_prefix is None or current.world_geometry_std_prefix is None:
            raise ValueError('Conditional mapping needs a geometry-only uncertainty prefix')
        world_std = np.maximum(current.world_geometry_std_prefix-previous.world_geometry_std_prefix,0.)
        local_std = np.abs(adjoint(inverse(current.pose)))@world_std
        return inverse(previous.pose)@current.pose,np.diag(local_std**2)
    world_std = np.maximum(current.world_std_prefix - previous.world_std_prefix, 0.)
    local_std = np.abs(adjoint(inverse(previous.pose))) @ world_std
    return inverse(previous.pose) @ current.pose, np.diag(local_std**2)


class StreamingPnPFrontend(MetricPnPFrontend):
    def __init__(self, K, config=None, device="cuda"):
        super().__init__(K, config, device)
        self.device = device
        self.rotation_tracker = None
        self.rotation_alignment = None
        if self.config.rotation_tracker == 'dpvo':
            from .config import MonoConfig
            from .dpvo_frontend import DPVOFrontend
            rotation_config = MonoConfig(frontend='dpvo', dpvo_checkpoint=self.config.dpvo_checkpoint,
                seed=self.config.seed, mask_people=self.config.mask_people, mask_interval=self.config.mask_interval,
                pose_model=self.config.pose_model, resolution=self.config.resolution,
                scale=replace(self.config.scale, mode='relative'))
            self.rotation_tracker = DPVOFrontend(K, rotation_config, device,
                geometry_model=self.geometry, external_masks=True)
            self.rotation_tracker.provide_mapping_depth = False
        self.depth_stream = torch.cuda.Stream(device=device, priority=0) if device.startswith("cuda") else None
        # Models were constructed on the calling stream before the worker.
        if self.depth_stream is not None:
            self.depth_stream.wait_stream(torch.cuda.current_stream(device))
        self.world_std_prefix = np.zeros(6)
        self.world_geometry_std_prefix = np.zeros(6)
        self.ready_depths = []
        self.pending_depths = []
        self.last_submitted = -1
        self.finished = False
        self.trace_metric_sources = self.config.trace_metric_sources
        if self.trace_metric_sources:
            from uuid import uuid4
            from .models import MODEL_REVISIONS
            self.metric_revision = MODEL_REVISIONS.get(self.config.metric_model)
            if self.metric_revision is None:
                raise ValueError("Metric source tracing requires a pinned metric model revision")
            self.metric_session_id = uuid4().hex
            self.anchor_metric_source = None
            self.anchor_scale_response = TranslationResponse()
            self.pose_scale_prefix = TranslationResponse()
            self.pose_scale_tail = None
        self.depth_worker = LatestWorker(self._predict_snapshot, "cross-metric-depth")

    def _capture_scale_response(self):
        if self.pose_scale_tail is None:
            return self.pose_scale_prefix
        source, displacement = self.pose_scale_tail
        return self.pose_scale_prefix.with_displacement(source, displacement)

    def _predict_snapshot(self, snapshot):
        with torch.inference_mode():
            if self.depth_stream is None:
                depth = self.metric.predict_metric(snapshot.rgb, self.K, snapshot.rgb.shape[:2])
            else:
                with torch.cuda.stream(self.depth_stream):
                    depth = self.metric.predict_metric(snapshot.rgb, self.K, snapshot.rgb.shape[:2])
                self.depth_stream.synchronize()
        return snapshot, depth

    def _receive(self, drain=False):
        received = []
        renewed = False
        for (snapshot, depth), timing in self.depth_worker.poll():
            self.pending_depths.append(((snapshot, depth), dict(timing, first_ready_frame=self.index)))
        retained = []
        for (snapshot, depth), timing in self.pending_depths:
            if snapshot.index <= self.anchor_index:
                continue
            if not drain and self.index-snapshot.index < self.config.teacher_lag_frames:
                retained.append(((snapshot, depth), timing))
                continue
            recovered = False
            if not snapshot.valid and self.config.delayed_recovery:
                # The baseline's reverse PnP becomes available when this
                # image's depth arrives. Correct the internal delayed anchor,
                # never a previously emitted pose. The next increment carries
                # the correction and retains the failed-frame uncertainty.
                if snapshot.rotation_prior is None:
                    reverse = self.refiner.estimate(snapshot.features, self.anchor_features, depth)
                else:
                    # Use the rotation available at this source image, never
                    # a later native pose or the receiving frame's rotation.
                    relative_rotation = (Rotation.from_matrix(self.anchor_pose[:3, :3]).inv()
                                         * Rotation.from_matrix(snapshot.rotation_prior)).as_matrix()
                    reverse = self.refiner.estimate(snapshot.features, self.anchor_features, depth,
                                                    rotation=relative_rotation)
                if reverse is not None:
                    relative = inverse(reverse[0])
                    snapshot = replace(snapshot, pose=self.anchor_pose @ relative, valid=True)
                    if getattr(self, "trace_metric_sources", False):
                        # Reverse PnP uses the newly delivered image's depth,
                        # not the old anchor's depth. Emitted trace stays held.
                        response = self.anchor_scale_response.with_displacement(
                            snapshot.metric_source.source_id, self.anchor_pose[:3, :3] @ relative[:3, 3])
                        snapshot = replace(snapshot, scale_response=response)
                    recovered = True
            timing = dict(timing, delayed_reverse_recovery=recovered,
                          held_ready_frames=self.index-timing["first_ready_frame"],
                          late_delivery_frames=max(0, self.index-snapshot.index-self.config.teacher_lag_frames)
                          if self.config.teacher_lag_frames else 0,
                          final_drain=drain)
            received.append(((snapshot, depth), timing))
            if (snapshot.index - self.anchor_index >= self.config.scale.interval or not self.last_valid
                    or (snapshot.valid and snapshot.refresh_anchor)):
                self.anchor_features, self.anchor_depth = snapshot.features, depth
                self.anchor_pose, self.anchor_index = snapshot.pose, snapshot.index
                if getattr(self, "trace_metric_sources", False):
                    self.anchor_metric_source = snapshot.metric_source
                    self.anchor_scale_response = snapshot.scale_response
                renewed = True
        self.pending_depths = retained
        self.ready_depths.extend(received)
        return renewed, received

    def take_depths(self):
        result, self.ready_depths = self.ready_depths, []
        return result

    def step(self, rgb, timestamp):
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("Frame timestamps must increase")
        start = perf_counter()
        renewed, received = self._receive()
        features = self.refiner.extract(rgb)
        self.rotation_prior = None
        rotation_estimate = None
        rotation_initialized = False
        if getattr(self, 'rotation_tracker', None) is not None:
            rotation_estimate = self.rotation_tracker.step(rgb, timestamp,
                exclusion_boxes=features.get('exclusion_boxes'))
            native_rotation = rotation_estimate.pose[:3, :3]
            rotation_initialized = bool(rotation_estimate.diagnostics['valid'] and np.isfinite(native_rotation).all())
            if rotation_initialized and self.rotation_alignment is not None:
                self.rotation_prior = (Rotation.from_matrix(self.rotation_alignment)
                                       * Rotation.from_matrix(native_rotation)).as_matrix()
        previous = self.metric_pose.copy()
        valid, count, error = True, 0, 0.
        bootstrap_seconds = 0.
        if self.anchor_features is not None:
            if self.rotation_prior is None:
                result = self.refiner.estimate(self.anchor_features, features, self.anchor_depth)
            else:
                relative_rotation = (Rotation.from_matrix(self.rotation_prior).inv()
                                     * Rotation.from_matrix(self.anchor_pose[:3, :3])).as_matrix()
                result = self.refiner.estimate(self.anchor_features, features, self.anchor_depth,
                                               rotation=relative_rotation)
            if result is not None:
                relative, count, error = result
                self.metric_pose = self.anchor_pose @ relative
                if getattr(self, "trace_metric_sources", False):
                    self.pose_scale_prefix = self.anchor_scale_response
                    self.pose_scale_tail = (self.anchor_metric_source.source_id,
                                           self.anchor_pose[:3, :3] @ relative[:3, 3])
            else:
                valid = False  # keep the held output and its failure in evaluation
                if self.rotation_prior is not None:
                    self.metric_pose[:3, :3] = self.rotation_prior
        if rotation_initialized and self.rotation_alignment is None and valid:
            self.rotation_alignment = (Rotation.from_matrix(self.metric_pose[:3, :3])
                                       * Rotation.from_matrix(native_rotation).inv()).as_matrix()
        delta = inverse(previous) @ self.metric_pose
        covariance = np.diag([self.config.translation_std_floor**2]*3 + [self.config.rotation_std_floor**2]*3)
        geometry_covariance = covariance.copy()
        covariance[:3, :3] += scale_translation_covariance(delta[:3, 3], self.scale_filter.uncertainty_variance)
        if not valid:
            covariance += np.eye(6)
            geometry_covariance += np.eye(6)
        transported = adjoint(previous) @ covariance @ adjoint(previous).T
        self.world_std_prefix += np.sqrt(np.maximum(np.diag(transported), 0.))
        geometry_transported = adjoint(self.metric_pose)@geometry_covariance@adjoint(self.metric_pose).T
        if hasattr(self,'world_geometry_std_prefix'):
            self.world_geometry_std_prefix += np.sqrt(geometry_transported.diagonal().clip(0))
        interval = min(self.config.scale.interval, self.config.mapping_interval) if self.provide_mapping_depth else self.config.scale.interval
        bootstrap = self.anchor_features is None
        # The experimental policy asks while the current pose still has a
        # verified geometric estimate. A depth attached to a held pose after
        # loss cannot recover the motion that was missed. Keep the existing
        # five-frame request bound and a single replaceable pending item.
        weak_support = (self.config.adaptive_anchor and not bootstrap and valid and count < 80)
        elapsed = self.index-self.last_submitted
        if self.config.stable_teacher_cadence:
            # Each interval starts at a fixed input index. An emergency may
            # defer its next regular request by the cooldown, but cannot shift
            # the grid permanently. A request satisfies the current interval;
            # missed intervals never create catch-up work. Short configured
            # intervals retain their original rate, and emergencies keep the
            # original five-frame bound. The latest-only worker stays bounded.
            periodic_due = self.index//interval > self.last_submitted//interval
            regular_request = periodic_due and elapsed >= min(5, interval)
        else:
            regular_request = elapsed >= interval
        emergency_request = (not valid or weak_support) and elapsed >= 5
        request = regular_request or emergency_request
        if bootstrap or request:
            snapshot = FrameSnapshot(self.index, timestamp, rgb.copy(), features, self.metric_pose.copy(),
                                     self.world_std_prefix.copy(), valid, refresh_anchor=weak_support,
                                     world_geometry_std_prefix=getattr(self,'world_geometry_std_prefix',None),
                                     rotation_prior=None if self.rotation_prior is None else self.rotation_prior.copy())
            if snapshot.world_geometry_std_prefix is not None:
                snapshot = replace(snapshot,world_geometry_std_prefix=snapshot.world_geometry_std_prefix.copy())
            if getattr(self, "trace_metric_sources", False):
                source = identify_prediction(snapshot.rgb, self.K, model_id=self.config.metric_model,
                                             revision=self.metric_revision, resolution=self.config.metric_resolution,
                                             session_id=self.metric_session_id)
                snapshot = replace(snapshot, metric_source=source, scale_response=self._capture_scale_response())
            self.last_submitted = self.index
            if bootstrap:
                tick = perf_counter()
                _, depth = self._predict_snapshot(snapshot)
                bootstrap_seconds = perf_counter()-tick
                self.anchor_features, self.anchor_depth = features, depth
                self.anchor_pose, self.anchor_index = self.metric_pose.copy(), self.index
                if getattr(self, "trace_metric_sources", False):
                    self.anchor_metric_source = snapshot.metric_source
                    self.anchor_scale_response = snapshot.scale_response
                self.ready_depths.append(((snapshot, depth), dict(queue_seconds=0., service_seconds=bootstrap_seconds,
                                                               turnaround_seconds=bootstrap_seconds)))
                renewed = True
            else:
                self.depth_worker.submit(snapshot)
        diagnostics = dict(frame=self.index, valid=valid, initializing=bootstrap, pose_source="streaming_pnp",
                           pnp_inliers=count, pnp_reprojection_median_px=error, anchor_renewed=renewed,
                           anchor_frame=self.anchor_index, anchor_age_frames=self.index-self.anchor_index,
                           correspondences=self.refiner.last_correspondences,
                           masked_keypoints=features.get("masked_keypoints", 0), person_boxes=features.get("person_boxes", 0),
                           scale=1., log_scale_std=float(np.sqrt(self.scale_filter.uncertainty_variance)),
                           metric_initialized=True, metric_seconds=bootstrap_seconds, bootstrap_seconds=bootstrap_seconds,
                           weak_anchor_support=weak_support,
                           metric_request=bool(bootstrap or request),
                           regular_metric_request=bool(not bootstrap and regular_request),
                           emergency_metric_request=bool(not bootstrap and emergency_request),
                           proactive_metric_request=bool(weak_support and request),
                           frontend_seconds=perf_counter()-start-bootstrap_seconds, total_seconds=perf_counter()-start,
                           teacher_updates=[dict(source_frame=s.index, age_frames=self.index-s.index, **timing)
                                            for (s, _), timing in received],
                           teacher_worker=self.depth_worker.statistics())
        if rotation_estimate is not None:
            diagnostics.update(rotation_tracker='dpvo', rotation_initialized=rotation_initialized,
                               rotation_prior_used=self.rotation_prior is not None,
                               rotation_tracker_seconds=rotation_estimate.diagnostics['total_seconds'],
                               native_rotation_matrix=rotation_estimate.pose[:3, :3].tolist(),
                               shared_person_mask=self.config.mask_people,
                               actual_torch_threads=torch.get_num_threads())
        if getattr(self, "trace_metric_sources", False):
            diagnostics.update(metric_anchor_source=self.anchor_metric_source.source_id,
                               metric_session_id=self.metric_session_id,
                               metric_source_trace_version=1)
        self.last_timestamp, self.last_valid = timestamp, valid
        self.index += 1
        if not self.provide_mapping_depth:
            self.take_depths()
        # A delayed anchor depth is not a depth observation of this RGB frame.
        return MonoEstimate(timestamp, self.metric_pose.copy(), delta, covariance, None, diagnostics)

    def finish(self):
        if not self.finished:
            self.depth_worker.close()
            self._receive(drain=True)
            self.finished = True

    def shutdown(self):
        self.finish()


class StreamingMonocularSystem(MonocularSystem):
    """A single owner updates the inherited CROSS mapper in capture order."""

    def __init__(self, K, image_size, config=None, system_config=None, device="cuda", frontend=None):
        frontend = frontend or StreamingPnPFrontend(K, config, device)
        self.pool = None
        if config is not None and config.mapping_process:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            from .mapping_process import initialize
            self.config, self.frontend = config, frontend
            self.map_alignment, self.initialized, self.last_estimate = np.eye(4), False, None
            self.mapper = None  # single owner lives in the child process
            self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
            self.pool.submit(initialize, K, image_size, config, system_config, device).result()
        else:
            super().__init__(K, image_size, config, system_config, device, frontend)
        self.map_stream = torch.cuda.Stream(device=device, priority=0) if device.startswith("cuda") else None
        if self.map_stream is not None:
            self.map_stream.wait_stream(torch.cuda.current_stream(device))
        self.previous_snapshot = None
        self.map_worker = LatestWorker(self._map_snapshot, "cross-global-observation")
        self.map_events = []
        self.finished = False
        self.source_biases = {}
        self.bias_packet_revision = 0
        self._bias_prefix_cache = None

    def _map_snapshot(self, item):
        if self.pool is not None:
            from .mapping_process import operation
            return self.pool.submit(operation, "step", item).result()
        snapshot, depth = item
        def process():
            conditional = getattr(getattr(self,'config',None),'conditional_sources',False)
            delta, covariance = snapshot_motion(self.previous_snapshot, snapshot,conditional=conditional)
            observation = dict(rgb=snapshot.rgb, depth=depth, conf=None, delta_pose=delta,
                               motion_covariance=covariance, timestamp=snapshot.timestamp)
            if getattr(snapshot, "metric_source", None) is not None:
                previous = self.previous_snapshot
                jacobians = relative_scale_response(previous.pose, previous.scale_response, snapshot.pose,
                                                     snapshot.scale_response) if previous is not None else {}
                observation["metric_source"] = snapshot.metric_source.record()
                observation["metric_input"] = dict(version=1, source_frame=snapshot.index,
                    previous_frame=previous.index if previous is not None else None,
                    source_id=snapshot.metric_source.source_id, session_id=snapshot.metric_source.session_id,
                    world_translation_response=snapshot.scale_response.record(),
                    motion_right_tangent_response=jacobians)
                if conditional:
                    from cross.core.conditional import SourceFactor
                    keys = tuple(sorted(jacobians))
                    J = np.column_stack([jacobians[k] for k in keys]) if keys else np.empty((6,0))
                    observation['motion_source_factor'] = SourceFactor(tuple('image:'+k for k in keys),J,
                        np.full(len(keys),self.config.source_log_std**2),log_depth_scale=True)
            self.mapper.step(observation)
            mapped = self.mapper.get_current_pose().matrix().detach().cpu().numpy()
            self.previous_snapshot = snapshot
            event = dict(source_frame=snapshot.index, source_timestamp=snapshot.timestamp,
                         permanent_keyframes=self.mapper.db.get_size(),
                         graph_nodes=len(self.mapper.hypothesis_manager.nodes),
                         hypotheses=len(self.mapper.hypothesis_manager.hypotheses),
                         mapping_event=getattr(self.mapper, "last_step_diagnostics", {}).copy())
            frontend_pose = snapshot.pose.copy()
            if conditional:
                state = self.mapper.hypothesis_manager.source_states[0]
                biases = {key.removeprefix('image:'):float(mean) for key,mean in zip(state.keys,state.mean)}
                frontend_pose[:3,3] = snapshot.scale_response.translation_at(frontend_pose[:3,3],biases)
                event['_source_biases'] = biases
                event['conditional_sources'] = dict(source_count=len(biases),max_abs_log_bias=max(map(abs,biases.values()),default=0.))
            return mapped @ inverse(frontend_pose), event
        with torch.inference_mode():
            if self.map_stream is None:
                return process()
            with torch.cuda.stream(self.map_stream):
                result = process()
            self.map_stream.synchronize()
            return result

    def _submit_depths(self):
        for (snapshot, depth), _ in self.frontend.take_depths():
            if snapshot.valid:
                # Mapping needs RGB/depth/motion, not frontend CUDA features.
                self.map_worker.submit((replace(snapshot, features={}), depth))

    def _receive_maps(self):
        events = []
        for (alignment, event), timing in self.map_worker.poll():
            self.map_alignment = alignment
            if '_source_biases' in event:
                self.source_biases = event.pop('_source_biases')
                self.bias_packet_revision += 1
            self.initialized = True
            events.append(dict(**event, **timing))
        self.map_events.extend(events)
        return events

    def step(self, rgb, timestamp):
        estimate = self.frontend.step(rgb, timestamp)
        self._submit_depths()
        events = self._receive_maps()
        estimate.diagnostics["frontend_pose"] = estimate.pose.tolist()
        if self.config.conditional_sources:
            prefix = self.frontend.pose_scale_prefix
            if (self._bias_prefix_cache is None or self._bias_prefix_cache[0] is not prefix
                    or self._bias_prefix_cache[1] != self.bias_packet_revision):
                correction = prefix.translation_at(np.zeros(3),self.source_biases)
                self._bias_prefix_cache = (prefix,self.bias_packet_revision,correction)
            correction = self._bias_prefix_cache[2].copy()
            tail = self.frontend.pose_scale_tail
            if tail is not None:
                source,displacement = tail
                correction -= (1.-np.exp(-self.source_biases.get(source,0.)))*displacement
            estimate.pose[:3,3] += correction
        estimate.pose = self.map_alignment @ estimate.pose
        estimate.diagnostics.update(mapping_updates=events, mapping_update=bool(events),
                                    mapping_worker=self.map_worker.statistics())
        self.last_estimate = estimate
        return estimate

    def finish(self):
        if not self.finished:
            try:
                self.frontend.finish()
                self._submit_depths()
            finally:
                self.map_worker.close()
                self._receive_maps()
            self.finished = True

    def save_map(self, path):
        self.finish()
        if self.pool is None:
            super().save_map(path)
        else:
            from .mapping_process import operation
            self.pool.submit(operation, "save", str(path)).result()

    def load_map(self, path):
        if self.frontend.index:
            raise RuntimeError("Load a map before processing images of a new session")
        if self.pool is None:
            super().load_map(path)
        else:
            from .mapping_process import operation
            self.pool.submit(operation, "load", str(path)).result()

    def warmup_mapping(self, rgb):
        if self.pool is not None:
            from .mapping_process import operation
            self.pool.submit(operation, "warmup", rgb).result()
        else:
            self.mapper.db.vpr_model.get_embedding(self.mapper.rgb_transform(rgb))
            if hasattr(self.mapper.pose_est, "refiner"):
                refiner = self.mapper.pose_est.refiner
                old_index, old_boxes = refiner.frame_index, refiner.boxes
                refiner.extract(rgb)
                refiner.frame_index, refiner.boxes = old_index, old_boxes

    def shutdown(self):
        try:
            self.finish()
        finally:
            try:
                self.frontend.shutdown()
            finally:
                if self.pool is None:
                    super().shutdown()
                else:
                    from .mapping_process import operation
                    try:
                        self.pool.submit(operation, "shutdown", None).result()
                    finally:
                        self.pool.shutdown()
