"""Causal monocular tracking with timestamped, bounded slow-model work.

The teacher supplies depth for the image it actually processed. Its delayed
output creates an anchor at that image's previously emitted local pose. No
old trajectory is rewritten and no future frame is consulted. The mapper
retains CROSS's global observation mixture and delayed commitment unchanged.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch

from .geometry import inverse
from .system import MonocularSystem
from .worker import LatestWorker


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


def snapshot_motion(previous, current):
    """All intermediate motion survives replacement of pending map images.

    Prefixes sum world-coordinate marginal standard deviations. Mapping to
    the previous camera uses |Ad| to conservatively allow unknown correlation;
    treating the diagonal as a full independent covariance would be unsafe.
    """
    if previous is None:
        return np.eye(4), np.zeros((6, 6))
    if current.index <= previous.index:
        raise ValueError("Mapping snapshots must increase")
    world_std = np.maximum(current.world_std_prefix - previous.world_std_prefix, 0.)
    local_std = np.abs(adjoint(inverse(previous.pose))) @ world_std
    return inverse(previous.pose) @ current.pose, np.diag(local_std**2)


class StreamingMonocularSystem(MonocularSystem):
    """A single owner updates the inherited CROSS mapper in capture order."""

    def __init__(self, K, image_size, config=None, system_config=None, device="cuda", frontend=None):
        if frontend is None:
            from .streaming_dpvo import StreamingDPVOFrontend
            frontend = StreamingDPVOFrontend(K, config, device)
        self.pool = None
        if config is not None and config.mapping_process:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

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

    def _map_snapshot(self, item):
        if self.pool is not None:
            from .mapping_process import operation
            return self.pool.submit(operation, "step", item).result()
        snapshot, depth = item
        def process():
            delta, covariance = snapshot_motion(self.previous_snapshot, snapshot)
            observation = dict(rgb=snapshot.rgb, depth=depth, conf=None, delta_pose=delta,
                               motion_covariance=covariance, timestamp=snapshot.timestamp)
            if self.previous_snapshot is None:
                # Choose the fresh map's coordinates from this frozen source
                # pose, not the later receiving pose. Loaded maps ignore this
                # gauge choice and retain CROSS's independent chart policy.
                observation['initial_chart_pose'] = snapshot.pose.copy()
            self.mapper.step(observation)
            mapped = self.mapper.get_current_pose().matrix().detach().cpu().numpy()
            self.previous_snapshot = snapshot
            event = dict(source_frame=snapshot.index, source_timestamp=snapshot.timestamp,
                         permanent_keyframes=self.mapper.db.get_size(),
                         graph_nodes=len(self.mapper.hypothesis_manager.nodes),
                         hypotheses=len(self.mapper.hypothesis_manager.hypotheses),
                         mapping_event=getattr(self.mapper, "last_step_diagnostics", {}).copy())
            return mapped @ inverse(snapshot.pose), event
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
            self.initialized = True
            events.append(dict(**event, **timing))
        self.map_events.extend(events)
        return events

    def step(self, rgb, timestamp):
        estimate = self.frontend.step(rgb, timestamp)
        self._submit_depths()
        events = self._receive_maps()
        estimate.diagnostics["frontend_pose"] = estimate.pose.tolist()
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
