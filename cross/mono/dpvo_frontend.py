"""DPVO motion with independent periodic image-only metric-scale observations.

DPVO's own loop closures are disabled: CROSS retains discrete map association
and commitment. Initial frames are explicitly marked uninitialized. The
finalized DPVO trajectory is available separately from causal online output.
"""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .config import MonoConfig
from .frontend import MonoEstimate
from .geometry import inverse, scale_translation_covariance
from .models import DA3Geometry, DA3MetricDepth
from .scale import LogScaleFilter, observe_sparse_scale


class DPVOFrontend:
    def __init__(self, K, config=None, device="cuda", geometry_model=None, external_masks=False):
        self.config = config or MonoConfig(frontend="dpvo")
        if not self.config.dpvo_checkpoint or not Path(self.config.dpvo_checkpoint).is_file():
            raise FileNotFoundError("Supply --dpvo-checkpoint pointing to released dpvo.pth")
        if device not in {"cuda", "cuda:0"}:
            raise ValueError("DPVO uses cuda:0; choose physical GPU with CUDA_VISIBLE_DEVICES")
        self.device = device
        self.K = np.asarray(K).copy()
        self.geometry = geometry_model or DA3Geometry(self.config.pose_model, device, self.config.resolution)
        self.metric = None if self.config.scale.mode == "relative" else DA3MetricDepth(
            self.config.metric_model, device, self.config.metric_resolution)
        self.scale_filter = LogScaleFilter(self.config.scale)
        self.tracker = None
        self.index = 0
        self.metric_pose = np.eye(4)
        self.unit_pose = np.eye(4)
        self.last_timestamp = None
        self.last_metric_index = -self.config.scale.interval
        self.rgb_memory = {}
        self.scale_history = []
        self.initialized_before = False
        self.provide_mapping_depth = True
        self.bootstrap_metric_calls = 0
        self.background_patchifier = None
        self.external_masks = external_masks
        self.vo_cpu_rng = torch.Generator().manual_seed(self.config.seed).get_state()
        self.vo_cuda_rng = torch.Generator(device="cuda").manual_seed(self.config.seed).get_state()

    def _pose_at(self, index):
        """Resolve an input frame through DPVO's keyframe-removal chain."""
        from dpvo.lietorch import SE3
        tracker = self.tracker
        entries = {int(tracker.pg.tstamps_[i]): tracker.pg.poses_[i] for i in range(tracker.n)}
        visited = set()
        transform = SE3.Identity(1, device="cuda")[0]
        while index not in entries:
            if index in visited or index not in tracker.pg.delta:
                raise RuntimeError("DPVO pose history is incomplete")
            visited.add(index)
            index, relative = tracker.pg.delta[index]
            transform = transform * relative
        return (transform * SE3(entries[index])).inv().matrix().float().cpu().numpy()

    @torch.inference_mode()
    def step(self, rgb, timestamp, exclusion_boxes=None):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected uint8 RGB")
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("Frame timestamps must increase")
        start = perf_counter()
        h, w = rgb.shape[:2]
        if h % 16 or w % 16:
            raise ValueError("DPVO input dimensions must be multiples of 16")
        if self.tracker is None:
            from dpvo.config import cfg
            from dpvo.dpvo import DPVO
            config = cfg.clone()
            config.LOOP_CLOSURE = False
            config.CLASSIC_LOOP_CLOSURE = False
            config.PATCHES_PER_FRAME = 96
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(self.config.seed)
                self.tracker = DPVO(config, self.config.dpvo_checkpoint, ht=h, wd=w, viz=False)
            if self.config.mask_people:
                from .background_patches import BackgroundPatchifier
                self.background_patchifier = BackgroundPatchifier(
                    self.tracker.network.patchify, self.device, self.config.mask_interval, self.external_masks)
                self.tracker.network.patchify = self.background_patchifier
        tracker = self.tracker
        if self.background_patchifier is not None:
            self.background_patchifier.observe(rgb, exclusion_boxes)
        self.rgb_memory[self.index] = rgb.copy()
        image = torch.as_tensor(rgb[..., ::-1].copy(), device="cuda").permute(2, 0, 1)  # DPVO expects BGR
        intrinsics = torch.tensor([self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]], device="cuda")
        # The geometry/metric models and BoQ can consume random numbers even
        # in eval mode (e.g. pixel subsampling). Keep VO patch choices identical
        # across scale ablations and frontend-only vs integrated runs.
        previous_keyframes = tracker.n
        with torch.random.fork_rng(devices=[0]):
            torch.set_rng_state(self.vo_cpu_rng)
            torch.cuda.set_rng_state(self.vo_cuda_rng)
            tracker(timestamp, image, intrinsics)
            self.vo_cpu_rng = torch.get_rng_state()
            self.vo_cuda_rng = torch.cuda.get_rng_state()
        # A streaming metric teacher owns another CUDA stream. Waiting for the
        # whole device here would make its delayed work block every fast pose.
        torch.cuda.current_stream().synchronize()
        pose_seconds = perf_counter() - start
        self.scale_filter.predict()
        initialized = bool(tracker.is_initialized)
        diagnostics = {"frame": self.index, "valid": initialized, "frontend_seconds": pose_seconds,
                       "pose_source": "dpvo", "initializing": not initialized}
        if self.background_patchifier is not None:
            diagnostics.update(background_patches=self.background_patchifier.valid_patches,
                               person_boxes=len(self.background_patchifier.current_boxes))
            if self.background_patchifier.valid_patches < 20:
                diagnostics["valid"] = False
        # Random inverse-depth initialization is ill-conditioned when startup
        # has little static parallax. A learned RGB-only depth initializes the
        # newly admitted patch depths; subsequent native BA remains free to
        # refine them. This is a starting guess, not a fixed depth measurement.
        if (self.config.dpvo_metric_bootstrap and not initialized and self.metric is not None
                and tracker.n > previous_keyframes):
            bootstrap_start = perf_counter()
            predicted = self.metric.predict_metric(rgb, self.K, (h, w))
            slot = tracker.n - 1
            uv = tracker.pg.patches_[slot, :, :2, 1, 1].float().cpu().numpy() * tracker.RES
            lookup = np.round(uv).astype(int)
            lookup[:, 0] = np.clip(lookup[:, 0], 0, w - 1)
            lookup[:, 1] = np.clip(lookup[:, 1], 0, h - 1)
            depths = predicted[lookup[:, 1], lookup[:, 0]]
            good = np.isfinite(depths) & (depths > 0.1) & (depths < 100)
            indices = torch.as_tensor(np.flatnonzero(good), device="cuda")
            inverse_depth = torch.as_tensor(1 / depths[good], device="cuda")
            tracker.pg.patches_[slot, indices, 2] = inverse_depth[:, None, None]
            self.bootstrap_metric_calls += 1
            diagnostics["bootstrap_metric_seconds"] = perf_counter() - bootstrap_start
        # Only mature patch depths constrain metric scale. They belong to a
        # past image still in the active window; no future image enters inference.
        due = self.index - self.last_metric_index >= self.config.scale.interval
        if self.config.scale.mode == "initial" and self.scale_filter.initialized:
            due = False
        metric_seconds = 0.0
        if initialized and due and self.metric is not None:
            slot = max(0, tracker.n - 4)
            input_id = int(tracker.pg.tstamps_[slot])
            if input_id in self.rgb_memory:
                metric_start = perf_counter()
                metric_depth = self.metric.predict_metric(self.rgb_memory[input_id], self.K, (h, w))
                patches = tracker.pg.patches_[slot, :, :, 1, 1].float().cpu().numpy()
                uv = patches[:, :2] * tracker.RES
                lookup = np.round(uv).astype(int)
                lookup[:, 0] = np.clip(lookup[:, 0], 0, w - 1)
                lookup[:, 1] = np.clip(lookup[:, 1], 0, h - 1)
                source = np.where(patches[:, 2] > 1e-8, 1 / np.maximum(patches[:, 2], 1e-8), np.nan)
                observed = observe_sparse_scale(metric_depth[lookup[:, 1], lookup[:, 0]], source, uv, (h, w),
                                                self.config.scale)
                self.scale_filter.update(observed)
                diagnostics["scale_observation"] = {
                    key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                    for key, value in asdict(observed).items()}
                diagnostics["scale_observation_input_frame"] = input_id
                self.last_metric_index = self.index
                metric_seconds = perf_counter() - metric_start
        delta = np.eye(4)
        if initialized:
            current = self._pose_at(self.index)
            # The VO gauge is anchored at its first camera (internal loop
            # closures/normalization are disabled). Include local-BA corrections
            # to the current pose; differencing two freshly optimized poses
            # would discard corrections to already-reported motion.
            next_metric = self.metric_pose.copy()
            next_metric[:3, :3] = current[:3, :3]
            next_metric[:3, 3] += self.scale_filter.scale * (current[:3, 3] - self.unit_pose[:3, 3])
            delta = inverse(self.metric_pose) @ next_metric
            self.unit_pose = current
            self.metric_pose = next_metric
            self.initialized_before = True
        covariance = np.diag([0.005**2] * 3 + [0.01**2] * 3)
        covariance[:3, :3] += scale_translation_covariance(delta[:3, 3], self.scale_filter.uncertainty_variance)
        if not initialized:
            covariance += np.eye(6)
        elif not diagnostics["valid"]:
            covariance += np.eye(6)
        # Dense depth is needed only on mapping updates. It is predicted from
        # current RGB; it is never sensor depth. It is not used as VO input.
        depth = np.ones((h, w), dtype=np.float32)
        if self.provide_mapping_depth and initialized and (self.index % self.config.mapping_interval == 0 or not getattr(self, "had_depth", False)):
            depth_start = perf_counter()
            if self.metric is not None:
                depth = self.metric.predict_metric(rgb, self.K, (h, w))
            else:
                prediction = self.geometry.predict([self.geometry.prepare(rgb)])
                depth = cv2.resize(prediction.depth[0], (w, h))
            self.had_depth = True
            diagnostics["mapping_depth_seconds"] = perf_counter() - depth_start
        # Keep only frames still eligible for scale observations (plus a small
        # safety tail for active-window reindexing). Images are bounded memory.
        active_ids = set(int(x) for x in tracker.pg.tstamps_[max(0, tracker.n - 12):tracker.n])
        self.rgb_memory = {k: v for k, v in self.rgb_memory.items() if k in active_ids}
        self.scale_history.append(self.scale_filter.scale)
        diagnostics.update(scale=self.scale_filter.scale, log_scale_std=float(np.sqrt(self.scale_filter.uncertainty_variance)),
                           metric_initialized=self.scale_filter.initialized, metric_seconds=metric_seconds,
                           pending_scale_observations=len(self.scale_filter.pending),
                           scale_reinitializations=self.scale_filter.reinitializations,
                           bootstrap_metric_calls=self.bootstrap_metric_calls,
                           accepted_metric_observations=self.scale_filter.accepted,
                           rejected_metric_observations=self.scale_filter.rejected,
                           total_seconds=perf_counter() - start)
        self.last_timestamp = timestamp
        self.index += 1
        return MonoEstimate(timestamp, self.metric_pose.copy(), delta, covariance, depth, diagnostics)

    def finalized_trajectory(self):
        """Return offline-refined unit poses; never overwrite causal outputs."""
        if not self.tracker or not self.tracker.is_initialized:
            return None
        poses, timestamps = self.tracker.terminate()
        matrices = np.broadcast_to(np.eye(4), (len(poses), 4, 4)).copy()
        matrices[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()
        matrices[:, :3, 3] = poses[:, :3]
        return timestamps, matrices
