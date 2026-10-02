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

from cross.imu import InertialScaleFilter, preintegrate

from .config import MonoConfig
from .frontend import MonoEstimate
from .geometry import inverse, scale_translation_covariance
from .models import DA3Geometry, DA3MetricDepth
from .scale import LogScaleFilter, observe_sparse_scale
from .scaled_motion import ScaledTranslation


def _so3_log(R):
    return Rotation.from_matrix(R).as_rotvec()


class DPVOFrontend:
    def __init__(self, K, config=None, device="cuda", geometry_model=None, external_masks=False, metric_model=None,
                 load_metric=True):
        self.config = config or MonoConfig(frontend="dpvo")
        if not self.config.dpvo_checkpoint or not Path(self.config.dpvo_checkpoint).is_file():
            raise FileNotFoundError("Supply --dpvo-checkpoint pointing to released dpvo.pth")
        if device not in {"cuda", "cuda:0"}:
            raise ValueError("DPVO uses cuda:0; choose physical GPU with CUDA_VISIBLE_DEVICES")
        self.device = device
        self.K = np.asarray(K).copy()
        self._geometry = geometry_model      # DA3 geometry, loaded on first use (not needed with input depth)
        # learned metric depth: scale observations (unless the IMU alone sets the scale) and mapping depth;
        # load_metric=False for a back end that needs no depth with IMU-only scale
        self.metric = None if self.config.scale.mode == "relative" or self.config.depth_input or not load_metric else (
            metric_model or DA3MetricDepth(self.config.metric_model, device, self.config.metric_resolution))
        self.depth_memory = {}               # input depth of the frames still eligible for scale observations
        # IMU scale (visual-inertial): the filter is created with the calibration of the first frame
        self.inertial = bool(self.config.imu.enabled)
        self.imu_calib = None
        self.accel_history = []
        self.prev_tracked = False
        self.imu_queue = []
        self._gyro_rotation = None
        # continuous start (set by the pipeline for back ends that can observe without metric motion): frames before
        # the metric trajectory is valid are reported as unknown motion instead of being withheld
        self.continuous_start = False
        self.output_valid = False
        self.gyro_rejections = 0
        self.gyro_bias = np.zeros(3)
        self.bias_samples = []
        self.gyro_world = None
        self.imu_buffer = np.zeros((0, 7))
        self.time_offset = 0.0              # camera clock = IMU clock + time_offset
        self.time_offset_done = False
        self.rate_log = []
        self.scale_filter = self._new_scale_filter()
        self.tracker = None
        self.index = 0
        self.metric_pose = np.eye(4)
        self.unit_pose = np.eye(4)
        self.scaled_translation = ScaledTranslation()
        self.last_timestamp = None
        self.last_metric_index = -self.config.scale.interval
        self.rgb_memory = {}
        self.scale_history = []
        self.initialized_before = False
        self.provide_mapping_depth = True
        self.bootstrap_metric_calls = 0
        self.background_patchifier = None
        self.external_masks = external_masks
        # A diverged tracker (non-finite pose) is restarted: its frame counter starts again at 0, and its new
        # arbitrary gauge is attached to the last emitted pose.
        self.frame_offset = 0
        self.gauge_origin = np.eye(4)
        self.restarts = 0
        # optional callable (previous_rgb, current_rgb) -> (T_previous_current or None, covisibility)
        self.pair_motion = None
        self.previous_rgb = self.previous_thumb = None
        self.bridges = 0
        self.vo_cpu_rng = torch.Generator().manual_seed(self.config.seed).get_state()
        self.vo_cuda_rng = torch.Generator(device="cuda").manual_seed(self.config.seed).get_state()

    def _new_scale_filter(self):
        if not self.inertial:
            return LogScaleFilter(self.config.scale)
        if self.imu_calib is None:
            return None
        c = self.imu_calib
        return InertialScaleFilter(self.config.imu, c.T_cam_imu, c.gyro_noise_density, c.accel_noise_density)

    def _depth_units(self, tracker):
        """Median patch depth of a recent keyframe (visual-odometry units): the scale of the visual noise."""
        inverse = tracker.pg.patches_[max(0, tracker.n - 2), :, 2, 1, 1].float().cpu().numpy()
        inverse = inverse[np.isfinite(inverse) & (inverse > 1e-6)]
        return float(np.median(1.0 / inverse)) if len(inverse) else None

    @property
    def geometry(self):
        if self._geometry is None:
            self._geometry = DA3Geometry(self.config.pose_model, self.device, self.config.resolution)
        return self._geometry

    @geometry.setter
    def geometry(self, model):
        self._geometry = model

    def track(self, frame):
        """Pipeline interface: one input frame (dict with rgb, timestamp and, with depth_input, depth; with the IMU,
        imu / imu_t0 / imu_t1 / imu_calib of cross.dataloader.imu)."""
        imu = None
        if self.inertial:
            if frame.get("imu_calib") is None:
                raise ValueError("IMU scale needs frames with an IMU stream (imu.txt / imu.json next to the images)")
            if self.imu_calib is None:
                self.imu_calib = frame["imu_calib"]
                self.scale_filter = self._new_scale_filter()
            imu = frame
        return self.step(frame["rgb"], frame["timestamp"],
                         depth=frame.get("depth") if self.config.depth_input else None, imu=imu)

    def input_index(self, tracker_stamp):
        """Input frame index of a tracker frame counter (the counter restarts with the tracker)."""
        return int(tracker_stamp) + self.frame_offset

    def _restart_tracker(self, next_frame=None):
        self.tracker = None
        self.background_patchifier = None
        self.frame_offset = self.index + 1 if next_frame is None else next_frame
        self.gauge_origin = self.metric_pose.copy()
        self.scaled_translation = ScaledTranslation()
        self.scale_filter = self._new_scale_filter()
        self.prev_tracked = False
        self.imu_queue = []
        self.gyro_world = None
        self.rgb_memory = {}
        self.depth_memory = {}
        self.restarts += 1

    @staticmethod
    def _thumb(rgb):
        small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
        return (small - small.mean()) / (small.std() + 1e-6)

    def _needs_mapping_depth(self, initialized, valid):
        """Dense depth for this frame: every mapping interval, and on the first valid frame (the mapper's first update)."""
        if not (self.provide_mapping_depth and initialized):
            return False
        return self.index % self.config.mapping_interval == 0 or (bool(valid) and not getattr(self, "had_depth", False))

    def _check_discontinuity(self, rgb):
        """Detect a jump with little shared content; restart DPVO from this frame and return the bridge."""
        if self.config.discontinuity_ncc <= 0:
            return None
        thumb, previous, previous_thumb = self._thumb(rgb), self.previous_rgb, self.previous_thumb
        self.previous_rgb, self.previous_thumb = rgb, thumb
        if previous_thumb is None or float((thumb * previous_thumb).mean()) >= self.config.discontinuity_ncc:
            return None
        if self.tracker is None or not self.tracker.is_initialized:
            return None
        T, covis = self.pair_motion(previous, rgb) if self.pair_motion is not None else (None, 0.)
        if T is None:
            pass
        elif covis >= 0.3:
            self.bridges += 1
        self._restart_tracker(next_frame=self.index)
        return T, covis

    def _pose_at(self, index):
        """Resolve an input frame through DPVO's keyframe-removal chain."""
        from dpvo.lietorch import SE3
        index -= self.frame_offset
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

    def initialize_tracker(self, height, width):
        """Load the native tracker without consuming an image or pose."""
        if height % 16 or width % 16:
            raise ValueError('DPVO input dimensions must be multiples of 16')
        if self.tracker is not None:
            return
        from dpvo.config import cfg
        from dpvo.dpvo import DPVO
        config = cfg.clone()
        config.LOOP_CLOSURE = False
        config.CLASSIC_LOOP_CLOSURE = False
        config.PATCHES_PER_FRAME = 96
        config.BUFFER_SIZE = 16384          # frames kept by the tracker; long sequences (KITTI 02) overflow the default 4096
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(self.config.seed)
            self.tracker = DPVO(config, self.config.dpvo_checkpoint, ht=height, wd=width, viz=False)
        if self.config.mask_people:
            from .background_patches import BackgroundPatchifier
            self.background_patchifier = BackgroundPatchifier(
                self.tracker.network.patchify, self.device, self.config.mask_interval, self.external_masks)
            self.tracker.network.patchify = self.background_patchifier

    @torch.inference_mode()
    def step(self, rgb, timestamp, exclusion_boxes=None, depth=None, imu=None):
        # The pinned upstream extensions launch kernels on CUDA's default
        # stream. Keep their PyTorch allocations and consumers on that same
        # stream; CROSS may call us from its high-priority tracking stream.
        caller = torch.cuda.current_stream(self.device)
        native = torch.cuda.default_stream(self.device)
        if caller == native:
            return self._step(rgb, timestamp, exclusion_boxes, depth, imu)
        native.wait_stream(caller)
        with torch.cuda.stream(native):
            result = self._step(rgb, timestamp, exclusion_boxes, depth, imu)
        caller.wait_stream(native)
        return result

    def _step(self, rgb, timestamp, exclusion_boxes=None, depth=None, imu=None):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected uint8 RGB")
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("Frame timestamps must increase")
        start = perf_counter()
        h, w = rgb.shape[:2]
        # DPVO needs sides that are multiples of 16: it tracks the top-left crop (same intrinsics, same pixel
        # coordinates), everything else (scale lookups, mapping depth) uses the full image
        th, tw = h - h % 16, w - w % 16
        if min(th, tw) < 16:
            raise ValueError("Image too small for DPVO")
        if self.config.depth_input:
            if depth is None or depth.shape[:2] != (h, w):
                raise ValueError("depth_input needs a depth map of the image size with every frame")
        track = rgb[:th, :tw]
        bridge = self._check_discontinuity(rgb)
        if self.config.depth_input:          # after a possible restart (which clears the memory)
            self.depth_memory[self.index] = np.asarray(depth, dtype=np.float32)
        self.initialize_tracker(th, tw)
        tracker = self.tracker
        if self.background_patchifier is not None:
            self.background_patchifier.observe(np.ascontiguousarray(track), exclusion_boxes)
        self.rgb_memory[self.index] = rgb.copy()
        image = torch.as_tensor(track[..., ::-1].copy(), device="cuda").permute(2, 0, 1)  # DPVO expects BGR
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
        restarted = False
        if initialized and not np.isfinite(self._pose_at(self.index)).all():
            # Diverged native BA (non-finite pose): hold the last pose, report tracking loss and start a new tracker
            # (new gauge and unit scale) from the next frame instead of aborting the run.
            self._restart_tracker()
            tracker = None
            initialized, restarted = False, True
        diagnostics = {"frame": self.index, "valid": initialized, "frontend_seconds": pose_seconds,
                       "pose_source": "dpvo", "initializing": not initialized, "tracker_restarted": restarted,
                       "tracker_restarts": self.restarts}
        if self.background_patchifier is not None:
            diagnostics.update(background_patches=self.background_patchifier.valid_patches,
                               person_boxes=len(self.background_patchifier.current_boxes))
            if self.background_patchifier.valid_patches < 20:
                diagnostics["valid"] = False
        # Random inverse-depth initialization is ill-conditioned when startup
        # has little static parallax. A learned RGB-only depth initializes the
        # newly admitted patch depths; subsequent native BA remains free to
        # refine them. This is a starting guess, not a fixed depth measurement.
        if (self.config.dpvo_metric_bootstrap and not initialized and self.metric is not None and tracker is not None
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
        self._gyro_rotation = None
        if self.inertial and imu is not None:
            diagnostics["imu"] = self._inertial_step(imu, tracker, initialized)
            if len(imu["imu"]) and imu["imu_t1"] > imu["imu_t0"]:
                raw = np.asarray(imu["imu"], dtype=np.float64).copy()
                raw[:, 1:4] -= self.gyro_bias
                R_cb = self.imu_calib.T_cam_imu[:3, :3]
                dR = preintegrate(raw, float(imu["imu_t0"]), float(imu["imu_t1"]), 0.0, 0.0).dR
                self._gyro_rotation = R_cb @ dR @ R_cb.T
        # Only mature patch depths constrain metric scale. They belong to a
        # past image still in the active window; no future image enters inference.
        due = self.index - self.last_metric_index >= self.config.scale.interval
        if self.inertial and not self.config.imu.depth_prior:
            due = False
        if self.config.scale.mode == "initial" and self.scale_filter.initialized:
            due = False
        metric_seconds = 0.0
        if initialized and due and (self.metric is not None or self.config.depth_input):
            slot = max(0, tracker.n - 4)
            input_id = self.input_index(tracker.pg.tstamps_[slot])
            if input_id in self.rgb_memory:
                metric_start = perf_counter()
                metric_depth = (self.depth_memory[input_id] if self.config.depth_input else
                                self.metric.predict_metric(self.rgb_memory[input_id], self.K, (h, w)))
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
        unit_translation = None
        valid_now = bool(diagnostics["valid"] and initialized and self.scale_filter.initialized)
        hold = self.continuous_start and not valid_now     # unknown motion until the metric trajectory is valid
        if initialized and not hold:
            current = self._pose_at(self.index)
            # The VO gauge is anchored at its first camera (internal loop
            # closures/normalization are disabled). Include local-BA corrections
            # to the current pose; differencing two freshly optimized poses
            # would discard corrections to already-reported motion.
            local = np.eye(4)
            local[:3, :3] = current[:3, :3]
            unit_translation = current[:3, 3].tolist()
            # Native initialization only establishes an arbitrary visual gauge.
            # Until a metric prior is accepted, its displacement is not metres.
            # The first accepted scale anchors all motion accumulated so far.
            local[:3, 3] = self.scaled_translation.update(
                current[:3, 3], self.scale_filter.scale if self.scale_filter.initialized else None)
            if self.continuous_start and not self.output_valid:
                # continue from the last reported pose: the motion before this frame was reported as unknown
                self.gauge_origin = self.metric_pose @ inverse(local)
            next_metric = self.gauge_origin @ local
            delta = inverse(self.metric_pose) @ next_metric
            self.unit_pose = current
            self.metric_pose = next_metric
            self.initialized_before = True
        diagnostics['valid'] = bool(diagnostics['valid'] and self.scale_filter.initialized)
        diagnostics.update(unit_translation=unit_translation, scale_application='anchored_startup_v1',
                           position_units=('tracker' if self.config.scale.mode == 'relative' else
                                           'metres' if self.scale_filter.initialized else 'unavailable'))
        covariance = np.diag([0.005**2] * 3 + [0.01**2] * 3)
        covariance[:3, :3] += scale_translation_covariance(delta[:3, 3], self.scale_filter.uncertainty_variance)
        if hold:
            # continuous start: frames before the metric trajectory is valid carry unknown motion (the gyro's rotation
            # when there is an IMU) with a wide covariance, so the back end can observe from the first frame
            rotation_std = self.config.imu.unknown_motion_std[1]
            if self._gyro_rotation is not None:
                delta[:3, :3] = self._gyro_rotation
                rotation_std = self.config.imu.unknown_motion_std[2]
            self.metric_pose = self.metric_pose @ delta
            covariance = np.diag([self.config.imu.unknown_motion_std[0] ** 2] * 3 + [rotation_std ** 2] * 3)
        elif not initialized:
            covariance += np.eye(6)
        elif not diagnostics["valid"]:
            covariance += np.eye(6)
        self.output_valid = bool(diagnostics["valid"])
        diagnostics["unknown_motion"] = bool(hold)
        if bridge is not None:
            # the jump itself: learned relative pose (inflated uncertainty), or unknown motion
            T, covis = bridge
            if T is not None:
                delta = T
                self.metric_pose = self.metric_pose @ T
                covariance = np.diag([0.1**2 + (0.2 * np.linalg.norm(T[:3, 3]))**2] * 3 + [0.05**2] * 3)
            else:
                delta = np.eye(4)
                covariance = np.eye(6)
            self.gauge_origin = self.metric_pose.copy()
            diagnostics.update(valid=False, discontinuity=True, bridge_covisibility=covis, bridged=T is not None)
        # Dense depth is needed only on mapping updates. It is predicted from
        # current RGB; it is never sensor depth. It is not used as VO input.
        mapping_depth = None      # no depth predicted for this frame (the pipeline predicts one if it maps it)
        if self.config.depth_input:
            mapping_depth = np.asarray(depth, dtype=np.float32)
        elif self._needs_mapping_depth(initialized, diagnostics["valid"]):
            depth_start = perf_counter()
            if self.metric is not None:
                mapping_depth = self.metric.predict_metric(rgb, self.K, (h, w))
            else:
                prediction = self.geometry.predict([self.geometry.prepare(rgb)])
                mapping_depth = cv2.resize(prediction.depth[0], (w, h))
            # the mapper initializes on the first VALID frame (metric scale accepted), which may fall between mapping
            # intervals: that frame needs depth even when earlier (still invalid) frames already had one
            self.had_depth = getattr(self, "had_depth", False) or bool(diagnostics["valid"])
            diagnostics["mapping_depth_seconds"] = perf_counter() - depth_start
        # Keep only frames still eligible for scale observations (plus a small
        # safety tail for active-window reindexing). Images are bounded memory.
        active_ids = set() if tracker is None else set(
            self.input_index(x) for x in tracker.pg.tstamps_[max(0, tracker.n - 12):tracker.n])
        self.rgb_memory = {k: v for k, v in self.rgb_memory.items() if k in active_ids}
        self.depth_memory = {k: v for k, v in self.depth_memory.items() if k in active_ids}
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
        return MonoEstimate(timestamp, self.metric_pose.copy(), delta, covariance, mapping_depth, diagnostics)

    def _inertial_step(self, imu, tracker, initialized):
        """Buffer the IMU samples and queue the last frame interval; intervals whose ends are tracked by DPVO enter the
        inertial scale filter imu.lag_frames frames late, with the refined DPVO poses of both ends.

        The filter works in a gyro-propagated world frame: the gyroscope (bias-corrected) carries the orientation from
        interval to interval and DPVO contributes its displacement in its own camera frame, so a drift of DPVO's
        global rotation (several degrees on some OpenLORIS sequences) does not tilt the gravity the filter sees.  The
        gyro bias is the robust running median of (gyro - DPVO) rotation rates over intervals where both agree, and
        the camera-IMU time offset is calibrated once from the same rates (time_offset_after intervals)."""
        cfg = self.config.imu
        c = self.imu_calib
        samples = np.asarray(imu["imu"], dtype=np.float64)
        t0, t1 = float(imu["imu_t0"]), float(imu["imu_t1"])
        if len(samples):
            if len(self.imu_buffer):
                samples = samples[samples[:, 0] > self.imu_buffer[-1, 0]]
            self.imu_buffer = np.concatenate([self.imu_buffer, samples])[-cfg.imu_buffer_samples:]
        info = {}
        if t1 > t0 and len(self.imu_buffer):
            sel = (self.imu_buffer[:, 0] >= t0 - self.time_offset - 0.05) & (self.imu_buffer[:, 0] <= t1 - self.time_offset + 0.05)
            if sel.any():
                self.accel_history = (self.accel_history + [self.imu_buffer[sel, 4:7].mean(0)])[-20:]
        tracked = bool(initialized and tracker is not None)
        if tracked and self.prev_tracked and t1 > t0 and self.index - 1 >= self.frame_offset:
            self.imu_queue.append((self.index - 1, t0, t1))
        elif not tracked:
            self.imu_queue = []
            self.gyro_world = None
        self.prev_tracked = tracked
        f = self.scale_filter
        R_cb = c.T_cam_imu[:3, :3]
        while self.imu_queue and self.imu_queue[0][0] + 1 <= self.index - cfg.lag_frames:
            i, a0, a1 = self.imu_queue.pop(0)
            P_i, P_j = self._pose_at(i).astype(np.float64), self._pose_at(i + 1).astype(np.float64)
            dR_vis = P_i[:3, :3].T @ P_j[:3, :3]                  # DPVO's rotation over the interval (camera frame)
            self._calibrate_time_offset(a0, a1, R_cb.T @ dR_vis @ R_cb)
            # the interval's samples on the camera clock (camera = IMU + offset), bias-corrected
            buf = self.imu_buffer
            lo = max(int(np.searchsorted(buf[:, 0] + self.time_offset, a0, side="right")) - 1, 0)
            hi = min(int(np.searchsorted(buf[:, 0] + self.time_offset, a1, side="left")) + 1, len(buf))
            corrected = buf[lo:hi].copy()
            corrected[:, 0] += self.time_offset
            corrected[:, 1:4] -= self.gyro_bias
            pre = preintegrate(corrected, a0, a1, c.gyro_noise_density, c.accel_noise_density)
            dR_gyro = R_cb @ pre.dR @ R_cb.T
            rel = dR_gyro.T @ dR_vis
            slip = float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1.0, 1.0))))
            ok = slip <= cfg.gyro_check_deg
            if ok and pre.dt > 0:
                # gyro bias sample (IMU frame): the residual rotation rate of the gyro w.r.t. DPVO, plus the current bias
                w_res = _so3_log(R_cb.T @ rel.T @ R_cb) / pre.dt
                self.bias_samples = (self.bias_samples + [self.gyro_bias + w_res])[-cfg.gyro_bias_window:]
                if len(self.bias_samples) >= 10:
                    self.gyro_bias = np.median(np.asarray(self.bias_samples), axis=0)
            self.gyro_rejections += int(not ok)
            if self.gyro_world is None or self.gyro_world[0] != i:
                self.gyro_world = (i, P_i[:3, :3].copy())          # (re)anchor the gyro frame at DPVO's rotation
            R_gi = self.gyro_world[1]
            R_gj = R_gi @ dR_gyro
            self.gyro_world = (i + 1, R_gj)
            z = R_gi @ (P_i[:3, :3].T @ (P_j[:3, 3] - P_i[:3, 3]))   # DPVO's displacement, camera frame i -> gyro world
            if not f.started:
                f.start(R_gi, np.mean(self.accel_history, axis=0))
            info = f.step(pre, R_gi, R_gj, z, depth_units=self._depth_units(tracker), visual_ok=ok)
            info.update(rotation_slip_deg=slip, gyro_rejections=self.gyro_rejections,
                        gyro_bias=self.gyro_bias.round(5).tolist(), time_offset=self.time_offset)
        return info

    def _calibrate_time_offset(self, a0, a1, dR_vis_imu):
        """Collect DPVO rotation rates (IMU frame); after time_offset_after intervals, the camera-IMU time offset that
        best aligns the gyro with them (a grid over +-time_offset_max s), kept when it clearly beats no offset."""
        cfg = self.config.imu
        if self.time_offset_done or cfg.time_offset_after <= 0 or a1 <= a0:
            return
        self.rate_log.append((0.5 * (a0 + a1), _so3_log(dR_vis_imu) / (a1 - a0)))
        if len(self.rate_log) < cfg.time_offset_after:
            return
        tm = np.array([r[0] for r in self.rate_log])
        rates = np.array([r[1] for r in self.rate_log])
        buf = self.imu_buffer
        errs = []
        for off in np.arange(-cfg.time_offset_max, cfg.time_offset_max + 1e-9, 0.005):
            g = np.stack([np.interp(tm - off, buf[:, 0], buf[:, 1 + j]) for j in range(3)], axis=1) - self.gyro_bias
            errs.append((float(np.sqrt(((g - rates) ** 2).sum(1).mean())), float(off)))
        best = min(errs)
        at_zero = min(errs, key=lambda e: abs(e[1]))
        if best[0] < 0.85 * at_zero[0]:
            self.time_offset = round(best[1], 3)
        self.time_offset_done = True
        self.rate_log = []

    def finalized_trajectory(self):
        """Return offline-refined unit poses; never overwrite causal outputs."""
        if not self.tracker or not self.tracker.is_initialized:
            return None
        poses, timestamps = self.tracker.terminate()
        matrices = np.broadcast_to(np.eye(4), (len(poses), 4, 4)).copy()
        matrices[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()
        matrices[:, :3, 3] = poses[:, :3]
        return timestamps, matrices
