"""Visual-inertial odometry from the feed-forward geometry model (VGGT-Omega) and the IMU, without DPVO.

Every frame, the IMU carries the pose: the gyroscope (bias-corrected) the rotation, and the velocity, gravity and
accelerometer bias of the inertial estimator (cross.imu.InertialScaleFilter) the position.  Every `interval` frames a
visual measurement corrects it: VGGT-Omega's relative pose between the last measured frame m and the current frame b.
The measurement comes from the back end's forward pass when the back end observed this frame with m as its temporal
anchor (Pipeline: backend_anchor / after_backend), and otherwise from a two-image forward pass of the same model.

VGGT-Omega's translations are in a gauge of its own for every forward pass.  The passes are chained through the frame
both share: m's depth in this pass against its depth in the pass where it was the current frame gives the ratio of
the two gauges, so all displacements are in one slowly drifting unit, like a visual odometry's, and the inertial
estimator recovers the metres per unit, with learned metric depth (Depth Anything 3) as a prior on it.

Before the scale is known the frontend reports invalid motion, or with a continuous start unknown motion (the gyro's
rotation, wide translation covariance).

With a stereo rig (T_right_in_left; the stereo mode) the stereo pair observes each pass's metric scale directly
(VgiGraph.add_stereo): the same graph, without learned depth and its bias.  Two ways (vgio_stereo_source): "depth",
classical stereo matching (SGBM) of the current pair against the pass's depth of the current frame, or "baseline", the
right image as one more view of the pass and its left-right translation against the calibrated baseline."""

from dataclasses import replace
from time import perf_counter

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from cross.imu import InertialScaleFilter, preintegrate
from cross.imu.vgi_graph import GraphConfig, VgiGraph

from .frontend import MonoEstimate
from .geometry import inverse, scale_translation_covariance
from .scale import observe_scale


def _so3_log(R):
    return Rotation.from_matrix(R).as_rotvec()


def _exp_rot(rotvec):
    return Rotation.from_rotvec(rotvec).as_matrix()


def _ortho(R):
    """The nearest rotation (products of preintegrated rotations drift off SO(3) by ~1e-6 per frame)."""
    U, _, Vt = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def _angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


class VggtImuFrontend:
    def __init__(self, K, config, device="cuda", metric_model=None, backend=None, rgb_transform=None,
                 depth_transform=None, interval: int = 3, depth_every: int = 3, context: int = 1,
                 keyframe_age: float = 0.0, visual_rotation: bool = False, rotation_gate_deg: float = 3.0,
                 graph: bool = False, T_right_in_left=None):
        self.config = config
        self.K = np.asarray(K, dtype=np.float64).copy()
        self.device = device
        self.metric = metric_model               # DA3 metric depth (scale prior), or None
        # stereo rig: the right camera's pose in the left camera (metres), or None (monocular)
        self.T_rl = None if T_right_in_left is None else np.asarray(T_right_in_left, dtype=np.float64).copy()
        self._right = None                       # right image of the current frame (stereo)
        self._sgbm = None                        # stereo matcher (vgio_stereo_source depth)
        self._sgbm_depth = None                  # metric stereo depth of the current frame (full resolution), if computed
        self.backend = backend                   # the VGGT-Omega backend (cross.cv.pose_est_ff), shared with the back end
        self.rgb_transform = rgb_transform       # the back end's image transform (same tensors: token-cache hits)
        self.depth_transform = depth_transform
        self.interval = int(interval)            # frames between visual measurements
        self.depth_every = int(depth_every)      # frames between learned-depth observations (0: config.scale.interval)
        self.last_depth_index = -10 ** 9
        # measured frames in a pass of its own besides the current one: 2 adds the frame measured before the last one
        # (the gauges are chained through both, and the model sees more of the scene)
        self.context = int(context)
        self.m_prev = None
        # keyframe-pinned gauge (keyframe_age > 0 s): passes of its own also contain a keyframe, an older measured
        # frame whose depth in chain units stays fixed, and the gauge is chained through it, so the chain only drifts
        # when the keyframe is replaced (after keyframe_age s) instead of at every measurement
        self.keyframe_age = float(keyframe_age)
        self.kf = None
        # reported orientation (visual_rotation): VGGT-Omega's rotation from the keyframe (or the last measured frame) at
        # every measurement that agrees with the gyro within rotation_gate_deg, the gyro in between.  The estimator stays
        # in the gyro-propagated frame (R_wc); translations are reported in the camera frame, so the two frames may drift
        # apart slowly without harm
        self.visual_rotation = bool(visual_rotation)
        self.rotation_gate_deg = float(rotation_gate_deg)
        self.R_out = np.eye(3)
        self._reported_R_out = None
        # local pose graph (cross.imu.vgi_graph): every visual measurement is a node; all pairs of a pass, the gauge
        # links through shared frames, learned depth (with its bias), the IMU and its biases are optimized together
        self.use_graph = bool(graph)
        self.graph = None
        self.g_rep = None                        # gravity in the reported frame (graph mode)
        self.rotation_gate_deg_graph = 2.0       # deg: a pass whose rotation disagrees with the gyro's is not used
        # adaptive measurement times: (min frames, max frames, min translation m, min rotation deg), or None
        self.adaptive = None
        # measurements aligned with the back end's observations (pipeline): (min frames, frames before an own pass)
        self.align = None
        # optional tracked features (graph mode, vgio_klt): corners detected on each measured frame and tracked frame
        # to frame (pyramidal Lucas-Kanade, forward-backward check); at the next measurement their essential matrix
        # with the calibrated intrinsics gives the rotation between the two frames, a factor when it has enough
        # inliers and agrees with the gyro.  Without texture there is no factor and the graph is unchanged
        self.klt = False
        self.klt_gray = None
        self.klt_m = None                        # corners on the last measured frame
        self.klt_cur = None                      # their tracked positions in the current frame
        self.standalone = True                   # no back end: every measurement is a forward pass of its own
        self.continuous_start = False
        self.imu_calib = None
        self.scale_filter = None
        self.imu_buffer = np.zeros((0, 7))
        self.accel_history = []
        self.gyro_bias = np.zeros(3)
        self.bias_samples = []
        self.time_offset = 0.0
        self.time_offset_done = False
        self._pp = None                          # principal-point correction of the passes' camera poses
        self._last_trans_ok = None               # time of the last pass translation accepted by the IMU test
        self._rest_spread, self._rest_nis, self._rest_prev = [], [], None   # zero-rate updates (_rest_rate)
        self._rot_checks = []                    # outcomes of the last gyro tests of the passes' rotations
        self.rate_log = []
        self._offset_rounds = 0
        self.index = 0
        self.last_timestamp = None
        # camera orientation in the gyro-propagated world frame and position / velocity of the IMU (metric)
        self.R_wc = np.eye(3)
        self.p_cam = np.zeros(3)
        self.v_imu = np.zeros(3)
        self.metric_pose = np.eye(4)             # pose reported for the last frame
        self.output_valid = False
        # the last measured frame m: index, timestamp, image, rotation, position, depth in chain units
        self.m = None
        self.pending = None                      # anchor handed to the back end for the current frame
        self._frame = None                       # (index, timestamp, rgb) of the current frame
        self._reported_internal = None           # internal pose of the last reported frame
        self._token = 0
        self.stats = dict(measurements=0, backend_measurements=0, own_calls=0, slips=0, chain_breaks=0,
                          depth_priors=0, model_seconds=0.0)
        self.last_info = {}

    # ------------------------------------------------------------------ IMU buffer
    def _add_imu(self, frame):
        samples = np.asarray(frame.get("imu", np.zeros((0, 7))), dtype=np.float64)
        if len(samples):
            if len(self.imu_buffer):
                samples = samples[samples[:, 0] > self.imu_buffer[-1, 0]]
            self.imu_buffer = np.concatenate([self.imu_buffer, samples])[-self.config.imu.imu_buffer_samples:]
        t0, t1 = frame.get("imu_t0"), frame.get("imu_t1")
        if t0 is not None and t1 is not None and t1 > t0 and len(self.imu_buffer):
            sel = ((self.imu_buffer[:, 0] >= t0 - self.time_offset - 0.05)
                   & (self.imu_buffer[:, 0] <= t1 - self.time_offset + 0.05))
            if sel.any():
                self.accel_history = (self.accel_history + [self.imu_buffer[sel, 4:7].mean(0)])[-20:]

    def _samples(self, a0, a1):
        """Raw IMU samples covering camera times a0..a1, on the camera clock (camera = IMU clock + time_offset)."""
        buf = self.imu_buffer
        if a1 <= a0 or len(buf) < 2:
            return None
        lo = max(int(np.searchsorted(buf[:, 0] + self.time_offset, a0, side="right")) - 1, 0)
        hi = min(int(np.searchsorted(buf[:, 0] + self.time_offset, a1, side="left")) + 1, len(buf))
        part = buf[lo:hi].copy()
        if len(part) < 2:
            return None
        part[:, 0] += self.time_offset
        return part

    def _preintegrate(self, a0, a1):
        """IMU between camera times a0 and a1, gyro-bias-corrected."""
        c = self.imu_calib
        part = self._samples(a0, a1)
        if part is None:
            return None
        part[:, 1:4] -= self.gyro_bias
        return preintegrate(part, a0, a1, c.gyro_noise_density, c.accel_noise_density)

    # ------------------------------------------------------------------ per frame
    def track(self, frame):
        start = perf_counter()
        rgb = frame["rgb"]
        self._right = frame.get("rgb_right") if self.T_rl is not None else None
        # the IMU clock of this frame: the end of its IMU window (cross.dataloader.imu, camera times)
        timestamp = float(frame.get("imu_t1", frame["timestamp"]))
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Frame times must increase")
        if frame.get("imu_calib") is None:
            raise ValueError("The VGGT-IMU frontend needs frames with an IMU stream")
        if self.imu_calib is None:
            self.imu_calib = frame["imu_calib"]
            self.time_offset = float(self.config.imu.vgio_time_offset)
            c = self.imu_calib
            if self.use_graph:
                ic = self.config.imu
                gcfg = GraphConfig(window=ic.vgio_graph_window, depth_bias=ic.vgio_depth_bias,
                                   trans_rel=ic.vgio_visual_std_rel, extra_accel_noise=ic.extra_accel_noise,
                                   depth_std_floor=ic.depth_prior_std_floor, huber=ic.huber, rot_std=ic.vgio_rot_std,
                                   depth_bias_std=ic.vgio_depth_bias_std, rot_rel=ic.vgio_rot_rel,
                                   accel_bias_std=ic.accel_bias_std, accel_bias_walk=ic.accel_bias_walk,
                                   gyro_dt_noise=ic.vgio_gyro_dt_noise, accel_dt_noise=ic.vgio_accel_dt_noise,
                                   rot_scale=ic.vgio_rot_scale, rot_scale_std=ic.vgio_rot_scale_std,
                                   time_offset=ic.vgio_graph_time_offset, time_offset_std=ic.vgio_time_offset_std,
                                   trans_sigma_predicted=ic.vgio_trans_sigma_predicted, robust_links=ic.vgio_robust_links,
                                   trans_sigma_bound=ic.vgio_trans_sigma_bound,
                                   depth_bias_drift=ic.vgio_depth_bias_drift,
                                   gyro_bias_walk=c.gyro_random_walk if ic.vgio_calib_gyro_walk else ic.vgio_gyro_bias_walk)
                self.graph = VgiGraph(gcfg, c.T_cam_imu, c.gyro_noise_density, c.accel_noise_density)
                self.graph.td = self.graph.td_init = self.time_offset
                if ic.vgio_graph_time_offset:
                    self.time_offset_done = True         # the graph estimates the offset
                self.scale_filter = _GraphEstimate(self.graph)
            else:
                imu_cfg = replace(self.config.imu, visual_std_rel=self.config.imu.vgio_visual_std_rel,
                                  visual_std_depth=self.config.imu.vgio_visual_std_depth,
                                  scale_drift=self.config.imu.vgio_scale_drift)
                self.scale_filter = InertialScaleFilter(imu_cfg, c.T_cam_imu, c.gyro_noise_density,
                                                        c.accel_noise_density)
        ic = self.config.imu
        if ic.vgio_align and self.align is None and not self.standalone:
            self.align = (ic.vgio_align_min, ic.vgio_align_max)
        self.klt = bool(ic.vgio_klt and self.use_graph)
        if self.klt:
            t0 = perf_counter()
            self._klt_track(rgb)
            self.stats["t_klt"] = self.stats.get("t_klt", 0.0) + perf_counter() - t0
        if ic.vgio_adaptive and self.adaptive is None:
            self.adaptive = (ic.vgio_min_interval, ic.vgio_max_interval, ic.vgio_min_translation, ic.vgio_min_rotation_deg)
        self._add_imu(frame)
        f = self.scale_filter
        c = self.imu_calib
        R_cb, t_cb = c.T_cam_imu[:3, :3], c.T_cam_imu[:3, 3]
        # the internal pose reported for the last frame: a visual measurement since then moved the internal state, and
        # the motion reported now includes that correction
        internal_prev = self._reported_internal if self._reported_internal is not None else self._internal_pose()
        gyro_rotation = None
        if self.last_timestamp is not None:
            pre = self._preintegrate(self.last_timestamp, timestamp)
            if pre is not None:
                gyro_rotation = _ortho(R_cb @ pre.dR @ R_cb.T)
                R_wb = self.R_wc @ R_cb
                if f.started:
                    # the IMU state carried by the estimator's velocity, gravity and accelerometer bias
                    p_imu = self.p_cam + self.R_wc @ t_cb
                    dp = pre.dp - pre.J_p @ f.b
                    dv = pre.dv - pre.J_v @ f.b
                    g = self.g_rep if (self.use_graph and self.g_rep is not None) else f.g
                    p_imu = p_imu + self.v_imu * pre.dt + 0.5 * g * pre.dt ** 2 + R_wb @ dp
                    self.v_imu = self.v_imu + g * pre.dt + R_wb @ dv
                    self.R_wc = _ortho(self.R_wc @ gyro_rotation)
                    self.p_cam = p_imu - self.R_wc @ t_cb
                else:
                    self.R_wc = _ortho(self.R_wc @ gyro_rotation)
                self.R_out = _ortho(self.R_out @ gyro_rotation)
        # relative motion between internal poses (the reported trajectory starts wherever the unknown motion left it)
        valid = bool(f.initialized and f.started)
        R_out_prev = self._reported_R_out if self._reported_R_out is not None else self.R_out
        delta = self._relative(internal_prev, R_out_prev, self._internal_pose(), self.R_out)
        if valid:
            step = float(np.linalg.norm(delta[:3, 3]))
            covariance = np.diag([(0.01 + 0.1 * step) ** 2] * 3 + [0.005 ** 2] * 3)
            covariance[:3, :3] += scale_translation_covariance(delta[:3, 3], f.uncertainty_variance)
        elif self.continuous_start:
            delta = np.eye(4)
            if gyro_rotation is not None:
                delta[:3, :3] = gyro_rotation
            std = self.config.imu.unknown_motion_std
            covariance = np.diag([std[0] ** 2] * 3 + [std[2 if gyro_rotation is not None else 1] ** 2] * 3)
        else:
            delta = np.eye(4)
            covariance = np.eye(6)
        self.metric_pose = self.metric_pose @ delta
        self._reported_internal = self._internal_pose()
        self._reported_R_out = self.R_out.copy()
        diagnostics = {"frame": self.index, "valid": valid, "unknown_motion": bool(not valid and self.continuous_start),
                       "pose_source": "vggt_imu", "scale": f.scale, "log_scale_std": float(np.sqrt(f.uncertainty_variance)),
                       "metric_initialized": bool(f.initialized), "imu": dict(self.last_info),
                       "measurements": self.stats["measurements"], "own_calls": self.stats["own_calls"],
                       "backend_measurements": self.stats["backend_measurements"]}
        self.output_valid = valid
        self._frame = (self.index, timestamp, rgb)
        self.pending = None
        if self.standalone:
            self.after_backend(None)
        diagnostics["frontend_seconds"] = perf_counter() - start
        self.stats["t_track"] = self.stats.get("t_track", 0.0) + diagnostics["frontend_seconds"]
        self.stats["frames"] = self.stats.get("frames", 0) + 1
        self.last_timestamp = timestamp
        self.index += 1
        return MonoEstimate(float(frame["timestamp"]), self.metric_pose.copy(), delta, covariance, None, diagnostics)

    def _relative(self, A, RA_out, B, RB_out):
        """Motion from state A to B: the translation in A's camera frame (from the estimator's frame), the rotation
        from the reported orientations (visual_rotation) or the estimator's."""
        T = inverse(A) @ B
        if self.visual_rotation:
            T[:3, :3] = RA_out.T @ RB_out
        return T

    def _internal_pose(self):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R_wc, self.p_cam
        return T

    def _m_pose(self):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.m["R_wc"], self.m["p_cam"]
        return T

    # ------------------------------------------------------------------ back-end coupling
    def backend_anchor(self, observing=True):
        """The temporal anchor for the back end's forward pass at this frame: the last measured frame with the
        frontend's relative pose to it (metric once the scale is known).  The pipeline asks on the frames it hands to
        the back end; the back end may still skip the pass (its observation cadence)."""
        if not observing or self.m is None or self._frame is None:
            return None
        k = self._frame[0] - self.m["index"]
        if self.align is not None and self.scale_filter is not None:
            if k < self.align[0]:
                return None                      # too soon after the last measurement
        elif not self._due(self._frame[0]):
            return None                          # no measurement due: the pass need not carry the anchor
        self._token += 1
        T = self._relative(self._m_pose(), self.m["R_out"], self._internal_pose(), self.R_out)
        self.pending = {"token": self._token, "rgb": self.m["rgb"], "T_prev_curr": T, "metric": bool(self.output_valid)}
        if self.keyframe_age > 0 and self.kf is not None and self.kf["index"] != self.m["index"]:
            self.pending["extra_rgb"] = [self.kf["rgb"]]       # the keyframe rides along (its pairs in the graph)
        return self.pending

    def after_backend(self, observed):
        """After the back end's step: its forward pass with our anchor (cross.cv.pose_est_ff last_frontend_obs), or
        None.  A due measurement without one is made with a forward pass of its own."""
        if self._frame is None:
            return
        index = self._frame[0]
        if self.align is not None and not self.standalone and self.m is not None:
            # aligned with the back end: measure on its passes; a pass of our own only after align[1] frames without
            due = index - self.m["index"] >= self.align[1]
        else:
            due = self._due(index)
        t0 = perf_counter()
        if self.m is not None and observed is not None and self.pending is not None \
                and observed.get("token") == self.pending["token"] and index > self.m["index"]:
            self.stats["backend_measurements"] += 1
            self._measure(observed)              # the back end's pass carries the anchor (offered only when due)
        elif due:
            self._measure(self._own_pass())
        if due or observed is not None:
            self.stats["t_measure"] = self.stats.get("t_measure", 0.0) + perf_counter() - t0
        self.pending = None

    def _due(self, index):
        """A visual measurement is due after `interval` frames, or (adaptive) once the camera moved min_translation m
        or turned min_rotation deg since the last one, between min_interval and max_interval frames."""
        if self.m is None:
            return True
        k = index - self.m["index"]
        ad = self.adaptive
        if ad is None or not self.scale_filter.initialized:
            return k >= self.interval
        if k < ad[0]:
            return False
        if k >= ad[1]:
            return True
        moved = float(np.linalg.norm(self.p_cam - self.m["p_cam"]))
        turned = _angle_deg(self.m["R_wc"].T @ self.R_wc)
        return moved >= ad[2] or turned >= ad[3]

    def _own_pass(self):
        """Forward pass [current, last measured (, measured before)] (the current image alone for the first frame)."""
        if self.backend is None:
            return None
        start = perf_counter()
        current = self.rgb_transform(self._frame[2])
        views = [current] if self.m is None else [current, self.rgb_transform(self.m["rgb"])]
        third = None
        if self.m is not None and self.keyframe_age > 0 and self.kf is not None and self.kf["index"] != self.m["index"]:
            third = "kf"
            views.append(self.rgb_transform(self.kf["rgb"]))
        elif self.m is not None and self.context >= 2 and self.m_prev is not None:
            third = "m_prev"
            views.append(self.rgb_transform(self.m_prev["rgb"]))
        if self._right is not None and self._stereo_views():
            views.append(self.rgb_transform(self._right))         # the stereo pair's right image, last
        images = torch.stack(views).float().to(self.backend.device if hasattr(self.backend, "device") else "cuda")
        if images.max() > 1.5:
            images = images / 255.0
        with torch.inference_mode():
            pred = self.backend.infer(images, n_depth=None)
        self.stats["own_calls"] += 1
        self.stats["model_seconds"] += perf_counter() - start
        out = {"c2w_curr": pred.c2w[0], "depth_curr": pred.depth[0], "conf_curr": _conf(pred, 0)}
        if self.m is not None:
            out.update(c2w_prev=pred.c2w[1], depth_prev=pred.depth[1], conf_prev=_conf(pred, 1))
        if third == "m_prev":
            out.update(depth_prev2=pred.depth[2], conf_prev2=_conf(pred, 2))
        elif third == "kf":
            out.update(depth_kf=pred.depth[2], conf_kf=_conf(pred, 2), c2w_kf=pred.c2w[2])
        if self._right is not None and self._stereo_views():
            out["c2w_right"] = pred.c2w[len(views) - 1]
        return out

    def _stereo_views(self):
        """The right image rides along in the frontend's own passes (only the baseline source needs it)."""
        return self.config.imu.vgio_stereo_source in ("baseline", "both")

    # ------------------------------------------------------------------ the visual measurement
    def _klt_track(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.klt_gray is not None and self.klt_cur is not None and len(self.klt_cur) >= 8:
            lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.klt_gray, gray, self.klt_cur, None, **lk)
            p0, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.klt_gray, p1, None, **lk)
            fb = np.linalg.norm((p0 - self.klt_cur)[:, 0], axis=1)
            ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (fb < 1.0)
            if ok.sum() >= 8:            # per step: tracking noise (forward-backward error) and image motion
                self.klt_steps.append((float(np.median(fb[ok])), float(np.median(np.linalg.norm((p1 - self.klt_cur)[ok, 0], axis=1)))))
            self.klt_cur, self.klt_m = p1[ok], self.klt_m[ok]
        self.klt_gray = gray

    def _klt_detect(self):
        pts = cv2.goodFeaturesToTrack(self.klt_gray, maxCorners=300, qualityLevel=0.01, minDistance=8)
        self.klt_steps = []
        self.klt_m = None if pts is None else pts.astype(np.float32)
        self.klt_cur = None if pts is None else pts.astype(np.float32).copy()

    def _klt_rotation(self, gyro_mb):
        """(R of the current camera in the last measured frame's camera, std) from the tracked corners, or None."""
        if self.klt_m is None or self.klt_cur is None or len(self.klt_cur) < 30:
            return None
        a, b = self.klt_m[:, 0].astype(np.float64), self.klt_cur[:, 0].astype(np.float64)
        E, mask = cv2.findEssentialMat(a, b, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            return None
        n, R, t, mask2 = cv2.recoverPose(E, a, b, self.K, mask=mask)
        inliers = int((mask2 > 0).sum()) if mask2 is not None else 0
        if inliers < 30 or inliers < 0.5 * len(a):
            return None
        R_mb = R.T                                # x_b = R x_m + t: camera b's orientation in camera m
        diff = _angle_deg(gyro_mb.T @ R_mb)
        if diff > 1.0:
            self.stats["klt_rejected"] = self.stats.get("klt_rejected", 0) + 1
            return None
        self.klt_last = (R_mb, inliers)
        # the noise of these rotations, online: the median disagreement with the gyro (good to ~0.1 deg over 0.3 s)
        # over the last 100 accepted ones; the norm of a 3-D error has its median at 1.54 sigma per axis.  KITTI 07:
        # ~0.12 deg against ground truth, OpenLORIS home ~0.5 deg (low parallax, slow robot)
        self.klt_diffs = (getattr(self, "klt_diffs", []) + [diff])[-100:]
        sigma = max(0.1, float(np.median(self.klt_diffs)) / 1.54) if len(self.klt_diffs) >= 10 else 0.5
        self.stats["klt_used"] = self.stats.get("klt_used", 0) + 1
        self.stats["klt_sigma_deg"] = round(sigma, 3)
        return R_mb, np.radians(sigma) * np.sqrt(100.0 / max(inliers, 30))

    def _rest_rate(self, samples, pre, a0, a1):
        """The gyro bias measured directly over an interval at rest: (mean of the gyro samples, its std per axis), or
        None.  At rest is decided by the images: the tracked corners moved no more than their tracking noise allows.
        The noise-only median displacement after n steps, from the forward-backward errors of the same tracks, is
        fb sqrt(n / 2) (per step sigma = fb / (1.18 sqrt 2), the median of a 2-D norm 1.18 sigma); the chained tracks
        drift more than that at a standstill (ROVER night 3x: 0.13-0.18 px; day 15-20x: 0.02-0.03 px, where the
        forward-backward error of sharp static images is ~0.001 px), and driving moves them >= 3 px (1st percentile on
        ROVER, KITTI, OpenLORIS).  The test: displacement <= max(6 fb sqrt(2 n), 0.25 px, KLT's sub-pixel precision),
        >= 12x below the slowest motion seen.  A frame repeated by the camera (no image motion at all), an IMU gap or a
        gyro spread above the one at earlier rests (99 % F bound) vetoes.  The std: the sample spread / sqrt N, scaled
        by the consistency of consecutive rest means (correlated samples, a slowly moving bias; used only once 30
        such comparisons exist), and the rotation rate the images still allow (displacement / (focal length x
        interval))."""
        steps = getattr(self, "klt_steps", [])
        if self.klt_m is None or self.klt_cur is None or len(self.klt_cur) < 30 or not steps:
            return None
        fb = float(np.median([s[0] for s in steps]))
        if min(s[1] for s in steps) < 1e-3 or fb <= 0.0:          # a repeated frame: no evidence of rest
            return None
        disp = float(np.median(np.linalg.norm((self.klt_cur - self.klt_m)[:, 0], axis=1)))
        thr = max(6.0 * fb * np.sqrt(2.0 * len(steps)), 0.25)
        info = {"disp": round(disp, 4), "thr": round(thr, 4)}
        if disp > thr or getattr(pre, "dropout", 0.0):
            return info | {"ok": False}
        sel = (samples[:, 0] + self.time_offset > a0) & (samples[:, 0] + self.time_offset <= a1)
        w = samples[sel, 1:4]
        n = len(w)
        h = float(np.median(np.diff(self.imu_buffer[-200:, 0]))) if len(self.imu_buffer) > 2 else 0.0
        if n < 10 or (h > 0 and n < 0.5 * (a1 - a0) / h):
            return info | {"ok": False}
        spread = float(np.mean(w.std(0)))
        floor = self._rest_spread
        info["spread"] = spread
        if len(floor) >= 3:
            f0 = float(np.median(floor))
            if spread ** 2 > f0 ** 2 * (1.0 + 2.33 * np.sqrt(2.0 / (n - 1))):
                return info | {"ok": False, "floor": f0}
        self._rest_spread = (floor + [spread])[-50:]
        mean, std = w.mean(0), np.maximum(w.std(0), 1e-6) / np.sqrt(n)
        if self._rest_prev is not None and self._rest_prev[2] == self.m.get("index"):
            z2 = (mean - self._rest_prev[0]) ** 2 / (std ** 2 + self._rest_prev[1] ** 2)
            self._rest_nis = (self._rest_nis + z2.tolist())[-300:]
        self._rest_prev = (mean, std, self._frame[0])
        if len(self._rest_nis) < 30:
            # the noise model is not validated yet (too few consecutive rest windows to check it): no factor.  Six
            # windows in the first 2 s of OpenLORIS office1-1 pinned a start-up bias for the whole session and broke
            # relocalization against that map (office1-7 LR 0.53 -> 0)
            return info | {"ok": False, "uncalibrated": True}
        k2 = max(1.0, float(np.median(self._rest_nis)) / 0.455)
        self.stats["rest"] = self.stats.get("rest", 0) + 1
        w_img = disp / (float(self.K[0, 0]) * (a1 - a0))            # rad/s: the rotation the images allow
        return mean, np.sqrt(k2 * std ** 2 + w_img ** 2), info | {"ok": True, "k": round(float(np.sqrt(k2)), 2)}

    def _noise_hat(self, gyro_mb, vggt_mb, klt, dt):
        """The rotation noise of the gyro, the tracked corners and the passes from their disagreements over the same
        intervals ("three-cornered hat"): with independent errors the variance of each pair's difference is the sum of
        their variances, so var_a = (s_ab^2 + s_ac^2 - s_bc^2) / 2 (s: robust per-axis std, the median of the angle
        norm / 1.538).  Sets the gyro's noise density for the next IMU factors; returns the corners' factor (R, std), or
        None before 20 triplets."""
        if vggt_mb is None or self.klt_last is None:
            return None
        G = self.graph
        R_k, inliers = self.klt_last
        R_v = vggt_mb
        if G.cfg.rot_scale:
            R_v = _exp_rot(_so3_log(vggt_mb) * np.exp(-G.kappa))
        hat = self.hat = (getattr(self, "hat", []) + [(_angle_deg(gyro_mb.T @ R_k), _angle_deg(gyro_mb.T @ R_v),
                                                         _angle_deg(R_k.T @ R_v), inliers, dt)])[-100:]
        if len(hat) < 20:
            return None
        h = np.array(hat)
        s2 = (np.median(h[:, :3], axis=0) / 1.538) ** 2                     # gk, gv, kv
        floor = 0.02 ** 2
        var_g = max((s2[0] + s2[1] - s2[2]) / 2, floor)
        var_k = max((s2[0] + s2[2] - s2[1]) / 2, floor)
        var_v = max((s2[1] + s2[2] - s2[0]) / 2, floor)
        nominal = float(self.imu_calib.gyro_noise_density)
        G.gyro_noise = float(np.clip(np.radians(np.sqrt(var_g)) / np.sqrt(np.mean(h[:, 4])), nominal / 10, nominal * 3))
        # the passes keep their std: the hat measures them over one interval (0.3 s), and the same std would weight
        # their keyframe pairs (2 s), which are several times worse (setting it: worse on 5 of 8 sequences)
        n_med = float(np.median(h[:, 3]))
        self.stats["hat_deg"] = [round(float(np.sqrt(v)), 3) for v in (var_g, var_k, var_v)]
        return klt[0], float(np.radians(np.sqrt(var_k)) * np.sqrt(n_med / max(inliers, 30)))

    def _learned_depth(self, rgb, pass_depth, index):
        """Learned metric depth of this frame against a depth map of it (same pixels): a scale observation, at most
        every depth_every frames."""
        every = self.depth_every if self.depth_every else self.config.scale.interval
        if self.metric is None or not self.config.imu.depth_prior or index - self.last_depth_index < every:
            return None
        self.last_depth_index = index
        t0 = perf_counter()
        metric = self.metric.predict_metric(rgb, self.K, rgb.shape[:2])
        self.stats["t_da3"] = self.stats.get("t_da3", 0.0) + perf_counter() - t0
        observed = self._metric_depth_scale(metric, pass_depth, self.config.scale)
        if observed is not None:
            self.stats["depth_priors"] += 1
        return observed

    def _metric_depth_scale(self, metric, pass_depth, scale_config):
        """A metric depth map of the current image (full resolution; learned or stereo) against the pass's depth of it:
        the log metres per unit of the pass (cross.mono.scale.observe_scale), or None if the sizes differ."""
        metric_v = self.depth_transform(torch.from_numpy(np.asarray(metric, dtype=np.float32))[None])[0].numpy()
        source = pass_depth.float().cpu().numpy()
        if metric_v.shape != source.shape:
            return None
        metric_v[~np.isfinite(metric_v)] = 0.0
        return observe_scale(metric_v, source, None, scale_config)

    def _stereo_scale(self, obs, depth, conf):
        """The metric scale of a pass from the current stereo pair: (log metres per pass unit, its std) of the source
        vgio_stereo_source, or None; and a log of both sources.  Neither has a bias state: the rig is calibrated."""
        if self.T_rl is None or obs is None:
            return None, None
        source = self.config.imu.vgio_stereo_source
        info = {}
        out = {}
        if source in ("depth", "both"):
            out["depth"], info["depth"] = self._stereo_depth_scale(depth)
        if source in ("baseline", "both"):
            out["baseline"], info["baseline"] = self._stereo_baseline_scale(obs, depth, conf)
        use = out.get("baseline" if source == "baseline" else "depth")
        self.stats["stereo_used"] = self.stats.get("stereo_used", 0) + int(use is not None)
        if use is not None:
            info |= {"log": round(use[0], 4), "std": round(use[1], 4)}
        return use, info

    def _stereo_depth_scale(self, pass_depth):
        """Classical stereo depth (SGBM) of the current pair against the pass's depth of the left image, as learned depth
        is in the monocular case (cross.mono.scale.observe_scale: tiles, inliers; its std, floored at vgio_stereo_std).
        Pixels with less than vgio_stereo_min_disparity px of disparity are left out (their depth is mostly noise)."""
        if self._right is None:
            return None, {"ok": False, "reason": "no right image"}
        from cross.dataloader.stereo_loader import SGBMDepth
        from cross.mono.config import ScaleConfig
        ic = self.config.imu
        rgb = self._frame[2]
        if self._sgbm is None:
            self._sgbm = SGBMDepth(rgb.shape[1])
        t0 = perf_counter()
        fx, nb = float(self.K[0, 0]), float(np.linalg.norm(self.T_rl[:3, 3]))
        metric = self._sgbm(rgb, np.asarray(self._right), fx, nb)
        metric[metric > fx * nb / max(ic.vgio_stereo_min_disparity, 1e-3)] = 0.0
        self._sgbm_depth = metric
        self.stats["t_sgbm"] = self.stats.get("t_sgbm", 0.0) + perf_counter() - t0
        o = self._metric_depth_scale(metric, pass_depth, ScaleConfig(observation_std_floor=ic.vgio_stereo_std))
        if o is None:
            return None, {"ok": False, "reason": "shape"}
        info = {"ok": bool(o.accepted), "log": round(float(o.log_scale), 4), "std": round(float(np.sqrt(o.variance)), 4)
                if np.isfinite(o.variance) else None, "pixels": int(o.pixels), "mad": round(float(o.log_mad), 3),
                "inl": round(float(o.inlier_fraction), 3)}
        if not o.accepted:
            self.stats["stereo_depth_rejected"] = self.stats.get("stereo_depth_rejected", 0) + 1
            return None, info
        return (float(o.log_scale), float(np.sqrt(o.variance))), info

    def _stereo_baseline_scale(self, obs, depth, conf):
        """The right image as one more view of the pass: its centre in the left camera's frame along the calibrated
        baseline, against the baseline's length.  Not used when the pass's rotation between the two cameras or its
        baseline direction disagrees with the calibration (a wrongly registered right view carries no scale).  The std
        grows with the scene depth over the baseline, both in the pass's units (scale-free).  VGGT-Omega's baseline is
        biased by scene (KITTI 07: 13 % too long against the pass's own translations, KITTI 01: 30 %; OpenLORIS T265
        2-7 % short), while its depths agree with its translations (cross_mono_ff_vgio analysis: 0.99-1.02)."""
        if obs.get("c2w_right") is None:
            return None, {"ok": False, "reason": "no right view"}
        ic = self.config.imu
        T = inverse(np.asarray(obs["c2w_curr"], dtype=np.float64)) @ np.asarray(obs["c2w_right"], dtype=np.float64)
        b = self.T_rl[:3, 3]
        nb = float(np.linalg.norm(b))
        t = T[:3, 3]
        along = float(t @ b) / nb
        cos = along / max(float(np.linalg.norm(t)), 1e-12)
        rot = _angle_deg(self.T_rl[:3, :3].T @ T[:3, :3])
        d_over_b = _median_depth(depth, conf) / max(along, 1e-12)
        info = {"rot_deg": round(rot, 3), "cos": round(cos, 4), "d_over_b": round(float(d_over_b), 2)}
        if along <= 0 or rot > ic.vgio_stereo_rot_gate_deg or cos < ic.vgio_stereo_dir_cos or not np.isfinite(d_over_b):
            self.stats["stereo_baseline_rejected"] = self.stats.get("stereo_baseline_rejected", 0) + 1
            return None, info | {"ok": False}
        std = float(np.hypot(ic.vgio_stereo_std, ic.vgio_stereo_depth_k * d_over_b))
        log_obs = float(np.log(nb) - np.log(along))
        return (log_obs, std), info | {"ok": True, "log": round(log_obs, 4), "std": round(std, 4)}

    def _stereo_pnp(self, gyro_mb, gyro_sigma_deg):
        """The metric pose of the current frame b in the last measured frame m from the corners tracked between them
        (vgio_klt) and the stereo depth of m: (R_mb, t_mb, 6x6 covariance, info), or (None, info).  The covariance comes
        from the corners themselves: the pixel noise is the RMS reprojection error of the inliers, each corner's depth
        error the disparity noise (the same pixel noise) through z^2 / (f b), both propagated through the projection.
        A metric motion independent of the passes (VGGT-Omega can under-report long translations: KITTI 01) and of the
        IMU; used when it has enough inliers and its rotation agrees with the gyro."""
        depth = None if self.m is None else self.m.get("sgbm")
        if depth is None or self.klt_m is None or self.klt_cur is None or len(self.klt_cur) < 12:
            return None, {"ok": False, "reason": "no tracks or depth"}
        a = self.klt_m[:, 0].astype(np.float64)
        b = self.klt_cur[:, 0].astype(np.float64)
        h, w = depth.shape
        ui, vi = np.round(a[:, 0]).astype(int), np.round(a[:, 1]).astype(int)
        inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        z = np.zeros(len(a))
        z[inside] = depth[vi[inside], ui[inside]]
        ok = z > 0
        if ok.sum() < 12:
            return None, {"ok": False, "reason": "few corners with depth", "n": int(ok.sum())}
        K = self.K
        X = np.stack([(a[ok, 0] - K[0, 2]) / K[0, 0], (a[ok, 1] - K[1, 2]) / K[1, 1], np.ones(ok.sum())], 1) * z[ok, None]
        u = b[ok]
        try:
            found, rvec, tvec, inl = cv2.solvePnPRansac(X, u, K, None, iterationsCount=100, reprojectionError=2.0,
                                                       confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        except cv2.error:
            found = False
        if not found or inl is None or len(inl) < 12 or len(inl) < 0.3 * len(u):
            return None, {"ok": False, "reason": "pnp", "n": int(len(u)), "inl": 0 if inl is None else int(len(inl))}
        inl = inl[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(X[inl], u[inl], K, None, rvec, tvec)
        R_bm = cv2.Rodrigues(rvec)[0]
        t_bm = tvec[:, 0]
        Xb = X[inl] @ R_bm.T + t_bm                               # the inliers in b's camera frame
        if (Xb[:, 2] <= 1e-6).any():
            return None, {"ok": False, "reason": "behind"}
        proj = np.stack([K[0, 0] * Xb[:, 0] / Xb[:, 2] + K[0, 2], K[1, 1] * Xb[:, 1] / Xb[:, 2] + K[1, 2]], 1)
        res = proj - u[inl]
        s_px = max(float(np.sqrt((res ** 2).sum(1).mean() / 2.0)), 0.1)
        # information of (dtheta, dt): left perturbation of T_bm (x = Exp(dtheta) R X + t + dt)
        x, y, zz = Xb[:, 0], Xb[:, 1], Xb[:, 2]
        Jp = np.zeros((len(Xb), 2, 3))                            # d projection / d point (b frame)
        Jp[:, 0, 0], Jp[:, 0, 2] = K[0, 0] / zz, -K[0, 0] * x / zz ** 2
        Jp[:, 1, 1], Jp[:, 1, 2] = K[1, 1] / zz, -K[1, 1] * y / zz ** 2
        Jpose = np.concatenate([-Jp @ _skew3(Xb), Jp], 2)          # (n, 2, 6)
        nb = float(np.linalg.norm(self.T_rl[:3, 3]))
        zm = X[inl, 2]
        sig_z = zm ** 2 * s_px / (K[0, 0] * nb)                   # depth noise of each corner (disparity noise s_px)
        ray = (X[inl] / zm[:, None]) @ R_bm.T                     # d point (b frame) / d depth in m
        jz = np.einsum("nij,nj->ni", Jp, ray)                     # (n, 2)
        Su = s_px ** 2 * np.eye(2)[None] + (sig_z ** 2)[:, None, None] * jz[:, :, None] * jz[:, None, :]
        Si = np.linalg.inv(Su)
        Hm = np.einsum("nki,nkl,nlj->ij", Jpose, Si, Jpose)
        try:
            cov_d = np.linalg.inv(Hm)
        except np.linalg.LinAlgError:
            return None, {"ok": False, "reason": "singular"}
        # to the factor's parameters: R_mb = R_bm^T with a right perturbation -dtheta, t_mb = -R_bm^T t_bm
        R_mb, t_mb = R_bm.T, -R_bm.T @ t_bm
        A = np.zeros((6, 6))
        A[0:3, 0:3] = -np.eye(3)
        A[3:6, 0:3] = -R_bm.T @ _skew3(t_bm[None])[0]
        A[3:6, 3:6] = -R_bm.T
        cov = A @ cov_d @ A.T
        diff = _angle_deg(gyro_mb.T @ R_mb)
        info = {"ok": True, "n": int(len(u)), "inl": int(len(inl)), "px": round(s_px, 3), "t": round(float(np.linalg.norm(t_mb)), 4),
                "t_std": round(float(np.sqrt(np.trace(cov[3:6, 3:6]))), 4), "gyro_deg": round(diff, 3)}
        if diff > max(self.rotation_gate_deg_graph, 3.0 * gyro_sigma_deg):
            self.stats["pnp_rejected"] = self.stats.get("pnp_rejected", 0) + 1
            return None, info | {"ok": False, "reason": "gyro"}
        self.stats["pnp_used"] = self.stats.get("pnp_used", 0) + 1
        return (R_mb, t_mb, cov), info

    def _measure_graph(self, obs):
        """A visual measurement as a node of the local pose graph."""
        index, timestamp, rgb = self._frame
        G = self.graph
        R_cb = self.imu_calib.T_cam_imu[:3, :3]
        if obs is None or not np.isfinite(np.asarray(obs["c2w_curr"])).all():
            self.last_info = {"measured": False}
            return
        depth_curr = obs["depth_curr"].float()
        conf_curr = obs.get("conf_curr")
        observed = self._learned_depth(rgb, depth_curr, index)
        da3 = (float(observed.log_scale), float(np.sqrt(observed.variance))) if observed is not None and observed.accepted \
            else None
        stereo, stereo_info = self._stereo_scale(obs, depth_curr, conf_curr)
        if self.m is None or not G.ids:
            if not self.accel_history:
                return
            lam0 = stereo[0] if stereo is not None else da3[0] if da3 is not None \
                else float(-np.log(max(_median_depth(depth_curr, conf_curr), 1e-6)))
            node = G.start(self.R_wc, np.mean(self.accel_history, axis=0), lam0, timestamp, bg0=self.gyro_bias)
            if da3 is not None:
                G.add_depth(node, *da3)
            if stereo is not None:
                G.add_stereo(node, *stereo)
            self._set_m(index, timestamp, rgb, None, conf_curr, node=node, pass_depth=depth_curr)
            if self.klt:
                self._klt_detect()
            T0 = np.eye(4)
            T0[:3, :3], T0[:3, 3] = G.R[node], G.p[node]
            self.m["rep"] = T0
            self.g_rep = G.g.copy()
            self.last_info = {"measured": True, "first": True}
            return
        a0, a1 = self.m["timestamp"], timestamp
        samples = self._samples(a0, a1)
        if samples is None:
            self.last_info = {"measured": False, "reason": "no imu"}
            return
        c2w_c = np.asarray(obs["c2w_curr"], dtype=np.float64)
        c2w_m = np.asarray(obs["c2w_prev"], dtype=np.float64)
        T_mb = inverse(c2w_m) @ c2w_c
        self._calibrate_time_offset(a0, a1, R_cb.T @ T_mb[:3, :3] @ R_cb)
        pre, jac = G.preintegrate(samples, a0, a1, self.time_offset)
        m_node = self.m["node"]
        # the pass is used only if its rotation m -> b agrees with the gyro's (a wrong pass would pull the shared
        # biases): the gyro over 0.3 s is good to ~0.25 deg
        gyro_mb = R_cb @ pre.dR @ R_cb.T
        vdiff = _angle_deg(gyro_mb.T @ T_mb[:3, :3])
        # the test is as wide as the gyro's own uncertainty over the interval allows (an interval bridging a dropout
        # of the IMU stream is uncertain, and its rotation cannot veto the pass)
        gyro_sigma_deg = float(np.degrees(np.sqrt(max(np.trace(pre.cov[0:3, 0:3]) / 3.0, 0.0))))
        pass_ok = vdiff <= max(self.rotation_gate_deg_graph, 3.0 * gyro_sigma_deg)
        self._rot_checks = (self._rot_checks + [bool(pass_ok)])[-5:]
        if pass_ok and self.config.imu.vgio_online_noise:
            # the passes' rotation noise, online: median disagreement with the gyro over the last 100 measurements
            self.vggt_diffs = (getattr(self, "vggt_diffs", []) + [vdiff])[-100:]
            if len(self.vggt_diffs) >= 10:
                G.cfg.rot_std = np.radians(max(0.2, float(np.median(self.vggt_diffs)) / 1.54))
                self.stats["vggt_rot_sigma_deg"] = round(float(np.degrees(G.cfg.rot_std)), 3)
        ratio, spread = _depth_ratio([(self.m["pass_depth"], self.m["conf"], obs["depth_prev"].float(), obs.get("conf_prev"))])
        link_ok = bool(np.isfinite(ratio) and spread < 0.25)
        lam_init = stereo[0] if stereo is not None else G.lam[m_node] + (np.log(ratio) if link_ok else 0.0)
        kfc = self._keyframe_consistency(obs, c2w_c, c2w_m, index, ratio if link_ok else None)
        j = G.add_node(pre, jac, lam_init, timestamp)
        pnp, pnp_info = (None, None)
        if self.klt and self.T_rl is not None and self.config.imu.vgio_stereo_pnp:
            pnp, pnp_info = self._stereo_pnp(gyro_mb, gyro_sigma_deg)
        if self.T_rl is not None:
            pass_t, pnp_t, gate = self._stereo_gate(m_node, j, lam_init, T_mb[:3, 3], pass_ok, pnp, timestamp)
            if pnp is not None and not pnp_t:
                pnp = None
                pnp_info["ok"], pnp_info["reason"] = False, "gate"
                self.stats["pnp_gated"] = self.stats.get("pnp_gated", 0) + 1
            gated = pass_ok and not pass_t
        else:
            gate = self._translation_gate(m_node, j, lam_init, T_mb[:3, 3], timestamp) if pass_ok else None
            gated = gate is not None and not gate["ok"]
        if gated:
            pass_ok = False
            self.stats["trans_gated"] = self.stats.get("trans_gated", 0) + 1
        self.klt_last = None
        if pnp is not None:
            G.add_metric_relative(m_node, j, *pnp)       # the tracks' full metric motion (their rotation included)
        klt = self._klt_rotation(gyro_mb) if self.klt and pnp is None else None
        if klt is not None and self.config.imu.vgio_noise_hat:
            klt = self._noise_hat(gyro_mb, T_mb[:3, :3] if pass_ok else None, klt, a1 - a0) or klt
        if klt is not None:
            G.add_rotation(m_node, j, *klt)
        rest = self._rest_rate(samples, pre, a0, a1) if self.klt and self.config.imu.vgio_zero_rate else None
        if isinstance(rest, tuple):
            G.add_zero_rate(j, rest[0], rest[1])
            rest = rest[2]
        if pass_ok:
            G.add_relative(m_node, j, j, T_mb[:3, :3], T_mb[:3, 3])
            if link_ok:
                G.add_link(j, m_node, np.log(ratio))
        else:
            self.stats["slips"] += 1
        kf_used = kf_log = False
        if pass_ok and obs.get("c2w_kf") is not None and self.kf is not None and self.kf.get("node") in G.R:
            c2w_k = np.asarray(obs["c2w_kf"], dtype=np.float64)
            k_node = self.kf["node"]
            T_kb, T_km = inverse(c2w_k) @ c2w_c, inverse(c2w_k) @ c2w_m
            kf_log = {"kf_index": int(self.kf["index"]), "kf_t": float(np.linalg.norm(T_kb[:3, 3]))}
            # against the graph's current rotation from the keyframe (over ~2 s the gyro alone drifts)
            pred_km = G.R[k_node].T @ G.R[m_node]
            if _angle_deg(pred_km.T @ T_km[:3, :3]) <= 2 * self.rotation_gate_deg_graph:
                G.add_relative(k_node, j, j, T_kb[:3, :3], T_kb[:3, 3])
                G.add_relative(k_node, m_node, j, T_km[:3, :3], T_km[:3, 3])
            r_k, s_k = _depth_ratio([(self.kf["pass_depth"], self.kf["conf"], obs["depth_kf"].float(), obs.get("conf_kf"))])
            if np.isfinite(r_k) and s_k < 0.25:
                G.add_link(j, k_node, np.log(r_k))
            kf_used = True
        if da3 is not None:
            G.add_depth(j, *da3)
        if stereo is not None:
            G.add_stereo(j, *stereo)
        est = self.scale_filter
        t0 = perf_counter()
        # the scale's std is taken until initialization only (as before: the translation gate and the reported
        # uncertainty read that value); the velocity's std, for the bounded prediction, after every solve
        lam_std_wanted = not est.initialized
        info = G.solve(need_std=lam_std_wanted or self.config.imu.vgio_trans_sigma_bound)
        G.marginalize()
        if G.cfg.time_offset and abs(G.td - self.time_offset) > 0.004:
            # the samples of the window again at the graph's offset (its first-order correction is good to a few ms)
            self.time_offset = float(G.td)
            G.repreintegrate(self._samples, self.time_offset)
            self.stats["offset_updates"] = self.stats.get("offset_updates", 0) + 1
        self.stats["t_graph"] = self.stats.get("t_graph", 0.0) + perf_counter() - t0
        if lam_std_wanted and np.isfinite(info.get("lam_std", float("nan"))):
            est.lam_std = info["lam_std"]
        if not est.initialized and len(G.ids) >= 4 and est.lam_std < self.config.imu.init_log_std:
            est.initialized = True
        _, R, p, v, lam = G.latest()
        # the reported trajectory continues with the graph's relative pose from the previous node: re-optimizing the
        # window moves every node (a new scale stretches it), and reporting the newest node's absolute pose would turn
        # that into jumps.  Velocity and gravity are rotated into the reported frame for the propagation that follows
        T_j = np.eye(4)
        T_j[:3, :3], T_j[:3, 3] = R, p
        if self.m.get("rep") is not None and m_node in G.R:
            T_m = np.eye(4)
            T_m[:3, :3], T_m[:3, 3] = G.R[m_node], G.p[m_node]
            rep = self.m["rep"] @ inverse(T_m) @ T_j
        else:
            rep = T_j.copy()
        # the reported pose is a running product over the whole session: kept a rigid transform, else any deviation of
        # the factors from SO(3) compounds (it shrank the KITTI 00 trajectory by 14 % over 1500 measurements)
        rep[:3, :3] = _ortho(rep[:3, :3])
        A = rep[:3, :3] @ R.T
        self.R_wc, self.p_cam = _ortho(rep[:3, :3]), rep[:3, 3].copy()
        self.v_imu = A @ v
        self.g_rep = A @ G.g
        self.gyro_bias = G.bg.copy()
        self.stats["measurements"] += 1
        self.last_info = {"measured": True, "nodes": len(G.ids), "lam": float(lam), "lam_std": est.lam_std,
                          "gyro_bias": G.bg.round(5).tolist(), "accel_bias": G.ba.round(4).tolist(),
                          "depth_bias": float(np.exp(G.beta)), "rot_scale": float(np.exp(G.kappa)),
                          "gravity_norm": float(np.linalg.norm(G.g)),
                          "speed": float(np.linalg.norm(v)), "link_ok": link_ok, "keyframe": kf_used,
                          "time_offset": self.time_offset, "graph_time_offset": float(G.td), "cost": info.get("cost"), "m_index": int(self.m["index"]),
                          "b_index": int(index), "da3": None if da3 is None else [round(da3[0], 4), round(da3[1], 4)],
                          "link_log": float(np.log(ratio)) if link_ok else None, "pass_ok": bool(pass_ok),
                          "pass_gyro_deg": round(vdiff, 3), "pass_t": float(np.linalg.norm(T_mb[:3, 3])),
                          "trans_gate": gate, "rest": rest, "kfc": kfc, "v_std": round(float(G.v_std), 4) if np.isfinite(G.v_std) else None,
                          "stereo": stereo_info, "pnp": pnp_info, **(kf_log or {})}
        self._set_m(index, timestamp, rgb, None, conf_curr, node=j, pass_depth=depth_curr)
        self.m["rep"] = rep
        if self.klt:
            self._klt_detect()

    def _keyframe_consistency(self, obs, c2w_c, c2w_m, index, ratio):
        """Vision-only consistency of consecutive passes: the previous pass measured keyframe -> its current frame (=
        this pass's previous frame m), this pass measures keyframe -> m again; in metres at the gauge the depth ratio
        of m carries, log |t_km (this)| + log ratio - log |t_kb (previous)| should be ~0 (a pass whose translation
        collapsed relative to its depth disagrees).  Returns it (or None); remembers this pass's keyframe -> current."""
        out = None
        if obs.get("c2w_kf") is None or self.kf is None:
            self._kf_prev = None
            return None
        c2w_k = np.asarray(obs["c2w_kf"], dtype=np.float64)
        t_kb = float(np.linalg.norm((inverse(c2w_k) @ c2w_c)[:3, 3]))
        t_km = float(np.linalg.norm((inverse(c2w_k) @ c2w_m)[:3, 3]))
        prev = getattr(self, "_kf_prev", None)
        if prev is not None and ratio is not None and prev[0] == self.kf["index"] and prev[1] == self.m["index"] \
                and min(prev[2], t_km) > 1e-9:
            out = round(float(np.log(t_km) + np.log(ratio) - np.log(prev[2])), 4)
        self._kf_prev = (self.kf["index"], index, t_kb)
        return out

    def _imu_prediction(self, m_node, j, timestamp):
        """The IMU's prediction of the motion since the last measured frame for the translation tests: (interval, speed
        of the new node's IMU-propagated position relative to the last measured one, time since the last accepted
        translation, whether the IMU is consistent with the passes: the gyro tests of the last five passes' rotations
        all passed)."""
        G = self.graph
        dt = max(timestamp - self.m["timestamp"], 1e-3)
        v_pred = float(np.linalg.norm(G.R[m_node].T @ (G.p[j] - G.p[m_node]))) / dt
        if self._last_trans_ok is None:
            self._last_trans_ok = self.m["timestamp"]
        gap = max(timestamp - self._last_trans_ok, dt)
        return dt, v_pred, gap, len(self._rot_checks) >= 5 and all(self._rot_checks)

    def _stereo_gate(self, m_node, j, lam_j, t_pass, pass_ok, pnp, timestamp):
        """Translation tests of the stereo case, with the IMU as the arbiter between the pass and the tracked corners'
        motion (stereo PnP).  A visual translation is accepted when its speed agrees with the IMU's prediction within 4
        sigma: the pass's sigma from its relative noise and the scale's uncertainty (small: the stereo pair observes
        it), the corners' from their covariance, the IMU's from the time since the last accepted translation (as in the
        monocular test, without its factor tolerance, which there covers the learned-depth scale; vgio_stereo_gate_factor
        > 1 restores it).  The two visual cues are not independent where moving objects fill the view (KITTI 01:
        vehicles alongside made the corners report ~1 m/s and the passes half the speed at a true 27 m/s), so they
        override the IMU only when it is not trustworthy (its gyro disagrees with the passes) or after
        vgio_trans_gate_max_gap s without an accepted translation; then a corner motion that disagrees with the pass is
        dropped.  Returns (pass translation ok, corners ok, info), or (True, True, None) before the scale is known."""
        ic = self.config.imu
        if float(ic.vgio_trans_gate) <= 1.0 or not self.scale_filter.initialized:
            return True, True, None
        G = self.graph
        dt, v_pred, gap, imu_consistent = self._imu_prediction(m_node, j, timestamp)
        lam_std = self.scale_filter.lam_std if np.isfinite(self.scale_filter.lam_std) else 1.0
        s_imu = 0.05 + G.cfg.accel_bias_std * gap
        g = float(ic.vgio_stereo_gate_factor)

        def agree(v, s):
            lr = np.log((v * dt + 0.02) / (v_pred * dt + 0.02))
            return bool((g > 1.0 and abs(lr) <= np.log(g)) or abs(v - v_pred) <= 4.0 * np.hypot(s, s_imu))
        v_pass = float(np.exp(lam_j) * np.linalg.norm(t_pass)) / dt
        s_pass = float(np.sqrt(G.cfg.trans_rel ** 2 + min(lam_std, 1.0) ** 2)) * v_pass
        info = {"v_imu": round(v_pred, 3), "tol_imu": round(4.0 * s_imu, 3), "v_pass": round(v_pass, 3),
                "imu_consistent": imu_consistent}
        trusted = imu_consistent and gap <= float(ic.vgio_trans_gate_max_gap)
        pass_t = agree(v_pass, s_pass) if pass_ok else False
        pnp_ok = True
        if pnp is not None:
            t_p = pnp[1]
            L = float(np.linalg.norm(t_p))
            u = t_p / max(L, 1e-9)
            v_pnp, s_pnp = L / dt, float(np.sqrt(max(u @ pnp[2][3:6, 3:6] @ u, 0.0))) / dt
            pnp_ok = agree(v_pnp, s_pnp)
            info.update(v_pnp=round(v_pnp, 3))
            if not trusted:
                both = abs(v_pass - v_pnp) <= 4.0 * np.hypot(s_pass, s_pnp)
                pnp_ok = pnp_ok or (pass_ok and both)
        if not trusted:
            pass_t = pass_ok
        if pass_t or pnp_ok and pnp is not None:
            self._last_trans_ok = timestamp
        info.update(ok=bool(pass_t), pnp_ok=bool(pnp_ok) if pnp is not None else None, trusted=trusted)
        return pass_t, pnp_ok, info

    def _translation_gate(self, m_node, j, lam_j, t_pass, timestamp):
        """Test of a pass's translation against the IMU's prediction (the counterpart of the gyro test of its rotation):
        the new node's IMU-propagated position relative to the last measured one, against the pass's translation in
        metres at the gauge it inherits, as velocities over the interval.  The IMU may outvote a pass only while it is
        consistent with the passes: the gyro tests of the last five passes' rotations all passed (a dropout or a clock
        jump of the IMU stream shows there first, and a drifting IMU velocity must not then lock out the passes that
        would correct it).  Rejected when the lengths differ by more than the factor vgio_trans_gate and the velocities
        by more than 4 sigma, sigma combining the pass's relative noise, the scale's uncertainty and the IMU velocity's,
        which grows with the time since the last accepted translation (accelerometer bias std x time).  After
        vgio_trans_gate_max_gap s without an accepted translation the pass is accepted whatever it says.  Before the
        scale is known no test.  Returns None (no test) or {"ok", "log_ratio", "dv", "tol"}."""
        g = float(self.config.imu.vgio_trans_gate)
        G = self.graph
        if g <= 1.0 or not self.scale_filter.initialized:
            return None
        dt, v_pred, gap, imu_consistent = self._imu_prediction(m_node, j, timestamp)
        v_meas = float(np.exp(lam_j) * np.linalg.norm(t_pass)) / dt
        lam_std = self.scale_filter.lam_std if np.isfinite(self.scale_filter.lam_std) else 1.0
        sig_v = float(np.sqrt((G.cfg.trans_rel ** 2 + lam_std ** 2) * v_meas ** 2
                              + (0.05 + G.cfg.accel_bias_std * gap) ** 2))
        lr = float(np.log((v_meas * dt + 0.02) / (v_pred * dt + 0.02)))
        ok = (not imu_consistent or abs(lr) <= np.log(g) or abs(v_meas - v_pred) <= 4.0 * sig_v
              or gap > float(self.config.imu.vgio_trans_gate_max_gap))
        if ok:
            self._last_trans_ok = timestamp
        return {"ok": bool(ok), "log_ratio": round(lr, 3), "dv": round(v_meas - v_pred, 3), "tol": round(4.0 * sig_v, 3),
                "imu_consistent": imu_consistent}

    def _model_to_cam(self, obs):
        """The pass's camera poses in the calibrated camera frame: VGGT-Omega places the principal point at the image
        centre (the back end's correction, cross.cv.pose_est_ff.principal_point_rotation; KITTI ~0.9 deg, OpenLORIS
        ~1.4 deg), which would otherwise act as a camera-IMU rotation error of that size."""
        if obs is None or not self.config.imu.vgio_pp_correction:
            return obs
        if self._pp is None:
            from cross.cv.pose_est_ff import principal_point_rotation
            h, w = self._frame[2].shape[:2]
            self._pp = np.eye(4)
            self._pp[:3, :3] = principal_point_rotation(self.K, w, h).T
        out = dict(obs)
        for k in ("c2w_curr", "c2w_prev", "c2w_kf", "c2w_right"):
            if out.get(k) is not None:
                out[k] = np.asarray(out[k], dtype=np.float64) @ self._pp
        return out

    def _measure(self, obs):
        obs = self._model_to_cam(obs)
        if self.use_graph:
            return self._measure_graph(obs)
        index, timestamp, rgb = self._frame
        cfg = self.config.imu
        f = self.scale_filter
        c = self.imu_calib
        R_cb = c.T_cam_imu[:3, :3]
        info = {}
        if obs is None or not np.isfinite(np.asarray(obs["c2w_curr"])).all():
            self.last_info = {"measured": False}
            return
        depth_curr = obs["depth_curr"].float()
        conf_curr = obs.get("conf_curr")
        if self.m is None:
            # first measured frame: the chain's unit is the median depth of this frame
            unit = 1.0 / max(_median_depth(depth_curr, conf_curr), 1e-6)
            self._set_m(index, timestamp, rgb, depth_curr * unit, conf_curr)
            self.last_info = {"measured": True, "first": True}
            return
        T = inverse(np.asarray(obs["c2w_prev"], dtype=np.float64)) @ np.asarray(obs["c2w_curr"], dtype=np.float64)
        depth_prev = obs["depth_prev"].float()
        # the pass's gauge in chain units, through m's depth in this pass and in the chain
        pairs = [(self.m["depth"], self.m["conf"], depth_prev, obs.get("conf_prev"))]
        if obs.get("depth_prev2") is not None and self.m_prev is not None:
            pairs.append((self.m_prev["depth"], self.m_prev["conf"], obs["depth_prev2"].float(), obs.get("conf_prev2")))
        via_kf = False
        if obs.get("depth_kf") is not None and self.kf is not None:
            kf_ratio, kf_spread = _depth_ratio([(self.kf["depth"], self.kf["conf"], obs["depth_kf"].float(),
                                                 obs.get("conf_kf"))])
            if np.isfinite(kf_ratio) and kf_spread < 0.25:
                pairs, via_kf = None, True
                ratio, spread = kf_ratio, kf_spread
        if pairs is not None:
            ratio, spread = _depth_ratio(pairs)
        chain_ok = np.isfinite(ratio) and spread < 0.25
        if not chain_ok:
            self.stats["chain_breaks"] += 1
            ratio = self.m["unit_ratio"]          # keep the last pass's gauge
        z_cam = ratio * T[:3, 3]
        a0, a1 = self.m["timestamp"], timestamp
        pre = self._preintegrate(a0, a1)
        if pre is None:
            self._set_m(index, timestamp, rgb, depth_curr * ratio, conf_curr, ratio)
            self.last_info = {"measured": False, "reason": "no imu"}
            return
        dR_vis = T[:3, :3]
        dR_gyro = R_cb @ pre.dR @ R_cb.T
        rel = dR_gyro.T @ dR_vis
        slip = _angle_deg(rel)
        ok = slip <= self.config.imu.gyro_check_deg and chain_ok
        if slip <= self.config.imu.gyro_check_deg and pre.dt > 0:
            w_res = _so3_log(R_cb.T @ rel.T @ R_cb) / pre.dt
            self.bias_samples = (self.bias_samples + [self.gyro_bias + w_res])[-cfg.gyro_bias_window:]
            if len(self.bias_samples) >= 10:
                self.gyro_bias = np.median(np.asarray(self.bias_samples), axis=0)
        self.stats["slips"] += int(slip > self.config.imu.gyro_check_deg)
        self._calibrate_time_offset(a0, a1, R_cb.T @ dR_vis @ R_cb)
        R_gi = self.m["R_wc"]
        R_gj = R_gi @ dR_gyro
        z = R_gi @ z_cam
        depth_units = float(ratio * _median_depth(depth_curr, conf_curr))
        if not f.started:
            f.start(R_gi, np.mean(self.accel_history, axis=0))
        # learned metric depth of this frame against its depth in chain units: a measurement of the scale
        da3_log_scale = None
        # at the learned-depth interval of the DPVO frontend (metric_interval frames): more frequent independent
        # observations outweigh the IMU (ROVER, where learned depth is 1.9x off: scale 2.1 vs 1.3 without them)
        every = self.depth_every if self.depth_every else self.config.scale.interval
        if self.metric is not None and cfg.depth_prior and index - self.last_depth_index >= every:
            self.last_depth_index = index
            metric = self.metric.predict_metric(rgb, self.K, rgb.shape[:2])
            metric_v = self.depth_transform(torch.from_numpy(np.asarray(metric, dtype=np.float32))[None])[0]
            chain_depth = (depth_curr * ratio).cpu().numpy()
            metric_v = metric_v.numpy()
            if metric_v.shape == chain_depth.shape:
                observed = observe_scale(metric_v, chain_depth, None, self.config.scale)
                f.update(observed)
                if observed.accepted:
                    da3_log_scale = float(observed.log_scale)
                self.stats["depth_priors"] += 1
        # the reported orientation of this frame from VGGT-Omega's rotation: from the keyframe when it is in the pass
        rot_gate = None
        if self.visual_rotation and slip <= self.config.imu.gyro_check_deg:
            if obs.get("c2w_kf") is not None and self.kf is not None and self.kf.get("R_out") is not None:
                R_vis = self.kf["R_out"] @ (inverse(np.asarray(obs["c2w_kf"], dtype=np.float64))
                                            @ np.asarray(obs["c2w_curr"], dtype=np.float64))[:3, :3]
            else:
                R_vis = self.m["R_out"] @ dR_vis
            R_vis = _ortho(R_vis)
            rot_gate = _angle_deg(self.R_out.T @ R_vis)
            if rot_gate <= self.rotation_gate_deg:
                self.R_out = R_vis
        was_initialized = f.initialized
        info = f.step(pre, R_gi, R_gj, z, depth_units=depth_units, visual_ok=ok)
        self.stats["measurements"] += 1
        # the measured position of the current frame: the last measured position plus the scaled displacement; the
        # velocity of the estimator.  The orientation stays the gyro's (VGGT's rotation checks it and its bias).
        if f.initialized:
            self.v_imu = f.velocity
            if ok:
                p_b = self.m["p_cam"] + f.scale * z
            else:
                p_b = self.p_cam                 # IMU propagation only
            R_b = self.R_wc
            if not was_initialized:
                p_b = self.p_cam                 # start of the metric trajectory: keep the reported position
            # the next frames are propagated from the measured state
            self.p_cam = p_b
            self.R_wc = R_b
        info.update(rotation_correction_deg=rot_gate, via_keyframe=via_kf, z_units=float(np.linalg.norm(z)), m_index=int(self.m["index"]), b_index=int(index),
                    da3_log_scale=da3_log_scale,
                    rotation_slip_deg=slip, chain_ratio=float(ratio), chain_spread=float(spread),
                    chain_ok=bool(chain_ok), gyro_bias=self.gyro_bias.round(5).tolist(), time_offset=self.time_offset,
                    measured=True, visual_ok=bool(ok))
        self.last_info = info
        self._set_m(index, timestamp, rgb, depth_curr * ratio, conf_curr, ratio)

    def _set_m(self, index, timestamp, rgb, depth_chain, conf, unit_ratio=1.0, node=None, pass_depth=None):
        self.m_prev = self.m
        sgbm, self._sgbm_depth = self._sgbm_depth, None          # the frame's stereo depth (for the next PnP)
        self.m = {"index": index, "timestamp": timestamp, "rgb": rgb, "R_wc": self.R_wc.copy(), "R_out": self.R_out.copy(),
                  "p_cam": self.p_cam.copy(), "depth": depth_chain, "conf": conf, "unit_ratio": unit_ratio,
                  "node": node, "pass_depth": pass_depth, "sgbm": sgbm}
        if self.keyframe_age > 0 and (self.kf is None or timestamp - self.kf["timestamp"] >= self.keyframe_age):
            self.kf = self.m                     # the new keyframe: its chain depth stays fixed from now on

    def _calibrate_time_offset(self, a0, a1, dR_vis_imu):
        """As DPVOFrontend._calibrate_time_offset, from the measured intervals' rotation rates."""
        cfg = self.config.imu
        if self.time_offset_done or cfg.time_offset_after <= 0 or a1 <= a0:
            return
        self.rate_log.append((0.5 * (a0 + a1), _so3_log(dR_vis_imu) / (a1 - a0), a1 - a0))
        # graph mode: a first calibration after 20 measurements (6 s), a final one after 100; the window's IMU
        # factors are preintegrated again with the new offset
        need = (20 if not self._offset_rounds else 100) if self.use_graph else max(30, cfg.time_offset_after // max(1, self.interval))
        if len(self.rate_log) < need:
            return
        tm = np.array([r[0] for r in self.rate_log])
        rates = np.array([r[1] for r in self.rate_log])
        if self.use_graph and float(np.sqrt((rates ** 2).sum(1).mean())) < 0.1:
            self.rate_log = self.rate_log[-300:]
            return                           # too little rotation to see an offset (a standing robot): wait
        buf = self.imu_buffer
        errs = []
        if self.use_graph:
            # the gyro integrated over each measured interval against the visual rotation of the interval (a rotation
            # vector to first order; robust median): sharper than rates at interval midpoints when the gyro is sampled
            # at the frame rate (KITTI's OXTS, 10 Hz: 0.07 deg per 0.3 s at the right offset, 0.22 deg at zero)
            t = buf[:, 0]
            w = buf[:, 1:4] - self.gyro_bias
            cum = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t)[:, None], axis=0)])
            half = np.array([0.5 * r[2] for r in self.rate_log])                  # interval half lengths
            target = rates * (2 * half)[:, None]                                  # visual rotation vectors
            for off in np.arange(-cfg.time_offset_max, cfg.time_offset_max + 1e-9, 0.0025):
                lo, hi = tm - half - off, tm + half - off
                integ = np.stack([np.interp(hi, t, cum[:, j]) - np.interp(lo, t, cum[:, j]) for j in range(3)], axis=1)
                errs.append((float(np.median(np.linalg.norm(integ - target, axis=1))), float(off)))
        else:
            for off in np.arange(-cfg.time_offset_max, cfg.time_offset_max + 1e-9, 0.005):
                # mean gyro rate over each interval (the rates are interval averages)
                g = np.stack([np.interp(tm - off, buf[:, 0], buf[:, 1 + j]) for j in range(3)], axis=1) - self.gyro_bias
                errs.append((float(np.sqrt(((g - rates) ** 2).sum(1).mean())), float(off)))
        best = min(errs)
        at_zero = min(errs, key=lambda e: abs(e[1]))
        old = self.time_offset
        if best[0] < (0.7 if self.use_graph else 0.85) * at_zero[0]:
            self.time_offset = round(best[1], 3)
        self._offset_rounds += 1
        if self.use_graph:
            self.time_offset_done = self._offset_rounds >= 2
            if self.graph is not None and self.time_offset != old:
                self.graph.repreintegrate(self._samples, self.time_offset)
            if not self.time_offset_done:
                return                                   # the rate log keeps growing for the final round
        self.time_offset_done = True
        self.rate_log = []

    def shutdown(self):
        from loguru import logger
        logger.info(f"VGGT-IMU frontend: {self.stats}")


class _GraphEstimate:
    """The local pose graph's newest node through the interface of InertialScaleFilter that track() reads."""

    def __init__(self, graph):
        self.graph = graph
        self.initialized = False
        self.lam_std = float("inf")

    @property
    def started(self):
        return bool(self.graph.ids)

    @property
    def g(self):
        return self.graph.g

    @property
    def b(self):
        return -self.graph.ba            # InertialScaleFilter's sign: dv(b) = dv - J_v b

    @property
    def scale(self):
        return float(np.exp(self.graph.lam[self.graph.ids[-1]])) if self.graph.ids else 1.0

    @property
    def uncertainty_variance(self):
        return min(self.lam_std ** 2, 1.0)

    @property
    def velocity(self):
        return self.graph.v[self.graph.ids[-1]].copy()


def _skew3(v):
    """Skew matrices of (n, 3) vectors."""
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], -1), np.stack([v[:, 2], z, -v[:, 0]], -1),
                     np.stack([-v[:, 1], v[:, 0], z], -1)], -2)


def _conf(pred, i):
    return None if pred.depth_conf is None else pred.depth_conf[i]


def _median_depth(depth, conf=None):
    d = depth.float()
    mask = torch.isfinite(d) & (d > 0)
    if conf is not None:
        cf = conf.float()
        mask &= cf >= torch.quantile(cf[mask].float(), 0.5) if mask.any() else mask
    return float(d[mask].median()) if mask.any() else float("nan")


def _depth_ratio(pairs):
    """Median of chain / pass depth over the confident pixels of each shared image (chain depth, chain confidence, pass
    depth, pass confidence), pooled, and its robust log spread."""
    logs = []
    for chain_depth, chain_conf, pass_depth, pass_conf in pairs:
        a, b = chain_depth.float(), pass_depth.float().to(chain_depth.device)
        if a.shape != b.shape:
            continue
        mask = torch.isfinite(a) & torch.isfinite(b) & (a > 0) & (b > 0)
        for cf in (chain_conf, pass_conf):
            if cf is not None and mask.any():
                cf = cf.float().to(a.device)
                mask &= cf >= torch.quantile(cf[mask], 0.3)
        if mask.sum() >= 100:
            logs.append(torch.log(a[mask]) - torch.log(b[mask]))
    if not logs:
        return float("nan"), float("inf")
    r = torch.cat(logs)
    center = r.median()
    spread = float(1.4826 * (r - center).abs().median())
    return float(torch.exp(center)), spread
