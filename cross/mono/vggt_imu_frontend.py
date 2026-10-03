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
rotation, wide translation covariance)."""

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
                 graph: bool = False):
        self.config = config
        self.K = np.asarray(K, dtype=np.float64).copy()
        self.device = device
        self.metric = metric_model               # DA3 metric depth (scale prior), or None
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
                                   time_offset=ic.vgio_graph_time_offset, time_offset_std=ic.vgio_time_offset_std)
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
        return out

    # ------------------------------------------------------------------ the visual measurement
    def _klt_track(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.klt_gray is not None and self.klt_cur is not None and len(self.klt_cur) >= 8:
            lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.klt_gray, gray, self.klt_cur, None, **lk)
            p0, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.klt_gray, p1, None, **lk)
            ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((p0 - self.klt_cur)[:, 0], axis=1) < 1.0)
            self.klt_cur, self.klt_m = p1[ok], self.klt_m[ok]
        self.klt_gray = gray

    def _klt_detect(self):
        pts = cv2.goodFeaturesToTrack(self.klt_gray, maxCorners=300, qualityLevel=0.01, minDistance=8)
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
        # the noise of these rotations, online: the median disagreement with the gyro (good to ~0.1 deg over 0.3 s)
        # over the last 100 accepted ones; the norm of a 3-D error has its median at 1.54 sigma per axis.  KITTI 07:
        # ~0.12 deg against ground truth, OpenLORIS home ~0.5 deg (low parallax, slow robot)
        self.klt_diffs = (getattr(self, "klt_diffs", []) + [diff])[-100:]
        sigma = max(0.1, float(np.median(self.klt_diffs)) / 1.54) if len(self.klt_diffs) >= 10 else 0.5
        self.stats["klt_used"] = self.stats.get("klt_used", 0) + 1
        self.stats["klt_sigma_deg"] = round(sigma, 3)
        return R_mb, np.radians(sigma) * np.sqrt(100.0 / max(inliers, 30))

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
        metric_v = self.depth_transform(torch.from_numpy(np.asarray(metric, dtype=np.float32))[None])[0].numpy()
        source = pass_depth.float().cpu().numpy()
        if metric_v.shape != source.shape:
            return None
        self.stats["depth_priors"] += 1
        return observe_scale(metric_v, source, None, self.config.scale)

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
        if self.m is None or not G.ids:
            if not self.accel_history:
                return
            lam0 = da3[0] if da3 is not None else float(-np.log(max(_median_depth(depth_curr, conf_curr), 1e-6)))
            node = G.start(self.R_wc, np.mean(self.accel_history, axis=0), lam0, timestamp, bg0=self.gyro_bias)
            if da3 is not None:
                G.add_depth(node, *da3)
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
        pass_ok = vdiff <= self.rotation_gate_deg_graph
        if pass_ok and self.config.imu.vgio_online_noise:
            # the passes' rotation noise, online: median disagreement with the gyro over the last 100 measurements
            self.vggt_diffs = (getattr(self, "vggt_diffs", []) + [vdiff])[-100:]
            if len(self.vggt_diffs) >= 10:
                G.cfg.rot_std = np.radians(max(0.2, float(np.median(self.vggt_diffs)) / 1.54))
                self.stats["vggt_rot_sigma_deg"] = round(float(np.degrees(G.cfg.rot_std)), 3)
        ratio, spread = _depth_ratio([(self.m["pass_depth"], self.m["conf"], obs["depth_prev"].float(), obs.get("conf_prev"))])
        link_ok = bool(np.isfinite(ratio) and spread < 0.25)
        lam_init = G.lam[m_node] + (np.log(ratio) if link_ok else 0.0)
        j = G.add_node(pre, jac, lam_init, timestamp)
        klt = self._klt_rotation(gyro_mb) if self.klt else None
        if klt is not None:
            G.add_rotation(m_node, j, *klt)
        if pass_ok:
            G.add_relative(m_node, j, j, T_mb[:3, :3], T_mb[:3, 3])
            if link_ok:
                G.add_link(j, m_node, np.log(ratio))
        else:
            self.stats["slips"] += 1
        kf_used = False
        if pass_ok and obs.get("c2w_kf") is not None and self.kf is not None and self.kf.get("node") in G.R:
            c2w_k = np.asarray(obs["c2w_kf"], dtype=np.float64)
            k_node = self.kf["node"]
            T_kb, T_km = inverse(c2w_k) @ c2w_c, inverse(c2w_k) @ c2w_m
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
        est = self.scale_filter
        t0 = perf_counter()
        info = G.solve(need_std=not est.initialized)
        G.marginalize()
        if G.cfg.time_offset and abs(G.td - self.time_offset) > 0.004:
            # the samples of the window again at the graph's offset (its first-order correction is good to a few ms)
            self.time_offset = float(G.td)
            G.repreintegrate(self._samples, self.time_offset)
            self.stats["offset_updates"] = self.stats.get("offset_updates", 0) + 1
        self.stats["t_graph"] = self.stats.get("t_graph", 0.0) + perf_counter() - t0
        if np.isfinite(info.get("lam_std", float("nan"))):
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
            rep = T_j
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
                          "b_index": int(index)}
        self._set_m(index, timestamp, rgb, None, conf_curr, node=j, pass_depth=depth_curr)
        self.m["rep"] = rep
        if self.klt:
            self._klt_detect()

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
        for k in ("c2w_curr", "c2w_prev", "c2w_kf"):
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
        self.m = {"index": index, "timestamp": timestamp, "rgb": rgb, "R_wc": self.R_wc.copy(), "R_out": self.R_out.copy(),
                  "p_cam": self.p_cam.copy(), "depth": depth_chain, "conf": conf, "unit_ratio": unit_ratio,
                  "node": node, "pass_depth": pass_depth}
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
