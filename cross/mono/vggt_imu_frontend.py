"""Visual-inertial odometry from the feed-forward geometry model (VGGT-Omega) and the IMU, without DPVO.

Every frame, the IMU carries the pose: the gyroscope (bias-corrected) the rotation, and the velocity, gravity and
accelerometer bias of the local pose graph (cross.imu.vgi_graph) the position.  Every `interval` frames a visual
measurement corrects it: VGGT-Omega's relative pose between the last measured frame m and the current frame b.  The
measurement comes from the back end's forward pass when the back end observed this frame with m as its temporal anchor
(Pipeline: backend_anchor / after_backend), and otherwise from a forward pass of the same model on [b, m, keyframe].

VGGT-Omega's translations are in a gauge of its own for every forward pass.  The passes are chained through the frames
they share: m's depth in this pass against its depth in the pass where it was the current frame gives the ratio of the
two gauges (a link of the graph), and the graph recovers the metres per unit, with learned metric depth (Depth
Anything 3) as a prior on it.

Before the scale is known the frontend reports invalid motion, or with a continuous start unknown motion (the gyro's
rotation, wide translation covariance).

With a stereo rig (T_right_in_left; the stereo mode) the same graph, without learned depth and its bias, gets two stereo
constraints: the current pair observes each pass's metric scale (VgiGraph.add_stereo; vgio_stereo_source "depth":
classical stereo matching (SGBM) against the pass's depth of the current frame, or "baseline": the right image as one
more view of the pass, against the calibrated baseline), and the corners tracked between measured frames, lifted to 3-D
with the stereo depth, give their metric motion (PnP; VgiGraph.add_metric_relative).  Visual translations are tested
against the IMU's prediction, the IMU arbitrating between the passes and the corners (_stereo_gate).

The GPU work lives in a service (cross.mono.vgio_service.VgioPassService): the passes, and what the graph needs from
their depth maps (gauge links, metric scale, the stereo depth at the corners), summarized to a few numbers.  A
measurement is a request made when its frame is captured (the corners are detected on that frame then, the next
request takes it as m) and closed when its summary arrives: in the same frame with a local service, possibly frames
later on a remote GPU server (cross.remote).  A late summary adds its node to the graph at its frame's time, and the
IMU carries the state from there to the last tracked frame again (_replay); the correction enters the next reported
motion, as an in-frame measurement's does."""

import json
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from cross.imu import preintegrate
from cross.imu.vgi_graph import GraphConfig, VgiGraph

from .frontend import MonoEstimate
from .geometry import inverse, scale_translation_covariance

_NO_TRACKS = (None, None, None, [], None)        # (corners on m, their positions now, ids, steps, origin frame)


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
                 graph: bool = True, T_right_in_left=None, local_service: bool = True):
        if not graph:
            raise ValueError("The VGGT-IMU frontend's inertial-filter variant (vgio_graph=false) was removed "
                             "(2026-10-06, remote split); the local pose graph is the frontend")
        self.config = config
        self.K = np.asarray(K, dtype=np.float64).copy()
        self.device = device
        # stereo rig: the right camera's pose in the left camera (metres), or None (monocular)
        self.T_rl = None if T_right_in_left is None else np.asarray(T_right_in_left, dtype=np.float64).copy()
        self._right = None                       # right image of the current frame (stereo)
        # the GPU side (passes, depth maps); None on a remote session's edge, where a server holds it (cross.remote)
        self.service = None
        if local_service:
            from .vgio_service import VgioPassService
            self.service = VgioPassService(K, config, backend=backend, rgb_transform=rgb_transform,
                                           depth_transform=depth_transform, metric_model=metric_model,
                                           T_right_in_left=T_right_in_left, depth_every=depth_every)
        self.interval = int(interval)            # frames between visual measurements
        # measured frames in a pass of its own besides the current one: 2 adds the frame measured before the last one
        # (the gauges are chained through both, and the model sees more of the scene)
        self.context = int(context)
        self.m_prev = None
        # keyframe-pinned gauge (keyframe_age > 0 s): passes of its own also contain a keyframe, an older measured
        # frame whose depth in chain units stays fixed, and the gauge is chained through it, so the chain only drifts
        # when the keyframe is replaced (after keyframe_age s) instead of at every measurement
        self.keyframe_age = float(keyframe_age)
        self.kf = None
        # reported orientation (visual_rotation): the gyro-propagated orientation R_out instead of the graph's (the
        # estimator stays in its own frame R_wc; translations are reported in the camera frame)
        self.visual_rotation = bool(visual_rotation)
        self.rotation_gate_deg = float(rotation_gate_deg)
        self.R_out = np.eye(3)
        self._reported_R_out = None
        # local pose graph (cross.imu.vgi_graph): every visual measurement is a node; all pairs of a pass, the gauge
        # links through shared frames, learned depth (with its bias), the IMU and its biases are optimized together
        self.use_graph = True
        self.graph = None
        self.g_rep = None                        # gravity in the reported frame
        self.rotation_gate_deg_graph = 2.0       # deg: a pass whose rotation disagrees with the gyro's is not used
        # adaptive measurement times: (min frames, max frames, min translation m, min rotation deg), or None
        self.adaptive = None
        # measurements aligned with the back end's observations (pipeline): (min frames, frames before an own pass)
        self.align = None
        # optional tracked features (vgio_klt): corners detected on each measured frame and tracked frame to frame
        # (pyramidal Lucas-Kanade, forward-backward check); at the next measurement their essential matrix with the
        # calibrated intrinsics gives the rotation between the two frames, a factor when it has enough inliers and
        # agrees with the gyro.  Without texture there is no factor and the graph is unchanged
        self.klt = False
        self.klt_gray = None
        self.klt_m = None                        # corners on the frame they were detected on (klt_origin)
        self.klt_cur = None                      # their tracked positions in the current frame
        self.klt_ids = None                      # their indices among the detected corners
        self.klt_steps = []
        self.klt_origin = None
        self.standalone = True                   # no back end: every measurement is a forward pass of its own
        self.continuous_start = False
        self.imu_calib = None
        self.scale_filter = None
        self.imu_buffer = np.zeros((0, 7))
        self.accel_history = []
        self.gyro_bias = np.zeros(3)
        self.time_offset = 0.0
        self.time_offset_done = False
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
        # the last measured frame m (a node of the graph): index, timestamp, pose, node, stereo depth at its corners
        self.m = None
        # requests: the current frame's (handed to the service or sent to the server), the open ones by token, and the
        # chain they predict (the frame the next request measures from, its keyframe, the frame before it)
        self.request = None
        self._pending = {}
        self._pred = {"m": None, "m_prev": None, "kf": None}
        self._hist = {}                          # frame index -> (R_wc, p_cam, R_out) after tracking (or replay)
        self._log = []                           # (index, timestamp) of the frames since the oldest open request
        self._frame = None                       # (index, timestamp, rgb) of the current frame
        self._reported_internal = None           # internal pose of the last reported frame
        self._token = 0
        self.stats = dict(measurements=0, backend_measurements=0, own_calls=0, slips=0, chain_breaks=0, replays=0)
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

    def _start(self, calib):
        self.imu_calib = calib
        self.time_offset = float(self.config.imu.vgio_time_offset)
        c = self.imu_calib
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
                           trans_sigma_bound=ic.vgio_trans_sigma_bound, debug_costs=ic.vgio_debug_costs,
                           depth_bias_drift=ic.vgio_depth_bias_drift,
                           gyro_bias_walk=c.gyro_random_walk if ic.vgio_calib_gyro_walk else ic.vgio_gyro_bias_walk)
        self.graph = VgiGraph(gcfg, c.T_cam_imu, c.gyro_noise_density, c.accel_noise_density)
        self.graph.td = self.graph.td_init = self.time_offset
        if ic.vgio_graph_time_offset:
            self.time_offset_done = True         # the graph estimates the offset
        self.scale_filter = _GraphEstimate(self.graph)

    def _propagate(self, t0, t1):
        """The IMU from camera time t0 to t1: rotation from the gyro, position and velocity from the graph's velocity,
        gravity and accelerometer bias once it has started.  Returns the gyro's rotation (camera frame) or None."""
        pre = self._preintegrate(t0, t1)
        if pre is None:
            return None
        f = self.scale_filter
        c = self.imu_calib
        R_cb, t_cb = c.T_cam_imu[:3, :3], c.T_cam_imu[:3, 3]
        gyro_rotation = _ortho(R_cb @ pre.dR @ R_cb.T)
        R_wb = self.R_wc @ R_cb
        if f.started:
            # the IMU state carried by the estimator's velocity, gravity and accelerometer bias
            p_imu = self.p_cam + self.R_wc @ t_cb
            dp = pre.dp - pre.J_p @ f.b
            dv = pre.dv - pre.J_v @ f.b
            g = self.g_rep if self.g_rep is not None else f.g
            p_imu = p_imu + self.v_imu * pre.dt + 0.5 * g * pre.dt ** 2 + R_wb @ dp
            self.v_imu = self.v_imu + g * pre.dt + R_wb @ dv
            self.R_wc = _ortho(self.R_wc @ gyro_rotation)
            self.p_cam = p_imu - self.R_wc @ t_cb
        else:
            self.R_wc = _ortho(self.R_wc @ gyro_rotation)
        self.R_out = _ortho(self.R_out @ gyro_rotation)
        return gyro_rotation

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
            self._start(frame["imu_calib"])
        ic = self.config.imu
        if ic.vgio_align and self.align is None and not self.standalone:
            self.align = (ic.vgio_align_min, ic.vgio_align_max)
        self.klt = bool(ic.vgio_klt)
        if self.klt:
            t0 = perf_counter()
            self._klt_track(rgb)
            self.stats["t_klt"] = self.stats.get("t_klt", 0.0) + perf_counter() - t0
        if ic.vgio_adaptive and self.adaptive is None:
            self.adaptive = (ic.vgio_min_interval, ic.vgio_max_interval, ic.vgio_min_translation, ic.vgio_min_rotation_deg)
        self._add_imu(frame)
        f = self.scale_filter
        # the internal pose reported for the last frame: a visual measurement since then moved the internal state, and
        # the motion reported now includes that correction
        internal_prev = self._reported_internal if self._reported_internal is not None else self._internal_pose()
        gyro_rotation = self._propagate(self.last_timestamp, timestamp) if self.last_timestamp is not None else None
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
        self._hist[self.index] = (self.R_wc.copy(), self.p_cam.copy(), self.R_out.copy())
        self._log.append((self.index, timestamp))
        # a measurement due now: requested at capture (aligned with the back end, the measurement may also come with
        # its anchor, backend_anchor)
        self.request = None
        pm = self._pred["m"]
        due = self.index - pm[0] >= self.align[1] if (self.align is not None and pm is not None) else self._due(self.index)
        if due:
            self.request = self._new_request(rgb, self._right, timestamp, conditional=False)
        if self.standalone:
            self.after_backend(None)
        diagnostics["frontend_seconds"] = perf_counter() - start
        self.stats["t_track"] = self.stats.get("t_track", 0.0) + diagnostics["frontend_seconds"]
        self.stats["frames"] = self.stats.get("frames", 0) + 1
        self.last_timestamp = timestamp
        self.index += 1
        self._prune()
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

    def _hist_pose(self, index):
        R, p, R_out = self._hist[index]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return T, R_out

    # ------------------------------------------------------------------ requests
    def _due(self, index):
        """A visual measurement is due after `interval` frames, or (adaptive) once the camera moved min_translation m
        or turned min_rotation deg since the last one, between min_interval and max_interval frames."""
        pm = self._pred["m"]
        if pm is None:
            return True
        k = index - pm[0]
        ad = self.adaptive
        if ad is None or not self.scale_filter.initialized:
            return k >= self.interval
        if k < ad[0]:
            return False
        if k >= ad[1]:
            return True
        T_m, _ = self._hist_pose(pm[0])
        moved = float(np.linalg.norm(self.p_cam - T_m[:3, 3]))
        turned = _angle_deg(T_m[:3, :3].T @ self.R_wc)
        return moved >= ad[2] or turned >= ad[3]

    def _new_request(self, rgb, right, timestamp, conditional=False):
        """A visual measurement of the current frame b from the frame the chain will have measured last (m): the request
        for the service (frame indices, the motion since m, the corners detected on b now) and its snapshot for the
        close.  A request that cannot add a node (no IMU since m; the first one without accelerometer samples) is still
        made (the service keeps its count of learned-depth observations), but it leaves the chain and the corners as
        they are.  A conditional request (aligned with the back end: measured only if the back end's pass carries it)
        detects its corners when it is closed, in the same frame."""
        b = self.index
        pm, pkf, pmp = self._pred["m"], self._pred["kf"], self._pred["m_prev"]
        doomed = None
        if pm is None:
            if not self.accel_history:
                doomed = "accel"
        elif self._samples(pm[1], timestamp) is None:
            doomed = "no imu"
        self._token += 1
        kf = pkf[0] if (pm is not None and self.keyframe_age > 0 and pkf is not None and pkf[0] != pm[0]) else None
        m_prev = pmp if (pm is not None and kf is None and self.context >= 2 and pmp is not None) else None
        T = None
        if pm is not None:
            T_m, R_out_m = self._hist_pose(pm[0])
            T = self._relative(T_m, R_out_m, self._internal_pose(), self.R_out)
        req = {"token": self._token, "index": b, "timestamp": timestamp, "m": None if pm is None else pm[0], "kf": kf,
               "m_prev": m_prev, "T_prev_curr": T, "metric": bool(self.output_valid), "anchor": False,
               "own": not conditional, "corners": None, "keep_from": None}
        snap = {"req": req, "index": b, "timestamp": timestamp, "R_wc": self.R_wc.copy(), "p_cam": self.p_cam.copy(),
                "v_imu": self.v_imu.copy(), "accel": list(self.accel_history),
                "tracks": (self.klt_m, self.klt_cur, self.klt_ids, list(self.klt_steps), self.klt_origin),
                "doomed": doomed, "conditional": bool(conditional), "advanced": False, "pred_before": dict(self._pred)}
        if doomed is None and not conditional:
            if self.klt:
                self._klt_detect()
                req["corners"] = None if self.klt_m is None else self.klt_m[:, 0].copy()
            self._advance_pred(b, timestamp)
            snap["advanced"] = True
        self._pending[self._token] = snap
        req["keep_from"] = self._keep_from()
        if self.service is not None:
            self.service.prune(req["keep_from"])
            self.service.add_frame(b, rgb, right)
        return req

    def _advance_pred(self, b, timestamp):
        pm, pkf = self._pred["m"], self._pred["kf"]
        kf = (b, timestamp) if self.keyframe_age > 0 and (pkf is None or timestamp - pkf[1] >= self.keyframe_age) else pkf
        self._pred = {"m": (b, timestamp), "m_prev": None if pm is None else pm[0], "kf": kf}

    def _keep_from(self):
        """The oldest frame an open or a later request can still reference."""
        idx = [s["index"] for s in self._pending.values()]
        idx += [x[0] for x in (self._pred["m"], self._pred["kf"]) if x is not None]
        idx += [x["index"] for x in (self.m, self.kf, self.m_prev) if x is not None]
        if self._pred["m_prev"] is not None:
            idx.append(self._pred["m_prev"])
        return min(idx) if idx else self.index

    def _prune(self):
        keep = self._keep_from()
        for k in [k for k in self._hist if k < keep]:
            del self._hist[k]
        oldest = min([s["index"] for s in self._pending.values()], default=self.index - 1)
        self._log = [e for e in self._log if e[0] >= min(oldest, self.index - 1)]

    # ------------------------------------------------------------------ back-end coupling (a local service)
    def backend_anchor(self, observing=True):
        """The temporal anchor for the back end's forward pass at this frame: the last measured frame with the
        frontend's relative pose to it (metric once the scale is known).  The pipeline asks on the frames it hands to
        the back end; the back end may still skip the pass (its observation cadence)."""
        if not observing or self._frame is None:
            return None
        req = self.request
        if req is None and self.align is not None and self._pred["m"] is not None \
                and self._frame[0] - self._pred["m"][0] >= self.align[0]:
            req = self.request = self._new_request(self._frame[2], self._right, self._frame[1], conditional=True)
        if req is None or req["m"] is None:
            return None
        req["anchor"] = True
        return self.service.anchor(req) if self.service is not None else req

    def after_backend(self, observed):
        """After the back end's step: its forward pass with our anchor (cross.cv.pose_est_ff last_frontend_obs), or
        None.  The request of this frame is measured on it, or with a forward pass of the service's own."""
        req, self.request = self.request, None
        if req is None or self.service is None:
            return
        t0 = perf_counter()
        self.close(req["token"], self.service.measure(req, observed))
        self.stats["t_measure"] = self.stats.get("t_measure", 0.0) + perf_counter() - t0

    def close(self, token, summary):
        """The service's summary of a request (cross.mono.vgio_service): its frame becomes a node of the graph.  A
        summary that comes frames after its request moves the state of its frame, and the IMU carries it to the last
        tracked frame again."""
        snap = self._pending.pop(token, None)
        if snap is None:
            return
        if summary is None:                      # nothing measured (a conditional request whose pass did not come)
            return
        self.stats["backend_measurements" if summary.get("source") == "backend" else "own_calls"] += 1
        frame, self._frame = self._frame, (snap["index"], snap["timestamp"], None)
        try:
            ok = not snap.get("superseded") and self._measure_graph(snap, summary)
        finally:
            self._frame = frame
        if self.config.imu.vgio_trace:
            with open(self.config.imu.vgio_trace, "a") as fh:
                fh.write(json.dumps({"index": snap["index"], "t": snap["timestamp"], **self.last_info},
                                    default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        if not ok:
            self._close_failed(snap)
            return
        if snap["conditional"]:
            # measured with the back end's pass in the same frame: the corners of the new m, as an in-frame measurement's
            if self.klt:
                self._klt_detect()
                corners = None if self.klt_m is None else self.klt_m[:, 0].copy()
                self.m["corner_z"] = self.service.corner_depth(snap["index"], corners) if self.service else None
            self._advance_pred(snap["index"], snap["timestamp"])
        if self._log and self._log[-1][0] > snap["index"]:
            self._replay(snap["index"], snap["timestamp"])

    def _close_failed(self, snap):
        """A request that added no node although its corners were detected and the chain went on from it: in the same
        frame everything is as before the request; later, the open requests made on it are dropped and the next ones
        measure from the last measured frame (its corners no longer match: no corner factors until the next node)."""
        if not snap["advanced"] or snap.get("superseded"):
            return
        self.stats["chain_breaks"] += 1
        if self._log and self._log[-1][0] == snap["index"]:
            self.klt_m, self.klt_cur, self.klt_ids, steps, self.klt_origin = snap["tracks"]
            self.klt_steps = list(steps)
            self._pred = dict(snap["pred_before"])
            return
        for other in self._pending.values():
            other["superseded"] = True
        m, kf, mp = self.m, self.kf, self.m_prev
        self._pred = {"m": None if m is None else (m["index"], m["timestamp"]),
                      "m_prev": None if mp is None else mp["index"],
                      "kf": None if kf is None else (kf["index"], kf["timestamp"])}

    def _replay(self, b, t_b):
        """The IMU from the newly measured frame b (time t_b) to the last tracked frame again (those frames were
        propagated from the state before the measurement)."""
        self.stats["replays"] += 1
        self.R_out = self._hist[b][2].copy() if b in self._hist else self.R_out
        t_prev = t_b
        for idx, t in self._log:
            if idx > b:
                self._propagate(t_prev, t)
                self._hist[idx] = (self.R_wc.copy(), self.p_cam.copy(), self.R_out.copy())
                t_prev = t

    # ------------------------------------------------------------------ tracked corners
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
            self.klt_cur, self.klt_m, self.klt_ids = p1[ok], self.klt_m[ok], self.klt_ids[ok]
        self.klt_gray = gray

    def _klt_detect(self):
        pts = cv2.goodFeaturesToTrack(self.klt_gray, maxCorners=300, qualityLevel=0.01, minDistance=8)
        self.klt_steps = []
        self.klt_m = None if pts is None else pts.astype(np.float32)
        self.klt_cur = None if pts is None else pts.astype(np.float32).copy()
        self.klt_ids = None if pts is None else np.arange(len(pts))
        self.klt_origin = self._frame[0] if self._frame is not None else None

    def _klt_rotation(self, gyro_mb, tracks):
        """(R of the current camera in the last measured frame's camera, std) from the tracked corners, or None."""
        klt_m, klt_cur = tracks[0], tracks[1]
        if klt_m is None or klt_cur is None or len(klt_cur) < 30:
            return None
        a, b = klt_m[:, 0].astype(np.float64), klt_cur[:, 0].astype(np.float64)
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

    def _rest_rate(self, samples, pre, a0, a1, tracks):
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
        klt_m, klt_cur, steps = tracks[0], tracks[1], tracks[3]
        if klt_m is None or klt_cur is None or len(klt_cur) < 30 or not steps:
            return None
        fb = float(np.median([s[0] for s in steps]))
        if min(s[1] for s in steps) < 1e-3 or fb <= 0.0:          # a repeated frame: no evidence of rest
            return None
        disp = float(np.median(np.linalg.norm((klt_cur - klt_m)[:, 0], axis=1)))
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

    def _stereo_pnp(self, gyro_mb, gyro_sigma_deg, tracks, corner_z):
        """The metric pose of the current frame b in the last measured frame m from the corners tracked between them
        (vgio_klt; tracks: their positions on m and on b, their ids among the corners detected on m) and the stereo
        depth of m at those corners (corner_z, by id): (R_mb, t_mb, 6x6 covariance, info), or (None, info).  The
        covariance comes from the corners themselves: the pixel noise is the RMS reprojection error of the inliers,
        each corner's depth error the disparity noise (the same pixel noise) through z^2 / (f b), both propagated
        through the projection.  A metric motion independent of the passes and of the IMU; used when it has enough
        inliers and its rotation agrees with the gyro as the corners' rotation-only factor must (it replaces that
        factor)."""
        klt_m, klt_cur, ids = tracks[0], tracks[1], tracks[2]
        if corner_z is None or klt_m is None or klt_cur is None or len(klt_cur) < 12:
            return None, {"ok": False, "reason": "no tracks or depth"}
        a = klt_m[:, 0].astype(np.float64)
        b = klt_cur[:, 0].astype(np.float64)
        z = np.asarray(corner_z, dtype=np.float64)[ids]
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
        if diff > max(1.0, 3.0 * gyro_sigma_deg):              # the corners' rotation-only factor's test (1 deg)
            self.stats["pnp_rejected"] = self.stats.get("pnp_rejected", 0) + 1
            return None, info | {"ok": False, "reason": "gyro"}
        if self.config.imu.vgio_stereo_pnp_rotation == "calibrated":
            # the rotation's noise self-calibrated like the corners' rotation-only factor: the median disagreement with
            # the gyro over the last 100 accepted ones (median of a 3-D error norm: 1.54 sigma per axis) as a floor of
            # the analytic covariance, which knows the pixel and depth noise but not their systematic errors
            self.pnp_diffs = (getattr(self, "pnp_diffs", []) + [diff])[-100:]
            if len(self.pnp_diffs) >= 10:
                s_r = np.radians(max(0.05, float(np.median(self.pnp_diffs)) / 1.54))
                w, V = np.linalg.eigh(cov[0:3, 0:3])
                extra = V @ np.diag(np.maximum(s_r ** 2 - w, 0.0)) @ V.T
                cov = cov.copy()
                cov[0:3, 0:3] += extra
                info["rot_floor_deg"] = round(float(np.degrees(s_r)), 3)
        self.stats["pnp_used"] = self.stats.get("pnp_used", 0) + 1
        return (R_mb, t_mb, cov), info

    # ------------------------------------------------------------------ the visual measurement
    def _measure_graph(self, snap, s):
        """A visual measurement (the service's summary s of the request snap) as a node of the local pose graph.
        Returns True when it added a node (its frame is the new last measured frame m)."""
        index, timestamp = snap["index"], snap["timestamp"]
        G = self.graph
        R_cb = self.imu_calib.T_cam_imu[:3, :3]
        if not s.get("finite"):
            self.last_info = {"measured": False}
            return False
        if snap["req"]["m"] != (None if self.m is None else self.m["index"]):
            self.last_info = {"measured": False, "reason": "chain"}
            return False
        da3, stereo, stereo_info = s.get("da3"), s.get("stereo"), s.get("stereo_info")
        if self.m is None or not G.ids:
            if not snap["accel"]:
                return False
            lam0 = stereo[0] if stereo is not None else da3[0] if da3 is not None else s["log_median_depth"]
            node = G.start(snap["R_wc"], np.mean(snap["accel"], axis=0), lam0, timestamp, bg0=self.gyro_bias)
            if da3 is not None:
                G.add_depth(node, *da3)
            if stereo is not None:
                G.add_stereo(node, *stereo)
            # the state at the measured frame (the frames since are propagated from it again, _replay)
            self.R_wc, self.p_cam, self.v_imu = snap["R_wc"].copy(), snap["p_cam"].copy(), snap["v_imu"].copy()
            self._set_m(index, timestamp, node=node, corner_z=s.get("corner_z"))
            T0 = np.eye(4)
            T0[:3, :3], T0[:3, 3] = G.R[node], G.p[node]
            self.m["rep"] = T0
            self.g_rep = G.g.copy()
            self.last_info = {"measured": True, "first": True}
            return True
        a0, a1 = self.m["timestamp"], timestamp
        samples = self._samples(a0, a1)
        if samples is None:
            self.last_info = {"measured": False, "reason": "no imu"}
            return False
        c2w_c, c2w_m = s["c2w_curr"], s["c2w_prev"]
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
        ratio, spread = s["link"]
        link_ok = bool(np.isfinite(ratio) and spread < 0.25)
        lam_init = stereo[0] if stereo is not None else G.lam[m_node] + (np.log(ratio) if link_ok else 0.0)
        kfc = self._keyframe_consistency(s, c2w_c, c2w_m, index, ratio if link_ok else None)
        j = G.add_node(pre, jac, lam_init, timestamp)
        # the corners tracked from m to this frame (snapshot of its capture), if they were detected on m
        tracks = snap["tracks"] if snap["tracks"][4] == self.m["index"] else _NO_TRACKS
        pnp, pnp_info = (None, None)
        if self.klt and self.T_rl is not None and self.config.imu.vgio_stereo_pnp:
            pnp, pnp_info = self._stereo_pnp(gyro_mb, gyro_sigma_deg, tracks, self.m.get("corner_z"))
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
        pnp_rot = self.config.imu.vgio_stereo_pnp_rotation not in (False, "false", "none", 0)
        if pnp is not None:
            R_p, t_p, cov_p = pnp
            if not pnp_rot:
                # the translation only (its marginal covariance): the corners' rotation stays the essential matrix's
                # factor below, whose noise is calibrated against the gyro
                cov_p = cov_p.copy()
                cov_p[0:3, :] = 0.0
                cov_p[:, 0:3] = 0.0
                cov_p[0:3, 0:3] = np.eye(3) * 1e4
            G.add_metric_relative(m_node, j, R_p, t_p, cov_p)
        klt = self._klt_rotation(gyro_mb, tracks) if self.klt and (pnp is None or not pnp_rot) else None
        if klt is not None and self.config.imu.vgio_noise_hat:
            klt = self._noise_hat(gyro_mb, T_mb[:3, :3] if pass_ok else None, klt, a1 - a0) or klt
        if klt is not None:
            G.add_rotation(m_node, j, *klt)
        rest = self._rest_rate(samples, pre, a0, a1, tracks) if self.klt and self.config.imu.vgio_zero_rate else None
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
        if pass_ok and s.get("c2w_kf") is not None and self.kf is not None and self.kf.get("node") in G.R \
                and snap["req"]["kf"] == self.kf["index"]:
            c2w_k = s["c2w_kf"]
            k_node = self.kf["node"]
            T_kb, T_km = inverse(c2w_k) @ c2w_c, inverse(c2w_k) @ c2w_m
            kf_log = {"kf_index": int(self.kf["index"]), "kf_t": float(np.linalg.norm(T_kb[:3, 3]))}
            # against the graph's current rotation from the keyframe (over ~2 s the gyro alone drifts)
            pred_km = G.R[k_node].T @ G.R[m_node]
            pairs_ok = True
            if self.T_rl is not None and gate is not None and self.config.imu.vgio_stereo_gate_pairs:
                # the keyframe pairs span ~2 s: their translations are tested against the graph as the pass's (one that
                # under-reports them, accepted on its short pair, pulled every velocity of the window)
                pairs_ok = (self._stereo_pair_ok(k_node, j, lam_init, T_kb[:3, 3])
                            and self._stereo_pair_ok(k_node, m_node, lam_init, T_km[:3, 3]))
                kf_log["pairs_ok"] = pairs_ok
                self.stats["kf_pairs_gated"] = self.stats.get("kf_pairs_gated", 0) + int(not pairs_ok)
            if pairs_ok and _angle_deg(pred_km.T @ T_km[:3, :3]) <= 2 * self.rotation_gate_deg_graph:
                G.add_relative(k_node, j, j, T_kb[:3, :3], T_kb[:3, 3])
                G.add_relative(k_node, m_node, j, T_km[:3, :3], T_km[:3, 3])
            r_k, s_k = s["link_kf"] if s.get("link_kf") is not None else (float("nan"), float("inf"))
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
        v_prop = float(np.linalg.norm(G.v[j]))
        info = G.solve(need_std=lam_std_wanted or self.config.imu.vgio_trans_sigma_bound or self.T_rl is not None)
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
                          "stereo": stereo_info, "pnp": pnp_info, "v_prop": round(v_prop, 3),
                          "costs": info.get("costs"), "source": s.get("source"), **(kf_log or {})}
        self._set_m(index, timestamp, node=j, corner_z=s.get("corner_z"))
        self.m["rep"] = rep
        return True

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
        # the IMU's velocity uncertainty: its drift since the last accepted translation, and the graph's own marginal
        # velocity std (last solve), large at a session start, where the IMU knows no speed yet (a relocalization
        # session must not lose its first passes to an unconverged velocity)
        v_std = float(G.v_std) if np.isfinite(G.v_std) else 1e3
        s_imu = float(np.hypot(0.05 + G.cfg.accel_bias_std * gap, v_std))
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

    def _stereo_pair_ok(self, a, b, lam, t_ab):
        """A longer pair of a pass (keyframe -> current, keyframe -> last measured) against the graph's motion between the
        two nodes: speeds within 4 sigma (the pass's relative noise and scale uncertainty; the IMU's velocity uncertainty
        over the pair's interval)."""
        G = self.graph
        dt = max(G.t[b] - G.t[a], 1e-3)
        v_pred = float(np.linalg.norm(G.R[a].T @ (G.p[b] - G.p[a]))) / dt
        v = float(np.exp(lam) * np.linalg.norm(t_ab)) / dt
        lam_std = self.scale_filter.lam_std if np.isfinite(self.scale_filter.lam_std) else 1.0
        v_std = float(G.v_std) if np.isfinite(G.v_std) else 1e3
        s = float(np.linalg.norm([np.sqrt(G.cfg.trans_rel ** 2 + min(lam_std, 1.0) ** 2) * v,
                                  0.05 + G.cfg.accel_bias_std * dt, v_std]))
        return bool(abs(v - v_pred) <= 4.0 * s)

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

    def _set_m(self, index, timestamp, node=None, corner_z=None):
        self.m_prev = self.m
        R_out = self._hist[index][2].copy() if index in self._hist else self.R_out.copy()
        self.m = {"index": index, "timestamp": timestamp, "R_wc": self.R_wc.copy(), "R_out": R_out,
                  "p_cam": self.p_cam.copy(), "node": node, "corner_z": corner_z}
        if self.keyframe_age > 0 and (self.kf is None or timestamp - self.kf["timestamp"] >= self.keyframe_age):
            self.kf = self.m                     # the new keyframe: its pass depth stays the gauge reference
        self._hist[index] = (self.R_wc.copy(), self.p_cam.copy(), R_out)

    def _calibrate_time_offset(self, a0, a1, dR_vis_imu):
        """As DPVOFrontend._calibrate_time_offset, from the measured intervals' rotation rates."""
        cfg = self.config.imu
        if self.time_offset_done or cfg.time_offset_after <= 0 or a1 <= a0:
            return
        self.rate_log.append((0.5 * (a0 + a1), _so3_log(dR_vis_imu) / (a1 - a0), a1 - a0))
        # a first calibration after 20 measurements (6 s), a final one after 100; the window's IMU factors are
        # preintegrated again with the new offset
        need = 20 if not self._offset_rounds else 100
        if len(self.rate_log) < need:
            return
        tm = np.array([r[0] for r in self.rate_log])
        rates = np.array([r[1] for r in self.rate_log])
        if float(np.sqrt((rates ** 2).sum(1).mean())) < 0.1:
            self.rate_log = self.rate_log[-300:]
            return                           # too little rotation to see an offset (a standing robot): wait
        buf = self.imu_buffer
        errs = []
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
        best = min(errs)
        at_zero = min(errs, key=lambda e: abs(e[1]))
        old = self.time_offset
        if best[0] < 0.7 * at_zero[0]:
            self.time_offset = round(best[1], 3)
        self._offset_rounds += 1
        self.time_offset_done = self._offset_rounds >= 2
        if self.graph is not None and self.time_offset != old:
            self.graph.repreintegrate(self._samples, self.time_offset)
        if not self.time_offset_done:
            return                                   # the rate log keeps growing for the final round
        self.rate_log = []

    def shutdown(self):
        from loguru import logger
        logger.info(f"VGGT-IMU frontend: {self.stats}" + (f", service: {self.service.stats}" if self.service else ""))


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
