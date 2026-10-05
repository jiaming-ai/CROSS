"""Metric scale of monocular visual odometry from an IMU: a sliding-window MAP estimate over the visual trajectory.

Visual odometry (DPVO) gives camera poses up to an unknown, slowly drifting scale; its rotations are accurate.  With
the rotations taken as known (the visual rotation at each frame, the gyroscope between frames), the IMU constrains the
metric motion, and the visual displacements are that motion in visual-odometry units.  Unknowns over a window of the
last W frame intervals (metric units; W = the visual odometry's world frame):

    v_k   velocity of the IMU at frame k, in W                       (k = 0..W)
    l_k   log scale at frame k: metres per visual-odometry unit is exp(l_k) (random walk: the drift of the visual odometry)
    g     gravity in W, pointing down (|g| = 9.81)
    b     accelerometer bias (IMU frame)

Residuals (each in the space where its noise is additive):

    IMU      v_{k+1} - v_k - g dt - R_k (dv_k - J_v b)                         ~ IMU noise (metric)
    visual   z_k - exp(-l_k) (v_k dt + g dt^2/2 + R_k (dp_k - J_p b) - L_k)    ~ visual noise (units)
    drift    l_{k+1} - l_k                                                     ~ scale_drift^2 dt
    gravity  |g| - 9.81;  learned depth  l_k - log(scale observed)  (optional, weak);  marginalization prior

with z_k the visual displacement of the camera (units), L_k = (R_WC,k+1 - R_WC,k) t_cb the lever arm of the IMU, and
(dv_k, dp_k, J_v, J_p) the preintegrated IMU (cross.imu.preintegration).  Gauss-Newton with Huber weights on the visual
residuals, warm-started every frame; the oldest frame is marginalized into a Gaussian prior (Schur complement) when the
window slides, so the velocity keeps the metric scale across stretches of constant velocity.

(Two Kalman filters were tried first: with the visual displacement as a regressor of the scale, its noise attenuated the
scale; with the IMU increment as a regressor of the inverse scale, the IMU noise inflated it, by 20-30 % on simulated
robot drives.  Both noises as residuals of a joint MAP estimate have neither bias.)

Start-up: until the scale is known, the window is solved for a grid of constant log scales (the problem is linear in
the other unknowns for a fixed scale) and Gauss-Newton starts from the best one, so a wrong initial guess cannot trap
it.  The scale counts as known once its marginal log std is below init_log_std.  A learned-depth scale observation (log
scale, variance) enters as a weak measurement of l at the newest frame.

The class exposes the interface of cross.mono.scale.LogScaleFilter (scale, initialized, uncertainty_variance,
update(observation)), so it replaces that filter in the DPVO frontend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preintegration import Preintegrated

GRAVITY = 9.81


@dataclass
class ImuConfig:
    enabled: bool = False
    # learned-depth scale observations (DA3 metric depth vs DPVO depth, every metric_interval frames) also enter as
    # measurements of the scale, each with at least depth_prior_std_floor.  Their errors are partly one domain bias (DA3
    # is 1.9x off on ROVER), but treating them as one prior per window (depth_prior_independent false, floor 0.3) let
    # the IMU dominate where its scale is weakly observed (slow indoor robots: OpenLORIS home 0.6x, office 0.93x), so
    # each observation counts (front end on six development sequences: scale within 2-3 % on office, cafe, KITTI 07
    # and SimChange; ROVER 1.37x vs 1.86x for learned depth alone; home 0.80x vs 1.04x)
    depth_prior: bool = True
    depth_prior_std_floor: float = 0.15
    depth_prior_independent: bool = True
    window: int = 100                            # frame intervals in the window (10 s at 10 Hz)
    scale_drift: float = 0.02                    # log-scale drift of the visual odometry per sqrt(second)
    # gravity in the (gyro-propagated) world frame moves with the residual gyro bias (m/s^2 per sqrt(second)); added to
    # the marginalization prior as the window slides, as is the accelerometer bias walk
    gravity_drift: float = 0.02
    accel_bias_walk: float = 0.005               # m/s^3/sqrt(Hz), larger than the datasheet for unmodelled errors
    accel_bias_std: float = 0.1                  # prior std of the accelerometer bias (m/s^2)
    gravity_std: float = 0.3                     # prior std of the initial gravity estimate (m/s^2, per axis)
    gravity_norm_std: float = 0.05               # |g| = 9.81 (m/s^2)
    # m/s^2/sqrt(Hz) added to the IMU noise: gravity leakage through visual rotation jitter (~0.2 deg).  0.1 helped
    # OpenLORIS home1-1 (front end 0.70 -> 0.45 m) but on the development split it made 4 of 5 maps and T3 worse
    extra_accel_noise: float = 0.01
    # per-axis noise of a frame-to-frame visual displacement, in visual-odometry units: a part proportional to the
    # scene depth (in units) and one proportional to the displacement
    visual_std_depth: float = 0.0005
    visual_std_rel: float = 0.01
    huber: float = 3.0                           # visual residuals beyond this many sigmas get Huber weights
    lag_frames: int = 3                          # measure the visual displacement this many frames late (refined poses)
    gyro_check_deg: float = 0.5                  # skip a visual displacement whose rotation differs from the gyro's by more
    gyro_bias_window: int = 600                  # intervals of the running median of the gyro bias (vs DPVO rotations)
    time_offset_after: int = 150                 # calibrate the camera-IMU time offset after this many intervals (0: off)
    time_offset_max: float = 0.15                # s, searched range
    imu_buffer_samples: int = 20000              # IMU samples kept by the frontend (> 60 s at 300 Hz)
    # continuous start (mono feed-forward back end): per-frame std of the unknown motion before DPVO's metric
    # trajectory is valid: translation (m), rotation without / with the gyroscope (rad)
    unknown_motion_std: tuple = (0.15, 0.1, 0.01)
    reset_nis: float = 50.0                      # restart the estimate when the mean visual NIS of the last
    reset_frames: int = 10                       # reset_frames intervals exceeds this (the visual and IMU motion disagree)
    init_log_std: float = 0.35                   # the scale counts as known once its log std is below this
    scale_prior_log_std: float = 3.0             # start-up grid: +-this around the learned-depth (or unit) scale
    grid_step: float = 0.2
    iterations: int = 3                          # Gauss-Newton iterations per frame
    # the scale stays within this factor of the latest learned-depth scale (or, without one, of the first estimate):
    # when visual odometry stops reporting translation while the IMU says the platform moves, the best fit is an
    # unbounded scale, which later multiplies DPVO's motion into astronomic distances (OpenLORIS home1-3).  Learned
    # depth was at most 1.9x off on the benchmark (ROVER).
    scale_band: float = 2.5
    # learned-depth bias as a state (log scale; prior std, drift per sqrt(s)): learned depth then measures l + bias, so
    # its observations follow the changes of the visual scale while the IMU sets the level (off: they measure l)
    depth_bias_state: bool = False
    depth_bias_std: float = 0.4
    depth_bias_drift: float = 0.005
    # vgio (VGGT-Omega + IMU, cross/mono/vggt_imu_frontend): frames between visual measurements, frames between
    # learned-depth observations (0: the DPVO frontend's metric interval), and the visual noise of a VGGT-Omega
    # displacement (in place of visual_std_rel / _depth)
    visual_interval: int = 3
    depth_every: int = 3
    vgio_visual_std_rel: float = 0.03
    vgio_visual_std_depth: float = 0.003
    vgio_scale_drift: float = 0.04               # chained VGGT-Omega gauges drift faster than DPVO's
    vgio_context: int = 2                        # measured frames besides the current one in a pass of the frontend
    vgio_keyframe_age: float = 2.0               # s; > 0: passes also contain a keyframe this old (graph: its pairs)
    vgio_visual_rotation: bool = False           # report VGGT-Omega's orientation (from the keyframe), not the gyro's
    vgio_graph: bool = True                      # local pose graph (cross.imu.vgi_graph) instead of this filter
    vgio_graph_window: int = 20                  # nodes
    vgio_depth_bias: bool = True                 # the graph estimates the learned-depth bias
    vgio_rot_std: float = 0.0087                 # rad, relative rotation of a pass in the graph (0.5 deg)
    vgio_depth_bias_std: float = 0.05            # prior std of the learned-depth log bias in the graph
    vgio_depth_bias_drift: float = 0.005         # its random walk (log per sqrt(s))
    vgio_rot_rel: float = 0.05                   # graph: rotation noise also grows with the angle (fraction)
    # adaptive measurement times (vgio_adaptive): after the camera moved / turned this much, within these frame counts
    vgio_rot_scale: bool = True                  # graph: calibrate the rotation scale of the passes against the gyro
    vgio_rot_scale_std: float = 0.05             # its prior std (log)
    vgio_online_noise: bool = False              # graph: the passes' rotation noise from their disagreement with the gyro
    vgio_time_offset: float = 0.0                # s, initial camera - IMU time offset (calibrated online from it)
    vgio_graph_time_offset: bool = False         # graph: the time offset as a variable of the graph (no separate
                                                 # calibration)
    vgio_time_offset_std: float = 0.05           # its prior std (s)
    vgio_pp_correction: bool = True              # the passes' camera poses corrected for the principal point offset
    vgio_noise_hat: bool = False                 # rotation noise of gyro, corners and passes from their disagreements
    vgio_calib_gyro_walk: bool = False           # graph: the gyro bias random walk of the sensor's calibration (else 1e-4)
    vgio_klt: bool = True                        # graph: rotation factors from tracked corners (optional, gated)
    # pipeline: measure on the back end's forward passes (at least vgio_align_min frames apart); a pass of the
    # frontend's own only after vgio_align_max frames without one.  Off: every visual_interval frames, on the back
    # end's pass when it observes at that frame, else a pass of its own (3 views).  Off is better: the back end's
    # passes carry map references, and measured on them alone the indoor scale came out 6 % short (development maps,
    # 5090: office 0.11 -> 0.06 m, KITTI 07 2.76 -> 1.79, home 0.46 -> 0.36, cafe 0.24 -> 0.17; time +0-14 %)
    vgio_align: bool = False
    vgio_align_min: int = 2
    vgio_align_max: int = 4
    # graph: a pass's translation must agree with the IMU's prediction in length within this factor or in velocity
    # within 4 sigma of the combined uncertainty (cross.mono.vggt_imu_frontend._translation_gate), else the pass is not
    # used (like a pass whose rotation disagrees with the gyro).  0 or 1: no test.  KITTI 01 without it: VGGT-Omega
    # reported 10-20 % of the motion for ~6 s on the highway, the graph followed it to 1 m/s at a true 25.7 m/s, and the
    # IMU (which only measures changes of velocity) kept the wrong speed: map scale 0.3, ATE 411 m
    vgio_trans_gate: float = 1.5
    vgio_trans_gate_max_gap: float = 15.0        # s: the longest the IMU may outvote the passes' translations
    # graph: random walk of the gyro bias (rad/s per sqrt(s)) when vgio_calib_gyro_walk is off.  The window's visual
    # rotations carry small systematic errors that a loose walk lets the bias follow: on ROVER the gyro alone, with the
    # bias of the initial standstill, turns 1079.9 deg for a true 1080.2 deg, while the graph's bias wandered 0.05-0.07
    # deg/s off with 1e-4 (heading drift 26 deg over 600 s, the integrated bias error 21 deg).  1e-5: drift 18 deg, front
    # end ATE 0.98x in geometric mean over 13 sequences (ROVER 0.91-0.96x, the rest within 3 %); 3e-6 changes nothing more
    vgio_gyro_bias_walk: float = 1e-5
    # graph: relative translation noise from the larger of the measured and predicted translation, Huber gauge links
    # (cross.imu.vgi_graph.GraphConfig.trans_sigma_predicted / robust_links).  ROVER night relocalization (T3 against
    # one map): the predicted-translation noise costs trials (with it 83, without 105 of 140, gate off), the Huber links
    # nothing; on KITTI 01 the predicted-translation noise and the gate are what stop the collapse
    # graph: the gyro bias measured directly while the images show the platform at rest (zero-rate update); the
    # visual rotations alone pin it to ~0.05 deg/s, and while driving they carry a motion-coupled error of that size
    # (ROVER: the bias learned at the start drifted 0.2 -> 0.15-0.29 deg/s, heading 15-30 deg)
    vgio_zero_rate: bool = True
    vgio_trans_sigma_predicted: bool = True
    vgio_robust_links: bool = True
    # ... with the predicted translation lowered by twice its uncertainty (the graph's marginal velocity std): a
    # prediction the graph does not know yet (session start) cannot weaken the measurements (GraphConfig.trans_sigma_bound;
    # ROVER night relocalization vs the v4 map 76 -> 91 of 140, KITTI 01 kept)
    vgio_trans_sigma_bound: bool = True
    vgio_gyro_dt_noise: float = 0.0              # graph: preintegration noise growing with the IMU sampling interval
    vgio_accel_dt_noise: float = 0.0
    vgio_adaptive: bool = False
    vgio_min_interval: int = 2
    vgio_max_interval: int = 4
    vgio_min_translation: float = 0.5
    vgio_min_rotation_deg: float = 3.0
    # stereo + IMU (the stereo mode with --odometry vgio).  The stereo pair observes each pass's metric scale
    # (VggtImuFrontend._stereo_scale; no bias state).  Source "depth": classical stereo depth (SGBM; pixels with >=
    # vgio_stereo_min_disparity px) against the pass's depth of the current frame, std floored at vgio_stereo_std.
    # Source "baseline": the right image as a view of the pass, its left-right translation against the calibrated
    # baseline (std hypot(vgio_stereo_std, vgio_stereo_depth_k * depth / baseline); not used when the pass's rotation
    # between the cameras or its baseline direction disagrees with the calibration).  VGGT-Omega places the right camera
    # too far on KITTI (scale 13 % low on 07, 30 % on 01) while its depths agree with its translations: SGBM depth is
    # within -4..+4 % on KITTI, ROVER, SimChange and the T265 indoors.  "both": both logged, depth used
    vgio_stereo_source: str = "depth"
    vgio_stereo_min_disparity: float = 2.0
    vgio_stereo_std: float = 0.02
    vgio_stereo_depth_k: float = 0.001
    vgio_stereo_rot_gate_deg: float = 3.0
    vgio_stereo_dir_cos: float = 0.95
    # stereo + IMU: the corners tracked between measured frames (vgio_klt) lifted to 3-D with the stereo depth of the
    # earlier frame give the metric motion by PnP, a factor with its own covariance (VggtImuFrontend._stereo_pnp) in
    # place of their rotation-only factor.  Front end: OpenLORIS office 0.055 -> 0.026 m, cafe 0.36 -> 0.26 m, SimChange
    # 0.057 -> 0.024 m, KITTI 07 relative error 2.5 -> 1.4 %; it locks onto vehicles alongside on a highway (KITTI 01),
    # which the translation tests catch
    vgio_stereo_pnp: bool = True
    # ... with its rotation (False: its translation only, the corners' rotation from the essential matrix as without it)
    vgio_stereo_pnp_rotation: bool = False
    # stereo + IMU: the translation tests with the IMU as the arbiter (VggtImuFrontend._stereo_gate): the pass and the
    # corners' motion must agree with the IMU's prediction within 4 sigma (vgio_stereo_gate_factor > 1 also accepts
    # within that factor, the monocular test's tolerance for the learned scale); vgio_stereo_gate_pairs: the pass's
    # keyframe pairs (~2 s) too, against the graph's motion (KITTI 01: a pass accepted on its short pair pulled every
    # velocity of the window through its long ones, 24 -> 2 m/s; ATE 266 -> 38 m with the test)
    vgio_stereo_gate_factor: float = 1.0
    vgio_stereo_gate_pairs: bool = True
    vgio_debug_costs: bool = False               # graph: log the cost of each factor type per solve (diagnostics)


@dataclass
class _Observation:      # the fields of cross.mono.scale.ScaleObservation that the filter reads
    log_scale: float
    variance: float
    accepted: bool = True
    reason: str = ""


@dataclass
class _Interval:
    dt: float
    R: np.ndarray            # R_WB at the start of the interval
    dv: np.ndarray
    dp: np.ndarray
    J_v: np.ndarray
    J_p: np.ndarray
    W_v: np.ndarray          # whitening of the IMU velocity residual (inverse Cholesky factor of its covariance in W)
    var_p: float             # metric variance (per axis) of the IMU displacement
    z: np.ndarray            # visual displacement (units)
    lever: np.ndarray        # (R_WC,j - R_WC,i) t_cb (metres)
    sigma_u: float           # visual noise (units, per axis)
    obs: tuple | None = None  # learned-depth (log scale, variance) observed at the end of the interval


class InertialScaleFilter:
    def __init__(self, config: ImuConfig, T_cam_imu: np.ndarray, gyro_noise: float, accel_noise: float):
        self.config = config
        self.R_cb = np.asarray(T_cam_imu, dtype=np.float64)[:3, :3]       # IMU orientation in the camera frame
        self.t_cb = np.asarray(T_cam_imu, dtype=np.float64)[:3, 3]        # IMU position in the camera frame (m)
        self.gyro_noise = float(gyro_noise)
        self.accel_noise = float(np.hypot(accel_noise, config.extra_accel_noise))
        self.intervals: list[_Interval] = []
        self.v = np.zeros((1, 3))           # v_0..v_W
        self.l = np.zeros(1)                # l_0..l_W
        self.g = np.zeros(3)
        self.g0 = np.zeros(3)
        self.b = np.zeros(3)
        self.beta = 0.0                     # learned-depth log bias (depth_bias_state)
        self.nb = 1 if config.depth_bias_state else 0
        # marginalization prior on (v_0, l_0, g, b[, beta]): cost |L^T (d + shift)|^2 / 2, d = x - x_lin
        self.prior = None
        self.started = False
        self.initialized = False
        self.accepted = self.rejected = self.reinitializations = 0
        self.gated = 0
        self.pending = []                 # (LogScaleFilter interface; no deferred observations here)
        self.prior_log_scale = None       # the last learned-depth observation before the start
        self.pending_obs = None
        self._log_std = float("inf")
        self.last_nis = float("nan")
        self._mean_step = None
        self._since_grid = 0
        self._recent_nis = []
        self.l_ref = None                 # centre of the allowed log-scale band (latest learned-depth scale)

    # ------------------------------------------------------------------ LogScaleFilter interface
    @property
    def mean(self) -> float:
        if self.started:
            return float(self.l[-1])
        return self.prior_log_scale[0] if self.prior_log_scale is not None else 0.0

    @property
    def scale(self) -> float:
        return float(np.exp(self.mean))

    @property
    def log_std(self) -> float:
        return self._log_std

    @property
    def uncertainty_variance(self) -> float:
        return min(self.log_std ** 2, 1.0)

    @property
    def variance(self) -> float:
        return self.uncertainty_variance

    @property
    def velocity(self) -> np.ndarray:
        return self.v[-1].copy()

    def predict(self, frames=1):
        """(LogScaleFilter interface) the estimate is updated in step() with the IMU data."""

    def update(self, observation) -> bool:
        """A learned-depth scale observation (log scale, variance): a weak measurement of the newest log scale."""
        if not self.config.depth_prior or not getattr(observation, "accepted", False):
            return False
        obs = (float(observation.log_scale), max(float(observation.variance), self.config.depth_prior_std_floor ** 2))
        self.l_ref = obs[0]
        if not self.started:
            # until the IMU window starts, the learned-depth scale is the estimate (as in the learned-depth mode, so a
            # session starts as early as without the IMU)
            self.prior_log_scale = obs
            self._log_std = float(np.sqrt(obs[1]))
            self._check_initialized()
        if self.intervals:
            self.intervals[-1].obs = obs
        else:
            self.pending_obs = obs
        self.accepted += 1
        return True

    # ------------------------------------------------------------------ steps
    def start(self, R_wc: np.ndarray, accel_mean: np.ndarray):
        """First frame with a visual pose: gravity from the mean specific force, scale from the learned-depth prior."""
        g = -(R_wc @ self.R_cb) @ np.asarray(accel_mean, dtype=np.float64)
        self.g = GRAVITY * g / max(np.linalg.norm(g), 1e-9)
        self.g0 = self.g.copy()
        self.b = np.zeros(3)
        self.l = np.array([self.prior_log_scale[0] if self.prior_log_scale is not None else 0.0])
        self.v = np.zeros((1, 3))
        self.started = True

    def step(self, pre: Preintegrated, R_wc_i: np.ndarray, R_wc_j: np.ndarray, dp_units: np.ndarray,
             depth_units: float | None = None, visual_ok: bool = True) -> dict:
        """Add the frame interval i -> j (IMU and visual displacement in units) and re-estimate the window.
        visual_ok False: the visual displacement is not used (the IMU alone links the two frames)."""
        cfg = self.config
        dt = pre.dt
        R_i = R_wc_i @ self.R_cb
        q = self.accel_noise ** 2
        cov_v = R_i @ pre.cov[3:6, 3:6] @ R_i.T + q * dt * np.eye(3)
        var_p = float(np.trace(pre.cov[6:9, 6:9]) / 3 + q * dt ** 3 / 3)
        z = np.asarray(dp_units, dtype=np.float64)
        step_len = float(np.linalg.norm(z))
        self._mean_step = step_len if self._mean_step is None else 0.98 * self._mean_step + 0.02 * step_len
        floor = cfg.visual_std_depth * depth_units if depth_units else 0.1 * self._mean_step
        sigma_u = float(np.hypot(cfg.visual_std_rel * step_len, max(floor, 1e-12))) if visual_ok else float("inf")
        it = _Interval(dt, R_i, pre.dv, pre.dp, pre.J_v, pre.J_p, np.linalg.inv(np.linalg.cholesky(cov_v)), var_p, z,
                       (R_wc_j - R_wc_i) @ self.t_cb, sigma_u, self.pending_obs)
        self.pending_obs = None
        self.intervals.append(it)
        # the new frame's states predicted from the last ones
        self.v = np.vstack([self.v, self.v[-1] + self.g * dt + R_i @ (pre.dv - pre.J_v @ self.b)])
        self.l = np.append(self.l, self.l[-1])
        if len(self.intervals) > cfg.window:
            self._marginalize_oldest()
        if not self.initialized and (self._since_grid == 0 or len(self.intervals) < 30):
            self._grid_search()
        self._since_grid = (self._since_grid + 1) % 10
        info = self._solve(cfg.iterations)
        self._recent_nis = (self._recent_nis + [info["nis"]])[-cfg.reset_frames:]
        if len(self._recent_nis) == cfg.reset_frames and np.mean(self._recent_nis) > cfg.reset_nis:
            # the visual and inertial motion disagree persistently (a visual-odometry failure, or a wrong estimate):
            # start over from the current gravity direction and the learned-depth / current scale
            self._reset()
            return {"nis": info["nis"], "scale": self.scale, "log_std": self._log_std, "reset": True,
                    "window": len(self.intervals)}
        self._check_initialized()
        return {"nis": info["nis"], "scale": self.scale, "log_std": self._log_std,
                "speed": float(np.linalg.norm(self.v[-1])), "gravity_norm": float(np.linalg.norm(self.g)),
                "accel_bias": self.b.round(4).tolist(), "window": len(self.intervals), "depth_bias": float(self.beta)}

    # ------------------------------------------------------------------ the least-squares problem
    def _stack(self, its):
        """The intervals' data as arrays (n leading)."""
        a = {k: np.stack([getattr(it, k) for it in its]) for k in ("R", "dv", "dp", "J_v", "J_p", "W_v", "z", "lever")}
        a["dt"] = np.array([it.dt for it in its])
        a["var_p"] = np.array([it.var_p for it in its])
        a["sigma_u"] = np.array([it.sigma_u for it in its])
        a["has_obs"] = np.array([it.obs is not None for it in its])
        a["obs"] = np.array([it.obs if it.obs is not None else (0.0, 1.0) for it in its])
        if not self.config.depth_prior_independent:
            a["obs"][:, 1] *= max(int(a["has_obs"].sum()), 1)    # all observations of the window weigh as one
        return a

    def _local(self, a, v0, v1, l0, l1, g, b, fixed_scale=False, beta=None):
        """Whitened residuals (n, 8) and Jacobians (n, 8, 14) of every interval over its local variables
        (v_k, l_k, v_k+1, l_k+1, g, b); rows: IMU (3), visual (3, Huber-weighted), drift, learned depth (0 if none).
        Also returns the visual errors in sigmas."""
        cfg = self.config
        n = len(a["dt"])
        dt = a["dt"]
        I3 = np.eye(3)
        r = np.zeros((n, 8))
        J = np.zeros((n, 8, 14 + self.nb))
        Rb = a["R"]
        imu = v1 - v0 - g[None] * dt[:, None] - np.einsum("nij,nj->ni", Rb, a["dv"] - np.einsum("nij,j->ni", a["J_v"], b))
        W = a["W_v"]
        r[:, 0:3] = np.einsum("nij,nj->ni", W, imu)
        J[:, 0:3, 4:7] = W
        J[:, 0:3, 0:3] = -W
        J[:, 0:3, 8:11] = -W * dt[:, None, None]
        J[:, 0:3, 11:14] = np.einsum("nij,njk,nkl->nil", W, Rb, a["J_v"])
        s_inv = np.exp(-l0)
        mdisp = (v0 * dt[:, None] + 0.5 * g[None] * dt[:, None] ** 2
                 + np.einsum("nij,nj->ni", Rb, a["dp"] - np.einsum("nij,j->ni", a["J_p"], b)) - a["lever"])
        used = np.isfinite(a["sigma_u"])
        sig = np.sqrt(np.where(used, a["sigma_u"], 1.0) ** 2 + s_inv ** 2 * a["var_p"])
        res = a["z"] - s_inv[:, None] * mdisp
        e = np.where(used, np.linalg.norm(res, axis=1) / sig, 0.0)
        w = np.where(used, np.where(e <= cfg.huber, 1.0, np.sqrt(cfg.huber / np.maximum(e, 1e-12))) / sig, 0.0)
        r[:, 3:6] = w[:, None] * res
        if not fixed_scale:
            J[:, 3:6, 3] = (w * s_inv)[:, None] * mdisp
        ws = (w * s_inv)[:, None, None]
        J[:, 3:6, 0:3] = -ws * dt[:, None, None] * I3
        J[:, 3:6, 8:11] = -ws * 0.5 * dt[:, None, None] ** 2 * I3
        J[:, 3:6, 11:14] = ws * np.einsum("nij,njk->nik", Rb, a["J_p"])
        if not fixed_scale:
            sd = cfg.scale_drift * np.sqrt(dt)
            r[:, 6] = (l1 - l0) / sd
            J[:, 6, 7], J[:, 6, 3] = 1 / sd, -1 / sd
            so = np.sqrt(a["obs"][:, 1])
            m = a["has_obs"].astype(float)
            bias = beta if (self.nb and beta is not None) else 0.0
            r[:, 7] = m * (l1 + bias - a["obs"][:, 0]) / so
            J[:, 7, 7] = m / so
            if self.nb:
                J[:, 7, 14] = m / so
        return r, J, e

    def _prior_rows(self, x_prior, fixed_scale=False):
        """Residuals of the marginalization prior (or the start-up priors) over (v0, l0, g, b) (10 columns)."""
        cfg = self.config
        if self.prior is not None:
            L, shift, x_lin = self.prior
            J = L.T.copy()
            if fixed_scale:
                J[:, 3] = 0.0
            return L.T @ (x_prior - x_lin + shift), J
        J = np.zeros((6 + self.nb, 10 + self.nb))
        J[:3, 4:7] = np.eye(3) / cfg.gravity_std
        J[3:6, 7:10] = np.eye(3) / cfg.accel_bias_std
        r = [(x_prior[4:7] - self.g0) / cfg.gravity_std, x_prior[7:10] / cfg.accel_bias_std]
        if self.nb:
            J[6, 10] = 1.0 / cfg.depth_bias_std
            r.append([x_prior[10] / cfg.depth_bias_std])
        return np.concatenate(r), J

    def _normal(self, a, v, l, g, b, beta=None, fixed_scale=False):
        """Normal equations (H, gradient), cost and visual errors of the window (columns: v_0..v_n, l_0..l_n, g, b
        [, beta])."""
        cfg = self.config
        n = len(a["dt"])
        m = 4 * (n + 1) + 6 + self.nb
        ig, ib, iq = 4 * (n + 1), 4 * (n + 1) + 3, 4 * (n + 1) + 6
        r, J, e = self._local(a, v[:-1], v[1:], l[:-1], l[1:], g, b, fixed_scale, beta)
        k = np.arange(n)
        idx = np.stack([3 * k, 3 * k + 1, 3 * k + 2, 3 * (n + 1) + k, 3 * k + 3, 3 * k + 4, 3 * k + 5,
                        3 * (n + 1) + k + 1] + [np.full(n, ig + j) for j in range(3)]
                       + [np.full(n, ib + j) for j in range(3)] + [np.full(n, iq)] * self.nb, axis=1)  # (n, 14 [+1])
        H = np.zeros((m, m))
        np.add.at(H, (idx[:, :, None], idx[:, None, :]), np.einsum("nri,nrj->nij", J, J))
        grad = np.zeros(m)
        np.add.at(grad, idx, np.einsum("nri,nr->ni", J, r))
        cost = float((r ** 2).sum())
        gn = float(np.linalg.norm(g))
        jg = g / max(gn, 1e-9) / cfg.gravity_norm_std
        rg = (gn - GRAVITY) / cfg.gravity_norm_std
        H[ig:ig + 3, ig:ig + 3] += np.outer(jg, jg)
        grad[ig:ig + 3] += jg * rg
        cost += rg ** 2
        cols = np.r_[0:3, 3 * (n + 1), ig:ig + 3, ib:ib + 3, iq:iq + self.nb]
        rp, Jp = self._prior_rows(np.concatenate([v[0], [l[0]], g, b, [beta] if self.nb else []]), fixed_scale)
        H[np.ix_(cols, cols)] += Jp.T @ Jp
        grad[cols] += Jp.T @ rp
        cost += float(rp @ rp)
        return H, grad, cost, e

    def _split(self, x):
        n = len(self.intervals)
        beta = float(x[4 * (n + 1) + 6]) if self.nb else None
        return (x[:3 * (n + 1)].reshape(-1, 3), x[3 * (n + 1):4 * (n + 1)], x[4 * (n + 1):4 * (n + 1) + 3],
                x[4 * (n + 1) + 3:4 * (n + 1) + 6], beta)

    def _x(self):
        return np.concatenate([self.v.reshape(-1), self.l, self.g, self.b, [self.beta] if self.nb else []])

    def _set(self, x):
        v, l, g, b, beta = self._split(x)
        self.v, self.l, self.g, self.b = v.copy(), self._clamp(l), g.copy(), b.copy()
        if self.nb:
            self.beta = beta

    def _solve(self, iterations):
        a = self._stack(self.intervals)
        x = self._x()
        n = len(self.intervals)
        damping = 1e-6
        H, grad, cost, vis = self._normal(a, *self._split(x))
        for _ in range(iterations):
            step = np.linalg.solve(H + damping * np.diag(np.diag(H) + 1e-9), -grad)
            x_new = x + step
            H_new, grad_new, c_new, vis_new = self._normal(a, *self._split(x_new))
            if c_new <= cost:
                x, H, grad, vis, cost = x_new, H_new, grad_new, vis_new, c_new
                damping = max(damping / 10, 1e-9)
            else:
                damping *= 10
        self._set(x)
        e = np.zeros(len(x))
        e[3 * (n + 1) + n] = 1.0
        try:
            self._log_std = float(np.sqrt(max(np.linalg.solve(H + 1e-12 * np.eye(len(x)), e)[3 * (n + 1) + n], 0.0)))
        except np.linalg.LinAlgError:
            self._log_std = float("inf")
        self.last_nis = float(vis[-1] ** 2) if len(vis) else float("nan")
        self.gated += int(len(vis) > 0 and vis[-1] > self.config.huber)
        return {"nis": self.last_nis, "cost": cost}

    def _grid_search(self):
        """Solve the window for constant log scales on a grid (linear in the rest up to |g|); start from the best."""
        cfg = self.config
        center = self.prior_log_scale[0] if self.prior_log_scale is not None else 0.0
        half = cfg.scale_prior_log_std if self.l_ref is None else min(cfg.scale_prior_log_std, np.log(cfg.scale_band))
        grid = center + np.arange(-half, half + 1e-9, cfg.grid_step)
        a = self._stack(self.intervals)
        n = len(self.intervals)
        x0 = self._x()
        keep = np.r_[0:3 * (n + 1), 4 * (n + 1):4 * (n + 1) + 6]      # (beta stays: no learned depth in these solves)
        best = None
        for lg in grid:
            x = x0.copy()
            x[3 * (n + 1):4 * (n + 1)] = lg
            for _ in range(2):           # linear for a fixed scale, up to the gravity norm: two Gauss-Newton steps
                H, grad, _, _ = self._normal(a, *self._split(x), fixed_scale=True)
                Hk = H[np.ix_(keep, keep)]
                x[keep] += np.linalg.solve(Hk + 1e-9 * np.eye(len(keep)), -grad[keep])
            _, _, c, _ = self._normal(a, *self._split(x), fixed_scale=True)
            n_obs = 1 if cfg.depth_prior_independent else max(sum(it.obs is not None for it in self.intervals), 1)
            bias = self.beta if self.nb else 0.0
            c += sum((lg + bias - it.obs[0]) ** 2 / (it.obs[1] * n_obs) for it in self.intervals if it.obs is not None)
            if best is None or c < best[0]:
                best = (c, x)
        self._set(best[1])

    def _marginalize_oldest(self):
        """Drop v_0, l_0 and interval 0: their factors (prior, IMU, visual, drift) become a Gaussian prior on
        (v_1, l_1, g, b) by the Schur complement, linearized at the current estimate."""
        cfg = self.config
        it = self.intervals[0]
        a = self._stack(self.intervals[:1])
        if not cfg.depth_prior_independent:
            a["has_obs"][:] = False                              # the learned-depth prior is not carried on
        r, J, _ = self._local(a, self.v[0:1], self.v[1:2], self.l[0:1], self.l[1:2], self.g, self.b, beta=self.beta)
        r, J = r[0], J[0]
        nc = 14 + self.nb
        x_lin = np.concatenate([self.v[0], [self.l[0]], self.v[1], [self.l[1]], self.g, self.b,
                                [self.beta] if self.nb else []])
        sel = np.r_[0:4, 8:nc]                                  # (v0, l0, g, b[, beta]) among the local columns
        rp, Jp = self._prior_rows(x_lin[sel])
        full = np.zeros((len(rp), nc))
        full[:, sel] = Jp
        r, J = np.concatenate([r, rp]), np.vstack([J, full])
        H, c = J.T @ J, J.T @ r
        mi, ri = np.r_[0:4], np.r_[4:nc]
        K = H[np.ix_(ri, mi)] @ np.linalg.inv(H[np.ix_(mi, mi)] + 1e-9 * np.eye(4))
        Hp = H[np.ix_(ri, ri)] - K @ H[np.ix_(mi, ri)]
        cp = c[ri] - K @ c[mi]
        Hp = 0.5 * (Hp + Hp.T) + 1e-9 * np.eye(len(ri))
        # cost 1/2 d^T Hp d + cp^T d: a Gaussian with mean x_lin - Hp^-1 cp; gravity and bias random walks over the
        # interval are added to its covariance (variables v1 l1 g b: g at 4:7, b at 7:10)
        mean = x_lin[ri] - np.linalg.solve(Hp, cp)
        cov = np.linalg.inv(Hp)
        cov[4:7, 4:7] += np.eye(3) * cfg.gravity_drift ** 2 * it.dt
        cov[7:10, 7:10] += np.eye(3) * cfg.accel_bias_walk ** 2 * it.dt
        if self.nb:
            cov[10, 10] += cfg.depth_bias_drift ** 2 * it.dt
        Hp = np.linalg.inv(0.5 * (cov + cov.T))
        L = np.linalg.cholesky(0.5 * (Hp + Hp.T))
        self.prior = (L, np.zeros(len(ri)), mean)
        self.intervals.pop(0)
        self.v = self.v[1:]
        self.l = self.l[1:]

    def _clamp(self, l):
        if self.l_ref is None:
            return l.copy()
        band = np.log(self.config.scale_band)
        return np.clip(l, self.l_ref - band, self.l_ref + band)

    def _check_initialized(self):
        """The scale counts as known once its log std is below init_log_std: with a learned-depth prior at once, with
        the IMU alone after at least 10 intervals of the window."""
        if self.initialized or self._log_std >= self.config.init_log_std:
            return
        if self.prior_log_scale is not None or len(self.intervals) >= 10:
            self.initialized = True
            if self.l_ref is None and self.started:
                self.l_ref = float(self.l[-1])

    def _reset(self):
        g_dir = self.g / max(np.linalg.norm(self.g), 1e-9)
        scale = self.l[-1]
        self.intervals = []
        self.prior = None
        self.g = GRAVITY * g_dir
        self.g0 = self.g.copy()
        self.b = np.zeros(3)
        self.v = np.zeros((1, 3))
        self.l = np.array([scale])
        self._recent_nis = []
        self._since_grid = 0
        self.reinitializations += 1
        # the scale stays usable (it was known before); its uncertainty restarts from the learned-depth prior level
        self._log_std = max(self._log_std, self.config.depth_prior_std_floor)

    def as_observation(self):
        return _Observation(self.mean, self.uncertainty_variance)
