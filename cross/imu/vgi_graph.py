"""Local pose graph of visual-inertial odometry with feed-forward relative poses (VGGT-Omega) and an IMU.

A sliding window of nodes, one per visual measurement (a forward pass of the geometry model on the current frame and
some earlier nodes' frames).  Unknowns:

    per node i     R_i, p_i   camera orientation and position in the world (metres)
                   v_i        velocity of the IMU (world)
                   lam_i      log metres per unit of node i's forward pass (each pass has a gauge of its own)
    global         g          gravity (world), |g| = 9.81
                   b_g, b_a   gyroscope and accelerometer biases (constant in the window, random walk on marginalization)
                   beta       log bias of learned metric depth (optional)
                   kappa      log scale of the model's rotation angles (optional; e.g. a focal length estimated
                              too short makes every rotation too large: KITTI 07, +4-6 %)
                   t_d        camera - IMU time offset (optional)

Factors:

    IMU            preintegration between consecutive nodes, with first-order corrections for the biases (and the
                   time offset)
    relative pose  every pair of views of a pass: rotation, and translation = exp(lam_pass) * the pass's translation
                   (Huber)
    gauge link     lam_j - lam_k = log of the depth ratio of a frame both passes contain
    learned depth  lam_i + beta = log(learned metric depth / pass depth) of node i's frame
    stereo         lam_i = the log metres per unit of node i's pass, observed by a stereo pair (metric, so no bias
                   state; Huber)
    metric pose    a relative pose in metres with a full 6x6 covariance, independent of the passes' scales (stereo:
                   corners tracked between two nodes' frames, 3-D from stereo depth; Huber)
    priors         |g|; start-up priors on the first node (gauge) and the biases; the marginalization prior

Gauss-Newton on the stacked tangent-space increment (right perturbations of the rotations); each factor's Jacobian is
analytic over its own variables (numpy, batched over the factors of a type) and scattered into the window's normal
equations.  The oldest node is marginalized by the Schur complement into a prior on the next node and the globals (its
relative-pose factors with later nodes are dropped)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .preintegration import preintegrate

GRAVITY = 9.81


@dataclass
class GraphConfig:
    window: int = 20                     # nodes
    rot_std: float = 0.0087              # rad, relative rotation of a pass (0.5 deg) ...
    rot_rel: float = 0.0                 # ... plus this fraction of the rotation angle (larger turns, larger errors)
    trans_rel: float = 0.03              # relative translation noise of a pass
    trans_floor: float = 0.01            # m
    link_std: float = 0.03               # gauge link (log)
    depth_std_floor: float = 0.15        # learned depth (log)
    depth_bias: bool = False             # estimate the learned-depth bias
    depth_bias_std: float = 0.3
    depth_bias_drift: float = 0.005      # per sqrt(s)
    rot_scale: bool = False              # estimate the rotation scale of the passes (calibrated against the gyro)
    rot_scale_std: float = 0.05
    rot_scale_drift: float = 0.001       # per sqrt(s)
    # the camera-IMU time offset as a variable (s; camera clock = IMU clock + offset), observed through the rotation
    # and the motion of the IMU factors; the samples are taken at the frontend's offset and corrected to first order
    time_offset: bool = False
    time_offset_std: float = 0.05
    time_offset_drift: float = 0.0001
    gravity_std: float = 0.3
    gravity_norm_std: float = 0.05
    gravity_drift: float = 0.02
    gyro_bias_std: float = 0.005         # rad/s, start-up prior (MEMS gyroscopes after start-up calibration)
    gyro_bias_walk: float = 0.0001       # rad/s per sqrt(s)
    accel_bias_std: float = 0.1
    accel_bias_walk: float = 0.005
    extra_accel_noise: float = 0.01
    # discretization error of the preintegration, grows with the sampling interval h of the IMU (noise densities
    # hypot(nominal, k * h)): negligible at 200-400 Hz, ~ the nominal noise for KITTI's 10 Hz OXTS
    gyro_dt_noise: float = 0.0
    accel_dt_noise: float = 0.0
    huber: float = 3.0
    iterations: int = 3
    # robustness to gross visual errors: the relative translation noise of a pass from the larger of its measured and
    # its predicted translation (False: the measured one), and Huber weights on the gauge links (False: none)
    trans_sigma_predicted: bool = True
    # ... with the predicted translation lowered by twice its uncertainty (from the velocity's marginal covariance): a
    # prediction the graph does not yet know (session start, poor vision) cannot weaken the measurements
    trans_sigma_bound: bool = False
    robust_links: bool = True
    debug_costs: bool = False            # solve() also reports the cost of each factor type (diagnostics)


def _exp(phi):
    return Rotation.from_rotvec(phi).as_matrix()


def _so3(R):
    """The nearest rotation.  Node rotations are chained (a new node starts from its predecessor through the
    extrinsic and the preintegrated rotation) and updated in place for the whole session, so rounding and an imperfect
    extrinsic accumulate unless every stored rotation is projected back (KITTI 00: singular values 1 +- 1e-4 after
    800 nodes without it)."""
    U, _, Vt = np.linalg.svd(R)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _log(R):
    return Rotation.from_matrix(R).as_rotvec()


def _skew(v):
    """Batched skew matrices of (..., 3) vectors."""
    z = np.zeros(v.shape[:-1])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1),
                     np.stack([v[..., 2], z, -v[..., 0]], -1),
                     np.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def _exp_b(phi):
    """Batched SO(3) exponential of (..., 3) rotation vectors."""
    t2 = (phi * phi).sum(-1)[..., None, None]
    small = t2 < 1e-6
    t2s = np.where(small, 1.0, t2)
    t = np.sqrt(t2s)
    A = np.where(small, 1 - t2 / 6 + t2 * t2 / 120, np.sin(t) / t)
    B = np.where(small, 0.5 - t2 / 24 + t2 * t2 / 720, (1 - np.cos(t)) / t2s)
    K = _skew(phi)
    return np.eye(3) + A * K + B * (K @ K)


def _log_b(R):
    """Batched SO(3) logarithm (rotations away from pi)."""
    s = 0.5 * np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    c = 0.5 * (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1)
    sn2 = (s * s).sum(-1)
    small = sn2 < 1e-8
    sn = np.sqrt(np.where(small, 1.0, sn2))
    f = np.where(small, 1 + sn2 / 6, np.arctan2(sn, c) / sn)
    return f[..., None] * s


def _jr(phi):
    """Batched right Jacobian of SO(3): Exp(phi + d) = Exp(phi) Exp(Jr(phi) d) to first order."""
    t2 = (phi * phi).sum(-1)[..., None, None]
    small = t2 < 1e-6
    t2s = np.where(small, 1.0, t2)
    t = np.sqrt(t2s)
    A = np.where(small, 0.5 - t2 / 24, (1 - np.cos(t)) / t2s)
    B = np.where(small, 1.0 / 6 - t2 / 120, (t - np.sin(t)) / (t2s * t))
    K = _skew(phi)
    return np.eye(3) - A * K + B * (K @ K)


def _jr_inv(phi):
    """Batched inverse right Jacobian: Log(Exp(phi) Exp(d)) = phi + Jr^-1(phi) d to first order."""
    t2 = (phi * phi).sum(-1)[..., None, None]
    small = t2 < 1e-6
    t2s = np.where(small, 1.0, t2)
    t = np.sqrt(t2s)
    C = np.where(small, 1.0 / 12 + t2 / 720, 1.0 / t2s - (1 + np.cos(t)) / (2 * t * np.sin(np.where(small, 1.0, t))))
    K = _skew(phi)
    return np.eye(3) + 0.5 * K + C * (K @ K)


def _T(M):
    return np.swapaxes(M, -1, -2)


def _mv(M, x):
    return np.einsum("...ij,...j->...i", M, x)


def _imu_block(Ri, pi, vi, Rj, pj, vj, q, g, bg, ba, Rcb, tcb, jac, td=None):
    """IMU factors (batched): whitened residuals (n, 9) and Jacobians (n, 9, 27 [+1]) over (rotation, position, velocity
    of nodes i and j, gravity, gyro bias, accelerometer bias[, time offset])."""
    dt = q["dt"]
    dbg = bg[None] - q["bgpre"]
    psi = _mv(q["JRg"], dbg)
    dv = q["dv"] + _mv(q["Jv"], np.broadcast_to(ba, dbg.shape)) + _mv(q["Jvg"], dbg)
    dp = q["dp"] + _mv(q["Jp"], np.broadcast_to(ba, dbg.shape)) + _mv(q["Jpg"], dbg)
    if td is not None:
        dtd = td - q["td0"]
        psi = psi + q["JRt"] * dtd[:, None]
        dv = dv + q["Jvt"] * dtd[:, None]
        dp = dp + q["Jpt"] * dtd[:, None]
    dRc = q["dR"] @ _exp_b(psi)
    Rbi, Rbj = Ri @ Rcb, Rj @ Rcb
    pbi, pbj = pi + _mv(Ri, np.broadcast_to(tcb, pi.shape)), pj + _mv(Rj, np.broadcast_to(tcb, pj.shape))
    E = _T(dRc) @ _T(Rbi) @ Rbj                                  # the rotation residual's rotation
    rR = _log_b(E)
    w = vj - vi - g[None] * dt[:, None]
    yv = _mv(_T(Rbi), w)
    rv = yv - dv
    yp = _mv(_T(Rbi), pbj - pbi - vi * dt[:, None] - 0.5 * g[None] * dt[:, None] ** 2)
    rp = yp - dp
    r9 = np.concatenate([rR, rv, rp], 1)
    r = _mv(q["W"], r9)
    if not jac:
        return r, None
    n = len(dt)
    Ji = _jr_inv(rR)
    RcbT = Rcb.T
    J9 = np.zeros((n, 9, 27 if td is None else 28))
    if td is not None:
        J9[:, 0:3, 27] = (-Ji @ _T(E) @ _jr(psi) @ q["JRt"][:, :, None])[:, :, 0]
        J9[:, 3:6, 27] = -q["Jvt"]
        J9[:, 6:9, 27] = -q["Jpt"]
    # rotation residual: theta_i, theta_j, gyro bias
    J9[:, 0:3, 0:3] = -Ji @ _T(Rbj) @ Rbi @ RcbT
    J9[:, 0:3, 9:12] = Ji @ RcbT
    J9[:, 0:3, 21:24] = -Ji @ _T(E) @ _jr(psi) @ q["JRg"]
    # velocity residual
    RbiT = _T(Rbi)
    J9[:, 3:6, 0:3] = _skew(yv) @ RcbT
    J9[:, 3:6, 6:9] = -RbiT
    J9[:, 3:6, 15:18] = RbiT
    J9[:, 3:6, 18:21] = -RbiT * dt[:, None, None]
    J9[:, 3:6, 21:24] = -q["Jvg"]
    J9[:, 3:6, 24:27] = -q["Jv"]
    # position residual (the IMU's position: the camera's plus the rotated lever arm)
    J9[:, 6:9, 0:3] = _skew(yp) @ RcbT + RcbT @ _skew(np.broadcast_to(tcb, (n, 3)))
    J9[:, 6:9, 3:6] = -RbiT
    J9[:, 6:9, 6:9] = -RbiT * dt[:, None, None]
    J9[:, 6:9, 9:12] = -RbiT @ Rj @ _skew(np.broadcast_to(tcb, (n, 3)))
    J9[:, 6:9, 12:15] = RbiT
    J9[:, 6:9, 18:21] = -0.5 * RbiT * (dt ** 2)[:, None, None]
    J9[:, 6:9, 21:24] = -q["Jpg"]
    J9[:, 6:9, 24:27] = -q["Jp"]
    return r, q["W"] @ J9


def _rel_block(Ra, pa, Rb, pb, lam_s, q, sig, kappa, jac):
    """Relative poses of the passes (batched): residuals (n, 6) and Jacobians (n, 6, 13 or 14) over (rotation and
    position of a, rotation and position of b, the pass's log scale[, the log rotation scale]).  The translation
    residual is in the pass's units (errors in variables)."""
    n = len(lam_s)
    M = _T(Ra) @ Rb
    if kappa is None:
        Rm = q["Rm"]
    else:
        sc = np.exp(-kappa)
        Rm = _exp_b(sc * q["rvm"])
    E = _T(Rm) @ M
    phi = _log_b(E)
    rs = q["rot_sig"][:, None]
    u = _mv(_T(Ra), pb - pa)
    el = np.exp(-lam_s)
    rt = (el[:, None] * u - q["tm"]) / sig[:, None]
    r = np.concatenate([phi / rs, rt], 1)
    if not jac:
        return r, None
    Ji = _jr_inv(phi)
    J = np.zeros((n, 6, 13 if kappa is None else 14))
    J[:, 0:3, 0:3] = -Ji @ _T(Rb) @ Ra / rs[:, :, None]
    J[:, 0:3, 6:9] = Ji / rs[:, :, None]
    k = (el / sig)[:, None, None]
    J[:, 3:6, 0:3] = k * _skew(u)
    J[:, 3:6, 3:6] = -k * _T(Ra)
    J[:, 3:6, 9:12] = k * _T(Ra)
    J[:, 3:6, 12] = -(el / sig)[:, None] * u
    if kappa is not None:
        J[:, 0:3, 13] = sc * _mv(Ji @ _T(M), q["rvm"]) / rs
    return r, J


def _rot_block(Ra, Rb, q, jac):
    """Rotation-only measurements (batched): residuals (n, 3) and Jacobians (n, 3, 6) over the rotations of a, b."""
    E = _T(q["Rm"]) @ _T(Ra) @ Rb
    phi = _log_b(E)
    sd = q["std"][:, None]
    r = phi / sd
    if not jac:
        return r, None
    Ji = _jr_inv(phi)
    J = np.concatenate([-Ji @ _T(Rb) @ Ra, Ji], 2) / sd[:, :, None]
    return r, J


def _met_block(Ra, pa, Rb, pb, q, jac):
    """Metric relative poses (batched): whitened residuals (n, 6) and Jacobians (n, 6, 12) over (rotation and position
    of a, rotation and position of b).  Residual [Log(Rm^T Ra^T Rb), Ra^T (pb - pa) - tm], whitened by W (W^T W =
    covariance^-1)."""
    E = _T(q["Rm"]) @ _T(Ra) @ Rb
    phi = _log_b(E)
    u = _mv(_T(Ra), pb - pa)
    r6 = np.concatenate([phi, u - q["tm"]], 1)
    r = _mv(q["W"], r6)
    if not jac:
        return r, None
    n = len(phi)
    Ji = _jr_inv(phi)
    J = np.zeros((n, 6, 12))
    J[:, 0:3, 0:3] = -Ji @ _T(Rb) @ Ra
    J[:, 0:3, 6:9] = Ji
    J[:, 3:6, 0:3] = _skew(u)
    J[:, 3:6, 3:6] = -_T(Ra)
    J[:, 3:6, 9:12] = _T(Ra)
    return r, q["W"] @ J


def _prior_rotation(R_ref, R, std):
    """log(R_ref^T R Exp(d)) / std at d = 0 and its Jacobian (3 x 3)."""
    phi = _log_b(R_ref.T @ R)
    return phi / std, _jr_inv(phi) / std

@dataclass
class _Imu:
    i: int                       # absolute node ids i -> i + 1
    dt: float
    dR: np.ndarray
    dv: np.ndarray
    dp: np.ndarray
    J_v: np.ndarray              # d dv / d b_a (dv(b_a) = dv + J_v b_a)
    J_p: np.ndarray
    JR_g: np.ndarray             # d log(dR) / d b_g
    Jv_g: np.ndarray
    Jp_g: np.ndarray
    bg0: np.ndarray              # gyro bias of the preintegration
    W: np.ndarray                # 9x9 whitening (rotation, velocity, position)
    t0: float = 0.0              # the interval (camera clock), to preintegrate again with another time offset
    t1: float = 0.0
    JR_t: np.ndarray = None      # d log(dR) / d time offset (time_offset), and of dv, dp
    Jv_t: np.ndarray = None
    Jp_t: np.ndarray = None
    td0: float = 0.0             # the time offset of the samples


@dataclass
class _Rot:
    a: int                       # a rotation-only measurement (e.g. tracked features with the calibrated camera)
    b: int
    R: np.ndarray
    std: float


@dataclass
class _Met:
    a: int                       # a metric relative pose: b in a's camera frame (metres)
    b: int
    R: np.ndarray
    t: np.ndarray
    W: np.ndarray                # 6x6 whitening of (rotation, translation)


@dataclass
class _Rel:
    a: int                       # absolute node ids, pose of b in a's camera frame (pass units)
    b: int
    s: int                       # node whose pass measured it (its lam scales the translation)
    R: np.ndarray
    t: np.ndarray


class VgiGraph:
    def __init__(self, config: GraphConfig, T_cam_imu: np.ndarray, gyro_noise: float, accel_noise: float):
        self.cfg = config
        self.R_cb = _so3(np.asarray(T_cam_imu, dtype=np.float64)[:3, :3])
        self.t_cb = np.asarray(T_cam_imu, dtype=np.float64)[:3, 3]
        self.gyro_noise = float(gyro_noise)
        self.accel_noise = float(np.hypot(accel_noise, config.extra_accel_noise))
        self.ids: list[int] = []                 # absolute ids of the window's nodes
        self.R: dict[int, np.ndarray] = {}
        self.p: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}
        self.lam: dict[int, float] = {}
        self.t: dict[int, float] = {}
        self.g = np.zeros(3)
        self.g0 = np.zeros(3)
        self.bg = np.zeros(3)
        self.bg0 = np.zeros(3)
        self.ba = np.zeros(3)
        self.beta = 0.0
        self.kappa = 0.0                    # log rotation scale of the passes (rot_scale)
        self.td = 0.0                       # camera - IMU time offset (time_offset)
        self.td_init = 0.0
        self.imu: list[_Imu] = []
        self.rel: list[_Rel] = []
        self.rots: list[_Rot] = []
        self.met: list[_Met] = []
        self.links: list[tuple] = []             # (j, k, log ratio, std): lam_j - lam_k = log ratio
        self.depth: dict[int, tuple] = {}        # node -> (log observation, std)
        self.stereo: dict[int, tuple] = {}       # node -> (log observation, std): stereo scale of its pass
        self.zrate: dict[int, tuple] = {}        # node -> (gyro mean (IMU frame), std (3,)) over an interval at rest
        self.prior = None                        # (L, node id, x_lin) on (node, globals)
        self.gauge = None
        self.next_id = 0
        self.last_info = {}
        # motion level seen by the IMU (running RMS of the bias-corrected rate and of the specific force's deviation
        # from its interval mean): scales the uncertainty of the IMU factors across sample dropouts
        self.motion_w = 0.0
        self.motion_a = 0.0
        self.v_std = np.inf                      # marginal std of the newest node's velocity (m/s, last solve)

    # ------------------------------------------------------------------ building
    @property
    def n_global(self):
        return 9 + int(self.cfg.depth_bias) + int(self.cfg.rot_scale) + int(self.cfg.time_offset)

    @property
    def i_kappa(self):
        return 9 + int(self.cfg.depth_bias)          # offset of kappa among the globals

    @property
    def i_td(self):
        return 9 + int(self.cfg.depth_bias) + int(self.cfg.rot_scale)

    def start(self, R_wc, accel_mean_body, lam0, t0, bg0=None):
        """The first node: gravity from the mean specific force, the gauge fixed at R_wc, p = 0."""
        g = -(R_wc @ self.R_cb) @ np.asarray(accel_mean_body, dtype=np.float64)
        self.g = GRAVITY * g / max(np.linalg.norm(g), 1e-9)
        self.g0 = self.g.copy()
        if bg0 is not None:
            self.bg, self.bg0 = np.asarray(bg0, dtype=np.float64).copy(), np.asarray(bg0, dtype=np.float64).copy()
        i = self._new_node(R_wc, np.zeros(3), np.zeros(3), lam0, t0)
        self.gauge = (np.asarray(R_wc, dtype=np.float64).copy(), np.zeros(3))      # the first node's pose stays fixed
        return i

    def _new_node(self, R, p, v, lam, t):
        i = self.next_id
        self.next_id += 1
        self.ids.append(i)
        self.R[i], self.p[i], self.v[i], self.lam[i], self.t[i] = _so3(R), p.copy(), v.copy(), float(lam), float(t)
        return i

    def preintegrate(self, samples, t0, t1, td0=None):
        """IMU of an interval (samples: raw rows t w a on the camera clock, taken at time offset td0) at the current
        gyro bias, with the gyro-bias Jacobians (analytic) and the time-offset Jacobians (central differences)."""
        h = float(np.median(np.diff(samples[:, 0]))) if len(samples) > 1 else 0.0
        gn = float(np.hypot(self.gyro_noise, self.cfg.gyro_dt_noise * h))
        an = float(np.hypot(self.accel_noise, self.cfg.accel_dt_noise * h))

        def run(bg, shift=0.0, full=True):
            s = samples.copy()
            s[:, 1:4] -= bg
            s[:, 0] += shift                     # a larger offset: every sample later on the camera clock
            return preintegrate(s, t0, t1, gn, an, full=full)
        base = run(self.bg)
        self._dropout(base, samples, t0, t1, h)
        if not self.cfg.time_offset:
            return base, (base.J_Rg, base.J_vg, base.J_pg)
        et = 2e-3
        pp, pm = run(self.bg, et, False), run(self.bg, -et, False)
        tj = dict(JR_t=_log(pm.dR.T @ pp.dR) / (2 * et), Jv_t=(pp.dv - pm.dv) / (2 * et),
                  Jp_t=(pp.dp - pm.dp) / (2 * et), td0=float(self.td if td0 is None else td0))
        return base, (base.J_Rg, base.J_vg, base.J_pg, tj)

    def _dropout(self, pre, samples, t0, t1, h):
        """Sample dropouts: the preintegration bridges a gap in the IMU stream by holding the last sample, as if the
        platform had kept its rate and acceleration (ROVER summer: gaps of 1.0-1.3 s; the bridged rotation failed the
        gyro test of the passes and the bridged velocity started a runaway).  The factor's covariance grows with the
        gaps (the parts longer than 5 nominal sample intervals): rotation by 3x the IMU's recent RMS rate x gap,
        velocity by 3x its recent RMS acceleration x gap, position by that x the interval.  The motion level is the
        IMU's own, so nothing here depends on the platform."""
        t = samples[:, 0]
        inside = (t >= t0 - h) & (t <= t1 + h)
        if inside.sum() >= 3:
            w = samples[inside, 1:4] - self.bg
            a = samples[inside, 4:7]
            rw = float(np.sqrt((w ** 2).sum(1).mean()))
            ra = float(np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean()))
            k = 0.05 if self.motion_w > 0 else 1.0
            self.motion_w += k * (rw - self.motion_w)
            self.motion_a += k * (ra - self.motion_a)
        d = np.diff(np.clip(t, t0, t1))
        gap = float(np.sum(d[d > max(5 * h, 1e-3)])) if h > 0 else 0.0
        if gap <= 0:
            return
        dt = max(t1 - t0, gap)
        sr = 3.0 * max(self.motion_w, 0.05) * gap
        sv = 3.0 * max(self.motion_a, 0.1) * gap
        pre.cov = pre.cov.copy()
        pre.cov[0:3, 0:3] += np.eye(3) * sr ** 2
        pre.cov[3:6, 3:6] += np.eye(3) * sv ** 2
        pre.cov[6:9, 6:9] += np.eye(3) * (sv * dt) ** 2
        pre.dropout = gap

    def add_node(self, pre, jac, lam, t):
        """A node after the last one, linked by the IMU; its initial state from the IMU propagation."""
        i = self.ids[-1]
        R_bi = self.R[i] @ self.R_cb
        dv = pre.dv + pre.J_v @ self.ba
        dp = pre.dp + pre.J_p @ self.ba
        dt = pre.dt
        p_bi = self.p[i] + self.R[i] @ self.t_cb
        v_j = self.v[i] + self.g * dt + R_bi @ dv
        p_bj = p_bi + self.v[i] * dt + 0.5 * self.g * dt ** 2 + R_bi @ dp
        R_j = R_bi @ pre.dR @ self.R_cb.T
        p_j = p_bj - R_j @ self.t_cb
        j = self._new_node(R_j, p_j, v_j, lam, t)
        cov = pre.cov + 1e-12 * np.eye(9)
        W = np.linalg.inv(np.linalg.cholesky(cov))
        self.imu.append(_Imu(i, dt, pre.dR, pre.dv, pre.dp, pre.J_v, pre.J_p, *jac[:3], self.bg.copy(), W,
                             self.t[i], float(t), **(jac[3] if len(jac) > 3 else {})))
        return j

    def repreintegrate(self, samples_of, td0=None):
        """Preintegrate every IMU factor of the window again (samples_of(t0, t1): raw samples on the camera clock, at
        time offset td0), after the camera-IMU time offset changed."""
        for k, f in enumerate(self.imu):
            samples = samples_of(f.t0, f.t1)
            if samples is None:
                continue
            pre, jac = self.preintegrate(samples, f.t0, f.t1, td0)
            W = np.linalg.inv(np.linalg.cholesky(pre.cov + 1e-12 * np.eye(9)))
            self.imu[k] = _Imu(f.i, pre.dt, pre.dR, pre.dv, pre.dp, pre.J_v, pre.J_p, *jac[:3], self.bg.copy(), W,
                               f.t0, f.t1, **(jac[3] if len(jac) > 3 else {}))

    def add_relative(self, a, b, s, R_ab, t_ab):
        if a in self.R and b in self.R and s in self.R:
            self.rel.append(_Rel(a, b, s, np.asarray(R_ab, dtype=np.float64), np.asarray(t_ab, dtype=np.float64)))

    def add_rotation(self, a, b, R_ab, std):
        """A relative rotation of the cameras (not subject to the passes' rotation scale)."""
        if a in self.R and b in self.R:
            self.rots.append(_Rot(a, b, np.asarray(R_ab, dtype=np.float64), float(std)))

    def add_metric_relative(self, a, b, R_ab, t_ab, cov):
        """A relative pose of the cameras in metres (b in a's frame) with its 6x6 covariance (rotation as a right
        perturbation of R_ab, translation in a's frame), not subject to the passes' scales."""
        if a in self.R and b in self.R:
            cov = 0.5 * (np.asarray(cov, dtype=np.float64) + np.asarray(cov, dtype=np.float64).T) + 1e-12 * np.eye(6)
            W = np.linalg.inv(np.linalg.cholesky(cov))
            self.met.append(_Met(a, b, _so3(np.asarray(R_ab, dtype=np.float64)), np.asarray(t_ab, dtype=np.float64), W))

    def add_link(self, j, k, log_ratio, std=None):
        if j in self.R and k in self.R and np.isfinite(log_ratio):
            self.links.append((j, k, float(log_ratio), float(std if std is not None else self.cfg.link_std)))

    def add_depth(self, i, log_obs, std):
        if i in self.R and np.isfinite(log_obs):
            self.depth[i] = (float(log_obs), max(float(std), self.cfg.depth_std_floor))

    def add_stereo(self, i, log_obs, std):
        """The metric scale of node i's pass from the stereo pair in it: lam_i = log_obs (std in log)."""
        if i in self.R and np.isfinite(log_obs) and np.isfinite(std) and std > 0:
            self.stereo[i] = (float(log_obs), float(std))

    # ------------------------------------------------------------------ the problem
    # The normal equations are assembled factor type by factor type: each factor's residual is a function of its own
    # few variables (an IMU factor: two nodes' rotation, position, velocity and the gravity and biases; a pass's
    # relative pose: two nodes' rotation and position, the scale of the pass and the rotation scale), its Jacobian is
    # analytic over those variables only (batched over the factors of the type), and J^T J and J^T r are scattered into
    # the dense system of the window.  The factors' constant data are stacked once per solve.

    def add_zero_rate(self, i, gyro_mean, std):
        """The gyro's mean over an interval ending at node i during which the platform was at rest: a direct
        measurement of the gyro bias (rad/s, IMU frame), independent of the visual rotations."""
        if i in self.R:
            self.zrate[i] = (np.asarray(gyro_mean, dtype=np.float64).copy(), np.asarray(std, dtype=np.float64).copy())

    def _prepare(self):
        """The factors' constant data as tensors, and their global columns (fixed during one solve).  Columns: node k at
        10 k (rotation 0:3, position 3:6, velocity 6:9, log scale 9), the globals at G = 10 n (gravity, gyro bias,
        accelerometer bias, then beta and kappa when estimated)."""
        cfg = self.cfg
        col = {i: k for k, i in enumerate(self.ids)}
        n = len(self.ids)
        G = 10 * n
        T = np.asarray
        P = {"col": col, "n": n, "G": G, "D": G + self.n_global}
        g9 = np.arange(G, G + 9)
        if self.imu:
            fs = self.imu
            ii = np.array([col[f.i] for f in fs])
            P["imu"] = dict(
                ii=ii, jj=ii + 1,
                dt=np.array([f.dt for f in fs]), dR=T(np.stack([f.dR for f in fs])), dv=T(np.stack([f.dv for f in fs])),
                dp=T(np.stack([f.dp for f in fs])), Jv=T(np.stack([f.J_v for f in fs])),
                Jp=T(np.stack([f.J_p for f in fs])), JRg=T(np.stack([f.JR_g for f in fs])),
                Jvg=T(np.stack([f.Jv_g for f in fs])), Jpg=T(np.stack([f.Jp_g for f in fs])),
                bgpre=T(np.stack([f.bg0 for f in fs])), W=T(np.stack([f.W for f in fs])),
                cols=np.concatenate([10 * ii[:, None] + np.arange(9)[None], 10 * (ii + 1)[:, None] + np.arange(9)[None],
                                     np.broadcast_to(g9, (len(fs), 9))]
                                    + ([np.full((len(fs), 1), G + self.i_td)] if cfg.time_offset else []), axis=1))
            if cfg.time_offset:
                z3 = np.zeros(3)
                P["imu"].update(JRt=T(np.stack([z3 if f.JR_t is None else f.JR_t for f in fs])),
                                Jvt=T(np.stack([z3 if f.Jv_t is None else f.Jv_t for f in fs])),
                                Jpt=T(np.stack([z3 if f.Jp_t is None else f.Jp_t for f in fs])),
                                td0=np.array([f.td0 for f in fs]))
        if self.rel:
            fs = self.rel
            aa = np.array([col[f.a] for f in fs])
            bb = np.array([col[f.b] for f in fs])
            ss = np.array([col[f.s] for f in fs])
            cols = [10 * aa[:, None] + np.arange(6)[None], 10 * bb[:, None] + np.arange(6)[None], (10 * ss + 9)[:, None]]
            if cfg.rot_scale:
                cols.append(np.full((len(fs), 1), G + self.i_kappa))
            Rm = np.stack([f.R for f in fs])
            rvm = _log_b(Rm)
            ang = np.linalg.norm(rvm, axis=1)
            P["rel"] = dict(
                aa=aa, bb=bb, ss=ss, Rm=Rm, rvm=rvm if cfg.rot_scale else None,
                tm=T(np.stack([f.t for f in fs])), tnorm=np.array([np.linalg.norm(f.t) for f in fs]),
                rot_sig=np.sqrt(cfg.rot_std ** 2 + (cfg.rot_rel * ang) ** 2),
                cols=np.concatenate(cols, axis=1))
        if self.rots:
            fs = self.rots
            aa = np.array([col[f.a] for f in fs])
            bb = np.array([col[f.b] for f in fs])
            P["rots"] = dict(aa=aa, bb=bb, Rm=T(np.stack([f.R for f in fs])), std=np.array([f.std for f in fs]),
                             cols=np.concatenate([10 * aa[:, None] + np.arange(3)[None],
                                                  10 * bb[:, None] + np.arange(3)[None]], axis=1))
        if self.met:
            fs = self.met
            aa = np.array([col[f.a] for f in fs])
            bb = np.array([col[f.b] for f in fs])
            P["met"] = dict(aa=aa, bb=bb, Rm=T(np.stack([f.R for f in fs])), tm=T(np.stack([f.t for f in fs])),
                            W=T(np.stack([f.W for f in fs])),
                            cols=np.concatenate([10 * aa[:, None] + np.arange(6)[None],
                                                 10 * bb[:, None] + np.arange(6)[None]], axis=1))
        if self.links:
            P["links"] = dict(jj=np.array([col[l[0]] for l in self.links]), kk=np.array([col[l[1]] for l in self.links]),
                              obs=np.array([l[2] for l in self.links]), std=np.array([l[3] for l in self.links]))
        zk = [i for i in self.zrate if i in col]
        if zk:
            P["zrate"] = dict(obs=np.stack([self.zrate[i][0] for i in zk]), std=np.stack([self.zrate[i][1] for i in zk]))
        dk = [i for i in self.depth if i in col]
        if dk:
            P["depth"] = dict(kk=np.array([col[i] for i in dk]), obs=np.array([self.depth[i][0] for i in dk]),
                              std=np.array([self.depth[i][1] for i in dk]))
        sk = [i for i in self.stereo if i in col]
        if sk:
            P["stereo"] = dict(kk=np.array([col[i] for i in sk]), obs=np.array([self.stereo[i][0] for i in sk]),
                               std=np.array([self.stereo[i][1] for i in sk]))
        return P

    def _blocks(self, P, jac=True):
        """Every factor type at the current estimate: (columns (n, k), unweighted residuals (n, m), Jacobians (n, m, k)
        or None, kind).  The Huber weights are applied by the caller."""
        cfg = self.cfg
        ids = self.ids
        R = np.stack([self.R[i] for i in ids])
        p = np.stack([self.p[i] for i in ids])
        v = np.stack([self.v[i] for i in ids])
        lam = np.array([self.lam[i] for i in ids])
        out = []
        if "imu" in P:
            q = P["imu"]
            ii, jj = q["ii"], q["jj"]
            r, J = _imu_block(R[ii], p[ii], v[ii], R[jj], p[jj], v[jj], q, self.g, self.bg, self.ba, self.R_cb, self.t_cb,
                              jac, self.td if cfg.time_offset else None)
            out.append((q["cols"], r, J, "imu"))
        if "rel" in P:
            q = P["rel"]
            aa, bb, ss = q["aa"], q["bb"], q["ss"]
            # translation noise in the pass's units, from the current scale (constant in the derivative, as before).
            # The relative part scales with the larger of the measured and the predicted translation: with the measured
            # one alone a translation the model under-reports claims to be precise in proportion (errors in variables
            # favour short measurements; KITTI 01: passes 7.7 m apart reporting 10-20 % of the motion dragged the speed
            # from 25 to 1 m/s while the IMU said constant velocity)
            tn = q["tnorm"]
            if cfg.trans_sigma_predicted:
                u = np.einsum("nji,nj->ni", R[aa], p[bb] - p[aa]) * np.exp(-lam[ss])[:, None]
                un = np.linalg.norm(u, axis=1)
                if cfg.trans_sigma_bound:
                    # the prediction only as far as it is known: its lower 2-sigma bound, the velocity's marginal std
                    # (newest node, last solve) over the pass interval (at a session start it is ~0)
                    tab = np.array([self.t[ids[b]] - self.t[ids[a]] for a, b in zip(aa, bb)])
                    un = np.maximum(un - 2.0 * self.v_std * tab * np.exp(-lam[ss]), 0.0)
                tn = np.maximum(tn, un)
            sig = np.sqrt((cfg.trans_rel * tn) ** 2 + (cfg.trans_floor * np.exp(-lam[ss])) ** 2)
            r, J = _rel_block(R[aa], p[aa], R[bb], p[bb], lam[ss], q, sig, self.kappa if cfg.rot_scale else None, jac)
            out.append((q["cols"], r, J, "rel"))
        if "rots" in P:
            q = P["rots"]
            r, J = _rot_block(R[q["aa"]], R[q["bb"]], q, jac)
            out.append((q["cols"], r, J, "rots"))
        if "met" in P:
            q = P["met"]
            r, J = _met_block(R[q["aa"]], p[q["aa"]], R[q["bb"]], p[q["bb"]], q, jac)
            out.append((q["cols"], r, J, "met"))
        if "links" in P:
            q = P["links"]
            r = ((lam[q["jj"]] - lam[q["kk"]] - q["obs"]) / q["std"])[:, None]
            J = np.stack([1.0 / q["std"], -1.0 / q["std"]], axis=1)[:, None, :]
            out.append((np.stack([10 * q["jj"] + 9, 10 * q["kk"] + 9], axis=1), r, J, "links"))
        G = P["G"]
        if "depth" in P:
            q = P["depth"]
            r = ((lam[q["kk"]] + (self.beta if cfg.depth_bias else 0.0) - q["obs"]) / q["std"])[:, None]
            cols, Js = [10 * q["kk"] + 9], [1.0 / q["std"]]
            if cfg.depth_bias:
                cols.append(np.full(len(q["kk"]), G + 9))
                Js.append(1.0 / q["std"])
            out.append((np.stack(cols, axis=1), r, np.stack(Js, axis=1)[:, None, :], "depth"))
        if "stereo" in P:
            q = P["stereo"]
            out.append(((10 * q["kk"] + 9)[:, None], ((lam[q["kk"]] - q["obs"]) / q["std"])[:, None],
                        (1.0 / q["std"])[:, None, None], "stereo"))
        if "zrate" in P:
            q = P["zrate"]
            n = len(q["obs"])
            out.append((np.broadcast_to(np.arange(G + 3, G + 6), (n, 3)), (self.bg[None] - q["obs"]) / q["std"],
                        (np.eye(3)[None] / q["std"][:, :, None]), "zrate"))
        # |g|
        gn = float(np.linalg.norm(self.g))
        out.append((np.arange(G, G + 3)[None], np.array([[(gn - GRAVITY) / cfg.gravity_norm_std]]),
                    (self.g / gn / cfg.gravity_norm_std)[None, None, :], "gnorm"))
        # priors
        nb, nk, nt = int(cfg.depth_bias), int(cfg.rot_scale), int(cfg.time_offset)
        if self.prior is None:
            i0 = ids[0]
            r_rot, J_rot = _prior_rotation(self.gauge[0], self.R[i0], 1e-3)
            rows = [r_rot, (self.p[i0] - self.gauge[1]) / 1e-3, (self.g - self.g0) / cfg.gravity_std,
                    (self.bg - self.bg0) / cfg.gyro_bias_std, self.ba / cfg.accel_bias_std]
            J = np.zeros((15 + nb + nk + nt, 15 + nb + nk + nt))
            J[0:3, 0:3] = J_rot
            J[3:6, 3:6] = np.eye(3) / 1e-3
            J[6:9, 6:9] = np.eye(3) / cfg.gravity_std
            J[9:12, 9:12] = np.eye(3) / cfg.gyro_bias_std
            J[12:15, 12:15] = np.eye(3) / cfg.accel_bias_std
            cols = list(range(10 * P["col"][i0], 10 * P["col"][i0] + 6)) + list(range(G, G + 9))
            if nb:
                rows.append([self.beta / cfg.depth_bias_std])
                J[15, 15] = 1.0 / cfg.depth_bias_std
                cols.append(G + 9)
            if nk:
                rows.append([self.kappa / cfg.rot_scale_std])
                J[15 + nb, 15 + nb] = 1.0 / cfg.rot_scale_std
                cols.append(G + self.i_kappa)
            if nt:
                rows.append([(self.td - self.td_init) / cfg.time_offset_std])
                J[15 + nb + nk, 15 + nb + nk] = 1.0 / cfg.time_offset_std
                cols.append(G + self.i_td)
            out.append((np.array(cols)[None], np.concatenate(rows)[None], J[None], "prior"))
        else:
            L, i0, xl = self.prior
            k0 = P["col"][i0]
            r_rot, J_rot = _prior_rotation(xl["R"], self.R[i0], 1.0)
            parts = [r_rot, self.p[i0] - xl["p"], self.v[i0] - xl["v"], [self.lam[i0] - xl["lam"]],
                     self.g - xl["g"], self.bg - xl["bg"], self.ba - xl["ba"]]
            if nb:
                parts.append([self.beta - xl["beta"]])
            if nk:
                parts.append([self.kappa - xl["kappa"]])
            if nt:
                parts.append([self.td - xl["td"]])
            d = np.concatenate(parts)
            Jd = np.eye(len(d))
            Jd[0:3, 0:3] = J_rot
            cols = list(range(10 * k0, 10 * k0 + 10)) + list(range(G, G + self.n_global))
            out.append((np.array(cols)[None], (L.T @ d)[None], (L.T @ Jd)[None], "prior"))
        return out

    def _huber(self, blocks):
        """Huber weights of the relative-pose, rotation-only, stereo-scale and gauge-link factors from their unweighted
        residuals (a gauge link is a visual measurement too: a pass whose depth of the shared frame jumps by 30-40 % gave
        links of 10-15 sigma at full weight)."""
        h = self.cfg.huber
        w = {}
        for _, r, _, kind in blocks:
            if kind in ("rel", "rots", "zrate", "stereo", "met") or (kind == "links" and self.cfg.robust_links):
                e = np.linalg.norm(r, axis=1) / np.sqrt(r.shape[1])
                w[kind] = np.where(e <= h, 1.0, np.sqrt(h / np.maximum(e, 1e-12)))
        return w

    @staticmethod
    def _weighted(blocks, w):
        out = []
        for cols, r, J, kind in blocks:
            if kind in w:
                r = r * w[kind][:, None]
                J = None if J is None else J * w[kind][:, None, None]
            out.append((cols, r, J, kind))
        return out

    @staticmethod
    def _assemble(blocks, D):
        H = np.zeros((D, D))
        grad = np.zeros(D)
        cost, n_res = 0.0, 0
        Hf = H.reshape(-1)
        for cols, r, J, _ in blocks:
            cost += float((r * r).sum())
            n_res += r.size
            np.add.at(Hf, (cols[:, :, None] * D + cols[:, None, :]).reshape(-1),
                      np.einsum("nrk,nrl->nkl", J, J).reshape(-1))
            np.add.at(grad, cols.reshape(-1), np.einsum("nrk,nr->nk", J, r).reshape(-1))
        return H, grad, cost, n_res

    def _system(self, P):
        """(H, gradient, cost, number of residuals, Huber weights) at the current estimate."""
        blocks = self._blocks(P, jac=True)
        w = self._huber(blocks)
        H, grad, cost, n_res = self._assemble(self._weighted(blocks, w), P["D"])
        return H, grad, cost, n_res, w

    def _cost(self, P, w):
        return float(sum((r * r).sum() for _, r, _, _ in self._weighted(self._blocks(P, jac=False), w)))

    def _apply(self, d):
        n = len(self.ids)
        dn = d[:10 * n].reshape(n, 10)
        for k, i in enumerate(self.ids):
            self.R[i] = _so3(self.R[i] @ _exp(dn[k, 0:3]))
            self.p[i] = self.p[i] + dn[k, 3:6]
            self.v[i] = self.v[i] + dn[k, 6:9]
            self.lam[i] = self.lam[i] + dn[k, 9]
        G = 10 * n
        self.g = self.g + d[G:G + 3]
        self.bg = self.bg + d[G + 3:G + 6]
        self.ba = self.ba + d[G + 6:G + 9]
        if self.cfg.depth_bias:
            self.beta = self.beta + d[G + 9]
        if self.cfg.rot_scale:
            self.kappa = self.kappa + d[G + self.i_kappa]
        if self.cfg.time_offset:
            self.td = self.td + d[G + self.i_td]

    def solve(self, iterations=None, need_std=True):
        """Gauss-Newton (Levenberg damping); need_std: the marginal std of the newest node's log scale."""
        damping = 1e-6
        info = {}
        H = None
        P = self._prepare()
        for _ in range(iterations or self.cfg.iterations):
            H, gr, cost, n_res, w = self._system(P)
            step = np.linalg.solve(H + damping * np.diag(np.diag(H) + 1e-9), -gr)
            saved = self._save()
            self._apply(step)
            c_new = self._cost(P, w)
            if c_new <= cost:
                damping = max(damping / 10, 1e-9)
                info = {"cost": c_new, "residuals": n_res}
            else:
                self._restore(saved)
                damping *= 10
                info = {"cost": cost, "residuals": n_res}
        info["lam_std"] = float("nan")
        if self.cfg.debug_costs:
            w = self._huber(self._blocks(P, jac=False))
            info["costs"] = {}
            for _, r, _, kind in self._weighted(self._blocks(P, jac=False), w):
                info["costs"][kind] = round(info["costs"].get(kind, 0.0) + float((r * r).sum()), 2)
        if need_std and H is not None:
            try:
                k = 10 * (len(self.ids) - 1)
                E = np.zeros((len(H), 4))
                E[[k + 9, k + 6, k + 7, k + 8], [0, 1, 2, 3]] = 1.0
                X = np.linalg.solve(H + 1e-12 * np.eye(len(H)), E)
                info["lam_std"] = float(np.sqrt(max(X[k + 9, 0], 0.0)))
                self.v_std = float(np.sqrt(max((X[k + 6, 1] + X[k + 7, 2] + X[k + 8, 3]) / 3.0, 0.0)))
            except np.linalg.LinAlgError:
                info["lam_std"] = float("inf")
                self.v_std = np.inf
        self.last_info = info
        return info

    def _save(self):
        return ({i: self.R[i].copy() for i in self.ids}, {i: self.p[i].copy() for i in self.ids},
                {i: self.v[i].copy() for i in self.ids}, dict(self.lam), self.g.copy(), self.bg.copy(), self.ba.copy(),
                self.beta, self.kappa, self.td)

    def _restore(self, s):
        self.R.update(s[0])
        self.p.update(s[1])
        self.v.update(s[2])
        self.lam.update(s[3])
        self.g, self.bg, self.ba, self.beta, self.kappa, self.td = s[4], s[5], s[6], s[7], s[8], s[9]

    # ------------------------------------------------------------------ marginalization
    def marginalize(self):
        """Drop the oldest node: its IMU factor, its relative poses with the next node, its gauge link, learned depth
        and stereo scale, and the current prior become a Gaussian prior on the next node and the globals (Schur
        complement); its relative poses with later nodes are dropped."""
        if len(self.ids) <= self.cfg.window:
            return
        i0, i1 = self.ids[0], self.ids[1]
        keep_rel = [f for f in self.rel if i0 not in (f.a, f.b, f.s)]
        mar_rel = [f for f in self.rel if (i0 in (f.a, f.b, f.s)) and {f.a, f.b, f.s} <= {i0, i1}]
        keep_rots = [f for f in self.rots if i0 not in (f.a, f.b)]
        mar_rots = [f for f in self.rots if (i0 in (f.a, f.b)) and {f.a, f.b} <= {i0, i1}]
        keep_met = [f for f in self.met if i0 not in (f.a, f.b)]
        mar_met = [f for f in self.met if (i0 in (f.a, f.b)) and {f.a, f.b} <= {i0, i1}]
        keep_links = [l for l in self.links if i0 not in (l[0], l[1])]
        mar_links = [l for l in self.links if (i0 in (l[0], l[1])) and {l[0], l[1]} <= {i0, i1}]
        # the factors involving only nodes 0 and 1 (and the globals), on a two-node sub-problem
        full = (self.ids, self.imu, self.rel, self.links, self.depth, self.zrate, self.stereo)
        self.ids = [i0, i1]
        self.imu = [f for f in self.imu if f.i == i0]
        self.rel = mar_rel
        self.rots = mar_rots
        self.met = mar_met
        self.links = mar_links
        self.depth = {i: o for i, o in full[4].items() if i == i0}
        self.zrate = {i: o for i, o in full[5].items() if i == i0}
        self.stereo = {i: o for i, o in full[6].items() if i == i0}
        H, c, _, _, _ = self._system(self._prepare())
        mi = np.arange(10)
        ri = np.arange(10, 20 + self.n_global)
        K = H[np.ix_(ri, mi)] @ np.linalg.inv(H[np.ix_(mi, mi)] + 1e-9 * np.eye(10))
        Hp = H[np.ix_(ri, ri)] - K @ H[np.ix_(mi, ri)]
        cp = c[ri] - K @ c[mi]
        Hp = 0.5 * (Hp + Hp.T) + 1e-9 * np.eye(len(ri))
        dt = self.t[i1] - self.t[i0]
        # the minimum of the quadratic as the prior's mean, with the random walks of the globals added
        mean_shift = -np.linalg.solve(Hp, cp)
        cov = np.linalg.inv(Hp)
        cfg = self.cfg
        cov[10:13, 10:13] += np.eye(3) * cfg.gravity_drift ** 2 * dt
        cov[13:16, 13:16] += np.eye(3) * cfg.gyro_bias_walk ** 2 * dt
        cov[16:19, 16:19] += np.eye(3) * cfg.accel_bias_walk ** 2 * dt
        if cfg.depth_bias:
            cov[19, 19] += cfg.depth_bias_drift ** 2 * dt
        if cfg.rot_scale:
            k = 10 + self.i_kappa
            cov[k, k] += cfg.rot_scale_drift ** 2 * dt
        if cfg.time_offset:
            k = 10 + self.i_td
            cov[k, k] += cfg.time_offset_drift ** 2 * dt
        # square root of the information (L L^T = cov^-1) by an eigendecomposition: a Cholesky factor fails when the
        # matrix is barely positive definite (KITTI 07: positions of hundreds of metres)
        w, V = np.linalg.eigh(0.5 * (cov + cov.T))
        w = np.maximum(w, 1e-12 * max(float(w.max()), 1e-12))
        L = V / np.sqrt(w)[None, :]
        # linearization point moved to the minimum (tangent-space shift of node 1 and the globals)
        xl = {"R": _so3(self.R[i1] @ _exp(mean_shift[0:3])), "p": self.p[i1] + mean_shift[3:6],
              "v": self.v[i1] + mean_shift[6:9], "lam": self.lam[i1] + mean_shift[9],
              "g": self.g + mean_shift[10:13], "bg": self.bg + mean_shift[13:16], "ba": self.ba + mean_shift[16:19],
              "beta": self.beta + (mean_shift[19] if cfg.depth_bias else 0.0),
              "kappa": self.kappa + (mean_shift[10 + self.i_kappa] if cfg.rot_scale else 0.0),
              "td": self.td + (mean_shift[10 + self.i_td] if cfg.time_offset else 0.0)}
        # restore the window without node 0
        self.ids = full[0][1:]
        self.imu = [f for f in full[1] if f.i != i0]
        self.rel = keep_rel
        self.rots = keep_rots
        self.met = keep_met
        self.links = keep_links
        self.depth = {i: o for i, o in full[4].items() if i != i0}
        self.zrate = {i: o for i, o in full[5].items() if i != i0}
        self.stereo = {i: o for i, o in full[6].items() if i != i0}
        self.prior = (L, i1, xl)
        for dct in (self.R, self.p, self.v, self.lam, self.t):
            dct.pop(i0, None)

    # ------------------------------------------------------------------ state
    def latest(self):
        i = self.ids[-1]
        return i, self.R[i], self.p[i], self.v[i], self.lam[i]
