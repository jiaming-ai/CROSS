"""Local pose graph of visual-inertial odometry with feed-forward relative poses (VGGT-Omega) and an IMU.

A sliding window of nodes, one per visual measurement (a forward pass of the geometry model on the current frame and
some earlier nodes' frames).  Unknowns:

    per node i     R_i, p_i   camera orientation and position in the world (metres)
                   v_i        velocity of the IMU (world)
                   lam_i      log metres per unit of node i's forward pass (each pass has a gauge of its own)
    global         g          gravity (world), |g| = 9.81
                   b_g, b_a   gyroscope and accelerometer biases (constant in the window, random walk on marginalization)
                   beta       log bias of learned metric depth (optional)

Factors:

    IMU            preintegration between consecutive nodes, with first-order bias corrections (gyro-bias Jacobians by
                   finite differences)
    relative pose  every pair of views of a pass: rotation, and translation = exp(lam_pass) * the pass's translation
                   (Huber)
    gauge link     lam_j - lam_k = log of the depth ratio of a frame both passes contain
    learned depth  lam_i + beta = log(learned metric depth / pass depth) of node i's frame
    priors         |g|; start-up priors on the first node (gauge) and the biases; the marginalization prior

Gauss-Newton on the stacked tangent-space increment; Jacobians by forward-mode automatic differentiation (torch, CPU,
float64).  The oldest node is marginalized by the Schur complement into a prior on the next node and the globals (its
relative-pose factors with later nodes are dropped)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
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


def _skew_t(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def _exp_t(phi):
    """SO(3) exponential (batched), differentiable at 0."""
    t2 = (phi * phi).sum(-1)[..., None, None]
    small = t2 < 1e-6
    t2s = torch.where(small, torch.ones_like(t2), t2)
    t = torch.sqrt(t2s)
    A = torch.where(small, 1 - t2 / 6 + t2 * t2 / 120, torch.sin(t) / t)
    B = torch.where(small, 0.5 - t2 / 24 + t2 * t2 / 720, (1 - torch.cos(t)) / t2s)
    K = _skew_t(phi)
    eye = torch.eye(3, dtype=phi.dtype).expand(K.shape)
    return eye + A * K + B * (K @ K)


def _log_t(R):
    """SO(3) logarithm (batched), differentiable near the identity."""
    s = 0.5 * torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    c = 0.5 * (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1)
    sn2 = (s * s).sum(-1)
    small = sn2 < 1e-8
    sn2s = torch.where(small, torch.ones_like(sn2), sn2)
    sn = torch.sqrt(sn2s)
    f = torch.where(small, 1 + sn2 / 6, torch.atan2(sn, c) / sn)
    return f[..., None] * s


def _exp(phi):
    return Rotation.from_rotvec(phi).as_matrix()


def _log(R):
    return Rotation.from_matrix(R).as_rotvec()


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
        self.R_cb = np.asarray(T_cam_imu, dtype=np.float64)[:3, :3]
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
        self.imu: list[_Imu] = []
        self.rel: list[_Rel] = []
        self.links: list[tuple] = []             # (j, k, log ratio, std): lam_j - lam_k = log ratio
        self.depth: dict[int, tuple] = {}        # node -> (log observation, std)
        self.prior = None                        # (L, node id, x_lin) on (node, globals)
        self.gauge = None
        self.next_id = 0
        self.last_info = {}

    # ------------------------------------------------------------------ building
    @property
    def n_global(self):
        return 9 + int(self.cfg.depth_bias)

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
        self.R[i], self.p[i], self.v[i], self.lam[i], self.t[i] = R.copy(), p.copy(), v.copy(), float(lam), float(t)
        return i

    def preintegrate(self, samples, t0, t1):
        """IMU of an interval (samples: raw rows t w a on the camera clock) at the current gyro bias, with the
        gyro-bias Jacobians by finite differences."""
        h = float(np.median(np.diff(samples[:, 0]))) if len(samples) > 1 else 0.0
        gn = float(np.hypot(self.gyro_noise, self.cfg.gyro_dt_noise * h))
        an = float(np.hypot(self.accel_noise, self.cfg.accel_dt_noise * h))

        def run(bg):
            s = samples.copy()
            s[:, 1:4] -= bg
            return preintegrate(s, t0, t1, gn, an)
        base = run(self.bg)
        eps = 1e-4
        JR, Jv, Jp = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
        for k in range(3):
            d = np.zeros(3)
            d[k] = eps
            pk = run(self.bg + d)
            JR[:, k] = _log(base.dR.T @ pk.dR) / eps
            Jv[:, k] = (pk.dv - base.dv) / eps
            Jp[:, k] = (pk.dp - base.dp) / eps
        return base, (JR, Jv, Jp)

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
        self.imu.append(_Imu(i, dt, pre.dR, pre.dv, pre.dp, pre.J_v, pre.J_p, *jac, self.bg.copy(), W,
                             self.t[i], float(t)))
        return j

    def repreintegrate(self, samples_of):
        """Preintegrate every IMU factor of the window again (samples_of(t0, t1): raw samples on the camera clock),
        after the camera-IMU time offset changed."""
        for k, f in enumerate(self.imu):
            samples = samples_of(f.t0, f.t1)
            if samples is None:
                continue
            pre, jac = self.preintegrate(samples, f.t0, f.t1)
            W = np.linalg.inv(np.linalg.cholesky(pre.cov + 1e-12 * np.eye(9)))
            self.imu[k] = _Imu(f.i, pre.dt, pre.dR, pre.dv, pre.dp, pre.J_v, pre.J_p, *jac, self.bg.copy(), W, f.t0, f.t1)

    def add_relative(self, a, b, s, R_ab, t_ab):
        if a in self.R and b in self.R and s in self.R:
            self.rel.append(_Rel(a, b, s, np.asarray(R_ab, dtype=np.float64), np.asarray(t_ab, dtype=np.float64)))

    def add_link(self, j, k, log_ratio, std=None):
        if j in self.R and k in self.R and np.isfinite(log_ratio):
            self.links.append((j, k, float(log_ratio), float(std if std is not None else self.cfg.link_std)))

    def add_depth(self, i, log_obs, std):
        if i in self.R and np.isfinite(log_obs):
            self.depth[i] = (float(log_obs), max(float(std), self.cfg.depth_std_floor))

    # ------------------------------------------------------------------ the problem
    def _residuals(self, delta, weights=None):
        """Stacked whitened residuals at the current estimate perturbed by delta (torch, float64)."""
        cfg = self.cfg
        ids = self.ids
        n = len(ids)
        col = {i: k for k, i in enumerate(ids)}
        d = delta
        dn = d[:10 * n].reshape(n, 10)
        G = 10 * n
        dg, dbg, dba = d[G:G + 3], d[G + 3:G + 6], d[G + 6:G + 9]
        dbeta = d[G + 9] if cfg.depth_bias else None
        T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)
        R0 = T(np.stack([self.R[i] for i in ids]))
        R = R0 @ _exp_t(dn[:, 0:3])
        p = T(np.stack([self.p[i] for i in ids])) + dn[:, 3:6]
        v = T(np.stack([self.v[i] for i in ids])) + dn[:, 6:9]
        lam = T(np.array([self.lam[i] for i in ids])) + dn[:, 9]
        g = T(self.g) + dg
        bg = T(self.bg) + dbg
        ba = T(self.ba) + dba
        beta = (self.beta + dbeta) if cfg.depth_bias else None
        Rcb, tcb = T(self.R_cb), T(self.t_cb)
        out = []
        # IMU
        if self.imu:
            ii = torch.as_tensor([col[f.i] for f in self.imu])
            jj = ii + 1
            dt = T([f.dt for f in self.imu])[:, None]
            Rbi, Rbj = R[ii] @ Rcb, R[jj] @ Rcb
            pbi = p[ii] + (R[ii] @ tcb)
            pbj = p[jj] + (R[jj] @ tcb)
            dbg_i = bg[None] - T(np.stack([f.bg0 for f in self.imu]))
            dR = T(np.stack([f.dR for f in self.imu])) @ _exp_t((T(np.stack([f.JR_g for f in self.imu])) @ dbg_i[..., None])[..., 0])
            dv = (T(np.stack([f.dv for f in self.imu])) + (T(np.stack([f.J_v for f in self.imu])) @ ba)
                  + (T(np.stack([f.Jv_g for f in self.imu])) @ dbg_i[..., None])[..., 0])
            dp = (T(np.stack([f.dp for f in self.imu])) + (T(np.stack([f.J_p for f in self.imu])) @ ba)
                  + (T(np.stack([f.Jp_g for f in self.imu])) @ dbg_i[..., None])[..., 0])
            rR = _log_t(dR.transpose(-1, -2) @ Rbi.transpose(-1, -2) @ Rbj)
            rv = (Rbi.transpose(-1, -2) @ (v[jj] - v[ii] - g * dt)[..., None])[..., 0] - dv
            rp = (Rbi.transpose(-1, -2) @ (pbj - pbi - v[ii] * dt - 0.5 * g * dt ** 2)[..., None])[..., 0] - dp
            r9 = torch.cat([rR, rv, rp], -1)
            W = T(np.stack([f.W for f in self.imu]))
            out.append((W @ r9[..., None])[..., 0].reshape(-1))
        # relative poses
        if self.rel:
            aa = torch.as_tensor([col[f.a] for f in self.rel])
            bb = torch.as_tensor([col[f.b] for f in self.rel])
            ss = torch.as_tensor([col[f.s] for f in self.rel])
            Rm = T(np.stack([f.R for f in self.rel]))
            tm = T(np.stack([f.t for f in self.rel]))
            ang = T(np.array([np.linalg.norm(_log(f.R)) for f in self.rel]))[:, None]
            rot_sig = torch.sqrt(cfg.rot_std ** 2 + (cfg.rot_rel * ang) ** 2)
            rr = _log_t(Rm.transpose(-1, -2) @ R[aa].transpose(-1, -2) @ R[bb]) / rot_sig
            # in the pass's units, where its noise is: the metric displacement over the pass's metres per unit (a
            # residual in metres would shrink the scale with the noise, errors in variables)
            rel_t = (R[aa].transpose(-1, -2) @ (p[bb] - p[aa])[..., None])[..., 0]
            sig = T(self._trans_sigma())[:, None]
            rt = (torch.exp(-lam[ss])[:, None] * rel_t - tm) / sig
            w = T(weights if weights is not None else np.ones(len(self.rel)))[:, None]
            out.append((w * torch.cat([rr, rt], -1)).reshape(-1))
        # gauge links
        if self.links:
            jj = torch.as_tensor([col[l[0]] for l in self.links])
            kk = torch.as_tensor([col[l[1]] for l in self.links])
            out.append((lam[jj] - lam[kk] - T([l[2] for l in self.links])) / T([l[3] for l in self.links]))
        # learned depth
        dk = [i for i in self.depth if i in col]
        if dk:
            idx = torch.as_tensor([col[i] for i in dk])
            o = T([self.depth[i][0] for i in dk])
            sd = T([self.depth[i][1] for i in dk])
            out.append((lam[idx] + (beta if beta is not None else 0.0) - o) / sd)
        # gravity norm
        out.append(((torch.linalg.norm(g) - GRAVITY) / cfg.gravity_norm_std)[None])
        # priors
        if self.prior is None:
            k0 = 0
            out.append(_log_t(T(self.gauge[0]).T @ R[k0]) / 1e-3)
            out.append((p[k0] - T(self.gauge[1])) / 1e-3)
            out.append((g - T(self.g0)) / cfg.gravity_std)
            out.append((bg - T(self.bg0)) / cfg.gyro_bias_std)
            out.append(ba / cfg.accel_bias_std)
            if beta is not None:
                out.append((beta / cfg.depth_bias_std)[None])
        else:
            L, i0, xl = self.prior
            k0 = col[i0]
            parts = [_log_t(T(xl["R"]).T @ R[k0]), p[k0] - T(xl["p"]), v[k0] - T(xl["v"]), (lam[k0] - xl["lam"])[None],
                     g - T(xl["g"]), bg - T(xl["bg"]), ba - T(xl["ba"])]
            if beta is not None:
                parts.append((beta - xl["beta"])[None])
            out.append(T(L).T @ torch.cat(parts))
        return torch.cat(out)

    def _trans_sigma(self):
        """Translation noise of each pass in its units: relative, with a floor of trans_floor metres."""
        cfg = self.cfg
        return np.array([np.hypot(cfg.trans_rel * np.linalg.norm(f.t), cfg.trans_floor * np.exp(-self.lam[f.s]))
                         for f in self.rel])

    def _rel_weights(self):
        """Huber weights of the relative-pose factors at the current estimate."""
        if not self.rel:
            return None
        with torch.no_grad():
            zero = torch.zeros(10 * len(self.ids) + self.n_global, dtype=torch.float64)
            r = self._residuals(zero)
        n_imu = 9 * len(self.imu)
        rr = r[n_imu:n_imu + 6 * len(self.rel)].reshape(-1, 6).numpy()
        e = np.linalg.norm(rr, axis=1) / np.sqrt(6)
        return np.where(e <= self.cfg.huber, 1.0, np.sqrt(self.cfg.huber / np.maximum(e, 1e-12)))

    def _apply(self, d):
        n = len(self.ids)
        dn = d[:10 * n].reshape(n, 10)
        for k, i in enumerate(self.ids):
            self.R[i] = self.R[i] @ _exp(dn[k, 0:3])
            self.p[i] = self.p[i] + dn[k, 3:6]
            self.v[i] = self.v[i] + dn[k, 6:9]
            self.lam[i] = self.lam[i] + dn[k, 9]
        G = 10 * n
        self.g = self.g + d[G:G + 3]
        self.bg = self.bg + d[G + 3:G + 6]
        self.ba = self.ba + d[G + 6:G + 9]
        if self.cfg.depth_bias:
            self.beta = self.beta + d[G + 9]

    def _system(self, weights):
        D = 10 * len(self.ids) + self.n_global
        zero = torch.zeros(D, dtype=torch.float64)
        f = lambda x: self._residuals(x, weights)
        J = torch.func.jacfwd(f)(zero).numpy()
        r = f(zero).detach().numpy()
        return J, r

    def _cost(self, weights):
        with torch.no_grad():
            r = self._residuals(torch.zeros(10 * len(self.ids) + self.n_global, dtype=torch.float64), weights)
        return float((r * r).sum())

    def solve(self, iterations=None, need_std=True):
        """Gauss-Newton (Levenberg damping); need_std: the marginal std of the newest node's log scale."""
        damping = 1e-6
        info = {}
        H = None
        for _ in range(iterations or self.cfg.iterations):
            w = self._rel_weights()
            J, r = self._system(w)
            H, gr = J.T @ J, J.T @ r
            cost = float(r @ r)
            step = np.linalg.solve(H + damping * np.diag(np.diag(H) + 1e-9), -gr)
            saved = self._save()
            self._apply(step)
            c_new = self._cost(w)
            if c_new <= cost:
                damping = max(damping / 10, 1e-9)
                info = {"cost": c_new, "residuals": len(r)}
            else:
                self._restore(saved)
                damping *= 10
                info = {"cost": cost, "residuals": len(r)}
        info["lam_std"] = float("nan")
        if need_std and H is not None:
            try:
                k = 10 * (len(self.ids) - 1) + 9
                e = np.zeros(len(H))
                e[k] = 1.0
                info["lam_std"] = float(np.sqrt(max(np.linalg.solve(H + 1e-12 * np.eye(len(H)), e)[k], 0.0)))
            except np.linalg.LinAlgError:
                info["lam_std"] = float("inf")
        self.last_info = info
        return info

    def _save(self):
        return ({i: self.R[i].copy() for i in self.ids}, {i: self.p[i].copy() for i in self.ids},
                {i: self.v[i].copy() for i in self.ids}, dict(self.lam), self.g.copy(), self.bg.copy(), self.ba.copy(),
                self.beta)

    def _restore(self, s):
        self.R.update(s[0])
        self.p.update(s[1])
        self.v.update(s[2])
        self.lam.update(s[3])
        self.g, self.bg, self.ba, self.beta = s[4], s[5], s[6], s[7]

    # ------------------------------------------------------------------ marginalization
    def marginalize(self):
        """Drop the oldest node: its IMU factor, its relative poses with the next node, its gauge link and learned
        depth, and the current prior become a Gaussian prior on the next node and the globals (Schur complement);
        its relative poses with later nodes are dropped."""
        if len(self.ids) <= self.cfg.window:
            return
        i0, i1 = self.ids[0], self.ids[1]
        keep_rel = [f for f in self.rel if i0 not in (f.a, f.b, f.s)]
        mar_rel = [f for f in self.rel if (i0 in (f.a, f.b, f.s)) and {f.a, f.b, f.s} <= {i0, i1}]
        keep_links = [l for l in self.links if i0 not in (l[0], l[1])]
        mar_links = [l for l in self.links if (i0 in (l[0], l[1])) and {l[0], l[1]} <= {i0, i1}]
        # the factors involving only nodes 0 and 1 (and the globals), on a two-node sub-problem
        full = (self.ids, self.imu, self.rel, self.links, self.depth)
        self.ids = [i0, i1]
        self.imu = [f for f in self.imu if f.i == i0]
        self.rel = mar_rel
        self.links = mar_links
        self.depth = {i: o for i, o in full[4].items() if i == i0}
        w = self._rel_weights()
        J, r = self._system(w)
        H, c = J.T @ J, J.T @ r
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
        # square root of the information (L L^T = cov^-1) by an eigendecomposition: a Cholesky factor fails when the
        # matrix is barely positive definite (KITTI 07: positions of hundreds of metres)
        w, V = np.linalg.eigh(0.5 * (cov + cov.T))
        w = np.maximum(w, 1e-12 * max(float(w.max()), 1e-12))
        L = V / np.sqrt(w)[None, :]
        # linearization point moved to the minimum (tangent-space shift of node 1 and the globals)
        xl = {"R": self.R[i1] @ _exp(mean_shift[0:3]), "p": self.p[i1] + mean_shift[3:6],
              "v": self.v[i1] + mean_shift[6:9], "lam": self.lam[i1] + mean_shift[9],
              "g": self.g + mean_shift[10:13], "bg": self.bg + mean_shift[13:16], "ba": self.ba + mean_shift[16:19],
              "beta": self.beta + (mean_shift[19] if cfg.depth_bias else 0.0)}
        # restore the window without node 0
        self.ids = full[0][1:]
        self.imu = [f for f in full[1] if f.i != i0]
        self.rel = keep_rel
        self.links = keep_links
        self.depth = {i: o for i, o in full[4].items() if i != i0}
        self.prior = (L, i1, xl)
        for dct in (self.R, self.p, self.v, self.lam, self.t):
            dct.pop(i0, None)

    # ------------------------------------------------------------------ state
    def latest(self):
        i = self.ids[-1]
        return i, self.R[i], self.p[i], self.v[i], self.lam[i]
