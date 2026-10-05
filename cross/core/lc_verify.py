"""Verified loop closure.

A loop closure is a visual measurement (relative pose between the current keyframe and an older keyframe) that
is *more informative than the graph's own estimate* of that relative pose.  Instead of deciding with counts, windows
and cooldowns, every candidate measurement is subjected to consistency tests that share one decision parameter,
the chi-square confidence level (6 degrees of freedom):

  1. in-pass consistency  -- the references registered together in one multi-view forward pass must agree with the
                             map about their mutual relative poses; a reference the model placed at the wrong place
                             (perceptual aliasing) contradicts the others by metres and is dropped before it can
                             become a proposal or an edge;
  2. prior consistency    -- the measurement must be consistent with the dead-reckoned relative pose along the
                             odometry chain between the two keyframes (or, in a relocalization session, along the
                             chain from the last keyframe anchored to the map), with the covariance of that chain
                             compounded through the adjoints;
  3. posterior consistency -- after the pose-graph optimisation the new measurements must fit the optimised graph;
                             measurements that remain outliers are removed and the graph is re-optimised.

All tests, and the pose-graph optimisation itself, use one calibrated noise model:
    visual edge   sigma = a + b * |t|          (fitted on ground truth: a few cm plus 1.5-2 % of the distance)
    odometry edge sigma = k * motion / sqrt(n) + floor   (noise proportional to every integrated reading)
    reference pair (same pass)   sigma = a + b * |t|
    stored map (relative pose of two map keyframes) sigma = a + b * |t|
Conventions: gtsam order (rotation, translation) inside this module; pypose order (tx ty tz rx ry rz) at the
boundaries with the rest of CROSS.
"""
from __future__ import annotations

import collections
import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import gtsam
import numpy as np
from loguru import logger
from scipy.stats import chi2 as _chi2

from cross.core.config import LoopClosureConfig, NoiseModelConfig
from cross.core.types import EdgeType


# ----------------------------------------------------------------------------- Lie helpers
def to_gtsam(x) -> gtsam.Pose3:
    """pypose SE3 / (7,) tensor or array [x y z qx qy qz qw] / (4,4) matrix / Edge -> gtsam.Pose3."""
    if isinstance(x, gtsam.Pose3):
        return x
    if hasattr(x, "mean_np"):
        x = x.mean_np
    if hasattr(x, "tensor"):
        x = x.tensor()
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-2:] == (4, 4):
        return gtsam.Pose3(gtsam.Rot3(x[:3, :3]), gtsam.Point3(x[0, 3], x[1, 3], x[2, 3]))
    x = x.reshape(-1)
    q = x[3:7] / max(np.linalg.norm(x[3:7]), 1e-12)
    return gtsam.Pose3(gtsam.Rot3.Quaternion(q[3], q[0], q[1], q[2]), gtsam.Point3(x[0], x[1], x[2]))


def logmap(T: gtsam.Pose3) -> np.ndarray:
    return np.asarray(gtsam.Pose3.Logmap(T), dtype=np.float64)


def residual(T_meas: gtsam.Pose3, T_pred: gtsam.Pose3) -> np.ndarray:
    """Right-perturbation residual Log(T_meas^-1 T_pred), gtsam order (rot, trans)."""
    return logmap(T_meas.between(T_pred))


def chi2_of(r: np.ndarray, cov: np.ndarray) -> float:
    try:
        return float(r @ np.linalg.solve(cov, r))
    except np.linalg.LinAlgError:
        return float("inf")


def chi2_threshold(confidence: float, dof: int = 6) -> float:
    return float(_chi2.ppf(confidence, dof))


def transport(cov: np.ndarray, T: gtsam.Pose3) -> np.ndarray:
    """Covariance of a right perturbation on X expressed as a right perturbation on X T (i.e. after composing T)."""
    A = T.inverse().AdjointMap()
    return A @ cov @ A.T


def rt_to_pypose(s: np.ndarray) -> np.ndarray:
    return np.array([s[3], s[4], s[5], s[0], s[1], s[2]], dtype=np.float64)


P90_3DOF = math.sqrt(_chi2.ppf(0.90, 3))   # 90th percentile of |e| for an isotropic 3-dof Gaussian = 2.5 sigma


def informative(cov: np.ndarray, sv: np.ndarray, rotation: bool = False, margin: float = 1.0) -> bool:
    """Information criterion: a measurement constrains the graph if the graph's own (un-inflated) prediction of the
    relative pose is less certain than the measurement in translation by `margin` (standard deviations; covariance
    traces, gtsam order r, t).  `sv`: the measurement's sigmas (6) or covariance (6, 6).
    `rotation=True` also admits measurements whose rotation is better known than the chain's: tested offline and
    rejected, because the calibrated rotation floor of the estimator (a few hundredths of a degree) is below the
    chain's rotation noise on nearly every edge, so the criterion degenerates to "all edges" (KITTI-07: 5.5 m map
    ATE against 0.7 m; Lone Monk seeds 1 / 2: 0.17 / 0.27 m against 0.14 / 0.44 m)."""
    sv = np.asarray(sv, dtype=np.float64)
    var = np.diag(sv) if sv.ndim == 2 else sv ** 2
    m2 = float(margin) ** 2
    if float(np.trace(cov[3:, 3:])) > m2 * float(np.sum(var[3:])):
        return True
    return bool(rotation and float(np.trace(cov[:3, :3])) > m2 * float(np.sum(var[:3])))


# ----------------------------------------------------------------------------- noise model
class NoiseModel:
    """Calibrated measurement noise (sigmas in gtsam order: rx ry rz tx ty tz)."""

    def __init__(self, cfg: NoiseModelConfig):
        self.cfg = cfg

    @staticmethod
    def _iso(sr: float, st: float) -> np.ndarray:
        return np.array([sr, sr, sr, st, st, st], dtype=np.float64)

    # a measurement is never more certain than these (numerical floor: a zero intercept would give a factor of
    # infinite weight for a measurement at zero distance)
    FLOOR_T, FLOOR_R = 1e-3, 1e-4

    def visual(self, dist: float, scale: float = 1.0) -> np.ndarray:
        c = self.cfg
        return self._iso(max(c.visual_r_a + c.visual_r_b * dist, self.FLOOR_R), max(c.visual_t_a + c.visual_t_b * dist, self.FLOOR_T)) * scale

    def odom(self, L: float, theta: float, n: int) -> np.ndarray:
        c = self.cfg
        n = max(int(n or 1), 1)
        return self._iso(c.odom_k_r * theta / math.sqrt(n) + c.odom_floor_r, c.odom_k_t * L / math.sqrt(n) + c.odom_floor_t)

    @staticmethod
    def split_cov(sr: float, st_along: float, st_cross: float, T) -> np.ndarray:
        """Covariance (gtsam order r, t; right perturbation at the measurement's end b) whose translation part is
        st_along along the measured bearing and st_cross across it (the bearing expressed in frame b)."""
        T = to_gtsam(T)
        t = np.asarray(T.translation(), dtype=np.float64)
        d = float(np.linalg.norm(t))
        C = np.zeros((6, 6))
        C[:3, :3] = np.eye(3) * sr ** 2
        C[3:, 3:] = np.eye(3) * st_cross ** 2
        if d > 1e-9:
            u = np.asarray(T.rotation().matrix()).T @ (t / d)
            C[3:, 3:] += (st_along ** 2 - st_cross ** 2) * np.outer(u, u)
        return C

    def visual_cov(self, T, scale: float = 1.0, scale_along: Optional[float] = None, scale_rot: Optional[float] = None) -> np.ndarray:
        """Visual measurement covariance: `scale` across the bearing, `scale_along` along it, `scale_rot` on the rotation
        (None: `scale`; scale_along None: the isotropic model)."""
        T = to_gtsam(T)
        s = self.visual(float(np.linalg.norm(T.translation())))
        sa = scale if scale_along is None else scale_along
        sr = scale if scale_rot is None else scale_rot
        return self.split_cov(s[0] * sr, s[3] * sa, s[3] * scale, T)

    def pair_cov(self, T, scale: float = 1.0, scale_along: Optional[float] = None, scale_rot: Optional[float] = None) -> np.ndarray:
        T = to_gtsam(T)
        s = self.pair(float(np.linalg.norm(T.translation())))
        sa = scale if scale_along is None else scale_along
        sr = scale if scale_rot is None else scale_rot
        return self.split_cov(s[0] * sr, s[3] * sa, s[3] * scale, T)

    def visual_cov_from_factor(self, f) -> np.ndarray:
        """Covariance of a stored visual factor with the noise scales it was measured under."""
        sc = float(getattr(f, "noise_scale", 1.0) or 1.0)
        sa = getattr(f, "noise_scale_along", None)
        sr = getattr(f, "noise_scale_rot", None)
        return self.visual_cov(f, sc, None if sa is None else float(sa), None if sr is None else float(sr))

    def factor_cov_gtsam(self, f) -> Optional[np.ndarray]:
        """Full covariance for the pose-graph optimisation (gtsam order) of an anisotropic visual factor, else None."""
        if f.type == EdgeType.VISUAL and getattr(f, "noise_scale_along", None) is not None:
            return self.visual_cov_from_factor(f)
        return None

    def pair(self, dist: float) -> np.ndarray:
        c = self.cfg
        return self._iso(max(c.pair_r_a + c.pair_r_b * dist, self.FLOOR_R), max(c.pair_t_a + c.pair_t_b * dist, self.FLOOR_T))

    def map_rel(self, dist: float) -> np.ndarray:
        c = self.cfg
        return self._iso(max(c.map_r_a + c.map_r_b * dist, self.FLOOR_R), max(c.map_t_a + c.map_t_b * dist, self.FLOOR_T))

    # --- from stored factors ---
    def visual_from_factor(self, f) -> np.ndarray:
        m = f.mean_np if hasattr(f, "mean_np") else f.mean.tensor().detach().cpu().numpy()
        return self.visual(float(np.linalg.norm(m[:3])), float(getattr(f, "noise_scale", 1.0) or 1.0))

    def odom_from_factor(self, f) -> np.ndarray:
        T = to_gtsam(f)
        L = float(np.linalg.norm(T.translation()))
        theta = float(np.linalg.norm(logmap(T)[:3]))
        return self.odom(L, theta, int(getattr(f, "n_frames", 1) or 1))

    def factor_sigmas_pypose(self, f) -> Optional[np.ndarray]:
        """Sigmas for the pose-graph optimisation (pypose order); None keeps the factor's own std."""
        if f.type == EdgeType.VISUAL:
            return rt_to_pypose(self.visual_from_factor(f))
        if f.type == EdgeType.ODOMETRY:
            return rt_to_pypose(self.odom_from_factor(f))
        return None


# ----------------------------------------------------------------------------- odometry chain
class ChainPredictor:
    """Dead-reckoned relative pose and covariance between two keyframes along the odometry chain.

    Prefix products make a query O(1): with P_i the pose of node i relative to the chain start and
    S_i = sum_{k<=i} Ad(P_k) Sigma_k Ad(P_k)^T (Sigma_k: covariance of edge k, right perturbation), the relative pose
    a -> b is P_a^{-1} P_b and its covariance (right perturbation at b) is Ad(P_b^{-1}) (S_b - S_a) Ad(P_b^{-1})^T.
    The tables follow the manager's odometry edges (mutation counter, count and newest key); a new edge at the end
    of a chain is appended in O(1), any other change rebuilds the tables (vectorised, ~1 ms per 500 edges)."""

    def __init__(self, hm, noise: NoiseModel):
        self.hm = hm
        self.noise = noise
        self._stamp = None
        self._sig: Dict[Tuple[int, int], np.ndarray] = {}
        self._next: Dict[int, Tuple[int, object]] = {}
        self._idx: Dict[int, Tuple[int, int]] = {}      # node -> (chain, position)
        self._P: list = []                                # per chain: (n, 4, 4) prefix poses
        self._S: list = []                                # per chain: (n, 6, 6) prefix covariance sums
        self._ends: list = []                             # per chain: last node

    # --- tables ---
    def _edge_sigma(self, key, e):
        s = self._sig.get(key)
        if s is None:
            s = self.noise.odom_from_factor(e)
            self._sig[key] = s
        return s

    @staticmethod
    def _adjoint(M: np.ndarray) -> np.ndarray:
        """Adjoint of SE(3) matrices (n, 4, 4) in gtsam tangent order (rotation, translation): [[R, 0], [[t]x R, R]]."""
        R, t = M[:, :3, :3], M[:, :3, 3]
        tx = np.zeros((len(M), 3, 3)); tx[:, 0, 1], tx[:, 0, 2], tx[:, 1, 0], tx[:, 1, 2], tx[:, 2, 0], tx[:, 2, 1] = -t[:, 2], t[:, 1], t[:, 2], -t[:, 0], -t[:, 1], t[:, 0]
        A = np.zeros((len(M), 6, 6)); A[:, :3, :3] = R; A[:, 3:, :3] = tx @ R; A[:, 3:, 3:] = R
        return A

    def _append(self, chain: int, node: int, key, e) -> None:
        P_prev = self._P[chain][-1]
        P = P_prev @ to_gtsam(e).matrix()
        A = self._adjoint(P[None])[0]
        M = A @ np.diag(self._edge_sigma(key, e) ** 2) @ A.T
        self._P[chain] = np.concatenate([self._P[chain], P[None]], 0)
        self._S[chain] = np.concatenate([self._S[chain], (self._S[chain][-1] + M)[None]], 0)
        self._idx[node] = (chain, len(self._P[chain]) - 1)
        self._ends[chain] = node

    def _rebuild(self) -> None:
        oe = self.hm.odom_edges
        self._next = {a: (b, e) for (a, b), e in oe.items()}
        preds = {b for (_, b) in oe}
        self._idx, self._P, self._S, self._ends = {}, [], [], []
        for root in sorted(a for a in self._next if a not in preds):
            nodes, keys, edges, cur = [root], [], [], root
            while cur in self._next and len(nodes) < 1000000:
                b, e = self._next[cur]
                if b in self._idx or b in nodes:
                    break
                keys.append((cur, b)); edges.append(e); nodes.append(b); cur = b
            n = len(edges)
            Ts = np.stack([to_gtsam(e).matrix() for e in edges], 0) if n else np.zeros((0, 4, 4))
            P = np.zeros((n + 1, 4, 4)); P[0] = np.eye(4)
            for i in range(n):
                P[i + 1] = P[i] @ Ts[i]
            sig = np.stack([self._edge_sigma(k, e) for k, e in zip(keys, edges)], 0) if n else np.zeros((0, 6))
            A = self._adjoint(P[1:])
            M = np.einsum("nij,nj,nkj->nik", A, sig ** 2, A)
            S = np.zeros((n + 1, 6, 6)); S[1:] = np.cumsum(M, 0)
            chain = len(self._P)
            self._P.append(P); self._S.append(S); self._ends.append(nodes[-1])
            for i, node in enumerate(nodes):
                self._idx[node] = (chain, i)

    def _refresh(self):
        oe = self.hm.odom_edges
        stamp = (getattr(self.hm, "odom_edges_version", 0), len(oe), max(oe) if oe else None)
        if stamp == self._stamp:
            return
        old = self._stamp
        self._stamp = stamp
        if old is not None and stamp[0] == old[0] + 1 and stamp[1] == old[1] + 1 and stamp[2] is not None:
            a, b = stamp[2]
            if a in self._idx and b not in self._idx and self._ends[self._idx[a][0]] == a and (a, b) in oe:
                self._next[a] = (b, oe[(a, b)])
                self._append(self._idx[a][0], b, (a, b), oe[(a, b)])
                return
        self._rebuild()

    # --- queries ---
    def path(self, a: int, b: int, max_len: int = 100000) -> Optional[list]:
        """Odometry edges from a to b (a earlier in the chain), or None."""
        self._refresh()
        edges, cur = [], a
        for _ in range(max_len):
            nxt = self._next.get(cur)
            if nxt is None:
                return None
            edges.append(((cur, nxt[0]), nxt[1]))
            cur = nxt[0]
            if cur == b:
                return edges
        return None

    def _forward(self, a: int, b: int, inflate: float):
        ca, ia = self._idx[a]; cb, ib = self._idx[b]
        P, S = self._P[ca], self._S[ca]
        Pa, Pb = gtsam.Pose3(P[ia]), gtsam.Pose3(P[ib])
        T = Pa.between(Pb)
        A = Pb.inverse().AdjointMap()
        cov = A @ (S[ib] - S[ia]) @ A.T * inflate ** 2
        return T, 0.5 * (cov + cov.T)

    def predict(self, a: int, b: int, inflate: float = 1.0) -> Tuple[Optional[gtsam.Pose3], Optional[np.ndarray]]:
        """T_ab and its covariance (right perturbation) along the chain; either order of a, b."""
        if a == b:
            return gtsam.Pose3(), np.zeros((6, 6))
        self._refresh()
        ia, ib = self._idx.get(a), self._idx.get(b)
        if ia is None or ib is None or ia[0] != ib[0]:
            return None, None
        if ia[1] < ib[1]:
            return self._forward(a, b, inflate)
        T, cov = self._forward(b, a, inflate)
        T = T.inverse()
        return T, transport(cov, T)      # right perturbation of T^-1

    def predict_walk(self, a: int, b: int, inflate: float = 1.0):
        """Reference implementation (edge-by-edge compounding), used by the tests."""
        if a == b:
            return gtsam.Pose3(), np.zeros((6, 6))
        forward = True
        edges = self.path(a, b)
        if edges is None:
            edges = self.path(b, a)
            forward = False
            if edges is None:
                return None, None
        Ts = [to_gtsam(e) for _, e in edges]
        cov = np.zeros((6, 6))
        tail = gtsam.Pose3()
        for k in range(len(edges) - 1, -1, -1):
            s = self._edge_sigma(edges[k][0], edges[k][1]) * inflate
            cov = cov + transport(np.diag(s ** 2), tail)
            tail = Ts[k].compose(tail)
        T = tail
        if not forward:
            T = T.inverse()
            cov = transport(cov, T)
        return T, cov


# ----------------------------------------------------------------------------- verifier
class LoopClosureVerifier:
    """Consistency tests for loop-closure candidates of hypothesis 0 (see module docstring)."""

    def __init__(self, system, cfg: LoopClosureConfig):
        self.system = system
        self.cfg = cfg
        self.noise = NoiseModel(cfg.noise)
        self.test_dof = getattr(cfg, "test_dof", "translation")
        self.translation_only = self.test_dof == "translation"
        # "split": the translation and the rotation residual are tested separately (3 dof each, same level); the
        # statistic is the larger of the two
        self.thr = chi2_threshold(cfg.confidence, 6 if self.test_dof == "full" else 3)
        self.chain = ChainPredictor(system.hypothesis_manager, self.noise)
        self.anchor: Optional[dict] = None       # {"map_kf", "kf", "T" (map_kf -> kf), "sigma"} of the latest verified session->map edge
        self.stats = {"inpass_rejected": 0, "prior_rejected": 0, "prior_tested": 0, "posterior_rejected": 0, "pgo": 0, "h0_loop_edges": 0}
        self.quarantine: list = []
        # online visual-noise scales (innovation-based), one per measurement type: references of the current
        # session and references of the stored map (a query under appearance change is noisy against the map only)
        self._innov = {"sess": collections.deque(maxlen=int(getattr(cfg, "adaptive_window", 150))),
                       "map": collections.deque(maxlen=int(getattr(cfg, "adaptive_window", 150)))}
        self.scales = {"sess": 1.0, "map": 1.0}
        self.scale = 1.0                # scale of the last reference type handled (kept for logging / stamping)
        # anisotropic model (cfg.anisotropic): `scales` act across the bearing, `scales_along` along it, `scales_rot` on
        # the rotation; their statistics are the across / along / rotation components of the innovation
        self.anisotropic = bool(getattr(cfg, "anisotropic", False))
        self.margin = float(getattr(cfg, "informative_margin", 1.0) or 1.0)
        self.scales_along = {"sess": 1.0, "map": 1.0}
        self.scales_rot = {"sess": 1.0, "map": 1.0}
        self._innov_r = {"sess": collections.deque(maxlen=int(getattr(cfg, "adaptive_window", 150))),
                         "map": collections.deque(maxlen=int(getattr(cfg, "adaptive_window", 150)))}
        self._innov_a = {"sess": collections.deque(maxlen=int(getattr(cfg, "adaptive_window", 150))),
                         "map": collections.deque(maxlen=int(getattr(cfg, "adaptive_window", 150)))}
        self._innov_c = {"sess": collections.deque(maxlen=int(getattr(cfg, "adaptive_window", 150))),
                         "map": collections.deque(maxlen=int(getattr(cfg, "adaptive_window", 150)))}
        # online metric scale of the estimator: median ratio of the raw measured translation to the odometry-chain
        # translation over short chains (<= 5 edges, >= scale_min_span_m, 1 m by default) of the last `adaptive_window` measurements; the
        # estimator's translations are divided by it (the odometry is metric, the feed-forward scale comes from the
        # stereo anchors and is biased where they are far: 0.82 on KITTI-06, 0.89 on KITTI-07, 1.0-1.2 on Lone Monk
        # depending on the place, so a constant cannot serve)
        self._ratio = collections.deque(maxlen=int(getattr(cfg, "adaptive_window", 150)))
        self.scale_ratio = float(getattr(cfg.noise, "visual_scale", 1.0) or 1.0)
        self.metric_correction = self.scale_ratio     # the value the estimator currently applies
        # odometry scale guard (LoopClosureConfig.odom_guard_factor); `guard_metric` is set by the System when the
        # estimator's translations are metric on their own (stereo / depth), the only case where they can judge the odometry
        self.guard_factor = float(getattr(cfg, "odom_guard_factor", 0.0) or 0.0)
        self.guard_metric = False
        self._guard = collections.deque(maxlen=max(int(getattr(cfg, "odom_guard_window", 10)), 3))
        self.odom_scale = 1.0                         # correction of the odometry's translations (applied by the System)

    def reset_session(self):
        self.anchor = None
        self.chain = ChainPredictor(self.system.hypothesis_manager, self.noise)
        self.stats = {k: 0 for k in self.stats}
        for q in list(self._innov.values()) + list(self._innov_a.values()) + list(self._innov_c.values()) + list(self._innov_r.values()):
            q.clear()
        self._ratio.clear()
        self._guard.clear()
        self.odom_scale = 1.0
        self.scales = {"sess": 1.0, "map": 1.0}
        self.scales_along = {"sess": 1.0, "map": 1.0}
        self.scales_rot = {"sess": 1.0, "map": 1.0}
        self.scale = 1.0

    MEDIAN_3DOF = 1.5382     # median of |e| for an isotropic 3-dof Gaussian, in units of sigma
    MEDIAN_1DOF = 0.6745     # median of |e| for a 1-dof Gaussian (along the bearing)
    MEDIAN_2DOF = 1.1774     # median of |e| for an isotropic 2-dof Gaussian (across the bearing)
    SCALE_MAX = 5.0          # sanity bound of the adaptation (a broken estimator is not a noisier one)

    def _update_scale(self):
        """Scale of the visual noise from the normalised prior-test residuals of the recent observations (median
        based, robust to the wrong-place references that the tests reject): 1 when the calibration holds, > 1 when
        the measurements against the map are noisier than on the calibration data (appearance change).  Needs a
        minimum of 20 samples; never below 1, never above SCALE_MAX."""
        if self.anisotropic:
            self._update_scale_split()
            return
        for kind, q in self._innov.items():
            if not getattr(self.cfg, "adaptive_scale", True) or len(q) < 20:
                self.scales[kind] = 1.0
                continue
            med = float(np.median(np.asarray(q)))
            self.scales[kind] = float(min(max(1.0, med / self.MEDIAN_3DOF), self.SCALE_MAX))
            # self-check of the prior: when the median normalised innovation of the map measurements exceeds what the
            # scale may absorb, the prior (the session anchor) is inconsistent with the measurements, not the other
            # way round; the anchor is dropped and the next verified or corroborated map edge re-anchors the session
            if kind == "map" and med / self.MEDIAN_3DOF > self.SCALE_MAX and self.anchor is not None:
                self.stats["anchor_dropped"] = self.stats.get("anchor_dropped", 0) + 1
                logger.info(f"prior inconsistent with the map measurements (median normalised innovation {med:.1f}): session anchor dropped")
                self.anchor = None
                q.clear()
                self.scales[kind] = 1.0

    def _update_scale_split(self):
        """Anisotropic model: the scales across the bearing, along it and of the rotation, from the medians of the
        across / along / rotation innovation components (each normalised with the base model plus the un-inflated chain
        variance in that direction); same bounds as the isotropic scale.  The anchor self-check of a relocalization
        session uses the across-track statistic: a wrong anchor shows in the direction, a far-away scene in the length."""
        for kind in ("sess", "map"):
            qa, qc, qr = self._innov_a[kind], self._innov_c[kind], self._innov_r[kind]
            if not getattr(self.cfg, "adaptive_scale", True) or len(qc) < 20:
                self.scales[kind] = self.scales_along[kind] = self.scales_rot[kind] = 1.0
                continue
            med_c = float(np.median(np.asarray(qc))) / self.MEDIAN_2DOF
            med_a = float(np.median(np.asarray(qa))) / self.MEDIAN_1DOF
            med_r = float(np.median(np.asarray(qr))) / self.MEDIAN_3DOF
            self.scales[kind] = float(min(max(1.0, med_c), self.SCALE_MAX))
            self.scales_along[kind] = float(min(max(1.0, med_a), self.SCALE_MAX))
            self.scales_rot[kind] = float(min(max(1.0, med_r), self.SCALE_MAX))
            if kind == "map" and med_c > self.SCALE_MAX and self.anchor is not None:
                self.stats["anchor_dropped"] = self.stats.get("anchor_dropped", 0) + 1
                logger.info(f"prior inconsistent with the map measurements (median normalised across-track innovation {med_c:.1f}): session anchor dropped")
                self.anchor = None
                qa.clear(); qc.clear(); qr.clear()
                self.scales[kind] = self.scales_along[kind] = self.scales_rot[kind] = 1.0

    def scale_for(self, kf_id: int) -> float:
        return self.scales["map"] if self._is_map_kf(int(kf_id)) else self.scales["sess"]

    def scale_along_for(self, kf_id: int) -> Optional[float]:
        """Along-track scale of the anisotropic model (None: isotropic model)."""
        if not self.anisotropic:
            return None
        return self.scales_along["map"] if self._is_map_kf(int(kf_id)) else self.scales_along["sess"]

    def scale_rot_for(self, kf_id: int) -> Optional[float]:
        """Rotation scale of the anisotropic model (None: isotropic model, the rotation uses `scale_for`)."""
        if not self.anisotropic:
            return None
        return self.scales_rot["map"] if self._is_map_kf(int(kf_id)) else self.scales_rot["sess"]

    def meas_cov(self, kf_id: int, T, unscaled: bool = False) -> np.ndarray:
        """Covariance of a visual measurement to keyframe kf_id under the current online scales."""
        if unscaled:
            return self.noise.visual_cov(T, 1.0, 1.0 if self.anisotropic else None)
        return self.noise.visual_cov(T, self.scale_for(kf_id), self.scale_along_for(kf_id), self.scale_rot_for(kf_id))

    # ------------------------------------------------------------------ helpers
    def chi2(self, r: np.ndarray, cov: np.ndarray, rotation: bool = False) -> float:
        """Test statistic: translation-only (3 dof), full (6 dof), or split: the translation marginal (3 dof), and with
        `rotation` (loop measurements between keyframes of the current session) the larger of it and the rotation
        marginal (3 dof).  Edges to a stored map and in-pass pairs keep the translation test: their predictions are
        short, so the estimator's rotation noise (4x the base model on OpenLORIS before the online scale adapts, and a
        relocalization trial is short) would decide the test."""
        if self.translation_only or (self.test_dof == "split" and not rotation):
            return chi2_of(r[3:], cov[3:, 3:])
        if self.test_dof == "split":
            return max(chi2_of(r[3:], cov[3:, 3:]), chi2_of(r[:3], cov[:3, :3]))
        return chi2_of(r, cov)

    def _both_session(self, a: int, b: int) -> bool:
        return not self._is_map_kf(int(a)) and not self._is_map_kf(int(b))

    def _session_start(self) -> int:
        return int(getattr(self.system, "_session_start_kf_id", 0))

    def _is_map_kf(self, kid: int) -> bool:
        return kid < self._session_start()

    def map_relative(self, a: int, b: int) -> Tuple[Optional[gtsam.Pose3], Optional[np.ndarray]]:
        """Relative pose of two keyframes as the graph currently knows it, with an uncertainty: the odometry chain
        (same session, not yet closed) or the stored map's consistency model (map keyframes)."""
        hm = self.system.hypothesis_manager
        if a not in hm.nodes or b not in hm.nodes:
            return None, None
        Ta, Tb = to_gtsam(hm.nodes[a].pose_mu[0]), to_gtsam(hm.nodes[b].pose_mu[0])
        T = Ta.between(Tb)
        d = float(np.linalg.norm(T.translation()))
        if self._is_map_kf(a) and self._is_map_kf(b):
            return T, np.diag(self.noise.map_rel(d) ** 2)
        if self._is_map_kf(a) != self._is_map_kf(b):
            return None, None            # a map keyframe and a session keyframe: relation only through the anchor
        _, cov = self.chain.predict(a, b)
        if cov is None:
            return T, np.diag(self.noise.map_rel(d) ** 2)
        # the graph's estimate is at least as good as the map model (loop closures already applied)
        cm = np.diag(self.noise.map_rel(d) ** 2)
        return T, (cov if np.trace(cov) < np.trace(cm) else cm)

    # ------------------------------------------------------------------ 1. in-pass consistency
    def inpass_gate(self, c2w_metric: np.ndarray, refs: Sequence, valid: np.ndarray, covis: Optional[np.ndarray] = None,
                    prior_ok: Optional[Sequence] = None) -> np.ndarray:
        """Largest set of mutually consistent references.  c2w_metric: (S, 4, 4) with index 0 = current view and
        1..B = references in the order of `refs`; valid: (B,) references that passed the covisibility test.
        Returns a (B,) mask: references to keep (invalid ones stay False)."""
        self.last_inpass_best = set()
        B = len(refs)
        keep = np.asarray(valid, dtype=bool).copy()
        idx = [i for i in range(B) if keep[i]]
        if len(idx) < 2:
            return keep
        P = {i: to_gtsam(c2w_metric[1 + i]) for i in idx}
        ok = {}
        n_tested = 0
        for i, j in itertools.combinations(idx, 2):
            M, cov_m = self.map_relative(int(refs[i].id), int(refs[j].id))
            if M is None:
                ok[(i, j)] = True          # relation unknown: no evidence either way
                continue
            Pij = P[i].between(P[j])
            d = float(np.linalg.norm(Pij.translation()))
            r = residual(Pij, M)
            sc = max(self.scale_for(int(refs[i].id)), self.scale_for(int(refs[j].id)))
            if self.anisotropic:
                sa = max(self.scale_along_for(int(refs[i].id)), self.scale_along_for(int(refs[j].id)))
                sr = max(self.scale_rot_for(int(refs[i].id)), self.scale_rot_for(int(refs[j].id)))
                c2 = self.chi2(r, self.noise.pair_cov(Pij, sc, sa, sr) + cov_m)
            else:
                c2 = self.chi2(r, np.diag((self.noise.pair(d) * sc) ** 2) + cov_m)
            ok[(i, j)] = c2 <= self.thr
            n_tested += 1
        if n_tested == 0 or all(ok.values()):
            return keep
        # best mutually consistent subset: most members with a positive prior verdict, then size, then covisibility.
        # A reference outside it is dropped when its own prior verdict is negative or when the subset is a strict
        # majority of the tested references; otherwise (e.g. one against one without prior evidence) it is kept.
        w = np.asarray(covis, dtype=float) if covis is not None else np.ones(B)
        pv = [(prior_ok[i] if (prior_ok is not None and i < len(prior_ok)) else None) for i in range(B)]
        best, best_key = None, None
        for size in range(len(idx), 0, -1):
            for sub in itertools.combinations(idx, size):
                if all(ok.get((i, j), True) for i, j in itertools.combinations(sub, 2)):
                    key = (sum(1 for i in sub if pv[i] is True), size, float(sum(w[i] for i in sub)))
                    if best_key is None or key > best_key:
                        best, best_key = sub, key
        if best is None:
            return keep
        # references corroborated by the pass: members of the best mutually consistent subset (used to anchor a
        # relocalization session only on map edges that at least one other reference of the same pass confirms)
        self.last_inpass_best = {int(refs[i].id) for i in best}
        majority = 2 * len(best) > len(idx)
        new = keep.copy()
        for i in idx:
            if i in best:
                continue
            if pv[i] is False or majority:
                new[i] = False
        n_rej = int(keep.sum() - new.sum())
        self.stats["inpass_rejected"] += n_rej
        if n_rej:
            logger.debug(f"in-pass consistency: dropped references {[int(refs[i].id) for i in idx if not new[i]]} (kept {[int(refs[i].id) for i in best or ()]})")
        return new

    # ------------------------------------------------------------------ 2. prior consistency
    def predict_to_current(self, ref_id: int, last_kf_id: int, T_since: gtsam.Pose3, cov_since: np.ndarray):
        """Prediction of T_ref_current = chain(ref -> last keyframe) (+) odometry since the last keyframe, or through
        the session anchor for a map keyframe.  Returns (T_pred, cov) or (None, None) when no relation exists."""
        infl = self.cfg.noise.gate_inflation
        if not self._is_map_kf(ref_id):
            T, cov = self.chain.predict(ref_id, last_kf_id, inflate=infl)
            if T is None:
                return None, None
            return T.compose(T_since), transport(cov, T_since) + cov_since
        if self.anchor is None:
            return None, None
        hm = self.system.hypothesis_manager
        a, a2 = ref_id, self.anchor["map_kf"]
        if a not in hm.nodes or a2 not in hm.nodes:
            return None, None
        T_map = to_gtsam(hm.nodes[a].pose_mu[0]).between(to_gtsam(hm.nodes[a2].pose_mu[0]))
        d_map = float(np.linalg.norm(T_map.translation()))
        T_chain, cov_chain = self.chain.predict(self.anchor["kf"], last_kf_id, inflate=infl)
        if T_chain is None:
            return None, None
        T_anc = self.anchor["T"]
        # T_pred = T_map (+) T_anc (+) T_chain (+) T_since ; transport every term to the end
        rest = T_anc.compose(T_chain).compose(T_since)
        cov = transport(np.diag(self.noise.map_rel(d_map) ** 2), rest)
        cov += transport(self.anchor["cov"] if "cov" in self.anchor else np.diag(self.anchor["sigma"] ** 2), T_chain.compose(T_since))
        cov += transport(cov_chain, T_since) + cov_since
        return T_map.compose(rest), cov

    def _guard_sample(self, sample: float) -> bool:
        """Odometry scale guard: record one measured / odometry translation ratio; returns whether the sample is
        consistent with the long-run ratio (and may update it).  Rescales the odometry when the recent samples agree on a
        departure of more than the guard factor."""
        if self.guard_factor <= 1.0 or not self.guard_metric or sample <= 0:
            return True
        ref = self.scale_ratio if self.scale_ratio > 0 else 1.0
        g = sample / ref
        self._guard.append(g)
        lim = math.log(self.guard_factor)
        if len(self._guard) == self._guard.maxlen:
            m = float(np.median(self._guard))
            if abs(math.log(m)) > lim:
                self.odom_scale *= m
                self._guard.clear()
                self.stats["odom_guard_updates"] = self.stats.get("odom_guard_updates", 0) + 1
                self.stats["odom_scale"] = round(self.odom_scale, 4)
                logger.info(f"odometry scale guard: measured / odometry translation {m:.3f} x the long-run ratio over the "
                            f"last {self._guard.maxlen} spans; odometry translations now scaled by {self.odom_scale:.4f}")
        return abs(math.log(g)) <= lim

    def prior_gate(self, refs: Sequence, T_meas: Sequence, last_kf_id: int, T_since, n_since: int) -> Tuple[list, list]:
        """For every reference: True (consistent with hypothesis 0), False (inconsistent), None (no relation).
        T_meas: measured T_ref_current (pypose SE3 or 7-vectors).  Also records, per reference, whether the
        measurement is a loop-closure candidate: the graph's own prediction of the relative pose is less certain
        than the measurement (translation covariance trace), which is the case for revisits, not for the keyframes
        just behind the robot (`self.last_loop_flags`)."""
        Ts = to_gtsam(T_since) if T_since is not None else gtsam.Pose3()
        L = float(np.linalg.norm(Ts.translation())); th = float(np.linalg.norm(logmap(Ts)[:3]))
        cov_since = np.diag((self.noise.odom(L, th, max(n_since, 1)) * self.cfg.noise.gate_inflation) ** 2)
        oks, chis, loops = [], [], []
        self._update_scale()
        for kf, Tm in zip(refs, T_meas):
            T_pred, cov = self.predict_to_current(int(kf.id), last_kf_id, Ts, cov_since)
            if T_pred is None:
                self.stats["prior_no_chain"] = self.stats.get("prior_no_chain", 0) + 1
                oks.append(None); chis.append(None); loops.append(True); continue
            Tm = to_gtsam(Tm)
            d = float(np.linalg.norm(Tm.translation()))
            r = residual(Tm, T_pred)
            kind = "map" if self._is_map_kf(int(kf.id)) else "sess"
            short = kind == "map"
            if kind == "sess":
                ia, ib = self.chain._idx.get(int(kf.id)), self.chain._idx.get(int(last_kf_id))
                L = float(np.linalg.norm(T_pred.translation()))
                short = ia is not None and ib is not None and ia[0] == ib[0] and abs(ia[1] - ib[1]) <= 5
                if short and L >= float(getattr(self.cfg, "scale_min_span_m", 1.0)):
                    sample = d * self.metric_correction / L
                    consistent = self._guard_sample(sample)       # the odometry scale guard (off: always True)
                    if consistent:
                        self._ratio.append(sample)
                    if consistent and len(self._ratio) >= 20:
                        self.scale_ratio = float(np.median(self._ratio))
                        self.stats["scale_ratio"] = round(self.scale_ratio, 4)
            self.scale = self.scales[kind]
            cov0 = cov / max(self.cfg.noise.gate_inflation, 1e-6) ** 2      # un-inflated chain covariance
            if self.anisotropic:
                Sv = self.noise.visual_cov(Tm, self.scale, self.scales_along[kind], self.scales_rot[kind])
                # loop candidate: the prediction (un-inflated chain) is less certain than the measurement by the margin
                # (measurements of the session's own keyframes; edges to a stored map keep the plain criterion)
                loops.append(informative(cov0, Sv, rotation=False, margin=self.margin if kind == "sess" else 1.0))
                # innovation statistics along / across the measured bearing (frame b), each normalised by the base model
                # and the un-inflated chain variance in that direction: medians 0.674 / 1.177 when the model holds
                if short and d > 1e-6:
                    u = np.asarray(Tm.rotation().matrix()).T @ (np.asarray(Tm.translation(), dtype=np.float64) / d)
                    rt = r[3:]
                    ra = float(rt @ u)
                    rc = float(np.linalg.norm(rt - ra * u))
                    va = float(u @ cov0[3:, 3:] @ u)
                    vc = max(float(np.trace(cov0[3:, 3:])) - va, 0.0)
                    sg = self.noise.visual(d)
                    base, base_r = float(sg[3]), float(sg[0])
                    self._innov_a[kind].append(abs(ra) / math.sqrt(base ** 2 + va))
                    self._innov_c[kind].append(rc / math.sqrt(base ** 2 + vc / 2.0))
                    self._innov_r[kind].append(float(np.linalg.norm(r[:3])) / math.sqrt(base_r ** 2 + float(np.trace(cov0[:3, :3])) / 3.0))
                c2 = self.chi2(r, cov + Sv, rotation=kind == "sess")
            else:
                sv = self.noise.visual(d, self.scale)
                # loop candidate: the prediction (un-inflated chain) is less certain than the measurement
                loops.append(informative(cov0, sv, rotation=False, margin=self.margin if kind == "sess" else 1.0))
                # innovation statistic: the translation residual of this very test, normalised by the un-scaled
                # covariance (prediction + calibrated measurement noise); its median over the recent observations is 1.54
                # when the calibration holds and grows when the measurements (against the map or the session) get noisier
                S0 = cov + np.diag(self.noise.visual(d) ** 2)
                self._innov[kind].append(float(np.linalg.norm(r[3:])) / math.sqrt(max(float(np.trace(S0[3:, 3:])) / 3.0, 1e-12)))
                c2 = self.chi2(r, cov + np.diag(self.noise.visual(d, self.scale) ** 2))
            self.stats["prior_tested"] += 1
            if c2 > self.thr:
                self.stats["prior_rejected"] += 1
            oks.append(bool(c2 <= self.thr)); chis.append(c2)
        self.last_loop_flags = loops
        self.stats["loop_candidates"] = self.stats.get("loop_candidates", 0) + sum(1 for l, o in zip(loops, oks) if l and o is not None)
        return oks, chis

    def corroborated(self, kf_id: int) -> bool:
        """The reference is in the best mutually consistent subset of its pass together with at least one other
        stored-map reference (the pair test against the stored map is tight, so this is independent evidence)."""
        best = getattr(self, "last_inpass_best", set()) or set()
        return int(kf_id) in best and sum(1 for k in best if self._is_map_kf(int(k))) >= 2

    def make_anchor(self, map_kf_id: int, kf_id: int, T_meas) -> dict:
        Tm = to_gtsam(T_meas)
        out = {"map_kf": int(map_kf_id), "kf": int(kf_id), "T": Tm,
               "sigma": self.noise.visual(float(np.linalg.norm(Tm.translation())), self.scales["map"])}
        if self.anisotropic:
            out["cov"] = self.noise.visual_cov(Tm, self.scales["map"], self.scales_along["map"], self.scales_rot["map"])
        return out

    def update_anchor(self, map_kf_id: int, kf_id: int, T_meas) -> None:
        self.anchor = self.make_anchor(map_kf_id, kf_id, T_meas)

    def anchored_chi2(self, anchor: dict, ref_id: int, T_meas, last_kf_id: int, T_since, n_since: int) -> Optional[float]:
        """chi^2 of the prior test of the map reference ref_id through a candidate session anchor (an earlier map edge
        of hypothesis 0), i.e. whether two map measurements of different observations agree through the odometry chain
        between them.  None when no chain relates them."""
        Ts = to_gtsam(T_since) if T_since is not None else gtsam.Pose3()
        L = float(np.linalg.norm(Ts.translation())); th = float(np.linalg.norm(logmap(Ts)[:3]))
        cov_since = np.diag((self.noise.odom(L, th, max(n_since, 1)) * self.cfg.noise.gate_inflation) ** 2)
        saved, self.anchor = self.anchor, anchor
        try:
            T_pred, cov = self.predict_to_current(int(ref_id), last_kf_id, Ts, cov_since)
        finally:
            self.anchor = saved
        if T_pred is None:
            return None
        Tm = to_gtsam(T_meas)
        d = float(np.linalg.norm(Tm.translation()))
        if self.anisotropic:
            return self.chi2(residual(Tm, T_pred), cov + self.noise.visual_cov(Tm, self.scales["map"], self.scales_along["map"], self.scales_rot["map"]))
        return self.chi2(residual(Tm, T_pred), cov + np.diag(self.noise.visual(d, self.scales["map"]) ** 2))

    # ------------------------------------------------------------------ 3. residuals against the graph / posterior
    def edge_cov(self, a: int, b: int, factor) -> np.ndarray:
        """Covariance of a stored visual measurement for residual tests against the graph: the measurement noise,
        plus the stored map's own inconsistency when the edge ties a session keyframe to a (fixed) map keyframe."""
        if getattr(factor, "noise_scale_along", None) is not None:
            cov = self.noise.visual_cov_from_factor(factor)
        else:
            s = self.noise.visual_from_factor(factor)
            cov = np.diag(s ** 2)
        if self._is_map_kf(a) != self._is_map_kf(b):
            d = float(np.linalg.norm((factor.mean_np if hasattr(factor, "mean_np") else factor.mean.tensor().detach().cpu().numpy())[:3]))
            cov = cov + np.diag(self.noise.map_rel(d) ** 2)
        return cov

    def graph_residual_chi2(self, a: int, b: int, factor) -> Optional[float]:
        """chi^2 of a stored visual measurement a -> b against the current hypothesis-0 keyframe poses."""
        hm = self.system.hypothesis_manager
        if a not in hm.nodes or b not in hm.nodes:
            return None
        pred = to_gtsam(hm.nodes[a].pose_mu[0]).between(to_gtsam(hm.nodes[b].pose_mu[0]))
        Tm = to_gtsam(factor)
        return self.chi2(residual(Tm, pred), self.edge_cov(a, b, factor), rotation=self._both_session(a, b))

    def posterior_outliers(self, pg, only_keys: Optional[set] = None, informative_only: bool = False) -> list:
        """Visual factors of an optimised pose graph whose residual at the optimised poses exceeds the threshold.
        Returns [(a, b, factor, chi2)] (hypothesis-0 factors only: both endpoints are keyframes)."""
        hm = self.system.hypothesis_manager
        vm = pg.vertex_map
        out = []
        for (a, b, factors) in pg.edges:
            if a not in hm.nodes or b not in hm.nodes or a not in vm or b not in vm:
                continue
            if only_keys is not None and (a, b) not in only_keys:
                continue
            pa = pg.optimized_poses.get(a, vm[a].pose)
            pb = pg.optimized_poses.get(b, vm[b].pose)
            pred = to_gtsam(pa).between(to_gtsam(pb))
            for f in factors:
                if f.type != EdgeType.VISUAL or (informative_only and getattr(f, "informative", True) is False):
                    continue
                c2 = self.chi2(residual(to_gtsam(f), pred), self.edge_cov(a, b, f), rotation=self._both_session(a, b))
                if c2 > self.thr:
                    out.append((a, b, f, c2))
        return out

    def merged_edge_outlier_fraction(self, pg) -> Tuple[float, int]:
        """Fraction of the merged hypothesis' visual edges (edges touching a twin vertex) that are posterior outliers."""
        hm = self.system.hypothesis_manager
        vm = pg.vertex_map
        n_out, n = 0, 0
        for (a, b, factors) in pg.edges:
            if a in hm.nodes and b in hm.nodes:
                continue
            if a not in vm or b not in vm:
                continue
            pa = pg.optimized_poses.get(a, vm[a].pose)
            pb = pg.optimized_poses.get(b, vm[b].pose)
            pred = to_gtsam(pa).between(to_gtsam(pb))
            for f in factors:
                if f.type != EdgeType.VISUAL:
                    continue
                # twin vertices carry the keyframe id of the session keyframe; edges of the merged hypothesis tie
                # session keyframes to map keyframes, so the map term applies
                ka = a if a in hm.nodes else vm[a].original_kf_id
                kb = b if b in hm.nodes else vm[b].original_kf_id
                c2 = self.chi2(residual(to_gtsam(f), pred), self.edge_cov(ka, kb, f), rotation=self._both_session(ka, kb))
                n += 1
                n_out += int(c2 > self.thr)
        return (n_out / n if n else 0.0), n

    def map_consistency(self) -> Optional[dict]:
        """Local consistency of the finished map without ground truth: tail-calibrated fit sigma(d) = a + b d of the
        posterior residuals of the hypothesis-0 visual edges against the final keyframe poses (translation and
        rotation norms), used as the stored map's consistency model in later sessions."""
        hm = self.system.hypothesis_manager
        rows = []
        for (a, b), factors in hm.hypotheses[0].visual_edges.items():
            if a not in hm.nodes or b not in hm.nodes:
                continue
            pred = to_gtsam(hm.nodes[a].pose_mu[0]).between(to_gtsam(hm.nodes[b].pose_mu[0]))
            for f in factors:
                if f.type != EdgeType.VISUAL:
                    continue
                r = residual(to_gtsam(f), pred)
                rows.append((float(np.linalg.norm(f.mean_np[:3])), float(np.linalg.norm(r[3:])), float(np.linalg.norm(r[:3]))))
        if len(rows) < 50:
            return None
        rows = np.asarray(rows)
        def fit(col):
            xs, ys, ws = [], [], []
            for lo, hi in zip([0, 0.5, 1, 2, 4, 8, 16, 32], [0.5, 1, 2, 4, 8, 16, 32, 64]):
                m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
                if m.sum() >= 15:
                    xs.append(float(np.median(rows[m, 0]))); ys.append(float(np.percentile(rows[m, col], 90) / P90_3DOF)); ws.append(math.sqrt(m.sum()))
            if not xs:
                return None
            if len(xs) == 1:
                return float(ys[0]), 0.0
            from scipy.optimize import nnls
            xs, ys, ws = np.asarray(xs), np.asarray(ys), np.asarray(ws)
            A = np.stack([np.ones_like(xs), xs], 1) * ws[:, None]
            coef, _ = nnls(A, ys * ws)
            return float(coef[0]), float(coef[1])
        ft, fr = fit(1), fit(2)
        if ft is None or fr is None:
            return None
        return {"map_t_a": max(ft[0], 0.01), "map_t_b": ft[1], "map_r_a": max(fr[0], 0.001), "map_r_b": fr[1], "n_edges": int(len(rows))}

    def remove_edge(self, a: int, b: int, factor) -> None:
        """Move a hypothesis-0 visual factor to the quarantine (it no longer constrains the graph)."""
        hm = self.system.hypothesis_manager
        h0 = hm.hypotheses[0]
        bucket = h0.visual_edges.get((a, b))
        if not bucket:
            return
        h0.visual_edges[(a, b)] = [f for f in bucket if f is not factor]
        if not h0.visual_edges[(a, b)]:
            del h0.visual_edges[(a, b)]
            if b in h0.visual_adjacency.get(a, set()):
                h0.visual_adjacency[a].discard(b)
            if a in h0.visual_adjacency.get(b, set()):
                h0.visual_adjacency[b].discard(a)
        self.quarantine.append((a, b, factor))
        self.stats["posterior_rejected"] += 1
