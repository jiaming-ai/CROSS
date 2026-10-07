"""GNSS measurement model and online quality control.

A fix is used only when three tests agree, each at the chi-square level of the verified loop closure
(`mapping.loop_closure.confidence`):

1. **Relative consistency** (drift free).  The last fixes of the current epoch (no gap) are compared with the robot's
   own track at the fix times (odometry / tracked poses in the map frame, levelled to the horizontal) up to a rigid 2-D
   transform fitted over the window.  The track is accurate over a few seconds, so the residuals measure the receiver's
   current noise: their robust NIS (median of the normalised squared residuals over the median of chi^2_2) inflates the
   prior noise when it exceeds 1 (multipath, reflections near buildings), and a fix that does not fit its window is
   rejected.  A position that stays put while the robot moves (a receiver holding its last fix indoors) or jumps fails
   this test.
2. **Reacquisition hold-off.**  After a gap (no fix for several nominal intervals; the interval is learned from the
   stream) a new epoch starts and its fixes are used only once the window holds `consistency_min_fixes` fixes that pass
   test 1 (receivers report converging positions for a while after reacquisition, typically when leaving a building).
3. **Absolute gate.**  The innovation against the predicted position (the caller's estimate carried through the map's
   ENU anchor) with covariance R (inflated) + P_pred (anchor + drift since the last used fix).  A run of fixes that pass
   tests 1-2 but are offset from the prediction by the same vector while the robot travels further than the offset is
   attributed to the robot's drift (multipath offsets change with the geometry as the robot moves), and accepted.

No receiver-specific constants: prior noise per fix class (receiver accuracy, HDOP, or mode / satellites for receivers
that report nothing better) is only a starting point; the relative test inflates it online and the posterior residuals
of the pose-graph optimisation rescale it (`GnssNoiseModel.update_posterior`)."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import chi2

# fix modes (normalised): receivers' own codes are mapped to these by the loaders
FIX_NONE, FIX_2D, FIX_3D, FIX_DGPS, FIX_RTK_FLOAT, FIX_RTK_FIXED = 0, 2, 3, 4, 5, 6
# prior 1-sigma horizontal error per fix class for receivers that report no accuracy (consumer-grade classes, m)
PRIOR_SIGMA_H = {FIX_2D: 10.0, FIX_3D: 5.0, FIX_DGPS: 2.0, FIX_RTK_FLOAT: 0.5, FIX_RTK_FIXED: 0.05}
MEDIAN_CHI2_2 = 2.0 * math.log(2.0)          # median of a chi-square with 2 dof


@dataclass
class GnssFix:
    t: float
    lat: float
    lon: float
    alt: float = float("nan")
    mode: Optional[int] = None        # normalised fix mode (FIX_*); None: unknown (treated as a 3-D fix)
    num_sats: Optional[int] = None    # satellites used; None or 0: not reported
    hdop: Optional[float] = None
    sigma_h: Optional[float] = None   # receiver's own horizontal accuracy (1 sigma, m) when it reports one
    sigma_v: Optional[float] = None

    @classmethod
    def from_obs(cls, d: dict) -> "GnssFix":
        g = lambda k: d.get(k) if d.get(k) is None or not isinstance(d.get(k), float) or math.isfinite(d.get(k)) else None
        return cls(t=float(d["t"]), lat=float(d["lat"]), lon=float(d["lon"]),
                   alt=float(d["alt"]) if d.get("alt") is not None else float("nan"),
                   mode=None if d.get("mode") is None else int(d["mode"]),
                   num_sats=None if g("num_sats") is None else int(d["num_sats"]),
                   hdop=g("hdop"), sigma_h=g("sigma_h"), sigma_v=g("sigma_v"))


@dataclass
class GnssModelConfig:
    uere: float = 3.0                   # user-equivalent range error (m) for receivers that report HDOP only
    ref_sats: int = 8                   # satellites at which the mode prior applies; fewer inflate it by sqrt(ref/n)
    sigma_v_factor: float = 2.0         # vertical / horizontal noise of a fix without its own vertical accuracy
    min_scale: float = 0.5              # bounds of the posterior rescaling of the prior noise
    max_scale: float = 20.0


class GnssNoiseModel:
    """Prior noise of a fix from what the receiver reports, times an online scale learned from posterior residuals."""

    def __init__(self, cfg: GnssModelConfig = None):
        self.cfg = cfg or GnssModelConfig()
        self.scale = 1.0
        self.n_posterior = 0

    def prior_sigma_h(self, fix: GnssFix) -> Optional[float]:
        c = self.cfg
        if fix.mode is not None and fix.mode < FIX_2D:
            return None
        if not (math.isfinite(fix.lat) and math.isfinite(fix.lon)):
            return None
        if fix.sigma_h is not None and fix.sigma_h > 0:
            return float(fix.sigma_h)
        if fix.hdop is not None and fix.hdop > 0:
            return c.uere * float(fix.hdop)
        s = PRIOR_SIGMA_H.get(FIX_3D if fix.mode is None else fix.mode, PRIOR_SIGMA_H[FIX_3D])
        if fix.num_sats:                  # 0 / None: not reported
            if fix.num_sats < 4 and (fix.mode is None or fix.mode >= FIX_3D):
                s *= 4.0
            s *= math.sqrt(max(1.0, c.ref_sats / float(fix.num_sats)))
        return s

    def sigma(self, fix: GnssFix):
        """(sigma_h, sigma_v) in metres, or None when the fix is unusable (no fix)."""
        sh = self.prior_sigma_h(fix)
        if sh is None:
            return None
        sv = fix.sigma_v if (fix.sigma_v is not None and fix.sigma_v > 0) else self.cfg.sigma_v_factor * sh
        return sh * self.scale, sv * self.scale

    def update_posterior(self, normalized_sq: np.ndarray, weight: float = 0.5):
        """Rescale the prior from the normalised squared horizontal residuals (2 dof) of the used fixes at a solution of
        the pose graph (robust: median / median of chi^2_2), smoothed by `weight`."""
        e = np.asarray(normalized_sq, float)
        e = e[np.isfinite(e)]
        if e.size < 10:
            return self.scale
        s2 = float(np.median(e) / MEDIAN_CHI2_2)
        target = self.scale * math.sqrt(max(s2, 1e-6))
        target = min(max(target, self.cfg.min_scale), self.cfg.max_scale)
        self.scale = float(math.exp((1 - weight) * math.log(self.scale) + weight * math.log(target)))
        self.n_posterior += int(e.size)
        return self.scale

    def state(self) -> dict:
        return {"scale": self.scale, "n_posterior": self.n_posterior}

    def load_state(self, s: dict):
        self.scale = float(s.get("scale", 1.0))
        self.n_posterior = int(s.get("n_posterior", 0))


def fit_rigid_2d(P: np.ndarray, Q: np.ndarray, w: np.ndarray):
    """Weighted least-squares rigid transform Q ~ R P + t in 2-D; returns (R 2x2, t 2)."""
    w = np.asarray(w, float)
    ws = w.sum()
    mp = (w[:, None] * P).sum(0) / ws
    mq = (w[:, None] * Q).sum(0) / ws
    H = ((P - mp) * w[:, None]).T @ (Q - mq)
    ang = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return R, mq - R @ mp


@dataclass
class GnssGateConfig:
    confidence: float = 0.999          # chi-square level (None in GeoConfig: the verified loop closure's)
    window_s: float = 20.0             # time span of the relative-consistency window (the robot's track is drift free
                                       # over it; long enough to see a receiver hold a stale position while moving)
    window: int = 40                   # at most this many fixes in the window (decimated in time)
    min_fixes: int = 5                 # fixes an epoch must hold (passing the relative test) before any is used
    holdoff_s: float = 10.0            # and the epoch's age (receivers converge for a while after reacquisition)
    gap_factor: float = 5.0            # a gap longer than gap_factor x the median fix interval starts a new epoch
    min_gap_s: float = 1.0
    min_spread_for_yaw: float = 0.0    # (diagnostic) spread of the window's track below which its yaw is not fitted


@dataclass
class GnssDecision:
    used: bool
    reason: str
    enu: Optional[np.ndarray] = None   # (3,) ENU position of the fix
    sigma_h: float = float("nan")      # horizontal 1-sigma after inflation (m)
    sigma_v: float = float("nan")
    inflation: float = 1.0             # variance inflation from the relative test
    rel_nis: float = float("nan")      # normalised squared residual of this fix in its window (after inflation)
    abs_nis: float = float("nan")      # normalised squared innovation against the prediction
    drift_reset: bool = False

    def cov(self) -> np.ndarray:
        return np.diag([self.sigma_h ** 2, self.sigma_h ** 2, self.sigma_v ** 2])


class GnssGate:
    """Online quality control of a GNSS stream (see the module docstring).  Feed every fix in time order with
    - `enu`: the fix in the local ENU frame,
    - `track_xy`: the robot's position at the fix time in its own (map) frame, levelled to 2-D (drift free over a
      window: odometry or the tracked pose),
    - `pred`: optional (pred_xy (2,), P (2, 2)) predicted ENU position and its covariance; None: no absolute gate (the
      map is not anchored yet)."""

    def __init__(self, noise: GnssNoiseModel = None, cfg: GnssGateConfig = None):
        self.noise = noise or GnssNoiseModel()
        self.cfg = cfg or GnssGateConfig()
        self.k2 = float(chi2.ppf(self.cfg.confidence, 2))
        self.k1 = math.sqrt(float(chi2.ppf(self.cfg.confidence, 1)))
        self.n_rel_reject, self.t_rel_reject = 0, None
        self.win = deque()                 # (t, enu_xy, track_xy, sigma_h) of the current epoch
        self.epoch_t0 = None
        self.last_t = None
        self.dts = deque(maxlen=200)
        self.offsets = deque()             # (innovation xy, S, track_xy) of consistent fixes rejected by the absolute gate
        self.stats = {"fixes": 0, "used": 0, "no_fix": 0, "holdoff": 0, "relative": 0, "absolute": 0, "drift": 0,
                      "stale": 0, "epochs": 0}

    def _gap(self) -> float:
        if not self.dts:
            return max(self.cfg.min_gap_s, 5.0)
        return max(self.cfg.min_gap_s, self.cfg.gap_factor * float(np.median(self.dts)))

    def reset_epoch(self, t0=None):
        self.epoch_t0 = t0
        self.win.clear()
        self.offsets.clear()
        self.stats["epochs"] += 1

    def process(self, fix: GnssFix, enu: np.ndarray, track_xy: np.ndarray, pred=None) -> GnssDecision:
        cfg = self.cfg
        self.stats["fixes"] += 1
        sig = self.noise.sigma(fix)
        if sig is None or not np.all(np.isfinite(enu[:2])):
            self.stats["no_fix"] += 1
            return GnssDecision(False, "no_fix")
        sh, sv = sig
        if self.last_t is not None:
            dt = float(fix.t - self.last_t)
            if dt > 0:
                if dt > self._gap():
                    self.reset_epoch(float(fix.t))
                else:
                    self.dts.append(dt)
        else:
            self.stats["epochs"] += 1
            self.epoch_t0 = float(fix.t)
        self.last_t = float(fix.t)
        enu = np.asarray(enu, float)
        txy = np.asarray(track_xy, float)[:2]
        # ---- 1. relative consistency of the window (leave-one-out for the new fix) ----
        prev = list(self.win)
        inflation, rel_nis = 1.0, float("nan")
        ok_rel = True
        if len(prev) >= 2:
            P = np.array([p[2] for p in prev])
            Q = np.array([p[1] for p in prev])
            S2 = np.array([p[3] for p in prev]) ** 2
            w = 1.0 / S2
            R, t = fit_rigid_2d(P, Q, w)
            res = Q - (P @ R.T + t)
            e = (res ** 2).sum(1) / S2
            dof_corr = max(2.0 * len(prev) - 3.0, 1.0) / (2.0 * len(prev))   # residual dof of the fit per point
            inflation = max(1.0, float(np.median(e)) / (MEDIAN_CHI2_2 * dof_corr))
            # scale of the fixes' displacements relative to the track's: ~1 for a working receiver, ~0 for one that
            # holds a stale position while the robot moves (a residual-size test cannot see that within the noise)
            mp = (w[:, None] * P).sum(0) / w.sum()
            mq = (w[:, None] * Q).sum(0) / w.sum()
            Rp = (P - mp) @ R.T
            info = float((w * (Rp ** 2).sum(1)).sum())
            if info > 0 and len(prev) >= 3:
                scale = float((w * (Rp * (Q - mq)).sum(1)).sum()) / info
                z2 = (scale - 1.0) ** 2 * info / inflation
                if z2 > self.k1 ** 2:
                    # the window as a whole contradicts the robot's motion: start a new epoch with this fix
                    self.stats["stale"] += 1
                    self.reset_epoch(float(fix.t))
                    self.win.append((float(fix.t), enu[:2].copy(), txy.copy(), sh))
                    self.n_rel_reject = 0
                    return GnssDecision(False, "stale", enu, sh, sv, inflation, float("nan"))
            r_new = enu[:2] - (R @ txy + t)
            rel_nis = float((r_new ** 2).sum() / (sh ** 2 * inflation))
            ok_rel = rel_nis <= self.k2
        if not ok_rel:
            # the new fix does not fit the window: it does not enter the window, so one jump does not inflate the
            # noise of the fixes that follow; a run of rejections means the window itself is wrong: new epoch
            self.stats["relative"] += 1
            self.n_rel_reject += 1
            if self.n_rel_reject == 1:
                self.t_rel_reject = float(fix.t)
            if self.n_rel_reject >= cfg.min_fixes and fix.t - self.t_rel_reject >= cfg.window_s / 4:
                self.reset_epoch(float(fix.t))
                self.win.append((float(fix.t), enu[:2].copy(), txy.copy(), sh))
                self.n_rel_reject = 0
            return GnssDecision(False, "relative", enu, sh * math.sqrt(inflation), sv * math.sqrt(inflation),
                                inflation, rel_nis)
        self.n_rel_reject = 0
        if not self.win or fix.t - self.win[-1][0] >= cfg.window_s / cfg.window:
            self.win.append((float(fix.t), enu[:2].copy(), txy.copy(), sh))
        while self.win and (fix.t - self.win[0][0] > cfg.window_s or len(self.win) > cfg.window):
            self.win.popleft()
        sh_i, sv_i = sh * math.sqrt(inflation), sv * math.sqrt(inflation)
        # ---- 2. reacquisition hold-off ----
        if len(self.win) < cfg.min_fixes or (self.epoch_t0 is not None and fix.t - self.epoch_t0 < cfg.holdoff_s):
            self.stats["holdoff"] += 1
            return GnssDecision(False, "holdoff", enu, sh_i, sv_i, inflation, rel_nis)
        # while the robot moves (further than the fix noise within the window), the epoch's fixes are used only once
        # the window shows that they follow its motion: the scale test above must detect a stale receiver (scale 0)
        # at the chi-square level with one sigma of margin (~84 % power)
        P = np.array([p[2] for p in self.win])
        w = 1.0 / np.array([p[3] for p in self.win]) ** 2
        Pc = P - (w[:, None] * P).sum(0) / w.sum()
        if float(np.linalg.norm(P.max(0) - P.min(0))) > sh:
            info = float((w * (Pc ** 2).sum(1)).sum()) / inflation
            if info < (self.k1 + 1.0) ** 2:
                self.stats["holdoff"] += 1
                return GnssDecision(False, "unverified", enu, sh_i, sv_i, inflation, rel_nis)
        # ---- 3. absolute gate ----
        if pred is None:
            self.stats["used"] += 1
            return GnssDecision(True, "unanchored", enu, sh_i, sv_i, inflation, rel_nis)
        pxy, Pc = np.asarray(pred[0], float)[:2], np.asarray(pred[1], float)[:2, :2]
        nu = enu[:2] - pxy
        S = sh_i ** 2 * np.eye(2) + Pc
        abs_nis = float(nu @ np.linalg.solve(S, nu))
        if abs_nis <= self.k2:
            self.offsets.clear()
            self.stats["used"] += 1
            return GnssDecision(True, "ok", enu, sh_i, sv_i, inflation, rel_nis, abs_nis)
        # consistent fixes, all offset from the prediction by the same vector while the robot moved further than the
        # offset: the prediction drifted, not the receiver
        self.offsets.append((nu, S, txy.copy(), float(fix.t)))
        while self.offsets and len(self.offsets) > cfg.window:
            self.offsets.popleft()
        if len(self.offsets) >= cfg.min_fixes and self.offsets[-1][3] - self.offsets[0][3] >= cfg.window_s / 2:
            nus = np.array([o[0] for o in self.offsets])
            m = nus.mean(0)
            spread = nus - m
            e = np.array([float(d @ np.linalg.solve(o[1] - Pc + 1e-9 * np.eye(2), d)) for d, o in zip(spread, self.offsets)])
            travel = float(np.linalg.norm(self.offsets[-1][2] - self.offsets[0][2]))
            if float(np.median(e)) <= MEDIAN_CHI2_2 * 2.0 and travel >= float(np.linalg.norm(m)):
                self.offsets.clear()
                self.stats["used"] += 1
                self.stats["drift"] += 1
                return GnssDecision(True, "drift", enu, sh_i, sv_i, inflation, rel_nis, abs_nis, drift_reset=True)
        self.stats["absolute"] += 1
        return GnssDecision(False, "absolute", enu, sh_i, sv_i, inflation, rel_nis, abs_nis)

    def state(self) -> dict:
        return {"noise": self.noise.state(), "stats": dict(self.stats)}
