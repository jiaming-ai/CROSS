"""Offline study of GNSS anchoring on a robot's odometry chain (no images): how much each part of the GNSS handling
(quality model, gating, online inflation) helps, across indoor / outdoor sessions.

A causal replay mirrors the CROSS integration (cross.geo + PoseGraph): keyframes every `--kf-dist` metres (or
`--kf-rot` rad) along the odometry; odometry between-factors with the CROSS default noise model; each GNSS fix goes
through the variant's handling and, when used, becomes a unary position factor on the nearest keyframe (carried by the
odometry between the fix time and the keyframe); the map frame is anchored to ENU (cross.geo.anchor) and the pose
graph is optimised (GTSAM LM, map frame, soft vertical-aware prior on the first keyframe) whenever the used fixes since
the last optimisation disagree with the current estimate at the chi-square level, and once at the end.

Variants:
  none      odometry only, anchored to ENU by the first minute of fixes (what the map has without GNSS factors)
  naive     every fix, fixed sigma (the prior of an unknown receiver), Gaussian, no gating
  robust    every fix, Cauchy kernel, posterior noise rescaling, no gating
  gated     cross.geo.gnss.GnssGate (relative consistency, stale receiver, hold-off, absolute gate) + robust + posterior
  gated_noinfl  gated without the online inflation (ablation)

Input: NCLT raw session folders (gps.csv, odometry_mu_100hz.csv, groundtruth_<s>.csv, gps_rtk.csv) or prepared folders
(gnss.txt, odom.txt, gt_body.txt; see the NCLT preparation README).  Output: per session and variant a JSON row, the
trajectories (ENU, lat/lon GeoJSON) for the page.

  python scripts/geo/gnss_pgo_study.py --nclt-raw /data/nclt/raw --sessions 2012-01-08 --out outputs/geo_study
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gtsam  # noqa: E402

from cross.core.config import NoiseModelConfig  # noqa: E402
from cross.geo.anchor import Anchor, AnchorConfig, rot_z, wrap  # noqa: E402
from cross.geo.geodesy import LocalFrame, geojson_linestring  # noqa: E402
from cross.geo.gnss import GnssErrorModel, GnssFix, GnssGate, GnssGateConfig, GnssNoiseModel, MEDIAN_CHI2_2  # noqa: E402
from scipy.stats import chi2  # noqa: E402

NCLT_LAT0, NCLT_LON0 = math.degrees(0.738167915410646), math.degrees(-1.46098650670922)


# ----------------------------------------------------------------------------------------------------------------------
# data
def euler_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty(np.shape(r) + (3, 3))
    R[..., 0, 0] = cy * cp; R[..., 0, 1] = cy * sp * sr - sy * cr; R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp; R[..., 1, 1] = sy * sp * sr + cy * cr; R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp; R[..., 2, 1] = cp * sr; R[..., 2, 2] = cp * cr
    return R


def load_nclt_raw(root: Path, session: str):
    """Odometry (t, R, p) in the odometry frame, GNSS fixes, ground truth (t, p) in the NCLT local frame, RTK fixes.
    NCLT frames: body x forward, y right, z down; odometry / ground-truth frames x north-ish, y east-ish, z down."""
    sdir = root / session if (root / session / "gps.csv").exists() else root / session / session
    od = np.genfromtxt(sdir / "odometry_mu_100hz.csv", delimiter=",")
    od = od[np.all(np.isfinite(od), 1)]
    gps = np.genfromtxt(sdir / "gps.csv", delimiter=",")
    gps = gps[np.isfinite(gps[:, 3]) & np.isfinite(gps[:, 4])]
    gtf = root / f"groundtruth_{session}.csv"
    if not gtf.exists():
        gtf = sdir / f"groundtruth_{session}.csv"
    gt = np.genfromtxt(gtf, delimiter=",")
    gt = gt[np.all(np.isfinite(gt[:, :4]), 1)]
    rtk = np.genfromtxt(sdir / "gps_rtk.csv", delimiter=",") if (sdir / "gps_rtk.csv").exists() else None
    from cross.dataloader.geo import fill_altitude
    gps = gps[np.argsort(gps[:, 0], kind="stable")]
    gps[:, 5] = fill_altitude(gps[:, 0] * 1e-6, gps[:, 5])
    fixes = []
    last = None
    for row in gps:
        t = row[0] * 1e-6
        if last is not None and t - last < 1e-3:
            continue
        last = t
        # NCLT gps.csv: mode is the NMEA message's (2/3 alternate), satellites are not reported: unknown quality
        fixes.append(GnssFix(t=t, lat=math.degrees(row[3]), lon=math.degrees(row[4]),
                             alt=row[5] if np.isfinite(row[5]) else float("nan"), mode=None, num_sats=None))
    odo = {"t": od[:, 0] * 1e-6, "p": od[:, 1:4], "R": euler_to_R(od[:, 4], od[:, 5], od[:, 6])}
    gtd = {"t": gt[:, 0] * 1e-6, "p": gt[:, 1:4]}
    rtkd = None
    if rtk is not None:
        rtk = rtk[np.isfinite(rtk[:, 3]) & np.isfinite(rtk[:, 4])]
        rtkd = {"t": rtk[:, 0] * 1e-6, "mode": rtk[:, 1], "sats": rtk[:, 2], "lat": np.degrees(rtk[:, 3]),
                "lon": np.degrees(rtk[:, 4]), "alt": rtk[:, 5]}
    return odo, fixes, gtd, rtkd


def rtk_good(rtkd):
    """RTK fixes with sky view: >= 5 satellites; sessions whose RTK log reports no satellites (from 2012-09): 3-D fixes."""
    if np.nanmax(rtkd["sats"]) > 0:
        return rtkd["sats"] >= 5
    return rtkd["mode"] >= 3


def gt_to_enu(gtd, rtkd, frame: LocalFrame):
    """Ground truth in ENU: the NCLT local frame is north-east-down near (NCLT_LAT0, NCLT_LON0); its exact placement
    is fitted (4-DOF, robust) to the RTK fixes with satellites >= 5, interpolated ground truth at the fix times."""
    p = gtd["p"]
    approx = np.stack([p[:, 1], p[:, 0], -p[:, 2]], 1)          # NED -> ENU axes
    if rtkd is None:
        return approx, None
    ok = rtk_good(rtkd) & (rtkd["t"] > gtd["t"][0]) & (rtkd["t"] < gtd["t"][-1])
    alt = rtkd["alt"][ok]
    zok = np.isfinite(alt)
    enu = frame.to_enu(rtkd["lat"][ok], rtkd["lon"][ok], np.where(zok, alt, frame.alt0))
    g = np.stack([np.interp(rtkd["t"][ok], gtd["t"], approx[:, i]) for i in range(3)], 1)
    a = Anchor(AnchorConfig(dof=4, max_t_std=100), vertical_map=np.array([0, 0, 1.0]))
    a.fit(g, enu, np.full(len(g), 1.0), np.where(zok, 3.0, 1e4))
    res = np.linalg.norm(a.to_enu(g)[:, :2] - enu[:, :2], axis=1)
    info = {"n_rtk": int(ok.sum()), "yaw_deg": math.degrees(a.yaw), "t": a.t.tolist(),
            "rtk_res_median": float(np.median(res)), "rtk_res_p90": float(np.percentile(res, 90))}
    return a.to_enu(approx), info


def interp_pose(odo, t):
    i = int(np.clip(np.searchsorted(odo["t"], t), 1, len(odo["t"]) - 1))
    a = (t - odo["t"][i - 1]) / max(odo["t"][i] - odo["t"][i - 1], 1e-9)
    p = (1 - a) * odo["p"][i - 1] + a * odo["p"][i]
    return odo["R"][i if a > 0.5 else i - 1], p


# ----------------------------------------------------------------------------------------------------------------------
# replay
def pose3(R, p):
    return gtsam.Pose3(gtsam.Rot3(np.asarray(R, float)), np.asarray(p, float))


class Replay:
    def __init__(self, odo, fixes, frame, variant, args):
        self.odo, self.fixes, self.frame, self.variant, self.args = odo, fixes, frame, variant, args
        self.noise = NoiseModelConfig()
        self.up = np.array([0, 0, -1.0])                                  # z down in the NCLT odometry frame
        self.anchor = Anchor(AnchorConfig(dof=4), vertical_map=self.up)
        gcfg = GnssGateConfig(confidence=args.confidence)
        self.gate = GnssGate(GnssNoiseModel(), gcfg)
        self.err = GnssErrorModel()
        self.drift_flag = False
        self.k2 = float(chi2.ppf(args.confidence, 2))
        self.conf = args.confidence

    def keyframes(self):
        """Keyframe indices into the odometry stream: every kf_dist metres or kf_rot radians."""
        t, p, R = self.odo["t"], self.odo["p"], self.odo["R"]
        idx = [0]
        for i in range(1, len(t)):
            j = idx[-1]
            d = np.linalg.norm(p[i] - p[j])
            ang = math.acos(np.clip((np.trace(R[j].T @ R[i]) - 1) / 2, -1, 1))
            if d >= self.args.kf_dist or ang >= self.args.kf_rot:
                idx.append(i)
        return np.array(idx)

    def odom_sigmas(self, Ra, pa, Rb, pb, n):
        c = self.noise
        L = float(np.linalg.norm(pb - pa))
        th = math.acos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))
        n = max(n, 1)
        st = c.odom_k_t * L / math.sqrt(n) + c.odom_floor_t
        sr = c.odom_k_r * th / math.sqrt(n) + c.odom_floor_r
        return np.array([sr, sr, sr, st, st, st])

    def run(self):
        a = self.args
        odo, fixes = self.odo, self.fixes
        kfi = self.keyframes()
        kt = odo["t"][kfi]
        # current estimate of every keyframe (map frame): odometry until optimised
        est_R = [odo["R"][i] for i in kfi]
        est_p = [odo["p"][i].copy() for i in kfi]
        n_kf = len(kfi)
        online = np.full((n_kf, 3), np.nan)          # ENU position reported when the keyframe was created
        used = []                                    # (kf index, enu (3), sigma_h, sigma_v)
        pending = []
        decisions = []
        n_opt, t_opt = 0, 0.0
        fi = 0
        last_used_t, last_used_dist = None, 0.0
        dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(odo["p"], axis=0), axis=1))])
        anchored_at = None
        first_minute = []
        last_opt_k = -10 ** 9
        for k in range(n_kf):
            # fixes up to this keyframe's time
            while fi < len(fixes) and fixes[fi].t <= kt[k]:
                fx = fixes[fi]
                fi += 1
                if fx.t < odo["t"][0] or fx.t > odo["t"][-1] or k == 0:
                    continue
                enu = self.frame.to_enu(fx.lat, fx.lon, fx.alt if math.isfinite(fx.alt) else 0.0)
                if not math.isfinite(fx.alt):
                    enu = np.array([enu[0], enu[1], np.nan])
                # robot position at the fix time: previous keyframe's estimate + odometry since
                Rf, pf = interp_pose(odo, fx.t)
                j = k - 1
                p_est = est_p[j] + est_R[j] @ (odo["R"][kfi[j]].T @ (pf - odo["p"][kfi[j]]))
                lev = self.anchor.R_level @ pf                              # drift-free local track (raw odometry)
                pred = None
                if self.anchor.ok:
                    d_since = dist[np.searchsorted(odo["t"], fx.t) - 1] - last_used_dist if last_used_t is not None else 1e3
                    sig_d = self.noise.odom_k_t * max(d_since, 0.0) + 0.5
                    P = self.anchor.position_cov_enu(p_est)[:2, :2] + sig_d ** 2 * np.eye(2)
                    pred = (self.anchor.to_enu(p_est)[:2], P)
                dec = self.handle_fix(fx, enu, lev[:2], pred)
                decisions.append((fx.t, dec[0], dec[1]))
                # carry the fix to keyframe j by the odometry between them
                off = est_R[j] @ (odo["R"][kfi[j]].T @ (pf - odo["p"][kfi[j]]))
                sc = self.gate.noise.scale
                if self.variant == "none":
                    if dec[0]:                               # anchor only: gated fixes align the map, no factors
                        first_minute.append((j, enu, dec[2] / sc, dec[3] / sc, off, fx.t))
                    continue
                if dec[1] == "drift":
                    self.drift_flag = True
                if dec[0]:
                    # sigmas without the posterior scale (applied at use, so that rescaling reaches every factor)
                    item = (j, enu, dec[2] / sc, dec[3] / sc, off, fx.t)
                    used.append(item)
                    pending.append(item)
                    last_used_t = fx.t
                    last_used_dist = dist[np.searchsorted(odo["t"], fx.t) - 1]
            # new keyframe k estimate: chained from k-1 by odometry
            if k > 0:
                Ra, pa, Rb, pb = odo["R"][kfi[k - 1]], odo["p"][kfi[k - 1]], odo["R"][kfi[k]], odo["p"][kfi[k]]
                dR = Ra.T @ Rb
                dp = Ra.T @ (pb - pa)
                est_R[k] = est_R[k - 1] @ dR
                est_p[k] = est_p[k - 1] + est_R[k - 1] @ dp
            # anchor (re)fit and the optimisation policy
            if self.variant == "none":
                if first_minute and (not self.anchor.ok or len(first_minute) % 50 == 0):
                    self.fit_anchor(first_minute, est_p)
            elif used:
                if not self.anchor.ok or (len(used) % 50 == 0):
                    self.fit_anchor(used, est_p)
                if self.anchor.ok and anchored_at is None:
                    anchored_at = k
                if (self.anchor.ok and (len(pending) >= 5 or (self.drift_flag and pending)) and k - last_opt_k >= self.args.opt_min_kf
                        and self.drifted(pending, est_p)):
                    last_opt_k = k
                    self.drift_flag = False
                    t0 = time.perf_counter()
                    est_R, est_p = self.optimize(kfi, est_R, est_p, used)
                    t_opt += time.perf_counter() - t0
                    n_opt += 1
                    pending = []
                    self.fit_anchor(used, est_p)
            if self.anchor.ok:
                online[k] = self.anchor.to_enu(est_p[k])
                if anchored_at is None:
                    anchored_at = k
        if self.variant == "none" and first_minute:
            self.fit_anchor(first_minute, est_p)
        if self.variant != "none" and used and self.anchor.ok:
            t0 = time.perf_counter()
            est_R, est_p = self.optimize(kfi, est_R, est_p, used, final=True)
            t_opt += time.perf_counter() - t0
            n_opt += 1
            self.fit_anchor(used, est_p)
        final = self.anchor.to_enu(np.array(est_p)) if self.anchor.ok else np.full((n_kf, 3), np.nan)
        return {"kf_t": kt, "online": online, "final": final, "n_used": len(used), "n_opt": n_opt, "t_opt": t_opt,
                "decisions": decisions, "gate_stats": dict(self.gate.stats), "noise_scale": self.gate.noise.scale,
                "anchored_at_kf": anchored_at, "used": [(u[0], u[5]) for u in used]}

    def handle_fix(self, fx, enu, track_xy, pred):
        """-> (used, reason, sigma_h, sigma_v)"""
        v = self.variant
        if v in ("naive", "robust"):
            s = self.gate.noise.sigma(fx)
            if s is None:
                return False, "no_fix", None, None
            return True, "all", s[0], s[1]
        if v in ("gated_noinfl", "gated_post"):
            # the gate's decisions as in "gated"; factors with the prior noise (gated_post: times the posterior scale,
            # without the relative test's inflation; gated_noinfl: neither)
            d = self.gate.process(fx, enu, track_xy, pred)
            sig = self.gate.noise.sigma(fx)
            return d.used, d.reason, (sig[0] if sig else None), (sig[1] if sig else None)
        d = self.gate.process(fx, enu, track_xy, pred)
        return d.used, d.reason, d.sigma_h, d.sigma_v

    def factor_interval(self) -> float:
        if self.args.factor_interval < 0:
            return float(self.err.tau)
        return float(self.args.factor_interval)

    def fit_anchor(self, used, est_p):
        sc = self.gate.noise.scale
        pm = np.array([est_p[u[0]] + u[4] for u in used])
        pe = np.array([u[1] for u in used])
        sh = np.array([u[2] for u in used]) * sc
        sv = np.array([u[3] for u in used]) * sc
        pe = pe.copy()
        bad_z = ~np.isfinite(pe[:, 2])
        pe[bad_z, 2] = 0.0
        sv = np.where(bad_z, 1e4, sv)
        self.anchor.fit(pm, pe, sh, sv)

    def drifted(self, pending, est_p) -> bool:
        """The used fixes since the last optimisation disagree with the current estimate (chi-square, 2 dof each):
        all of them, or the most recent ones (as cross.geo.manager.GeoManager.should_optimize); a fix whose offset the
        gate attributed to drift triggers at once."""
        if self.drift_flag:
            return True
        sc = self.gate.noise.scale
        es = []
        for (j, enu, sh, sv, off, t) in pending:
            r = enu[:2] - self.anchor.to_enu(est_p[j] + off)[:2]
            es.append(float(r @ r) / (sh * sc) ** 2)
        m = min(len(es), 5)
        return sum(es) > chi2.ppf(self.conf, 2 * len(es)) or sum(es[-m:]) > chi2.ppf(self.conf, 2 * m)

    def optimize(self, kfi, est_R, est_p, used, final=False):
        odo = self.odo
        graph = gtsam.NonlinearFactorGraph()
        init = gtsam.Values()
        n = len(est_p)
        for i in range(n):
            init.insert(i, pose3(est_R[i], est_p[i]))
        # soft vertical-aware prior on the first keyframe: roll / pitch tight (the map's vertical), yaw and position
        # free (the GNSS factors fix them through the anchor)
        v_body = est_R[0].T @ self.up
        Sr = (0.01 ** 2) * np.eye(3) + (1.0 ** 2 - 0.01 ** 2) * np.outer(v_body, v_body)
        cov = np.zeros((6, 6))
        cov[:3, :3] = Sr
        cov[3:, 3:] = (1e3) ** 2 * np.eye(3)
        graph.add(gtsam.PriorFactorPose3(0, pose3(est_R[0], est_p[0]), gtsam.noiseModel.Gaussian.Covariance(cov)))
        for i in range(1, n):
            Ra, pa, Rb, pb = odo["R"][kfi[i - 1]], odo["p"][kfi[i - 1]], odo["R"][kfi[i]], odo["p"][kfi[i]]
            nfr = max(int(round((odo["t"][kfi[i]] - odo["t"][kfi[i - 1]]) * self.args.frame_rate)), 1)
            meas = pose3(Ra.T @ Rb, Ra.T @ (pb - pa))
            graph.add(gtsam.BetweenFactorPose3(i - 1, i, meas, gtsam.noiseModel.Diagonal.Sigmas(self.odom_sigmas(Ra, pa, Rb, pb, nfr))))
        robust = self.variant != "naive"
        A = self.anchor
        res_keys = []
        sc = self.gate.noise.scale
        all_used = used
        fi = self.factor_interval()
        if self.variant not in ("naive",) and fi > 0:
            kept, last_t = [], -1e18
            for u in used:
                if u[5] - last_t >= fi:
                    kept.append(u)
                    last_t = u[5]
            used = kept
        for (j, enu, sh, sv, off, t) in used:
            sh, sv = sh * sc, sv * sc
            if self.args.no_altitude:
                enu = np.array([enu[0], enu[1], np.nan])
            target = A.to_map(np.array([enu[0], enu[1], enu[2] if np.isfinite(enu[2]) else 0.0]))
            # factor on keyframe j's position: the fix minus the odometry offset between keyframe and fix
            Rme = A.R.T      # ENU -> map
            Cenu = np.diag([sh ** 2, sh ** 2, (sv if np.isfinite(enu[2]) else 1e3) ** 2])
            C = Rme @ Cenu @ Rme.T + 1e-6 * np.eye(3)
            base = gtsam.noiseModel.Gaussian.Covariance(C)
            nm = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy.Create(math.sqrt(self.k2)), base) if robust else base
            graph.add(gtsam.GPSFactor(j, target - off, nm))
            res_keys.append((j, target - off, C))
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(50)
        result = gtsam.LevenbergMarquardtOptimizer(graph, init, params).optimize()
        new_R = [result.atPose3(i).rotation().matrix() for i in range(n)]
        new_p = [result.atPose3(i).translation() for i in range(n)]
        if robust and all_used:
            # posterior rescaling of the GNSS noise: horizontal residuals of every used fix at the solution
            e, r2 = [], []
            for (j, enu, sh, sv, off, t) in all_used:
                r = (enu - A.to_enu(new_p[j] + off))[:2]
                e.append(float(r @ r) / (sh * sc) ** 2)
                r2.append(float(r @ r))
            if self.variant != "gated_noinfl" and len(e) >= 20:
                self.gate.noise.update_posterior(np.array(e))
            # correlation time of the receiver error from the posterior residuals
            ep = np.zeros(len(all_used), int)
            ts = np.array([u[5] for u in all_used])
            if len(ts) > 1:
                gap = max(5.0 * float(np.median(np.diff(ts))), 1.0)
                ep = np.concatenate([[0], np.cumsum(np.diff(ts) > gap)])
            rr = np.array([(enu - A.to_enu(new_p[j] + off))[:2] for (j, enu, sh, sv, off, t) in all_used])
            self.err.update_from_residuals(ts, rr, ep)
        return new_R, new_p


# ----------------------------------------------------------------------------------------------------------------------
def evaluate(res, gt_enu, gtd, sky):
    kt = res["kf_t"]
    g = np.stack([np.interp(kt, gtd["t"], gt_enu[:, i]) for i in range(3)], 1)
    inside = (kt >= gtd["t"][0]) & (kt <= gtd["t"][-1])
    out = {}
    for name in ("online", "final"):
        P = res[name]
        ok = inside & np.all(np.isfinite(P[:, :2]), 1)
        if ok.sum() < 10:
            out[name] = None
            continue
        e = np.linalg.norm(P[ok, :2] - g[ok, :2], axis=1)
        # shape: after the best 2-D rigid alignment
        from cross.geo.gnss import fit_rigid_2d
        Rr, tt = fit_rigid_2d(P[ok, :2], g[ok, :2], np.ones(ok.sum()))
        es = np.linalg.norm(P[ok, :2] @ Rr.T + tt - g[ok, :2], axis=1)
        sk = sky(kt[ok])
        dz = P[ok, 2] - g[ok, 2]
        dz = dz[np.isfinite(dz)]
        row = {"vert_rmse": float(np.sqrt(((dz - np.median(dz)) ** 2).mean())) if len(dz) else None,
               "vert_range_est": float(np.ptp(P[ok, 2])) if ok.sum() else None, "vert_range_gt": float(np.ptp(g[ok, 2])),
               "ate_geo_rmse": float(np.sqrt((e ** 2).mean())), "ate_geo_median": float(np.median(e)),
               "ate_geo_p95": float(np.percentile(e, 95)), "ate_geo_max": float(e.max()),
               "ate_shape_rmse": float(np.sqrt((es ** 2).mean())), "n": int(ok.sum())}
        if (~sk).sum() > 0:
            row["indoor_rmse"] = float(np.sqrt((e[~sk] ** 2).mean()))
            row["indoor_max"] = float(e[~sk].max())
            row["indoor_frac"] = float((~sk).mean())
        row["outdoor_rmse"] = float(np.sqrt((e[sk] ** 2).mean())) if sk.sum() else None
        out[name] = row
        out[name + "_err"] = e
        out[name + "_ok"] = ok
    return out


def sky_visibility(rtkd, gtd):
    """Sky-visibility label from the RTK receiver (a proxy for outdoor): a good fix (rtk_good) within 3 s, and
    indoor stretches shorter than 10 s count as outdoor (gaps of the receiver)."""
    if rtkd is None:
        return lambda t: np.ones(len(t), bool)
    good = rtkd["t"][rtk_good(rtkd)]

    def f(t):
        t = np.asarray(t)
        i = np.clip(np.searchsorted(good, t), 1, len(good) - 1)
        d = np.minimum(np.abs(good[i] - t), np.abs(good[i - 1] - t))
        out = d <= 3.0
        # fill short indoor runs
        idx = np.where(~out)[0]
        if len(idx):
            runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for r in runs:
                if t[r[-1]] - t[r[0]] < 10.0:
                    out[r] = True
        return out
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nclt-raw", type=Path, required=True)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["none", "naive", "robust", "gated", "gated_noinfl"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kf-dist", type=float, default=1.0)
    ap.add_argument("--kf-rot", type=float, default=0.26)
    ap.add_argument("--frame-rate", type=float, default=5.0, help="camera frames per second (odometry noise model)")
    ap.add_argument("--confidence", type=float, default=0.999)
    ap.add_argument("--factor-interval", type=float, default=0.0,
                    help="seconds between GNSS factors (0: every used fix; -1: from the online error model)")
    ap.add_argument("--tag", default="", help="suffix of the variant names in the output")
    ap.add_argument("--no-altitude", action="store_true", help="horizontal GNSS factors only (no altitude)")
    ap.add_argument("--opt-min-kf", type=int, default=25, help="keyframes between two optimisations (rate limit)")
    ap.add_argument("--max-time", type=float, default=0.0, help="seconds of the session to use (0: all)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = LocalFrame(NCLT_LAT0, NCLT_LON0, 270.0)
    for s in args.sessions:
        t0 = time.time()
        odo, fixes, gtd, rtkd = load_nclt_raw(args.nclt_raw, s)
        if args.max_time > 0:
            tmax = odo["t"][0] + args.max_time
            m = odo["t"] <= tmax
            odo = {k: v[m] for k, v in odo.items()}
            fixes = [f for f in fixes if f.t <= tmax]
        gt_enu, gtinfo = gt_to_enu(gtd, rtkd, frame)
        sky = sky_visibility(rtkd, gtd)
        sdir = args.out / s
        sdir.mkdir(exist_ok=True)
        rows = {}
        geo = {"type": "FeatureCollection", "features": []}
        step = max(1, len(gtd["t"]) // 4000)
        geo["features"].append(geojson_linestring(frame.to_lla(gt_enu[::step]), {"name": "ground truth", "session": s}))
        # raw fixes (sub-sampled) for the map view
        fe = np.array([frame.to_enu(f.lat, f.lon, 0.0) for f in fixes[::5]])
        geo["features"].append(geojson_linestring(frame.to_lla(fe), {"name": "raw GNSS", "session": s}))
        for v in args.variants:
            tv = time.time()
            rep = Replay(odo, fixes, frame, v, args)
            res = rep.run()
            ev = evaluate(res, gt_enu, gtd, sky)
            row = {k: ev[k] for k in ("online", "final")}
            row.update({"n_used": res["n_used"], "n_fixes": len(fixes), "n_kf": len(res["kf_t"]), "n_opt": res["n_opt"],
                        "t_opt_s": res["t_opt"], "gate": res["gate_stats"], "noise_scale": res["noise_scale"],
                        "anchored_at_kf": res["anchored_at_kf"], "wall_s": time.time() - tv,
                        "err_model": rep.err.state(), "factor_interval": rep.factor_interval()})
            v = v + args.tag
            rows[v] = row
            P = res["final"] if np.isfinite(res["final"]).all() else res["online"]
            okp = np.all(np.isfinite(P[:, :2]), 1)
            geo["features"].append(geojson_linestring(frame.to_lla(np.nan_to_num(P[okp])), {"name": v, "session": s}))
            np.savez_compressed(sdir / f"traj_{v}.npz", kf_t=res["kf_t"], online=res["online"], final=res["final"],
                                err_final=ev.get("final_err", np.array([])), ok_final=ev.get("final_ok", np.array([])),
                                sky=sky(res["kf_t"]), dec_t=np.array([d[0] for d in res["decisions"]]),
                                dec_used=np.array([d[1] for d in res["decisions"]]),
                                dec_reason=np.array([d[2] for d in res["decisions"]]))
            f = row["final"] or {}
            print(f"{s} {v:13s} geo RMSE {f.get('ate_geo_rmse', float('nan')):7.2f} m  shape {f.get('ate_shape_rmse', float('nan')):6.2f}"
                  f"  indoor {f.get('indoor_rmse', float('nan')):6.2f}  used {res['n_used']}/{len(fixes)}  opt {res['n_opt']}"
                  f" ({res['t_opt']:.1f}s)  {time.time() - tv:.0f}s", flush=True)
        summary = {"session": s, "gt_alignment": gtinfo, "n_odom": int(len(odo["t"])),
                   "duration_s": float(odo["t"][-1] - odo["t"][0]),
                   "length_m": float(np.linalg.norm(np.diff(odo["p"], axis=0), axis=1).sum()),
                   "variants": rows, "wall_s": time.time() - t0}
        (sdir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        (sdir / "tracks.geojson").write_text(json.dumps(geo))


if __name__ == "__main__":
    main()
