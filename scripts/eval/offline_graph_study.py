#!/usr/bin/env python3
"""Offline loop-closure study on a dumped pose graph (scripts/lc/graph_io.py format, ground truth included).

1. Ground-truth labelling of every visual edge (translation / rotation error of the measurement).
2. Noise-model calibration: visual edges as a function of distance (and covisibility), odometry edges as a
   function of length / number of frames.
3. Prior-consistency test of loop candidates (same-session edges whose endpoints are >= --loop-gap steps apart):
   the measurement is compared with the dead-reckoned relative pose along the odometry chain between the two
   keyframes, with the chain covariance compounded through the adjoints (right perturbations), chi^2 with 6 dof.
   Pairwise (cycle) consistency of loop edges that share a map segment and a query segment (PCM-style).
4. Posterior test: pose-graph optimisation from the odometry initialisation with different noise models and
   robust back ends (Gaussian, Huber, GNC-TLS), residuals after the optimisation, map ATE (Umeyama SE(3) alignment
   over all keyframes), and injected false loop closures (aliasing model: true relative pose plus a 3-15 m
   horizontal offset and a yaw offset) with their rejection rate.

usage: python scripts/lc/offline_graph_study.py graph_s0.json --out outputs/lcstudy/<scene>/study_s0 [--snr 10]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import gtsam
import numpy as np
from scipy.stats import chi2

CHI2_LEVELS = {"p0.99": chi2.ppf(0.99, 6), "p0.999": chi2.ppf(0.999, 6), "p1-1e-6": chi2.ppf(1 - 1e-6, 6)}


# ----------------------------------------------------------------------------- Lie helpers (gtsam order: rot, trans)
def pose7(p):
    return gtsam.Pose3(gtsam.Rot3.Quaternion(p[6], p[3], p[4], p[5]), gtsam.Point3(p[0], p[1], p[2]))


def pose16(m):
    m = np.asarray(m, dtype=np.float64).reshape(4, 4)
    return gtsam.Pose3(gtsam.Rot3(m[:3, :3]), gtsam.Point3(m[0, 3], m[1, 3], m[2, 3]))


def sig_rt(std6):
    """pypose std order (tx ty tz rx ry rz) -> gtsam order (rx ry rz tx ty tz)."""
    s = np.asarray(std6, dtype=np.float64)
    return np.array([s[3], s[4], s[5], s[0], s[1], s[2]])


def logmap(T):
    return np.asarray(gtsam.Pose3.Logmap(T), dtype=np.float64)


def residual(T_meas, T_ref):
    """r = Log(T_meas^-1 T_ref) (right perturbation of the measurement), gtsam order (rot, trans)."""
    return logmap(T_meas.between(T_ref))


def chi2_of(r, cov):
    return float(r @ np.linalg.solve(cov, r))


def umeyama_ate(est: dict, gt: dict):
    ids = [i for i in est if i in gt]
    if len(ids) < 3:
        return float("nan")
    src = np.array([est[i].translation() for i in ids]); dst = np.array([gt[i].translation() for i in ids])
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_d - Rm @ mu_s
    return float(np.sqrt(np.mean(np.sum(((Rm @ src.T).T + t - dst) ** 2, axis=1))))


# ----------------------------------------------------------------------------- noise models
class NoiseModels:
    """Odometry / visual covariance models used by the study (all return sigmas in gtsam order)."""

    def __init__(self, fit: dict, snr: float | None):
        self.fit = fit
        self.snr = snr

    # --- odometry ---
    def odom_system(self, e):
        return sig_rt(e["std"])

    def odom_fitted(self, e, L, theta, n_frames, inflate=1.0):
        """sigma_t = k_t L / sqrt(n) + floor_t, sigma_r = k_r theta / sqrt(n) + floor_r (noise proportional to the
        motion of every integrated reading, n readings per edge), times an inflation factor."""
        f = self.fit["odom"]
        n = max(int(n_frames or 1), 1)
        st = (f["k_t"] * L / math.sqrt(n) + f["floor_t"]) * inflate
        sr = (f["k_r"] * theta / math.sqrt(n) + f["floor_r"]) * inflate
        return np.array([sr, sr, sr, st, st, st])

    def odom_oracle(self, e, L, theta, n_frames):
        """Simulated SNR noise: per frame sigma = |motion| / (snr sqrt(3)) per axis; edge = n frames of L/n each."""
        n = max(int(n_frames or 1), 1)
        st = max(L / (self.snr * math.sqrt(3) * math.sqrt(n)), 1e-3)
        sr = max(theta / (self.snr * math.sqrt(3) * math.sqrt(n)), 1e-3)
        return np.array([sr, sr, sr, st, st, st])

    # --- visual ---
    def visual_system(self, e):
        return sig_rt(e["std"])

    def visual_fitted(self, e, d, conf):
        f = self.fit["visual"]
        st = f["t_a"] + f["t_b"] * d
        sr = f["r_a"] + f["r_b"] * d
        m = 1.0
        if conf is not None and f.get("conf_ref"):
            # low covisibility widens the distribution (measured per bin); interpolate the multiplier
            m = float(np.interp(conf, f["conf_ref"]["c"], f["conf_ref"]["m"]))
        return np.array([sr * m, sr * m, sr * m, st * m, st * m, st * m])

    def map_sigma(self, d):
        """Local consistency of the stored map between two keyframes d metres apart (fitted from the map's GT)."""
        f = self.fit.get("map", {"t_a": 0.05, "t_b": 0.01, "r_a": 0.005, "r_b": 0.001})
        st = f["t_a"] + f["t_b"] * d; sr = f["r_a"] + f["r_b"] * d
        return np.array([sr, sr, sr, st, st, st])


# ----------------------------------------------------------------------------- study
def robust_sigma(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 5:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


P90_3DOF = math.sqrt(chi2.ppf(0.90, 3))   # |e| of a 3-dof isotropic Gaussian: p90 = 2.5 sigma


def tail_sigma(norms):
    """Per-axis sigma such that the 90th percentile of the 3-vector norm matches (tail-calibrated)."""
    return float(np.percentile(norms, 90) / P90_3DOF) if len(norms) else float("nan")


def fit_scale_model(d, norms, bins, min_n=20):
    """sigma(d) = a + b d, a, b >= 0, fitted (weighted NNLS) to tail-calibrated per-bin sigmas of the residual norms."""
    from scipy.optimize import nnls
    xs, ys, ws = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() >= min_n:
            xs.append(float(np.median(d[m]))); ys.append(tail_sigma(norms[m])); ws.append(math.sqrt(m.sum()))
    xs, ys, ws = np.asarray(xs), np.asarray(ys), np.asarray(ws)
    if len(xs) == 0:
        return {"a": 0.05, "b": 0.0, "bins": []}
    if len(xs) == 1:
        return {"a": float(ys[0]), "b": 0.0, "bins": [{"d": float(xs[0]), "sigma": float(ys[0])}]}
    A = np.stack([np.ones_like(xs), xs], 1) * ws[:, None]
    coef, _ = nnls(A, ys * ws)
    return {"a": float(coef[0]), "b": float(coef[1]), "bins": [{"d": float(x), "sigma": float(y)} for x, y in zip(xs, ys)]}


class GraphStudy:
    def __init__(self, path, snr=None, loop_gap=200, local_gap=100, seed=0, false_t=1.0, false_r=10.0):
        self.false_t, self.false_r = false_t, false_r
        g = json.loads(Path(path).read_text())
        self.meta = g["meta"]
        self.snr = snr if snr is not None else g["meta"].get("snr")
        self.nodes = {n["id"]: n for n in g["nodes"]}
        self.gt = {n["id"]: pose16(n["gt"]) for n in g["nodes"] if n.get("gt") is not None}
        self.step = {n["id"]: n.get("step_created") for n in g["nodes"]}
        self.session = {n["id"]: n.get("session", 0) for n in g["nodes"]}
        self.odom = g["odom"]
        self.visual = g["visual"]
        self.loop_gap, self.local_gap = loop_gap, local_gap
        self.rng = np.random.default_rng(seed)
        self.next_of = {e["a"]: (e["b"], e) for e in self.odom}
        for e in self.odom:
            e["T"] = pose7(e["mean"])
            a, b = e["a"], e["b"]
            e["L"] = float(np.linalg.norm(np.asarray(e["mean"][:3])))
            e["theta"] = float(np.linalg.norm(logmap(e["T"])[:3]))
            e["n_frames"] = (self.step[b] - self.step[a]) if (self.step.get(a) is not None and self.step.get(b) is not None) else None
            if a in self.gt and b in self.gt:
                e["r_gt"] = residual(e["T"], self.gt[a].between(self.gt[b]))
        for e in self.visual:
            e["T"] = pose7(e["mean"])
            a, b = e["a"], e["b"]
            e["d"] = float(np.linalg.norm(np.asarray(e["mean"][:3])))
            same = self.session.get(a) == self.session.get(b)
            gap = abs(self.step[b] - self.step[a]) if (same and self.step.get(a) is not None and self.step.get(b) is not None) else None
            e["gap"] = gap
            e["kind"] = "cross" if not same else ("unknown" if gap is None else ("local" if gap < local_gap else ("loop" if gap >= loop_gap else "mid")))
            if a in self.gt and b in self.gt:
                r = residual(e["T"], self.gt[a].between(self.gt[b]))
                e["r_gt"] = r
                e["e_r"] = float(np.degrees(np.linalg.norm(r[:3])))
                e["e_t"] = float(np.linalg.norm(r[3:]))
                e["gt_false"] = bool(e["e_t"] > self.false_t or e["e_r"] > self.false_r)
            e["injected"] = False

    # ------------------------------------------------------------------ 1 + 2: labels and calibration
    def calibrate(self):
        out = {}
        vis = [e for e in self.visual if "r_gt" in e]
        out["n_visual"] = len(self.visual); out["n_visual_gt"] = len(vis)
        out["kinds"] = {}
        for k in ("local", "mid", "loop", "cross"):
            sel = [e for e in vis if e["kind"] == k]
            if sel:
                et = np.array([e["e_t"] for e in sel]); er = np.array([e["e_r"] for e in sel])
                out["kinds"][k] = {"n": len(sel), "e_t_p50": float(np.median(et)), "e_t_p90": float(np.percentile(et, 90)),
                                   "e_t_p99": float(np.percentile(et, 99)), "e_r_p50": float(np.median(er)), "e_r_p99": float(np.percentile(er, 99)),
                                   "frac_false": float(np.mean([e["gt_false"] for e in sel]))}
        inl = [e for e in vis if not e["gt_false"]]
        d = np.array([e["d"] for e in inl])
        nt = np.array([np.linalg.norm(e["r_gt"][3:]) for e in inl]) if inl else np.zeros(0)
        nr = np.array([np.linalg.norm(e["r_gt"][:3]) for e in inl]) if inl else np.zeros(0)
        bins = [0, 0.5, 1, 2, 4, 8, 16, 32, 64]
        ft = fit_scale_model(d, nt, bins); fr = fit_scale_model(d, nr, bins)
        # covisibility multiplier: tail sigma of the distance-normalised residual per covisibility bin
        conf_ref = None
        cs = np.array([e["conf"] if e.get("conf") is not None else np.nan for e in inl])
        if np.isfinite(cs).sum() > 100:
            norm = nt / (ft["a"] + ft["b"] * d)
            ref = tail_sigma(norm[np.isfinite(cs)])
            cb = [0.15, 0.25, 0.35, 0.5, 0.7, 1.01]; c_mid, mult = [], []
            for lo, hi in zip(cb[:-1], cb[1:]):
                m = np.isfinite(cs) & (cs >= lo) & (cs < hi)
                if m.sum() >= 30:
                    c_mid.append(float(np.median(cs[m]))); mult.append(float(max(tail_sigma(norm[m]) / max(ref, 1e-9), 0.5)))
            if len(c_mid) >= 2:
                conf_ref = {"c": c_mid, "m": mult}
        out["visual_fit"] = {"t_a": ft["a"], "t_b": ft["b"], "t_bins": ft["bins"], "r_a": fr["a"], "r_b": fr["b"], "r_bins": fr["bins"], "conf_ref": conf_ref}
        # local consistency of the stored map (final keyframe poses vs GT) as a function of keyframe distance
        map_ids = [i for i, n in self.nodes.items() if n.get("session", 0) == 0 and i in self.gt]
        if len(map_ids) > 20:
            P = {i: pose7(self.nodes[i]["pose"]) for i in map_ids}
            dd, et, er = [], [], []
            for _ in range(min(20000, 30 * len(map_ids))):
                a, b = self.rng.choice(map_ids, 2, replace=False)
                r = residual(P[a].between(P[b]), self.gt[a].between(self.gt[b]))
                dd.append(float(np.linalg.norm(self.gt[a].between(self.gt[b]).translation()))); et.append(np.linalg.norm(r[3:])); er.append(np.linalg.norm(r[:3]))
            dd, et, er = np.asarray(dd), np.asarray(et), np.asarray(er)
            mt = fit_scale_model(dd, et, [0, 1, 2, 4, 8, 16, 32, 64, 128]); mr = fit_scale_model(dd, er, [0, 1, 2, 4, 8, 16, 32, 64, 128])
            out["map_fit"] = {"t_a": mt["a"], "t_b": mt["b"], "t_bins": mt["bins"], "r_a": mr["a"], "r_b": mr["b"], "r_bins": mr["bins"]}
        # calibration of the system's own std: per-axis |z| quantiles
        z = np.concatenate([np.abs(e["r_gt"] / np.maximum(sig_rt(e["std"]), 1e-6)) for e in inl]) if inl else np.zeros(0)
        out["system_std_abs_z"] = {"p50": float(np.median(z)), "p95": float(np.percentile(z, 95)), "p99": float(np.percentile(z, 99)),
                                   "median_over_0.674": float(np.median(z) / 0.674)} if len(z) else None
        # odometry
        od = [e for e in self.odom if "r_gt" in e]
        if od:
            L = np.array([e["L"] for e in od]); th = np.array([e["theta"] for e in od]); nf = np.array([e["n_frames"] or 1 for e in od], float)
            et = np.array([np.linalg.norm(e["r_gt"][3:]) for e in od]); er = np.array([np.linalg.norm(e["r_gt"][:3]) for e in od])
            # per-unit-motion model: sigma = k * motion / sqrt(n) + floor (tail-calibrated on the 3-dof norms)
            mt = L > 0.05; mr = th > 0.05
            k_t = tail_sigma(et[mt] * np.sqrt(nf[mt]) / L[mt]) if mt.sum() > 20 else 0.03
            k_r = tail_sigma(er[mr] * np.sqrt(nf[mr]) / th[mr]) if mr.sum() > 20 else 0.03
            floor_t = tail_sigma(et[~mt]) if (~mt).sum() > 20 else 0.002
            floor_r = tail_sigma(er[~mr]) if (~mr).sum() > 20 else 0.001
            fo_t = {"a": floor_t, "b": k_t, "bins": []}; fo_r = {"a": floor_r, "b": k_r, "bins": []}
            zo = np.concatenate([np.abs(e["r_gt"] / np.maximum(sig_rt(e["std"]), 1e-6)) for e in od])
            oracle = None
            if self.snr:
                zs = []
                for e in od:
                    s = NoiseModels({}, self.snr).odom_oracle(e, e["L"], e["theta"], e["n_frames"])
                    zs.append(np.abs(e["r_gt"] / s))
                zs = np.concatenate(zs)
                oracle = {"abs_z_p50_over_0.674": float(np.median(zs) / 0.674), "abs_z_p99": float(np.percentile(zs, 99))}
            out["odom"] = {"n": len(od), "L_p50": float(np.median(L)), "n_frames_p50": float(np.median(nf)),
                           "e_t_p50": float(np.median(et)), "e_t_p99": float(np.percentile(et, 99)),
                           "e_r_deg_p50": float(np.degrees(np.median(er))), "e_r_deg_p99": float(np.degrees(np.percentile(er, 99))),
                           "fit": {"k_t": fo_t["b"], "floor_t": fo_t["a"], "k_r": fo_r["b"], "floor_r": fo_r["a"]},
                           "system_std_abs_z": {"p50_over_0.674": float(np.median(zo) / 0.674), "p99": float(np.percentile(zo, 99))},
                           "oracle_snr_model": oracle}
        self.fit = {"visual": out["visual_fit"], "odom": out.get("odom", {}).get("fit", {"k_t": 0.03, "floor_t": 0.002, "k_r": 0.03, "floor_r": 0.001}),
                    "map": out.get("map_fit")}
        self.models = NoiseModels(self.fit, self.snr)
        self.calib = out
        return out

    # ------------------------------------------------------------------ chain covariance
    def chain(self, a, b, odom_model):
        """Dead-reckoned relative pose a->b along the odometry chain and its covariance (right perturbation)."""
        if a == b:
            return gtsam.Pose3(), np.zeros((6, 6))
        edges = []
        cur = a
        for _ in range(len(self.odom) + 1):
            nxt = self.next_of.get(cur)
            if nxt is None:
                return None, None
            edges.append(nxt[1]); cur = nxt[0]
            if cur == b:
                break
        else:
            return None, None
        if cur != b:
            return None, None
        # tails P_k = T_k ... T_n
        tails = [None] * (len(edges) + 1)
        tails[len(edges)] = gtsam.Pose3()
        for k in range(len(edges) - 1, -1, -1):
            tails[k] = edges[k]["T"].compose(tails[k + 1])
        cov = np.zeros((6, 6))
        for k, e in enumerate(edges):
            s = self.odom_sigma(e, odom_model)
            A = tails[k + 1].inverse().AdjointMap()
            cov += A @ np.diag(s ** 2) @ A.T
        return tails[0], cov

    def odom_sigma(self, e, model):
        """model: 'system' | 'oracle' | 'fitted' | 'fitted*<inflation>'"""
        if model == "system":
            return self.models.odom_system(e)
        if model == "oracle":
            return self.models.odom_oracle(e, e["L"], e["theta"], e["n_frames"])
        infl = float(model.split("*")[1]) if "*" in model else 1.0
        return self.models.odom_fitted(e, e["L"], e["theta"], e["n_frames"], inflate=infl)

    def visual_sigma(self, e, model):
        if model == "system":
            return self.models.visual_system(e)
        return self.models.visual_fitted(e, e["d"], e.get("conf"))

    # ------------------------------------------------------------------ 3: prior test
    def anchor_prediction(self, e, om, vm):
        """Cross-session edge a (map) -> b (session): prediction through the latest earlier session keyframe b' that
        has a correct map edge a' -> b' (the anchor), the session chain b' -> b and the map relation a -> a'."""
        b = e["b"]; a = e["a"]
        prev = [x for x in self.visual if x["kind"] == "cross" and x["b"] < b and "r_gt" in x and not x["gt_false"] and not x["injected"]]
        if not prev:
            return None, None
        anc = max(prev, key=lambda x: x["b"])
        T_chain, cov_chain = self.chain(anc["b"], b, om)
        if T_chain is None:
            return None, None
        Ma, Ma2 = pose7(self.nodes[a]["pose"]), pose7(self.nodes[anc["a"]]["pose"])
        T_map = Ma.between(Ma2)
        d_map = float(np.linalg.norm(T_map.translation()))
        T_pred = T_map.compose(anc["T"]).compose(T_chain)
        s_anc = self.visual_sigma(anc, vm); s_map = self.models.map_sigma(d_map)
        # transport the anchor and map terms to the end of the chain (right perturbation of T_pred)
        A1 = T_chain.inverse().AdjointMap()
        A2 = anc["T"].compose(T_chain).inverse().AdjointMap()
        cov = cov_chain + A1 @ np.diag(s_anc ** 2) @ A1.T + A2 @ np.diag(s_map ** 2) @ A2.T
        return T_pred, cov

    def prior_test(self, models=(("system", "system"), ("fitted", "fitted"), ("fitted*2", "fitted"), ("fitted*3", "fitted"), ("oracle", "fitted"))):
        cands = [e for e in self.visual if e["kind"] in ("loop", "cross") and "r_gt" in e]
        out = {"n_loop_candidates": len(cands), "n_same_session": sum(1 for e in cands if e["kind"] == "loop"),
               "n_cross_session": sum(1 for e in cands if e["kind"] == "cross"), "models": {}}
        for om, vm in models:
            key = f"odom={om},visual={vm}"
            chis = []
            for e in cands:
                if e["kind"] == "loop":
                    a, b = (e["a"], e["b"]) if e["a"] < e["b"] else (e["b"], e["a"])
                    T_pred, cov = self.chain(a, b, om)
                    T_meas = e["T"] if e["a"] < e["b"] else e["T"].inverse()
                else:
                    T_pred, cov = self.anchor_prediction(e, om, vm)
                    T_meas = e["T"]
                if T_pred is None:
                    e[f"chi2_prior[{key}]"] = None; continue
                s = self.visual_sigma(e, vm)
                r = residual(T_meas, T_pred)
                c2 = chi2_of(r, cov + np.diag(s ** 2))
                e[f"chi2_prior[{key}]"] = c2
                chis.append((c2, e["gt_false"], e["injected"]))
            arr = np.array([[c, f, i] for c, f, i in chis], dtype=float) if chis else np.zeros((0, 3))
            res = {"n": len(arr)}
            for name, thr in CHI2_LEVELS.items():
                acc = arr[:, 0] <= thr if len(arr) else np.zeros(0, bool)
                tr = arr[:, 1] == 0
                res[name] = {"thr": float(thr), "TPR": float(acc[tr].mean()) if tr.any() else None,
                             "FPR": float(acc[~tr].mean()) if (~tr).any() else None,
                             "n_true": int(tr.sum()), "n_false": int((~tr).sum()),
                             "n_false_accepted": int((acc & ~tr).sum())}
            if len(arr):
                res["chi2_true_quantiles"] = {q: float(np.percentile(arr[arr[:, 1] == 0, 0], q)) for q in (50, 90, 99)} if (arr[:, 1] == 0).any() else None
                res["chi2_false_quantiles"] = {q: float(np.percentile(arr[arr[:, 1] == 1, 0], q)) for q in (1, 10, 50)} if (arr[:, 1] == 1).any() else None
            out["models"][key] = res
        return out

    def pairwise_test(self, vm="fitted", om="fitted", window=40, thr_key="p0.999"):
        """Cycle consistency of pairs of loop edges (i, j) whose query keyframes and map keyframes are both within
        `window` steps: T_i, chain(b_i -> b_j), T_j^-1, chain(a_j -> a_i) should compose to identity."""
        cands = [e for e in self.visual if e["kind"] in ("loop", "cross") and "r_gt" in e]
        thr = CHI2_LEVELS[thr_key]
        for e in cands:
            e["pcm_partners"] = 0; e["pcm_consistent"] = 0
        for i, ei in enumerate(cands):
            for ej in cands[i + 1:]:
                if ei["kind"] != ej["kind"]:
                    continue
                if abs(self.step[ei["b"]] - self.step[ej["b"]]) > window or abs(ei["a"] - ej["a"]) > window:
                    continue
                if ei["b"] == ej["b"] and ei["a"] == ej["a"]:
                    continue
                if ei["kind"] == "cross":
                    Tb, Cb = self.chain(min(ei["b"], ej["b"]), max(ei["b"], ej["b"]), om)
                    if Tb is None:
                        continue
                    if ei["b"] > ej["b"]:
                        A = Tb.inverse().AdjointMap(); Tb, Cb = Tb.inverse(), A @ Cb @ A.T
                    Ta = pose7(self.nodes[ej["a"]]["pose"]).between(pose7(self.nodes[ei["a"]]["pose"]))
                    Ca = np.diag(self.models.map_sigma(float(np.linalg.norm(Ta.translation()))) ** 2)
                    cyc = ei["T"].compose(Tb).compose(ej["T"].inverse()).compose(Ta)
                    r = logmap(cyc); si, sj = self.visual_sigma(ei, vm), self.visual_sigma(ej, vm)
                    c2 = chi2_of(r, np.diag(si ** 2) + np.diag(sj ** 2) + Cb + Ca)
                    ei["pcm_partners"] += 1; ej["pcm_partners"] += 1
                    if c2 <= thr:
                        ei["pcm_consistent"] += 1; ej["pcm_consistent"] += 1
                    continue
                # orient chains along increasing ids
                def seg(x, y):
                    if x <= y:
                        T, C = self.chain(x, y, om); return T, C
                    T, C = self.chain(y, x, om)
                    if T is None:
                        return None, None
                    A = T.inverse().AdjointMap()
                    return T.inverse(), A @ C @ A.T
                Tb, Cb = seg(ei["b"], ej["b"]); Ta, Ca = seg(ej["a"], ei["a"])
                if Tb is None or Ta is None:
                    continue
                # cycle: a_i -> b_i (T_i) -> b_j (Tb) -> a_j (T_j^-1) -> a_i (Ta)
                cyc = ei["T"].compose(Tb).compose(ej["T"].inverse()).compose(Ta)
                r = logmap(cyc)
                si, sj = self.visual_sigma(ei, vm), self.visual_sigma(ej, vm)
                # first-order covariance: sum of the four terms transported to the cycle frame (approximate: no adjoints
                # for the short chains, which are near identity in the loop frame)
                cov = np.diag(si ** 2) + np.diag(sj ** 2) + Cb + Ca
                c2 = chi2_of(r, cov)
                ei["pcm_partners"] += 1; ej["pcm_partners"] += 1
                if c2 <= thr:
                    ei["pcm_consistent"] += 1; ej["pcm_consistent"] += 1
        tr = [e for e in cands if not e["gt_false"]]; fa = [e for e in cands if e["gt_false"]]
        def stats(sel):
            if not sel:
                return None
            return {"n": len(sel), "frac_with_partner": float(np.mean([e["pcm_partners"] > 0 for e in sel])),
                    "frac_supported": float(np.mean([e["pcm_consistent"] > 0 for e in sel])),
                    "frac_supported_of_those_with_partner": float(np.mean([e["pcm_consistent"] > 0 for e in sel if e["pcm_partners"] > 0])) if any(e["pcm_partners"] > 0 for e in sel) else None}
        return {"window": window, "thr": thr_key, "true": stats(tr), "false": stats(fa)}

    # ------------------------------------------------------------------ 4: PGO variants
    def build_graph(self, om, vm, robust=None, sigma_scale=1.0, edge_filter=None):
        graph = gtsam.NonlinearFactorGraph()
        ids = sorted(self.nodes.keys())
        first = ids[0]
        graph.add(gtsam.PriorFactorPose3(first, gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))))
        known = [0]
        for e in self.odom:
            s = self.odom_sigma(e, om) * sigma_scale
            known.append(graph.size())
            graph.add(gtsam.BetweenFactorPose3(e["a"], e["b"], e["T"], gtsam.noiseModel.Diagonal.Sigmas(s)))
        vis_idx = []
        for e in self.visual:
            if edge_filter is not None and not edge_filter(e):
                continue
            s = self.visual_sigma(e, vm) * sigma_scale
            base = gtsam.noiseModel.Diagonal.Sigmas(s)
            if robust == "huber":
                base = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.0), base)
            vis_idx.append((graph.size(), e))
            graph.add(gtsam.BetweenFactorPose3(e["a"], e["b"], e["T"], base))
        return graph, known, vis_idx

    def odom_init(self):
        vals = gtsam.Values()
        ids = sorted(self.nodes.keys())
        cur = gtsam.Pose3(); vals.insert(ids[0], cur); pos = {ids[0]: cur}
        k = ids[0]
        for _ in range(len(self.odom) + 1):
            nxt = self.next_of.get(k)
            if nxt is None:
                break
            cur = cur.compose(nxt[1]["T"]); vals.insert(nxt[0], cur); pos[nxt[0]] = cur; k = nxt[0]
        for i in ids:
            if not vals.exists(i):
                vals.insert(i, gtsam.Pose3())
        return vals, pos

    def solve(self, graph, init, method="lm", known=None, chi2_thr=None):
        t0 = time.time()
        if method == "gnc":
            p = gtsam.GncLMParams()
            p.setLossType(gtsam.GncLossType.TLS)
            if known:
                p.setKnownInliers(known)
            try:
                p.setVerbosityGNC(p.Verbosity.SILENT)
            except Exception:
                pass
            opt = gtsam.GncLMOptimizer(graph, init, p)
            res = opt.optimize()
            w = np.asarray(opt.getWeights())
        else:
            lp = gtsam.LevenbergMarquardtParams(); lp.setMaxIterations(100)
            res = gtsam.LevenbergMarquardtOptimizer(graph, init, lp).optimize()
            w = None
        return res, w, time.time() - t0

    def pgo_variants(self, inject=0, out_dir=None):
        """Compare back ends.  GNC: the TLS inlier threshold of this gtsam build is fixed at factor error 0.5 r^2 <= 1,
        so all sigmas are scaled by sqrt(chi2_thr / 2) to place the threshold at chi2_6(p)."""
        gt = self.gt
        init, dead = self.odom_init()
        results = {"ate_odom_only": umeyama_ate(dead, gt), "ate_online": umeyama_ate({i: pose7(n["pose"]) for i, n in self.nodes.items()}, gt)}
        injected = self.inject_false_loops(inject) if inject else []
        thr = CHI2_LEVELS["p0.999"]
        variants = [
            ("system+huber (current PGO)", "system", "system", "huber", "lm", 1.0, None),
            ("fitted gaussian", "fitted", "fitted", None, "lm", 1.0, None),
            ("fitted huber", "fitted", "fitted", "huber", "lm", 1.0, None),
            ("fitted huber, informative edges only", "fitted", "fitted", "huber", "lm", 1.0, "informative"),
            ("fitted huber, informative, init=online poses", "fitted", "fitted", "huber", "lm", 1.0, "informative:online"),
            ("fitted GNC-TLS", "fitted", "fitted", None, "gnc", math.sqrt(thr / 2.0), None),
            ("fitted prior-gate + GNC-TLS", "fitted", "fitted", None, "gnc", math.sqrt(thr / 2.0), "prior"),
        ]
        key_prior = "chi2_prior[odom=fitted*2,visual=fitted]"
        # information criterion: a visual edge constrains the graph only if the odometry chain between its
        # endpoints is less certain than the measurement (translation covariance trace); cross-session edges always
        for e in self.visual:
            if e["kind"] == "cross" or e["injected"]:
                e["informative"] = True; continue
            a, b = (e["a"], e["b"]) if e["a"] < e["b"] else (e["b"], e["a"])
            _, cov = self.chain(a, b, "fitted")
            sv = self.visual_sigma(e, "fitted")
            e["informative"] = bool(cov is None or np.trace(cov[3:, 3:]) > float(np.sum(sv[3:] ** 2)))
        results["n_informative"] = int(sum(1 for e in self.visual if e.get("informative")))
        for name, om, vm, robust, method, scale, gate in variants:
            def filt(e, gate=gate):
                if gate == "prior" and e["kind"] == "loop":
                    c = e.get(key_prior)
                    return c is not None and c <= thr
                if gate and gate.startswith("informative"):
                    return bool(e.get("informative", True))
                return True
            graph, known, vis_idx = self.build_graph(om, vm, robust=robust, sigma_scale=scale, edge_filter=filt)
            init_v = init
            if gate and gate.endswith(":online"):
                init_v = gtsam.Values()
                for i in sorted(self.nodes):
                    init_v.insert(i, pose7(self.nodes[i]["pose"]))
            try:
                res, w, dt = self.solve(graph, init_v, method=method, known=known)
            except Exception as ex:
                results[name] = {"error": str(ex)}; continue
            est = {i: res.atPose3(i) for i in self.nodes}
            rec = {"ate": umeyama_ate(est, gt), "time_s": dt, "n_visual_used": len(vis_idx)}
            # posterior residuals of the visual edges (chi^2 against the measurement covariance)
            post = []
            for fi, e in vis_idx:
                s = self.visual_sigma(e, vm)
                r = residual(e["T"], est[e["a"]].between(est[e["b"]]))
                c2 = chi2_of(r, np.diag(s ** 2))
                e[f"chi2_post[{name}]"] = c2
                if w is not None:
                    e[f"gnc_w[{name}]"] = float(w[fi])
                post.append((c2, e.get("gt_false", False), e["injected"], float(w[fi]) if w is not None else None, e["kind"]))
            arr = [p for p in post if p[4] in ("loop", "cross")]
            if arr:
                tr = [p for p in arr if not p[1]]; fa = [p for p in arr if p[1]]
                rec["post_true_accept_p0.999"] = float(np.mean([p[0] <= thr for p in tr])) if tr else None
                rec["post_false_accept_p0.999"] = float(np.mean([p[0] <= thr for p in fa])) if fa else None
                if w is not None:
                    rec["gnc_true_inlier_frac"] = float(np.mean([p[3] > 0.5 for p in tr])) if tr else None
                    rec["gnc_false_inlier_frac"] = float(np.mean([p[3] > 0.5 for p in fa])) if fa else None
                inj = [p for p in post if p[2]]
                if inj:
                    rec["injected_post_accept_p0.999"] = float(np.mean([p[0] <= thr for p in inj]))
                    if w is not None:
                        rec["injected_gnc_inlier_frac"] = float(np.mean([p[3] > 0.5 for p in inj]))
            results[name] = rec
        results["n_injected"] = len(injected)
        if injected:
            key = key_prior
            acc = [e.get(key) is not None and e[key] <= thr for e in injected]
            results["injected_prior_accept_p0.999"] = float(np.mean(acc))
        return results

    def reloc_posterior_test(self, vm="fitted", om="fitted", robust="huber"):
        """Relocalization graph: map keyframes fixed at their stored poses, session keyframes optimised from their
        online poses with calibrated noise; posterior chi-square (translation, 3 dof) of every session->map edge
        against the optimised poses, scored with the ground-truth labels."""
        sess = self.meta.get("session", 0)
        if sess == 0:
            return None
        thr3 = chi2.ppf(0.999, 3)
        graph = gtsam.NonlinearFactorGraph(); init = gtsam.Values()
        ids = sorted(self.nodes.keys())
        for i in ids:
            init.insert(i, pose7(self.nodes[i]["pose"]))
        fixed = [i for i in ids if self.session.get(i, 0) == 0]
        for i in fixed:
            graph.add(gtsam.PriorFactorPose3(i, init.atPose3(i), gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))))
        for e in self.odom:
            if self.session.get(e["a"], 0) == 0 and self.session.get(e["b"], 0) == 0:
                continue
            graph.add(gtsam.BetweenFactorPose3(e["a"], e["b"], e["T"], gtsam.noiseModel.Diagonal.Sigmas(self.odom_sigma(e, om))))
        cross = [e for e in self.visual if e["kind"] == "cross" and "r_gt" in e and not e["injected"]]
        sess_vis = [e for e in self.visual if e["kind"] != "cross" and self.session.get(e["a"], 0) != 0 and self.session.get(e["b"], 0) != 0]
        for e in cross + sess_vis:
            s_ = self.visual_sigma(e, vm)
            if e["kind"] == "cross":
                s_ = np.sqrt(s_ ** 2 + self.models.map_sigma(e["d"]) ** 2)
            base = gtsam.noiseModel.Diagonal.Sigmas(s_)
            if robust == "huber":
                base = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.0), base)
            graph.add(gtsam.BetweenFactorPose3(e["a"], e["b"], e["T"], base))
        lp = gtsam.LevenbergMarquardtParams(); lp.setMaxIterations(100)
        t0 = time.time()
        res = gtsam.LevenbergMarquardtOptimizer(graph, init, lp).optimize()
        dt = time.time() - t0
        tp = fp = fn = tn = 0
        for e in cross:
            s_ = self.visual_sigma(e, vm); s_ = np.sqrt(s_ ** 2 + self.models.map_sigma(e["d"]) ** 2)
            r = residual(e["T"], res.atPose3(e["a"]).between(res.atPose3(e["b"])))
            c2 = float(r[3:] @ (r[3:] / (s_[3:] ** 2)))
            e["chi2_post_reloc"] = c2
            acc = c2 <= thr3
            if e["gt_false"]:
                fp += int(acc); tn += int(not acc)
            else:
                tp += int(acc); fn += int(not acc)
        # session ATE (session keyframes vs GT, alignment through the fixed map: compare map-frame poses directly
        # after aligning the map to GT with Umeyama over the map keyframes)
        est_s = {i: res.atPose3(i) for i in ids if self.session.get(i, 0) != 0}
        return {"n_cross": len(cross), "n_false": sum(e["gt_false"] for e in cross), "time_s": dt,
                "true_accept": tp / max(tp + fn, 1), "false_accept": fp / max(fp + tn, 1), "false_rejected": tn, "true_rejected": fn}

    def inject_false_loops(self, n):
        ids = [i for i in self.nodes if i in self.gt and self.step.get(i) is not None]
        ids.sort()
        out = []
        tries = 0
        while len(out) < n and tries < 50 * n:
            tries += 1
            a, b = sorted(self.rng.choice(ids, 2, replace=False))
            if self.session[a] != self.session[b] or self.step[b] - self.step[a] < self.loop_gap:
                continue
            T_gt = self.gt[a].between(self.gt[b])
            if np.linalg.norm(T_gt.translation()) > 30:
                continue
            ang = self.rng.uniform(0, 2 * math.pi); mag = self.rng.uniform(3.0, 15.0)
            off = gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), gtsam.Point3(mag * math.cos(ang), 0.0, mag * math.sin(ang)))
            yaw = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.0, self.rng.uniform(-0.5, 0.5), 0.0), gtsam.Point3(0, 0, 0))
            T_false = T_gt.compose(off).compose(yaw)
            e = {"a": int(a), "b": int(b), "mean": None, "std": [0.05, 0.05, 0.05, 0.02, 0.02, 0.02], "conf": 0.5, "rw": 0.6,
                 "T": T_false, "d": float(np.linalg.norm(T_false.translation())), "gap": int(self.step[b] - self.step[a]),
                 "kind": "loop", "injected": True, "gt_false": True}
            r = residual(T_false, T_gt); e["r_gt"] = r; e["e_t"] = float(np.linalg.norm(r[3:])); e["e_r"] = float(np.degrees(np.linalg.norm(r[:3])))
            self.visual.append(e); out.append(e)
        # prior chi2 for the injected edges (fitted models, several odometry inflations)
        for e in out:
            for om in ("fitted", "fitted*2", "fitted*3"):
                T_chain, cov = self.chain(e["a"], e["b"], om)
                if T_chain is not None:
                    s = self.visual_sigma(e, "fitted")
                    e[f"chi2_prior[odom={om},visual=fitted]"] = chi2_of(residual(e["T"], T_chain), cov + np.diag(s ** 2))
        return out

    def edge_table(self):
        rows = []
        for e in self.visual:
            rows.append({k: v for k, v in e.items() if k not in ("T", "mean", "std", "r_gt") and not isinstance(v, np.ndarray)})
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--out", required=True)
    ap.add_argument("--snr", type=float, default=None)
    ap.add_argument("--loop-gap", type=int, default=200)
    ap.add_argument("--inject", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pgo", action="store_true")
    ap.add_argument("--false-thr", type=float, default=1.0, help="translation error above which an edge is a false loop closure (m)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    st = GraphStudy(args.graph, snr=args.snr, loop_gap=args.loop_gap, seed=args.seed, false_t=args.false_thr); self_snr = st.snr
    res = {"graph": args.graph, "meta": st.meta, "n_nodes": len(st.nodes), "n_gt": len(st.gt), "n_odom": len(st.odom)}
    res["calibration"] = st.calibrate()
    res["prior_test"] = st.prior_test()
    res["pairwise_test"] = st.pairwise_test()
    if not args.no_pgo:
        res["pgo"] = st.pgo_variants(inject=args.inject)
    if st.meta.get("session", 0) != 0:
        res["reloc_posterior"] = st.reloc_posterior_test()
        print(f"   reloc posterior test (map fixed, calibrated Huber): {res['reloc_posterior']}")
    (out / "study.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "edges.json").write_text(json.dumps(st.edge_table(), default=float))
    c = res["calibration"]
    print(f"== {args.graph}: {len(st.nodes)} nodes ({len(st.gt)} with GT), {len(st.odom)} odom edges, {c['n_visual']} visual edges")
    for k, v in c["kinds"].items():
        print(f"   {k:6s} n={v['n']:5d} e_t p50/p90/p99 = {v['e_t_p50']:.3f}/{v['e_t_p90']:.3f}/{v['e_t_p99']:.3f} m  e_r p50/p99 = {v['e_r_p50']:.2f}/{v['e_r_p99']:.2f} deg  false={v['frac_false']:.4f}")
    vf = c["visual_fit"]
    print(f"   visual fit (tail-calibrated): sigma_t = {vf['t_a']:.4f} + {vf['t_b']:.4f} d  m,  sigma_r = {math.degrees(vf['r_a']):.3f} + {math.degrees(vf['r_b']):.4f} d  deg;  bins {[(round(b['d'],1), round(b['sigma'],3)) for b in vf['t_bins']]}; conf multipliers {vf['conf_ref']}")
    if c.get("map_fit"):
        mf = c["map_fit"]; print(f"   map consistency: sigma_t = {mf['t_a']:.3f} + {mf['t_b']:.4f} d m, sigma_r = {math.degrees(mf['r_a']):.2f} + {math.degrees(mf['r_b']):.3f} d deg; bins {[(round(b['d'],1), round(b['sigma'],3)) for b in mf['t_bins']]}")
    if c.get("system_std_abs_z"):
        print(f"   system visual std: median|z|/0.674 = {c['system_std_abs_z']['median_over_0.674']:.3f} (calibrated = 1)")
    if c.get("odom"):
        o = c["odom"]
        print(f"   odom: n={o['n']} L p50 {o['L_p50']:.2f} m, frames/edge {o['n_frames_p50']:.0f}; e_t p50/p99 {o['e_t_p50']:.4f}/{o['e_t_p99']:.4f} m, e_r p50/p99 {o['e_r_deg_p50']:.3f}/{o['e_r_deg_p99']:.3f} deg")
        print(f"         fit sigma_t = {o['fit']['k_t']:.4f} L/sqrt(n) + {o['fit']['floor_t']:.4f}, sigma_r = {o['fit']['k_r']:.4f} theta/sqrt(n) + {o['fit']['floor_r']:.4f} (oracle k = 1/(snr sqrt3) = {1/(self_snr*math.sqrt(3)) if self_snr else float('nan'):.4f}); system std median|z|/0.674 = {o['system_std_abs_z']['p50_over_0.674']:.3f}; oracle {o['oracle_snr_model']}")
    p = res["prior_test"]
    print(f"   prior test on {p['n_loop_candidates']} loop candidates ({p['n_same_session']} same-session, {p['n_cross_session']} session->map):")
    for k, v in p["models"].items():
        print(f"     {k}: " + "; ".join(f"{lv}: TPR {x['TPR']} FPR {x['FPR']} (false accepted {x['n_false_accepted']}/{x['n_false']})" for lv, x in v.items() if lv.startswith("p")))
        print(f"        chi2 true q50/90/99 {v.get('chi2_true_quantiles')}  false q1/10/50 {v.get('chi2_false_quantiles')}")
    print(f"   pairwise: {res['pairwise_test']}")
    if "pgo" in res:
        print(f"   PGO: ATE odom-only {res['pgo']['ate_odom_only']:.3f} m, online result {res['pgo']['ate_online']:.3f} m; injected {res['pgo']['n_injected']} false loops (prior-gate accepts {res['pgo'].get('injected_prior_accept_p0.999')})")
        for k, v in res["pgo"].items():
            if isinstance(v, dict):
                print(f"     {k:32s} " + ", ".join(f"{kk}={vv:.3f}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in v.items()))


if __name__ == "__main__":
    main()
