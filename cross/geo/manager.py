"""Geo anchoring of a CROSS map: GNSS fixes and compass headings -> map<->ENU anchor, GNSS factors in the pose graph,
geo-gated retrieval and proposal tests (cross.core.system calls it; see GeoConfig in cross/core/config.py).

Per step the System hands over the odometry increment, the belief of hypothesis 0 (map frame) and the GNSS fix /
compass sample of the frame, if any.  The manager
- keeps a drift-free local track (the integrated odometry) for the GNSS relative-consistency test,
- runs every fix through cross.geo.gnss.GnssGate (absolute gate against the belief carried through the anchor once
  the map is anchored and the session is in the map frame),
- attaches used fixes to the latest keyframe (with the odometry offset between keyframe and fix), at most one factor
  per correlation time of the receiver error (cross.geo.gnss.GnssErrorModel),
- fits the anchor T_ENU<-map (cross.geo.anchor) in a mapping session; a loaded map brings its anchor,
- decides when the pose graph should be optimised (the factors added since the last optimisation disagree with the
  current keyframe poses at the chi-square level), rescales the noise from the posterior residuals afterwards,
- provides GNSS location priors for retrieval and a GNSS consistency test of relocalization proposals,
- stores its state with the map (ENU origin, anchor, per-keyframe fixes, noise / compass calibration) and exports the
  keyframes' latitude / longitude."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from scipy.stats import chi2

from .anchor import Anchor, AnchorConfig, wrap
from .compass import Compass, CompassConfig
from .geodesy import LocalFrame, geojson_linestring, geojson_points
from .gnss import GnssErrorModel, GnssFix, GnssGate, GnssGateConfig, GnssModelConfig, GnssNoiseModel, MEDIAN_CHI2_2


def pose7_to_matrix(v) -> np.ndarray:
    """[tx ty tz qx qy qz qw] -> 4x4 (numpy; ~10 us, a pypose SE3 -> matrix costs ~250 us per call)."""
    x, y, z, w = float(v[3]), float(v[4]), float(v[5]), float(v[6])
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [float(v[0]), float(v[1]), float(v[2])]
    return T


def _up_from_vertical(v: np.ndarray, first_R: Optional[np.ndarray] = None) -> np.ndarray:
    """Up direction of the map frame from the place projection's vertical (sign: opposite to the first keyframe's
    camera y axis, which points down for a forward-looking camera)."""
    v = np.asarray(v, float) / max(np.linalg.norm(v), 1e-12)
    down = (first_R[:, 1] if first_R is not None else np.array([0, 1.0, 0]))
    return -v if float(v @ down) > 0 else v


class GeoManager:
    def __init__(self, cfg, lc_confidence: float = 0.999, odom_k_t: float = 0.06):
        self.cfg = cfg
        conf = cfg.confidence if cfg.confidence is not None else lc_confidence
        self.conf = conf
        self.k2 = float(chi2.ppf(conf, 2))
        self.odom_k_t = float(cfg.drift_rate if cfg.drift_rate is not None else odom_k_t)
        self.noise = GnssNoiseModel(GnssModelConfig(uere=cfg.uere, sigma_v_factor=cfg.sigma_v_factor))
        self.gate = GnssGate(self.noise, GnssGateConfig(confidence=conf, window_s=cfg.window_s,
                                                        min_fixes=cfg.min_fixes, holdoff_s=cfg.holdoff_s))
        self.err = GnssErrorModel(prior_tau=cfg.factor_interval_prior_s)
        self.compass = Compass(CompassConfig(confidence=conf, sigma_deg=cfg.compass_sigma_deg,
                                             offset_deg=cfg.compass_offset_deg, frame=cfg.compass_frame))
        self.anchor = Anchor(AnchorConfig(dof=cfg.anchor_dof, max_yaw_std_deg=cfg.anchor_max_yaw_std_deg))
        self.frame: Optional[LocalFrame] = None
        self.anchor_fixed = False                  # True once a loaded map supplied the anchor
        # local track: integrated odometry (4x4), drift free over the relative test's window
        self.T_odo = np.eye(4)
        self.dist = 0.0
        self.kf_odo: Dict[int, np.ndarray] = {}    # keyframe id -> odometry pose when it was created
        self.factors: Dict[int, dict] = {}         # keyframe id -> {"enu", "sh", "sv", "delta", "t"}  (base sigmas)
        self.fix_log: List[dict] = []              # every used fix (for anchoring and posterior statistics)
        self.pending: List[int] = []               # keyframes whose factor arrived since the last optimisation
        self.last_factor_t = -1e18
        self.last_used_dist = None
        self.window_ref_kf = None
        self.drift_flag = False
        self.last_decision = None
        self.current = None                        # the latest used fix of this step: {"enu", "sh", "t", "map"}
        self.n_opt = 0
        self.last_opt_kf = -10 ** 9
        self.n_kf = 0
        self.compass_last = None
        self.now = None
        self.robust_c = (float(cfg.robust_c) if cfg.robust_c is not None else math.sqrt(self.k2)) if cfg.robust else None
        self.stats = {"factors": 0, "opt": 0, "proposals_rejected": 0, "proposals_tested": 0, "focus": 0}
        # pool_sessions / compass_anchor: this session's index in the map, the stored sessions' fixes and headings
        self.session = 0
        self.map_fix_log: List[dict] = []
        self.heading_log: List[dict] = []          # compass headings attached to keyframes: {"kf", "Rd", "yaw", "t", "session"}
        self.map_heading_log: List[dict] = []
        self.last_heading_t = -1e18
        # component_anchors: kf id -> component root (its min kf id), from the system; anchors of placed separate groups
        self.components_fn = None
        self.comp_anchors: Dict[int, Anchor] = {}

    # ------------------------------------------------------------------ state
    @property
    def anchored(self) -> bool:
        return self.frame is not None and self.anchor.ok

    # ------------------------------------------------------------------ separate groups (component_anchors)
    def _components(self, nodes: Optional[dict] = None):
        """(kf id -> component root, root of the main group) or (None, None) without component_anchors."""
        if not self.cfg.component_anchors or self.components_fn is None:
            return None, None
        comp = self.components_fn()
        if not comp:
            return None, None
        return comp, comp.get(min(comp))

    def anchor_of(self, kf_id) -> Optional[Anchor]:
        """The anchor that places keyframe `kf_id` (the map anchor for the main group or without component_anchors; a
        placed group's own anchor; None for an unplaced group or before the map is anchored)."""
        if self.frame is None:
            return None
        comp, main = self._components()
        if comp is None or kf_id is None or comp.get(int(kf_id), main) == main:
            return self.anchor if self.anchor.ok else None
        return self.comp_anchors.get(comp[int(kf_id)])

    def _placed_anchors(self) -> list:
        """[(root or None for the main group, anchor)] of every placed anchor."""
        out = [(None, self.anchor)] if self.anchored else []
        if self.cfg.component_anchors and self.frame is not None:
            out += sorted(self.comp_anchors.items())
        return out

    def set_vertical(self, v_map: np.ndarray, first_R: Optional[np.ndarray] = None):
        if self.anchor_fixed:
            return
        up = _up_from_vertical(v_map, first_R)
        if self.anchor.R is None or float(up @ self.anchor.up_map) < 0.9999:
            self.anchor.set_vertical(up)

    def reset_session(self):
        """A new session on a loaded map: the local track and the gate restart; the anchor stays."""
        self.T_odo = np.eye(4)
        self.dist = 0.0
        self.gate = GnssGate(self.noise, self.gate.cfg)
        self.pending = []
        self.current = None
        self.last_factor_t = -1e18
        self.last_used_dist = None
        self.session += 1
        self.last_heading_t = -1e18

    # ------------------------------------------------------------------ per step
    def tick(self, t: float):
        self.now = float(t)

    def _current_fix(self):
        """The latest used fix while it is recent (fix_max_age_s), with the extra uncertainty of the odometry
        displacement since then (its direction in the map frame is not known before the session is merged)."""
        c = self.current
        if c is None:
            return None, 0.0
        if self.now is not None and self.now - c["t"] > self.cfg.fix_max_age_s:
            return None, 0.0
        disp = float(np.linalg.norm((np.linalg.inv(c["odo"]) @ self.T_odo)[:3, 3])) if c.get("odo") is not None else 0.0
        return c, disp

    def on_motion(self, delta: Optional[np.ndarray]):
        if delta is None:
            return
        self.T_odo = self.T_odo @ delta
        self.dist += float(np.linalg.norm(delta[:3, 3]))

    def observe(self, t: float, gnss: Optional[dict], compass: Optional[dict], T_map_cam: Optional[np.ndarray],
                in_map_frame: bool, belief_std_t: float = 0.0, last_kf_id: Optional[int] = None, nodes: dict = None):
        """Process the frame's GNSS fix / compass sample.  `T_map_cam`: hypothesis 0's pose; `in_map_frame`: whether
        it is a pose in the map's frame (mapping session, or a relocalization session merged into the map)."""
        self.now = float(t)
        if compass is not None:
            self._observe_compass(compass, T_map_cam, in_map_frame, last_kf_id)
        if gnss is None:
            return None
        self.current = None
        fix = GnssFix.from_obs(gnss)
        if self.frame is None:
            if self.noise.sigma(fix) is None:
                return None
            self.frame = LocalFrame(fix.lat, fix.lon, fix.alt if math.isfinite(fix.alt) else 0.0)
            logger.info(f"geo: local ENU origin {self.frame.lat0:.7f}, {self.frame.lon0:.7f}, {self.frame.alt0:.1f} m")
        enu = self.frame.to_enu(fix.lat, fix.lon, fix.alt if math.isfinite(fix.alt) else self.frame.alt0)
        z_ok = math.isfinite(fix.alt)
        lev = self.anchor.R_level @ self.T_odo[:3, 3]
        pred = None
        # the anchor of the group the session is in (component_anchors; else the map anchor); none for an unplaced group
        A = self.anchor_of(last_kf_id) if self.cfg.component_anchors else (self.anchor if self.anchored else None)
        if A is not None and in_map_frame and T_map_cam is not None:
            p = T_map_cam[:3, 3]
            d_since = (self.dist - self.last_used_dist) if self.last_used_dist is not None else 1e3
            sig = self.odom_k_t * max(d_since, 0.0) + max(belief_std_t, 0.0) + self.cfg.pred_floor
            P = A.position_cov_enu(p)[:2, :2] + sig ** 2 * np.eye(2)
            pred = (A.to_enu(p)[:2], P)
        if self.cfg.gate:
            dec = self.gate.process(fix, enu, lev[:2], pred)
        else:
            from .gnss import GnssDecision
            sig = self.noise.sigma(fix)
            dec = GnssDecision(sig is not None, "ungated" if sig is not None else "no_fix", enu,
                               *(sig if sig is not None else (float("nan"), float("nan"))))
        self.last_decision = dec
        if not dec.used:
            return dec
        self.last_used_dist = self.dist
        sc = self.noise.scale
        if math.isfinite(dec.inflation) and dec.inflation > 1.0:
            # the relative test's inflation steers the gate only; factors keep the prior noise
            dec.sigma_h, dec.sigma_v = dec.sigma_h / math.sqrt(dec.inflation), dec.sigma_v / math.sqrt(dec.inflation)
        rec = {"t": float(fix.t), "enu": np.array([enu[0], enu[1], enu[2] if z_ok else np.nan]),
               "sh": dec.sigma_h / sc, "sv": dec.sigma_v / sc, "in_map": bool(in_map_frame), "kf": None, "delta": None,
               "map": None if T_map_cam is None else T_map_cam[:3, 3].copy(), "odo": self.T_odo.copy(),
               "epoch": int(self.gate.stats["epochs"]), "session": int(self.session)}
        if last_kf_id is not None and last_kf_id in self.kf_odo:
            # attached to the latest keyframe with the odometry between them: its map position follows the keyframe
            rec["kf"] = int(last_kf_id)
            rec["delta"] = (np.linalg.inv(self.kf_odo[last_kf_id]) @ self.T_odo)[:3, 3].copy()
        self.current = rec
        if in_map_frame and rec["kf"] is not None:
            self.fix_log.append(rec)
            if len(self.fix_log) > self.cfg.max_fix_log:
                self.fix_log = self.fix_log[-self.cfg.max_fix_log:]
            # one factor per correlation time of the receiver error, on the latest keyframe (and every fix whose offset
            # the gate attributed to the chain's drift)
            if not self.cfg.decimate or fix.t - self.last_factor_t >= self.err.tau or dec.drift_reset:
                if dec.drift_reset:
                    self.drift_flag = True
                self.factors[rec["kf"]] = {"enu": rec["enu"], "sh": rec["sh"], "sv": rec["sv"], "delta": rec["delta"],
                                           "t": rec["t"]}
                self.pending.append(rec["kf"])
                self.last_factor_t = float(fix.t)
                self.stats["factors"] += 1
            if not self.anchor_fixed and nodes is not None and (
                    not self.anchor.ok or len(self.fix_log) % self.cfg.anchor_refit_every == 0):
                self.fit_anchor(nodes)
        return dec

    def _observe_compass(self, sample, T_map_cam, in_map_frame, last_kf_id=None):
        outdoor_ok = self.last_decision is not None and self.last_decision.used
        h = self.compass.heading(sample, outdoor_ok=outdoor_ok)
        if h is None:
            self.compass_last = None          # disturbed: no heading this frame
            return
        self.compass_last = h
        self.compass_t = self.now
        A = self.anchor_of(last_kf_id) if self.cfg.component_anchors else (self.anchor if self.anchored else None)
        if A is not None and in_map_frame and outdoor_ok and T_map_cam is not None:
            cam_yaw = A.heading_enu(T_map_cam[:3, :3])
            self.compass.add_offset_sample(h[0], cam_yaw)
        # compass_anchor: the camera's ENU heading, attached to the latest keyframe with the odometry rotation since it
        if (self.cfg.compass_anchor and in_map_frame and self.compass.offset_external and last_kf_id is not None
                and last_kf_id in self.kf_odo and h[1] <= math.radians(self.cfg.compass_max_sigma_deg)
                and self.now - self.last_heading_t >= self.cfg.compass_anchor_interval_s):
            self.last_heading_t = self.now
            Rd = (np.linalg.inv(self.kf_odo[last_kf_id]) @ self.T_odo)[:3, :3].copy()
            self.heading_log.append({"kf": int(last_kf_id), "Rd": Rd, "yaw": float(self.compass.camera_yaw(h[0])),
                                     "t": float(self.now), "session": int(self.session)})

    def on_keyframe(self, kf_id: int):
        self.kf_odo[int(kf_id)] = self.T_odo.copy()
        self.n_kf += 1

    # ------------------------------------------------------------------ anchor
    def _position(self, nodes: dict, kid: int, delta: np.ndarray) -> Optional[np.ndarray]:
        kf = nodes.get(kid)
        if kf is None:
            k2 = self._reattach(kid, nodes)
            if k2 is None:
                return None
            M = np.linalg.inv(self.kf_odo[k2]) @ self.kf_odo[kid]
            delta = M[:3, :3] @ delta + M[:3, 3]
            kf = nodes[k2]
        T = kf.pose_mu[0].matrix().detach().cpu().numpy().astype(np.float64)
        return T[:3, 3] + T[:3, :3] @ delta

    def _positions(self, nodes: dict, recs: list) -> list:
        """`_position` of every record (None where the keyframe is gone for good).  The poses of the keyframes still in the
        graph are converted to matrices in one batch: a pypose indexing + matrix() per fix costs ~250 us, and a long
        session logs 10^4 fixes (same values)."""
        out = [None] * len(recs)
        present = [i for i, r in enumerate(recs) if r["kf"] in nodes]
        if present:
            import pypose as pp
            import torch
            rows = torch.stack([nodes[recs[i]["kf"]].plain_row("pose_mu", 0) for i in present])
            Ts = pp.SE3(rows).matrix().detach().cpu().numpy().astype(np.float64)
            for j, i in enumerate(present):
                T = Ts[j]
                out[i] = T[:3, 3] + T[:3, :3] @ recs[i]["delta"]
        for i, r in enumerate(recs):
            if out[i] is None:
                out[i] = self._position(nodes, r["kf"], r["delta"])
        return out

    def fit_anchor(self, nodes: dict) -> bool:
        """Fit T_ENU<-map to the used fixes of this map, at the current poses of the keyframes they are attached to
        (pool_sessions: also the stored sessions' fixes; compass_anchor: plus one heading per session;
        component_anchors: the map anchor from the main group's fixes, one anchor per separate group from its own)."""
        if self.anchor_fixed:
            return self.anchor.ok
        log = (self.map_fix_log + self.fix_log) if self.cfg.pool_sessions else self.fix_log
        recs, pm = [], []
        for r, p in zip(log, self._positions(nodes, log)):
            if p is not None:
                recs.append(r)
                pm.append(p)
        comp, main = self._components(nodes)
        if comp is None:
            return self._fit_one(self.anchor, recs, pm, nodes, None, main_anchor=True)
        groups = {}
        for r, p in zip(recs, pm):
            k = r["kf"] if r["kf"] in comp else self._reattach(r["kf"], nodes)
            if k is None or k not in comp:
                continue
            groups.setdefault(comp[k], ([], []))
            groups[comp[k]][0].append(r)
            groups[comp[k]][1].append(p)
        mr, mp = groups.pop(main, ([], []))
        ok = self._fit_one(self.anchor, mr, mp, nodes, lambda k: comp.get(k) == main, main_anchor=True)
        placed = {}
        for root, (gr, gp) in groups.items():
            if len(gr) < self.cfg.component_min_fixes:
                continue
            A = Anchor(AnchorConfig(dof=self.cfg.anchor_dof, max_yaw_std_deg=180.0, max_t_std=1e9),
                       vertical_map=self.anchor.up_map)
            self._fit_one(A, gr, gp, nodes, lambda k, root=root: comp.get(k) == root, main_anchor=False)
            if A.cov is None:
                continue
            # placed when every keyframe of the group is located to component_max_pos_std_m (yaw lever + translation)
            ids = [k for k, c in comp.items() if c == root and k in nodes]
            P = np.array([nodes[k].pose_mu[0].tensor()[:3].detach().cpu().double().numpy() for k in ids[::max(1, len(ids) // 200)]])
            worst = max(math.sqrt(max(np.linalg.eigvalsh(A.position_cov_enu(q)[:2, :2]))) for q in P)
            if worst <= self.cfg.component_max_pos_std_m:
                placed[root] = A
                if root not in self.comp_anchors:
                    logger.info(f"geo: separate group (keyframes from {root}, {len(ids)} kf) placed in ENU by its own "
                                f"{len(gr)} fixes (yaw {math.degrees(A.yaw):.1f} deg +- {math.degrees(A.yaw_std()):.1f}, "
                                f"position std <= {worst:.1f} m)")
        self.comp_anchors = placed
        self.stats["groups_placed"] = len(placed)
        self.stats["groups_with_fixes"] = len(groups)
        return ok

    def _fit_one(self, A: Anchor, recs: list, pm: list, nodes: dict, kf_filter, main_anchor: bool) -> bool:
        """Fit anchor `A` to the records `recs` at map positions `pm` (+ compass session headings of the keyframes that
        pass `kf_filter`)."""
        if len(recs) < 3:
            return False
        pm = np.array(pm)
        sc = self.noise.scale
        pe = np.array([r["enu"] for r in recs])
        sh = np.array([r["sh"] for r in recs]) * sc
        sv = np.array([r["sv"] for r in recs]) * sc
        bad = ~np.isfinite(pe[:, 2])
        pe = pe.copy()
        pe[bad, 2] = 0.0
        sv = np.where(bad, 1e4, sv)
        was = A.ok
        # correlated errors: the fit's covariance assumes independent fixes; scale it by the number of fixes per
        # correlation time of the receiver error so that the observability test is not overconfident (per session:
        # the sessions' receiver errors are independent of each other)
        by_s = {}
        for r in recs:
            by_s.setdefault(r.get("session", 0), []).append(r["t"])
        n_eff = sum(max(1.0, (max(t) - min(t)) / max(self.err.tau, 1e-3)) for t in by_s.values())
        infl = max(1.0, len(recs) / n_eff)
        heads = self._session_headings(nodes, kf_filter, A) if self.cfg.compass_anchor else None
        if heads:
            # the GNSS weights carry the inflation, so the session headings weigh in at their own noise
            A.fit(pm, pe, sh * math.sqrt(infl), sv * math.sqrt(infl), heading_map=np.zeros(len(heads)),
                  heading_enu=np.array([h[0] for h in heads]), heading_sigma=np.array([h[1] for h in heads]))
        else:
            A.fit(pm, pe, sh, sv)
            if A.cov is not None:
                A.cov = A.cov * infl
        if not main_anchor:
            return A.ok
        self.stats["anchor_fixes"] = len(recs)
        self.stats["anchor_sessions"] = len(by_s)
        self.stats["anchor_headings"] = len(heads or [])
        ok = A.ok
        if ok and not was:
            logger.info(f"geo: map anchored to ENU (yaw {math.degrees(A.yaw):.1f} deg +- "
                        f"{math.degrees(A.yaw_std()):.1f}, {len(recs)} fixes of {len(by_s)} sessions, "
                        f"{len(heads or [])} compass sessions)")
        return ok

    def _session_headings(self, nodes: dict, kf_filter=None, A: Optional[Anchor] = None) -> list:
        """[(yaw offset ENU - levelled map, sigma)] per session with compass headings: the robust circular mean over the
        session's headings of (compass camera heading in ENU - the keyframe's heading in the levelled map), with noise
        compass_session_sigma_deg (the session-to-session spread of a phone compass, which dominates the in-session
        noise); a session whose headings scatter more than compass_max_spread_deg adds none."""
        log = (self.map_heading_log + self.heading_log) if self.cfg.pool_sessions else self.heading_log
        log = [r for r in log if r["kf"] in nodes and (kf_filter is None or kf_filter(r["kf"]))]
        if not log:
            return []
        A = A or self.anchor
        import pypose as pp
        import torch
        rows = torch.stack([nodes[r["kf"]].plain_row("pose_mu", 0) for r in log])
        Rs = pp.SE3(rows).matrix().detach().cpu().numpy().astype(np.float64)[:, :3, :3]
        by_s = {}
        for r, R in zip(log, Rs):
            d = float(wrap(r["yaw"] - A.heading_map_level(R @ r["Rd"])))
            by_s.setdefault(r["session"], []).append(d)
        out = []
        s0 = math.radians(self.cfg.compass_session_sigma_deg)
        for ses, d in by_s.items():
            d = np.asarray(d)
            if len(d) < 10:
                continue
            c = math.atan2(np.sin(d).mean(), np.cos(d).mean())
            e = wrap(d - c)
            med = float(np.median(e))
            spread = 1.4826 * float(np.median(np.abs(e - med)))
            if spread > math.radians(self.cfg.compass_max_spread_deg):
                continue
            out.append((float(wrap(c + med)), s0))
        return out

    # ------------------------------------------------------------------ pose graph
    def pgo_factors(self, nodes: dict, node_ids) -> list:
        """[(kf_id, target_map (3,), cov_map (3, 3))] for the keyframes of `node_ids` that carry a GNSS factor (none
        with pose_graph off: anchor only)."""
        if not self.cfg.pose_graph or not self.anchored or not self.factors:
            return []
        out = []
        sc = self.noise.scale
        Rme = self.anchor.R.T
        comp, main = self._components(nodes)
        for kid, f in list(self.factors.items()):
            if comp is not None and comp.get(kid, main) != main:
                continue                       # a separate group's fix: the map anchor does not place it
                kid2 = self._reattach(kid, nodes)
                if kid2 is None or kid2 not in node_ids:
                    continue
                f = self.factors.pop(kid)
                odo = self.kf_odo[kid]
                f["delta"] = (np.linalg.inv(self.kf_odo[kid2]) @ odo)[:3, :3] @ f["delta"] + (np.linalg.inv(self.kf_odo[kid2]) @ odo)[:3, 3]
                if kid2 in self.factors:
                    continue
                self.factors[kid2] = f
                kid = kid2
            kf = nodes[kid]
            T = kf.pose_mu[0].matrix().detach().cpu().numpy().astype(np.float64)
            enu = f["enu"]
            z_ok = bool(np.isfinite(enu[2]))
            target = self.anchor.to_map(np.array([enu[0], enu[1], enu[2] if z_ok else 0.0])) - T[:3, :3] @ f["delta"]
            Cenu = np.diag([(f["sh"] * sc) ** 2, (f["sh"] * sc) ** 2, ((f["sv"] * sc) if z_ok else 1e3) ** 2])
            out.append((kid, target, Rme @ Cenu @ Rme.T + 1e-6 * np.eye(3)))
        return out

    def _reattach(self, kid: int, nodes: dict) -> Optional[int]:
        """A keyframe that carried a factor was removed (temporary keyframe merged away): the nearest earlier keyframe."""
        if kid not in self.kf_odo:
            return None
        cands = [k for k in self.kf_odo if k in nodes and k < kid]
        if not cands:
            cands = [k for k in self.kf_odo if k in nodes]
        return max(cands) if cands else None

    def soft_gauge_cov(self, R_first: np.ndarray) -> np.ndarray:
        """6x6 covariance (gtsam order: rotation, translation) of the soft prior on the first keyframe that replaces
        its hard fix when GNSS factors determine the map's position and heading: roll / pitch tight (the vertical),
        yaw and position free."""
        up_body = R_first.T @ self.anchor.up_map
        s_small, s_big = math.radians(self.cfg.gauge_tilt_deg), 1.0
        Sr = s_small ** 2 * np.eye(3) + (s_big ** 2 - s_small ** 2) * np.outer(up_body, up_body)
        C = np.zeros((6, 6))
        C[:3, :3] = Sr
        C[3:, 3:] = (1e3) ** 2 * np.eye(3)
        return C

    def should_optimize(self, nodes: dict, n_kf_now: int) -> bool:
        """The factors added since the last optimisation disagree with the current keyframe poses."""
        if not self.cfg.pose_graph:
            return False
        if not self.anchored or (len(self.pending) < self.cfg.opt_min_factors and not self.drift_flag) or not self.pending:
            return False
        if n_kf_now - self.last_opt_kf < self.cfg.opt_min_keyframes:
            return False
        if self.drift_flag:
            return True                       # the gate attributed an offset to the chain's drift: correct it now
        sc = self.noise.scale
        es = []
        for kid in self.pending:
            f = self.factors.get(kid)
            if f is None or kid not in nodes:
                continue
            T = nodes[kid].pose_mu[0].matrix().detach().cpu().numpy().astype(np.float64)
            p = self.anchor.to_enu(T[:3, 3] + T[:3, :3] @ f["delta"])
            r = f["enu"][:2] - p[:2]
            es.append(float(r @ r) / (f["sh"] * sc) ** 2)
        if not es:
            return False
        # all factors since the last optimisation, and the most recent ones alone (a drift that grows is diluted by
        # the older, still consistent factors)
        m = min(len(es), self.cfg.opt_min_factors)
        return sum(es) > chi2.ppf(self.conf, 2 * len(es)) or sum(es[-m:]) > chi2.ppf(self.conf, 2 * m)

    def after_optimize(self, nodes: dict, n_kf_now: int):
        """Posterior statistics, error model, anchor refit, after a pose-graph optimisation with GNSS factors."""
        self.pending = []
        self.drift_flag = False
        self.last_opt_kf = n_kf_now
        # the latest keyframe the GNSS factors constrained at this optimisation: a later GNSS-triggered optimisation
        # of a large map may keep the keyframes before it fixed (windowed PGO), everything since is free
        live = [k for k in self.factors if k in nodes]
        self.window_ref_kf = max(live) if live else None
        self.n_opt += 1
        self.stats["opt"] += 1
        if not self.factors:
            return
        sc = self.noise.scale
        # posterior residuals of every used fix of this map (attached to keyframes): noise scale and correlation time
        e, rr, tt, ee = [], [], [], []
        for rec, p in zip(self.fix_log, self._positions(nodes, self.fix_log)):
            if p is None:
                continue
            r = rec["enu"][:2] - self.anchor.to_enu(p)[:2]
            e.append(float(r @ r) / (rec["sh"] * sc) ** 2)
            rr.append(r)
            tt.append(rec["t"])
            ee.append(rec.get("epoch", 0))
        if len(e) >= 10:
            self.err.update_from_residuals(np.array(tt), np.array(rr), np.array(ee))
        # the logged fixes' map positions follow the optimised keyframes they are attached to
        self.fit_anchor(nodes)

    # ------------------------------------------------------------------ retrieval / relocalization
    def focus_region(self):
        """(center in the map frame (3,), horizontal radius (m), up vector of the map) of the stored keyframes a trusted
        fix allows (cfg.retrieval_focus), or None without a usable fix / anchor."""
        regs = self.focus_regions()
        return regs[0][:3] if regs else None

    def focus_regions(self) -> list:
        """[(center (3,), radius, up, group root or None for the main group)]: the focus region of the current fix in
        the frame of every placed anchor (one without component_anchors)."""
        out = []
        for center, sigma, src, root in self._priors():
            out.append((center, math.sqrt(self.k2) * sigma + float(self.cfg.focus_margin_m),
                        np.asarray(self.anchor.up_map, float), root))
        return out

    def location_priors(self) -> list:
        """[(center_map (3,), sigma (m), source)] for retrieval: the current fix through the anchor (component_anchors:
        through every placed anchor)."""
        return [p[:3] for p in self._priors()]

    def _priors(self) -> list:
        c, disp = self._current_fix()
        if not self.anchored and not self.comp_anchors or c is None:
            return []
        enu = np.array([c["enu"][0], c["enu"][1], c["enu"][2] if np.isfinite(c["enu"][2]) else 0.0])
        s = float(c["sh"] * self.noise.scale)
        out = []
        for root, A in self._placed_anchors():
            p = A.to_map(enu)
            sa = float(np.sqrt(np.trace(A.position_cov_enu(p)[:2, :2]) / 2.0))
            out.append((p, math.sqrt(s ** 2 + sa ** 2 + disp ** 2), "gnss" if root is None else f"gnss:{root}", root))
        return out

    def proposal_consistent(self, p_map: np.ndarray, kf_id: Optional[int] = None) -> Optional[bool]:
        """Chi-square test (2 dof, horizontal) of a proposed current camera position (map frame) against the current
        fix; None when there is no usable fix this frame (component_anchors: or when the reference keyframe `kf_id` is in
        an unplaced group; a placed group's reference is tested through its group's anchor)."""
        c, disp = self._current_fix()
        A = self.anchor_of(kf_id) if self.cfg.component_anchors else (self.anchor if self.anchored else None)
        if A is None or c is None:
            return None
        p = A.to_enu(np.asarray(p_map, float))
        r = c["enu"][:2] - p[:2]
        S = (c["sh"] * self.noise.scale) ** 2 * np.eye(2) + A.position_cov_enu(p_map)[:2, :2] \
            + (self.cfg.pred_floor ** 2 + disp ** 2) * np.eye(2)
        self.stats["proposals_tested"] += 1
        ok = float(r @ np.linalg.solve(S, r)) <= self.k2
        if not ok:
            self.stats["proposals_rejected"] += 1
        return ok

    def heading_consistent(self, R_map_cam: np.ndarray, kf_id: Optional[int] = None) -> Optional[bool]:
        """Chi-square test (1 dof) of a proposed camera orientation (map frame) against the compass heading of this
        frame (undisturbed, offset calibrated); None without a usable compass heading (or in an unplaced group)."""
        A = self.anchor_of(kf_id) if self.cfg.component_anchors else (self.anchor if self.anchored else None)
        if A is None or self.compass_last is None or self.compass.offset is None:
            return None
        if self.now is not None and self.now - getattr(self, "compass_t", -1e18) > self.cfg.fix_max_age_s:
            return None
        yaw_c = self.compass.camera_yaw(self.compass_last[0])
        yaw_p = A.heading_enu(np.asarray(R_map_cam, float))
        spread = getattr(self.compass, "offset_spread", None) or 0.0
        sig = math.sqrt(max(self.compass_last[1], spread) ** 2 + A.yaw_std() ** 2)
        self.stats["heading_tested"] = self.stats.get("heading_tested", 0) + 1
        ok = float(wrap(yaw_p - yaw_c)) ** 2 <= (sig ** 2) * float(chi2.ppf(self.conf, 1))
        if not ok:
            self.stats["heading_rejected"] = self.stats.get("heading_rejected", 0) + 1
        return ok

    # ------------------------------------------------------------------ persistence / export
    def state(self) -> dict:
        """Persistent state, stored with the map (System.save_map: "geo").  Per-keyframe data as numpy columns (map
        format v2 keeps the graph as columns too): factors (kf, enu, sigmas, offset, time)."""
        # the stored map's factors (a loaded map) are kept with this session's (a map saved again keeps its fixes)
        f = {**getattr(self, "map_factors", {}), **self.factors}
        ks = sorted(f)
        extra = {}
        if self.cfg.pool_sessions:
            # every used fix of every session (the anchor of the whole map is refitted from them later)
            fx = self.map_fix_log + self.fix_log
            extra["fixes"] = {"kf": np.asarray([r["kf"] for r in fx], np.int64),
                              "enu": np.asarray([r["enu"] for r in fx], np.float64).reshape(-1, 3),
                              "sh": np.asarray([r["sh"] for r in fx], np.float64),
                              "sv": np.asarray([r["sv"] for r in fx], np.float64),
                              "delta": np.asarray([r["delta"] for r in fx], np.float64).reshape(-1, 3),
                              "t": np.asarray([r["t"] for r in fx], np.float64),
                              "session": np.asarray([r.get("session", 0) for r in fx], np.int64)}
            hx = self.map_heading_log + self.heading_log
            extra["headings"] = {"kf": np.asarray([r["kf"] for r in hx], np.int64),
                                 "Rd": np.asarray([r["Rd"] for r in hx], np.float64).reshape(-1, 3, 3),
                                 "yaw": np.asarray([r["yaw"] for r in hx], np.float64),
                                 "t": np.asarray([r["t"] for r in hx], np.float64),
                                 "session": np.asarray([r["session"] for r in hx], np.int64)}
        if self.cfg.component_anchors:
            extra["component_anchors"] = [{"root": int(r), "anchor": A.state()} for r, A in sorted(self.comp_anchors.items())]
        return {"version": 2, **extra, "frame": None if self.frame is None else self.frame.state(),
                "anchor": self.anchor.state(), "noise": self.noise.state(), "err": self.err.state(),
                "compass": self.compass.state(),
                "factors": {"kf": np.asarray(ks, np.int64),
                            "enu": np.asarray([f[k]["enu"] for k in ks], np.float64).reshape(-1, 3),
                            "sh": np.asarray([f[k]["sh"] for k in ks], np.float64),
                            "sv": np.asarray([f[k]["sv"] for k in ks], np.float64),
                            "delta": np.asarray([f[k]["delta"] for k in ks], np.float64).reshape(-1, 3),
                            "t": np.asarray([f[k]["t"] for k in ks], np.float64)},
                "gate": dict(self.gate.stats), "stats": dict(self.stats)}

    @staticmethod
    def factor_records(fac) -> Dict[int, dict]:
        """Factors of a stored state: the columns of version 2, or the dict of version 1."""
        if not fac:
            return {}
        if isinstance(fac, dict) and "kf" in fac and not isinstance(next(iter(fac.values())), dict):
            return {int(k): {"enu": np.asarray(e), "sh": float(a), "sv": float(b), "delta": np.asarray(d), "t": float(t)}
                    for k, e, a, b, d, t in zip(fac["kf"], fac["enu"], fac["sh"], fac["sv"], fac["delta"], fac["t"])}
        return {int(k): v for k, v in fac.items()}

    def load_state(self, s: dict):
        if not s or s.get("frame") is None:
            return
        self.frame = LocalFrame.from_state(s["frame"])
        self.anchor.load_state(s["anchor"])
        # pool_sessions: the anchor keeps being refitted with the stored and this session's fixes
        self.anchor_fixed = self.anchor.ok and not self.cfg.pool_sessions
        if self.cfg.pool_sessions:
            fx = s.get("fixes") or {}
            self.map_fix_log = [{"kf": int(k), "enu": np.asarray(e, float), "sh": float(a), "sv": float(b),
                                 "delta": np.asarray(d, float), "t": float(t), "session": int(ses)}
                                for k, e, a, b, d, t, ses in zip(fx.get("kf", []), fx.get("enu", []), fx.get("sh", []),
                                                                 fx.get("sv", []), fx.get("delta", []), fx.get("t", []),
                                                                 fx.get("session", []))]
            hx = s.get("headings") or {}
            self.map_heading_log = [{"kf": int(k), "Rd": np.asarray(R, float), "yaw": float(y), "t": float(t),
                                     "session": int(ses)}
                                    for k, R, y, t, ses in zip(hx.get("kf", []), hx.get("Rd", []), hx.get("yaw", []),
                                                               hx.get("t", []), hx.get("session", []))]
            ss = [r["session"] for r in self.map_fix_log + self.map_heading_log]
            self.session = max(ss) if ss else 0
        if self.cfg.component_anchors:
            self.comp_anchors = {}
            for c in s.get("component_anchors") or []:
                A = Anchor(AnchorConfig(dof=self.cfg.anchor_dof, max_yaw_std_deg=180.0, max_t_std=1e9))
                A.load_state(c["anchor"])
                self.comp_anchors[int(c["root"])] = A
        self.noise.load_state(s.get("noise", {}))
        if s.get("err"):
            self.err.tau = float(s["err"].get("tau", self.err.tau))
            self.err.sigma2_axis = s["err"].get("sigma2_axis")
        self.compass.load_state(s.get("compass", {}), freeze=self.cfg.compass_anchor)
        # the stored map's fixes stay with it (its keyframes are fixed in a relocalization session)
        self.map_factors = self.factor_records(s.get("factors"))

    @staticmethod
    def lla_columns(lla: Dict[int, list]) -> dict:
        ks = sorted(lla)
        return {"kf": np.asarray(ks, np.int64), "lla": np.asarray([lla[k] for k in ks], np.float64).reshape(-1, 3)}

    @staticmethod
    def lla_records(c) -> Dict[int, list]:
        """Keyframe latitude / longitude / altitude of a stored map: columns (v2) or a dict (v1)."""
        if not c:
            return {}
        if isinstance(c, dict) and "kf" in c and "lla" in c:
            return {int(k): [float(x) for x in v] for k, v in zip(c["kf"], c["lla"])}
        return {int(k): v for k, v in c.items()}

    def keyframe_lla(self, nodes: dict) -> Dict[int, list]:
        """Latitude / longitude / altitude of every keyframe (camera position), once anchored (component_anchors: every
        keyframe of a placed group through its group's anchor; keyframes of unplaced groups have none)."""
        comp, main = self._components(nodes)
        if comp is None:
            if not self.anchored:
                return {}
            ids = sorted(nodes)
            P = np.array([nodes[i].pose_mu[0].tensor().detach().cpu().numpy().reshape(-1)[:3] for i in ids], float)
            lla = self.frame.to_lla(self.anchor.to_enu(P))
            return {int(i): [float(a), float(b), float(c)] for i, (a, b, c) in zip(ids, lla)}
        out = {}
        for root, A in self._placed_anchors():
            ids = sorted(k for k in nodes if comp.get(k, main) == (main if root is None else root))
            if not ids:
                continue
            P = np.array([nodes[i].pose_mu[0].tensor().detach().cpu().numpy().reshape(-1)[:3] for i in ids], float)
            for i, (a, b, c) in zip(ids, self.frame.to_lla(A.to_enu(P))):
                out[int(i)] = [float(a), float(b), float(c)]
        return out

    def geojson(self, nodes: dict, name: str = "map") -> dict:
        lla = self.keyframe_lla(nodes)
        if not lla:
            return {"type": "FeatureCollection", "features": []}
        arr = np.array([lla[k] for k in sorted(lla)])
        feats = [geojson_linestring(arr, {"name": name, "keyframes": len(arr)})]
        fx = [self.factors[k]["enu"] for k in sorted(self.factors)]
        if fx:
            e = np.array(fx)
            e[~np.isfinite(e[:, 2]), 2] = self.frame.alt0 * 0
            feats += geojson_points(self.frame.to_lla(e), [{"kind": "gnss_factor"}] * len(e))
        return {"type": "FeatureCollection", "features": feats}
