"""Alignment of the map frame to the local ENU frame: T_ENU<-map.

4-DOF (default): the map's vertical is known (the place projection's vertical, or the camera y axis of the first
keyframe), so the map is levelled by a fixed rotation and only the heading (yaw about Up) and a translation are
estimated.  6-DOF: rotation and translation free (maps without a known vertical).  Both are weighted least squares,
iterated with a Cauchy kernel on the Mahalanobis residuals (IRLS) so that fixes the gate let through by mistake do not
pull the anchor.  Compass headings, once their offset (declination + mounting) is known, add heading measurements and
make the yaw observable from one fix.  The anchor reports its covariance (yaw, t) and whether it is observable: yaw
needs horizontal spread of the fixed positions (or a compass), i.e. its standard deviation below `max_yaw_std`."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation R with R a = b (unit vectors)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-12:                      # opposite: rotate by pi about any axis orthogonal to a
        ax = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0, 1.0, 0])
        ax /= np.linalg.norm(ax)
        return 2 * np.outer(ax, ax) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class AnchorConfig:
    dof: int = 4                       # 4: yaw + translation with the map's vertical; 6: free rotation
    cauchy_c: float = 2.4477           # Cauchy kernel scale on the Mahalanobis residual (sqrt chi2_2(0.95))
    iterations: int = 10
    max_yaw_std_deg: float = 3.0       # the anchor is usable (observable) below this yaw standard deviation
    max_t_std: float = 10.0            # and below this horizontal translation standard deviation (m)


class Anchor:
    """T_ENU<-map = (R, t): p_enu = R p_map + t."""

    def __init__(self, cfg: AnchorConfig = None, vertical_map: Optional[np.ndarray] = None):
        self.cfg = cfg or AnchorConfig()
        self.set_vertical(vertical_map if vertical_map is not None else np.array([0.0, -1.0, 0.0]))
        self.R = None
        self.t = None
        self.yaw = None
        self.cov = None                    # 4x4 covariance of (yaw, tx, ty, tz) (4-DOF) or 6x6 (6-DOF: rotvec, t)
        self.n_used = 0
        self.weights = None

    def set_vertical(self, up_map: np.ndarray):
        """Up direction of the map frame (unit vector in map coordinates).  The default (-y) is the up direction of a
        camera frame (x right, y down, z forward) of a level forward-looking camera at the first keyframe."""
        self.up_map = np.asarray(up_map, float) / np.linalg.norm(up_map)
        self.R_level = rotation_between(self.up_map, np.array([0, 0, 1.0]))   # levelled = R_level @ map

    @property
    def ok(self) -> bool:
        if self.R is None or self.cov is None:
            return False
        if self.cfg.dof == 4:
            ys, ts = math.sqrt(max(self.cov[0, 0], 0)), math.sqrt(max(self.cov[1, 1], self.cov[2, 2], 0))
        else:
            ys, ts = math.sqrt(max(self.cov[2, 2], 0)), math.sqrt(max(self.cov[3, 3], self.cov[4, 4], 0))
        return ys <= math.radians(self.cfg.max_yaw_std_deg) and ts <= self.cfg.max_t_std

    def yaw_std(self) -> float:
        if self.cov is None:
            return float("inf")
        return math.sqrt(max(self.cov[0, 0] if self.cfg.dof == 4 else self.cov[2, 2], 0.0))

    # ------------------------------------------------------------------
    def fit(self, p_map: np.ndarray, p_enu: np.ndarray, sigma_h: np.ndarray, sigma_v: Optional[np.ndarray] = None,
            heading_map: Optional[np.ndarray] = None, heading_enu: Optional[np.ndarray] = None,
            heading_sigma: Optional[np.ndarray] = None) -> bool:
        """Fit from positions (N, 3) in the map and ENU frames with 1-sigma horizontal (vertical) noise, plus optional
        heading pairs (yaw of the same direction in the levelled map frame and in ENU, radians).  Returns self.ok."""
        p_map = np.asarray(p_map, float).reshape(-1, 3)
        p_enu = np.asarray(p_enu, float).reshape(-1, 3)
        sh = np.asarray(sigma_h, float).reshape(-1)
        sv = np.asarray(sigma_v, float).reshape(-1) if sigma_v is not None else 3.0 * sh
        n = len(p_map)
        nh = 0 if heading_map is None else len(heading_map)
        if n == 0:
            return False
        if self.cfg.dof == 6:
            return self._fit6(p_map, p_enu, sh, sv)
        q = p_map @ self.R_level.T                          # levelled map coordinates
        w = np.ones(n)
        wh = np.ones(nh)
        yaw = self.yaw if self.yaw is not None else None
        for it in range(self.cfg.iterations):
            W = w / sh ** 2
            if yaw is None or it == 0:
                # closed-form weighted Procrustes in 2-D (+ heading pairs as unit-vector correspondences)
                ws = W.sum()
                mq = (W[:, None] * q[:, :2]).sum(0) / ws
                me = (W[:, None] * p_enu[:, :2]).sum(0) / ws
                H = ((q[:, :2] - mq) * W[:, None]).T @ (p_enu[:, :2] - me)
                Sxy = H[0, 1] - H[1, 0]
                Cxx = H[0, 0] + H[1, 1]
                if nh:
                    # a heading pair (a, b) with weight 1/sigma^2 adds sin/cos of (b - a) like a unit-length pair
                    d = wrap(np.asarray(heading_enu) - np.asarray(heading_map))
                    whw = wh / np.asarray(heading_sigma) ** 2
                    # scale headings comparably to positions: a heading residual of x rad at lever L costs (x L / s)^2
                    Sxy += float((whw * np.sin(d)).sum())
                    Cxx += float((whw * np.cos(d)).sum())
                if abs(Sxy) + abs(Cxx) < 1e-12:
                    return False
                yaw = math.atan2(Sxy, Cxx)
            Rz = rot_z(yaw)
            # Gauss-Newton refinement on (yaw, tx, ty) with the vertical offset separate
            for _ in range(2):
                pr = q @ Rz.T
                tx = ((W * (p_enu[:, 0] - pr[:, 0])).sum()) / W.sum()
                ty = ((W * (p_enu[:, 1] - pr[:, 1])).sum()) / W.sum()
                Wv = w / sv ** 2
                tz = ((Wv * (p_enu[:, 2] - pr[:, 2])).sum()) / Wv.sum() if np.isfinite(p_enu[:, 2]).all() else 0.0
                r = p_enu[:, :2] - pr[:, :2] - np.array([tx, ty])
                # derivative of R(yaw) q wrt yaw: [-y, x] of the rotated point
                J = np.stack([-pr[:, 1], pr[:, 0]], 1)
                num = (W[:, None] * J * r).sum()
                den = (W[:, None] * J * J).sum() - ((W[:, None] * J).sum(0) ** 2).sum() / W.sum()
                if nh:
                    rh = wrap(np.asarray(heading_enu) - np.asarray(heading_map) - yaw)
                    whw = wh / np.asarray(heading_sigma) ** 2
                    num += float((whw * rh).sum())
                    den += float(whw.sum())
                if den > 1e-12:
                    yaw = float(wrap(yaw + num / den))
                Rz = rot_z(yaw)
            pr = q @ Rz.T
            t = np.array([tx, ty, tz])
            res = p_enu - pr - t
            m2 = (res[:, :2] ** 2).sum(1) / sh ** 2
            w = 1.0 / (1.0 + m2 / self.cfg.cauchy_c ** 2)
            if nh:
                rh = wrap(np.asarray(heading_enu) - np.asarray(heading_map) - yaw)
                wh = 1.0 / (1.0 + (rh / np.asarray(heading_sigma)) ** 2 / self.cfg.cauchy_c ** 2)
        # covariance of (yaw, tx, ty, tz): inverse of the weighted normal matrix
        W = w / sh ** 2
        pr = q @ rot_z(yaw).T
        A = np.zeros((4, 4))
        J = np.zeros((n, 2, 4))
        J[:, 0, 0], J[:, 1, 0] = -pr[:, 1], pr[:, 0]
        J[:, 0, 1] = 1.0
        J[:, 1, 2] = 1.0
        A += np.einsum("n,nia,nib->ab", W, J, J)
        A[3, 3] += float((w / sv ** 2).sum())
        if nh:
            A[0, 0] += float((wh / np.asarray(heading_sigma) ** 2).sum())
        try:
            cov = np.linalg.inv(A + 1e-12 * np.eye(4))
        except np.linalg.LinAlgError:
            return False
        self.yaw = float(yaw)
        self.R = rot_z(yaw) @ self.R_level
        self.t = np.array([tx, ty, tz])
        self.cov = cov
        self.n_used = int((w > 0.5).sum())
        self.weights = w
        return self.ok

    def _fit6(self, p_map, p_enu, sh, sv) -> bool:
        w = np.ones(len(p_map))
        for _ in range(self.cfg.iterations):
            W = w / sh ** 2
            ws = W.sum()
            mp = (W[:, None] * p_map).sum(0) / ws
            me = (W[:, None] * p_enu).sum(0) / ws
            H = ((p_map - mp) * W[:, None]).T @ (p_enu - me)
            U, S, Vt = np.linalg.svd(H)
            D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
            R = Vt.T @ D @ U.T
            t = me - R @ mp
            res = p_enu - p_map @ R.T - t
            m2 = (res[:, :2] ** 2).sum(1) / sh ** 2 + res[:, 2] ** 2 / sv ** 2
            w = 1.0 / (1.0 + m2 / (2 * self.cfg.cauchy_c ** 2))
        W = w / sh ** 2
        pr = p_map @ R.T
        A = np.zeros((6, 6))
        for p, wi, svi in zip(pr, W, w / sv ** 2):
            Jr = np.array([[0, p[2], -p[1]], [-p[2], 0, p[0]], [p[1], -p[0], 0]])   # d(R p)/d(rotvec) = -[p]x
            J = np.hstack([Jr, np.eye(3)])
            Wm = np.diag([wi, wi, svi])
            A += J.T @ Wm @ J
        try:
            self.cov = np.linalg.inv(A + 1e-12 * np.eye(6))
        except np.linalg.LinAlgError:
            return False
        self.R, self.t = R, t
        up = R @ self.up_map
        self.yaw = float(math.atan2(R[1, 0], R[0, 0]))
        self.n_used = int((w > 0.5).sum())
        self.weights = w
        return self.ok

    # ------------------------------------------------------------------
    def to_enu(self, p_map: np.ndarray) -> np.ndarray:
        return np.asarray(p_map, float) @ self.R.T + self.t

    def to_map(self, p_enu: np.ndarray) -> np.ndarray:
        return (np.asarray(p_enu, float) - self.t) @ self.R

    def heading_enu(self, R_map_cam: np.ndarray, forward=(0.0, 0.0, 1.0)) -> float:
        """ENU yaw (from East, counter-clockwise) of the camera's forward axis."""
        f = self.R @ (np.asarray(R_map_cam) @ np.asarray(forward, float))
        return math.atan2(f[1], f[0])

    def heading_map_level(self, R_map_cam: np.ndarray, forward=(0.0, 0.0, 1.0)) -> float:
        """Yaw of the camera's forward axis in the levelled map frame (the quantity the anchor's yaw rotates)."""
        f = self.R_level @ (np.asarray(R_map_cam) @ np.asarray(forward, float))
        return math.atan2(f[1], f[0])

    def position_cov_enu(self, p_map: np.ndarray) -> np.ndarray:
        """3x3 covariance of R p_map + t from the anchor's own uncertainty (lever arm of the yaw)."""
        if self.cov is None:
            return np.eye(3) * 1e6
        if self.cfg.dof == 4:
            pr = rot_z(self.yaw) @ (self.R_level @ np.asarray(p_map, float))
            J = np.zeros((3, 4))
            J[0, 0], J[1, 0] = -pr[1], pr[0]
            J[0, 1] = J[1, 2] = J[2, 3] = 1.0
        else:
            p = self.R @ np.asarray(p_map, float)
            J = np.hstack([np.array([[0, p[2], -p[1]], [-p[2], 0, p[0]], [p[1], -p[0], 0]]), np.eye(3)])
        return J @ self.cov @ J.T

    def state(self) -> dict:
        return {"dof": self.cfg.dof, "up_map": self.up_map.tolist(),
                "R": None if self.R is None else self.R.tolist(), "t": None if self.t is None else self.t.tolist(),
                "yaw": self.yaw, "cov": None if self.cov is None else self.cov.tolist(), "n_used": self.n_used}

    def load_state(self, s: dict):
        self.set_vertical(np.asarray(s["up_map"]))
        self.R = None if s.get("R") is None else np.asarray(s["R"])
        self.t = None if s.get("t") is None else np.asarray(s["t"])
        self.yaw = s.get("yaw")
        self.cov = None if s.get("cov") is None else np.asarray(s["cov"])
        self.n_used = int(s.get("n_used", 0))
