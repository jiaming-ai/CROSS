"""Metric scale recovery for feed-forward multi-view pose predictions.

A feed-forward geometry model (VGGT-Omega, Depth Anything 3, ...) predicts camera
poses for a set of images up to one global similarity transform.  If a calibrated
stereo pair (left, right) is part of the same forward pass, the predicted relative
transform between the two views must equal the known rig transform B up to that
scale, which yields a scalar anchor  s_i = ||b|| / ||trans(X_L^-1 X_R)||.

Additional anchors of the same form can come from any two views whose metric
relative translation is known, e.g. two temporally adjacent frames connected by
odometry.  All anchors are fused with a robust location estimate in log space
(multiplicative noise model); the estimator and its weights follow the
stereo_vggt study (constraints.py) and are re-implemented here in torch-free numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


def invert_poses(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    out = np.zeros_like(T)
    R = T[..., :3, :3]
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    # project onto SO(3) first: a slightly scaled / non-orthogonal matrix (float32 quaternion drift)
    # makes the trace formula report degrees of error where there is a fraction of a degree
    U, _, Vt = np.linalg.svd(R)
    d = np.sign(np.linalg.det(U @ Vt))
    if R.ndim == 2:
        R = U @ np.diag([1.0, 1.0, float(d)]) @ Vt
    else:
        D = np.repeat(np.eye(3)[None], len(R), 0); D[:, 2, 2] = d
        R = U @ D @ Vt
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


@dataclass
class ScaleAnchor:
    """One metric anchor: a pair of views with known metric relative translation."""
    idx_a: int                       # view index of the first camera in the forward pass
    idx_b: int                       # view index of the second camera
    T_ab: np.ndarray                 # known metric transform of b expressed in a (4x4)
    kind: str = "stereo"             # "stereo", "odom" or "map" (known relative pose between two references)
    weight: float = 1.0              # prior weight (e.g. 1.0 stereo, smaller for odometry)


@dataclass
class ScaleEstimate:
    scale: float
    valid: bool
    n_anchors: int
    n_used: int
    ratios: np.ndarray = field(default_factory=lambda: np.zeros(0))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rot_err_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dir_cos: np.ndarray = field(default_factory=lambda: np.zeros(0))
    kinds: list = field(default_factory=list)
    log_std: float = 0.0             # dispersion of accepted log ratios (relative scale std)
    method: str = "none"

    def to_dict(self) -> dict:
        return {
            "scale": float(self.scale),
            "valid": bool(self.valid),
            "n_anchors": int(self.n_anchors),
            "n_used": int(self.n_used),
            "ratios": self.ratios.tolist(),
            "weights": self.weights.tolist(),
            "rot_err_deg": self.rot_err_deg.tolist(),
            "dir_cos": self.dir_cos.tolist(),
            "kinds": list(self.kinds),
            "log_std": float(self.log_std),
            "method": self.method,
        }


def _huber_log_location(log_ratios: np.ndarray, weights: np.ndarray) -> float:
    center = float(np.median(log_ratios))
    mad = float(np.median(np.abs(log_ratios - center)))
    f_scale = max(1.4826 * mad, 0.03)
    sw = np.sqrt(np.maximum(weights, 1e-8))
    res = least_squares(
        lambda x: sw * (x[0] - log_ratios),
        x0=np.asarray([center]),
        loss="huber",
        f_scale=f_scale,
        max_nfev=100,
    )
    return float(res.x[0])


def estimate_scale(
    c2w: np.ndarray,
    anchors: list[ScaleAnchor],
    *,
    method: str = "adaptive",
    max_rot_err_deg: float = 20.0,
    min_dir_cos: float = 0.5,
    min_pred_norm: float = 1e-4,
    rot_sigma_deg: float = 10.0,
    weight_by_baseline: bool = False,
) -> ScaleEstimate:
    """Estimate the metric scale of a set of predicted camera-to-world poses.

    Args:
        c2w: (S, 4, 4) predicted camera-to-world poses (any global gauge).
        anchors: known-metric pairs inside this forward pass.
        method: "adaptive" (mean if ratio CV < 0.2 else log-Huber), "huber_log",
                "median", "mean", "norm_ls".
        max_rot_err_deg / min_dir_cos: hard rejection of anchors whose predicted
                relative rotation / baseline direction disagrees with the known
                rig transform (a wrongly registered stereo view carries no scale
                information).  Direction is only checked for stereo anchors.
    Returns:
        ScaleEstimate (valid=False if no anchor survives).
    """
    if len(anchors) == 0:
        return ScaleEstimate(scale=1.0, valid=False, n_anchors=0, n_used=0, method=method)

    c2w = np.asarray(c2w, dtype=np.float64)
    ratios, rot_errs, dir_cos, prior_w, kinds, pred_norms = [], [], [], [], [], []
    for a in anchors:
        rel = invert_poses(c2w[a.idx_a]) @ c2w[a.idx_b]          # predicted b in a
        t_pred = rel[:3, 3]
        n_pred = float(np.linalg.norm(t_pred))
        t_known = a.T_ab[:3, 3]
        n_known = float(np.linalg.norm(t_known))
        rot_err = float(rotation_angle_deg(rel[:3, :3] @ a.T_ab[:3, :3].T))
        cos = float(np.dot(t_pred, t_known) / max(n_pred * n_known, 1e-12))
        ratios.append(n_known / max(n_pred, min_pred_norm))
        rot_errs.append(rot_err)
        dir_cos.append(cos)
        prior_w.append(a.weight)
        kinds.append(a.kind)
        pred_norms.append(n_pred)

    ratios = np.asarray(ratios)
    rot_errs = np.asarray(rot_errs)
    dir_cos = np.asarray(dir_cos)
    prior_w = np.asarray(prior_w)
    pred_norms = np.asarray(pred_norms)

    accept = (rot_errs <= max_rot_err_deg) & (pred_norms > min_pred_norm)
    stereo_mask = np.asarray([k in ("stereo", "map") for k in kinds])
    accept &= (~stereo_mask) | (dir_cos >= min_dir_cos)

    est = ScaleEstimate(
        scale=1.0, valid=False, n_anchors=len(anchors), n_used=int(accept.sum()),
        ratios=ratios, rot_err_deg=rot_errs, dir_cos=dir_cos, kinds=kinds, method=method,
    )
    if est.n_used == 0:
        est.weights = np.zeros_like(ratios)
        return est

    r = ratios[accept]
    log_r = np.log(np.maximum(r, 1e-12))
    # soft weights: rotation agreement, and magnitude consistency (log-MAD gaussian)
    rot_w = np.exp(-0.5 * (rot_errs[accept] / rot_sigma_deg) ** 2)
    center = np.median(log_r)
    mad = np.median(np.abs(log_r - center))
    sigma = max(1.4826 * float(mad), 0.15)
    mag_w = np.exp(-0.5 * ((log_r - center) / sigma) ** 2)
    w = np.maximum(rot_w * mag_w * prior_w[accept], 1e-6)
    if weight_by_baseline:
        # an additive error in the predicted camera centres gives a ratio error inversely proportional to the
        # predicted baseline: weight by the (normalised) predicted baseline length
        bl = pred_norms[accept] / max(float(np.max(pred_norms[accept])), 1e-12)
        w = np.maximum(w * bl, 1e-6)

    if method == "median":
        log_s = float(np.median(log_r))
    elif method == "mean":
        log_s = float(np.log(np.mean(r)))
    elif method == "norm_ls":
        # s = sum(w * ||b|| * ||a||) / sum(w * ||a||^2) with a = predicted, b = known
        a = pred_norms[accept]
        b = a * r
        log_s = float(np.log(np.sum(w * a * b) / max(np.sum(w * a * a), 1e-12)))
    elif method == "huber_log":
        log_s = _huber_log_location(log_r, w)
    elif method == "adaptive":
        cv = float(np.std(r) / max(np.mean(r), 1e-12)) if len(r) > 1 else 0.0
        log_s = float(np.log(np.sum(w * r) / max(np.sum(w), 1e-12))) if cv < 0.20 else _huber_log_location(log_r, w)
    else:
        raise ValueError(f"Unknown scale method {method}")

    est.scale = float(np.exp(log_s))
    est.valid = np.isfinite(est.scale) and est.scale > 0
    weights = np.zeros_like(ratios)
    weights[accept] = w
    est.weights = weights
    if len(log_r) > 1:
        est.log_std = float(np.sqrt(np.sum(w * (log_r - log_s) ** 2) / max(np.sum(w), 1e-12)))
    else:
        est.log_std = 0.0
    return est


def scale_camera_centers(c2w: np.ndarray, scale: float, origin_index: int = 0) -> np.ndarray:
    """Scale all camera centers about the center of view `origin_index`."""
    out = np.asarray(c2w, dtype=np.float64).copy()
    origin = out[origin_index, :3, 3].copy()
    out[:, :3, 3] = origin + scale * (out[:, :3, 3] - origin)
    return out
