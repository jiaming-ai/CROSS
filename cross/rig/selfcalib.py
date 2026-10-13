"""Self-calibration of a two-phone stereo rig from its own recordings (no calibration target).

Two phones mounted rigidly side by side each record camera images, ARKit poses (metric, gravity-aligned VIO) and
optionally LiDAR depth (cross/rig/recording.py).  From one or more recordings this estimates:

1. Clock offset d between the phones (B clock = A clock + d for the same instant): both cameras turn at the same
   angular speed at every instant, so d maximises the correlation of |w_A(t)| and |w_B(t + d)|.
2. Extrinsic X = T_B<-A (camera A in camera B; x_B = R x_A + T, OpenCV camera axes) by hand-eye calibration: relative
   motions over the same interval satisfy dA = X^-1 dB X.  R aligns A's rotation axes to B's; T from the linear system
   (R_dB - I) T = R t_dA - t_dB.  Metric, since ARKit is.
3. Image refinement: SIFT matches between a B image and the A image nearest in time; the relative pose of the two
   exposures is X composed with B's own motion between them (ARKit), so timing differences are compensated.  R, the
   direction of T and d are refined by minimising the Sampson epipolar error (robust loss).
4. LiDAR scale check: the inlier matches are triangulated with the refined rig and compared with B's LiDAR depth; the
   median ratio is the correction of the baseline length (an independent metric reference).

The relative pose of any B frame and A frame (for rectifying a stereo pair taken a few ms apart) is
pair_pose(recA, recB, X, d, iB, jA).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from cross.rig.recording import Recording


# ---------------------------------------------------------------------------------------------------- time offset
def angular_speed(rec: Recording) -> Tuple[np.ndarray, np.ndarray]:
    """(mid times, |w| rad/s) of consecutive frames."""
    R = Rotation.from_matrix(rec.T[:, :3, :3])
    w = np.linalg.norm((R[:-1].inv() * R[1:]).as_rotvec(), axis=1) / np.maximum(np.diff(rec.t), 1e-6)
    return 0.5 * (rec.t[:-1] + rec.t[1:]), w


def estimate_clock_offset(recA: Recording, recB: Recording, search: float = 1.0, step: float = 0.002,
                          grid_dt: float = 0.01) -> Tuple[float, float]:
    """(d, correlation): B clock = A clock + d.  search: largest |d| tried (s)."""
    tA, wA = angular_speed(recA)
    tB, wB = angular_speed(recB)
    lo, hi = max(tA[0], tB[0] + search), min(tA[-1], tB[-1] - search)     # g and g + d inside both, |d| <= search
    if hi - lo < 10.0:
        raise ValueError("the recordings overlap by less than 10 s")
    g = np.arange(lo, hi, grid_dt)
    a = np.interp(g, tA, wA)
    a = (a - a.mean()) / max(a.std(), 1e-9)
    ds = np.arange(-search, search + step / 2, step)
    cs = np.empty(len(ds))
    for k, d in enumerate(ds):
        b = np.interp(g + d, tB, wB)
        cs[k] = np.mean(a * (b - b.mean()) / max(b.std(), 1e-9))
    k = int(np.argmax(cs))
    off = 0.0
    if 0 < k < len(cs) - 1:
        y0, y1, y2 = cs[k - 1], cs[k], cs[k + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return float(ds[k] + off * step), float(cs[k])


def pair_frames(recA: Recording, recB: Recording, d: float, max_dt: float = 0.06):
    """For every B frame the A frame closest in time: (iB, jA, dt = A time - B time in seconds, physical)."""
    tA_on_B = recA.t + d                      # A frame times on B's clock
    j = np.clip(np.searchsorted(tA_on_B, recB.t), 1, len(recA) - 1)
    j = np.where(np.abs(tA_on_B[j - 1] - recB.t) < np.abs(tA_on_B[j] - recB.t), j - 1, j)
    dt = tA_on_B[j] - recB.t
    keep = np.abs(dt) <= max_dt
    return np.nonzero(keep)[0], j[keep], dt[keep]


# ---------------------------------------------------------------------------------------------------- hand-eye
def handeye(recA: Recording, recB: Recording, d: float, gap: float = 1.0, min_rot_deg: float = 10.0,
            max_pairs: int = 4000, trim: float = 0.2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, dict]:
    """(R, T, stats) of X = T_B<-A from relative motions over `gap` seconds that rotate at least min_rot_deg."""
    tB = recB.t[(recB.t - d > recA.t[0]) & (recB.t - d < recA.t[-1])]
    TB = recB.pose_at(tB)
    TA = recA.pose_at(tB - d)                 # A's poses at the same instants
    rng = np.random.default_rng(seed)
    I = rng.integers(0, len(tB), size=max_pairs * 8)
    J = np.searchsorted(tB, tB[I] + gap)
    ok = J < len(tB)
    I, J = I[ok], J[ok]
    dA = np.linalg.inv(TA[I]) @ TA[J]
    dB = np.linalg.inv(TB[I]) @ TB[J]
    ra = Rotation.from_matrix(dA[:, :3, :3]).as_rotvec()
    rb = Rotation.from_matrix(dB[:, :3, :3]).as_rotvec()
    big = np.degrees(np.linalg.norm(rb, axis=1)) >= min_rot_deg
    ra, rb, dA, dB = ra[big][:max_pairs], rb[big][:max_pairs], dA[big][:max_pairs], dB[big][:max_pairs]
    if len(ra) < 50:
        raise ValueError(f"only {len(ra)} motion pairs rotate >= {min_rot_deg} deg: the walk turns too little")
    U, _, Vt = np.linalg.svd(ra.T @ rb)       # rb = R ra
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    A = (dB[:, :3, :3] - np.eye(3)).reshape(-1, 3)
    b = (np.einsum("ij,nj->ni", R, dA[:, :3, 3]) - dB[:, :3, 3]).reshape(-1)
    T, *_ = np.linalg.lstsq(A, b, rcond=None)
    e = np.linalg.norm((A @ T - b).reshape(-1, 3), axis=1)
    m = e <= np.quantile(e, 1 - trim)          # robust re-fit without the worst pairs
    Am, bm = A.reshape(-1, 3, 3)[m].reshape(-1, 3), b.reshape(-1, 3)[m].reshape(-1)
    T, *_ = np.linalg.lstsq(Am, bm, rcond=None)
    cosang = np.einsum("ni,ni->n", rb, ra @ R.T) / (np.linalg.norm(rb, axis=1) * np.linalg.norm(ra, axis=1) + 1e-12)
    # standard error of T from the residual spread (rough: pairs are not independent)
    res = (Am @ T - bm)
    cov = np.linalg.inv(Am.T @ Am) * float(res @ res) / max(len(bm) - 3, 1)
    stats = {"pairs": int(len(ra)), "axis_residual_deg_median": float(np.degrees(np.median(np.arccos(np.clip(cosang, -1, 1))))),
             "t_residual_m_median": float(np.median(e[m])), "T_std_m": [float(x) for x in np.sqrt(np.diag(cov))]}
    return R, T, stats


# ---------------------------------------------------------------------------------------------------- pair geometry
def pair_pose(recB: Recording, R: np.ndarray, T: np.ndarray, d: float, tB: np.ndarray, tA: np.ndarray) -> np.ndarray:
    """(n, 4, 4) pose of camera A at its exposures tA (A clock) in camera B at its exposures tB (B clock):
    T_B(tB)^-1 T_B(tA + d) X, with B's own motion between the two instants (ARKit)."""
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, T
    return np.linalg.inv(recB.pose_at(tB)) @ recB.pose_at(np.asarray(tA) + d) @ X


def _match(imB, imA, max_features: int = 3000, ratio: float = 0.75):
    import cv2
    sift = cv2.SIFT_create(nfeatures=max_features)
    kb, db = sift.detectAndCompute(imB, None)
    ka, da = sift.detectAndCompute(imA, None)
    if db is None or da is None or len(kb) < 20 or len(ka) < 20:
        return None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 30:
        return None
    pb = np.float64([kb[g.queryIdx].pt for g in good])
    pa = np.float64([ka[g.trainIdx].pt for g in good])
    F, inl = cv2.findFundamentalMat(pb, pa, cv2.FM_RANSAC, 1.5, 0.999)
    if F is None:
        return None
    inl = inl.ravel().astype(bool)
    return (pb[inl], pa[inl]) if inl.sum() >= 20 else None


def _sampson(xb, xa, Rr, tr):
    """Sampson errors (normalised coordinates) of x_b^T E x_a = 0, E = [t]x R (x_b = R x_a + t)."""
    tx = np.array([[0, -tr[2], tr[1]], [tr[2], 0, -tr[0]], [-tr[1], tr[0], 0]])
    E = tx @ Rr
    Ea = xa @ E.T
    Eb = xb @ E
    num = np.einsum("ni,ni->n", xb, Ea)
    den = Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Eb[:, 0] ** 2 + Eb[:, 1] ** 2
    return num / np.sqrt(np.maximum(den, 1e-18))


@dataclass
class ImageMatches:
    tB: float
    tA: float
    iB: int
    xb: np.ndarray        # (n, 3) normalised homogeneous coordinates in B
    xa: np.ndarray        # (n, 3) in A
    pb: np.ndarray        # (n, 2) pixel coordinates in B (full resolution)


def collect_matches(recA: Recording, recB: Recording, d: float, n_frames: int = 120, scale: float = 0.5,
                    seed: int = 0) -> List[ImageMatches]:
    """SIFT matches of up to n_frames B frames (the calmer ones, spread over the walk) with the A frame nearest in time."""
    iB, jA, dt = pair_frames(recA, recB, d)
    if len(iB) == 0:
        return []
    tm, w = angular_speed(recB)
    wB = np.interp(recB.t[iB], tm, w)
    calm = iB[wB <= np.quantile(wB, 0.5)]              # less motion blur / rolling shutter
    pick = calm[np.linspace(0, len(calm) - 1, min(n_frames, len(calm))).astype(int)]
    lookup = dict(zip(iB.tolist(), jA.tolist()))
    KB, KA = recB.K(), recA.K()
    out = []
    for i in pick:
        j = lookup[int(i)]
        try:
            m = _match(recB.image(int(i), scale=scale), recA.image(int(j), scale=scale))
        except FileNotFoundError:
            continue
        if m is None:
            continue
        pb, pa = m[0] / scale, m[1] / scale
        KBi, KAi = recB.K(int(i)), recA.K(int(j))
        xb = np.c_[pb, np.ones(len(pb))] @ np.linalg.inv(KBi).T
        xa = np.c_[pa, np.ones(len(pa))] @ np.linalg.inv(KAi).T
        out.append(ImageMatches(float(recB.t[i]), float(recA.t[j]), int(i), xb, xa, pb))
    return out


def refine_with_images(recB: Recording, matches: List[ImageMatches], R: np.ndarray, T: np.ndarray, d: float,
                       refine_offset: bool = True, f_px: float = 1300.0):
    """(R, T, d, stats): rotation, translation direction (length kept) and clock offset minimising the Sampson
    error of the matches, each pair compensated for B's motion between its two exposures."""
    from scipy.optimize import least_squares
    if not matches:
        raise ValueError("no image matches")
    tb = np.array([m.tB for m in matches])
    ta = np.array([m.tA for m in matches])
    L = float(np.linalg.norm(T))
    t0 = T / L
    e1 = np.cross(t0, [0, 0, 1.0]) if abs(t0[2]) < 0.9 else np.cross(t0, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(t0, e1)

    def unpack(p):
        Rp = Rotation.from_rotvec(p[:3]).as_matrix() @ R
        tp = t0 + p[3] * e1 + p[4] * e2
        tp = L * tp / np.linalg.norm(tp)
        dp = d + (p[5] if refine_offset else 0.0)
        return Rp, tp, dp

    def residuals(p):
        Rp, tp, dp = unpack(p)
        P = pair_pose(recB, Rp, tp, dp, tb, ta)
        return np.concatenate([_sampson(m.xb, m.xa, P[k, :3, :3], P[k, :3, 3]) for k, m in enumerate(matches)])

    p0 = np.zeros(6)
    r0 = residuals(p0)
    sol = least_squares(residuals, p0, loss="soft_l1", f_scale=1.0 / f_px, x_scale=[1e-3] * 5 + [1e-2],
                        diff_step=1e-6)
    Rr, tr, dr = unpack(sol.x)
    r1 = residuals(sol.x)
    stats = {"frames": len(matches), "matches": int(sum(len(m.xb) for m in matches)),
             "sampson_px_median_before": float(np.median(np.abs(r0)) * f_px),
             "sampson_px_median_after": float(np.median(np.abs(r1)) * f_px),
             "rotation_change_deg": float(np.degrees(np.linalg.norm(sol.x[:3]))),
             "direction_change_deg": float(np.degrees(np.arccos(np.clip(float(tr @ T) / (L * L), -1, 1)))),
             "offset_change_ms": float(1000 * (dr - d))}
    return Rr, tr, dr, stats


def lidar_baseline_ratio(recB: Recording, matches: List[ImageMatches], R: np.ndarray, T: np.ndarray, d: float,
                         zmin: float = 0.5, zmax: float = 5.0) -> Optional[dict]:
    """Median of LiDAR depth / triangulated depth of the matches (triangulated with the rig): the factor that corrects
    the baseline length; None without LiDAR."""
    ratios = []
    for m in matches:
        dep = recB.depth(m.iB)
        if dep is None:
            return None
        D, (sx, sy) = dep
        P = pair_pose(recB, R, T, d, np.array([m.tB]), np.array([m.tA]))[0]
        Rr, tr = P[:3, :3], P[:3, 3]
        # depth along B's ray: z_b x_b = z_a R x_a + t  -> least squares in (z_b, z_a)
        Rx = m.xa @ Rr.T
        A = np.stack([m.xb, -Rx], 2)                       # (n, 3, 2)
        z = np.einsum("nij,nj->ni", np.linalg.pinv(A), np.broadcast_to(tr, (len(m.xb), 3)))
        zb = z[:, 0]
        u = np.clip((m.pb[:, 0] / sx).astype(int), 0, D.shape[1] - 1)
        v = np.clip((m.pb[:, 1] / sy).astype(int), 0, D.shape[0] - 1)
        zl = D[v, u]
        ok = np.isfinite(zl) & (zl > zmin) & (zl < zmax) & (zb > zmin) & (zb < 3 * zmax)
        ratios.append(zl[ok] / zb[ok])
    r = np.concatenate(ratios) if ratios else np.zeros(0)
    if len(r) < 50:
        return None
    return {"ratio_median": float(np.median(r)), "ratio_p25_p75": [float(np.percentile(r, 25)), float(np.percentile(r, 75))],
            "points": int(len(r))}


# ---------------------------------------------------------------------------------------------------- driver
@dataclass
class RigCalibration:
    R: np.ndarray                              # x_B = R x_A + T (OpenCV camera axes; A = right phone, B = left)
    T: np.ndarray
    K_A: np.ndarray
    K_B: np.ndarray
    size: Tuple[int, int]                      # image width, height
    clock_offsets_s: Dict[str, float] = field(default_factory=dict)
    per_recording: Dict[str, dict] = field(default_factory=dict)

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(self.T))

    def to_dict(self) -> dict:
        return {"R": self.R.tolist(), "T": self.T.tolist(), "baseline_m": self.baseline,
                "image_width": int(self.size[0]), "image_height": int(self.size[1]),
                "camera_matrix_A": self.K_A.tolist(), "camera_matrix_B": self.K_B.tolist(),
                "distortion_A": [0.0] * 5, "distortion_B": [0.0] * 5,
                "convention": "x_B = R x_A + T, OpenCV camera axes of the raw (unrotated) images; B = left phone",
                "method": "self-calibration from ARKit motion (hand-eye) + image epipolar refinement + LiDAR scale "
                          "(cross/rig/selfcalib.py)",
                "clock_offsets_s": self.clock_offsets_s, "per_recording": self.per_recording}


def calibrate_recording(recA: Recording, recB: Recording, images: bool = True, n_frames: int = 120,
                        lidar: bool = True, log=print) -> dict:
    d, corr = estimate_clock_offset(recA, recB)
    log(f"  clock offset {1000 * d:+.1f} ms (angular-speed correlation {corr:.3f})")
    R, T, he = handeye(recA, recB, d)
    log(f"  hand-eye: baseline {np.linalg.norm(T):.4f} m (std {np.linalg.norm(he['T_std_m']):.4f}), "
        f"axis residual {he['axis_residual_deg_median']:.2f} deg over {he['pairs']} motion pairs")
    out = {"clock_offset_s": d, "offset_correlation": corr, "handeye": he, "R_handeye": R.tolist(), "T_handeye": T.tolist()}
    if images:
        matches = collect_matches(recA, recB, d, n_frames=n_frames)
        if matches:
            R, T, d, st = refine_with_images(recB, matches, R, T, d)
            out["images"] = st
            log(f"  images: {st['frames']} frames, {st['matches']} matches, Sampson "
                f"{st['sampson_px_median_before']:.2f} -> {st['sampson_px_median_after']:.2f} px; rotation "
                f"{st['rotation_change_deg']:+.2f} deg, offset {st['offset_change_ms']:+.1f} ms")
            if lidar:
                lr = lidar_baseline_ratio(recB, matches, R, T, d)
                if lr is not None:
                    out["lidar"] = lr
                    log(f"  LiDAR / triangulated depth {lr['ratio_median']:.3f} over {lr['points']} points "
                        f"-> baseline {np.linalg.norm(T) * lr['ratio_median']:.4f} m")
    out.update(R=R.tolist(), T=T.tolist(), clock_offset_s=d)
    return out


def combine(results: Dict[str, dict], recs: Dict[str, Tuple[Recording, Recording]], use_lidar_scale: bool = True
            ) -> RigCalibration:
    """One rig calibration from several recordings of the same mounting: chordal mean of the rotations, mean
    translation direction, baseline from LiDAR (when available) or hand-eye; the spread is kept per recording."""
    Rs = np.stack([np.asarray(r["R"]) for r in results.values()])
    U, _, Vt = np.linalg.svd(Rs.sum(0))
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    Ts = np.stack([np.asarray(r["T"]) for r in results.values()])
    lens = []
    for r in results.values():
        L = float(np.linalg.norm(r["T"]))
        if use_lidar_scale and "lidar" in r:
            L *= r["lidar"]["ratio_median"]
        lens.append(L)
    direction = (Ts / np.linalg.norm(Ts, axis=1, keepdims=True)).mean(0)
    T = direction / np.linalg.norm(direction) * float(np.median(lens))
    recA, recB = next(iter(recs.values()))
    per = {}
    for k, r in results.items():
        per[k] = {"clock_offset_ms": round(1000 * r["clock_offset_s"], 2),
                  "rotation_from_combined_deg": round(float(np.degrees(np.linalg.norm(
                      Rotation.from_matrix(np.asarray(r["R"]) @ R.T).as_rotvec()))), 3),
                  "baseline_m": round(lens[list(results).index(k)], 4),
                  "handeye_baseline_m": round(float(np.linalg.norm(r["T_handeye"])), 4)}
        for key in ("images", "lidar"):
            if key in r:
                per[k][key] = r[key]
    return RigCalibration(R=R, T=T, K_A=recA.K(), K_B=recB.K(), size=recB.size,
                          clock_offsets_s={k: r["clock_offset_s"] for k, r in results.items()}, per_recording=per)
