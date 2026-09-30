"""Experimental calibrated relative geometry with a learned metric magnitude.

Rotation and translation direction come from calibrated image correspondences.
The reference depth sets magnitude once; current depth only screens compatibility.
Thresholds and the downstream covariance floor remain uncalibrated heuristics.
Classical two-view geometry is not a new contribution of CROSS.
"""
import cv2
import numpy as np


def estimate_metric_two_view(xy_ref, xy_cur, ref_depth, cur_depth, K, shape):
    """Return T_ref_current: geometric direction/rotation, reference metric scale.

    Depth is used once for magnitude, with the second image a compatibility
    screen. The thresholds are declared diagnostics, not calibrated likelihoods.
    """
    p, q = np.asarray(xy_ref, dtype=np.float64), np.asarray(xy_cur, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if p.ndim != 2 or p.shape[1:] != (2,) or q.shape != p.shape:
        raise ValueError("Expected equal N by 2 point arrays")
    if K.shape != (3, 3) or not np.isfinite(K).all() or np.linalg.det(K) <= 0:
        raise ValueError("Expected a finite nonsingular camera calibration")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        return None, dict(reason="nonfinite_matches", accepted=False, matches=len(p))
    audit = dict(reason='few_matches', matches=len(p), accepted=False)
    if len(p) < 20:
        return None, audit
    E, mask = cv2.findEssentialMat(p, q, K, method=cv2.USAC_MAGSAC, prob=.999,
                                  threshold=1., maxIters=1000)
    audit['reason'] = 'epipolar_consensus'
    if E is None or E.shape != (3, 3) or mask is None:
        return None, audit
    mask = mask.reshape(-1).astype(bool)
    audit['epipolar_inliers'] = int(mask.sum())
    if mask.sum() < 20 or mask.mean() < .3:
        return None, audit
    H, hm = cv2.findHomography(p, q, cv2.USAC_MAGSAC, 3., maxIters=1000, confidence=.999)
    audit['homography_inliers'] = 0 if hm is None else int(hm.sum())
    _, R, t, visible = cv2.recoverPose(E, p, q, K, mask=mask.astype(np.uint8))
    visible = visible.reshape(-1).astype(bool)
    audit.update(reason='cheirality', visible_inliers=int(visible.sum()),
                 cheirality_fraction=float(visible.sum()/mask.sum()))
    if visible.sum() < 20 or visible.sum()/mask.sum() < .9:
        return None, audit
    rays1 = np.c_[p, np.ones(len(p))] @ np.linalg.inv(K).T
    rays2 = np.c_[q, np.ones(len(q))] @ np.linalg.inv(K).T
    ray1_unit = rays1 / np.linalg.norm(rays1, axis=1, keepdims=True)
    ray2_unit = rays2 @ R
    ray2_unit /= np.linalg.norm(ray2_unit, axis=1, keepdims=True)
    parallax = np.degrees(np.arccos(np.clip((ray1_unit*ray2_unit).sum(axis=1), -1, 1)))
    audit.update(reason='low_parallax', median_parallax_deg=float(np.median(parallax[visible])))
    if audit['median_parallax_deg'] < 1.:
        return None, audit
    homogeneous = cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)], np.c_[R,t],
                                        rays1[:,:2].T, rays2[:,:2].T).T
    with np.errstate(divide='ignore', invalid='ignore'):
        X = homogeneous[:,:3]/homogeneous[:,3:]
        Y = X @ R.T + t.ravel()
        a = X @ K.T; b = Y @ K.T
        error1 = np.linalg.norm(a[:,:2]/a[:,2:]-p, axis=1)
        error2 = np.linalg.norm(b[:,:2]/b[:,2:]-q, axis=1)
    good = visible & (parallax >= .5) & (X[:,2] > 0) & (Y[:,2] > 0)
    good &= np.isfinite(X).all(axis=1) & (error1 <= 3.) & (error2 <= 3.)
    height, width = shape
    def depth_at(depth, pixels):
        if depth.ndim == 1:
            return depth
        uv = np.rint(pixels * [depth.shape[1]/width,depth.shape[0]/height]).astype(int)
        uv[:,0] = np.clip(uv[:,0],0,depth.shape[1]-1)
        uv[:,1] = np.clip(uv[:,1],0,depth.shape[0]-1)
        return depth[uv[:,1],uv[:,0]]
    z1, z2 = depth_at(ref_depth, p), depth_at(cur_depth, q)
    good &= np.isfinite(z1) & np.isfinite(z2) & (z1 > .01) & (z2 > .01)
    audit.update(reason='triangulation_support', metric_inliers=int(good.sum()))
    if good.sum() < 20 or good.mean() < .3:
        return None, audit
    tiles = [len(np.unique(np.floor(x[good]/[width,height]*4).astype(int),axis=0)) for x in (p,q)]
    audit.update(reason='poor_spatial_coverage', tiles=tiles)
    if min(tiles) < 4:
        return None, audit
    log_ratios = [np.log(z1[good]/X[good,2]),np.log(z2[good]/Y[good,2])]
    centers = np.array([np.median(x) for x in log_ratios])
    spreads = [float(1.4826*np.median(abs(x-c))) for x,c in zip(log_ratios,centers)]
    audit.update(reason='inconsistent_depth_shape', log_scales=centers.tolist(),
                 log_mad=spreads, scale_ratio=float(np.exp(centers[0]-centers[1])))
    if max(spreads) > .2:
        return None, audit
    audit['reason'] = 'inconsistent_metric_scale'
    if abs(centers[0]-centers[1]) > 3*np.sqrt(2)*.12:
        return None, audit
    result = np.eye(4); result[:3,:3] = R.T
    result[:3,3] = -R.T @ t.ravel() * np.exp(centers[0])
    audit.update(reason='accepted', accepted=True,
                 median_reprojection_px=float(np.median(np.maximum(error1,error2)[good])))
    return result, audit
