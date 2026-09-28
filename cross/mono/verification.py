"""Image correspondence checks before admitting learned retrieval proposals."""

import cv2
import numpy as np

from .geometry import inverse


def reprojection_inliers(reference_pixels, current_pixels, depth, K_ref, K_current, T_ref_current,
                         max_error=6.0):
    reference_pixels = np.asarray(reference_pixels, dtype=np.float64)
    current_pixels = np.asarray(current_pixels, dtype=np.float64)
    uv = np.round(reference_pixels).astype(int)
    uv[:, 0] = np.clip(uv[:, 0], 0, depth.shape[1] - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, depth.shape[0] - 1)
    z = depth[uv[:, 1], uv[:, 0]]
    xyz_ref = np.c_[reference_pixels, np.ones(len(uv))] @ np.linalg.inv(K_ref).T * z[:, None]
    T_current_ref = inverse(T_ref_current)
    xyz_current = xyz_ref @ T_current_ref[:3, :3].T + T_current_ref[:3, 3]
    projected = xyz_current @ np.asarray(K_current).T
    safe_z = np.where(np.abs(projected[:, 2:]) > 1e-9, projected[:, 2:], np.nan)
    projected = projected[:, :2] / safe_z
    errors = np.linalg.norm(projected - current_pixels, axis=1)
    return np.isfinite(errors) & (xyz_current[:, 2] > 0) & (z > 0) & (errors <= max_error)


def verify_pair(reference_rgb, current_rgb, prediction, minimum_inliers=15):
    from .geometry import relative_pose

    h, w = prediction.depth[0].shape
    images = [cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
              for image in [reference_rgb, current_rgb]]
    detector = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
    features = [detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None) for image in images]
    if any(desc is None or len(desc) < minimum_inliers for _, desc in features):
        return False, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = matcher.knnMatch(features[0][1], features[1][1], k=2)
    reverse = matcher.knnMatch(features[1][1], features[0][1], k=2)
    reverse_pairs = {(m.queryIdx, m.trainIdx) for pair in reverse if len(pair) == 2
                     for m, n in [pair] if m.distance < 0.8 * n.distance}
    pairs = [(m.queryIdx, m.trainIdx) for pair in forward if len(pair) == 2
             for m, n in [pair] if m.distance < 0.8 * n.distance
             and (m.trainIdx, m.queryIdx) in reverse_pairs]
    if len(pairs) < minimum_inliers:
        return False, 0.0
    ref_uv = np.array([features[0][0][i].pt for i, _ in pairs])
    cur_uv = np.array([features[1][0][j].pt for _, j in pairs])
    inliers = reprojection_inliers(ref_uv, cur_uv, prediction.depth[0], prediction.intrinsics[0],
                                  prediction.intrinsics[1],
                                  relative_pose(prediction.extrinsics[0], prediction.extrinsics[1]))
    if inliers.sum() < minimum_inliers or inliers.mean() < 0.25:
        return False, float(inliers.mean())
    tiles = np.floor(ref_uv[inliers] / np.array([w, h]) * 4).astype(int)
    coverage = len(np.unique(tiles, axis=0))
    return coverage >= 4, float(inliers.mean())
