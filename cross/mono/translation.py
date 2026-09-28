"""Robust metric translation conditional on a scale-free rotation estimate."""

import numpy as np


def translation_given_rotation(points, pixels, K, rotation, threshold=3.0, trials=128):
    """Estimate T_current_reference translation; reject moving correspondences.

    With fixed R, projection supplies two linear equations in translation
    per correspondence. Two-point RANSAC is followed by inlier least squares.
    All input depths may be learned, and their shared bias remains unobservable.
    """
    points, pixels = np.asarray(points), np.asarray(pixels)
    if len(points) < 20:
        return None
    rotated = points @ np.asarray(rotation).T
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T
    A = np.zeros((len(points), 2, 3))
    A[:, 0, 0], A[:, 1, 1] = 1, 1
    A[:, :, 2] = -rays[:, :2]
    b = rays[:, :2] * rotated[:, 2:] - rotated[:, :2]
    samples = np.random.default_rng(0).integers(0, len(points), (trials, 2))
    design, values = A[samples].reshape(trials, 4, 3), b[samples].reshape(trials, 4, 1)
    transpose = design.transpose(0, 2, 1)
    proposals = np.linalg.solve(transpose @ design + np.eye(3)[None] * 1e-10,
                                transpose @ values)[:, :, 0]

    def errors(translations):
        xyz = rotated[None] + translations[:, None]
        uvz = xyz @ np.asarray(K).T
        uv = uvz[:, :, :2] / np.maximum(uvz[:, :, 2:], 1e-8)
        residual = np.linalg.norm(uv - pixels[None], axis=2)
        residual[xyz[:, :, 2] <= 0.01] = np.inf
        return residual

    residuals = errors(proposals)
    best = np.argmax((residuals < threshold).sum(axis=1))
    inliers = residuals[best] < threshold
    translation = proposals[best]
    for _ in range(3):
        if inliers.sum() < 20 or inliers.mean() < 0.3:
            return None
        translation, _, rank, _ = np.linalg.lstsq(A[inliers].reshape(-1, 3), b[inliers].reshape(-1), rcond=None)
        if rank < 3:
            return None
        residual = errors(translation[None])[0]
        inliers = residual < threshold
    if inliers.sum() < 20 or inliers.mean() < 0.3 or not np.isfinite(translation).all():
        return None
    return translation, np.flatnonzero(inliers), float(np.median(residual[inliers]))
