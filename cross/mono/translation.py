"""Robust metric translation conditional on a scale-free rotation estimate."""

import numpy as np


def _refine_translation_pixels(rotated, pixels, K, translation):
    """Refine the same pixel residual used to select and verify consensus.

    Algebraic ray errors multiply image errors by depth. Fitting them without
    whitening can let distant points destroy a valid pixel consensus. A small
    damped solve keeps rotation fixed and rejects steps behind the camera.
    """
    translation = translation.copy()

    def linearize(value):
        xyz = rotated + value
        if np.any(xyz[:, 2] <= .01):
            return None
        projected = xyz @ K.T
        uv = projected[:, :2] / projected[:, 2:]
        jacobian = (K[None, :2, :] - uv[:, :, None] * K[None, None, 2, :]) / projected[:, 2, None, None]
        return (uv - pixels).reshape(-1), jacobian.reshape(-1, 3)

    for _ in range(8):
        current = linearize(translation)
        if current is None:
            return None
        residual, jacobian = current
        if np.linalg.matrix_rank(jacobian) < 3:
            return None
        hessian = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        diagonal = np.diag(np.maximum(np.diag(hessian), 1e-12))
        damping, accepted = 1e-6, False
        for _ in range(8):
            delta = np.linalg.solve(hessian + damping * diagonal, -gradient)
            candidate = translation + delta
            trial = linearize(candidate)
            if trial is not None and trial[0] @ trial[0] <= residual @ residual:
                translation, accepted = candidate, True
                break
            damping *= 10
        # Relative scene length keeps stopping consistent under metric scaling.
        if not accepted or np.linalg.norm(delta) <= 1e-7 * np.median(np.linalg.norm(rotated, axis=1)):
            break
    return translation


def translation_given_rotation(points, pixels, K, rotation, threshold=3.0, trials=128):
    """Estimate T_current_reference translation; reject moving correspondences.

    With fixed R, projection supplies two linear equations in translation
    per correspondence. Two-point RANSAC is followed by pixel reprojection
    refinement with the same consensus threshold.
    All input depths may be learned, and their shared bias remains unobservable.
    """
    points, pixels, K = np.asarray(points), np.asarray(pixels), np.asarray(K)
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
        translation = _refine_translation_pixels(rotated[inliers], pixels[inliers], K, translation)
        if translation is None:
            return None
        residual = errors(translation[None])[0]
        inliers = residual < threshold
    if inliers.sum() < 20 or inliers.mean() < 0.3 or not np.isfinite(translation).all():
        return None
    return translation, np.flatnonzero(inliers), float(np.median(residual[inliers]))
