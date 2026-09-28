"""Compare rotation-only and metric-pose explanations of static matches."""

import cv2
import numpy as np


def rotation_only_model(points, pixels, K, full_rotation, full_translation, pixel_std=1.0):
    """BIC test for whether three translation parameters earn their complexity.

    Both models are evaluated on the same geometrically verified matches.
    A one-pixel localization-noise floor prevents negligible subpixel gains
    from becoming spurious translation, especially in low-parallax scenes.
    """
    rvec = cv2.Rodrigues(np.asarray(full_rotation))[0]
    zero = np.zeros(3)
    for _ in range(5):
        projected, jacobian = cv2.projectPoints(points, rvec, zero, K, None)
        residual = (projected.reshape(-1, 2) - pixels).reshape(-1)
        J = jacobian[:, :3]
        update = np.linalg.solve(J.T @ J + np.eye(3) * 1e-8, -J.T @ residual)
        rvec += update[:, None]
        if np.linalg.norm(update) < 1e-7:
            break
    projected = cv2.projectPoints(points, rvec, zero, K, None)[0].reshape(-1, 2)
    full = cv2.projectPoints(points, cv2.Rodrigues(np.asarray(full_rotation))[0], full_translation, K, None)[0].reshape(-1, 2)
    rotation_cost = float(np.sum((projected - pixels)**2))
    full_cost = float(np.sum((full - pixels)**2))
    complexity = 3 * pixel_std**2 * np.log(2 * len(points))
    return cv2.Rodrigues(rvec)[0], rotation_cost - full_cost <= complexity
