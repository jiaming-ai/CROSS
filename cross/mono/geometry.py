"""OpenCV camera coordinates; T_a_b maps points in b into a."""

import numpy as np
from scipy.spatial.transform import Rotation


def inverse(T):
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def homogeneous(extrinsics):
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics.copy()
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Expected 3x4 or 4x4 extrinsics, got {extrinsics.shape}")
    out = np.broadcast_to(np.eye(4), (*extrinsics.shape[:-2], 4, 4)).copy()
    out[..., :3, :] = extrinsics
    return out


def relative_pose(world_to_ref, world_to_current, scale=1.0):
    """Convert decoded world-to-camera extrinsics to T_ref_current."""
    out = homogeneous(world_to_ref) @ inverse(homogeneous(world_to_current))
    out[:3, 3] *= scale
    return out


def mean_pose(poses, weights=None):
    poses = np.asarray(poses)
    weights = np.ones(len(poses)) if weights is None else np.asarray(weights)
    weights = weights / weights.sum()
    out = np.eye(4)
    out[:3, 3] = np.sum(poses[:, :3, 3] * weights[:, None], axis=0)
    out[:3, :3] = Rotation.from_matrix(poses[:, :3, :3]).mean(weights=weights).as_matrix()
    return out


def scale_translation_covariance(translation, log_scale_variance):
    """First-order covariance from one shared log-scale random variable.

    J_lambda = translation; the rank-one covariance must not be treated as
    three independent observations of scale.
    """
    t = np.asarray(translation)
    return np.outer(t, t) * log_scale_variance
