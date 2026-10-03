"""Place coordinates for proposal clustering / hypothesis matching (cross/utils/lie_tensor.py: SE3Projection)."""
import math

import numpy as np
import pypose as pp
import torch

from cross.utils.lie_tensor import (SE3Projection, estimate_vertical, project_SE3, split_clusters_by_vertical,
                                    vertical_vector)


def _poses(t, R):
    q = pp.mat2SO3(torch.as_tensor(R, dtype=torch.float32)).tensor()
    return pp.SE3(torch.cat([torch.as_tensor(t, dtype=torch.float32), q], dim=-1))


def _rot(axis, angle):
    return pp.so3(torch.as_tensor(axis, dtype=torch.float64) * angle).matrix().numpy()


def test_y_vertical_reproduces_original_distances():
    torch.manual_seed(0)
    T = pp.randn_SE3(64)
    T = pp.SE3(torch.cat([T.tensor()[:, :3] * 5, T.tensor()[:, 3:]], dim=1))
    for v in ("y", [0.0, -1.0, 0.0]):
        a, b = project_SE3(T), project_SE3(T, projection=SE3Projection(v))
        d_old = torch.cdist(a, a, compute_mode="donot_use_mm_for_euclid_dist")
        d_new = torch.cdist(b, b, compute_mode="donot_use_mm_for_euclid_dist")
        assert torch.allclose(d_old, d_new, atol=1e-4)


def test_down_looking_vertical_z():
    """Vertical z: a displacement along y separates places, one along z (altitude) does not unless weighted."""
    T = _poses([[0, 0, 0], [0, 3, 0], [0, 0, 3]], np.stack([np.eye(3)] * 3))
    p = project_SE3(T, projection=SE3Projection("z"))
    assert torch.norm(p[0] - p[1]) > 2.9 and torch.norm(p[0] - p[2]) < 1e-6
    p = project_SE3(T, projection=SE3Projection("z", vertical_weight=0.5))
    assert abs(float(torch.norm(p[0] - p[2])) - 1.5) < 1e-5
    # heading about z: a 90 deg turn about the optical axis moves (cos, sin) by sqrt(2)
    T2 = _poses([[0, 0, 0], [0, 0, 0]], np.stack([np.eye(3), _rot([0, 0, 1], math.pi / 2)]))
    p2 = project_SE3(T2, projection=SE3Projection("z"))
    assert abs(float(torch.norm(p2[0] - p2[1])) - math.sqrt(2)) < 1e-5


def _turning_trajectory(tilt_deg, n=60, turn=2 * math.pi, roll_noise_deg=0.0, seed=0):
    """Camera pitched down by tilt_deg on a robot turning about world vertical (-y of a level OpenCV camera); poses
    expressed in the first camera frame (the CROSS map frame)."""
    rng = np.random.default_rng(seed)
    mount = _rot([1, 0, 0], -math.radians(tilt_deg))            # camera-to-body
    world_up = np.array([0.0, -1.0, 0.0])
    Rs = []
    for k in range(n):
        noise = _rot(rng.normal(size=3) / np.sqrt(3), math.radians(roll_noise_deg) * rng.normal())
        Rs.append(_rot(world_up, turn * k / n) @ noise @ mount)
    Rs = np.stack(Rs)
    Rs = np.einsum("ij,njk->nik", Rs[0].T, Rs)                  # relative to the first camera
    return torch.as_tensor(Rs), mount.T @ world_up               # true vertical in the first camera frame


def test_estimate_vertical_tilted_camera():
    R, v_true = _turning_trajectory(tilt_deg=25, roll_noise_deg=2)
    v, ok = estimate_vertical(R, vertical_vector("y"))
    assert ok and abs(float(v @ torch.as_tensor(v_true))) > 0.995
    # no turns (straight drive): the prior is kept
    R0, _ = _turning_trajectory(tilt_deg=25, turn=0.0)
    v0, ok0 = estimate_vertical(R0, vertical_vector("y"))
    assert not ok0 and torch.equal(v0, vertical_vector("y"))


def test_tilted_camera_projection_uses_true_horizontal():
    """With a 30 deg pitched camera, the original projection mixes height into the horizontal coordinates; the
    estimated vertical removes it."""
    R, v_true = _turning_trajectory(tilt_deg=30)
    v, _ = estimate_vertical(R, vertical_vector("y"))
    up = torch.as_tensor(v_true, dtype=torch.float32)
    T = _poses(torch.stack([torch.zeros(3), 2.0 * up]), np.stack([np.eye(3)] * 2))   # same place, 2 m higher
    p_old = project_SE3(T)
    p_new = project_SE3(T, projection=SE3Projection(v))
    assert torch.norm(p_old[0] - p_old[1]) > 0.9 and torch.norm(p_new[0] - p_new[1]) < 1e-3


def test_split_clusters_by_vertical():
    labels = np.array([0, 0, 0, 0, 1, -1])
    vert = np.array([0.0, 0.1, 1.5, 1.6, 0.0, 9.0])
    scores = np.array([1.0, 5.0, 2.0, 0.5, 1.0, 1.0])
    out = split_clusters_by_vertical(labels, vert, scores, gate=0.5)
    assert out[0] == out[1] == 0 and out[2] == out[3] not in (0, 1, -1) and out[4] == 1 and out[5] == -1
