# some of the functions are borrowed from pypose

import math
import pypose as pp
import torch
from cross.utils.rotation import quaternion_to_euler_torch


def normalize_se3(pose: pp.LieTensor, max_deviation: float = None) -> pp.LieTensor:
    """Remove roundoff in a near-unit SE3 quaternion without changing position.

    Repeated float32 products otherwise drift off the group, making inverse
    and matrix conversion inconsistent. Reject invalid inputs instead of
    treating normalization as a repair for arbitrary poses: a non-finite pose or a (near-)zero quaternion raises.
    Quaternions further from unit norm (e.g. odometry read from low-precision text files) are normalized too,
    unless max_deviation is given: then | |q| - 1 | above it also raises (the conditional-source filter).
    """
    data = pose.tensor().clone()
    norm = torch.linalg.vector_norm(data[..., 3:], dim=-1, keepdim=True)
    limit = 0.5 if max_deviation is None else max_deviation
    if not torch.isfinite(data).all() or torch.any((norm - 1).abs() > limit):
        raise ValueError('Pose mean has a non-unit or non-finite quaternion')
    data[..., 3:] /= norm
    return pp.SE3(data)

def normalize_SE3(x):
    """Return `x` with unit quaternions (batched SE3 LieTensor or raw (..., 7) tensor).

    pypose never renormalizes: every float32 composition (odometry accumulation, motion update,
    candidate poses `kf_pose @ rel`) shrinks |q| by ~1e-6, which compounds over thousands of steps
    into |q| ~ 0.98.  A non-unit quaternion is not a rotation for pypose (Act/matrix blend the
    rotation with the identity and scale translations), so poses must be renormalized wherever
    they are composed.
    """
    t = x.tensor() if hasattr(x, "ltype") else x
    q = t[..., 3:7]
    n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    out = torch.cat([t[..., :3], q / n], dim=-1)
    return pp.SE3(out)

def vec2skew(input:torch.Tensor) -> torch.Tensor:
    r"""
    Convert batched vectors to skew matrices.

    Args:
        input (Tensor): the tensor :math:`\mathbf{x}` to convert.

    Return:
        Tensor: the skew matrices :math:`\mathbf{y}`.

    Shape:
        Input: :obj:`(*, 3)`

        Output: :obj:`(*, 3, 3)`

    .. math::
        {\displaystyle \mathbf{y}_i={\begin{bmatrix}\,\,
        0&\!-x_{i,3}&\,\,\,x_{i,2}\\\,\,\,x_{i,3}&0&\!-x_{i,1}
        \\\!-x_{i,2}&\,\,x_{i,1}&\,\,0\end{bmatrix}}}

    Note:
        The last dimension of the input tensor has to be 3.

    Example:
        >>> pp.vec2skew(torch.randn(1,3))
        tensor([[[ 0.0000, -2.2059, -1.2761],
                [ 2.2059,  0.0000,  0.2929],
                [ 1.2761, -0.2929,  0.0000]]])
    """
    v = input.tensor() if hasattr(input, 'ltype') else input
    assert v.shape[-1] == 3, "Last dim should be 3"
    O = torch.zeros(v.shape[:-1], device=v.device, dtype=v.dtype, requires_grad=v.requires_grad)
    return torch.stack([torch.stack([        O, -v[...,2],  v[...,1]], dim=-1),
                        torch.stack([ v[...,2],         O, -v[...,0]], dim=-1),
                        torch.stack([-v[...,1],  v[...,0],         O], dim=-1)], dim=-2)
    
# def SO3_Adj(X: pp.LieTensor) -> pp.LieTensor:
#     I3x3 = torch.eye(3, device=X.device, dtype=X.dtype).expand(X.shape[:-1]+(3, 3))
#     Xv, Xw = X[..., :3], X[..., 3:]
#     Xw_3x3 = Xw.tensor().unsqueeze(-1) * I3x3
#     return 2.0 * Xw.unsqueeze(-1) * (Xw_3x3 + vec2skew(Xv)) - I3x3 + 2.0 * Xv.unsqueeze(-1) * Xv.unsqueeze(-2)

# def SE3_Adj(X: pp.LieTensor) -> pp.LieTensor:
#     Adj = torch.zeros((X.shape[:-1]+(6, 6)), device=X.device, dtype=X.dtype, requires_grad=False)
#     t, q = X[..., :3], X[..., 3:]
#     R3x3 = SO3_Adj(q)
#     tx = vec2skew(t)
#     Adj[..., :3, :3] = R3x3
#     Adj[..., :3, 3:] = torch.matmul(tx, R3x3)
#     Adj[..., 3:, 3:] = R3x3
#     return Adj

    
def SE3_Adj(T: pp.LieTensor) -> torch.Tensor:
    """Manually constructs the 6x6 Adjoint matrix for a batch of SE3 objects.
    
    Args:
        T (pp.SE3): A pypose.SE3 tensor of shape (B, K) or any other shape.

    Returns:
        torch.Tensor: The Adjoint matrices, shape (..., 6, 6)
    """
    # Extract rotation matrix (R) and translation vector (t)
    # T.matrix() returns a tensor of shape (..., 4, 4)
    T_matrix = T.matrix()
    R = T_matrix[..., :3, :3]

    # Create the 3x3 skew-symmetric matrix from the translation vector
    t_skew = pp.vec2skew(T.tensor()[..., :3]) # Shape: (..., 3, 3)

    # Initialize the 6x6 Adjoint matrix
    batch_shape = T.shape[:-1]
    device = T.device
    dtype = T_matrix.dtype
    Adj = torch.zeros(*batch_shape, 6, 6, device=device, dtype=dtype)

    # Fill the block matrix
    Adj[..., :3, :3] = R                  # Top-left
    Adj[..., 3:, 3:] = R                  # Bottom-right
    Adj[..., :3, 3:] = t_skew @ R         # Top-right

    return Adj


_AXES = {"x": 0, "y": 1, "z": 2}


def vertical_vector(vertical) -> torch.Tensor:
    """Unit vector of a vertical given as a map-frame axis name ("x", "y", "z") or as a 3-vector."""
    if isinstance(vertical, str):
        v = torch.zeros(3, dtype=torch.float64)
        v[_AXES[vertical.lower()]] = 1.0
        return v
    v = torch.as_tensor(vertical, dtype=torch.float64).reshape(3)
    return v / torch.linalg.vector_norm(v).clamp_min(1e-12)


def estimate_vertical(rotations: torch.Tensor, prior: torch.Tensor, min_turn_var: float = 0.01,
                      max_tilt_ratio: float = 0.2):
    """Vertical of the map frame from the camera orientations of a trajectory: the axis the camera turns about.

    Under rotations about one axis v, every camera axis moves on a circle around v, so the scatter of the three axis
    directions has no spread along v. The vertical is the eigenvector of the smallest eigenvalue of that scatter,
    accepted when the turns are large enough (second eigenvalue >= min_turn_var; 0.01 is a heading spread of about
    +-10 deg) and the motion is close to planar (smallest / second eigenvalue <= max_tilt_ratio). Otherwise the prior
    is returned. Uses only the estimated poses (no ground truth); the sign is aligned with the prior.

    Args:
        rotations: (N, 3, 3) camera-to-map rotation matrices.
        prior: (3,) unit vector returned when the trajectory does not determine the vertical.

    Returns:
        (vertical (3,) float64 unit vector, True if estimated / False if the prior was kept)
    """
    prior = prior.to(torch.float64)
    R = rotations.detach().to("cpu", torch.float64)
    if R.shape[0] < 3:
        return prior, False
    axes = R.transpose(1, 2).reshape(-1, 3, 3)                 # [n, i] = direction of camera axis i in the map
    d = axes - axes.mean(dim=0, keepdim=True)
    scatter = torch.einsum("nij,nik->jk", d, d) / R.shape[0]
    evals, evecs = torch.linalg.eigh(scatter)                  # ascending
    if evals[1] < min_turn_var or evals[0] > max_tilt_ratio * evals[1]:
        return prior, False
    v = evecs[:, 0]
    return (v if float(v @ prior) >= 0 else -v), True


class SE3Projection:
    """Place coordinates of poses for proposal clustering and proposal-to-hypothesis matching.

    (h1, h2, w * v, cos psi, sin psi): position along two horizontal axes of the map frame, the vertical position
    weighted by w (w = 0 drops it), and the heading psi of the camera's most horizontal axis about the vertical.
    With the vertical along y and w = 0 the distances are those of the original (x, z, yaw) projection.
    """

    def __init__(self, vertical, vertical_weight: float = 0.0):
        self.vertical = vertical_vector(vertical)
        self.vertical_weight = float(vertical_weight)
        v = self.vertical
        # heading reference: a camera axis of the first frame (= map frame) within 45 deg of horizontal, the optical
        # axis (z) if it is, else x (a down-looking camera), else y
        self.ref_axis = next(i for i in (2, 0, 1) if abs(float(v[i])) < math.sqrt(0.5) or i == 1)
        e = torch.zeros(3, dtype=torch.float64)
        e[self.ref_axis] = 1.0
        h1 = e - (e @ v) * v
        self.h1 = h1 / torch.linalg.vector_norm(h1)
        self.h2 = torch.linalg.cross(v, self.h1)

    def vertical_offset(self, T: pp.LieTensor) -> torch.Tensor:
        """Vertical position (metres along the vertical) of a batch of poses, shape (N,)."""
        t = T.tensor()[:, :3]
        return t @ self.vertical.to(t.device, t.dtype)

    def __call__(self, T: pp.LieTensor) -> torch.Tensor:
        t = T.tensor()[:, :3]
        dev, dt = t.device, t.dtype
        v, h1, h2 = (a.to(dev, dt) for a in (self.vertical, self.h1, self.h2))
        r = pp.SE3(T.tensor()).rotation().matrix()[:, :, self.ref_axis]
        psi = torch.atan2(r @ h2, r @ h1)
        cols = [t @ h1, t @ h2]
        if self.vertical_weight > 0:
            cols.append(self.vertical_weight * (t @ v))
        return torch.stack(cols + [torch.cos(psi), torch.sin(psi)], dim=1)

    def __repr__(self):
        v = [round(float(a), 4) for a in self.vertical]
        return f"SE3Projection(vertical={v}, weight={self.vertical_weight}, heading axis={'xyz'[self.ref_axis]})"


def split_clusters_by_vertical(labels, vertical, scores, gate: float):
    """Split each cluster (label >= 0) into groups whose vertical positions are within `gate` of the group's best-scoring
    member, taking members by decreasing score; the best group keeps the label, the others get new labels.

    Args:
        labels: (N,) int cluster labels (-1 = noise, left as is).
        vertical: (N,) vertical positions.
        scores: (N,) member scores.
    """
    import numpy as np
    labels = np.asarray(labels)
    out = labels.copy()
    next_label = int(labels.max()) + 1 if labels.size else 0
    for k in sorted(set(labels.tolist()) - {-1}):
        idx = np.where(labels == k)[0]
        remaining = idx[np.argsort(-np.asarray(scores)[idx], kind="stable")]
        first = True
        while remaining.size:
            near = np.abs(vertical[remaining] - vertical[remaining[0]]) <= gate
            if not first:
                out[remaining[near]] = next_label
                next_label += 1
            first = False
            remaining = remaining[~near]
    return out


def project_SE3(T: pp.LieTensor, use_heights: bool = False, projection: "SE3Projection" = None) -> torch.Tensor:
    """Project a batch of SE3 objects to (x, z, cos yaw, sin yaw), or to the place coordinates of `projection`.

    Args:
        T (pp.SE3): A pypose.SE3 tensor of shape (N, 7).
        projection: an SE3Projection; None keeps the original ground-robot projection (camera y vertical).

    Returns:
        torch.Tensor: The projected poses, shape (N, 4) (N, 5 with a weighted vertical coordinate)
    """
    if projection is not None:
        return projection(T)
    if use_heights:
        euler_angles = quaternion_to_euler_torch(T.tensor()[:, 3:])
        yaw = euler_angles[:, :1]
        return torch.cat([T.tensor()[:, :3], yaw], dim=1)

    else:

        euler_angles = quaternion_to_euler_torch(T.tensor()[:, 3:])
        yaw = euler_angles[:, 0]

        # Map yaw to the unit circle to avoid discontinuity
        yaw_cos = torch.cos(yaw)
        yaw_sin = torch.sin(yaw)
        
        translations = T.tensor()[:, :3]
        x = translations[:, 0]
        z = translations[:, 2]
        return torch.stack([x, z, yaw_cos, yaw_sin], dim=1)


def so3_angle_between(
    q1: torch.Tensor,
    q2: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Geodesic angle between two rotations given in quaternion (…,4).
    Returns a tensor of shape (…) with values in [0, π] (radians).
    """
    w1, v1 = q1[..., 3:4], q1[..., :3]        # split scalar / vector
    w2, v2 = q2[..., 3:4], q2[..., :3]

    w_rel = w2 * w1 + (v2 * v1).sum(-1, keepdim=True)   # scalar part
    v_rel = (w2 * (-v1) + w1 * v2 +
             torch.cross(v2, -v1, dim=-1))              # vector part

    # Δθ = 2·atan2(‖v_rel‖, w_rel)
    angle = 2.0 * torch.atan2(torch.linalg.norm(v_rel, dim=-1),
                              w_rel.squeeze(-1).clamp(-1.0, 1.0))

    # Numerical safety: map tiny negatives to zero, huge to π
    return angle.clamp(min=0.0, max=math.pi)

def rotation_angle_from_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Extracts the rotation angle θ ∈ [0, π] from a unit quaternion q = [ x, y, z, w].
    Input: q of shape (..., 4), assumed to be unit-norm
    Output: angle tensor of shape (...,)
    """
    w = q[..., 3].clamp(-1.0 + eps, 1.0 - eps)  # Clamp to avoid NaNs in acos
    angle = 2.0 * torch.acos(w)
    return angle
