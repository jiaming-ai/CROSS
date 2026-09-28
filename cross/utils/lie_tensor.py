# some of the functions are borrowed from pypose

import math
import pypose as pp
import torch
from cross.utils.rotation import quaternion_to_euler_torch

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


def project_SE3(T: pp.LieTensor, use_heights: bool = False) -> torch.Tensor:
    """Project a batch of SE3 objects to (x, z, yaw).
    
    Args:
        T (pp.SE3): A pypose.SE3 tensor of shape (B, K) or any other shape.
        
    Returns:
        torch.Tensor: The projected 3D points, shape (..., 3)
    """
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