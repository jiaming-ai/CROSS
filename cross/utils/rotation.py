import torch
import numpy as np

from kornia.geometry import conversions as KC

    
def quaternion_to_rotation_matrix_numpy(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a rotation matrix.
    Args:
        q: A numpy array of shape (4,) representing the quaternion.
    Returns:
        A numpy array of shape (3, 3) representing the rotation matrix.
    """
    # Unpack the quaternion
    qx, qy, qz, qw = q

    r00 = 1 - 2 * (qy ** 2 + qz ** 2)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)

    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx ** 2 + qz ** 2)
    r12 = 2 * (qy * qz - qx * qw)

    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx ** 2 + qy ** 2)

    # Create the rotation matrix
    rotation_matrix = np.array([[r00, r01, r02],
                                [r10, r11, r12],
                                [r20, r21, r22]])

    return rotation_matrix


    


    

def quaternion_to_euler_torch(
    q: torch.Tensor,
    use_degrees: bool = False,
) -> torch.Tensor:
    """
    Convert a batch of quaternions to Euler angles (yaw, pitch, roll).

    Args:
        q (torch.Tensor): shape (B,4), each row = [x, y, z, w] (OpenCV convention).
        use_degrees (bool): if True, output angles in degrees; else in radians.

    Returns:
        torch.Tensor: shape (B,3), columns = [yaw, pitch, roll].
    """
    # Split and (optionally) conjugate
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    # Build only the needed R elements
    r02 = 2.0 * (x * z + y * w)
    r12 = 2.0 * (y * z - x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)

    # Compute angles
    pitch = torch.asin(torch.clamp(-r12, -1.0, 1.0))        # rotation about X
    yaw   = -torch.atan2(r02, r22)                         # rotation about Y (flip for +Y up)
    roll  =  torch.atan2(r10, r11)                         # rotation about Z

    # Stack into (B,3)
    angles = torch.stack((yaw, pitch, roll), dim=1)

    if use_degrees:
        angles = angles * (180.0 / torch.pi)

    return angles