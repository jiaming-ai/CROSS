import torch
import numpy as np

from kornia.geometry import conversions as KC

def rotation_matrix_to_euler_angles(R, use_degrees=True):
    """
    Extract Euler angles from a rotation matrix.
    Assumes ZYX rotation order (extrinsic rotations).
    
    Args:
    R (numpy.ndarray): 3x3 rotation matrix
    use_degrees (bool): If True, return angles in degrees; otherwise in radians
    
    Returns:
    tuple: (roll, pitch, yaw) in degrees or radians
    """
    assert(np.isclose(np.linalg.det(R), 1.0))

    sy = np.sqrt(R[0,0] * R[0,0] +  R[0,1] * R[0,1])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[1,2], R[2,2])  # roll
        y = np.arctan2(-R[0,2], sy)     # pitch
        z = np.arctan2(R[0,1], R[0,0])  # yaw
    else:
        x = np.arctan2(-R[2,1], R[1,1])
        y = np.arctan2(-R[0,2], sy)
        z = 0

    angles = np.array([x, y, z])
    
    if use_degrees:
        return np.degrees(angles)
    else:
        return angles
    
def quaternion_to_rotation_matrix_torch(q: np.ndarray) -> torch.Tensor:
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
    rotation_matrix = torch.tensor([[r00, r01, r02],
                                    [r10, r11, r12],
                                    [r20, r21, r22]])

    return rotation_matrix
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


def rotation_matrix_to_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Convert a 3x3 rotation matrix to axis-angle representation.
    
    Args:
        R (np.ndarray): 3x3 rotation matrix
    
    Returns:
        tuple: (axis, angle_in_degrees)
            axis (np.ndarray): A 3D unit vector representing the rotation axis
            angle_in_degrees (float): The rotation angle in degrees
    """
    # Ensure R is a valid rotation matrix
    assert np.allclose(np.dot(R, R.T), np.eye(3)) and np.isclose(np.linalg.det(R), 1)

    # Calculate the angle of rotation
    angle_rad = np.arccos((np.trace(R) - 1) / 2)
    angle_deg = np.degrees(angle_rad)

    # If angle is close to 0 or 180 degrees, handle special cases
    if np.isclose(angle_rad, 0):
        return np.array([0, 0, 1]), 0.0
    elif np.isclose(angle_rad, np.pi):
        # Find the largest diagonal element
        i = np.argmax(np.diag(R))
        axis = np.sqrt((R[i, i] + 1) / 2)
        axis = np.array([
            R[0, i] / (2 * axis),
            R[1, i] / (2 * axis),
            R[2, i] / (2 * axis)
        ])
        return axis, 180.0

    # Calculate the rotation axis
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    axis /= np.linalg.norm(axis)

    return axis, angle_deg

def rotation_matrix_to_axis_angle_tensor(R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a 3x3 rotation matrix to axis-angle representation using PyTorch tensors.
    
    Args:
        R (torch.Tensor): 3x3 rotation matrix
    
    Returns:
        tuple: (axis, angle_in_degrees)
            axis (torch.Tensor): A 3D unit vector representing the rotation axis
            angle_in_degrees (torch.Tensor): The rotation angle in degrees
    """
    # Ensure R is a valid rotation matrix with some tolerance
    identity = torch.eye(3, device=R.device)
    tol = 1e-4  # Increased tolerance for numerical stability
    is_valid = (torch.allclose(torch.matmul(R, R.transpose(-2, -1)), identity, atol=tol) and 
                torch.allclose(torch.linalg.det(R), torch.tensor(1.0, device=R.device), atol=tol))
    
    if not is_valid:
        # Project to valid rotation matrix if needed
        U, _, Vh = torch.linalg.svd(R)
        R = torch.matmul(U, Vh)

    # Calculate the angle of rotation
    angle_rad = torch.acos((torch.trace(R) - 1) / 2)
    angle_deg = torch.rad2deg(angle_rad)

    # If angle is close to 0 or 180 degrees, handle special cases
    if torch.isclose(angle_rad, torch.tensor(0.0)):
        return torch.tensor([0.0, 0.0, 1.0], device=R.device), torch.tensor(0.0, device=R.device)
    elif torch.isclose(angle_rad, torch.tensor(torch.pi)):
        # Find the largest diagonal element
        i = torch.argmax(torch.diag(R))
        axis = torch.sqrt((R[i, i] + 1) / 2)
        axis = torch.tensor([
            R[0, i] / (2 * axis),
            R[1, i] / (2 * axis),
            R[2, i] / (2 * axis)
        ], device=R.device)
        return axis, torch.tensor(180.0, device=R.device)

    # Calculate the rotation axis
    axis = torch.tensor([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ], device=R.device)
    axis /= torch.linalg.norm(axis)

    return axis, angle_deg

    

def add_batched_gaussian_noise_to_pose(T, std_trans, std_rot):
    """
    Adds batched Gaussian noise to a batch of 4x4 homogeneous transformation matrices (poses).

    Args:
        T (torch.Tensor): Batched 4x4 homogeneous transformation matrices (B, 4, 4).
        std_trans (float): Standard deviation for translation noise (isotropic).
        std_rot (float): Standard deviation for rotation noise (isotropic, in rotation vector magnitude).
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Batched noisy 4x4 homogeneous transformation matrices (B, 4, 4).
    """
    batch_size = T.shape[0]
    device = T.device

    # 1. Map SE(3) to se(3) (extract translation and rotation vector) - Batched
    R_matrices = T[:, :3, :3]  # (B, 3, 3)
    t_vectors = T[:, :3, 3]   # (B, 3)

    # Kornia's rotation_matrix_to_angle_axis expects (B, 3, 3) and returns angle axis in (B, 3)
    rot_vecs = KC.rotation_matrix_to_axis_angle(R_matrices) # (B, 3) - rotation vectors (angle-axis)

    xi_t = torch.cat([t_vectors, rot_vecs], dim=-1)  # (B, 6) - batched se(3) representation

    # 2. Add Gaussian noise in se(3) - Batched
    noise_trans = torch.randn(batch_size, 3, device=device) * std_trans
    noise_rot = torch.randn(batch_size, 3, device=device) * std_rot
    noise_xi = torch.cat([noise_trans, noise_rot], dim=-1) # (B, 6)
    xi_noisy = xi_t + noise_xi # (B, 6)

    # 3. Map noisy se(3) back to SE(3) - Batched
    t_noisy = xi_noisy[:, :3] # (B, 3)
    rot_vec_noisy = xi_noisy[:, 3:] # (B, 3)

    # Kornia's angle_axis_to_rotation_matrix expects (B, 3) and returns (B, 3, 3)
    R_noisy = KC.axis_angle_to_rotation_matrix(rot_vec_noisy) # (B, 3, 3)

    T_noisy = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1) # (B, 4, 4) - batched identity matrices
    T_noisy[:, :3, :3] = R_noisy # (B, 4, 4)
    T_noisy[:, :3, 3] = t_noisy # (B, 4, 4)

    return T_noisy

    
def quaternion_to_euler(q: np.ndarray, use_degrees: bool = True, invert: bool = True):
    # optionally invert to go from world→camera to camera→world
    if invert:
        x, y, z, w = -q[0], -q[1], -q[2], q[3]
    else:
        x, y, z, w = q

    # build parts of rotation matrix
    r02 = 2*(x*z + y*w)
    r12 = 2*(y*z - x*w)
    r22 = 1 - 2*(x*x + y*y)
    r10 = 2*(x*y + z*w)
    r11 = 1 - 2*(x*x + z*z)

    # pitch (X-axis)
    pitch = np.arcsin(np.clip(-r12, -1, 1))

    # yaw (Y-axis) — camera’s Y points down, so flip the sign to get “+Y up”
    yaw = -np.arctan2(r02, r22)

    # roll (Z-axis) stays the same
    roll = np.arctan2(r10, r11)

    angles = np.array([yaw, pitch, roll])
    if use_degrees:
        angles = np.degrees(angles)
    return angles

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