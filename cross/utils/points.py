from typing import Optional, Union, Tuple, List
import torch
import numpy as np
from dataclasses import dataclass
from torch import Tensor
# from torch_geometric.nn.pool.consecutive import consecutive_cluster
# from torch_geometric.nn.pool.voxel_grid import voxel_grid
# from torch_geometric.utils import add_self_loops, scatter
from cross.core.types import Camera
from cross.utils.profile import timeit, timeblock

def project_xyz_to_pixel(xyz: torch.Tensor, camera: Camera):
    """project xyz to pixel using simple pinhole camera model"""
    # Extract camera parameters
    fx, fy = camera.fx, camera.fy
    cx, cy = camera.px, camera.py
    
    # Convert from (x: forward, y: left, z: up) to (x: right, y: down, z: forward)
    x, y, z = xyz.unbind(-1)
    x_cam, y_cam, z_cam = -y, -z, x
    
    # Avoid division by zero
    z_cam = torch.clamp(z_cam, min=1e-7)
    
    # Calculate pixel coordinates
    px = (fx * x_cam / z_cam) + cx
    py = (fy * y_cam / z_cam) + cy
    
    pixel_coords = torch.stack([px, py], dim=-1)
    
    return pixel_coords
 

def check_xyz_in_image(xyz: torch.Tensor, camera: Camera):
    """Check if xyz points are visible in the image
    Args:
        xyz: the xyz points to check, with shape (N, 3)
        camera: the camera parameters
    Returns:
        is_visible: a boolean tensor with shape (N,)
        pixel_coords: the pixel coordinates of the xyz points, with shape (N, 2)
    """
    # Project xyz to pixel coordinates
    pixel_coords = project_xyz_to_pixel(xyz, camera)
    
    # Extract x, y, z components
    x, y, z = xyz.unbind(-1)
    
    # Check if points are in front of the camera (z > 0)
    in_front = z > 0
    
    # Check if pixel coordinates are within image bounds
    px, py = pixel_coords.unbind(-1)
    in_bounds = (
        (px >= 0) & (px < camera.frame_width) &
        (py >= 0) & (py < camera.frame_height)
    )
    
    # Combine conditions
    is_visible = in_front & in_bounds
    
    return is_visible, pixel_coords

@timeit
def project_mp_to_uv(
    mp: torch.Tensor, 
    cam_pose: torch.Tensor,
    camera: Camera
) -> torch.Tensor:
    """Project map points to uv coordinates"""
    # project to camera frame
    T_cam_world = torch.inverse(cam_pose)
    mp_tensor = mp.to(cam_pose.device) @ T_cam_world[:3, :3].T \
        + T_cam_world[:3, 3].unsqueeze(0)
    is_visible, pixel_coords = check_xyz_in_image(
        mp_tensor, 
        camera
    )
    return is_visible, pixel_coords

@timeit
def project_mp_to_uv_no_check(
    mp: torch.Tensor, 
    cam_pose: torch.Tensor,
    camera: Camera
) -> torch.Tensor:
    """Project map points to uv coordinates"""
    # project to camera frame
    T_cam_world = torch.inverse(cam_pose)
    mp_tensor = mp.to(cam_pose.device) @ T_cam_world[:3, :3].T \
        + T_cam_world[:3, 3].unsqueeze(0)
    pixel_coords = project_xyz_to_pixel(
        mp_tensor, 
        camera
    )
    return pixel_coords


def unproject_masked_depth_to_xyz_coordinates(
    depth: torch.Tensor,
    pose: torch.Tensor,
    inv_intrinsics: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Returns the XYZ coordinates for a batch posed RGBD image.

    Args:
        depth: The depth tensor, with shape (B, 1, H, W)
        mask: The mask tensor, with the same shape as the depth tensor,
            where True means that the point should be masked (not included)
        inv_intrinsics: The inverse intrinsics, with shape (B, 3, 3)
        pose: The poses, with shape (B, 4, 4)

    Returns:
        XYZ coordinates, with shape (N, 3) where N is the number of points in
        the depth image which are unmasked
    """

    batch_size, _, height, width = depth.shape
    if mask is None:
        mask = torch.full_like(depth, fill_value=False, dtype=torch.bool)
    flipped_mask = ~mask

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=depth.device),
        torch.arange(0, height, device=depth.device),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1)[None, :, :].repeat_interleave(batch_size, dim=0)
    xy = xy[flipped_mask.squeeze(1)]
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Associates poses and intrinsics with XYZ coordinates.
    inv_intrinsics = inv_intrinsics[:, None, None, :, :].expand(
        batch_size, height, width, 3, 3
    )[flipped_mask.squeeze(1)]
    pose = pose[:, None, None, :, :].expand(batch_size, height, width, 4, 4)[
        flipped_mask.squeeze(1)
    ]
    depth = depth[flipped_mask]

    # Applies intrinsics and extrinsics.
    xyz = xyz.to(inv_intrinsics).unsqueeze(1) @ inv_intrinsics.permute([0, 2, 1])
    xyz = xyz * depth[:, None, None]
    xyz = (xyz[..., None, :] * pose[..., None, :3, :3]).sum(dim=-1) + pose[
        ..., None, :3, 3
    ]
    xyz = xyz.squeeze(1)

    return xyz

def depth_to_xyz(depth: torch.Tensor, camera: Camera):
    """get depth from numpy using simple pinhole camera model"""
    # TODO: convert to torch:
    # xs, ys = torch.meshgrid(
    #     torch.arange(0, width), torch.arange(0, height), indexing="xy", device=depth.device
    # )
    h, w = depth.shape[-2:]
    indices = np.indices((h, w), dtype=np.float32).transpose(
        1, 2, 0
    )
    z = depth

    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    x = torch.tensor(indices[:, :, 1] - camera.px).to(z.device) * (z / camera.fx)
    y = torch.tensor(indices[:, :, 0] - camera.py).to(z.device) * (z / camera.fy)

    # Should now be batch x height x width x 3, after this:
    # xyz is in different frame: x: right, y: down, z: forward
    # convert to x: forward, y: left, z: up
    # xyz = torch.stack([z, -x, -y], dim=-1)
    xyz = torch.stack([x, y, z], dim=-1)
    return xyz


def voxelize(
    pos: Tensor,
    voxel_size: float,
    batch: Optional[Tensor] = None,
    start: Optional[Union[float, Tensor]] = None,
    end: Optional[Union[float, Tensor]] = None,
) -> Tuple[Tensor]:
    """Returns voxel indices and packed (consecutive) indices for points

    Args:
        pos (Tensor): [N, 3] locations
        voxel_size (float): Size (resolution) of each voxel in the grid
        batch (Optional[Tensor], optional): Batch index of each point in pos. Defaults to None.
        start (Optional[Union[float, Tensor]], optional): Mins along each coordinate for the voxel grid.
            Defaults to None, in which case the starts are inferred from min values in pos.
        end (Optional[Union[float, Tensor]], optional):  Maxes along each coordinate for the voxel grid.
            Defaults to None, in which case the starts are inferred from max values in pos.
    Returns:
        voxel_idx (LongTensor): Idx of each point's voxel coordinate. E.g. [0, 0, 4, 3, 3, 4]
        cluster_consecutive_idx (LongTensor): Packed idx -- contiguous in cluster ID. E.g. [0, 0, 2, 1, 1, 2]
        batch_sample: See https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/max_pool.html
    """
    voxel_cluster = voxel_grid(
        pos=pos, batch=batch, size=voxel_size, start=start, end=end
    )
    cluster_consecutive_idx, perm = consecutive_cluster(voxel_cluster)
    batch_sample = batch[perm] if batch is not None else None
    cluster_idx = voxel_cluster
    return cluster_idx, cluster_consecutive_idx, batch_sample

def reduce_pointcloud(
    voxel_cluster: Tensor,
    pos: Tensor,
    features: Tensor,
    weights: Optional[Tensor] = None,
    rgbs: Optional[Tensor] = None,
    feature_reduce: str = "mean",
) -> Tuple[Tensor]:
    """Pools values within each voxel

    Args:
        voxel_cluster (LongTensor): [N] IDs of each point
        pos (Tensor): [N, 3] position of each point
        features (Tensor): [N, D] features at each point
        weights (Optional[Tensor], optional): [N,] weights of each point. Defaults to None.
        rgbs (Optional[Tensor], optional): [N, 3] colors of each point. Defaults to None.
        feature_reduce (str, optional): Feature reduction method. Defaults to 'mean'.

    Raises:
        NotImplementedError: if unknown reduction method

    Returns:
        pos_cluster (Tensor): weighted average position within each voxel
        feature_cluster (Tensor): aggregated feature of each voxel
        weights_cluster (Tensor): aggregated weights of each voxel
        rgb_cluster (Tensor): colors of each voxel
    """
    if weights is None:
        weights = torch.ones_like(pos[..., 0])
    weights_cluster = scatter(weights, voxel_cluster, dim=0, reduce="sum")

    pos_cluster = scatter_weighted_mean(
        pos, weights, voxel_cluster, weights_cluster, dim=0
    )

    if rgbs is not None:
        rgb_cluster = scatter_weighted_mean(
            rgbs, weights, voxel_cluster, weights_cluster, dim=0
        )
    else:
        rgb_cluster = None

    if features is None:
        return pos_cluster, None, weights_cluster, rgb_cluster

    if feature_reduce == "mean":
        feature_cluster = scatter_weighted_mean(
            features, weights, voxel_cluster, weights_cluster, dim=0
        )
    elif feature_reduce == "max":
        feature_cluster = scatter(features, voxel_cluster, dim=0, reduce="max")
    elif feature_reduce == "sum":
        feature_cluster = scatter(
            features * weights[:, None], voxel_cluster, dim=0, reduce="sum"
        )
    else:
        raise NotImplementedError(f"Unknown feature reduction method {feature_reduce}")

    return pos_cluster, feature_cluster, weights_cluster, rgb_cluster



def scatter_weighted_mean(
    features: Tensor,
    weights: Tensor,
    cluster: Tensor,
    weights_cluster: Tensor,
    dim: int,
) -> Tensor:
    """_summary_

    Args:
        features (Tensor): [N, D] features at each point
        weights (Optional[Tensor], optional): [N,] weights of each point. Defaults to None.
        cluster (LongTensor): [N] IDs of each point (clusters.max() should be <= N, or you'll OOM)
        weights_cluster (Tensor): [N,] aggregated weights of each cluster, used to normalize
        dim (int): Dimension along which to do the reduction -- should be 0

    Returns:
        Tensor: Agggregated features, weighted by weights and normalized by weights_cluster
    """
    assert dim == 0, "Dim != 0 not yet implemented"
    feature_cluster = scatter(
        features * weights[:, None], cluster, dim=dim, reduce="sum"
    )
    feature_cluster = feature_cluster / weights_cluster[:, None]
    return feature_cluster

