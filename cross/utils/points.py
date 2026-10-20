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


