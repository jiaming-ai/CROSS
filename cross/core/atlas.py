import torch

def new_atlas_center(system):
    """Create a new atlas center adaptively based on current map size.

    Places the new origin outside the current map bounds with a margin
    proportional to the map size, ensuring collision-free placement while
    keeping it as close as possible.

    Returns:
        List[float]: [x, y, z] coordinates for the new atlas center
    """
    all_kfs = system.hypothesis_manager.nodes.values()
    all_poses = [kf.pose_mu[0] for kf in all_kfs]
    all_poses = torch.stack(all_poses).to(system.device)
    all_poses = all_poses.tensor()
    all_poses = all_poses[:, :3]  # (N, 3) - extract translation only

    # Compute bounding box of existing map
    x_min, x_max = all_poses[:, 0].min(), all_poses[:, 0].max()
    y_min, y_max = all_poses[:, 1].min(), all_poses[:, 1].max()
    z_min, z_max = all_poses[:, 2].min(), all_poses[:, 2].max()

    # Compute map extents
    x_extent = x_max - x_min
    y_extent = y_max - y_min
    z_extent = z_max - z_min

    # Use the maximum horizontal extent (x and z) to determine margin
    # Vertical (y) is typically smaller and less relevant for margin calculation
    horizontal_extent = max(x_extent, z_extent)

    # Adaptive margin: 30% of horizontal extent, with bounds [2m, 20m]
    # - Minimum 2m for very small maps
    # - Maximum 20m for very large maps to avoid excessive separation
    # - 30% scaling for medium-sized maps
    margin = max(2.0, min(20.0, float(horizontal_extent) * 0.3))

    # Place new origin at x_max + margin (forward/positive x direction)
    # Keep y and z at their max values to stay in similar vertical/lateral region
    new_x_origin = float(x_max) + margin
    new_y_origin = float(y_max)
    new_z_origin = float(z_max)

    return [new_x_origin, new_y_origin, new_z_origin]