from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List, Union
import torch

import pypose as pp

from cross.core.types import Camera, Keyframe
from cross.core.config import PoseEstConfig, PoseEstType, config_to_dict as _cfg_to_dict
from cross.utils.profile import timeit
from cross.cv.kp_det import get_kp_det
from cross.cv.kp_match import get_kp_match
from loguru import logger
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dataclasses

from cross.utils.points import depth_to_xyz
from cross.visualization.viz2d import create_window, show_plot
from cross.utils.rotation import quaternion_to_rotation_matrix_numpy
from kornia.geometry import conversions as KC


def visualize_pose_estimation_matches(
    ref_image: torch.Tensor,
    curr_image: torch.Tensor,
    ref_keypoints: torch.Tensor,
    curr_keypoints: torch.Tensor,
    matches: Dict,
    retrieval_scores: List[float] = None,
    keyframes: List = None,
    est_poses_valid: Optional[torch.Tensor] = None,
    est_valid_masks: Optional[np.ndarray] = None,
    est_covs_valid: Optional[torch.Tensor] = None,
    device: str = "cuda",
    viz_fig=None,
    top_k: int = 6
):
    """Visualize pose estimation matches between reference and current images.
    
    Args:
        ref_image: Reference images (B, 3, H, W)
        curr_image: Current image (3, H, W)
        ref_keypoints: Reference keypoints (B, M, 2)
        curr_keypoints: Current keypoints (B, N, 2)
        matches: Match results dictionary
        retrieval_scores: Optional retrieval scores for ranking
        keyframes: Optional keyframes for timestamps
        est_poses_valid: Optional valid poses for pose information
        est_valid_masks: Optional valid masks for pose information
        est_covs_valid: Optional valid covariances for pose information
        device: Device string
        viz_fig: Optional existing figure to reuse
        top_k: Number of top matches to visualize
        
    Returns:
        viz_fig: The figure used for visualization
    """
    batch_size = ref_image.shape[0]
    
    # Determine which reference images to visualize
    if retrieval_scores is not None:
        retrieval_scores = torch.tensor(retrieval_scores).to(device)
        # Get top k indices based on retrieval weights
        _, top_indices = torch.topk(retrieval_scores, min(top_k, len(retrieval_scores)), largest=True)
        top_indices = top_indices.cpu().numpy()
        k = len(top_indices)
    else:
        # Use first k images if no weights provided
        k = min(top_k, batch_size)
        top_indices = list(range(k))
    
    timestamps = torch.tensor([kf.timestamp for kf in keyframes]) if keyframes else None

    # Calculate grid layout: max 4 columns, 1-3 rows
    cols = min(4, k)
    rows = (k + cols - 1) // cols  # Ceiling division
    rows = min(3, rows)  # Max 3 rows
    
    # Create or reuse persistent figure
    if viz_fig is None:
        viz_fig = create_window()
    
    # Set figure size and clear
    viz_fig.set_size_inches(4*cols, 3*rows)
    viz_fig.clear()
    axes = viz_fig.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    # Convert current image to numpy once
    curr_img_np = curr_image.permute(1, 2, 0).cpu().numpy()
    curr_img_np = np.clip(curr_img_np, 0, 1)
    curr_kps_np = curr_keypoints[0].cpu().numpy()  # Same for all since repeated
    
    # Batch process reference images and keypoints
    ref_imgs_np = []
    ref_kps_list = []
    match_counts = []
    scores = []
    
    for i, idx in enumerate(top_indices):
        # Convert reference image
        ref_img_np = ref_image[idx].permute(1, 2, 0).cpu().numpy()
        ref_img_np = np.clip(ref_img_np, 0, 1)
        ref_imgs_np.append(ref_img_np)
        
        # Get keypoints
        ref_kps_np = ref_keypoints[idx].cpu().numpy()
        ref_kps_list.append(ref_kps_np)
        
        # Get match count
        match_count = len(matches["matches"][idx])
        match_counts.append(match_count)
        
        # Get score
        score = retrieval_scores[idx].item() if retrieval_scores is not None else 0.0
        scores.append(score)
    
    # Plot each reference-current pair
    for i, idx in enumerate(top_indices):
        row = i // cols
        col = i % cols
        
        if row >= len(axes) or col >= len(axes[0]):
            break
        
        ax = axes[row][col]
        
        # Create side-by-side images
        ref_img = ref_imgs_np[i]
        h, w = ref_img.shape[:2]
        combined_img = np.zeros((h, w*2, 3))
        combined_img[:, :w] = ref_img
        combined_img[:, w:] = curr_img_np
        
        ax.imshow(combined_img)
        
        # Create title with timestamp or index
        if timestamps is not None:
            title = f'{timestamps[idx]} | {scores[i]:.2f} | {match_counts[i]}'
        else:
            title = f'{idx} | {scores[i]:.2f} | {match_counts[i]}'
        
        # Add estimated pose information to the title
        pose_info_str = " | Pose: N/A" # Default
        if est_valid_masks is not None and est_poses_valid is not None and est_poses_valid.nelement() > 0:
            if idx < len(est_valid_masks) and est_valid_masks[idx]:
                # This original image (idx) had a valid pose. Find its position in est_poses_valid.
                # Count True values in est_valid_masks up to (but not including) idx.
                pose_idx_in_valid_tensor = np.sum(est_valid_masks[:idx])
                
                if pose_idx_in_valid_tensor < est_poses_valid.shape[0]:
                    current_pose_tensor = est_poses_valid[pose_idx_in_valid_tensor].cpu().tensor()
                    t_vec = current_pose_tensor[:3].numpy()
                    
                    try:
                        # current_pose_tensor is a 7-element torch.Tensor [tx, ty, tz, qx, qy, qz, qw]
                        # Extract quaternion components directly from the pypose tensor
                        q_x_val = current_pose_tensor[3] 
                        q_y_val = current_pose_tensor[4]
                        q_z_val = current_pose_tensor[5]
                        q_w_val = current_pose_tensor[6] # pypose stores w as the last component

                        # Convert quaternion to Euler angles (roll, pitch, yaw in radians)
                        # Kornia's euler_from_quaternion expects (w, x, y, z)
                        roll_rad, pitch_rad, yaw_rad = KC.euler_from_quaternion(q_w_val, q_x_val, q_y_val, q_z_val)
                        
                        # Convert to degrees
                        # .item() is used to get scalar Python numbers from single-element Tensors
                        yaw_deg = yaw_rad.item() * 180.0 / np.pi
                        pitch_deg = pitch_rad.item() * 180.0 / np.pi
                        roll_deg = roll_rad.item() * 180.0 / np.pi

                        pose_info_str = f" | T[{t_vec[0]:.1f},{t_vec[1]:.1f},{t_vec[2]:.1f}], YPR[{yaw_deg:.1f},{pitch_deg:.1f},{roll_deg:.1f}]deg"
                    except Exception as e:
                        logger.warning(f"Error converting pose {current_pose_tensor} to RPY for viz: {e}")
                        pose_info_str = " | Pose: RPY_ERR"
                else:
                    # This case implies a logic error if est_valid_masks[idx] is True
                    logger.error(f"Pose index mapping error for idx {idx}, pose_idx_in_valid_tensor {pose_idx_in_valid_tensor}, est_poses_valid shape {est_poses_valid.shape}")
                    pose_info_str = " | Pose: Idx_ERR"
            elif idx < len(est_valid_masks) and not est_valid_masks[idx]:
                pose_info_str = " | Pose: Invalid"
            # If idx >= len(est_valid_masks), it defaults to "N/A"
            
        title += pose_info_str
        ax.set_title(title, fontsize=9) # Reduced fontsize for longer titles
        ax.axis('off')
        
        # Plot keypoints
        ref_kps = ref_kps_list[i]
        curr_kps_shifted = curr_kps_np.copy()
        curr_kps_shifted[:, 0] += w  # Shift current keypoints to right side
        
        # Plot reference keypoints (red)
        if len(ref_kps) > 0:
            ax.scatter(ref_kps[:, 0], ref_kps[:, 1], c='red', s=1, alpha=0.5)
        
        # Plot current keypoints (blue)
        if len(curr_kps_shifted) > 0:
            ax.scatter(curr_kps_shifted[:, 0], curr_kps_shifted[:, 1], c='blue', s=1, alpha=0.5)
        
        # Plot matches (green lines) - limit to 30 for performance
        if match_counts[i] > 0:
            match_indices = matches["matches"][idx].cpu().numpy()
            ref_matched_kps = ref_kps[match_indices[:, 0]]
            curr_matched_kps = curr_kps_np[match_indices[:, 1]]
            curr_matched_kps_shifted = curr_matched_kps.copy()
            curr_matched_kps_shifted[:, 0] += w
            
            for j in range(min(len(ref_matched_kps), 30)):
                ax.plot([ref_matched_kps[j, 0], curr_matched_kps_shifted[j, 0]], 
                       [ref_matched_kps[j, 1], curr_matched_kps_shifted[j, 1]], 
                       'lime', linewidth=0.5, alpha=0.7)
    
    # Hide unused subplots
    for i in range(k, rows * cols):
        row = i // cols
        col = i % cols
        if row < len(axes) and col < len(axes[0]):
            axes[row][col].axis('off')
    
    viz_fig.tight_layout()
    show_plot(blocking=False)
    
    return viz_fig

def estimate_covariance_piecewise_interp(
    inlier_counts: torch.Tensor,
    base_cov_diag: torch.Tensor,
    steps: torch.Tensor = torch.tensor([10, 20, 30, 50, 90]),
    scales: torch.Tensor = torch.tensor([6.0, 3.0, 2.0, 1.5, 1.2, 1.0]),
) -> torch.Tensor:
    """
    Estimates a batch of diagonal covariances using a piecewise function with
    constant ends and interpolated middle segments.

    The mapping from inlier count to variance scale is defined as follows:
    - If `count < steps[0]`, scale is `scales[0]`.
    - If `count >= steps[-1]`, scale is `scales[-1]`.
    - For a `count` in the interval `[steps[i], steps[i+1])`, the scale is
      linearly interpolated between the point `(steps[i], scales[i+1])` and
      `(steps[i+1], scales[i+2])`.

    Args:
        inlier_counts (torch.Tensor): (B,) tensor of inlier counts.
        base_cov_diag (torch.Tensor): (6,) tensor for the base diagonal covariance.
        steps (torch.Tensor): A sorted 1D tensor of N inlier counts defining the
                               boundaries. E.g., `torch.tensor([10, 20, 80])`.
        scales (torch.Tensor): A 1D tensor of N+1 variance scale factors. The first and
                                last are for the constant regions, the middle ones are
                                for the interpolation points. E.g., `torch.tensor([5.0, 3.0, 1.0, 0.5])`.

    Returns:
        torch.Tensor: (B, 6) tensor of estimated diagonal covariances.
    """
    # --- Input Validation ---
    if not len(scales) == len(steps) + 1:
        raise ValueError("Length of `scales` must be length of `steps` + 1.")
    # assert torch.all(steps[:-1] <= steps[1:]), "`steps` tensor must be sorted."

    # Move definition tensors to the correct device
    steps = steps.to(inlier_counts.device, dtype=torch.float32)
    scales = scales.to(inlier_counts.device, dtype=torch.float32)
    inlier_counts_f = inlier_counts.float()

    # Initialize the output tensor with a default value (e.g., from the first scale)
    final_scales = torch.full_like(inlier_counts_f, scales[0])

    # --- Handle all interpolation regions with a loop ---
    # This loop covers all intervals [steps[i], steps[i+1])
    for i in range(len(steps) - 1):
        # Define the points for the linear segment
        lower_bound = steps[i]
        upper_bound = steps[i+1]
        lower_scale = scales[i+1] # Note the indexing: scales[i+1]
        upper_scale = scales[i+2] # and scales[i+2]

        # Create a mask for counts within this interval
        mask = (inlier_counts_f >= lower_bound) & (inlier_counts_f < upper_bound)
        
        if mask.any():
            counts_in_interval = inlier_counts_f[mask]
            
            # Normalize progress within the interval to [0, 1]
            progress = (counts_in_interval - lower_bound) / (upper_bound - lower_bound)
            
            # Linearly interpolate (LERP) and assign
            interp_scales = lower_scale + progress * (upper_scale - lower_scale)
            final_scales[mask] = interp_scales
            
    # --- Handle the two constant regions at the ends ---
    # The first region (< steps[0]) is already set by the initialization.
    # The last region (>= steps[-1]) needs to be set now.
    final_scales[inlier_counts_f >= steps[-1]] = scales[-1]
    
    # Compute final covariance matrix diagonals
    final_cov_diag = final_scales.unsqueeze(1) * base_cov_diag.to(final_scales.device)
    
    return final_cov_diag, final_scales


@timeit
def pnp_ransac(
    points3d: torch.Tensor, 
    points2d: torch.Tensor, 
    camera: Camera,
    iterations_count: int = 50,
    reprojection_error: float = 3.0,
    confidence: float = 0.99
) -> Optional[torch.Tensor]:
    """Estimate camera pose using PnP RANSAC.
    
    Args:
        points3d: Model 3D coordinates (N x 3)
        points2d: Scene 2D coordinates (N x 2)
        camera: Camera
        iterations_count: Number of RANSAC iterations
        reprojection_error: Maximum reprojection error
        confidence: RANSAC confidence level
    
    Returns:
        pose: 7-dim tensor [tx, ty, tz, qx, qy, qz, qw] or None if failed
    """
    # Add check for minimum number of points
    if len(points3d) < 4 or len(points2d) < 4:
        logger.warning(f"Not enough points for PnP RANSAC. Need at least 4, got {len(points3d)}")
        return None

    assert points3d.shape[0] == points2d.shape[0], "Mismatched number of 2D and 3D points"

    # Convert torch tensors to numpy for OpenCV compatibility
    # Use float64 for both arrays to satisfy OpenCV type requirements.
    points3d_np = points3d.detach().cpu().numpy().astype(np.float64)
    points2d_np = points2d.detach().cpu().numpy().astype(np.float64)
    camera_matrix_np = camera.K.astype(np.float64)
    
    # Solve PnP with RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points3d_np, points2d_np,
        camera_matrix_np,
        distCoeffs=None,
        iterationsCount=iterations_count,
        reprojectionError=reprojection_error,
        confidence=confidence,
    )
    
    if not success:
        return None, 0
    
    R_opencv, _ = cv2.Rodrigues(rvec)

    # OpenCV returns R, t that transform points from the reference (object)
    # frame into the *current camera* frame:
    #     X_cam = R * X_ref + t
    # For SLAM we want the camera pose (camera -> reference transform), i.e.:
    #     X_ref = R^T * X_cam - R^T * t
    # Therefore, the camera translation expressed in the reference frame is
    #     t_ref = -R^T * t
    R_ref_cam = R_opencv.T            # rotation from camera to reference
    t_ref_cam = -R_ref_cam @ tvec     # translation of camera origin in reference frame

    T = torch.eye(4)
    T[:3, :3] = torch.from_numpy(R_ref_cam).float()
    T[:3, 3] = torch.from_numpy(t_ref_cam.squeeze()).float()

    pose = pp.from_matrix(T, ltype=pp.SE3_type)

    return pose, inliers.shape[0]


def visualize_pose_3d(poses: list, ref_image: torch.Tensor, curr_image: torch.Tensor, 
                     scale: float = 1.0, save_path: str = None):
    """Visualize estimated poses in 3D with camera frames.
    
    Args:
        poses: List of 7-dim pose tensors [tx, ty, tz, qx, qy, qz, qw]
        ref_image: Reference image tensor (B, 3, H, W) or (3, H, W)
        curr_image: Current image tensor (3, H, W)
        scale: Scale factor for camera frame axes
        save_path: Optional path to save the visualization
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Create subplots: 3D view, reference image, current image  
    ax_3d = fig.add_subplot(131, projection='3d')
    ax_ref = fig.add_subplot(132)
    ax_curr = fig.add_subplot(133)
    
    # Plot reference camera at origin
    ax_3d.quiver(0, 0, 0, scale, 0, 0, color='red', arrow_length_ratio=0.1, label='X (right)')
    ax_3d.quiver(0, 0, 0, 0, scale, 0, color='green', arrow_length_ratio=0.1, label='Y (down)')
    ax_3d.quiver(0, 0, 0, 0, 0, scale, color='blue', arrow_length_ratio=0.1, label='Z (forward)')
    
    # Plot current camera poses
    for i, pose in enumerate(poses):
        if pose is not None:
            # Extract translation and quaternion
            t = pose[:3].cpu().numpy()
            quat = pose[3:].cpu().numpy()  # [qx, qy, qz, qw]
            
            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix_numpy(quat)
            
            # Plot camera position
            ax_3d.scatter(t[0], t[1], t[2], c='orange', s=50, label=f'Camera {i+1}' if i == 0 else "")
            
            # Plot camera coordinate axes
            axis_colors = ['red', 'green', 'blue']
            axis_labels = ['X', 'Y', 'Z']
            for j, (color, label) in enumerate(zip(axis_colors, axis_labels)):
                axis = R[:, j] * scale * 0.5  # Scale down relative to reference
                ax_3d.quiver(t[0], t[1], t[2], 
                           axis[0], axis[1], axis[2], 
                           color=color, alpha=0.7, arrow_length_ratio=0.1)
    
    # Set 3D plot properties
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y') 
    ax_3d.set_zlabel('Z')
    ax_3d.legend()
    ax_3d.set_title('Camera Poses')
    
    # Plot reference image
    if ref_image.dim() == 4:  # Batch dimension
        ref_img_np = ref_image[0].permute(1, 2, 0).cpu().numpy()
    else:
        ref_img_np = ref_image.permute(1, 2, 0).cpu().numpy()
    
    # Ensure image is in [0, 1] range for display
    if ref_img_np.max() > 1.0:
        ref_img_np = ref_img_np / 255.0
    
    ax_ref.imshow(np.clip(ref_img_np, 0, 1))
    ax_ref.set_title('Reference Image')
    ax_ref.axis('off')
    
    # Plot current image
    curr_img_np = curr_image.permute(1, 2, 0).cpu().numpy()
    if curr_img_np.max() > 1.0:
        curr_img_np = curr_img_np / 255.0
        
    ax_curr.imshow(np.clip(curr_img_np, 0, 1))
    ax_curr.set_title('Current Image')
    ax_curr.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


class PoseEst(ABC):
    """Estimate relative pose between two images"""
    def __init__(self, device: str, config: Union[PoseEstConfig, Dict]):
        self.device = device
        if isinstance(config, PoseEstConfig):
            self.config = config
        else:
            # Legacy dict support
            from cross.core.config import _from_dict
            self.config = _from_dict(PoseEstConfig, config)

        # base covariance diagonal for PnP RANSAC
        # x, y, z, qx, qy, qz
        # z is the camera direction, so it should be larger
        self.base_cov_diag = torch.tensor([0.1, 0.1, 0.15, 0.1, 0.1, 0.1])

    @abstractmethod
    def estimate_pose(self, image: torch.Tensor, depth: torch.Tensor):
        pass


def _dc_to_dict(obj) -> dict:
    """Convert a dataclass to a plain dict for legacy consumers."""
    if dataclasses.is_dataclass(obj):
        from enum import Enum
        result = {}
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if isinstance(v, Enum):
                result[f.name] = v.value
            elif dataclasses.is_dataclass(v):
                result[f.name] = _dc_to_dict(v)
            else:
                result[f.name] = v
        return result
    return obj


class PoseEstPnP(PoseEst):
    """Estimate relative pose between two images using PnP"""
    def __init__(self, device: str, config: Union[PoseEstConfig, Dict], camera: Camera = None):
        super().__init__(device, config)
        cfg = self.config  # now always PoseEstConfig
        self.max_depth = cfg.max_depth
        # kp_det and kp_match factories still expect plain dicts
        det_dict = _dc_to_dict(cfg.kp_detector)
        match_dict = _dc_to_dict(cfg.kp_matcher)
        self.kp_det = get_kp_det(
            cfg.kp_detector.type.value,
            device,
            det_dict,
        )
        self.kp_match = get_kp_match(
            cfg.kp_matcher.type.value,
            cfg.kp_detector.type.value,
            device,
            match_dict,
        )
        self.camera = camera

        # Initialize persistent visualization figure
        self.viz_fig = None
        self.visualize = False

    def _match_keypoints_batched(self, ref_kp_det_res: List, curr_kp_det_res, img_size: Tuple[int, int]):
        """
        Perform batched keypoint matching when all reference images have the same number of keypoints.
        
        Args:
            ref_kp_det_res: List of DetectionResult objects for reference images
            curr_kp_det_res: Single DetectionResult object for current image
            img_size: Image size (W, H)
            
        Returns:
            matches: Match results from kp_match.match()
        """
        batch_size = len(ref_kp_det_res)
        
        # Extract keypoints and descriptors from detection results
        ref_keypoints = torch.stack([res.keypoints for res in ref_kp_det_res])  # [B x M x 2]
        ref_descriptors = torch.stack([res.descriptors for res in ref_kp_det_res])  # [B x M x D]
        
        # curr_kp_det_res is a single DetectionResult object, expand to batch
        curr_keypoints = curr_kp_det_res.keypoints.unsqueeze(0).repeat(batch_size, 1, 1)  # [B x N x 2]
        curr_descriptors = curr_kp_det_res.descriptors.unsqueeze(0).repeat(batch_size, 1, 1)  # [B x N x D]
        
        data = {
            "image0": {
                "image_size": img_size,
                "keypoints": ref_keypoints,  # [B x M x 2]
                "descriptors": ref_descriptors,  # [B x M x D]
            },
            "image1": {
                "image_size": img_size,
                "keypoints": curr_keypoints,  # [B x N x 2]
                "descriptors": curr_descriptors,  # [B x N x D]
            }
        }
        
        return self.kp_match.match(data)

    def _match_keypoints_grouped(self, ref_kp_det_res: List, curr_kp_det_res, img_size: Tuple[int, int]):
        """
        Perform grouped keypoint matching when reference images have different numbers of keypoints.
        Groups images by keypoint count and performs batched inference within each group.
        
        Args:
            ref_kp_det_res: List of DetectionResult objects for reference images
            curr_kp_det_res: Single DetectionResult object for current image
            img_size: Image size (W, H)
            
        Returns:
            matches: Match results organized by original batch order
        """
        batch_size = len(ref_kp_det_res)
        
        # Group indices by keypoint count
        kp_count_groups = {}
        for i, res in enumerate(ref_kp_det_res):
            kp_count = len(res.keypoints)
            if kp_count not in kp_count_groups:
                kp_count_groups[kp_count] = []
            kp_count_groups[kp_count].append(i)
        
        # Process each group with batched inference
        all_matches = {}
        
        for kp_count, indices in kp_count_groups.items():
            if kp_count == 0:
                # Skip images with no keypoints
                for idx in indices:
                    all_matches[idx] = torch.empty((0, 2), dtype=torch.long, device=curr_kp_det_res.keypoints.device)
                continue
            if len(indices) == 1:
                # Single image - process individually
                idx = indices[0]
                ref_res = ref_kp_det_res[idx]
                
                # Create single-item batch
                ref_keypoints = ref_res.keypoints.unsqueeze(0)  # [1 x M x 2]
                ref_descriptors = ref_res.descriptors.unsqueeze(0)  # [1 x M x D]
                curr_keypoints = curr_kp_det_res.keypoints.unsqueeze(0)  # [1 x N x 2]
                curr_descriptors = curr_kp_det_res.descriptors.unsqueeze(0)  # [1 x N x D]
                
                data = {
                    "image0": {
                        "image_size": img_size,
                        "keypoints": ref_keypoints,
                        "descriptors": ref_descriptors,
                    },
                    "image1": {
                        "image_size": img_size,
                        "keypoints": curr_keypoints,
                        "descriptors": curr_descriptors,
                    }
                }
                
                group_matches = self.kp_match.match(data)
                all_matches[idx] = group_matches["matches"][0]  # Extract single result
                
            else:
                # Multiple images with same keypoint count - batch process
                group_ref_res = [ref_kp_det_res[i] for i in indices]
                group_batch_size = len(group_ref_res)
                
                # Stack keypoints and descriptors for this group
                ref_keypoints = torch.stack([res.keypoints for res in group_ref_res])  # [Group_B x M x 2]
                ref_descriptors = torch.stack([res.descriptors for res in group_ref_res])  # [Group_B x M x D]
                
                # Expand current keypoints/descriptors for this group
                curr_keypoints = curr_kp_det_res.keypoints.unsqueeze(0).repeat(group_batch_size, 1, 1)  # [Group_B x N x 2]
                curr_descriptors = curr_kp_det_res.descriptors.unsqueeze(0).repeat(group_batch_size, 1, 1)  # [Group_B x N x D]
                
                data = {
                    "image0": {
                        "image_size": img_size,
                        "keypoints": ref_keypoints,
                        "descriptors": ref_descriptors,
                    },
                    "image1": {
                        "image_size": img_size,
                        "keypoints": curr_keypoints,
                        "descriptors": curr_descriptors,
                    }
                }
                
                group_matches = self.kp_match.match(data)
                
                # Store results for each index in this group
                for group_idx, original_idx in enumerate(indices):
                    all_matches[original_idx] = group_matches["matches"][group_idx]
        
        # Reconstruct matches in original batch order
        matches = {"matches": [all_matches[i] for i in range(batch_size)]}
        
        return matches

    def match_keypoints_adaptive(self, ref_kp_det_res: List, curr_kp_det_res, img_size: Tuple[int, int]):
        """Adaptively match keypoints using batched inference when possible.
        Falls back to grouped matching when keypoint counts differ.
        
        Args:
            ref_kp_det_res: List of DetectionResult objects for reference images
            curr_kp_det_res: Single DetectionResult object for current image
            img_size: Image size (W, H)
            
        Returns:
            matches: Match results from kp_match.match()
        """
        # Check if all ref images have the same number of keypoints
        ref_kp_counts = [len(res.keypoints) for res in ref_kp_det_res]
        
        if len(set(ref_kp_counts)) == 1:
            # All have same number of keypoints - use efficient batched inference
            return self._match_keypoints_batched(ref_kp_det_res, curr_kp_det_res, img_size)
        else:
            # Different numbers of keypoints - use grouped processing
            logger.debug(f"Keypoint counts differ: {ref_kp_counts}. Using grouped matching.")
            return self._match_keypoints_grouped(ref_kp_det_res, curr_kp_det_res, img_size)

    @timeit
    @torch.inference_mode()
    def estimate_pose(
        self, 
        ref_image: torch.Tensor, 
        ref_depth: torch.Tensor,
        curr_image: torch.Tensor,
        curr_depth: torch.Tensor,
        retrieval_scores: List[float] = None,
        keyframes: List[Keyframe] = None,
    ):
        """Estimate relative pose between two images using PnP
        TODO: cache the keypoints and descriptors
        Args:
            ref_image: (B, 3, H, W)
            ref_depth: (B, H, W)
            curr_image: (3, H, W)
            curr_depth: (H, W)
            retrieval_scores: (B,) optional, for visualization only
            keyframes: (B,) optional, for visualization only
        Returns:
            valid_poses: SE3 (B, 7). The pose is T_ref_cam
            valid_covs: se3 (B, 6)
            valid_masks: (B,)
            confidences: (B,)
        """
        ref_image = ref_image.to(self.device)
        curr_image = curr_image.to(self.device)
        ref_depth = ref_depth.to(self.device)
        curr_depth = curr_depth.to(self.device)

        batch_size = ref_image.shape[0]

        # Batch detect: concatenate ref and curr images for single forward pass
        combined_images = torch.cat([curr_image.unsqueeze(0), ref_image], dim=0)  # [B+1, 3, H, W]
        all_kp_det_res = self.kp_det.detect(combined_images)  # Returns list of (B+1) DetectionResult
        ref_kp_det_res = all_kp_det_res[1:]  
        curr_kp_det_res = all_kp_det_res[0]  

        # Match keypoints in batch
        img_size = curr_image.shape[1:][::-1] # H, W
        
        matches = self.match_keypoints_adaptive(ref_kp_det_res, curr_kp_det_res, img_size)

        ref_points_3d = depth_to_xyz(ref_depth, self.camera) # [B x H x W x 3]
        valid_depth_mask = ref_depth < self.max_depth # max depth is 20m
        

        # Process matches and estimate pose for each batch
        poses = []
        inlier_counts = []
        match_counts = []
        valid_masks = []
        for b in range(batch_size):
                
            # Get matched keypoint indices (following the example pattern)
            match_indices = matches["matches"][b]  # [N_matches x 2]
            ref_matched_kps = ref_kp_det_res[b].keypoints[match_indices[:, 0]].long()  # [N_matches x 2]
            curr_matched_kps = curr_kp_det_res.keypoints[match_indices[:, 1]]  # [N_matches x 2]

            # Get the corresponding current keypoints that have valid reference depth
            ref_matched_kps_3d = ref_points_3d[b][ref_matched_kps[:, 1], ref_matched_kps[:, 0]]
            
            # Filter out 3D points that are too far away using valid_depth_mask
            ref_matched_depth_mask = valid_depth_mask[b][ref_matched_kps[:, 1], ref_matched_kps[:, 0]]
            ref_matched_kps_3d_filtered = ref_matched_kps_3d[ref_matched_depth_mask]
            curr_matched_kps_filtered = curr_matched_kps[ref_matched_depth_mask]

            if len(ref_matched_kps_3d_filtered) < self.config.kf_match_threshold:
                valid_masks.append(False)
                continue
            
            # Estimate pose using PnP RANSAC
            # Use filtered reference 3D points and corresponding current 2D points
            pose, inliers = pnp_ransac(
                ref_matched_kps_3d_filtered,
                curr_matched_kps_filtered,
                self.camera,
                iterations_count=100,
                reprojection_error=5.0,
                confidence=0.99
            )
            if pose is None or inliers < self.config.inlier_count_threshold:
                valid_masks.append(False)
                continue
            
            poses.append(pose)
            inlier_counts.append(inliers)
            match_counts.append(len(ref_matched_kps_3d_filtered))
            valid_masks.append(True)


        valid_masks = np.array(valid_masks)
        # compute variances
        confidences = torch.tensor([])
        if valid_masks.sum() > 0:
            # covs, var_scales = estimate_covariance_piecewise_interp(
            #     torch.tensor(inlier_counts),
            #     self.base_cov_diag,
            # )
            # TODO: use this or inlier count?
            # confidences = 1 / np.array(var_scales) 
            confidences = torch.tensor(inlier_counts)
            # covs = pp.se3(covs)
            poses = torch.stack(poses, dim=0)
        
        if self.visualize:
            # Reconstruct keypoints for visualization (needs consistent format)
            ref_keypoints_for_viz = []
            max_kp_count = max(len(res.keypoints) for res in ref_kp_det_res)
            
            for res in ref_kp_det_res:
                kps = res.keypoints
                if len(kps) < max_kp_count:
                    # Pad with zeros to match max count
                    padding = torch.zeros((max_kp_count - len(kps), 2), device=kps.device, dtype=kps.dtype)
                    kps = torch.cat([kps, padding], dim=0)
                ref_keypoints_for_viz.append(kps)
            
            ref_keypoints_viz = torch.stack(ref_keypoints_for_viz)
            curr_keypoints_viz = curr_kp_det_res.keypoints.unsqueeze(0).repeat(batch_size, 1, 1)
            
            self.viz_fig = visualize_pose_estimation_matches(
                ref_image=ref_image,
                curr_image=curr_image,
                ref_keypoints=ref_keypoints_viz,
                curr_keypoints=curr_keypoints_viz,
                matches=matches,
                retrieval_scores=retrieval_scores,
                keyframes=keyframes,
                est_poses_valid=poses,
                est_valid_masks=valid_masks,
                # est_covs_valid=covs,
                device=self.device,
                viz_fig=self.viz_fig,
                top_k=9
            )


        # # 3D visualization if requested
        # if visualize_3d:
        #     visualize_pose_3d(poses, ref_image, curr_image, scale=0.5)
            
        return poses, valid_masks, confidences
