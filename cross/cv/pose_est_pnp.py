import dataclasses
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pypose as pp
import torch
from loguru import logger

from cross.core.config import PoseEstConfig
from cross.core.types import Camera, Keyframe
from cross.cv.kp_det import get_kp_det
from cross.cv.kp_match import get_kp_match
from cross.utils.points import depth_to_xyz
from cross.utils.profile import timeit


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
        **context,
    ):
        """Estimate relative pose between two images using PnP
        (context: optional keyword arguments of the retrieval step, e.g. ref_loaded / session_unanchored; unused here)
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
        valid_depth_mask = (ref_depth > 0) & (ref_depth < self.max_depth)  # 0 = invalid (SGBM / GT holes), max depth is 20m


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
            confidences = torch.tensor(inlier_counts)
            poses = torch.stack(poses, dim=0)

        return poses, valid_masks, confidences
