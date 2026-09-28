import numpy as np
import rerun as rr
from typing import Optional, Dict, Any, List, Union

import torch
from cross.core.types import Keyframe, Camera
from cross.core.config import VisualizationConfig
from collections import defaultdict
import rerun.blueprint as rrb
import pypose as pp
from loguru import logger
from cross.core.hypothesis import HypothesisManager, Hypothesis

class RRViz:
    # Transformation from GT (X-Fwd, Y-Left, Z-Up) to OpenCV/Rerun (X-Right, Y-Down, Z-Fwd)
    # x_cv = -y_gt
    # y_cv = -z_gt
    # z_cv =  x_gt
    GT_TO_OPENCV_TRANSFORM = np.array([
        [ 0, -1,  0,  0],
        [ 0,  0, -1,  0],
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1]
    ], dtype=np.float64)

    def __init__(
        self,
        camera: Camera = None,
        recording_id: str = "CROSS_visualization",
        hypothesis_manager: HypothesisManager = None,
        visualize: bool = True,
        config: Union[VisualizationConfig, None] = None,
        **kwargs,
        ):
        """Initialize RRViz with rerun recording.

        Args:
            camera: Camera object with intrinsics and frame dimensions
            recording_id: Name for the rerun recording
            hypothesis_manager: Hypothesis manager containing keyframes to visualize
            config: VisualizationConfig (preferred). Legacy **kwargs still accepted.
        """
        self.recording_id = recording_id
        self.camera = camera
        self.hypothesis_manager: HypothesisManager = hypothesis_manager
        self.visualize = visualize

        ################ internal states ################
        self.step_idx = 0
        self.first_camera_pose_inv = None
        self.is_first_frame = True
        self.accumulated_points = []
        self.accumulated_colors = []
        self.gt_trajectory_segments = [[]]  # List of trajectory segments (for handling kidnapping)

        # Odometry trajectory tracking
        self.odom_accumulated = pp.identity_SE3()
        self.odom_trajectory_segments = [[]]  # List of odometry trajectory segments
        self.is_first_odom = True

        # Track keyframe IDs at kidnapping boundaries
        self.kidnapped_kf_ids = set()

        self.alive_pose_components = set()
        self.alive_proposal_components = set()
        self.alive_kf_components = defaultdict(set)

        ################ visualization settings ################
        vcfg = config or VisualizationConfig()
        # Support legacy kwargs override (e.g. from planner.py creating RRViz directly)
        self.visualize_pointcloud = kwargs.get('visualize_pointcloud', vcfg.visualize_pointcloud)
        self.visualize_camera = kwargs.get('visualize_camera', vcfg.visualize_camera)
        self.visualize_current_gmm_state = kwargs.get('visualize_current_gmm_state', vcfg.visualize_current_gmm_state)
        self.visualize_keyframe_gmms = kwargs.get('visualize_keyframe_gmms', vcfg.visualize_keyframe_gmms)
        self.visualize_system_data = kwargs.get('visualize_system_data', vcfg.visualize_system_data)
        self.visualize_hypotheses = kwargs.get('visualize_hypotheses', vcfg.visualize_hypotheses)
        self.visualize_trajectory = kwargs.get('visualize_trajectory', vcfg.visualize_trajectory)
        self.visualize_odom_trajectory = kwargs.get('visualize_odom_trajectory', vcfg.visualize_odom_trajectory)
        self.visualize_pinhole_camera = kwargs.get('visualize_pinhole_camera', vcfg.visualize_pinhole_camera)
        self.visualize_keyframe_gmm_trajectory = kwargs.get('visualize_keyframe_gmm_trajectory', vcfg.visualize_keyframe_gmm_trajectory)

        self.min_weight_for_gmm = 1e-5
        self.max_points_per_cloud = 10000  # For downsampling
        self.voxel_size = 0.2  # 

        self._init_rerun()


    def _init_rerun(self):
        # Initialize rerun
        if self.visualize:
            rr.init(self.recording_id, spawn=True)

            bg = rrb.Background(color=(255, 255, 255, 255))
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin="/world",
                        background=[255, 255, 255],
                        line_grid=rrb.archetypes.LineGrid3D(visible=False),
                    ),
                    rrb.Vertical(
                        rrb.Spatial2DView(
                            origin="/world/current_camera_gt",
                        ),
                        rrb.TextDocumentView(
                            origin="/hypothesis_lifecycle",
                        ),
                        row_shares=[1, 3],
                    ),

                    rrb.TextDocumentView(
                        origin="/system_data",
                    ),
                    column_shares=[3, 1, 1],
                ),
                rrb.BlueprintPanel(state="expanded"),
                rrb.SelectionPanel(state="collapsed"),
                rrb.TimePanel(state="collapsed"),
            )
            rr.send_blueprint(blueprint)

            # Set up world coordinate system to RDF (X-Right, Y-Down, Z-Forward)
            rr.log("world", rr.ViewCoordinates.RDF, static=True)

    def _voxel_downsample(self, points: np.ndarray, colors: np.ndarray, 
                        voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Downsample point cloud using voxel grid (one point per voxel).
        
        Args:
            points: (N, 3) array of 3D points
            colors: (N, 3) array of RGB colors
            voxel_size: Size of each voxel
            
        Returns:
            Downsampled points and colors
        """
        if len(points) == 0:
            return points, colors
            
        # Quantize points to voxel grid
        voxel_coords = np.floor(points / voxel_size).astype(np.int32)
        
        # Use dictionary to keep track of unique voxels
        voxel_dict = {}
        
        for i, (voxel_coord, point, color) in enumerate(zip(voxel_coords, points, colors)):
            voxel_key = tuple(voxel_coord)
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = (point, color)
        
        # Extract downsampled points and colors
        if len(voxel_dict) == 0:
            return np.empty((0, 3)), np.empty((0, 3))
            
        downsampled_points = np.array([item[0] for item in voxel_dict.values()])
        downsampled_colors = np.array([item[1] for item in voxel_dict.values()])
        
        return downsampled_points, downsampled_colors

    def _project_depth_to_pointcloud(self, depth: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project depth image to 3D point cloud using camera intrinsics.
        
        Args:
            depth: Depth image (H, W) in meters
            rgb: RGB image (H, W, 3)
            
        Returns:
            points: (N, 3) array of 3D points
            colors: (N, 3) array of RGB colors
        """
        if self.camera is None:
            raise ValueError("Camera intrinsics not set")
            
        h, w = depth.shape
        fx, fy = self.camera.fx, self.camera.fy
        cx, cy = self.camera.px, self.camera.py
        
        # Create pixel coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Filter out invalid depth values
        valid_mask = (depth > 0) & (depth < 10.0)  # Reasonable depth range
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth[valid_mask]
        colors_valid = rgb[valid_mask]
        
        # Project to 3D
        z = depth_valid
        x = (u_valid - cx) * z / fx
        y = (v_valid - cy) * z / fy
        
        points = np.stack([x, y, z], axis=1)

        # remove ceilling points
        mask = (points[:, 1] > -0.5) & (points[:, 1] < 1.5)  # -y is up
        points = points[mask]
        colors_valid = colors_valid[mask]
        
        return points, colors_valid

    def _process_gt_pose(self, world_pose_gt: np.ndarray) -> np.ndarray:
        """Process GT pose and apply coordinate transformations.
        
        Args:
            world_pose_gt: 4x4 transformation matrix in GT coordinates
            
        Returns:
            world_pose_transformed_gt: Transformed pose relative to first frame
        """
        # M = self.GT_TO_OPENCV_TRANSFORM  # Transforms points from GT to RDF
        # inv_M = np.linalg.inv(M)
        
        # Convert world_pose_gt to be T_World(RDF) <- Camera(RDF)
        # Original world_pose_gt is T_World(GT) <- Camera(GT)
        # current_pose_rdf_convention = M @ world_pose_gt @ inv_M
        
        if self.is_first_frame:
            # self.first_camera_pose_inv becomes T_Cam1(RDF) <- World(RDF_initial)
            self.first_camera_pose_inv = np.linalg.inv(world_pose_gt)
            # world_pose_transformed is T_Cam1(RDF) <- Cam1(RDF)
            world_pose_transformed_gt = np.eye(4, dtype=np.float64)
            self.is_first_frame = False
        else:
            # world_pose_transformed is (T_Cam1(RDF) <- World(RDF_initial)) @ (T_World(RDF_initial) <- CamN(RDF))
            # which simplifies to T_Cam1(RDF) <- CamN(RDF)
            world_pose_transformed_gt = self.first_camera_pose_inv @ world_pose_gt

        # Add keyframe position for trajectory (to current segment)
        current_position_gt = world_pose_transformed_gt[:3, 3]
        self.gt_trajectory_segments[-1].append(current_position_gt)
        return world_pose_transformed_gt

    def _process_pointcloud(self, is_kf: bool, depth: np.ndarray, rgb: np.ndarray, 
                          world_pose_transformed_gt: np.ndarray):
        """Process point cloud for keyframes and accumulate global map.
        
        Args:
            is_kf: Whether this step is a keyframe
            depth: Depth image
            rgb: RGB image
            world_pose_transformed_gt: Transformed pose
        """
        if not is_kf:
            return
            
        # Project depth to point cloud, only for keyframes
        points_cam, colors = self._project_depth_to_pointcloud(depth, rgb)
        
        # Transform points to world coordinates
        points_cam_homo = np.hstack([points_cam, np.ones((len(points_cam), 1))])
        points_world_homo = (world_pose_transformed_gt @ points_cam_homo.T).T
        points_world = points_world_homo[:, :3]
        
        # Downsample current frame points using voxel grid
        points_world, colors = self._voxel_downsample(
            points_world, colors, self.voxel_size)

        self.accumulated_points.append(points_world)
        self.accumulated_colors.append(colors)
        
        # Combine all accumulated points
        all_points = np.vstack(self.accumulated_points) if self.accumulated_points else np.empty((0, 3))
        all_colors = np.vstack(self.accumulated_colors) if self.accumulated_colors else np.empty((0, 3))
        
        # Global voxel downsampling if too many points
        if len(all_points) > self.max_points_per_cloud:
            all_points, all_colors = self._voxel_downsample(
                all_points, all_colors, self.voxel_size)  # Larger voxels for global map
        
        # Log accumulated point cloud
        rr.log("world/pointcloud", rr.Points3D(
            positions=all_points,
            colors=all_colors.astype(np.uint8),
            ),
            static=True,
        )
        
    def _log_trajectory(self):
        """Log GT trajectory with support for multiple segments (after kidnapping)."""
        # Collect all non-empty segments with at least 2 points
        valid_segments = [np.array(seg) for seg in self.gt_trajectory_segments if len(seg) > 1]

        if len(valid_segments) > 0:
            # Log all segments together
            rr.log("world/gt_trajectory", rr.LineStrips3D(
                strips=valid_segments,
                colors=[[0, 255, 0]] * len(valid_segments)  # Green trajectory for GT
            ))

    def _process_odom_trajectory(self, state_info: Dict[str, Any]):
        """Process odometry delta and accumulate trajectory.

        Accumulates odometry deltas similar to OdomAccumulator pattern,
        building a trajectory from relative motion measurements.

        Args:
            state_info: Dictionary containing delta_pose
        """
        if not state_info or "delta_pose" not in state_info:
            return

        delta_pose = state_info["delta_pose"]

        # Handle None delta_pose gracefully
        if delta_pose is None:
            return

        # Convert numpy array to pypose SE3 if needed
        if isinstance(delta_pose, np.ndarray):
            delta_pose = pp.from_matrix(delta_pose, pp.SE3_type).float()

        # Ensure delta_pose is a pypose tensor
        if not isinstance(delta_pose, pp.LieTensor):
            return

        # First odometry reading: initialize on the same device as delta_pose
        if self.is_first_odom:
            self.odom_accumulated = pp.identity_SE3().to(delta_pose.device)
            self.odom_trajectory_segments[-1].append(np.array([0.0, 0.0, 0.0]))
            self.is_first_odom = False

        # Accumulate: compose with previous accumulated pose
        self.odom_accumulated = self.odom_accumulated @ delta_pose

        # Extract position from accumulated pose
        # pypose SE3 tensor format: [x, y, z, qx, qy, qz, qw]
        pose_tensor = self.odom_accumulated.tensor()
        if pose_tensor.is_cuda:
            pose_tensor = pose_tensor.cpu()
        position = pose_tensor[:3].numpy()

        self.odom_trajectory_segments[-1].append(position)

    def _visualize_odom_trajectory(self):
        """Visualize accumulated odometry trajectory with support for multiple segments."""
        # Collect all non-empty segments with at least 2 points
        valid_segments = [np.array(seg) for seg in self.odom_trajectory_segments if len(seg) > 1]

        if len(valid_segments) > 0:
            rr.log("world/odom_trajectory", rr.LineStrips3D(
                strips=valid_segments,
                colors=[[0, 150, 255]] * len(valid_segments)  # Cyan/blue trajectory for odometry
            ))

    def _log_camera_poses(self, is_kf: bool, step_idx: int, rgb: np.ndarray, 
                         world_pose_transformed_gt: np.ndarray):
        """Log camera poses and images for current frame and keyframes.
        
        Args:
            is_kf: Whether this step is a keyframe
            step_idx: Current step index
            rgb: RGB image
            world_pose_transformed_gt: Transformed pose
        """
        # Log current GT camera pose transform
        rr.log("world/current_camera_gt", rr.Transform3D(
            translation=world_pose_transformed_gt[:3, 3],
            mat3x3=world_pose_transformed_gt[:3, :3]
        ))

        rr.log("world/current_camera_gt", rr.Image(rgb)) # GT view
        rr.log("world/current_camera_gt", rr.Pinhole( # Redundant if logged with keyframe
            resolution=[self.camera.frame_width, self.camera.frame_height],
            focal_length=[self.camera.fx, self.camera.fy],
            principal_point=[self.camera.px, self.camera.py],
            camera_xyz=rr.ViewCoordinates.RDF, 
        ))
        
        # Only log pinhole camera for keyframes and make them persistent
        if is_kf:
            keyframe_entity_gt = f"world/gt_keyframes/keyframe_{step_idx}"

            # Log current camera pose transform for GT
            rr.log(keyframe_entity_gt, rr.Transform3D(
                translation=world_pose_transformed_gt[:3, 3],
                mat3x3=world_pose_transformed_gt[:3, :3]
            ))
            
            # Log transform for keyframe camera GT
            rr.log(keyframe_entity_gt, rr.Image(rgb))
            if self.visualize_pinhole_camera:
                rr.log(keyframe_entity_gt, rr.Pinhole(
                    resolution=[self.camera.frame_width, self.camera.frame_height],
                    focal_length=[self.camera.fx, self.camera.fy],
                    principal_point=[self.camera.px, self.camera.py],
                    camera_xyz=rr.ViewCoordinates.RDF, 
                    ),
                )

    def _visualize_current_gmm_state(self, state_info: Dict[str, Any]):
        """Visualize current estimated GMM state.
        
        Args:
            state_info: Dictionary containing current GMM state
        """
        if not state_info or "current_mu" not in state_info:
            return
            
        current_mu_se3 = state_info["current_mu"] # pypose.SE3 (K, 7)
        current_sigma_se3 = state_info["current_sigma"] # pypose.se3 (K, 6)
        current_weights = state_info["current_weights"] # torch.Tensor (K,)

        # The estimated poses are T_world_origin <- T_camera_current
        # No need for GT_TO_OPENCV_TRANSFORM or first_camera_pose_inv
        # as these are already in the correct frame (first camera is origin, RDF)
        
        for i in range(current_mu_se3.shape[0]):
            gmm_component_entity = f"world/estimated_pose/component_{i}"

            if current_weights[i] < self.min_weight_for_gmm: # Skip components with negligible weight
                if i in self.alive_pose_components:
                    rr.log(gmm_component_entity, rr.Clear(recursive=True))
                    self.alive_pose_components.remove(i)
                continue

            if i not in self.alive_pose_components:
                self.alive_pose_components.add(i)

            
            # Extract translation and rotation for rr.Transform3D
            # pypose.SE3 stores as [x, y, z, qx, qy, qz, qw]
            # rr.Transform3D needs translation as (x,y,z) and rotation as mat3x3 or quaternion [x,y,z,w]
            pose_tensor = current_mu_se3[i].tensor() 
            translation = pose_tensor[:3].cpu().numpy()
            # pypose quat is [x,y,z,w], rerun also expects [x,y,z,w] for RotationQuat
            rotation_quat = pose_tensor[3:].cpu().numpy() 

            rr.log(
                gmm_component_entity,
                rr.Transform3D(
                    translation=translation,
                    rotation=rr.Quaternion(xyzw=rotation_quat) # Use xyzw
                )
            )

            # Visualize the mean pose as a set of axes
            # Scale axes by weight for emphasis
            axis_length = 0.1
            
            rr.log(
                f"{gmm_component_entity}/mean_axes",
                rr.Arrows3D(
                    origins=[[0,0,0],[0,0,0],[0,0,0]],
                    vectors=[[axis_length,0,0],[0,axis_length,0],[0,0,axis_length]],
                    colors=[[255,0,0],[0,255,0],[0,0,255]] # R,G,B for X,Y,Z
                )
            )

            # Visualize uncertainty (sigma)
            # Sigma is in se3 (tangent space), representing variance of [vx,vy,vz,wx,wy,wz]
            # We can visualize the translational part as a box/sphere.
            # Rotational uncertainty is harder to visualize directly as a simple shape.
            # For now, let's use the translational stddevs (sqrt of variance) for box sizes.
            sigma_vec = current_sigma_se3[i].tensor().cpu().numpy()
            # Clamp sigma values to avoid very small or negative (if any instability) values before sqrt
            clamped_sigma_vec = np.maximum(sigma_vec, 1e-6)

            # Scale the box size by a factor for visibility, can be adjusted
            box_half_sizes = np.clip(clamped_sigma_vec[:3], 0.05, 0.5) 
            
            rr.log(
                f"{gmm_component_entity}/uncertainty_translation",
                rr.Ellipsoids3D(
                    half_sizes=[box_half_sizes], # expects (N,3)
                    centers=[[0,0,0]], # Relative to the component's transform
                    colors=[[0,180,204, 120]], # Semi-transparent gray
                    fill_mode=3 # Solid
                )
            )

            

        if state_info and "proposal_mu" in state_info:
            proposal_mu = state_info["proposal_mu"]
            proposal_sigma = state_info["proposal_sigma"]
            proposal_weights = state_info["proposal_weights"].clone()

            # remove components with infinite sigma, which is "imaginary"
            infi_sigma = proposal_sigma.min(dim=-1).values > 1e3
            proposal_weights[infi_sigma] = 0.0

            for i in range(current_mu_se3.shape[0]):
                gmm_component_entity = f"world/proposal_pose/component_{i}"

                if proposal_weights[i] < self.min_weight_for_gmm: # Skip components with negligible weight
                    if i in self.alive_proposal_components:
                        rr.log(gmm_component_entity, rr.Clear(recursive=True))
                        self.alive_proposal_components.remove(i)
                    continue

                if i not in self.alive_proposal_components:
                    self.alive_proposal_components.add(i)

                
                # Extract translation and rotation for rr.Transform3D
                # pypose.SE3 stores as [x, y, z, qx, qy, qz, qw]
                # rr.Transform3D needs translation as (x,y,z) and rotation as mat3x3 or quaternion [x,y,z,w]
                pose_tensor = proposal_mu[i].tensor() 
                translation = pose_tensor[:3].cpu().numpy()
                # pypose quat is [x,y,z,w], rerun also expects [x,y,z,w] for RotationQuat
                rotation_quat = pose_tensor[3:].cpu().numpy() 

                rr.log(
                    gmm_component_entity,
                    rr.Transform3D(
                        translation=translation,
                        rotation=rr.Quaternion(xyzw=rotation_quat) # Use xyzw
                    )
                )

                # Visualize the mean pose as a set of axes
                # Scale axes by weight for emphasis
                axis_length = 0.1 
                
                rr.log(
                    f"{gmm_component_entity}/mean_axes",
                    rr.Arrows3D(
                        origins=[[0,0,0],[0,0,0],[0,0,0]],
                        vectors=[[axis_length,0,0],[0,axis_length,0],[0,0,axis_length]],
                        colors=[[255,0,0],[0,255,0],[0,0,255]] # R,G,B for X,Y,Z
                    )
                )

                # Visualize uncertainty (sigma)
                # Sigma is in se3 (tangent space), representing variance of [vx,vy,vz,wx,wy,wz]
                # We can visualize the translational part as a box/sphere.
                # Rotational uncertainty is harder to visualize directly as a simple shape.
                # For now, let's use the translational stddevs (sqrt of variance) for box sizes.
                sigma_vec = proposal_sigma[i].tensor().cpu().numpy()
                # Clamp sigma values to avoid very small or negative (if any instability) values before sqrt
                clamped_sigma_vec = np.maximum(sigma_vec, 1e-6)
                # Scale the box size by a factor for visibility, can be adjusted
                box_half_sizes = np.clip(clamped_sigma_vec[:3]*2, 0.01, 0.5) 
                
                rr.log(
                    f"{gmm_component_entity}/uncertainty_translation",
                    rr.Ellipsoids3D(
                        half_sizes=[box_half_sizes], # expects (N,3)
                        centers=[[0,0,0]], # Relative to the component's transform
                        colors=[[102, 255, 102, 120]], # Semi-transparent green
                        fill_mode=3 # Solid
                    )
                )


           

    def _log_system_data(self, is_kf: bool, state_info: Dict[str, Any]):
        """Log system data to rerun."""
        if not state_info:
            return
            
        markdown_content = self._generate_system_data_markdown(is_kf, state_info)
        
        # Add hypotheses information if available
        hypotheses = state_info.get("hypotheses", [])
        if hypotheses:
            hypotheses_content = self._generate_hypotheses_markdown(hypotheses)
            markdown_content += "\n\n" + hypotheses_content
        
        # Log the markdown document
        rr.log("/system_data", rr.TextDocument(markdown_content, media_type=rr.MediaType.MARKDOWN))

    def _generate_system_data_markdown(self, is_kf: bool, state_info: Dict[str, Any]) -> str:
        """Generate markdown content for system data visualization."""
        lines = []
        lines.append("# System Data Summary\n")
        
        # Extract data from state_info
        keyframe_timestamps = [kf.timestamp for kf in state_info.get("retrieved_keyframes", [])]
        valid_keyframe_timestamps = [kf.timestamp for kf in state_info.get("valid_keyframes", [])]
        retrieval_scores = state_info.get("retrieval_scores", [])
        valid_masks = state_info.get("valid_masks", [])
        valid_retrieval_weights = state_info.get("valid_retrieval_weights", [])
        confidences = state_info.get("pose_est_conf", [])
        
        # Check if we have pose estimation data
        # Need to verify: key exists, has data (len > 0), and all required keys are present
        has_pose_data = (
            len(state_info.get("valid_ref_mus", [])) > 0 and
            "valid_ref_stds" in state_info and
            "valid_ref_component_weights" in state_info
        )
        
        if has_pose_data:
            valid_ref_mus = state_info["valid_ref_mus"]  # (B, K, 7)
            valid_ref_sigmas = state_info["valid_ref_stds"]  # (B, K, 6)
            valid_ref_component_weights = state_info["valid_ref_component_weights"]  # (B, K)
            convolved_mus = state_info.get("convolved_mus", None)  # (B, K, 7)
            convolved_sigmas = state_info.get("convolved_stds", None)  # (B, K, 6)
            valid_poses = state_info.get("valid_poses", None)  # (B, 7)
            valid_stds = state_info.get("valid_stds", None)  # (B, 6)

        # proposal GMM
        proposal_mu = state_info.get("proposal_mu", None)
        proposal_sigma = state_info.get("proposal_sigma", None) 
        proposal_weights = state_info.get("proposal_weights", None)
        
        # Current State Estimate GMM
        if proposal_mu is not None and proposal_weights is not None:
            lines.append("## Proposal GMM\n")
            lines.append("| Component | Weight | Mean (x,y,z) | Mean (qx,qy,qz,qw) | Variance (vx,vy,vz,wx,wy,wz) |\n")
            lines.append("|-----------|--------|--------------|-------------------|-----------------------------|\n")
            
            for comp_idx in range(proposal_mu.shape[0]):
                weight = proposal_weights[comp_idx].item()
                
                # Skip components with negligible weight
                if weight < self.min_weight_for_gmm:
                    continue
                    
                pose_tensor = proposal_mu[comp_idx].tensor().cpu().numpy()
                trans = pose_tensor[:3]
                rot = pose_tensor[3:]  # qx,qy,qz,qw
                
                if proposal_sigma is not None:
                    sigma_tensor = proposal_sigma[comp_idx].tensor().cpu().numpy()
                    sigma_str = (f"({sigma_tensor[0]:.2f},{sigma_tensor[1]:.2f},{sigma_tensor[2]:.2f},"
                               f"{sigma_tensor[3]:.2f},{sigma_tensor[4]:.2f},{sigma_tensor[5]:.2f})")
                else:
                    sigma_str = "N/A"
                
                lines.append(f"| {comp_idx} | {weight:.2f} | "
                           f"({trans[0]:.2f},{trans[1]:.2f},{trans[2]:.2f}) | "
                           f"({rot[0]:.2f},{rot[1]:.2f},{rot[2]:.2f},{rot[3]:.2f}) | "
                           f"{sigma_str} |\n")
            
            lines.append("\n")           
        # Current state GMM
        current_mu = state_info.get("current_mu", None)
        current_sigma = state_info.get("current_sigma", None) 
        current_weights = state_info.get("current_weights", None)
        
        # Current State Estimate GMM
        if current_mu is not None and current_weights is not None:
            lines.append("## Current State Estimate GMM\n")
            lines.append("| Component | Weight | Mean (x,y,z) | Mean (qx,qy,qz,qw) | Variance (vx,vy,vz,wx,wy,wz) |\n")
            lines.append("|-----------|--------|--------------|-------------------|-----------------------------|\n")
            
            for comp_idx in range(current_mu.shape[0]):
                weight = current_weights[comp_idx].item()
                
                # Skip components with negligible weight
                if weight < self.min_weight_for_gmm:
                    continue
                    
                pose_tensor = current_mu[comp_idx].tensor().cpu().numpy()
                trans = pose_tensor[:3]
                rot = pose_tensor[3:]  # qx,qy,qz,qw
                
               
                
                if current_sigma is not None:
                    sigma_tensor = current_sigma[comp_idx].tensor().cpu().numpy()
                    sigma_str = (f"({sigma_tensor[0]:.2f},{sigma_tensor[1]:.2f},{sigma_tensor[2]:.2f},"
                               f"{sigma_tensor[3]:.2f},{sigma_tensor[4]:.2f},{sigma_tensor[5]:.2f})")
                else:
                    sigma_str = "N/A"
                
                lines.append(f"| {comp_idx} | {weight:.2f} | "
                           f"({trans[0]:.2f},{trans[1]:.2f},{trans[2]:.2f}) | "
                           f"({rot[0]:.2f},{rot[1]:.2f},{rot[2]:.2f},{rot[3]:.2f}) | "
                           f"{sigma_str} |\n")
            
            lines.append("\n")

        # Retrieved Keyframes Table
        lines.append("## Retrieved Keyframes\n")
        lines.append("| KF Timestamp | Retrieval Score | Valid |  Pose_est_conf| Rel Pose (x,y,z) | Rel Pose (qx,qy,qz,qw) | Rel Pose Variance |\n")
        lines.append("|-------|------------|-------------|-----------|------------------|----------------------|------------------|\n")
        
        valid_kf_idx = 0
        for i, (score, is_valid) in enumerate(zip(retrieval_scores, valid_masks)):
            valid_str = "✓" if is_valid else "✗"
            
            # Extract relative pose information if available
            if is_valid and valid_poses is not None:
                pose_tensor = valid_poses[valid_kf_idx].tensor().cpu().numpy()
                rel_pose_trans = f"({pose_tensor[0]:.2f},{pose_tensor[1]:.2f},{pose_tensor[2]:.2f})"
                rel_pose_rot = f"({pose_tensor[3]:.2f},{pose_tensor[4]:.2f},{pose_tensor[5]:.2f},{pose_tensor[6]:.2f})"
                
                if valid_stds is not None:
                    cov_tensor = valid_stds[valid_kf_idx].tensor().cpu().numpy()
                    rel_pose_var = f"({cov_tensor[0]:.2f},{cov_tensor[1]:.2f},{cov_tensor[2]:.2f})"
                else:
                    rel_pose_var = "N/A"
                confidence = f"{confidences[valid_kf_idx]:.2f}"
                valid_kf_idx += 1
            else:
                rel_pose_trans = "N/A"
                rel_pose_rot = "N/A"
                rel_pose_var = "N/A"
                confidence = "N/A"
            
            lines.append(f"| {keyframe_timestamps[i]} | {score:.2f} | {valid_str} | {confidence} | {rel_pose_trans} | {rel_pose_rot} | {rel_pose_var} |\n")
        
        lines.append("\n")
        
        # 2. Valid Keyframe GMMs
        if has_pose_data and len(valid_ref_mus) > 0:
            lines.append("## Valid Keyframe GMMs\n")
            
            valid_kf_idx = 0
            for i, is_valid in enumerate(valid_masks):
                if not is_valid:
                    continue
                    
                lines.append(f"##### Keyframe {valid_keyframe_timestamps[valid_kf_idx]} GMM Components\n")
                lines.append("| Component | Weight | Mean (x,y,z) | Mean (qx,qy,qz,qw) | Variance (vx,vy,vz,wx,wy,wz) | Convolved Mean (x,y,z) | Convolved Mean (qx,qy,qz,qw) | Convolved Variance |\n")
                lines.append("|-----------|--------|--------------|-------------------|-----------------------------|-----------------------|-----------------------------|-----------------|\n")
                
                # Get GMM components for this keyframe
                kf_mus = valid_ref_mus[valid_kf_idx]  # (K, 7)
                kf_sigmas = valid_ref_sigmas[valid_kf_idx]  # (K, 6)
                kf_weights = valid_ref_component_weights[valid_kf_idx]  # (K,)
                
                # Get convolved data if available
                conv_mus = None
                conv_sigmas = None
                if convolved_mus is not None:
                    conv_mus = convolved_mus[valid_kf_idx]  # (K, 7)
                if convolved_sigmas is not None:
                    conv_sigmas = convolved_sigmas[valid_kf_idx]  # (K, 6)
                
                for comp_idx in range(kf_mus.shape[0]):
                    weight = kf_weights[comp_idx].item()
                    
                    # Skip components with negligible weight
                    if weight < self.min_weight_for_gmm:
                        continue
                        
                    pose_tensor = kf_mus[comp_idx].tensor().cpu().numpy()
                    trans = pose_tensor[:3]
                    rot = pose_tensor[3:]  # qx,qy,qz,qw
                    
                    sigma_tensor = kf_sigmas[comp_idx].tensor().cpu().numpy()
                    
                    # Convolved data
                    if conv_mus is not None:
                        conv_pose_tensor = conv_mus[comp_idx].tensor().cpu().numpy()
                        conv_trans = f"({conv_pose_tensor[0]:.2f},{conv_pose_tensor[1]:.2f},{conv_pose_tensor[2]:.2f})"
                        conv_rot = f"({conv_pose_tensor[3]:.2f},{conv_pose_tensor[4]:.2f},{conv_pose_tensor[5]:.2f},{conv_pose_tensor[6]:.2f})"
                    else:
                        conv_trans = "N/A"
                        conv_rot = "N/A"
                    
                    if conv_sigmas is not None:
                        conv_sigma_tensor = conv_sigmas[comp_idx].tensor().cpu().numpy()
                        conv_var = f"({conv_sigma_tensor[0]:.2f},{conv_sigma_tensor[1]:.2f},{conv_sigma_tensor[2]:.2f})"
                    else:
                        conv_var = "N/A"
                    
                    lines.append(f"| {comp_idx} | {weight:.2f} | "
                               f"({trans[0]:.2f},{trans[1]:.2f},{trans[2]:.2f}) | "
                               f"({rot[0]:.2f},{rot[1]:.2f},{rot[2]:.2f},{rot[3]:.2f}) | "
                               f"({sigma_tensor[0]:.2f},{sigma_tensor[1]:.2f},{sigma_tensor[2]:.2f},"
                               f"{sigma_tensor[3]:.2f},{sigma_tensor[4]:.2f},{sigma_tensor[5]:.2f}) | "
                               f"{conv_trans} | {conv_rot} | {conv_var} |\n")
                
                lines.append("\n")
                valid_kf_idx += 1
        
        
        # Additional Statistics
        lines.append("## Statistics\n")
        lines.append(f"- **Is Keyframe**: {is_kf}\n")
        
        # Count total, temporary, and permanent keyframes
        all_keyframes = self.hypothesis_manager.nodes.values()
        total_num_kf = len(all_keyframes)
        temp_kf_count = sum(1 for kf in all_keyframes if kf.temporary)
        permanent_kf_count = total_num_kf - temp_kf_count
        
        lines.append(f"- **Total Number of Keyframes**: {total_num_kf}\n")
        lines.append(f"- **Permanent Keyframes**: {permanent_kf_count}\n")
        lines.append(f"- **Temporary Keyframes**: {temp_kf_count}\n")
        
        if valid_retrieval_weights is not None and len(valid_retrieval_weights) > 0:
            lines.append(f"- **Valid Retrievals**: {len(valid_retrieval_weights)}\n")
            lines.append(f"- **Max Retrieval Score**: {max(valid_retrieval_weights).item():.2f}\n")
            lines.append(f"- **Mean Retrieval Score**: {valid_retrieval_weights.mean().item():.2f}\n")
        
        # if len(match_counts) > 0:
        #     valid_match_counts = [mc for mc, vm in zip(match_counts, valid_masks) if vm]
        #     if valid_match_counts:
        #         lines.append(f"- **Max Matches**: {max(valid_match_counts)}\n")
        #         lines.append(f"- **Mean Matches**: {sum(valid_match_counts)/len(valid_match_counts):.2f}\n")
        
        return "".join(lines)


    def _visualize_keyframe_gmm_trajectory(self):
        """Visualize trajectory connecting first components of keyframe GMMs with kidnapping segmentation."""
        if self.hypothesis_manager is None:
            return

        # Collect positions and segment at kidnapping boundaries
        segments = []
        current_segment = []

        # Get all keyframes sorted by their ID to maintain temporal order
        all_keyframes = sorted(self.hypothesis_manager.nodes.items(), key=lambda x: x[0])

        for kf_id, keyframe in all_keyframes:
            # Skip temporary keyframes
            if keyframe.temporary:
                continue

            # Skip if no pose data
            if keyframe.pose_mu is None or keyframe.pose_weights is None:
                continue

            # Skip if first component has negligible weight
            if keyframe.pose_weights[0] < self.min_weight_for_gmm:
                continue

            # Check if this is a kidnapping boundary
            if kf_id in self.kidnapped_kf_ids and len(current_segment) > 0:
                # Save current segment and start a new one
                segments.append(np.array(current_segment))
                current_segment = []

            # Extract position from first component (index 0)
            pose_tensor = keyframe.pose_mu[0].tensor()  # [x, y, z, qx, qy, qz, qw]
            position = pose_tensor[:3].cpu().numpy()
            current_segment.append(position)

        # Add the last segment if it has points
        if len(current_segment) > 0:
            segments.append(np.array(current_segment))

        # Log all segments with at least 2 points
        valid_segments = [seg for seg in segments if len(seg) >= 2]
        if len(valid_segments) > 0:
            rr.log("world/keyframe_gmm_trajectory", rr.LineStrips3D(
                strips=valid_segments,
                colors=[[255, 165, 0]] * len(valid_segments)  # Orange trajectory for keyframe GMM estimates
            ))

    def _visualize_keyframe_gmms(self):
        """Visualize GMM pose distributions for all keyframes in the hypothesis manager."""
        if self.hypothesis_manager is None:
            return

        # Get all keyframes from the hypothesis manager
        all_keyframes = self.hypothesis_manager.nodes.values()

        for kf_idx, keyframe in enumerate(all_keyframes):
            if keyframe.temporary:
                continue
                
            if keyframe.pose_mu is None or keyframe.pose_weights is None:
                continue
                
            # Keyframe GMM components
            pose_mu_se3 = keyframe.pose_mu  # pypose.SE3 (K, 7)
            pose_sigma_se3 = keyframe.pose_std  # pypose.se3 (K, 6) 
            pose_weights = keyframe.pose_weights  # torch.Tensor (K,)
            
            for comp_idx in range(pose_mu_se3.shape[0]):
                kf_component_entity = f"world/keyframe_gmms/kf_{kf_idx}/component_{comp_idx}"

                if pose_weights[comp_idx] < self.min_weight_for_gmm:  # Skip negligible components
                    if comp_idx in self.alive_kf_components[kf_idx]:
                        rr.log(kf_component_entity, rr.Clear(recursive=True))
                        self.alive_kf_components[kf_idx].remove(comp_idx)
                    continue
                    
                if comp_idx not in self.alive_kf_components[kf_idx]:
                    self.alive_kf_components[kf_idx].add(comp_idx)

                # Extract pose information
                pose_tensor = pose_mu_se3[comp_idx].tensor()
                translation = pose_tensor[:3].cpu().numpy()
                rotation_quat = pose_tensor[3:].cpu().numpy()
                
                # Log the transform
                rr.log(
                    kf_component_entity,
                    rr.Transform3D(
                        translation=translation,
                        rotation=rr.Quaternion(xyzw=rotation_quat)
                    )
                )
                
                # Visualize mean pose with axes scaled by weight
                # Use different base size for keyframe vs current estimate
                base_axis_length = 0.05  # Smaller for keyframes to reduce clutter
                weight_scale = pose_weights[comp_idx].item()
                axis_length = base_axis_length + weight_scale * 0.2  # Scale between 0.05 and 0.25
                
                rr.log(
                    f"{kf_component_entity}/axes",
                    rr.Arrows3D(
                        origins=[[0,0,0],[0,0,0],[0,0,0]],
                        vectors=[[axis_length,0,0],[0,axis_length,0],[0,0,axis_length]],
                        colors=[[200,100,100],[100,200,100],[100,100,200]]  # Muted colors for keyframes
                    )
                )
                
                # Visualize uncertainty if available
                if pose_sigma_se3 is not None:
                    sigma_vec = pose_sigma_se3[comp_idx].tensor().cpu().numpy()
                    clamped_sigma_vec = np.maximum(sigma_vec, 1e-6)
                    std_devs = np.sqrt(clamped_sigma_vec[:3])  # Translational uncertainties
                    
                    # Smaller uncertainty boxes for keyframes
                    box_half_sizes = np.clip(std_devs * 0.05, 0.02, 0.5)
                    
                    rr.log(
                        f"{kf_component_entity}/uncertainty",
                        rr.Boxes3D(
                            half_sizes=[box_half_sizes],
                            centers=[[0,0,0]],
                            colors=[[180,10,10 ]]  # More transparent for keyframes
                        )
                    )


    def set_hypothesis_manager(self, hypothesis_manager: HypothesisManager):
        """Update the hypothesis manager reference.

        Useful when reusing visualizer across multiple System instances.

        Args:
            hypothesis_manager: New hypothesis manager to track
        """
        self.hypothesis_manager = hypothesis_manager

    def reset(self, new_session: bool = False):
        """Reset the visualization for a new session or kidnapping event.

        Args:
            new_session: If True, clears ALL data and resets step counter.
                        If False (kidnapping), starts new trajectory segments.
        """
        if new_session:
            # Full reset (e.g., loading a new map)
            self.step_idx = 0
            self.gt_trajectory_segments = [[]]
            self.odom_trajectory_segments = [[]]
            self.kidnapped_kf_ids = set()
            self.accumulated_points = []
            self.accumulated_colors = []
        else:
            # Kidnapping reset - segment trajectories
            if len(self.gt_trajectory_segments[-1]) > 0:
                self.gt_trajectory_segments.append([])
            if len(self.odom_trajectory_segments[-1]) > 0:
                self.odom_trajectory_segments.append([])

        # Always reset coordinate frame references
        self.first_camera_pose_inv = None
        self.is_first_frame = True
        self.odom_accumulated = pp.identity_SE3()
        self.is_first_odom = True

        # self._init_rerun()

    def visualize_tracking_step(
        self,
        kf: Keyframe,
        gt_info: Dict[str, Any],
        state_info: Dict[str, Any],
        step_idx: int = None,
        **kwargs,
    ):
        """Process one step of visualization data.

        Args:
            kf: Keyframe
            step_idx: (Deprecated) Ignored - visualizer maintains its own step counter
            gt_info: Dictionary containing rgb, depth, world_pose, etc.
            state_info: Dictionary containing current GMM state
        """
        _ = step_idx  # Suppress unused parameter warning (kept for backward compatibility)
        if not self.visualize or gt_info is None:
            return

        # Use visualizer's own step counter (increments across System instances)
        rr.set_time("step", sequence=self.step_idx)
        self.step_idx += 1

        is_kf = kf is not None and not kf.temporary

        # Track kidnapping boundaries: if this is a permanent keyframe AND we just reset
        # the coordinate frame (is_first_frame was True before processing), mark it
        is_kidnapped_frame = is_kf and self.is_first_frame

        if self.visualize_system_data:
            self._log_system_data(is_kf, state_info)
        
        # Extract GT data
        rgb = gt_info["rgb"]
        depth = gt_info["depth"] 
        world_pose_gt = gt_info["world_pose"]  # 4x4 transformation matrix in GT coordinates

        # Process GT pose transformations
        world_pose_transformed_gt = self._process_gt_pose(world_pose_gt)        # Add keyframe position for trajectory
        
        # Process point cloud for keyframes
        if self.visualize_pointcloud:
            self._process_pointcloud(is_kf, depth, rgb, world_pose_transformed_gt)
        
        # Log trajectory
        if self.visualize_trajectory:
            self._log_trajectory()

        # Process and visualize odometry trajectory
        if self.visualize_odom_trajectory:
            self._process_odom_trajectory(state_info)
            self._visualize_odom_trajectory()

        # Log camera poses and images
        if self.visualize_camera:
            self._log_camera_poses(is_kf, step_idx, rgb, world_pose_transformed_gt)
        
        # Visualize current estimated GMM state
        if self.visualize_current_gmm_state:
            self._visualize_current_gmm_state(state_info)
        
        # Visualize all keyframe GMM distributions (called periodically or on keyframes)
        if is_kf and self.visualize_keyframe_gmms:  # Update keyframe visualizations when we have a new keyframe
            self._visualize_keyframe_gmms()

        if is_kf and self.visualize_keyframe_gmm_trajectory:  # Update keyframe visualizations when we have a new keyframe
            self._visualize_keyframe_gmm_trajectory()

        if self.visualize_hypotheses:
            self._visualize_hypo_dict(state_info)
        
        # Mark kidnapping boundary after all processing
        if is_kidnapped_frame:
            self.kidnapped_kf_ids.add(kf.id)


    def _visualize_hypo_dict(self, state_info: Dict[str, Any]):
        """Visualize hypotheses lifecycle and evidence metrics."""
        if not state_info:
            return

        # Generate comprehensive lifecycle visualization
        lifecycle_content = self._generate_hypothesis_lifecycle_markdown()
        rr.log("/hypothesis_lifecycle", rr.TextDocument(lifecycle_content, media_type=rr.MediaType.MARKDOWN))



    def _determine_component_stage(self, comp_idx: int) -> tuple[str, str]:
        """Determine the lifecycle stage of a component.

        Returns:
            (emoji, stage_name) tuple
        """
        if not hasattr(self.hypothesis_manager, 'realized'):
            return "❓", "Unknown"

        realized = bool(self.hypothesis_manager.realized[comp_idx].item()) if comp_idx < self.hypothesis_manager.realized.numel() else False
        ttl = int(self.hypothesis_manager.ttl[comp_idx].item()) if comp_idx < self.hypothesis_manager.ttl.numel() else 0
        weight = float(self.hypothesis_manager.dist[2][comp_idx].item()) if self.hypothesis_manager.dist is not None else 0.0

        # Determine stage
        if not realized and ttl == 0 and weight < 1e-6:
            return "⚪", "Free"
        elif not realized and ttl > 0:
            return "🟡", "Tracking (Pending)"
        elif realized:
            return "🟢", "Tracking (Realized)"
        else:
            return "❓", "Unknown"

    def _create_text_progress_bar(self, value: float, max_value: float, width: int = 10,
                                   threshold: float = None) -> str:
        """Create ASCII progress bar with optional threshold indicator.

        Args:
            value: Current value
            max_value: Maximum value for the bar
            width: Width in characters
            threshold: Optional threshold to mark on the bar
        """
        if max_value <= 0:
            return "░" * width

        filled = int((value / max_value) * width)
        filled = max(0, min(width, filled))

        # Create bar
        bar = "█" * filled + "░" * (width - filled)

        # Add threshold marker if provided
        if threshold is not None and max_value > 0:
            threshold_pos = int((threshold / max_value) * width)
            threshold_pos = max(0, min(width - 1, threshold_pos))
            bar_list = list(bar)
            if threshold_pos < len(bar_list):
                bar_list[threshold_pos] = "│"
            bar = "".join(bar_list)

        return bar

    def _create_llr_heatmap_row(self, llr_values: torch.Tensor) -> str:
        """Create block character heatmap for LLR history.

        Args:
            llr_values: Tensor of LLR values (window length)
        """
        # Block chars from light to dark
        blocks = [" ", "░", "▒", "▓", "█"]

        # Normalize to 0-4 range for block selection
        max_val = max(llr_values.max().item(), 0.1)  # Avoid division by zero
        normalized = (llr_values / max_val) * 4
        normalized = torch.clamp(normalized, 0, 4).long()

        heatmap = "".join([blocks[int(val.item())] for val in normalized])
        return heatmap

    def _generate_hypothesis_lifecycle_markdown(self) -> str:
        """Generate comprehensive markdown for hypothesis lifecycle visualization."""
        lines = []
        lines.append("# Hypothesis Lifecycle Status\n\n")

        hm = self.hypothesis_manager
        if hm is None or hm.dist is None:
            lines.append("*No hypothesis manager available*\n")
            return "".join(lines)

        n_components = hm.n_components

        # Section A: Component Lifecycle Status Table
        lines.append("## 🔄 Component Status\n\n")
        lines.append("| Comp | Stage | Weight | TTL | Sum Pos | Hit Rate | Realized | Newborn |\n")
        lines.append("|------|-------|--------|-----|---------|----------|----------|----------|\n")

        for comp_idx in range(n_components):
            emoji, stage_name = self._determine_component_stage(comp_idx)

            weight = float(hm.dist[2][comp_idx].item())
            ttl = int(hm.ttl[comp_idx].item()) if comp_idx < hm.ttl.numel() else 0
            sum_pos = float(hm.last_sum_pos[comp_idx].item()) if hasattr(hm, 'last_sum_pos') and comp_idx < hm.last_sum_pos.numel() else 0.0
            hit_rate = float(hm.last_hit_rate[comp_idx].item()) if hasattr(hm, 'last_hit_rate') and comp_idx < hm.last_hit_rate.numel() else 0.0
            realized = "✓" if (comp_idx < hm.realized.numel() and bool(hm.realized[comp_idx].item())) else "✗"
            newborn = "✓" if (comp_idx < hm.newborn.numel() and bool(hm.newborn[comp_idx].item())) else "✗"

            lines.append(f"| {comp_idx} | {emoji} {stage_name} | {weight:.4f} | {ttl} | {sum_pos:.2f} | {hit_rate:.2f} | {realized} | {newborn} |\n")

        lines.append("\n")

        # Show thresholds
        if hasattr(hm, 'realize_sum_thresh') and hasattr(hm, 'realize_hitrate_thresh'):
            lines.append(f"*Thresholds: Realize ({hm.realize_sum_thresh:.1f}/{hm.realize_hitrate_thresh:.1f}), " +
                        f"Detect Overlap ({hm.detect_overlap_sum_thresh:.1f}/{hm.detect_overlap_hitrate_thresh:.1f}), " +
                        f"Detect Conf ({hm.detect_conf_hitrate_thresh:.1f}), " +
                        f"TTL Band ({hm.ttl_sum_thresh:.2f}/{hm.ttl_hitrate_thresh:.2f})*\n\n")

        # Section B: LLR History Heatmap
        if hasattr(hm, 'llr_hist'):
            lines.append("## 🔥 LLR History Heatmap\n\n")
            lines.append("*Recent evidence (oldest → newest)*\n\n")

            # Show window size
            window_size = hm.llr_hist_length
            ptr = hm.llr_hist_ptr

            lines.append("| Comp | " + " | ".join([f"t-{i}" for i in range(window_size-1, -1, -1)]) + " | Heatmap |\n")
            lines.append("|------|" + "|".join(["-----" for _ in range(window_size)]) + "|----------|\n")

            for comp_idx in range(n_components):
                # Get history for this component in chronological order (oldest first)
                hist = hm.llr_hist[comp_idx]
                # Reorder from circular buffer: from ptr to end, then from 0 to ptr
                ordered_hist = torch.cat([hist[ptr:], hist[:ptr]])

                # Show numeric values
                hist_str = " | ".join([f"{val:.2f}" for val in ordered_hist])

                # Create heatmap
                heatmap = self._create_llr_heatmap_row(ordered_hist)

                lines.append(f"| {comp_idx} | {hist_str} | `{heatmap}` |\n")

            lines.append("\n")

        # Section B2: Log-C History Heatmap
        if hasattr(hm, 'log_c_hist'):
            lines.append("## 🔥 Log-C History Heatmap\n\n")
            lines.append("*Recent log-likelihood differences (log_c - log_c[0]): oldest → newest*\n\n")

            window_size = hm.llr_hist_length
            ptr = hm.llr_hist_ptr

            lines.append("| Comp | " + " | ".join([f"t-{i}" for i in range(window_size-1, -1, -1)]) + " | Heatmap |\n")
            lines.append("|------|" + "|".join(["-----" for _ in range(window_size)]) + "|----------|\n")

            for comp_idx in range(n_components):
                hist = hm.log_c_hist[comp_idx]
                # Reorder from circular buffer
                ordered_hist = torch.cat([hist[ptr:], hist[:ptr]])

                # Show numeric values
                hist_str = " | ".join([f"{val:.2f}" for val in ordered_hist])

                # Create heatmap (using absolute values for visualization)
                heatmap = self._create_llr_heatmap_row(torch.abs(ordered_hist))

                lines.append(f"| {comp_idx} | {hist_str} | `{heatmap}` |\n")

            lines.append("\n")

        # Section B3: Log-Conf History Heatmap
        if hasattr(hm, 'log_conf_hist'):
            lines.append("## 🔥 Log-Conf History Heatmap\n\n")
            lines.append("*Recent log confidence ratios (log(conf/conf[0])): oldest → newest*\n\n")

            window_size = hm.llr_hist_length
            ptr = hm.llr_hist_ptr

            lines.append("| Comp | " + " | ".join([f"t-{i}" for i in range(window_size-1, -1, -1)]) + " | Heatmap |\n")
            lines.append("|------|" + "|".join(["-----" for _ in range(window_size)]) + "|----------|\n")

            for comp_idx in range(n_components):
                hist = hm.log_conf_hist[comp_idx]
                # Reorder from circular buffer
                ordered_hist = torch.cat([hist[ptr:], hist[:ptr]])

                # Show numeric values
                hist_str = " | ".join([f"{val:.2f}" for val in ordered_hist])

                # Create heatmap (using absolute values for visualization)
                heatmap = self._create_llr_heatmap_row(torch.abs(ordered_hist))

                lines.append(f"| {comp_idx} | {hist_str} | `{heatmap}` |\n")

            lines.append("\n")

        # Section C: Hypothesis Branch Information
        lines.append("## Hypothesis Branches\n\n")

        if hm.hypotheses:
            lines.append("| ID | Start KF | Visual Edges | Adjacency Nodes | Status |\n")
            lines.append("|----|----------|--------------|-----------------|--------|\n")

            for comp_id, hypo in hm.hypotheses.items():
                edge_count = len(hypo.visual_edges)
                adj_count = len(hypo.visual_adjacency)

                # Determine if active
                weight = float(hm.dist[2][comp_id].item())
                status = "🟢 Active" if weight > 1e-5 else "⚪ Inactive"

                lines.append(f"| {comp_id} | {hypo.start_idx} | {edge_count} | {adj_count} | {status} |\n")
        else:
            lines.append("*No hypothesis branches*\n")

        lines.append("\n")

        # Summary stats
        lines.append("## 📈 Summary\n\n")
        active_count = sum(1 for i in range(n_components) if float(hm.dist[2][i].item()) > 1e-5)
        realized_count = sum(1 for i in range(n_components) if i < hm.realized.numel() and bool(hm.realized[i].item()))
        lines.append(f"- **Active components**: {active_count}/{n_components}\n")
        lines.append(f"- **Realized branches**: {realized_count}\n")
        lines.append(f"- **Step counter**: {hm.step_counter}\n")
        return "".join(lines)

    def _generate_hypotheses_markdown(self, hypotheses: List[Hypothesis]):
        """Generate markdown for hypotheses (legacy function, use _generate_hypothesis_lifecycle_markdown instead)."""
        lines = []
        lines.append("# Hypotheses Summary\n")

        if not hypotheses:
            lines.append("No hypotheses available.\n")
            return "".join(lines)

        # Create table header
        lines.append("| ID | Start Keyframe |\n")
        lines.append("|----|----------------|\n")

        # Add each hypothesis as a row
        for hypo in hypotheses.values():
            lines.append(f"| {hypo.component_id} | {hypo.start_idx} |\n")

        lines.append("\n")

        return "".join(lines)

    def _log_graph_legend(self):
        """Log graph visualization legend as a static text document."""
        legend_text = """# Pose Graph Visualization Legend

## Node Colors
- 🔴 **Red**: Optimizable nodes (will be adjusted during optimization)
- 🟢 **Green**: Fixed nodes (anchors, held constant)
- ⚪ **Gray**: Other nodes

## Edge Colors
- 🟢 **Green**: Odometry edges (sequential motion between keyframes)
- 🔵 **Blue**: Visual edges (loop closures from visual recognition)
- 🔴 **Red**: Loop closure edges (constraints between hypotheses)

## Node Labels
Format: `comp_id | kf_id`
- `comp_id`: Hypothesis/component ID
- `kf_id`: Keyframe ID (or Temp ID for auxiliary vertices)

## Graph Structure
Each hypothesis is visualized as a separate directed graph showing the pose optimization problem structure.
"""
        rr.log("/graph/legend", rr.TextDocument(legend_text, media_type=rr.MediaType.MARKDOWN), static=True)

    def _visualize_graph(self, pgo_info: Dict[str, Any]):
        """Visualize pose graph with nodes (Vertex) and edges.
        
        Node colors:
        - Red: Optimizable nodes
        - Green: Fixed nodes
        - Gray: Other nodes
        
        Edge colors:
        - Green: Odometry edges
        - Blue: Visual edges
        - Red: Loop closure edges
        
        Args:
            pgo_info: Dictionary containing:
                - nodes: List[Vertex] - List of graph vertices
                - edges: List[Tuple[int, int, List[Edge]]] - List of edges with their factors
                - hypothesis_id: int - Which hypothesis this graph represents
                - optim_nodes_ids: Optional[Set[int]] - IDs of nodes to be optimized
                - fixed_nodes_ids: Optional[Set[int]] - IDs of fixed nodes
        """
        nodes = pgo_info.get("nodes", [])
        edges = pgo_info.get("edges", [])
        hypothesis_id = pgo_info.get("hypothesis_id", 0)
        optim_nodes_ids = pgo_info.get("optim_nodes_ids", set())
        fixed_nodes_ids = pgo_info.get("fixed_nodes_ids", set())
        
        if not nodes:
            return
        
        # Build node information
        node_ids = []
        node_labels = []
        node_colors = []
        
        for vertex in nodes:
            node_id = vertex.kf_id
            node_ids.append(node_id)
            
            # Node label: "comp_id | kf_id"
            # For temp vertices (created for other hypotheses), kf_id is the temp ID
            # We need to map back to the original keyframe if possible
            if node_id in self.hypothesis_manager.nodes:
                # Original keyframe
                original_kf = self.hypothesis_manager.nodes[node_id]
                label = f"{hypothesis_id} | KF{node_id}"
            else:
                # Temp vertex (likely from another hypothesis merged for LC)
                label = f"? | Temp{node_id}"
            
            node_labels.append(label)
            
            # Node color based on optimization status
            if node_id in optim_nodes_ids:
                node_colors.append([255, 100, 100])  # Red for optimizable nodes
            elif node_id in fixed_nodes_ids:
                node_colors.append([100, 255, 100])  # Green for fixed nodes
            else:
                node_colors.append([150, 150, 150])  # Gray for other nodes
        
        # Build edge information with different colors per edge type
        edge_list = []
        edge_colors = []
        
        # Color scheme:
        # - Odometry: Green [0, 200, 0]
        # - Visual: Blue [0, 100, 255]
        # - Loop Closure: Red [255, 0, 0]
        
        edge_type_colors = {
            "odometry": [0, 200, 0],
            "visual": [0, 100, 255],
            "loop_closure": [255, 0, 0],
        }
        
        for u, v, edge_factors in edges:
            # Determine edge type (use the first factor's type if multiple)
            if edge_factors:
                edge_type = edge_factors[0].type
                color = edge_type_colors.get(edge_type, [128, 128, 128])  # Default gray
            else:
                color = [128, 128, 128]
            
            edge_list.append((u, v))
            edge_colors.append(color)
        
        # Log the graph
        graph_entity = f"graph/hypothesis_{hypothesis_id}"
        
        if node_ids and edge_list:
            rr.log(
                graph_entity,
                [
                    rr.GraphNodes(
                        node_ids=node_ids,
                        labels=node_labels,
                        colors=node_colors,
                    ),
                    rr.GraphEdges(
                        edges=edge_list,
                        colors=edge_colors,
                        graph_type="directed"
                    ),
                ]
            )
        elif node_ids:
            # Only nodes, no edges
            rr.log(
                graph_entity,
                rr.GraphNodes(
                    node_ids=node_ids,
                    labels=node_labels,
                    colors=node_colors,
                )
            )
        
        # Log graph legend/info (static, only needs to be logged once)
        self._log_graph_legend()

    
    def visualize_pgo_step(self, pgo_info: Dict[str, Any], idx: int = None):
        """Visualize PGO step.

        Args:
            pgo_info: PGO information dictionary
            idx: (Deprecated) Ignored - uses current visualizer step counter
        """
        _ = idx  # Suppress unused parameter warning (kept for backward compatibility)
        if not self.visualize:
            return

        # Use visualizer's own step counter (don't increment, PGO happens within a step)
        rr.set_time("step", sequence=self.step_idx - 1)

        self._visualize_graph(pgo_info)
        

    