import numpy as np
from typing import Tuple, Union
from collections import deque
from cross.core.atlas import new_atlas_center
from cross.utils.lie_tensor import project_SE3, rotation_angle_from_quat
from cross.utils.profile import timeit
from cross.utils.fps import fps_monitor, start_fps_monitoring, stop_fps_monitoring
from cross.core.hypothesis import HypothesisManager
from cross.core.odom_accum import OdomAccumulator
from cross.core.types import Camera, Keyframe, EdgeType
from cross.core.simple_topo import SimpleTopo, SimpleTopoConfig
from cross.core.config import (
    SystemConfig, load_config, config_to_dict,
    PoseEstType, FilterMode,
)
import torch
import threading
import time
import json
import atexit
from loguru import logger
import sys
from cross.db.db import KeyframeDatabase
from cross.utils.probabilities import (
    convolve_gmm_batch_SE3,
)
import pypose as pp
from cross.visualization.viz_rr import RRViz
from cross.core.pgo import PoseGraph
from sklearn.cluster import DBSCAN
import copy
from pathlib import Path

import pickle
import os
import queue

logger.remove()
logger.add("logs/system.log", level="DEBUG", mode="w")  
logger.add(sys.stdout, level="INFO")


torch.set_printoptions(
    precision=4,          # two digits after the decimal
    sci_mode=False,        # turn off 1.23e+04 style
    linewidth=300,
)
class System:
    """Pose-aware topological mapping system.
    Important: everything is in OPENCV convention.
    """
    def __init__(
        self,
        device: str = 'cuda',
        visualize: bool = False,
        debug: bool = False,
        camera: Camera = None,
        visualizer: 'RRViz' = None,
        config: Union[SystemConfig, str, Path, None] = None,
        **kwargs,
    ):
        """
        Args:
            camera: the camera model
            device: the device to use for the map
            visualizer: optional RRViz instance to reuse across sessions. If None, creates a new one.
            config: SystemConfig instance, path to a YAML file, or None for defaults.
                    Additional **kwargs with matching section names (e.g. ``tracking={...}``)
                    are deep-merged on top for backward compatibility.
        """
        ########## system ####################
        if config is None:
            self.config = load_config(**kwargs) if kwargs else SystemConfig()
        elif isinstance(config, (str, Path)):
            self.config = load_config(config, **kwargs)
        elif isinstance(config, SystemConfig):
            # Apply any extra kwargs on top
            if kwargs:
                self.config = load_config(**{**config_to_dict(config), **kwargs})
            else:
                self.config = config
        else:
            raise TypeError(f"config must be SystemConfig, path, or None, got {type(config)}")

        # print the formatted config
        logger.info(f"System config: {json.dumps(config_to_dict(self.config), indent=4)}")

        self.async_update = self.config.async_update
        self.local_update_thread = None
        self.local_update_thread_running = threading.Event()
        self._cur_obs_lock = threading.Lock()

        self.device = device
        self.storage_device = "cuda" #"cpu"
        self.visualize = visualize
        self.debug = debug
        self.use_depth_pred = self.config.depth_pred.use_depth_pred
        if self.use_depth_pred:
            from cross.cv.depth_pred_uni import DepthPredUni
            self.depth_pred = DepthPredUni(device=self.device)

        
        ################ counter ################
        self._processed_frame_num = 0


        ########### tracking ###########
        self._cur_obs_queue = deque(maxlen=10)
        self._prev_obs = None
        self._odometry_readings = []
        self.use_odometry = self.config.tracking.use_odometry
        self.new_kf_after_n_unsuccessful_steps = self.config.tracking.new_kf_after_n_unsuccessful_steps
        self.base_measurement_std_diag = torch.tensor([0.2, 0.2, 0.3, 0.2, 0.2, 0.2])

        # Odometry uncertainty parameters (distance-based, not step-based)
        # For 1m translation -> 0.1m std, for 1 rad rotation -> 0.1 rad std
        # NOTE: for good odometry, 0.2~0.5
        # otherwise if the odometry is noisy, consider 0.5~1
        self.odom_std_per_meter = self.config.tracking.odom_std_per_meter
        self.odom_std_per_radian = self.config.tracking.odom_std_per_radian
        self.odom_min_std_translation = self.config.tracking.odom_min_std_translation
        self.odom_min_std_rotation = self.config.tracking.odom_min_std_rotation

        if self.use_odometry:
            self.odom_accumulator = OdomAccumulator(
                std_per_meter=self.odom_std_per_meter,
                std_per_radian=self.odom_std_per_radian,
                min_std_translation=self.odom_min_std_translation,
                min_std_rotation=self.odom_min_std_rotation,
            )
            self.odom_accumulator.register_item("since_last_step")
            self.odom_accumulator.register_item("since_last_add_kf")
            self.odom_accumulator.register_item("since_last_retrieval")

        self.use_VO = self.config.tracking.use_VO
        # Retrieval filtering mode configuration
        self.filter_mode = self.config.tracking.filter_mode
        self.adaptive_filter_cfg = self.config.tracking.adaptive_filter

        self._unsuccessful_retrieval_steps = 0

        self._last_retrieved_results = None
        
        # pose estimation model
        self.pose_est_type = self.config.pose_est.type
        if self.pose_est_type == PoseEstType.PNP:
            from cross.cv.pose_est_pnp import PoseEstPnP
            self.pose_est = PoseEstPnP(self.device, self.config.pose_est, camera)
        elif self.pose_est_type == PoseEstType.VGGT:
            from cross.cv.pose_est_vggt import PoseEstVGGT
            self.pose_est = PoseEstVGGT(self.device)

        ########### mapping ###########
        # kf parameters
        self.kf_gmm_n_components = self.config.mapping.kf_gmm_n_components
        self.kf_retrieval_threshold_new_kf = self.config.mapping.kf_retrieval_threshold_new_kf
        self.kf_match_threshold_new_kf = self.config.mapping.kf_match_threshold_new_kf
        self.new_component_weight_threshold = self.config.mapping.new_component_weight_threshold
        self.last_added_kf_id = None

        # database
        self.db : KeyframeDatabase = KeyframeDatabase(
            self,
            device=device,
            config=self.config.retrieval,
        )

        # hypothesis manager
        self.hypothesis_manager : HypothesisManager = HypothesisManager(
            system=self,
            n_components=self.kf_gmm_n_components,
            config=self.config.mapping.hypothesis,
        )

        # Topological map (odometry + proximity) used for lightweight planning
        topo_cfg = SimpleTopoConfig(
            proximity_distance_thresh=self.config.mapping.topo.proximity_distance_thresh,
            proximity_std_trans=self.config.mapping.topo.proximity_std_trans,
            proximity_std_rot=self.config.mapping.topo.proximity_std_rot,
            use_proximity_grid=self.config.mapping.topo.use_proximity_grid,
            enable_incremental_proximity=self.config.mapping.topo.enable_incremental_proximity,
        )
        self.topo_map: SimpleTopo = SimpleTopo(system=self, config=topo_cfg)

        # Initialize async loop-closure engine (latest-wins), if enabled
        lc_cfg = self.config.mapping.loop_closure
        self._pgo_apply_queue: 'queue.Queue' = queue.Queue(maxsize=lc_cfg.queue_size)
        self._lc_engine = None
        if lc_cfg.async_:
            from cross.core.lc_engine import LoopClosureEngine
            self._lc_engine = LoopClosureEngine(
                hypothesis_manager=self.hypothesis_manager,
                apply_queue=self._pgo_apply_queue,
                device=self.device,
                depth=1000,
                k_hop=2,
                queue_size=lc_cfg.queue_size,
            )
            self._lc_engine.start()

        # Local smoothing tracking
        self._last_smoothing_step = 0

        ################ camera ################
        original_camera = copy.deepcopy(camera)
        if self.pose_est_type == PoseEstType.VGGT:
            from cross.utils.camera import get_transforms_vggt
            self.rgb_transform, self.depth_transform = get_transforms_vggt(camera)
        else:
            from cross.utils.camera import get_transforms_target_max
            self.rgb_transform, self.depth_transform = get_transforms_target_max(camera)
        # Expose the (possibly resized/cropped) camera intrinsics used for all downstream geometry.
        self.camera = camera

        ################ visualization ################
        if visualizer is not None:
            # Reuse provided visualizer and update its hypothesis manager
            self.visualizer = visualizer
            self.visualizer.set_hypothesis_manager(self.hypothesis_manager)
            self.visualize = True
        elif self.visualize:
            # Create new visualizer with default settings
            self.visualizer = RRViz(
                camera=original_camera,
                hypothesis_manager=self.hypothesis_manager,
                visualize=self.visualize,
                config=self.config.visualization,
            )
        start_fps_monitoring()

        ################# clean up #################

        atexit.register(self.shutdown)


        if self.async_update:
            self.init_update_thread()

    @property
    def anchor_count(self):
        """Get the number of anchors."""
        return self.db._next_atlas_id
    
    @property
    def processed_count(self):
        """Get the processed count."""
        return self._processed_frame_num
    

    def init_update_thread(self):
        """Initialize the local update thread."""
        logger.info("Initializing the local update thread...")
        self.local_update_thread = threading.Thread(
            target=self.run_worker,
            name="local_update_thread",
            daemon=False,
        )
        self.local_update_thread_running.set()
        self.local_update_thread.start()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def shutdown(self):
        """Clean up the system."""
        logger.info("Shutting down the system...")
        time.sleep(1) # grace period for the visualizer to finish
        # Stop LC engine first
        if hasattr(self, "_lc_engine") and self._lc_engine is not None:
            try:
                self._lc_engine.stop()
            except Exception:
                pass
        if hasattr(self, "local_update_thread") and self.local_update_thread is not None:
            self.local_update_thread_running.clear()
            self.local_update_thread.join(timeout=1)

        stop_fps_monitoring()

    def get_all_keyframes(self):
        """Get all keyframes from the database."""
        return self.db.get_all_keyframes()
    
    def _init_system(
        self,
        rgb_image: torch.Tensor,
        depth_image: torch.Tensor,
        timestamp: float = None,
    ):
        """Initialize the system."""
        
        # currently atlas not used. So we just use the last atlas.
        all_atlases = self.db.get_all_atlases()
        if len(all_atlases) == 0:
            new_atlas = self.db.create_atlas()
        else:
            new_atlas = all_atlases[-1]
        self.current_atlas = new_atlas

        # init the current state gmm
        if self.db.get_size() > 0:
            # init from previous map
            new_pose = new_atlas_center(self)
            mu = pp.identity_SE3(self.kf_gmm_n_components, device=self.storage_device)
            mu[0,:3] = torch.tensor(new_pose)
            sigma = pp.identity_se3(self.kf_gmm_n_components, device=self.storage_device)
            weights = torch.zeros(self.kf_gmm_n_components, device=self.storage_device)
            weights[0] = 1.0

        else:
            # init from scratch
            mu = pp.identity_SE3(self.kf_gmm_n_components, device=self.storage_device)
            sigma= pp.identity_se3(self.kf_gmm_n_components, device=self.storage_device)
            weights = torch.zeros(self.kf_gmm_n_components, device=self.storage_device)
            weights[0] = 1.0

        self.hypothesis_manager.dist = (mu.to(self.device), sigma.to(self.device), weights.to(self.device))

        # insert the initial keyframe
        kf = self.db.insert(
            self._processed_frame_num,
            rgb_image.to(self.storage_device), 
            depth_image.to(self.storage_device) if depth_image is not None else None,
            mu=mu,
            sigma=sigma,
            weights=weights,
            atlas=new_atlas,
            timestamp=timestamp,
            temporary=False,
        )
        self.hypothesis_manager.add_node(kf)
        self.last_added_kf_id = kf.id
        self.odom_accumulator.reset_odom()
        return kf
        
    def run_worker(self):
        """Run worker."""
        while self.local_update_thread_running.is_set():
            # update the map
            if len(self._cur_obs_queue) == 0:
                logger.info("Waiting for observation...")
                time.sleep(1)
                continue
            self._step_work()

        logger.info("Local update thread finished")
    
    def get_current_pose(self):
        """Get the current pose.
        Note: this is not accurate in the global frame. 
        It should be only used for local planning.
        """
        return self.hypothesis_manager.dist[0][0]
    
    def get_current_kf(self):
        """Get the current node.
        This is used for planning.
        To get the current node, we simply compute the distance between the current node and all nodes,
        and return the node with the smallest distance.
        """
        def ang_wrap(x: torch.Tensor) -> torch.Tensor:
            # Normalize angle to [-pi, pi]
            return torch.atan2(torch.sin(x), torch.cos(x))

        def quat_to_yaw(qx, qy, qz, qw):
            # Yaw (around +Z) from quaternion (x, y, z, w)
            # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
            two = torch.tensor(2.0, device=qx.device, dtype=qx.dtype)
            num = two * (qw * qz + qx * qy)
            den = 1.0 - two * (qy * qy + qz * qz)
            return torch.atan2(num, den)
    
        current_pose = self.hypothesis_manager.dist[0][0]

        # TODO: we only consider the first component. is it ok?
        # TODO: we should only compute distance between current
        # and surrounding nodes, not all nodes, to make it more efficient
        all_nodes = list(self.hypothesis_manager.nodes.values())
        all_node_poses = [n.pose_mu[0] for n in all_nodes]
        all_node_poses = torch.stack(all_node_poses).to(self.device)

        # consider only the translation part
        distances = torch.norm(all_node_poses.tensor()[:, :3] - \
            current_pose.tensor()[None,:3], dim=1)
        
        closest_node_id = torch.argmin(distances).item()
        closest_node = all_nodes[closest_node_id]

        # get closest permanent node

        all_perm_nodes = [kf for kf in self.hypothesis_manager.nodes.values() if not kf.temporary]
        all_perm_node_poses = [n.pose_mu[0] for n in all_perm_nodes]
        all_perm_node_poses = torch.stack(all_perm_node_poses).to(self.device)
        all_perm_distances = torch.norm(all_perm_node_poses.tensor()[:, :3] - \
            current_pose.tensor()[None,:3], dim=1)
        closest_perm_node_id = torch.argmin(all_perm_distances).item()
        closest_perm_node = all_perm_nodes[closest_perm_node_id]


        # nearby_earliest_node: we find all kf that within a certain threshold
        # and return the earliest kf
        threshold = 1.5
        within_nodes_idx = torch.where(distances < threshold)[0].tolist()
        
        if len(within_nodes_idx) == 0:
            # if no nodes are within threshold, return the closest node
            nearby_earliest_node = closest_node
        else:
            within_nodes = [all_nodes[i] for i in within_nodes_idx]
            within_nodes = sorted(within_nodes, key=lambda x: x.id)
            # return the earliest kf
            nearby_earliest_node = within_nodes[0]

            # get aligned node
            cp = current_pose.tensor()
            c_qx, c_qy, c_qz, c_qw = cp[3], cp[4], cp[5], cp[6]
            current_yaw = quat_to_yaw(c_qx, c_qy, c_qz, c_qw)

            ap = all_node_poses.tensor()
            qx, qy, qz, qw = ap[:, 3], ap[:, 4], ap[:, 5], ap[:, 6]
            all_node_yaws = quat_to_yaw(qx, qy, qz, qw)

            thresh_deg = 45.0
            thresh_rad = thresh_deg * 3.141592653589793 / 180.0

            within_yaw_diffs = ang_wrap(all_node_yaws[within_nodes_idx] - current_yaw)
            aligned_mask = torch.abs(within_yaw_diffs) <= thresh_rad
            aligned_indices_local = torch.nonzero(aligned_mask, as_tuple=False).squeeze(-1).tolist()
            if isinstance(aligned_indices_local, int):
                aligned_indices_local = [aligned_indices_local]

            if len(aligned_indices_local) == 0:
                nearby_earliest_node_aligned = nearby_earliest_node
            else:
                aligned_global_indices = [within_nodes_idx[j] for j in aligned_indices_local]
                aligned_nodes = [all_nodes[i] for i in aligned_global_indices]
                aligned_nodes = sorted(aligned_nodes, key=lambda x: x.id)
                nearby_earliest_node_aligned = aligned_nodes[0]
        
        return {
            "nearby_earliest_node": nearby_earliest_node,
            "closest_node": closest_node,
            "closest_perm_node": closest_perm_node,
            "nearby_earliest_node_aligned": nearby_earliest_node_aligned,
        }
    
    
    def save_map(self, save_path: str):
        """Save the map.
        Saves only persistent graph structure:
        - permanent kfs, embeddings, and atlases in db
        - temporary kfs in hypothesis manager
        - odometry and visual edges (hypothesis 0 only)
        - class variables for ID tracking
        - system config

        Tracking state (GMM dist, metadata) is NOT saved.
        A new session will re-initialize tracking state on the first step.
        """

        logger.info(f"Saving map to {save_path}...")

        # Create directory if it doesn't exist
        save_path = os.path.join(os.getcwd(), save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # --- 1. Save Database State (includes atlases) ---
        db_data = self.db.save_state()

        # --- 2. Save Hypothesis Manager State (graph structure only, hypothesis 0 only) ---
        hypo_data = self.hypothesis_manager.save_state()

        # --- 3. Save Class Variables ---
        class_vars = {
            "keyframe_next_id": Keyframe._next_id,
        }

        # --- 4. Combine All Data ---
        save_data = {
            "config": config_to_dict(self.config),
            "db_data": db_data,
            "hypo_data": hypo_data,
            "class_vars": class_vars,
            "current_atlas_id": self.current_atlas.id if hasattr(self, 'current_atlas') else None,
        }

        # --- 5. Save to Disk ---
        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Map saved successfully to {save_path}")
        logger.info(f"  - Saved {len(db_data['keyframes'])} permanent keyframes")
        logger.info(f"  - Saved {len(db_data['atlases'])} atlases")
        logger.info(f"  - Saved {len(hypo_data['temp_keyframes'])} temporary keyframes")
        logger.info(f"  - Saved {len(hypo_data['odom_edges'])} odometry edges")
        logger.info(f"  - Saved {sum(len(h['visual_edges']) for h in hypo_data['hypotheses_data'].values())} visual edges (hypothesis 0)")
        logger.info(f"  - Tracking state (GMM dist, metadata) NOT saved - will re-initialize on first step")

        # Note: planning system (sparse graph) will be rebuilt on load if enabled

    def load_map(self, load_path: str):
        """Load the map.
        Loads persistent graph structure:
        - permanent kfs, embeddings, and atlases from db
        - temporary kfs and edges (hypothesis 0 only) from hypothesis manager
        - class variables for ID tracking

        Tracking state is NOT loaded and remains uninitialized (dist=None).
        On the first step() call after loading, _init_system() will:
        - Create a new keyframe at an estimated starting pose
        - Initialize tracking state (GMM dist, etc.)
        - If robot is in a previously visited area, loop closure will merge trajectories
        """
        load_path = os.path.join(os.getcwd(), load_path)
        logger.info(f"Loading map from {load_path}...")

        # --- 1. Load Data from Disk ---
        with open(load_path, "rb") as f:
            save_data = pickle.load(f)

        # --- 2. Restore Class Variables ---
        Keyframe._next_id = save_data["class_vars"]["keyframe_next_id"]

        # --- 3. Restore Database (includes atlases and keyframes) ---
        existing_keyframes = self.db.load_state(
            save_data["db_data"],
            self.storage_device
        )

        # --- 4. Restore Hypothesis Manager (graph structure only, hypothesis 0 only) ---
        self.hypothesis_manager.load_state(
            save_data["hypo_data"],
            self.db,
            self.storage_device,
            self.device,
            existing_keyframes
        )

        # Recompute next ID: object re-instantiation during load bumps the counter
        Keyframe._next_id = (max(self.hypothesis_manager.nodes.keys()) + 1) if self.hypothesis_manager.nodes else 0  

        # --- 5. Restore Current Atlas ---
        if save_data["current_atlas_id"] is not None:
            self.current_atlas = self.db.get_atlas(save_data["current_atlas_id"])
        else:
            # Create a new atlas if none exists
            atlases = self.db.get_all_atlases()
            if atlases:
                self.current_atlas = atlases[-1]
            else:
                self.current_atlas = self.db.create_atlas()

        # --- 6. Reset Visualizer ---
        # Clear visualization state to avoid timeline conflicts when step counter resets
        if self.visualize:
            self.visualizer.reset(new_session=False)

        logger.info(f"Map loaded successfully from {load_path}")
        logger.info(f"  - Loaded {len(save_data['db_data']['keyframes'])} permanent keyframes")
        logger.info(f"  - Loaded {len(save_data['db_data']['atlases'])} atlases")
        logger.info(f"  - Loaded {len(save_data['hypo_data']['temp_keyframes'])} temporary keyframes")
        logger.info(f"  - Loaded {len(save_data['hypo_data']['odom_edges'])} odometry edges")
        logger.info(f"  - Loaded {sum(len(h['visual_edges']) for h in save_data['hypo_data']['hypotheses_data'].values())} visual edges (hypothesis 0)")
        logger.info(f"  - Current Keyframe ID counter: {Keyframe._next_id}")
        logger.info(f"  - Tracking state (GMM dist) NOT loaded - will initialize on first step()")

        # Note: Planning system will rebuild sparse graph automatically when initialized (see PlanningSystem.__init__)

    @fps_monitor("Data Receive")
    @timeit
    def step(
        self,
        obs: dict,
        **kwargs,
    ):
        """Update the map.
        This is the main function that is called to update the map.
        Args:
            obs: the observation dict containing:
                - rgb_image: the rgb image, np.ndarray
                - depth_image: the depth image, np.ndarray
                - confidence_map: the confidence map of the depth image, np.ndarray
                - delta_pose: the delta pose between current and last step, np.ndarray
                - timestamp: the timestamp of the observation
        """


        # first accumulate the odometry
        if self.use_odometry:
            self.odom_accumulator.update_odom(obs["delta_pose"])
        
        if obs.get("rgb", None) is not None:

            # push obs to queue if rgb image is not None
            if self.async_update:
                self._step_async(obs, **kwargs)
            else:
                self._step_sync(obs, **kwargs)

        if kwargs.get("get_current_kf", False):
            return {
                "current_kf": self.get_current_kf(),
            }

    def _step_async(
        self,
        obs: dict,
        **kwargs,
    ):
        """Update the map.
        This is the main function that is called to update the map.
        Args:
            obs: the observation dict containing:
                - rgb: the rgb image, np.ndarray
                - depth: the depth image, np.ndarray
                - conf: the confidence map of the depth image, np.ndarray
                - delta_pose: the delta pose between current and last step, np.ndarray
                - timestamp: the timestamp of the observation
        """
        with self._cur_obs_lock:
            self._cur_obs_queue.append(obs)

    def _step_sync(
        self,
        obs: dict,
        **kwargs,
    ):
        """Update the map synchronized
        This is primarily used for testing
        """
        self._cur_obs_queue.append(obs)

        self._step_work(**kwargs)

    @fps_monitor("Process Step")
    @timeit
    def _step_work(self, **kwargs):
        """main function for step
        """
        logger.debug(f"Step work at step {self._processed_frame_num}")
        self._processed_frame_num += 1
        
        # get all synchronized observations
        with self._cur_obs_lock:
            last_obs = self._cur_obs_queue.popleft()

        rgb_image = last_obs["rgb"]
        depth_image = last_obs["depth"]
        confidence_map = last_obs["conf"]
        timestamp = last_obs.get("timestamp", None)

        if timestamp is None:
            timestamp = self._processed_frame_num
 
        
        ############ preprocess the image ############
        if self.use_depth_pred:
            pred_depth, confidence, output_dict = self.depth_pred.predict(rgb_image)
            depth_image = pred_depth.cpu().numpy()

            # add for visualization
            if 'data' in kwargs:
                kwargs['data']['depth'] = depth_image
        
        rgb_image = self.rgb_transform(rgb_image) # (3, H, W)
        if depth_image is not None:
            depth_image = torch.from_numpy(depth_image).float().unsqueeze(0) # (1, H, W)
            depth_image = self.depth_transform(depth_image) 

        ############ initialize the system ############
        if self._processed_frame_num == 1:
            kf = self._init_system(rgb_image, depth_image, timestamp=timestamp)

            if self.visualize:
                self.visualizer.visualize_tracking_step(
                    kf = kf,
                    state_info = None,
                    gt_info = kwargs.get("data"),
                    step_idx = self._processed_frame_num,
                )
            return

        ret = self._construct_motion_dist()

        self._prev_obs = (rgb_image, depth_image, confidence_map)

        #################################
        # Build pose update mask based on filter mode
        # By default, update all components with retrieval filtering
        #################################
        pose_update_mask = torch.ones(self.kf_gmm_n_components, dtype=torch.bool, device=self.device)
        if self.filter_mode == FilterMode.SKIP_ACTIVE:
            # Skip retrieval-based filtering for component 0 (active world)
            pose_update_mask[0] = False
        elif self.filter_mode == FilterMode.ADAPTIVE:
            # Dynamically decide for component 0 based on odometry uncertainty
            allow_filter_active = True
            delta_std = ret.get("delta_std", None)
            if delta_std is not None:
                std_t = delta_std.tensor()[:3]
                std_r = delta_std.tensor()[3:]
                tmax = torch.max(std_t).item()
                rmax = torch.max(std_r).item()
                t_thresh = self.adaptive_filter_cfg.trans_thresh
                r_thresh = self.adaptive_filter_cfg.rot_thresh
                allow_filter_active = (tmax >= t_thresh) or (rmax >= r_thresh)
            else:
                # If no motion std (kidnapped/missing), treat as high uncertainty -> allow filtering
                allow_filter_active = True
            pose_update_mask[0] = allow_filter_active

        ################################
        # first update motion prior 
        ################################
        # update the motion model gmm with odometry
        if not ret["kidnapped"]:
            self.hypothesis_manager.motion_update(ret["delta_pose"], ret["delta_std"])
            logger.debug(f"Updated the current state gmm with odometry at step {self._processed_frame_num}")
        
        else:
            # NOTE: we assumes that the system will always have some odom / imu readings
            # and when not supplied, it's kidnapped event.
            # We might add logic to handle temporary no sensor readings due to sensor failure in the future.
            logger.info(f"Kidnapped event detected at step {self._processed_frame_num} - resetting tracking state")

            # Reset hypothesis manager tracking state (metadata, evidence)
            self.hypothesis_manager.reset_tracking_state()

            # Reset visualizer coordinate frames and start new trajectory segments
            if self.visualize:
                self.visualizer.reset(new_session=False)

            # Re-initialize system (GMM dist, new keyframe)
            kf = self._init_system(rgb_image, depth_image, timestamp=timestamp)
            if self.visualize:
                self.visualizer.visualize_tracking_step(
                    kf = kf,
                    state_info = None,
                    gt_info = kwargs.get("data"),
                    step_idx = self._processed_frame_num,
                )
            return

        ################################
        # update the observation likelihood
        ################################
        ret.update(self._construct_observation_dist(rgb_image, depth_image))
        # if no proposal, continue with motion-only update
        if len(ret["valid_keyframes"]) == 0:
            logger.info(f"No valid keyframes found. Continuing with motion-only update")
            self._unsuccessful_retrieval_steps += 1

            # Populate ret with current state after motion update
            current_mu, current_sigma, current_weights = self.hypothesis_manager.dist
            ret['current_mu'] = current_mu
            ret['current_sigma'] = current_sigma
            ret['current_weights'] = current_weights
            ret['hypotheses'] = self.hypothesis_manager.hypotheses

            # Handle force-add keyframe if threshold exceeded
            new_kf = None
            if self._unsuccessful_retrieval_steps > self.new_kf_after_n_unsuccessful_steps:
                new_kf = self._add_new_kf(rgb_image, depth_image, force_permanent=True, force_add=True)
                self._unsuccessful_retrieval_steps = 0
                logger.info(f"Added new keyframe at step {self._processed_frame_num} due to unsuccessful retrieval")

            # Visualize even without valid keyframes
            if self.visualize:
                self.visualizer.visualize_tracking_step(
                    kf=new_kf,
                    state_info=ret,
                    gt_info=kwargs.get("data"),
                    step_idx=self._processed_frame_num,
                )
            return
        else:
            self._unsuccessful_retrieval_steps = 0
        
        #########################################################
        # analytical gmm filtering
        # first merge the proposal with DBSCAN clustering, 
        # then align the components with the current GMM
        # then filter the components with the current GMM
        #########################################################
        (proposal_gmm_mu, 
         proposal_gmm_sigma, 
         proposal_gmm_weights, 
         proposal_gmm_confidence, 
         edge_mapping) = self._merge_and_align_components(ret)


        self.hypothesis_manager.gmm_filtering(
            proposal_gmm_mu,
            proposal_gmm_sigma,
            proposal_gmm_weights,
            proposal_gmm_confidence,
            pose_update_mask=pose_update_mask,
        )
        current_mu, current_sigma, current_weights = self.hypothesis_manager.dist

        ret['current_mu'] = current_mu
        ret['current_sigma'] = current_sigma
        ret['current_weights'] = current_weights
        ret['proposal_mu'] = proposal_gmm_mu
        ret['proposal_sigma'] = proposal_gmm_sigma
        ret['proposal_weights'] = proposal_gmm_weights
        ret['proposal_confidence'] = proposal_gmm_confidence
        ret['hypotheses'] = self.hypothesis_manager.hypotheses
        ret['edge_mapping'] = edge_mapping

        #########################
        # detect loop closure
        #########################
        lc_result = self.hypothesis_manager.detect_loop_closure(ret)
        
        

        #########################
        # insert new keyframe
        # if loop closure is detected, we will force add a kf for pgo
        #########################
        new_kf = self._add_new_kf(
            rgb_image, 
            depth_image, 
            ret=ret,
            edge_mapping=edge_mapping,
            timestamp=timestamp,
            force_permanent=False,
            force_add=lc_result["loop_closure"],
        )

        if lc_result["loop_closure"]:
            logger.info(
                f"LC detected at step {self._processed_frame_num}, hypo id: {lc_result['loop_closure_hypo_id']}"
            )
            if self._lc_engine is not None:
                self._lc_engine.submit(lc_result["loop_closure_hypo_id"])
                ret["loop_closure_submitted"] = True
            else:
                # Fallback to synchronous handling
                pgo_info = self.hypothesis_manager.handle_loop_closure(
                    lc_result["loop_closure_hypo_id"]
                )
                if pgo_info.get("success"):
                    ret["loop_closure_pgo"] = pgo_info
                else:
                    message = pgo_info.get("message", "Loop closure handling failed without message")
                    logger.warning(message)

        # Apply any pending PGO results from the async engine and optionally force smoothing
        applied = self._apply_pending_pgo_results(ret)
        # Periodic local smoothing
        self._maybe_local_smoothing(force=applied)

        if self.visualize:
            self.visualizer.visualize_tracking_step(
                kf = new_kf,
                state_info = ret,
                gt_info = kwargs.get("data"),
                step_idx = self._processed_frame_num,
            )
   

    def _construct_motion_dist(
        self,
    ):
        """Construct the motion distribution.
        """
        ret = {"kidnapped": False}
        if self.use_odometry:
            delta_pose, std = self.odom_accumulator.get_since_last_reading("since_last_step")
            ret["delta_pose"] = delta_pose
            ret["delta_std"] = std
            
        elif self.use_VO:
            ret["delta_pose"] = ret["vo_delta_pose"]
            ret["delta_std"] = ret["vo_delta_std"]
        else:
            raise ValueError("No odometry or VO provided")
        
        # handle kidnapped event
        if delta_pose is None:
            ret["kidnapped"] = True
        
        return ret

    def _apply_pending_pgo_results(self, ret: dict = None) -> bool:
        """Apply any pending PGO results from the async LC engine.
        Returns True if something was applied.
        """
        applied = False
        latest = None
        # Drain queue; latest-wins
        while True:
            try:
                item = self._pgo_apply_queue.get_nowait()
                latest = item
            except Exception:
                break
        if latest is None:
            return False

        pgo_info = self.hypothesis_manager.apply_pgo_result(latest)
        applied = pgo_info.get("success", False)
        if applied:
            if ret is not None:
                ret["loop_closure_pgo"] = pgo_info
        return applied

    def _maybe_local_smoothing(self, force: bool = False) -> bool:
        """Run a small local PGO over the last window keyframes (k_hop=1) on the frontend.
        Returns True if smoothing applied.
        """
        ls_cfg = self.config.mapping.local_smoothing
        period = ls_cfg.period_steps
        if not (ls_cfg.enabled and (self._processed_frame_num - self._last_smoothing_step) < period) \
            and not force:
            return False

        target_node_id = self.last_added_kf_id if self.last_added_kf_id is not None else max(self.hypothesis_manager.nodes.keys())
        window_kfs = ls_cfg.window_kfs
        k_hop = ls_cfg.k_hop

        # Build graph inside lock for a consistent snapshot
        with self.hypothesis_manager.graph_lock:
            pg = PoseGraph(self.hypothesis_manager, depth=window_kfs, k_hop=k_hop, device=self.device)
            pg.construct_for_local_smoothing(target_node_id=target_node_id, window_kfs=window_kfs, k_hop=k_hop)

        if not pg.vertices or not pg.edges:
            return False

        # Fix earliest original KF in window
        original_kf_ids = [v.id for v in pg.vertices if v.id in self.hypothesis_manager.nodes]
        if not original_kf_ids:
            return False
        fixed_node_id = min(original_kf_ids)
        optim_node_ids = set([v.id for v in pg.vertices]) - {fixed_node_id}

        pg.solve(optim_node_ids=optim_node_ids, fixed_node_ids={fixed_node_id})

        # Apply smoothing updates to hypothesis 0
        std_reduction = self.config.pgo.std_reduction_factor
        with self.hypothesis_manager.graph_lock:
            for node_id, optimized_pose in pg.optimized_poses.items():
                if node_id in self.hypothesis_manager.nodes:
                    kf = self.hypothesis_manager.nodes[node_id]
                    kf.pose_mu[0] = optimized_pose
                    kf.pose_std[0] = kf.pose_std[0] * std_reduction
                    kf.last_pgo_step = int(self.hypothesis_manager.step_counter)

            # realign tracking dist to the latest KF after smoothing
            if self.last_added_kf_id is not None and self.last_added_kf_id in self.hypothesis_manager.nodes:
                last_kf = self.hypothesis_manager.nodes[self.last_added_kf_id]
                self.hypothesis_manager.dist[0][0] = last_kf.pose_mu[0]
                self.hypothesis_manager.dist[1][0] = last_kf.pose_std[0]
                self.hypothesis_manager.dist[2][0] = 1

        self._last_smoothing_step = self._processed_frame_num
        return True
    
    @timeit
    def _add_new_kf(
        self,
        rgb_image: torch.Tensor,
        depth_image: torch.Tensor,
        ret: dict = None,
        edge_mapping: dict = None,
        timestamp: float = None,
        force_permanent: bool = False,
        force_add: bool = False,
    ):
        """Add a new keyframe to the database.
        We constantly add kfs as robot moves.
        Kf is permanent if the the new kf is not too similar to the previous kf.
        Otherwise, it's virtual and will be merged later.
        """
        # force true => add permanent kf
        is_temp_kf = False if force_permanent else True
        delta_pose, std = self.odom_accumulator.get_since_last_reading("since_last_add_kf", reset=False)

        # if the robot moves too little, skip adding a kf
        if not force_add and rotation_angle_from_quat(delta_pose.tensor()[3:]) < 0.1 and \
            torch.norm(delta_pose.tensor()[:3]) < 0.2:
            return None

        if ret is not None:
            ret_weights = ret["valid_retrieval_weights"]
            confidence = ret["pose_est_conf"]

            # if current obs is not similar to other kfs, add a permanent kf
            if ret_weights.max() < self.kf_retrieval_threshold_new_kf or \
                (self.pose_est_type == PoseEstType.PNP and confidence.max() < self.kf_match_threshold_new_kf):
                is_temp_kf = False

        mu, sigma, weights = self.hypothesis_manager.get_active_dist()

        if not is_temp_kf:
            if weights.nonzero().numel() == 0:
                return None
            # insert kf into the database for permanent kf
            keyframe = self.db.insert(
                self._processed_frame_num,
                rgb_image.to(self.storage_device), 
                depth_image.to(self.storage_device) if depth_image is not None else None, 
                mu=mu.to(self.storage_device), 
                sigma=sigma.to(self.storage_device), 
                weights=weights.to(self.storage_device),    
                atlas=self.current_atlas,
                timestamp=timestamp,
                temporary=is_temp_kf,
            )
            logger.debug(f"Add permanent keyframe at step {self._processed_frame_num}. Total keyframes: {self.db.get_size()}")
        else:
            # otherwise create a temporary keyframe
            keyframe = Keyframe(
                pose_mu=mu.to(self.storage_device),
                pose_std=sigma.to(self.storage_device),
                pose_weights=weights.to(self.storage_device),
                timestamp=timestamp,
                temporary=is_temp_kf,
            )
            logger.debug(f"Add temporary keyframe at step {self._processed_frame_num}. Total keyframes: {self.db.get_size()}")
        
        self.hypothesis_manager.add_node(keyframe)

        # ---- Add new relative pose measurements to the graph ----
        current_kf_id = keyframe.id
        if ret is not None:
            # The results from pose estimation are our new edges
            valid_keyframes = ret["valid_keyframes"]
            
            # adding visual constraints
            for i, kf_i in enumerate(valid_keyframes):
                # this happends when tracking slot is full
                if kf_i.id not in edge_mapping:
                    continue
                source_comp_id, dest_comp_id = edge_mapping[kf_i.id]
                self.hypothesis_manager.add_edge(
                    id1=kf_i.id,
                    id2=current_kf_id,
                    rel_pose_mean=ret["valid_poses"][i],
                    rel_pose_std=ret["valid_stds"][i],
                    type=EdgeType.VISUAL,
                    from_comp_id=source_comp_id,
                    to_comp_id=dest_comp_id,
                )

        # adding odometry constraints
        if delta_pose is not None:
            self.hypothesis_manager.add_edge(
                id1=self.last_added_kf_id, # since odom edge is from last added kf to current kf
                id2=current_kf_id,
                rel_pose_mean=delta_pose,
                rel_pose_std=std,
                type=EdgeType.ODOMETRY,
            )
        self.odom_accumulator.reset_item("since_last_add_kf")
        self.last_added_kf_id = current_kf_id

        return keyframe

    @timeit
    def _merge_and_align_components(
        self,
        ret: dict,
        dbscan_eps: float = 1.0,
        dbscan_min_samples: int = 1, # Set to 1 to ensure every component is in a cluster
    ) -> Tuple[pp.LieTensor, pp.LieTensor, torch.Tensor]:
        """Clusters candidate absolute poses to generate distinct pose hypotheses.

        This method takes the convolved means (candidate absolute poses), clusters
        them using DBSCAN in a relevant (x, z, yaw) space, and then for each
        cluster, finds the best representative pose based on geometric quality.
        The cluster std is computed from the actual dispersion of member poses
        in se(3), optionally weighted by confidence.

        Args:
            ret (dict): The dictionary from the initial part of pose estimation.
            dbscan_eps (float): The DBSCAN epsilon parameter.
            dbscan_min_samples (int): The DBSCAN min_samples parameter.

        Returns:
            List[Dict]: A list of hypothesis dictionaries. Each dictionary contains:
                - 'pose' (pp.LieTensor): The representative pose for the hypothesis.
                - 'std' (pp.LieTensor): The associated std.
                - 'score' (float): A quality score for the hypothesis (e.g., inlier count).
                - 'inlier_count' (int): The number of inliers for the best pose in the cluster.
        """

        convolved_mus = ret["convolved_mus"].to(self.device)  # (B, K, 7)
        convolved_stds = ret["convolved_stds"].to(self.device)  # (B, K, 6)
        confidences = ret['pose_est_conf'].to(self.device)  # (B)
        component_weights = ret["valid_ref_component_weights"].to(self.device)  # (B, K)
        B, K = convolved_mus.shape[:2]

        # --- 1. Flatten Data and Track Sources in Batch ---

        # Create a boolean mask for all components with significant weight.
        valid_mask = component_weights > 1e-3  # Shape: (B, K)

        # Early exit if no components are valid.
        # Return default values matching align_proposal_prior signature
        assert torch.any(valid_mask), "No valid components found. Something is wrong."

        # Calculate scores for all B*K components at once using broadcasting.
        # Unsqueeze confidences from (B,) to (B, 1) to multiply with (B, K).
        all_scores = confidences.unsqueeze(1) * component_weights # Shape: (B, K)

        # Apply the mask to efficiently filter and flatten the data.
        # The result `flat_...` will have shape (N, ...), where N is the number of valid components.
        flat_mus = convolved_mus[valid_mask]
        flat_stds = convolved_stds[valid_mask]
        flat_scores = all_scores[valid_mask]
        flat_comp_weights = component_weights[valid_mask]

        # Get the original (b, k) indices for the valid components.
        # torch.where() returns a tuple of (row_indices, col_indices).
        valid_indices_tuple = torch.where(valid_mask)
        # Stack them to create an (N, 2) tensor of [b, k] source pairs.
        source_indices_map = torch.stack(valid_indices_tuple, dim=1)

        # change batch_id to kf_id
        kf_ids = torch.tensor([kf.id for kf in ret['valid_keyframes']], device=self.device)
        source_indices_map[:, 0] = kf_ids[source_indices_map[:, 0]]
        

        # --- 1. Project to Clustering Space (x, z, yaw) ---
        samples_se3 = pp.SE3(flat_mus)
        projected_data = project_SE3(samples_se3)
        clustering_data = projected_data.cpu().numpy()

        
        # --- 2. Run DBSCAN ---
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(clustering_data)
        labels = db.labels_
        unique_labels = set(labels)
        
        hypotheses = []
        
        # --- 3. Find Best Representative for Each Cluster ---

        # config: weighting and floors for dispersion-based std
        cluster_std_cfg = self.config.mapping.cluster_std
        use_conf_weight = cluster_std_cfg.use_conf_weight
        min_t = cluster_std_cfg.min_std_translation
        min_r = cluster_std_cfg.min_std_rotation

        for k in unique_labels:
            if k == -1: continue # Ignore noise points

            cluster_mask = (labels == k)
            
            # Get all candidates belonging to this cluster
            cluster_indices = np.where(cluster_mask)[0]
            
            # combine the retrieval score and the inlier count
            combined_scores = flat_scores[cluster_indices]

            # Find the best candidate within this cluster based on the number of inliers
            best_candidate_in_cluster_idx = cluster_indices[torch.argmax(combined_scores)]
            
            # The representative pose is the highest-scoring in cluster.
            representative_pose = pp.SE3(flat_mus[best_candidate_in_cluster_idx])

            # Compute cluster std from dispersion in se(3)
            # Fallback to candidate std if singleton cluster
            if len(cluster_indices) == 1:
                representative_std = pp.se3(flat_stds[best_candidate_in_cluster_idx])
            else:
                # Residuals r_i = Log(rep^{-1} ∘ μ_i) in prior tangent
                cluster_poses = pp.SE3(flat_mus[cluster_indices])
                residuals = (representative_pose.Inv() @ cluster_poses).Log().tensor()  # (M, 6)

                # Weights: either conf*comp_weight or component weight only
                if use_conf_weight:
                    w = flat_scores[cluster_indices].clone()
                else:
                    w = flat_comp_weights[cluster_indices].clone()

                w_sum = w.sum()
                if w_sum <= 1e-12 or torch.isnan(w_sum):
                    w = torch.ones_like(w) / w.numel()
                else:
                    w = w / w_sum

                # Weighted mean
                r_bar = (w.unsqueeze(-1) * residuals).sum(dim=0)  # (6,)
                # Weighted variance (pure dispersion)
                diffs = residuals - r_bar.unsqueeze(0)
                var = (w.unsqueeze(-1) * (diffs * diffs)).sum(dim=0)  # (6,)
                std = torch.sqrt(torch.clamp(var, min=1e-12))

                # Apply floors separately for translation (0..2) and rotation (3..5)
                floors = torch.tensor(
                    [min_t, min_t, min_t, min_r, min_r, min_r],
                    device=std.device,
                    dtype=std.dtype,
                )
                std = torch.maximum(std, floors)
                representative_std = pp.se3(std.unsqueeze(0)).squeeze(0)

                logger.debug(
                    f"Cluster {k}: size={len(cluster_indices)}, use_conf_weight={use_conf_weight}, std={std.tolist()}"
                )

            hypothesis_score = flat_scores[cluster_indices].sum()

            cluster_sources = source_indices_map[cluster_indices]

            hypotheses.append({
                'pose': representative_pose, # cluster representative pose
                'std': representative_std, # std from cluster dispersion (se(3))
                'score': hypothesis_score, # cluster score
                'source_indices': cluster_sources.tolist(), # M, 2
            })

        # make sure the hypotheses are in the same order as the current state GMM
        return self.hypothesis_manager.align_proposal_prior(hypotheses)

    def _retrieve_keyframes(self, rgb_image: torch.Tensor):
        """Retrieve the keyframes from the database, with caching for efficiency.
        We skip (to speed up) retrieval if:
        1) rotation is small since last kf (should be last retrieved results instead?)
        2) we have recent retrieval results

        TODO:
        Since we have odom, we can use it to make the decision based on delta pose since
        last retrieval. Optionally, we can use predicted essential matrix to check correspondence
        between some ORB features between the current and last retrieved results, and compute the
        sampson error. This helps when camera not moving but scene changes a lot.
        """
        retrieve = True
        if self.use_odometry:
            delta_pose, _ = self.odom_accumulator.get_since_last_reading("since_last_retrieval", reset=False, return_std=False)
            if rotation_angle_from_quat(delta_pose.tensor()[3:]) < 0.15 and \
                torch.norm(delta_pose.tensor()[:3]) < 0.2 and \
                self._last_retrieved_results is not None and \
                len(self._last_retrieved_results["scores"]) > 0:

                # TODO: max score < threshold should be considered, and last retrieval results
                logger.debug(f"Skipping retrieval... (delta pose: {delta_pose})")
                retrieve = False

        if retrieve:
            results = self.db.query(rgb_image) 
            self._last_retrieved_results = results
            self.odom_accumulator.reset_item("since_last_retrieval")
            return results
        else:
            return self._last_retrieved_results
    
    def _get_std_diag(
        self, 
        confidences: torch.Tensor,
        retrieval_scores: torch.Tensor,
        ):
        """Get the std diagonal based on the confidence.
        Args:
            confidences: (B,)
            retrieval_scores: (B,)
        Returns:
            std_diag: (B, 6)
        """
        if len(confidences) == 0:
            return []
        
        if self.pose_est_type == PoseEstType.PNP:
            # conf is num of inliers
            conf = confidences / self.config.pose_est.kp_detector.n_keypoints
        else:
            conf = confidences
        # the median of both confidences and retrieval scores is 0.5
        # so we multiply 4 to make the std roughtly same scale as base_std_diag
        std_diag = self.base_measurement_std_diag.unsqueeze(0) / (4 * conf.unsqueeze(1) * retrieval_scores.unsqueeze(1))
        return pp.se3(std_diag)
    
    @timeit
    def _construct_observation_dist(
        self,
        rgb_image: torch.Tensor,
        depth_image: torch.Tensor = None,
    ):
        """Construct the proposal distribution
        The proposal distribution is defined as a GMM, where
        - the mixing weights are the scores of the place recognition candidates,
        - the means are the poses of the estimated poses,
        - the stds are the scores of the confidence for the estimated poses.

        """
        ret = {
            "retrieval_scores": [],
            "retrieved_keyframes": [],
            "valid_masks": [],
            "valid_keyframes": [],
            "valid_retrieval_weights": [],
            "valid_retrieval_weights_normalized": [],
            "valid_ref_mus": [],
        }
        # first retrieve the image
        results = self._retrieve_keyframes(rgb_image) 
        if len(results["scores"]) == 0:
            logger.debug(f"No valid retrieval results found.")
            return ret

        retrieval_scores = results["scores"]
        keyframes = results["keyframes"]
        ref_rgbs = [p.raw_rgb_image for p in keyframes]
        ref_depths = [p.depth_image for p in keyframes] if depth_image is not None else None

        # insert VO pose est
        if self._prev_obs is not None and self.use_VO:
            ref_rgbs = [self._prev_obs[0]] + ref_rgbs
            ref_depths = [self._prev_obs[1]] + ref_depths

        ref_rgbs = torch.stack(ref_rgbs)
        ref_depths = torch.cat(ref_depths, dim=0) if ref_depths is not None else None

        # then estimate the relative pose
        valid_poses, valid_masks, confidences = self.pose_est.estimate_pose(
            ref_rgbs,
            ref_depths,
            rgb_image,
            depth_image,
        )
        # no valid pose
        if valid_masks.sum() == 0:
            logger.debug(f"No valid pose found.")
            return ret

        valid_keyframes = [k for k, m in zip(keyframes, valid_masks) if m]
        if len(valid_keyframes) == 0:
            logger.debug(f"No valid keyframes found.")
            return ret

        # valid retrieval weights
        retrieval_weights = torch.tensor([w for w, m in zip(retrieval_scores, valid_masks) if m])
        retrieval_weights_normalized = self._normalize_retrieval_weights(retrieval_weights)

        valid_stds = self._get_std_diag(confidences, retrieval_weights)

        # extract VO pose est
        vo_delta_pose = None
        vo_delta_std = None
        vo_delta_confidence = None
        if self.use_VO and self._prev_obs is not None:

            if valid_masks[0]:
                # retrieve the VO pose est
                vo_delta_pose = valid_poses[0]
                vo_delta_std = valid_stds[0]
                vo_delta_confidence = confidences[0]

                # update the valid poses, stds, match counts, valid masks, and var scales
                valid_poses = valid_poses[1:]
                valid_stds = valid_stds[1:]
                confidences = confidences[1:]
                
            valid_masks = valid_masks[1:]

        valid_poses = valid_poses.to(self.device)
        valid_stds = valid_stds.to(self.device)

        valid_ref_mus = torch.stack([p.pose_mu for p in valid_keyframes], dim=0).to(self.device) # (B, K, 7)
        valid_ref_stds = torch.stack([p.pose_std for p in valid_keyframes], dim=0).to(self.device) # (B, K, 6)
        valid_ref_component_weights = torch.stack([p.pose_weights for p in valid_keyframes], dim=0).to(self.device) # (B, K)

        # convolve the Gaussian with the GMM of the L keyframes
        convolved_mus, convolved_stds = convolve_gmm_batch_SE3(
            valid_ref_mus,
            valid_ref_stds,
            valid_poses,
            valid_stds,
            valid_ref_component_weights,
        )


        return {
            # retrieval
            "retrieval_scores": retrieval_scores,
            "retrieved_keyframes": keyframes,
            "valid_masks": valid_masks,
            "valid_keyframes": valid_keyframes,
            "valid_retrieval_weights": retrieval_weights,
            "valid_retrieval_weights_normalized": retrieval_weights_normalized,
            "valid_ref_mus": valid_ref_mus,
            "valid_ref_stds": valid_ref_stds,
            "valid_ref_component_weights": valid_ref_component_weights,
            # pose est
            "valid_poses": valid_poses,
            "valid_stds": valid_stds,
            "pose_est_conf": confidences,
            # GMM
            "convolved_mus": convolved_mus,
            "convolved_stds": convolved_stds,
            # VO pose est
            "vo_delta_pose": vo_delta_pose,
            "vo_delta_std": vo_delta_std,
            "vo_delta_confidence": vo_delta_confidence,
        }

    def _normalize_retrieval_weights(
        self,
        weights: torch.Tensor,
    ):
        """Normalize the weights.
        The weights are normalized to sum to 1.
        We can use different strategies:
        1) softmax
        2) linear
        3) etc.
        Ideally this normalized weights should represent the true probability of the retrieval results.
        Args:
            weights: (B,)
        Returns:
            normalized_weights: (B,)
        """
        return weights / weights.sum()

