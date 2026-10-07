import math
import numpy as np
from typing import Tuple, Union
from collections import deque
from cross.core.atlas import new_atlas_center
from cross.utils.lie_tensor import (normalize_se3, project_SE3, rotation_angle_from_quat, SE3Projection, estimate_vertical,
                                    split_clusters_by_vertical, vertical_vector)
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
from cross.utils.lie_tensor import normalize_SE3
from sklearn.cluster import DBSCAN
import copy
from pathlib import Path

import pickle

import os
import queue

logger.remove()
logger.add(os.environ.get("CROSS_LOG_FILE", "logs/system.log"), level="DEBUG", mode="w")  
logger.add(sys.stdout, level="INFO")


torch.set_printoptions(
    precision=4,          # two digits after the decimal
    sci_mode=False,        # turn off 1.23e+04 style
    linewidth=300,
)

def _scale_translation(delta, s: float):
    """An odometry reading (4x4 array or pypose SE3, translation first) with its translation multiplied by s."""
    if isinstance(delta, np.ndarray):
        out = np.array(delta, dtype=np.float64, copy=True)
        out[:3, 3] *= s
        return out
    t = delta.tensor().clone()
    t[..., :3] *= s
    return pp.SE3(t)


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
        T_right_in_left: np.ndarray = None,
        pose_estimator=None,
        **kwargs,
    ):
        """
        Args:
            camera: the camera model
            device: the device to use for the map
            visualizer: optional RRViz instance to reuse across sessions. If None, creates a new one.
            T_right_in_left: optional 4x4 stereo rig calibration (right camera pose in the left camera
                    frame, OpenCV convention). Required for the feed-forward estimator's stereo anchors.
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
        self.storage_device = device
        self.state_device = torch.device(self.config.state_device or device)
        self.visualize = visualize
        self.debug = debug
        self.use_depth_pred = self.config.depth_pred.use_depth_pred
        if self.use_depth_pred:
            from cross.cv.depth_pred_uni import DepthPredUni
            self.depth_pred = DepthPredUni(device=self.device)

        
        ################ counter ################
        self._processed_frame_num = 0
        # Immutable provenance of the graph present at map load. New query
        # nodes can share its atlas, so atlas IDs do not identify sessions.
        self.loaded_node_ids = frozenset()
        self._session_localized = False        # session_localized(): hypothesis 0 linked to the stored map


        ########### tracking ###########
        self._cur_obs_queue = deque(maxlen=10)
        self._prev_obs = None
        self._odometry_readings = []
        self.use_odometry = self.config.tracking.use_odometry
        self.new_kf_after_n_unsuccessful_steps = self.config.tracking.new_kf_after_n_unsuccessful_steps
        self.base_measurement_std_diag = torch.tensor([0.2, 0.2, 0.3, 0.2, 0.2, 0.2])
        if self.config.pose_est.type == PoseEstType.FF and self.config.pose_est.ff.base_measurement_std:
            self.base_measurement_std_diag = torch.tensor([float(v) for v in self.config.pose_est.ff.base_measurement_std])

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
                device=self.state_device,
            )
            self.odom_accumulator.register_item("since_last_step")
            self.odom_accumulator.register_item("since_last_add_kf")
            self.odom_accumulator.register_item("since_last_retrieval")
            self.odom_accumulator.register_item("since_last_obs")

        self.use_VO = self.config.tracking.use_VO
        # right images are only stored when stored keyframe pairs are used as scale anchors (storage option)
        self._store_right_images = self.config.pose_est.type == PoseEstType.FF and (
            self.config.pose_est.ff.n_ref_anchors > 0 or self.config.pose_est.ff.store_right_images)
        # Retrieval filtering mode configuration
        self.filter_mode = self.config.tracking.filter_mode
        self.adaptive_filter_cfg = self.config.tracking.adaptive_filter

        self._unsuccessful_retrieval_steps = 0

        self._last_retrieved_results = None
        # keyframes with id >= _session_start_kf_id belong to the current session (relocalization: the
        # map keyframes of previous sessions must not be crowded out of the retrieval by them)
        self._session_start_kf_id = 0
        # locality-aware retrieval (RetrievalConfig.locality): path length since the last anchoring event, external
        # position priors (System.add_location_prior) and the last step's search regions (diagnostics)
        self._locality_path = 0.0
        self._location_priors = []
        self.last_locality = None
        self._anchor_pending = deque()   # candidate session anchors (anchor_corroborate_window)
        self._contra_pending = deque()   # rejected map measurements that may contradict the anchor (anchor_contradict_min)
        self._session_start_frame = 0
        # observation cadence bookkeeping (feed-forward estimator)
        self._steps_since_obs = 0
        self._last_obs_mapped = False
        self._last_obs_rgb = None
        # intra-hypothesis loop closure bookkeeping: (step, n_long_range_edges) of recent keyframes
        self._intra_lc_events: deque = deque()
        self._last_intra_pgo_step = -10**9
        self.T_right_in_left = T_right_in_left
        
        # pose estimation model
        self.pose_est_type = self.config.pose_est.type
        if pose_estimator is not None:
            self.pose_est = pose_estimator
        elif self.pose_est_type == PoseEstType.PNP:
            from cross.cv.pose_est_pnp import PoseEstPnP
            self.pose_est = PoseEstPnP(self.device, self.config.pose_est, camera)
        elif self.pose_est_type == PoseEstType.VGGT:
            from cross.cv.pose_est_vggt import PoseEstVGGT
            self.pose_est = PoseEstVGGT(self.device)
        elif self.pose_est_type == PoseEstType.FF:
            from cross.cv.pose_est_ff import PoseEstFeedForward
            self.pose_est = PoseEstFeedForward(self.device, self.config.pose_est.ff, T_right_in_left)
        elif self.pose_est_type in {PoseEstType.DA3, PoseEstType.METRIC_PNP}:
            raise ValueError("Monocular retrieval requires an injected pose_estimator; use MonocularSystem")

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

        # place coordinates for proposal clustering and proposal-to-hypothesis matching (None: the original (x, z, yaw)
        # projection); with estimate_vertical the vertical is re-estimated from the keyframes as they are added
        pcfg = self.config.mapping.projection
        self._projection_prior = vertical_vector(pcfg.vertical)
        self._projection_n_nodes = -1
        self.place_projection = self._make_place_projection(self._projection_prior)

        # Topological map (odometry + proximity) used for lightweight planning
        topo_cfg = SimpleTopoConfig(
            proximity_distance_thresh=self.config.mapping.topo.proximity_distance_thresh,
            proximity_std_trans=self.config.mapping.topo.proximity_std_trans,
            proximity_std_rot=self.config.mapping.topo.proximity_std_rot,
            use_proximity_grid=self.config.mapping.topo.use_proximity_grid,
            enable_incremental_proximity=self.config.mapping.topo.enable_incremental_proximity,
            enabled=self.config.mapping.topo.enabled,
        )
        self.topo_map: SimpleTopo = SimpleTopo(system=self, config=topo_cfg)

        # Verified loop closure (consistency tests + calibrated noise model), or the heuristic intra-hypothesis PGO
        lc_cfg = self.config.mapping.loop_closure
        self._lc_verifier = None
        if lc_cfg.mode == "verified":
            if lc_cfg.noise_file:
                import yaml as _yaml
                from cross.core.config import NoiseModelConfig, _from_dict
                with open(lc_cfg.noise_file) as _f:
                    lc_cfg.noise = _from_dict(NoiseModelConfig, _yaml.safe_load(_f) or {})
                logger.info(f"Loop closure noise model loaded from {lc_cfg.noise_file}")
            from cross.core.lc_verify import LoopClosureVerifier
            self._lc_verifier = LoopClosureVerifier(self, lc_cfg)
            # the odometry scale guard needs translations that are metric without the odometry: PnP on depth, or the
            # feed-forward estimator with a stereo rig (not the monocular mode, whose scale comes from the odometry)
            self._lc_verifier.guard_metric = self.pose_est_type == PoseEstType.PNP or (
                self.pose_est_type == PoseEstType.FF and T_right_in_left is not None)
            # calibrated metric scale of the feed-forward estimator (measured / true translation, from the odometry)
            if float(getattr(lc_cfg.noise, "visual_scale", 1.0) or 1.0) != 1.0 and hasattr(self, "pose_est"):
                self.pose_est.metric_scale_correction = float(lc_cfg.noise.visual_scale)
                logger.info(f"Feed-forward metric scale correction: translations divided by {lc_cfg.noise.visual_scale:.3f}")
            if self.config.tracking.odom_std_from_noise_model and self.use_odometry:
                nz = lc_cfg.noise
                acc = self.odom_accumulator
                acc.std_per_meter, acc.std_per_radian = float(nz.odom_k_t), float(nz.odom_k_r)
                acc.min_std_translation, acc.min_std_rotation = float(nz.odom_floor_t), float(nz.odom_floor_r)
                self.odom_std_per_meter, self.odom_std_per_radian = acc.std_per_meter, acc.std_per_radian
                self.odom_min_std_translation, self.odom_min_std_rotation = acc.min_std_translation, acc.min_std_rotation
                logger.info(f"Tracking odometry std from the noise model: {nz.odom_k_t:.4f} per m, {nz.odom_k_r:.4f} per rad, "
                            f"floors {nz.odom_floor_t:.4f} m / {nz.odom_floor_r:.4f} rad per step")
        self._last_pgo_step = -10**9
        # Initialize async loop-closure engine (latest-wins), if enabled
        self._pgo_apply_queue: 'queue.Queue' = queue.Queue(maxsize=lc_cfg.queue_size)
        self._lc_engine = None
        if lc_cfg.async_:
            from cross.core.lc_engine import LoopClosureEngine
            self._lc_engine = LoopClosureEngine(
                hypothesis_manager=self.hypothesis_manager,
                apply_queue=self._pgo_apply_queue,
                device=self.state_device,
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
        elif self.pose_est_type == PoseEstType.FF:
            from cross.utils.camera import get_transforms_ff
            self.rgb_transform, self.depth_transform = get_transforms_ff(
                camera, image_resolution=self.config.pose_est.ff.image_resolution)
        else:
            from cross.utils.camera import get_transforms_target_max
            self.rgb_transform, self.depth_transform = get_transforms_target_max(camera)
        # Expose the (possibly resized/cropped) camera intrinsics used for all downstream geometry.
        self.camera = camera
        if self.pose_est_type == PoseEstType.FF and hasattr(self.pose_est, "set_camera"):
            self.pose_est.set_camera(camera.K, camera.frame_width, camera.frame_height)

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
        # the exit hook would otherwise keep every shut-down system (and its GPU map) alive until process exit
        atexit.unregister(self.shutdown)
        if self.visualize:
            time.sleep(1)  # grace period for the visualizer to finish
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
        rgb_right: torch.Tensor = None,
        initial_chart_pose=None,
    ):
        """Initialize tracking, optionally choosing a fresh map's coordinates.

        A delayed monocular snapshot may arrive after local motion has already
        been emitted. Its frontend pose can select the fresh map's gauge so
        initialization does not reset those coordinates. This is a coordinate
        choice, not another measurement or a constraint on a loaded map.
        Existing maps retain the original independent-atlas initialization.
        """
        
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
            mu = pp.identity_SE3(self.kf_gmm_n_components, device=self.state_device)
            mu[0,:3] = torch.tensor(new_pose)
            sigma = pp.identity_se3(self.kf_gmm_n_components, device=self.state_device)
            weights = torch.zeros(self.kf_gmm_n_components, device=self.state_device)
            weights[0] = 1.0

        else:
            # init from scratch
            mu = pp.identity_SE3(self.kf_gmm_n_components, device=self.state_device)
            if initial_chart_pose is not None:
                matrix = np.asarray(initial_chart_pose, dtype=np.float64)
                if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                        or not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-6, rtol=0)
                        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-5, rtol=0)):
                    raise ValueError('Initial chart pose must be a finite rigid 4x4 camera pose')
                mu[0] = normalize_se3(pp.mat2SE3(torch.as_tensor(matrix, dtype=mu.dtype,
                                                               device=self.state_device)))
            sigma= pp.identity_se3(self.kf_gmm_n_components, device=self.state_device)
            weights = torch.zeros(self.kf_gmm_n_components, device=self.state_device)
            weights[0] = 1.0

        # .to() aliases when storage and tracking use the same device. The
        # first saved node must remain a snapshot when live motion mutates
        # its prior, including its conditional source-response center.
        self.hypothesis_manager.dist = (mu.to(self.state_device).clone(), sigma.to(self.state_device).clone(),
                                       weights.to(self.state_device).clone())
        self.hypothesis_manager.start_tracking_chart()
        if self.config.mapping.hypothesis.conditional_sources:
            self.hypothesis_manager.initialize_source_filter()

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
            raw_rgb_right=rgb_right.to(self.storage_device) if (rgb_right is not None and self._store_right_images) else None,
            pose_charts=self.hypothesis_manager.get_active_charts(),
            metric_source=getattr(self, "_current_metric_source", None),
        )
        kf.step_created = int(self._processed_frame_num)
        self.hypothesis_manager.add_node(kf)
        self.last_added_kf_id = kf.id
        self.odom_accumulator.reset_odom()
        self._steps_since_obs = 0
        self._last_obs_mapped = False
        self._last_obs_rgb = rgb_image
        self._session_start_frame = self._processed_frame_num
        if getattr(self, "_lc_verifier", None) is not None:
            self._lc_verifier.reset_session()
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
        if self.hypothesis_manager.chart_aware:
            chart = int(self.hypothesis_manager.component_charts[0])
            all_nodes = [n for n in all_nodes if int(n.pose_charts[0]) == chart]
        all_node_poses = [n.pose_mu[0] for n in all_nodes]
        all_node_poses = torch.stack(all_node_poses).to(self.state_device)

        # consider only the translation part
        distances = torch.norm(all_node_poses.tensor()[:, :3] - \
            current_pose.tensor()[None,:3], dim=1)
        
        closest_node_id = torch.argmin(distances).item()
        closest_node = all_nodes[closest_node_id]

        # get closest permanent node

        all_perm_nodes = [kf for kf in all_nodes if not kf.temporary]
        all_perm_node_poses = [n.pose_mu[0] for n in all_perm_nodes]
        all_perm_node_poses = torch.stack(all_perm_node_poses).to(self.state_device)
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
        # never persist a map whose belief lives in a non-zero component (only hypothesis 0 is saved)
        self.hypothesis_manager.maybe_adopt_dominant_hypothesis(force=True)

        # verified mode: the stored map is the optimum of every informative measurement (calibrated Huber factors).
        # Online, the graph is optimised when a loop-closure candidate disagrees with it; the revisit edges recorded
        # after that optimisation are consistent with the graph and never triggered another one, so they enter here
        # (one optimisation of the whole map, well under a second for 500 keyframes)
        if self._lc_verifier is not None and 0 in self.hypothesis_manager.hypotheses:
            h0 = self.hypothesis_manager.hypotheses[0]
            n_inf = sum(1 for fs in h0.visual_edges.values() for f in fs if getattr(f, "informative", True))
            if n_inf > 0 and len(self.hypothesis_manager.nodes) >= 10:
                t_final = time.perf_counter()
                try:
                    v = self._lc_verifier
                    res = self.hypothesis_manager.handle_loop_closure(0, global_opt=bool(self.config.mapping.loop_closure.global_final_opt))
                    v.stats["pgo"] = v.stats.get("pgo", 0) + 1
                    n_rej = 0
                    pg0 = res.get("pose_graph")
                    logger.info(f"Final optimisation round 0: initial {getattr(pg0, 'initial_cost', None)} -> {res.get('cost')} "
                                f"({getattr(pg0, 'lm_iterations', None)} it, {len(getattr(pg0, 'vertices', []))} vertices, factors {getattr(pg0, 'n_factors', None)})")
                    # diagnostics: did the write-back land in the keyframes the graph reads?
                    mism = []
                    for nid, p in (getattr(pg0, "optimized_poses", None) or {}).items():
                        kf_ = self.hypothesis_manager.nodes.get(nid)
                        if kf_ is None:
                            continue
                        a_ = kf_.pose_mu[0].tensor().detach().cpu().numpy().reshape(-1); b_ = p.tensor().detach().cpu().numpy().reshape(-1)
                        d_ = float(np.linalg.norm(a_[:3] - b_[:3]))
                        qa_, qb_ = a_[3:7] / max(np.linalg.norm(a_[3:7]), 1e-12), b_[3:7] / max(np.linalg.norm(b_[3:7]), 1e-12)
                        ang_ = float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(qa_, qb_)))))))
                        if d_ > 1e-3 or ang_ > 0.1:
                            mism.append((int(nid), round(d_, 3), round(ang_, 1), bool(kf_.temporary), tuple(kf_.pose_mu.shape), str(kf_.pose_mu.device), str(kf_.pose_mu.dtype),
                                         [round(float(x), 4) for x in a_[3:7]], [round(float(x), 4) for x in b_[3:7]]))
                    logger.info(f"Final optimisation write-back check: {len(mism)} of {len(getattr(pg0, 'optimized_poses', {}))} nodes differ from the optimised pose; e.g. {mism[:8]}")
                    # posterior test over every informative measurement: the ones the robust optimisation cannot
                    # reconcile are quarantined and the map re-optimised (at most three rounds)
                    for _ in range(3):
                        if not (res.get("success") and self.config.mapping.loop_closure.use_posterior
                                and getattr(self.config.mapping.loop_closure, "posterior_action", "remove") == "remove"):
                            break
                        outliers = v.posterior_outliers(res["pose_graph"], informative_only=True)
                        if not outliers:
                            break
                        c_after_apply = self.hypothesis_manager.graph_cost_now()
                        try:   # diagnostics: the most expensive factor of the fresh graph vs the same pair in the optimiser's graph
                            import gtsam as _g
                            pgd = PoseGraph(self.hypothesis_manager, depth=1000, k_hop=2, device=self.state_device, noise_fn=self.hypothesis_manager.pgo_noise_fn(), skip_fn=self.hypothesis_manager.pgo_skip_fn())
                            pgd.eval_only = True
                            with self.hypothesis_manager.graph_lock:
                                pgd.construct_for_loop_closure(target_node_id=max(self.hypothesis_manager.nodes.keys()), other_hypothesis_id=0)
                            ids_ = [vv.id for vv in pgd.vertices if vv.id in self.hypothesis_manager.nodes]
                            pgd.solve(optim_node_ids=set(ids_) - {min(ids_)}, fixed_node_ids={min(ids_)})
                            gd, vd = pgd._graph, pgd._initial
                            worst_i = max(range(gd.size()), key=lambda i: gd.at(i).error(vd))
                            fd = gd.at(worst_i); ka, kb = list(fd.keys())[:2]
                            g0, r0, i0 = pg0._graph, pg0._result, pg0._initial
                            same = [i for i in range(g0.size()) if list(g0.at(i).keys())[:2] == [ka, kb]]
                            f0 = g0.at(same[0]) if same else None
                            def p7(P): q = P.rotation().toQuaternion(); t = P.translation(); return [round(float(x), 4) for x in (t[0], t[1], t[2], q.x(), q.y(), q.z(), q.w())]
                            logger.info(f"Worst factor diagnostics: keys {ka},{kb}: fresh error {fd.error(vd):.1f}; optimiser factor error at result {f0.error(r0) if f0 else None:.1f}, at fresh values {f0.error(vd) if f0 else None:.1f}; "
                                        f"fresh measurement {p7(fd.measured())} optimiser measurement {p7(f0.measured()) if f0 else None}; "
                                        f"fresh sigmas {np.round(fd.noiseModel().sigmas() if hasattr(fd.noiseModel(), 'sigmas') else fd.noiseModel().noise().sigmas(), 4).tolist()} "
                                        f"optimiser sigmas {np.round(f0.noiseModel().sigmas() if hasattr(f0.noiseModel(), 'sigmas') else f0.noiseModel().noise().sigmas(), 4).tolist() if f0 else None}; "
                                        f"values a: result {p7(r0.atPose3(ka))} fresh {p7(vd.atPose3(ka))}; b: result {p7(r0.atPose3(kb))} fresh {p7(vd.atPose3(kb))}")
                        except Exception as ex:
                            logger.warning(f"worst-factor diagnostics failed: {ex}")
                        sizes = [len(self.hypothesis_manager.hypotheses[0].visual_edges.get((a, b), [])) for (a, b, f, c2) in outliers]
                        for (a, b, f, c2) in outliers:
                            v.remove_edge(a, b, f)
                        c_after_remove = self.hypothesis_manager.graph_cost_now()
                        logger.info(f"Final optimisation diagnostics: cost after write-back {c_after_apply[0]:.1f} {c_after_apply[1]} worst {c_after_apply[2]}; "
                                    f"bucket sizes of removed {sizes}; cost after removal {c_after_remove[0]:.1f} {c_after_remove[1]} worst {c_after_remove[2]}")
                        n_rej += len(outliers)
                        res = self.hypothesis_manager.handle_loop_closure(0, global_opt=bool(self.config.mapping.loop_closure.global_final_opt))
                        v.stats["pgo"] = v.stats.get("pgo", 0) + 1
                        pgk = res.get("pose_graph")
                        logger.info(f"Final optimisation round: {len(outliers)} rejected (chi2 max {max(o[3] for o in outliers):.0f}, e.g. {[(o[0], o[1]) for o in outliers][:4]}); "
                                    f"initial {getattr(pgk, 'initial_cost', None)} -> {res.get('cost')} ({getattr(pgk, 'lm_iterations', None)} it, "
                                    f"{len(getattr(pgk, 'vertices', []))} vertices, factors {getattr(pgk, 'n_factors', None)})")
                    logger.info(f"Final optimisation of the map before saving: {n_inf} informative visual edges, {n_rej} rejected, "
                                f"cost {res.get('cost')} ({time.perf_counter() - t_final:.2f} s)")
                except Exception as ex:   # the map is saved as it is rather than not at all
                    logger.warning(f"Final optimisation of the map failed: {ex}")

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
            "coordinate_charts_version": 1 if self.hypothesis_manager.chart_aware else 0,
            "schmidt_map_geometry_version": ((2 if self.hypothesis_manager.map_geometry_basis == 'factor' else 1)
                                             if self.hypothesis_manager.schmidt_map_geometry else 0),
            "conditional_sources_version": 1 if (self.hypothesis_manager.source_states is not None or
                                                   self.hypothesis_manager.saved_source_belief is not None) else 0,
            "config": config_to_dict(self.config),
            "db_data": db_data,
            "hypo_data": hypo_data,
            "class_vars": class_vars,
            "current_atlas_id": self.current_atlas.id if hasattr(self, 'current_atlas') else None,
        }
        # local consistency of the map from its own posterior residuals (no ground truth): the map-consistency
        # model of the verified loop closure in later sessions
        if self._lc_verifier is not None:
            try:
                mc = self._lc_verifier.map_consistency()
                if mc:
                    save_data["map_consistency"] = mc
                    logger.info(f"Map consistency model (posterior residuals of {mc['n_edges']} edges): "
                                f"sigma_t = {mc['map_t_a']:.3f} + {mc['map_t_b']:.4f} d m")
            except Exception as ex:
                logger.warning(f"map consistency model failed: {ex}")

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

    def session_localized(self) -> bool:
        """Whether hypothesis 0 is expressed in the frame of the stored map.

        A session that loaded a map starts in a coordinate frame of its own, placed outside the stored map
        (atlas.new_atlas_center); its pose becomes a pose in the map only when a merge links one of its keyframes to a
        stored keyframe in hypothesis 0's graph (the merged hypothesis brings its session->map edges along).  Until then
        the pose must not be reported as a pose in the map.  A mapping session (nothing loaded) is its own map.  Once
        linked, the session stays localized, since the keyframe that held the link may be a temporary one removed later;
        replacing hypothesis 0 (merge, adoption) resets the flag and the link is checked again."""
        if not self.loaded_node_ids:
            return True
        if self._session_localized:
            return True
        h0 = self.hypothesis_manager.hypotheses.get(0)
        if h0 is None:
            return False
        start, loaded = self._session_start_kf_id, self.loaded_node_ids
        # only the session's keyframes (ids from start up to the id counter) can hold the link: look them up instead
        # of scanning the adjacency of the whole map when the map is much larger than the session
        adj = h0.visual_adjacency
        n_session = Keyframe._next_id - start
        keys = range(start, Keyframe._next_id) if 0 < 4 * n_session < len(adj) else [k for k in adj if k >= start]
        for kid in keys:
            nbrs = adj.get(kid)
            if nbrs is not None and not loaded.isdisjoint(nbrs):
                self._session_localized = True
                break
        return self._session_localized

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
        if save_data.get("coordinate_charts_version", 0) and not self.hypothesis_manager.chart_aware:
            raise ValueError("This map contains coordinate charts; enable chart-aware mapping to load it")
        if bool(save_data.get('conditional_sources_version',0)) != self.config.mapping.hypothesis.conditional_sources:
            raise ValueError('Conditional source maps require the same inference mode; rebuild a legacy reference map')
        from .map_geometry import validate_load_mode
        validate_load_mode(save_data,self.hypothesis_manager.schmidt_map_geometry,
                           self.hypothesis_manager.map_geometry_basis)

        # --- 2. Restore Class Variables ---
        Keyframe._next_id = save_data["class_vars"]["keyframe_next_id"]

        # --- 3. Restore Database (includes atlases and keyframes) ---
        existing_keyframes = self.db.load_state(
            save_data["db_data"],
            self.storage_device, pose_device=self.state_device
        )

        # --- 4. Restore Hypothesis Manager (graph structure only, hypothesis 0 only) ---
        self.hypothesis_manager.load_state(
            save_data["hypo_data"],
            self.db,
            self.state_device,
            self.state_device,
            existing_keyframes
        )

        # Recompute next ID: object re-instantiation during load bumps the counter
        Keyframe._next_id = (max(self.hypothesis_manager.nodes.keys()) + 1) if self.hypothesis_manager.nodes else 0  
        self.loaded_node_ids = frozenset(self.hypothesis_manager.nodes)
        self.hypothesis_manager.reference_support.start(self.loaded_node_ids)
        self._session_start_kf_id = Keyframe._next_id
        self._session_localized = False
        self._projection_n_nodes = -1                  # re-estimate the vertical from the loaded map
        self._anchor_pending.clear()
        self._contra_pending.clear()
        self._last_retrieved_results = None

        # --- 4b. map-consistency model stored with the map (verified loop closure) ---
        if self._lc_verifier is not None and save_data.get("map_consistency"):
            mc = save_data["map_consistency"]
            for k in ("map_t_a", "map_t_b", "map_r_a", "map_r_b"):
                setattr(self.config.mapping.loop_closure.noise, k, float(mc[k]))
            logger.info(f"Map consistency model from the map: sigma_t = {mc['map_t_a']:.3f} + {mc['map_t_b']:.4f} d m")

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

        if self.pose_est_type == PoseEstType.FF and hasattr(self.pose_est, "precompute"):
            # tokens of every map image now (exact, a few seconds once per process: the cache keeps them across map
            # reloads), instead of a slower pass whenever a map keyframe is retrieved for the first time
            from cross.db.db import as_float_image
            kfs = self.db.get_all_keyframes()
            images = [as_float_image(k.raw_rgb_image) for k in kfs if k.raw_rgb_image is not None]
            images += [as_float_image(k.raw_rgb_right) for k in kfs if getattr(k, "raw_rgb_right", None) is not None]
            self.pose_est.precompute(images)

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
                - rgb: the rgb image, np.ndarray
                - rgb_right: optional right stereo image, np.ndarray (feed-forward estimator anchor)
                - depth: the depth image, np.ndarray
                - conf: the confidence map of the depth image, np.ndarray
                - delta_pose: the delta pose between current and last step, np.ndarray
                - timestamp: the timestamp of the observation
                - initial_chart_pose: optional fresh-map gauge at this image;
                  only used for the first image when the database is empty
        """


        # first accumulate the odometry (translations rescaled by the odometry scale guard when it has fired)
        if self.use_odometry:
            v = getattr(self, "_lc_verifier", None)       # absent in test stubs built without __init__
            if v is not None and v.odom_scale != 1.0 and obs.get("delta_pose") is not None:
                obs = dict(obs, delta_pose=_scale_translation(obs["delta_pose"], v.odom_scale))
            self.odom_accumulator.update_odom(obs["delta_pose"], covariance=obs.get("motion_covariance"),
                                              source_factor=obs.get('motion_source_factor'))
        
        # a remote session (cross.remote) steps every frame the edge maps, but sends the image only of the frames it
        # expects the back end to observe (remote_frame without rgb: the step without an observation)
        if obs.get("rgb", None) is not None or obs.get("remote_frame", False):

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
        self.last_step_diagnostics = {"loop_closure_detected": False, "loop_closure_applied": False}
        self._processed_frame_num += 1
        
        # get all synchronized observations
        with self._cur_obs_lock:
            last_obs = self._cur_obs_queue.popleft()

        rgb_image = last_obs["rgb"]
        self._frontend_anchor = last_obs.get("frontend_anchor")
        depth_image = last_obs.get("depth", None)
        confidence_map = last_obs.get("conf", None)
        rgb_right = last_obs.get("rgb_right", None)
        timestamp = last_obs.get("timestamp", None)
        self._current_metric_source = last_obs.get("metric_source")
        if "metric_input" in last_obs:
            # These are frontend input derivatives, not filtered node-pose
            # sensitivities. Retain them as diagnostics for the coupled model.
            self.last_step_diagnostics["metric_input"] = last_obs["metric_input"]

        if timestamp is None:
            timestamp = self._processed_frame_num
 
        
        ############ preprocess the image ############
        if rgb_image is None and (self._processed_frame_num == 1 or self.hypothesis_manager.dist is None):
            raise ValueError("A remote session must send the image of its first mapped frame")
        if self.use_depth_pred and rgb_image is not None:
            pred_depth, confidence, output_dict = self.depth_pred.predict(rgb_image)
            depth_image = pred_depth.cpu().numpy()

            # add for visualization
            if 'data' in kwargs:
                kwargs['data']['depth'] = depth_image
        
        if rgb_image is not None:
            rgb_image = self.rgb_transform(rgb_image) # (3, H, W)
        if rgb_right is not None:
            rgb_right = self.rgb_transform(rgb_right)
        if depth_image is not None:
            depth_image = torch.from_numpy(depth_image).float().unsqueeze(0) # (1, H, W)
            depth_image = self.depth_transform(depth_image) 

        ############ initialize the system ############
        if self._processed_frame_num == 1:
            kf = self._init_system(rgb_image, depth_image, timestamp=timestamp, rgb_right=rgb_right,
                                   initial_chart_pose=last_obs.get('initial_chart_pose'))

            if self.visualize:
                self.visualizer.visualize_tracking_step(
                    kf = kf,
                    state_info = None,
                    gt_info = kwargs.get("data"),
                    step_idx = self._processed_frame_num,
                )
            return

        ret = self._construct_motion_dist()

        if rgb_image is not None:
            self._prev_obs = (rgb_image, depth_image, confidence_map)

        #################################
        # Build pose update mask based on filter mode
        # By default, update all components with retrieval filtering
        #################################
        pose_update_mask = torch.ones(self.kf_gmm_n_components, dtype=torch.bool, device=self.state_device)
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
            self.hypothesis_manager.motion_update(ret["delta_pose"], ret["delta_std"],
                                                  source_factor=ret.get('motion_source_factor'))
            if self.config.retrieval.locality.enabled:
                self._locality_path += float(torch.linalg.norm(ret["delta_pose"].tensor().reshape(-1)[:3]))
            logger.debug(f"Updated the current state gmm with odometry at step {self._processed_frame_num}")
        
        else:
            # NOTE: we assumes that the system will always have some odom / imu readings
            # and when not supplied, it's kidnapped event.
            # We might add logic to handle temporary no sensor readings due to sensor failure in the future.
            logger.info(f"Kidnapped event detected at step {self._processed_frame_num} - resetting tracking state")
            if rgb_image is None:
                raise ValueError("A remote session must send the image of a frame after missing odometry")

            # Reset hypothesis manager tracking state (metadata, evidence)
            self.hypothesis_manager.reset_tracking_state()

            # Reset visualizer coordinate frames and start new trajectory segments
            if self.visualize:
                self.visualizer.reset(new_session=False)

            # Re-initialize system (GMM dist, new keyframe)
            kf = self._init_system(rgb_image, depth_image, timestamp=timestamp, rgb_right=rgb_right)
            if self.visualize:
                self.visualizer.visualize_tracking_step(
                    kf = kf,
                    state_info = None,
                    gt_info = kwargs.get("data"),
                    step_idx = self._processed_frame_num,
                )
            return

        ################################
        # observation cadence gating: with a feed-forward estimator the observation
        # (retrieval + multi-view forward pass) is expensive, while odometry is cheap.
        # Skip the observation until the robot moved enough (or N steps elapsed) and
        # let the motion model carry the belief in between.
        ################################
        skip = self._should_skip_observation()
        if not skip and rgb_image is None:
            # a remote session's edge did not send this frame's image: the observation waits for the next one
            skip = True
            self.last_step_diagnostics["observation_deferred"] = True
        if skip:
            current_mu, current_sigma, current_weights = self.hypothesis_manager.dist
            ret['current_mu'] = current_mu
            ret['current_sigma'] = current_sigma
            ret['current_weights'] = current_weights
            ret['hypotheses'] = self.hypothesis_manager.hypotheses
            ret['observation_skipped'] = True
            ret['valid_keyframes'] = []
            if self.visualize:
                self.visualizer.visualize_tracking_step(
                    kf=None, state_info=ret, gt_info=kwargs.get("data"), step_idx=self._processed_frame_num,
                )
            return

        # temporal anchor for the feed-forward estimator: previous observed frame + odometry
        odom_anchor = None
        if self.use_odometry:
            T_prev_curr, _ = self.odom_accumulator.get_since_last_reading("since_last_obs", reset=True, return_std=False)
            if self._last_obs_rgb is not None and T_prev_curr is not None:
                odom_anchor = {"image": self._last_obs_rgb, "T_prev_curr": T_prev_curr.matrix().cpu().numpy()}
        fa = getattr(self, "_frontend_anchor", None)
        if fa is not None:
            # the frontend's anchor (vgio): its last measured frame and its motion since then replace the previous
            # observation as the temporal anchor (cross.pipeline.Pipeline.advance)
            odom_anchor = {"image": self.rgb_transform(fa["rgb"]), "T_prev_curr": np.asarray(fa["T_prev_curr"]),
                           "frontend": True, "metric": bool(fa.get("metric")), "token": fa.get("token"),
                           "extra_images": [self.rgb_transform(x) for x in fa.get("extra_rgb", [])]}
        self._steps_since_obs = 0
        self._last_obs_rgb = rgb_image

        ################################
        # update the observation likelihood
        ################################
        ret.update(self._construct_observation_dist(rgb_image, depth_image, rgb_right=rgb_right, odom_anchor=odom_anchor))
        self.last_step_diagnostics["verified_keyframes"] = len(ret["valid_keyframes"])
        self.last_step_diagnostics["retrieval_audit"] = ret["retrieval_audit"]
        self.last_step_diagnostics["loaded_node_count"] = len(self.loaded_node_ids)
        if ret.get("h0_ok") is not None:
            # prior-consistency verdicts of the verified loop closure for this observation's references
            self.last_step_diagnostics["prior_gate"] = dict(
                keyframes=[int(k.id) for k in ret["valid_keyframes"]],
                consistent=[None if o is None else bool(o) for o in ret["h0_ok"]],
                chi2=[None if c is None else float(c) for c in (ret.get("h0_chi2") or [])],
                loop_candidate=[bool(x) for x in (ret.get("h0_loop") or [])])
        if getattr(self, "_lc_verifier", None) is not None:
            self.last_step_diagnostics["verifier_stats"] = {k: (float(v) if isinstance(v, float) else v)
                                                            for k, v in self._lc_verifier.stats.items()}
        # if no proposal, continue with motion-only update
        if len(ret["valid_keyframes"]) == 0:
            self.hypothesis_manager.reference_support.observe(self._processed_frame_num, {})
            logger.info(f"No valid keyframes found. Continuing with motion-only update")
            self._unsuccessful_retrieval_steps += 1
            self._last_obs_mapped = False

            # Populate ret with current state after motion update
            current_mu, current_sigma, current_weights = self.hypothesis_manager.dist
            ret['current_mu'] = current_mu
            ret['current_sigma'] = current_sigma
            ret['current_weights'] = current_weights
            ret['hypotheses'] = self.hypothesis_manager.hypotheses

            # Handle force-add keyframe if threshold exceeded
            new_kf = None
            if self._unsuccessful_retrieval_steps > self.new_kf_after_n_unsuccessful_steps:
                new_kf = self._add_new_kf(rgb_image, depth_image, force_permanent=True, force_add=True,
                                          timestamp=timestamp, rgb_right=rgb_right)
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
        self.hypothesis_manager.reference_support.observe(
            self._processed_frame_num, edge_mapping,
            torch.where(self.hypothesis_manager.newborn)[0].tolist())
        self.last_step_diagnostics["proposal_audit"] = self.hypothesis_manager.last_alignment_audit


        # hypothesis 0 is not moved by measurements that the odometry chain already knows better (references just
        # behind the robot): fusing them as independent evidence re-applies the estimator's bias at every observation
        # (their keyframe poses came from the same belief) and drifts the map; they still weigh the hypotheses and
        # become graph edges
        variance_update_mask = None
        if self.config.mapping.hypothesis.h0_informative_only and self.hypothesis_manager.comp0_informative is False:
            pose_update_mask[0] = False
            ret["h0_pose_update_skipped"] = True
            if getattr(self.config.mapping.hypothesis, "h0_skip_keeps_variance", False) and self._session_start_kf_id == 0:
                variance_update_mask = torch.zeros_like(pose_update_mask)
                variance_update_mask[0] = True

        self.hypothesis_manager.gmm_filtering(
            proposal_gmm_mu,
            proposal_gmm_sigma,
            proposal_gmm_weights,
            proposal_gmm_confidence,
            pose_update_mask=pose_update_mask,
            variance_update_mask=variance_update_mask,
            **(dict(source_factors={i:m.factor for i,m in self.hypothesis_manager.aligned_conditional_models.items()},
                    source_covariances={i:m.geometry_covariance for i,m in self.hypothesis_manager.aligned_conditional_models.items()})
               if self.hypothesis_manager.source_states is not None else {}),
        )
        if self.hypothesis_manager.source_states is not None:
            self.last_step_diagnostics['conditional_filter'] = self.hypothesis_manager.last_conditional_audit
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
        new_kf, lc_result = self._add_keyframe_and_detect_loop(rgb_image, depth_image, ret, edge_mapping, timestamp,
                                                               rgb_right=rgb_right)
        self._last_obs_mapped = new_kf is None or bool(getattr(new_kf, "temporary", False))

        if lc_result["loop_closure"]:
            logger.info(
                f"LC detected at step {self._processed_frame_num}, hypo id: {lc_result['loop_closure_hypo_id']}"
            )
            self.note_anchor("loop_closure")
            self._last_merge_step = self._processed_frame_num
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
                    self.last_step_diagnostics["loop_closure_applied"] = True
                else:
                    message = pgo_info.get("message", "Loop closure handling failed without message")
                    logger.warning(message)

        # Apply any pending PGO results from the async engine and optionally force smoothing
        applied = self._apply_pending_pgo_results(ret)
        # Periodic local smoothing
        self._maybe_local_smoothing(force=applied)
        # a realized hypothesis that has taken over the belief for long without a loop closure becomes hypothesis 0
        if not lc_result["loop_closure"] and not ret.get("loop_closure_submitted", False):
            if self._lc_verifier is not None:
                done = self._maybe_verified_loop_closure(ret, new_kf, edge_mapping)
            else:
                done = self._maybe_intra_hypothesis_loop_closure(ret, new_kf, edge_mapping)
            if done:
                self.note_anchor("loop_closure")
            if not done:
                self.hypothesis_manager.maybe_adopt_dominant_hypothesis()

        if self.visualize:
            self.visualizer.visualize_tracking_step(
                kf = new_kf,
                state_info = ret,
                gt_info = kwargs.get("data"),
                step_idx = self._processed_frame_num,
            )
   

    def _add_keyframe_and_detect_loop(self, rgb_image, depth_image, ret, edge_mapping, timestamp, rgb_right=None):
        """Check newly realized branches after their current graph edges exist.

        Realization happens inside add_node. A final successful observation can
        both qualify a branch and provide its last required historical support;
        it must not need a future successful retrieval merely to check the same
        evidence. Detection reads history and never accumulates it a second time.
        """
        manager = self.hypothesis_manager
        lc_result = self._mapping_merge_gate(manager.detect_loop_closure(ret))
        branches_before = set(manager.hypotheses)
        new_kf = self._add_new_kf(rgb_image, depth_image, ret=ret, edge_mapping=edge_mapping,
                                timestamp=timestamp, force_permanent=False, force_add=lc_result["loop_closure"],
                                rgb_right=rgb_right)
        recheck = (self.config.mapping.hypothesis.lc_recheck_after_realization
                   and not lc_result["loop_closure"] and new_kf is not None
                   and bool(set(manager.hypotheses) - branches_before))
        if recheck:
            lc_result = self._mapping_merge_gate(manager.detect_loop_closure(ret))
        self.last_step_diagnostics.update(loop_closure_detected=bool(lc_result["loop_closure"]),
                                         commitment_audit=manager.last_loop_audit,
                                         commitment_rechecked_after_realization=recheck)
        return new_kf, lc_result

    def _mapping_merge_gate(self, lc_result: dict) -> dict:
        """In a session without a loaded map, another hypothesis is not merged into hypothesis 0 (unless
        hypothesis.merge_in_mapping_session): hypothesis 0 is the odometry chain the map is built on and never lost, a
        revisit consistent with it is closed by the verified loop closure, and a hypothesis exists only because its
        measurements were inconsistent with that chain (aliased places)."""
        cfg = getattr(getattr(getattr(self, "config", None), "mapping", None), "hypothesis", None)
        if (lc_result.get("loop_closure") and int(getattr(self, "_session_start_kf_id", 0)) == 0
                and not getattr(cfg, "merge_in_mapping_session", True)):
            logger.debug(f"merge of hypothesis {lc_result.get('loop_closure_hypo_id')} not applied in a mapping session")
            if getattr(self, "_lc_verifier", None) is not None:
                self._lc_verifier.stats["mapping_merges_skipped"] = self._lc_verifier.stats.get("mapping_merges_skipped", 0) + 1
            return dict(lc_result, loop_closure=False)
        return lc_result

    def _verify_references(self, valid_keyframes, valid_poses, keyframes, valid_masks, ret):
        """Tests 2 and 1 of the verified loop closure on the references of the current forward pass.
        Returns (keep mask over valid_keyframes, prior verdicts over valid_keyframes or None)."""
        v = self._lc_verifier
        lc_cfg = self.config.mapping.loop_closure
        h0_ok = None
        if lc_cfg.use_prior and self.use_odometry and self.last_added_kf_id is not None:
            try:
                T_since, _ = self.odom_accumulator.get_since_last_reading("since_last_add_kf", reset=False, return_std=False)
                last_kf = self.hypothesis_manager.nodes.get(self.last_added_kf_id)
                last_step = getattr(last_kf, "step_created", None) if last_kf is not None else None
                n_since = max(int(self._processed_frame_num) - int(last_step), 1) if last_step is not None else 1
                h0_ok, h0_chi2 = v.prior_gate(valid_keyframes, [valid_poses[i] for i in range(len(valid_keyframes))],
                                              self.last_added_kf_id, T_since, n_since)
                ret["h0_chi2"] = h0_chi2
                ret["h0_loop"] = list(getattr(v, "last_loop_flags", []))
                # online metric-scale correction of the estimator for the following measurements (only the feed-forward
                # estimator applies it; PnP translations come from metric depth, and updating the correction the verifier
                # assumes applied without applying it would compound the ratio at every observation)
                if self.pose_est_type == PoseEstType.FF and abs(v.scale_ratio - v.metric_correction) > 1e-6:
                    v.metric_correction = v.scale_ratio
                    self.pose_est.metric_scale_correction = v.scale_ratio
                if lc_cfg.anchor_contradict_min > 0 and v.anchor is not None and self._session_start_kf_id > 0 \
                        and any(o is False for o in h0_ok):
                    self._contradict_anchor(valid_keyframes, valid_poses, h0_ok, T_since, n_since)
                if lc_cfg.anchor_corroborate_window > 0 and v.anchor is None and self._session_start_kf_id > 0:
                    self._corroborate_anchor(valid_keyframes, valid_poses, h0_ok, T_since, n_since)
                logger.debug(f"prior gate: {[(int(k.id), o, None if c is None else round(c, 1), l) for k, o, c, l in zip(valid_keyframes, h0_ok, h0_chi2, ret['h0_loop'])]} last_kf {self.last_added_kf_id}")
                if any(o is False for o in h0_ok):
                    logger.debug(f"prior consistency: {[(int(k.id), None if c is None else round(c, 1)) for k, c in zip(valid_keyframes, h0_chi2)]} (scales {v.scales})")
            except Exception as ex:   # the verifier must never break the observation
                logger.warning(f"prior consistency test failed: {ex}")
                h0_ok = None
        keep = np.ones(len(valid_keyframes), dtype=bool)
        if lc_cfg.use_inpass and len(valid_keyframes) >= 2:
            info = getattr(self.pose_est, "last_info", {}) or {}
            if info.get("valid") and "c2w_metric" in info:
                prior_full = None
                if h0_ok is not None:
                    prior_full = [None] * len(keyframes)
                    for k, i in enumerate(np.where(valid_masks)[0]):
                        prior_full[int(i)] = h0_ok[k]
                try:
                    full_keep = v.inpass_gate(np.asarray(info["c2w_metric"]), keyframes, valid_masks, covis=info.get("covis"), prior_ok=prior_full)
                    keep = full_keep[valid_masks]
                except Exception as ex:
                    logger.warning(f"in-pass consistency test failed: {ex}")
        return keep, h0_ok

    def _kf_position(self, kf_id):
        node = self.hypothesis_manager.nodes.get(int(kf_id)) if kf_id is not None else None
        return None if node is None else node.pose_mu[0].tensor()[:3].detach().cpu().numpy()

    def _separated(self, kf_id, others) -> bool:
        """Session keyframe kf_id is at least anchor_min_separation from every keyframe position in `others`."""
        sep = self.config.mapping.loop_closure.anchor_min_separation
        if sep <= 0:
            return True
        p = self._kf_position(kf_id)
        if p is None:
            return False
        return all(q is None or float(np.linalg.norm(p - q)) >= sep for q in others)

    def _corroborate_anchor(self, valid_keyframes, valid_poses, h0_ok, T_since, n_since) -> None:
        """Unanchored relocalization session: a map reference whose measurement passes the prior test through a pending
        map edge of hypothesis 0 (another observation within anchor_corroborate_window steps, another map keyframe) is
        marked consistent (h0_ok True), so that its edge anchors the session."""
        v = self._lc_verifier
        step = self._processed_frame_num
        pend = self._anchor_pending
        while pend and pend[0][0] < step - self.config.mapping.loop_closure.anchor_corroborate_window:
            pend.popleft()
        for i, kf in enumerate(valid_keyframes):
            if h0_ok[i] is not None or int(kf.id) >= self._session_start_kf_id:
                continue
            here = [self._kf_position(self.last_added_kf_id)]
            for st, map_kf, anchor in pend:
                if st == step or map_kf == int(kf.id) or not self._separated(anchor["kf"], here):
                    continue
                c2 = v.anchored_chi2(anchor, int(kf.id), valid_poses[i], self.last_added_kf_id, T_since, n_since)
                if c2 is not None and c2 <= v.thr:
                    h0_ok[i] = True
                    v.stats["anchor_corroborated"] = v.stats.get("anchor_corroborated", 0) + 1
                    logger.debug(f"anchor corroborated: map kf {int(kf.id)} agrees with map kf {map_kf} of step {st} (chi2 {c2:.1f})")
                    break

    def _contradict_anchor(self, valid_keyframes, valid_poses, h0_ok, T_since, n_since) -> None:
        """Anchored relocalization session: a map reference rejected by the prior test that agrees with at least
        anchor_contradict_min earlier rejected map measurements (distinct observations and map keyframes, within
        anchor_contradict_window steps) outvotes the anchor.  The anchor is dropped and the verdicts of this pass's map
        references are voided (they were tested against the dropped anchor)."""
        v = self._lc_verifier
        lc_cfg = self.config.mapping.loop_closure
        step = self._processed_frame_num
        pend = self._contra_pending
        while pend and pend[0][0] < step - lc_cfg.anchor_contradict_window:
            pend.popleft()
        hm = self.hypothesis_manager
        T_belief = None
        if lc_cfg.anchor_contradict_min_offset > 0 and self.last_added_kf_id in hm.nodes:
            T_belief = pp.SE3(hm.nodes[self.last_added_kf_id].pose_mu[0]).matrix().detach().cpu().numpy().astype(np.float64)
            if T_since is not None:
                T_belief = T_belief @ pp.SE3(T_since).matrix().detach().cpu().numpy().astype(np.float64).reshape(4, 4)
        for i, kf in enumerate(valid_keyframes):
            if h0_ok[i] is not False or int(kf.id) >= self._session_start_kf_id:
                continue
            if T_belief is not None:
                T_impl = pp.SE3(kf.pose_mu[0]).matrix().detach().cpu().numpy().astype(np.float64) \
                    @ pp.SE3(valid_poses[i]).matrix().detach().cpu().numpy().astype(np.float64).reshape(4, 4)
                if float(np.linalg.norm(T_impl[:3, 3] - T_belief[:3, 3])) < lc_cfg.anchor_contradict_min_offset:
                    continue
            steps, maps = set(), set()
            places = [self._kf_position(self.last_added_kf_id)]   # agreeing measurements must come from separated places
            for st, map_kf, cand in reversed(pend):
                if st == step or map_kf == int(kf.id) or st in steps or map_kf in maps or not self._separated(cand["kf"], places):
                    continue
                c2 = v.anchored_chi2(cand, int(kf.id), valid_poses[i], self.last_added_kf_id, T_since, n_since)
                if c2 is not None and c2 <= v.thr:
                    steps.add(st); maps.add(map_kf); places.append(self._kf_position(cand["kf"]))
            if len(steps) >= lc_cfg.anchor_contradict_min:
                v.anchor = None
                v.stats["anchor_contradicted"] = v.stats.get("anchor_contradicted", 0) + 1
                logger.info(f"session anchor contradicted: map kf {int(kf.id)} agrees with {len(steps)} earlier rejected map "
                            f"measurements (steps {sorted(steps)}); anchor dropped")
                pend.clear()
                self._anchor_pending.clear()
                for j, kj in enumerate(valid_keyframes):
                    if int(kj.id) < self._session_start_kf_id:
                        h0_ok[j] = None
                return

    def _maybe_verified_loop_closure(self, ret: dict, new_kf, edge_mapping: dict) -> bool:
        """Verified loop closure of hypothesis 0 (cross/core/lc_verify.py).

        The visual edges added to hypothesis 0 in this step have passed the in-pass and prior consistency tests.
        If any of them is inconsistent with the poses the graph currently holds (residual beyond the calibrated
        noise), the graph is optimised with the calibrated noise model; new edges that remain outliers afterwards are
        quarantined and the graph is re-optimised.  Returns True when an optimisation was run."""
        v = self._lc_verifier
        if v is None or new_kf is None or ret is None or not self.use_odometry:
            return False
        hm = self.hypothesis_manager
        if getattr(hm, "no_pgo_for_lc", False):
            return False
        new_keys, significant = [], False
        loop_flags = ret.get("h0_loop")
        lc_cfg = self.config.mapping.loop_closure
        for i, kf in enumerate(ret.get("valid_keyframes", [])):
            m = edge_mapping.get(kf.id)
            if m is None or m[1] != 0:
                continue
            bucket = hm.hypotheses[0].visual_edges.get((kf.id, new_kf.id), [])
            if not bucket:
                continue
            new_keys.append((kf.id, new_kf.id))
            # temporal corroboration of loop candidates: a revisit only constrains the graph once a second loop
            # measurement (another reference, within corroborate_window observations) implies the same correction of the
            # current pose.  Until then the edge stays in the graph as non-informative (skipped by the optimisation).
            is_loop = loop_flags is not None and i < len(loop_flags) and bool(loop_flags[i])
            is_map_ref = kf.id < self._session_start_kf_id
            # loop candidates of this session, and (with map_edges_trigger) edges to the stored map, need corroboration
            # from another observation before they may constrain / optimise the graph
            if lc_cfg.corroborate_window > 0 and ((is_loop and not is_map_ref) or (is_map_ref and lc_cfg.map_edges_trigger)):
                if not self._corroborate_loop(kf.id, new_kf, bucket[-1]):
                    continue
            # only loop-closure candidates (measurement more informative than the graph's prediction) can trigger
            # an optimisation; the scatter of the measurements to the keyframes just behind the robot cannot.  An edge to
            # a keyframe of a previous session (fixed map) can, when map_edges_trigger is set: its residual against the
            # session graph measures the session's drift relative to the map (the verifier's session anchor makes it
            # look predictable, but the anchor is itself only an earlier map edge plus odometry)
            is_map_edge = self.config.mapping.loop_closure.map_edges_trigger and kf.id < self._session_start_kf_id
            if loop_flags is not None and i < len(loop_flags) and not loop_flags[i] and not is_map_edge:
                continue
            c2 = v.graph_residual_chi2(kf.id, new_kf.id, bucket[-1])
            if c2 is not None and c2 > v.thr:
                significant = True
        if not significant:
            return False
        step = self._processed_frame_num
        t_pgo = time.perf_counter()
        # test-before-apply: the candidate solution is only written back when its new edges pass the posterior test;
        # otherwise the outliers are quarantined and the graph is re-solved from the *unmodified* poses (re-solving from
        # a solution already distorted by a wrong loop edge left the map in a bad minimum: jumps of 5-11 m on long surveys)
        defer = bool(self.config.mapping.loop_closure.test_before_apply) and self.config.mapping.loop_closure.use_posterior \
            and getattr(self.config.mapping.loop_closure, "posterior_action", "remove") == "remove"
        pgo_info = hm.handle_loop_closure(0, apply=not defer)
        if not pgo_info.get("success"):
            logger.warning(pgo_info.get("message", "verified loop-closure PGO failed without message"))
            return False
        v.stats["pgo"] += 1
        v.stats["pgo_time"] = v.stats.get("pgo_time", 0.0) + (time.perf_counter() - t_pgo)
        self._last_pgo_step = step
        applied = not defer
        if self.config.mapping.loop_closure.use_posterior:
            outliers = v.posterior_outliers(pgo_info["pose_graph"], only_keys=set(new_keys))
            if outliers:
                v.stats["posterior_flagged"] = v.stats.get("posterior_flagged", 0) + len(outliers)
                if getattr(self.config.mapping.loop_closure, "posterior_action", "remove") == "remove":
                    for (a, b, f, c2) in outliers:
                        v.remove_edge(a, b, f)
                    logger.info(f"Verified loop closure at step {step}: {len(outliers)} of {len(new_keys)} new edges rejected after the "
                                f"optimisation (chi2 {[round(o[3], 1) for o in outliers][:6]}, edges {[(o[0], o[1]) for o in outliers][:6]}) -> re-optimising")
                    if defer and len(outliers) == len(new_keys):
                        # every new edge was an outlier: the graph is unchanged, nothing to re-solve or apply
                        return False
                    pgo_info = hm.handle_loop_closure(0)
                    applied = True
                else:
                    logger.info(f"Verified loop closure at step {step}: {len(outliers)} of {len(new_keys)} new edges remain outliers after the "
                                f"optimisation (chi2 {[round(o[3], 1) for o in outliers][:6]}); kept (robust optimisation)")
        if not applied:
            hm.apply_pgo_result({"success": True, "pose_graph": pgo_info.get("pose_graph"),
                                 "optimized_poses": pgo_info.get("optimized_poses", {}), "other_hypothesis_id": 0})
        pg_ = pgo_info.get("pose_graph"); nf = getattr(pg_, "n_factors", {})
        logger.info(f"Verified loop closure at step {step}: {len(new_keys)} new hypothesis-0 edges, PGO cost {pgo_info.get('cost')} "
                    f"(initial {getattr(pg_, 'initial_cost', None)}, {getattr(pg_, 'lm_iterations', None)} LM iterations; "
                    f"{time.perf_counter() - t_pgo:.2f} s, {len(hm.nodes)} keyframes, {len(getattr(pg_, 'vertices', []))} vertices, factors {nf})")
        ret["loop_closure_pgo"] = pgo_info
        ret["verified_loop_closure"] = True
        return True

    def _corroborate_loop(self, ref_id: int, new_kf, factor) -> bool:
        """True when the loop edge ref_id -> new_kf is corroborated by a pending loop edge from another reference whose
        implied correction of the current pose agrees (translation / rotation tolerance); both edges then become
        informative.  An uncorroborated edge is marked non-informative and kept pending for corroborate_window steps."""
        cfg = self.config.mapping.loop_closure
        hm = self.hypothesis_manager
        step = self._processed_frame_num
        pend = getattr(self, "_loop_pending", None)
        if pend is None:
            pend = self._loop_pending = deque()
        while pend and pend[0][0] < step - cfg.corroborate_window:
            pend.popleft()
        try:
            T_ref = pp.SE3(hm.nodes[ref_id].pose_mu[0]).matrix().detach().cpu().numpy().astype(np.float64)
            T_cur = pp.SE3(new_kf.pose_mu[0]).matrix().detach().cpu().numpy().astype(np.float64)
            T_meas = pp.SE3(factor.mean).matrix().detach().cpu().numpy().astype(np.float64).reshape(4, 4)
        except Exception:
            return True
        # implied pose of the current keyframe from this loop edge, as a correction in the world frame
        C = (T_ref @ T_meas) @ np.linalg.inv(T_cur)
        ok = False
        for (st, rid, f_old, C_old) in pend:
            # corroboration must come from another observation: references of one forward pass share its (possibly
            # wrong) geometry and agree with each other by construction
            if rid == ref_id or st == step:
                continue
            dC = np.linalg.inv(C_old) @ C
            dt = float(np.linalg.norm(dC[:3, 3]))
            dr = float(np.degrees(np.arccos(np.clip((np.trace(dC[:3, :3]) - 1) / 2, -1, 1))))
            if dt < cfg.corroborate_tol_t and dr < cfg.corroborate_tol_r_deg:
                f_old.informative = True
                ok = True
        if ok:
            factor.informative = True
            v = self._lc_verifier
            if v is not None:
                v.stats["loops_corroborated"] = v.stats.get("loops_corroborated", 0) + 1
            return True
        factor.informative = False
        pend.append((step, ref_id, factor, C))
        return False

    def _maybe_intra_hypothesis_loop_closure(self, ret: dict, new_kf, edge_mapping: dict) -> bool:
        """Pose-graph optimisation of hypothesis 0 triggered by long-range visual edges.

        Counts the visual edges of the keyframe added this step that connect hypothesis 0 to keyframes of the
        current session created before the retrieval recency window (`RetrievalConfig.recent_window_steps`),
        i.e. edges that close a loop inside the tracked hypothesis.  When enough of them accumulate within
        `intra_window_steps`, the whole hypothesis-0 graph is optimised (`HypothesisManager.handle_loop_closure(0)`:
        odometry chain + visual edges, earliest keyframe fixed, map keyframes fixed after a map load) and the
        tracking distribution is re-aligned to the optimised keyframe.  Returns True when a PGO was run.
        """
        cfg = self.config.mapping.loop_closure
        if not cfg.intra_enabled or new_kf is None or ret is None or not self.use_odometry:
            return False
        hm = self.hypothesis_manager
        if getattr(hm, "no_pgo_for_lc", False):
            return False
        step = self._processed_frame_num
        if step - getattr(self, "_last_merge_step", -10**9) < cfg.intra_after_merge_cooldown_steps:
            return False
        conf = ret.get("pose_est_conf")
        rels = ret.get("valid_poses")
        stds = ret.get("valid_stds")
        n_long, res_t, res_r, res_sig, oldest = 0, 0.0, 0.0, 0.0, None
        new_pose = new_kf.pose_mu[0]
        for i, kf in enumerate(ret.get("valid_keyframes", [])):
            if kf.id not in edge_mapping:
                continue
            src_comp, dst_comp = edge_mapping[kf.id]
            if src_comp != 0 or dst_comp != 0:
                continue
            if kf.id < self._session_start_kf_id:
                continue        # map keyframe of a previous session: the session is anchored to it, not a loop
            age = step - int(getattr(kf, "step_created", -10**9))
            if age < cfg.intra_min_loop_steps:
                continue        # keyframe just behind the robot
            if conf is not None and float(conf[i]) < cfg.intra_min_conf:
                continue
            # residual of the measurement against the relative pose the graph currently implies
            pred = kf.pose_mu[0].to(new_pose.device).Inv() @ new_pose
            r = (pred.Inv() @ rels[i].to(new_pose.device)).Log().tensor()
            res_t = max(res_t, float(torch.norm(r[:3])))
            res_r = max(res_r, float(torch.norm(r[3:])))
            if stds is not None:
                sd = stds.tensor()[i].to(r.device).clamp(min=1e-3)
                res_sig = max(res_sig, float(torch.norm(r / sd)) / math.sqrt(6.0))
            oldest = kf.id if oldest is None else min(oldest, kf.id)
            n_long += 1
        if n_long > 0:
            self._intra_lc_events.append((step, n_long, res_t, res_r, oldest, res_sig))
        while self._intra_lc_events and self._intra_lc_events[0][0] < step - cfg.intra_window_steps:
            self._intra_lc_events.popleft()
        total = sum(e[1] for e in self._intra_lc_events)
        if total < cfg.intra_min_edges or step - self._last_intra_pgo_step < cfg.intra_cooldown_steps:
            return False
        max_t = max(e[2] for e in self._intra_lc_events)
        max_r = math.degrees(max(e[3] for e in self._intra_lc_events))
        max_sig = max(e[5] for e in self._intra_lc_events)
        if (max_t < cfg.intra_min_residual_t and max_r < cfg.intra_min_residual_r_deg) or max_sig < cfg.intra_min_residual_sigma:
            return False        # the revisit is already consistent with the graph (absolutely, or within the edge noise)
        logger.info(f"Intra-hypothesis loop closure at step {step}: {total} long-range visual edges (oldest keyframe "
                    f"{min(e[4] for e in self._intra_lc_events)}) in the last {cfg.intra_window_steps} steps, "
                    f"max residual {max_t:.2f} m / {max_r:.1f} deg ({max_sig:.1f} sigma) -> PGO of hypothesis 0")
        self._last_intra_pgo_step = step
        self._intra_lc_events.clear()
        pgo_info = hm.handle_loop_closure(0)
        if pgo_info.get("success"):
            ret["loop_closure_pgo"] = pgo_info
            ret["intra_loop_closure"] = True
            return True
        logger.warning(pgo_info.get("message", "intra-hypothesis PGO failed without message"))
        return False

    def _should_skip_observation(self) -> bool:
        """Observation cadence (pose_est.obs_*): skip retrieval + relative pose estimation at this step?"""
        cfg = self.config.pose_est
        if not self.use_odometry or (cfg.obs_min_translation <= 0 and cfg.obs_min_rotation <= 0 and cfg.obs_max_interval_steps <= 1):
            return False
        self._steps_since_obs += 1
        if self._processed_frame_num - self._session_start_frame <= cfg.obs_warmup_steps:
            return False   # warm-up after (re)initialisation: observe every frame to relocalize quickly
        max_interval, min_t, min_r = cfg.obs_max_interval_steps, cfg.obs_min_translation, cfg.obs_min_rotation
        if cfg.obs_confident_max_interval_steps > 0 and self._last_obs_mapped:
            dist = self.hypothesis_manager.dist
            if dist is not None and dist[2].numel() > 0 and float(dist[2].max()) >= cfg.obs_confident_weight:
                max_interval = cfg.obs_confident_max_interval_steps
                min_t, min_r = cfg.obs_confident_min_translation, cfg.obs_confident_min_rotation
        if self._steps_since_obs >= max_interval:
            return False
        delta_pose, _ = self.odom_accumulator.get_since_last_reading("since_last_obs", reset=False, return_std=False)
        if delta_pose is None:
            return False
        moved = min_t > 0 and bool(torch.norm(delta_pose.tensor()[:3]) >= min_t)
        rotated = min_r > 0 and bool(rotation_angle_from_quat(delta_pose.tensor()[3:]) >= min_r)
        return not (moved or rotated)

    def _construct_motion_dist(
        self,
    ):
        """Construct the motion distribution.
        """
        ret = {"kidnapped": False}
        if self.use_odometry:
            ret['motion_source_factor'] = self.odom_accumulator.source_since_last_reading('since_last_step')
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
            pg = PoseGraph(self.hypothesis_manager, depth=window_kfs, k_hop=k_hop, device=self.state_device)
            pg.construct_for_local_smoothing(target_node_id=target_node_id, window_kfs=window_kfs, k_hop=k_hop)

        if not pg.vertices or not pg.edges:
            return False

        # Fix earliest original KF in window
        original_kf_ids = [v.id for v in pg.vertices if v.id in self.hypothesis_manager.nodes]
        if not original_kf_ids:
            return False
        fixed_node_id = pg.preferred_fixed_node if self.hypothesis_manager.chart_aware else min(original_kf_ids)
        optim_node_ids = set([v.id for v in pg.vertices]) - {fixed_node_id}

        pg.solve(optim_node_ids=optim_node_ids, fixed_node_ids={fixed_node_id})

        if self.hypothesis_manager.source_states is not None:
            applied = self.hypothesis_manager.apply_pgo_result(dict(pose_graph=pg,optimized_poses=pg.optimized_poses,
                                                                   other_hypothesis_id=0))
            self._last_smoothing_step = self._processed_frame_num
            return applied['success']

        # Apply smoothing updates to hypothesis 0
        std_reduction = self.config.pgo.std_reduction_factor
        self.hypothesis_manager.pose_epoch += 1
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
        rgb_right: torch.Tensor = None,
    ):
        """Add a new keyframe to the database.
        We constantly add kfs as robot moves.
        Kf is permanent if the the new kf is not too similar to the previous kf.
        Otherwise, it's virtual and will be merged later.
        """
        # force true => add permanent kf
        is_temp_kf = False if force_permanent else True
        delta_pose, std = self.odom_accumulator.get_since_last_reading("since_last_add_kf", reset=False)
        conditional_motion = None
        if self.hypothesis_manager.source_states is not None:
            from cross.core.conditional_pose import ConditionalPose
            factor = self.odom_accumulator.source_since_last_reading('since_last_add_kf')
            conditional_motion = ConditionalPose(np.diag(std.tensor().double().cpu().numpy()**2),factor)

        # if the robot moves too little, skip adding a kf
        if not force_add and rotation_angle_from_quat(delta_pose.tensor()[3:]) < 0.1 and \
            torch.norm(delta_pose.tensor()[:3]) < 0.2:
            return None

        if ret is not None:
            ret_weights = ret["valid_retrieval_weights"]
            confidence = ret["pose_est_conf"]

            # if current obs is not similar to other kfs, add a permanent kf
            if ret_weights.max() < self.kf_retrieval_threshold_new_kf or \
                (self.pose_est_type == PoseEstType.PNP and confidence.max() < self.kf_match_threshold_new_kf) or \
                (self.pose_est_type == PoseEstType.FF and confidence.max() < self.config.pose_est.ff.kf_conf_threshold_new_kf):
                is_temp_kf = False

        mu, sigma, weights = self.hypothesis_manager.get_active_dist()
        pose_charts = self.hypothesis_manager.get_active_charts()

        if not is_temp_kf:
            if weights.nonzero().numel() == 0:
                return None
            # insert kf into the database for permanent kf
            keyframe = self.db.insert(
                self._processed_frame_num,
                rgb_image.to(self.storage_device), 
                depth_image.to(self.storage_device) if depth_image is not None else None, 
                mu=mu.to(self.state_device), 
                sigma=sigma.to(self.state_device), 
                weights=weights.to(self.state_device),    
                atlas=self.current_atlas,
                timestamp=timestamp,
                temporary=is_temp_kf,
                raw_rgb_right=rgb_right.to(self.storage_device) if (rgb_right is not None and self._store_right_images) else None,
                pose_charts=pose_charts,
                metric_source=getattr(self, "_current_metric_source", None),
            )
            logger.debug(f"Add permanent keyframe at step {self._processed_frame_num}. Total keyframes: {self.db.get_size()}")
        else:
            # otherwise create a temporary keyframe
            keyframe = Keyframe(
                pose_mu=mu.to(self.state_device),
                pose_std=sigma.to(self.state_device),
                pose_weights=weights.to(self.state_device),
                timestamp=timestamp,
                temporary=is_temp_kf,
                pose_charts=pose_charts,
                metric_source=getattr(self, "_current_metric_source", None),
            )
            logger.debug(f"Add temporary keyframe at step {self._processed_frame_num}. Total keyframes: {self.db.get_size()}")
        
        keyframe.step_created = int(self._processed_frame_num)
        self.hypothesis_manager.add_node(keyframe)

        # ---- Add new relative pose measurements to the graph ----
        current_kf_id = keyframe.id
        if ret is not None:
            # The results from pose estimation are our new edges
            valid_keyframes = ret["valid_keyframes"]
            
            # adding visual constraints
            confs = ret.get("pose_est_conf")
            h0_ok = ret.get("h0_ok")
            loop_flags = ret.get("h0_loop")
            logger.debug(f"kf {current_kf_id} edges from {[int(k.id) for k in valid_keyframes]} loop flags {loop_flags} mapping {[int(k.id) in edge_mapping for k in valid_keyframes]}")
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
                    conditional_pose=ret.get('relative_conditional_poses',[None]*len(valid_keyframes))[i],
                    meta={"conf": float(confs[i]) if confs is not None else None,
                          "noise_scale": float(self._lc_verifier.scale_for(kf_i.id)) if self._lc_verifier is not None else 1.0,
                          # anisotropic noise model: the scale along the measured bearing (None: isotropic model)
                          "noise_scale_along": self._lc_verifier.scale_along_for(kf_i.id) if self._lc_verifier is not None else None,
                          "noise_scale_rot": self._lc_verifier.scale_rot_for(kf_i.id) if self._lc_verifier is not None else None,
                          # information criterion: the measurement constrains the graph only if the odometry chain
                          # did not already know the relative pose better (loop candidates, session->map edges)
                          "informative": (bool(loop_flags[i]) if (loop_flags is not None and i < len(loop_flags)) else True)
                                         or (self.config.mapping.hypothesis.map_refs_informative not in (False, "off", None)
                                             and kf_i.id < self._session_start_kf_id),
                          "scale_corr": float(self._lc_verifier.metric_correction) if self._lc_verifier is not None else 1.0},
                )
                # a map edge anchors the session to the map (prior test of later references) when it passed the prior
                # test, or, without a usable prior, when another map reference of the same pass corroborates it; a lone
                # untested edge never anchors (a wrong first anchor made the prior test reject the true measurements
                # that followed)
                if self._lc_verifier is not None and dest_comp_id == 0 and kf_i.id < self._session_start_kf_id:
                    verdict = None if h0_ok is None else h0_ok[i]
                    if verdict is True or (verdict is None and self._lc_verifier.corroborated(kf_i.id)):
                        self._lc_verifier.update_anchor(kf_i.id, current_kf_id, ret["valid_poses"][i])
                        self._anchor_pending.clear()
                        self.note_anchor("map_edge")
                    elif verdict is None and self._lc_verifier.anchor is None and self.config.mapping.loop_closure.anchor_corroborate_window > 0:
                        # candidate anchor: a later map measurement of another observation may corroborate it
                        self._anchor_pending.append((self._processed_frame_num, int(kf_i.id),
                                                     self._lc_verifier.make_anchor(kf_i.id, current_kf_id, ret["valid_poses"][i])))
                # a map measurement rejected by the prior test (whichever hypothesis it feeds) may, with others, contradict
                # the session anchor
                if (self._lc_verifier is not None and kf_i.id < self._session_start_kf_id and h0_ok is not None
                        and h0_ok[i] is False and self._lc_verifier.anchor is not None
                        and self.config.mapping.loop_closure.anchor_contradict_min > 0):
                    self._contra_pending.append((self._processed_frame_num, int(kf_i.id),
                                                 self._lc_verifier.make_anchor(kf_i.id, current_kf_id, ret["valid_poses"][i])))

        # adding odometry constraints
        if delta_pose is not None:
            last_kf = self.hypothesis_manager.nodes.get(self.last_added_kf_id)
            last_step = getattr(last_kf, "step_created", None) if last_kf is not None else None
            meta = {"n_frames": max(int(self._processed_frame_num) - int(last_step), 1) if last_step is not None else None}
            v = getattr(self, "_lc_verifier", None)
            if v is not None and getattr(v, "odom_fault", 0.0) > 0:   # the odometry scale guard has measured an odometry fault
                meta["odom_fault"] = v.odom_fault
            self.hypothesis_manager.add_edge(
                id1=self.last_added_kf_id, # since odom edge is from last added kf to current kf
                id2=current_kf_id,
                rel_pose_mean=delta_pose,
                rel_pose_std=std,
                type=EdgeType.ODOMETRY,
                conditional_pose=conditional_motion,
                meta=meta,
            )
        self.odom_accumulator.reset_item("since_last_add_kf")
        self.last_added_kf_id = current_kf_id

        return keyframe

    def _make_place_projection(self, vertical):
        """SE3Projection for a vertical, or None for the original projection (vertical y, no vertical coordinate)."""
        pcfg = self.config.mapping.projection
        is_y = bool(torch.equal(vertical, vertical_vector("y")))
        if is_y and not pcfg.estimate_vertical and pcfg.vertical_weight <= 0:
            return None
        return SE3Projection(vertical, pcfg.vertical_weight)

    def _update_place_projection(self):
        """Re-estimate the vertical from the keyframe orientations (every 10 new keyframes; the loaded map's keyframes
        in a relocalization session)."""
        if not self.config.mapping.projection.estimate_vertical:
            return
        nodes = self.hypothesis_manager.nodes
        ids = list(getattr(self, "loaded_node_ids", None) or nodes.keys())
        if self._projection_n_nodes >= 0 and len(ids) - self._projection_n_nodes < 10:
            return
        self._projection_n_nodes = len(ids)
        poses = [nodes[i].pose_mu[0].tensor().detach().reshape(-1).cpu() for i in ids if i in nodes]
        if len(poses) < 3:
            return
        R = pp.SE3(torch.stack(poses)).rotation().matrix()
        v, estimated = estimate_vertical(R, self._projection_prior)
        self.place_projection = SE3Projection(v, self.config.mapping.projection.vertical_weight)
        logger.debug(f"place projection from {len(poses)} keyframes ({'estimated' if estimated else 'prior'}): "
                     f"{self.place_projection}")

    def vertical_offsets(self, T: pp.LieTensor) -> torch.Tensor:
        """Vertical position of a batch of poses along the place projection's vertical (camera y by default)."""
        if self.place_projection is not None:
            return self.place_projection.vertical_offset(T)
        return T.tensor()[:, 1]

    @timeit
    def _merge_and_align_components(
        self,
        ret: dict,
        dbscan_eps: float = None,
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
        if dbscan_eps is None:
            dbscan_eps = self.config.mapping.cluster_eps

        convolved_mus = ret["convolved_mus"].to(self.state_device)  # (B, K, 7)
        convolved_stds = ret["convolved_stds"].to(self.state_device)  # (B, K, 6)
        confidences = ret['pose_est_conf'].to(self.state_device)  # (B)
        component_weights = ret["valid_ref_component_weights"].to(self.state_device)  # (B, K)
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
        kf_ids = torch.tensor([kf.id for kf in ret['valid_keyframes']], device=self.state_device)
        source_indices_map[:, 0] = kf_ids[source_indices_map[:, 0]]
        

        # --- 1. Project to Clustering Space (place coordinates: horizontal position, [vertical], heading) ---
        samples_se3 = pp.SE3(flat_mus)
        self._update_place_projection()
        projected_data = project_SE3(samples_se3, projection=self.place_projection)
        clustering_data = projected_data.cpu().numpy()

        
        # --- 2. Run DBSCAN ---
        if self.hypothesis_manager.chart_aware:
            from cross.core.charts import cluster_by_chart
            charts = torch.stack([kf.pose_charts for kf in ret['valid_keyframes']]).to(self.state_device)
            flat_charts = charts[valid_mask].cpu().numpy()
            labels = cluster_by_chart(clustering_data, flat_charts, dbscan_eps, dbscan_min_samples)
        else:
            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(clustering_data)
            labels = db.labels_
        gate = self.config.mapping.projection.vertical_gate
        if gate > 0:
            labels = split_clusters_by_vertical(labels, self.vertical_offsets(samples_se3).cpu().numpy(),
                                                flat_scores.detach().cpu().numpy(), gate)
        # a reference the prior test found inconsistent with hypothesis 0 never shares a proposal with consistent ones:
        # a mixed cluster counted as consistent (any member) and informative (any member), so the inconsistent member's
        # edge was filed under hypothesis 0 as a loop edge (OpenLORIS corridor1-1, RGB-D: 289 of 290 hypothesis-0 loop
        # edges joined keyframes 35-75 m apart and bent the map by 23 deg)
        h0_ok_all = ret.get("h0_ok")
        if h0_ok_all is not None and getattr(self.config.mapping.loop_closure, "pure_verdict_clusters", False):
            labels = np.asarray(labels).copy()
            flat_b = valid_indices_tuple[0].tolist()
            bad = np.array([h0_ok_all[int(b)] is False for b in flat_b], dtype=bool)
            next_label = int(labels.max()) + 1 if len(labels) else 0
            for lab in sorted(set(labels.tolist())):
                if lab == -1:
                    continue
                m = labels == lab
                if bad[m].any() and (~bad[m]).any():
                    labels[m & bad] = next_label
                    next_label += 1
        unique_labels = sorted(set(labels))
        
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
                if cluster_std_cfg.floor_by_member_std:
                    member_min = torch.min(torch.as_tensor(flat_stds[cluster_indices]).reshape(len(cluster_indices), 6), dim=0).values
                    std = torch.maximum(std, member_min.to(std.device, std.dtype))
                representative_std = pp.se3(std.unsqueeze(0)).squeeze(0)

                logger.debug(
                    f"Cluster {k}: size={len(cluster_indices)}, use_conf_weight={use_conf_weight}, std={std.tolist()}"
                )

            hypothesis_score = flat_scores[cluster_indices].sum()

            cluster_sources = source_indices_map[cluster_indices]

            # prior-consistency verdict of the cluster w.r.t. hypothesis 0 (verified loop closure): True if any source
            # reference is consistent, False if all tested sources are inconsistent, None if untested
            h0_flag = None
            h0_ok = ret.get("h0_ok")
            if h0_ok is not None:
                verdicts = [h0_ok[int(b)] for b in valid_indices_tuple[0][cluster_indices].tolist()]
                if any(v is True for v in verdicts):
                    h0_flag = True
                elif verdicts and all(v is False for v in verdicts):
                    h0_flag = False

            # informative cluster: at least one reference whose relative pose the odometry chain knows less well than
            # the measurement (verified-LC loop flag; references without a chain, e.g. map keyframes of a relocalization
            # session, count as informative)
            loops = ret.get("h0_loop")
            if loops is None:
                informative = True
            else:
                bs = valid_indices_tuple[0][cluster_indices].tolist()
                informative = any((int(b) >= len(loops)) or loops[int(b)] is not False for b in bs)
            # keyframes of previous sessions are fixed and independent of this session's odometry drift: a measurement
            # to them is informative about hypothesis 0 even when the verifier's session anchor predicts it (the anchor is
            # itself only the previous map measurement plus odometry); without this, a relocalized session drifted away
            # from the map with its odometry (0.06 m after relocalization -> 0.7 m in a real-data session)
            mri = self.config.mapping.hypothesis.map_refs_informative
            if not informative and mri and mri != "off":
                srcs = [int(x) for x in cluster_sources[:, 0].tolist()]
                if any(k < self._session_start_kf_id for k in srcs):
                    if mri == "belief":
                        # informative when hypothesis 0 is less certain (translation) than this measurement: fused map
                        # measurements shrink the belief, so correlated follow-ups are skipped until the odometry has
                        # grown the uncertainty again (bounds the drift without re-applying one bias every frame)
                        prior_var_t = float((self.hypothesis_manager.dist[1][0].tensor()[:3] ** 2).sum())
                        meas_var_t = float((torch.as_tensor(representative_std.tensor() if hasattr(representative_std, "tensor") else representative_std)[:3] ** 2).sum())
                        informative = prior_var_t > meas_var_t
                    else:
                        informative = True

            hypotheses.append({
                'pose': representative_pose, # cluster representative pose
                'std': representative_std, # std from cluster dispersion (se(3))
                'score': hypothesis_score, # cluster score
                'source_indices': cluster_sources.tolist(), # M, 2
                'h0_ok': h0_flag,
                'informative': informative,
            })
            if self.hypothesis_manager.chart_aware:
                hypotheses[-1]['chart_id'] = int(flat_charts[best_candidate_in_cluster_idx])
            if self.hypothesis_manager.source_states is not None:
                from cross.core.conditional_pose import ConditionalPose
                b,c = [int(v[best_candidate_in_cluster_idx]) for v in valid_indices_tuple]
                message = ret['convolved_conditional_poses'][b][c]
                S = message.geometry_covariance.copy()
                if len(cluster_indices) > 1:
                    # Dispersion does not replace the selected factor's
                    # conditional geometric uncertainty. Nearby duplicates
                    # cannot turn a noisy node/PnP fit into exact geometry.
                    S += np.diag(representative_std.tensor().double().cpu().numpy()**2)
                message = ConditionalPose(S,message.factor)
                hypotheses[-1]['conditional_pose'] = message
                hypotheses[-1]['std'] = pp.se3(torch.as_tensor(S.diagonal().copy(),
                                            device=self.state_device,dtype=representative_std.dtype).clamp_min(0).sqrt())

        # make sure the hypotheses are in the same order as the current state GMM
        return self.hypothesis_manager.align_proposal_prior(hypotheses)

    def note_anchor(self, source: str = "external") -> None:
        """The current pose was just tied to the map (loop closure, verified map edge, an accepted GPS fix): the
        drift term of the locality search radius restarts from zero."""
        self._locality_path = 0.0

    def add_location_prior(self, center, sigma: float, source: str = "external", ttl_steps: int = 1) -> None:
        """A prior on the robot's current position in the map frame (e.g. a GPS fix carried into the map frame by the
        geo anchor).  With RetrievalConfig.locality enabled, retrieval scores the keyframes within
        r_min + k_sigma * sigma of `center` among the locality slots.  Valid for the next `ttl_steps` steps (call it
        before System.step of the frame it belongs to)."""
        self._location_priors.append({"center": np.asarray(center, dtype=np.float64).reshape(3), "sigma": float(sigma),
                                      "source": str(source), "until": self._processed_frame_num + int(ttl_steps)})

    def _locality_regions(self):
        """Search regions (centers (m, 3), radii (m,), sources) of locality-aware retrieval: one per active hypothesis
        (hypothesis 0 with the drift term) and one per valid external prior."""
        lc = self.config.retrieval.locality
        centers, radii, sources = [], [], []
        dist = self.hypothesis_manager.dist
        if dist is not None:
            mu, sigma, w = dist
            T = mu.tensor().detach().reshape(-1, 7).cpu().double().numpy()
            S = sigma.tensor().detach().reshape(-1, 6).cpu().double().numpy()
            W = w.detach().reshape(-1).cpu().double().numpy()
            for k in range(len(W)):
                if W[k] < lc.min_weight and k != 0:
                    continue
                r = lc.r_min + lc.k_sigma * float(np.linalg.norm(S[k, :3]))
                if k == 0:
                    r += lc.drift_rate * self._locality_path
                centers.append(T[k, :3]); radii.append(min(r, lc.r_max)); sources.append(f"h{k}")
        self._location_priors = [p for p in self._location_priors if p["until"] >= self._processed_frame_num]
        for p in self._location_priors:
            centers.append(p["center"]); radii.append(min(lc.r_min + lc.k_sigma * p["sigma"], lc.r_max))
            sources.append(p["source"])
        return np.asarray(centers, dtype=np.float64).reshape(-1, 3), np.asarray(radii, dtype=np.float64), sources

    def _locality_merge(self, rgb_image, ranked, max_kf_id=None, min_kf_id=None, n=None):
        """`ranked` [(score, kf)] (global appearance ranking) -> the best keyframes inside the search regions first
        (up to locality.slots), then the global ranking without them."""
        lc = self.config.retrieval.locality
        if not lc.enabled or self.db.get_size() < max(lc.min_keyframes, 1):
            return ranked
        centers, radii, sources = self._locality_regions()
        info = {"regions": len(radii), "radii": [round(float(r), 1) for r in radii], "sources": sources,
                "candidates": 0, "local": 0}
        self.last_locality = info
        if not len(radii):
            return ranked
        rows = self.db.rows_near(centers, radii, epoch=self.hypothesis_manager.pose_epoch)
        if max_kf_id is not None or min_kf_id is not None:
            ids = self.db.index.ids[torch.as_tensor(rows, dtype=torch.long, device=self.db.device)].cpu().numpy() if len(rows) else rows
            keep = np.ones(len(rows), dtype=bool)
            if max_kf_id is not None:
                keep &= ids < max_kf_id
            if min_kf_id is not None:
                keep &= ids >= min_kf_id
            rows = rows[keep]
        info["candidates"] = int(len(rows))
        if not len(rows):
            return ranked
        local = self.db.query(rgb_image, rows=rows, top_k=lc.slots,
                              score_threshold=lc.score_threshold if lc.score_threshold is not None
                              else (self.config.retrieval.map_score_threshold if max_kf_id is not None else None))
        take = list(zip(local["scores"], local["keyframes"]))[:lc.slots]
        info["local"] = len(take)
        ids = {kf.id for _, kf in take}
        return take + [(sc, kf) for sc, kf in ranked if kf.id not in ids]

    @timeit
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
            if self.config.retrieval.historical_slots > 0:
                results = self.db.query(rgb_image, reserved_keyframe_ids=self.loaded_node_ids,
                                        reserved_count=self.config.retrieval.historical_slots,
                                        reserved_min_score=self.config.retrieval.historical_min_score)
            else:
                results = self.db.query(rgb_image)
                if self._session_start_kf_id > 0:
                    # session-aware retrieval: guarantee map keyframes (previous sessions) most of the budget
                    own_slots = self.config.retrieval.own_session_slots
                    map_res = self.db.query(rgb_image, max_kf_id=self._session_start_kf_id,
                                            score_threshold=self.config.retrieval.map_score_threshold)
                    logger.debug(f"retrieval: map kfs {[(kf.id, round(sc, 3)) for sc, kf in zip(map_res['scores'], map_res['keyframes'])][:6]}, "
                                 f"own {[(kf.id, round(sc, 3)) for sc, kf in zip(results['scores'], results['keyframes']) if kf.id >= self._session_start_kf_id][:3]}")
                    own = [(sc, kf) for sc, kf in zip(results["scores"], results["keyframes"]) if kf.id >= self._session_start_kf_id][:own_slots]
                    n_map = max(self.db.top_k - len(own), 0)
                    map_list = self._locality_merge(rgb_image, list(zip(map_res["scores"], map_res["keyframes"])),
                                                    max_kf_id=self._session_start_kf_id)
                    guided = self._pose_guided_map_keyframes(rgb_image)
                    if guided:
                        ids = {kf.id for _, kf in guided}
                        map_list = guided + [(sc, kf) for sc, kf in map_list if kf.id not in ids]
                    merged = map_list[:n_map] + own
                    results = {"scores": [m[0] for m in merged], "keyframes": [m[1] for m in merged]}
                elif self.config.retrieval.recent_window_steps > 0 and self._processed_frame_num > self.config.retrieval.recent_window_steps:
                    # mapping: recency-aware split so that revisited places (loop closures) get retrieval budget
                    cut = self.hypothesis_manager.latest_kf_id_before(self._processed_frame_num - self.config.retrieval.recent_window_steps)
                    if cut is not None:
                        rcfg = self.config.retrieval
                        recent_slots = max(rcfg.own_session_slots, rcfg.recent_split_recent_slots)
                        old_res = self.db.query(rgb_image, max_kf_id=cut, score_threshold=rcfg.map_score_threshold)
                        recent = [(sc, kf) for sc, kf in zip(results["scores"], results["keyframes"]) if kf.id > cut][:recent_slots]
                        best = max([float(sc) for sc in results["scores"]] + [0.0])
                        # old keyframes only when they really look like the query: rank-only admission handed the pose
                        # estimator aliased places (repeated arcades / rooms) and spawned false hypotheses
                        min_score = max(rcfg.recent_split_min_score, rcfg.recent_split_rel_score * best)
                        old = [(sc, kf) for sc, kf in zip(old_res["scores"], old_res["keyframes"]) if float(sc) >= min_score]
                        n_old = max(self.db.top_k - len(recent), 0)
                        merged = old[:n_old] + recent
                        if merged:
                            results = {"scores": [m[0] for m in merged], "keyframes": [m[1] for m in merged]}
                        logger.debug(f"retrieval(map): old kfs {[(kf.id, round(sc, 3)) for sc, kf in zip(old_res['scores'], old_res['keyframes'])][:6]}, "
                                     f"recent {[(kf.id, round(sc, 3)) for sc, kf in recent]}")
            if self._session_start_kf_id == 0 and self.config.retrieval.locality.enabled:
                merged = self._locality_merge(rgb_image, list(zip(results["scores"], results["keyframes"])))[:self.db.top_k]
                results = {"scores": [m[0] for m in merged], "keyframes": [m[1] for m in merged]}
            self._last_retrieved_results = results
            self.odom_accumulator.reset_item("since_last_retrieval")
            return results
        else:
            return self._last_retrieved_results
    
    def _pose_guided_map_keyframes(self, rgb_image):
        """Map keyframes whose viewed region overlaps the current one, predicted from the tracked belief (see
        RetrievalConfig.pose_guided_*).  Only in a relocalization session that is anchored to the map."""
        rc = self.config.retrieval
        v = self._lc_verifier
        if rc.pose_guided_k <= 0 or self._session_start_kf_id <= 0 or v is None or v.anchor is None:
            return []
        cache = getattr(self, "_pg_cache", None)
        if cache is None or cache["session"] != self._session_start_kf_id:
            kfs = [kf for kf in self.db.get_all_keyframes() if kf.id < self._session_start_kf_id]
            if not kfs:
                return []
            M = np.stack([pp.SE3(kf.pose_mu[0]).matrix().detach().cpu().numpy() for kf in kfs]).astype(np.float64)
            cache = {"session": self._session_start_kf_id, "kfs": kfs, "p": M[:, :3, 3] + rc.pose_guided_depth * M[:, :3, 2], "z": M[:, :3, 2]}
            self._pg_cache = cache
        Mc = pp.SE3(self.hypothesis_manager.dist[0][0]).matrix().detach().cpu().numpy().astype(np.float64)
        pc = Mc[:3, 3] + rc.pose_guided_depth * Mc[:3, 2]
        d = np.linalg.norm(cache["p"] - pc[None], axis=1)
        cosang = cache["z"] @ Mc[:3, 2]
        ok = (d < rc.pose_guided_radius) & (cosang > np.cos(np.radians(rc.pose_guided_max_angle_deg)))
        idx = np.where(ok)[0][np.argsort(d[ok])][:rc.pose_guided_k]
        if len(idx) == 0:
            return []
        kfs = [cache["kfs"][i] for i in idx]
        sims = self.db.similarities(rgb_image, [kf.id for kf in kfs])
        logger.debug(f"pose-guided retrieval: {[(int(kf.id), round(float(d[i]), 2)) for kf, i in zip(kfs, idx)]}")
        return [(max(sims.get(int(kf.id), 0.3), 0.05), kf) for kf in kfs]

    def _get_std_diag(
        self, 
        confidences: torch.Tensor,
        retrieval_scores: torch.Tensor,
        poses: torch.Tensor = None,
        keyframes: list = None,
        ):
        """Get the std diagonal based on the confidence.
        Args:
            confidences: (B,)
            retrieval_scores: (B,)
            poses: (B, 7) relative poses (used to inflate the translation std with the
                   relative scale uncertainty of the feed-forward estimator)
        Returns:
            std_diag: (B, 6)
        """
        if len(confidences) == 0:
            return []
        if self.config.pose_est.meas_std_from_noise_model and self._lc_verifier is not None and poses is not None:
            v = self._lc_verifier
            d = torch.norm(poses.tensor()[:, :3], dim=1).detach().cpu().numpy()
            rows = []
            for i in range(len(d)):
                sc = v.scale_for(keyframes[i].id) if keyframes is not None else 1.0
                sg = v.noise.visual(float(d[i]), scale=sc)        # gtsam order: r r r t t t
                rows.append([sg[3], sg[4], sg[5], sg[0], sg[1], sg[2]])
            return pp.se3(torch.tensor(rows, dtype=torch.float32))
        
        if self.pose_est_type == PoseEstType.PNP:
            # conf is num of inliers
            conf = confidences / self.config.pose_est.kp_detector.n_keypoints
        elif self.pose_est_type == PoseEstType.FF:
            # conf is a covisibility ratio in [0, 1]
            conf = confidences.clamp(min=1e-2)
        else:
            conf = confidences
        # the median of both confidences and retrieval scores is 0.5
        # so we multiply 4 to make the std roughtly same scale as base_std_diag
        std_diag = self.base_measurement_std_diag.unsqueeze(0) / (4 * conf.unsqueeze(1) * retrieval_scores.unsqueeze(1))
        if self.pose_est_type == PoseEstType.FF and self.config.pose_est.ff.scale_std_inflation and poses is not None:
            rel = self.pose_est.scale_rel_std
            t_norm = torch.norm(poses.tensor()[:, :3].to(std_diag.device), dim=1, keepdim=True)
            std_diag = std_diag.clone()
            std_diag[:, :3] = torch.sqrt(std_diag[:, :3] ** 2 + (rel * t_norm) ** 2)
        return pp.se3(std_diag)
    
    @timeit
    def _map_anchor_pairs(self, keyframes):
        """Pairs of references with their metric relative pose from the map (component-0 means), used as
        long-baseline scale anchors by the feed-forward estimator.  Pairs are chosen by decreasing distance
        within [map_anchor_min_dist, map_anchor_max_dist] (all of them with map_anchor_min_pair_covis)."""
        ffc = self.config.pose_est.ff
        if not ffc.use_map_anchors or len(keyframes) < 2:
            return None
        mats = [np.asarray(pp.SE3(kf.pose_mu[0]).matrix().detach().cpu().numpy(), dtype=np.float64) for kf in keyframes]
        pairs = []
        for i in range(len(mats)):
            for j in range(i + 1, len(mats)):
                T_ij = np.linalg.inv(mats[i]) @ mats[j]
                d = float(np.linalg.norm(T_ij[:3, 3]))
                if ffc.map_anchor_min_dist <= d <= ffc.map_anchor_max_dist:
                    pairs.append((d, i, j, T_ij))
        pairs.sort(key=lambda x: -x[0])
        # with the pair-covisibility filter every pair is a candidate (the estimator keeps the longest that pass)
        n = None if getattr(ffc, "map_anchor_min_pair_covis", 0.0) > 0 else ffc.map_anchor_max_pairs
        return [(i, j, T) for _, i, j, T in pairs[:n]] or None

    def _construct_observation_dist(
        self,
        rgb_image: torch.Tensor,
        depth_image: torch.Tensor = None,
        rgb_right: torch.Tensor = None,
        odom_anchor: dict = None,
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
            "retrieval_audit": [],
        }
        # first retrieve the image
        results = self._retrieve_keyframes(rgb_image) 
        if len(results["scores"]) == 0:
            logger.debug(f"No valid retrieval results found.")
            return ret

        retrieval_scores = results["scores"]
        keyframes = results["keyframes"]
        if self.pose_est_type == PoseEstType.FF:
            # bound the context size of the multi-view forward pass
            max_refs = self.config.pose_est.ff.max_refs
            retrieval_scores = retrieval_scores[:max_refs]
            keyframes = keyframes[:max_refs]
        from cross.db.db import as_float_image
        ref_rgbs = [as_float_image(p.raw_rgb_image) for p in keyframes]
        ref_depths = [as_float_image(p.depth_image) for p in keyframes] if depth_image is not None else None

        # insert VO pose est
        if self._prev_obs is not None and self.use_VO:
            ref_rgbs = [self._prev_obs[0]] + ref_rgbs
            ref_depths = [self._prev_obs[1]] + ref_depths

        ref_rgbs = torch.stack(ref_rgbs)
        ref_depths = torch.cat(ref_depths, dim=0) if ref_depths is not None else None

        # then estimate the relative pose
        source_context = {}
        if getattr(self, "_current_metric_source", None) is not None:
            source_context = dict(ref_metric_sources=[getattr(k, "metric_source", None) for k in keyframes],
                                  curr_metric_source=self._current_metric_source)
        if self.hypothesis_manager.source_states is not None:
            source_context['excluded_factor_ids'] = set().union(*(s.seen_factors for s in self.hypothesis_manager.source_states if s is not None))
        # which references belong to a loaded map, and whether this session is still unjoined to it
        source_context['ref_loaded'] = [k.id in self.loaded_node_ids for k in keyframes]
        source_context['session_unanchored'] = bool(self.hypothesis_manager.reference_support.unanchored)
        if self.pose_est_type == PoseEstType.FF:
            valid_poses, valid_masks, confidences = self.pose_est.estimate_pose(
                ref_rgbs, None, rgb_image, None,
                curr_image_right=rgb_right,
                ref_images_right=[as_float_image(p.raw_rgb_right) for p in keyframes],
                odom_anchor=odom_anchor,
                ref_rel_poses=self._map_anchor_pairs(keyframes),
                **source_context,
            )
        else:
            valid_poses, valid_masks, confidences = self.pose_est.estimate_pose(
                ref_rgbs,
                ref_depths,
                rgb_image,
                depth_image,
                **source_context,
            )
        # Record even failed geometry. Previously the early return erased
        # retrieved IDs whenever no pair survived verification.
        pair_audits = getattr(self.pose_est, "last_pair_audit", None)
        offset = int(self._prev_obs is not None and self.use_VO)
        for i, (keyframe, score) in enumerate(zip(keyframes, retrieval_scores)):
            pair_index = i + offset
            item = dict(keyframe_id=int(keyframe.id), loaded=keyframe.id in self.loaded_node_ids,
                        retrieval_score=float(score), verified=bool(valid_masks[pair_index]))
            if pair_audits is not None and pair_index < len(pair_audits):
                item["geometry"] = pair_audits[pair_index]
            ret["retrieval_audit"].append(item)
        # no valid pose
        if valid_masks.sum() == 0:
            logger.debug(f"No valid pose found.")
            return ret

        valid_keyframes = [k for k, m in zip(keyframes, valid_masks) if m]
        if len(valid_keyframes) == 0:
            logger.debug(f"No valid keyframes found.")
            return ret

        # verified loop closure: prior consistency of every measurement with hypothesis 0 (test 2), then the in-pass
        # consistency of the references of this forward pass (test 1, prior verdicts as tie-break); references dropped
        # by test 1 never become proposals or edges, the verdicts of test 2 steer the proposal alignment
        h0_ok = None
        # (the in-pass test needs the pass-internal relative poses of a multi-view estimator; others get the prior test)
        if self._lc_verifier is not None:
            keep, h0_ok = self._verify_references(valid_keyframes, valid_poses, keyframes, np.asarray(valid_masks, dtype=bool), ret)
            if keep is not None and int(keep.sum()) < len(valid_keyframes):
                sel = torch.as_tensor(keep)
                valid_poses = valid_poses[sel]
                confidences = confidences[sel]
                if h0_ok is not None:
                    h0_ok = [o for o, m in zip(h0_ok, keep) if m]
                if ret.get("h0_loop") is not None and len(ret["h0_loop"]) == len(keep):
                    ret["h0_loop"] = [o for o, m in zip(ret["h0_loop"], keep) if m]
                idx = np.where(np.asarray(valid_masks, dtype=bool))[0]
                vm = np.zeros(len(keyframes), dtype=bool); vm[idx[keep]] = True
                valid_masks = vm
                valid_keyframes = [k for k, m in zip(keyframes, valid_masks) if m]
                if len(valid_keyframes) == 0:
                    logger.debug("No valid keyframes after the consistency tests.")
                    return ret
        ret["h0_ok"] = h0_ok
        # one forward pass registers the current view once: its loop measurements to several (consecutive) keyframes of
        # the session share that registration's error and also carry the model's geometry between those keyframes, so
        # only the nearest loop candidate(s) of a pass (the smallest measured distance: the least noisy measurement under
        # the noise model) constrain the graph
        n_per_pass = int(getattr(self.config.mapping.loop_closure, "loop_edges_per_pass", 0) or 0)
        loop_flags = ret.get("h0_loop")
        if n_per_pass > 0 and loop_flags is not None and len(loop_flags) == len(valid_keyframes):
            cand = [i for i, kf in enumerate(valid_keyframes)
                    if loop_flags[i] and kf.id >= self._session_start_kf_id and (h0_ok is None or h0_ok[i] is not False)]
            if len(cand) > n_per_pass:
                dist = torch.norm(valid_poses.tensor()[:, :3], dim=1).detach().cpu().numpy()
                order = sorted(cand, key=lambda i: float(dist[i]))
                keep = set(order[:n_per_pass])
                ret["h0_loop"] = [bool(f) and (i not in cand or i in keep) for i, f in enumerate(loop_flags)]

        # valid retrieval weights
        retrieval_weights = torch.tensor([w for w, m in zip(retrieval_scores, valid_masks) if m])
        retrieval_weights_normalized = self._normalize_retrieval_weights(retrieval_weights)

        valid_stds = self._get_std_diag(confidences, retrieval_weights, poses=valid_poses, keyframes=valid_keyframes)
        # Monocular relative-pose adapters provide calibrated metric-scale
        # uncertainty. The legacy filter stores diagonal standard deviations.
        supplied_stds = getattr(self.pose_est, "last_stds", None)
        if supplied_stds is not None:
            if supplied_stds.shape != valid_stds.shape:
                raise ValueError("Pose estimator uncertainty does not match valid poses")
            valid_stds = pp.se3(torch.maximum(valid_stds.tensor(), supplied_stds.to(valid_stds.device)))

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

        valid_poses = valid_poses.to(self.state_device)
        valid_stds = valid_stds.to(self.state_device)


        valid_ref_mus = torch.stack([p.pose_mu for p in valid_keyframes], dim=0).to(self.state_device) # (B, K, 7)
        valid_ref_stds = torch.stack([p.pose_std for p in valid_keyframes], dim=0).to(self.state_device) # (B, K, 6)
        valid_ref_component_weights = torch.stack([p.pose_weights for p in valid_keyframes], dim=0).to(self.state_device) # (B, K)

        # convolve the Gaussian with the GMM of the L keyframes
        convolved_mus, convolved_stds = convolve_gmm_batch_SE3(
            normalize_SE3(valid_ref_mus),
            valid_ref_stds,
            valid_poses,
            valid_stds,
            valid_ref_component_weights,
        )
        convolved_mus = normalize_SE3(convolved_mus)

        conditional_ret = {}
        if self.hypothesis_manager.source_states is not None:
            from cross.core.conditional_pose import ConditionalPose, compose
            relative_models = getattr(self.pose_est, 'last_conditional_poses', None)
            if relative_models is None or len(relative_models) != len(valid_keyframes):
                raise ValueError('Conditional retrieval needs one model for every verified relative pose')
            models = []
            for i,(kf,relative) in enumerate(zip(valid_keyframes,relative_models)):
                if kf.conditional_poses is None:
                    raise ValueError('A retrieved node has no conditional pose model; rebuild the reference map')
                # Retain CROSS confidence/retrieval-dependent geometry weights,
                # without reintroducing the old teacher-scale marginal.
                relative = ConditionalPose(np.diag(valid_stds[i].tensor().double().cpu().numpy()**2),relative.factor)
                relative_models[i] = relative
                row = []
                for c,node_model in enumerate(kf.conditional_poses):
                    if valid_ref_component_weights[i,c] <= 1e-3:
                        row.append(None)
                        continue
                    if node_model is None:
                        raise ValueError('An active retrieved component has no conditional pose model')
                    pose,model = compose(kf.pose_mu[c].matrix().double().cpu().numpy(),node_model,
                        valid_poses[i].matrix().double().cpu().numpy(),relative,self.hypothesis_manager.source_states[0])
                    convolved_mus[i,c] = pp.from_matrix(torch.as_tensor(pose,device=self.state_device,dtype=convolved_mus.dtype),pp.SE3_type)
                    convolved_stds[i,c] = pp.se3(torch.as_tensor(model.geometry_covariance.diagonal().copy(),
                                                device=self.state_device,dtype=convolved_stds.dtype).clamp_min(0).sqrt())
                    row.append(model)
                models.append(row)
            conditional_ret = dict(convolved_conditional_poses=models,relative_conditional_poses=relative_models)

        return {
            **conditional_ret,
            # retrieval
            "retrieval_audit": ret["retrieval_audit"],
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
            # verified loop closure: prior verdicts / chi2 / loop-candidate flags per valid keyframe (steer the proposal
            # alignment, the PGO trigger and which measurements constrain the graph)
            "h0_ok": h0_ok,
            "h0_chi2": ret.get("h0_chi2"),
            "h0_loop": ret.get("h0_loop"),
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
