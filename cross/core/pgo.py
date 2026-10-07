import bisect
import collections
import weakref
import numpy as np
import torch
import pypose as pp
import gtsam
from typing import Tuple, List, Dict, Set, Optional
from dataclasses import dataclass
from loguru import logger

from cross.core.types import Edge, EdgeType
from cross.utils.profile import timeit, timeblock


def pypose_to_gtsam_pose3(pose) -> 'gtsam.Pose3':
    """Converts a pypose SE3 LieTensor (or a (7,) array) to a gtsam.Pose3 object."""
    pose_tensor = np.asarray(pose, dtype=np.float64).reshape(-1) if isinstance(pose, np.ndarray) else pose.tensor().detach().cpu().numpy().astype(np.float64).reshape(-1)
    # pypose quat: [qx, qy, qz, qw]; gtsam quat: [w, x, y, z].  The quaternion is renormalised: gtsam builds the
    # rotation matrix assuming a unit quaternion, and a non-unit one gives a scaled, non-orthonormal matrix whose
    # errors compound through the pypose <-> gtsam round trip of every optimisation (|q| reached 1.67 after ~100)
    q = pose_tensor[3:7] / max(float(np.linalg.norm(pose_tensor[3:7])), 1e-12)
    rot = gtsam.Rot3(q[3], q[0], q[1], q[2])
    trans = gtsam.Point3(pose_tensor[0], pose_tensor[1], pose_tensor[2])
    
    return gtsam.Pose3(rot, trans)


def _pypose_row(pose: 'gtsam.Pose3') -> np.ndarray:
    """gtsam.Pose3 -> (7,) float64 [x y z qx qy qz qw] (unit quaternion)."""
    rot_quat = np.asarray(pose.rotation().toQuaternion().coeffs(), dtype=np.float64)  # [x, y, z, w]
    rot_quat = rot_quat / max(float(np.linalg.norm(rot_quat)), 1e-12)
    trans = pose.translation()  # [x, y, z]
    # pypose quat: [qx, qy, qz, qw]
    return np.concatenate([trans, rot_quat])


def gtsam_to_pypose_pose3(pose: 'gtsam.Pose3', device: str) -> pp.LieTensor:
    """Converts a gtsam.Pose3 object to a pypose SE3 LieTensor."""
    return pp.SE3(torch.from_numpy(_pypose_row(pose)).to(dtype=torch.float32, device=device))


# torch-function dispatch of LieTensor subclasses (pypose) costs ~90 us per indexing / assignment; the pose graph
# reads and writes thousands of keyframe rows per optimisation, so those go through plain tensor ops (same values)
_plain_tensor_ops = torch._C.DisableTorchFunctionSubclass


def as_se3(t: torch.Tensor) -> pp.LieTensor:
    """An SE3 LieTensor over the tensor `t` (no copy), as pp.SE3(t) and pypose's own result wrapping build it."""
    lt = torch.Tensor.as_subclass(t, pp.LieTensor)
    lt.ltype = pp.SE3_type
    return lt


# Per-factor caches of the GTSAM measurement and noise model.  Keys are the factor objects (weak: a removed factor
# drops its entries); each entry stores the exact inputs it was computed from and is used only when they match.
_measurement_cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_noise_cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


class Vertex:
    """Represents a vertex (node) in the pose graph.

    `pose` / `std` are the rows of the keyframe's belief (`pose_mu[comp]`, `pose_std[comp]`, views as before); they
    are materialised on first access from the tensors captured at construction (`src`), since most vertices of a
    large graph are only read numerically by PoseGraph.solve (`pose_row`)."""

    def __init__(self, id: int, pose: Optional[pp.LieTensor] = None, std: Optional[pp.LieTensor] = None,
                 original_kf_id: int = 0, original_comp_id: int = 0, temporary: bool = False, src=None):
        self.id = id  # Original keyframe ID (or temp vertex ID for multi-hypothesis)
        self.original_kf_id = original_kf_id  # Original keyframe ID
        self.original_comp_id = original_comp_id  # Component/hypothesis ID
        self.temporary = temporary  # Whether the vertex is a temp kf
        self._pose = pose
        self._std = std
        self._src = src  # (pose_mu, pose_std, component) of the keyframe at construction

    @property
    def pose(self) -> pp.LieTensor:
        if self._pose is None and self._src is not None:
            self._pose = self._src[0][self._src[2]]
        return self._pose

    @pose.setter
    def pose(self, value):
        self._pose = value

    @property
    def std(self) -> pp.LieTensor:
        if self._std is None and self._src is not None:
            self._std = self._src[1][self._src[2]]
        return self._std

    @std.setter
    def std(self, value):
        self._std = value

    def pose_row(self) -> torch.Tensor:
        """The pose as a plain (7,) tensor (the values of `pose`, without creating a LieTensor)."""
        if self._pose is not None or self._src is None:
            return self._pose.tensor()
        with _plain_tensor_ops():
            return self._src[0][self._src[2]]

    def __repr__(self):
        return (f"Vertex(id={self.id}, original_kf_id={self.original_kf_id}, original_comp_id={self.original_comp_id}, "
                f"temporary={self.temporary})")


class PoseGraph:
    """Pose Graph for Loop Closure
    """
    
    def __init__(
        self,
        hypothesis_manager,
        depth: int = 100,
        k_hop: int = 2,
        device: str = "cuda",
        uncertainty_scales: Optional[Dict[EdgeType, float]] = None,
        noise_fn=None,
        skip_fn=None,
        ):
        """
        Args:
            nodes: Dictionary of keyframe ID to Keyframe objects
            odom_edges: Dictionary of odometry edges
            depth: Odometry expansion depth
            k_hop: Visual edge expansion hops
            device: Device for tensor operations
            uncertainty_scales: Optional scaling factors for edge uncertainties by type
            noise_fn: optional callable factor -> sigmas (pypose order, 6) or a covariance (6, 6, gtsam order r, t)
                      replacing the factor's own std (calibrated noise model of the verified loop closure); None keeps
                      the stored std
            skip_fn: optional callable factor -> bool; True excludes the factor from the optimisation (visual
                     measurements that the odometry chain already explains better: the information criterion of
                     the verified loop closure)
        """
        self.noise_fn = noise_fn
        self.skip_fn = skip_fn
        self.nodes = hypothesis_manager.nodes
        self.odom_edges = hypothesis_manager.odom_edges
        self.hypothesis_manager = hypothesis_manager
        self.chart_aware = hypothesis_manager.chart_aware
        self.source_component_charts = hypothesis_manager.component_charts.tolist()
        self.source_component_generations = list(hypothesis_manager.component_generations)
        self.output_chart = None
        self.preferred_fixed_node = None
        self.device = device
        self.depth = depth
        self.k_hop = k_hop

        # Uncertainty scaling factors
        self.uncertainty_scales = uncertainty_scales or {
            EdgeType.LOOP_CLOSURE: 1.0,
            EdgeType.ODOMETRY: 1.0,
            EdgeType.VISUAL: 1.0,
        }

        # Robust kernel settings for visual edges (read from system config if available)
        system = getattr(hypothesis_manager, "system", None)
        if system is not None and hasattr(system, "config"):
            pgo_cfg = system.config.pgo
            self.visual_robust_enabled = pgo_cfg.visual_robust_enabled
            self.visual_robust_type = pgo_cfg.visual_robust_type.value if hasattr(pgo_cfg.visual_robust_type, "value") else pgo_cfg.visual_robust_type
            self.visual_robust_delta = float(pgo_cfg.visual_robust_delta)

        # Constructed graph structure (filled by construct methods)
        self.vertices: List[Vertex] = []
        self.edges: List[Tuple[int, int, List]] = []
        self.vertex_map: Dict[int, Vertex] = {}  # vertex_id -> Vertex

        # Optimization results
        self.optimized_poses: Dict[int, pp.LieTensor] = {}
        self.optimization_cost: float = 0.0
    
    def _make_between_noise_model(self, factor, gtsam_sigmas: np.ndarray, gtsam_cov: Optional[np.ndarray] = None) -> 'gtsam.noiseModel.Base':
        """
        Create a GTSAM noise model for a between factor (diagonal sigmas, or a full covariance in gtsam order),
        optionally wrapping visual factors in a robust kernel to downweight outliers.
        """
        if gtsam_cov is not None:
            base_noise = gtsam.noiseModel.Gaussian.Covariance(np.asarray(gtsam_cov, dtype=np.float64))
        else:
            base_noise = gtsam.noiseModel.Diagonal.Sigmas(gtsam_sigmas)

        # Only visual edges get robust kernels; others remain Gaussian.
        if factor.type != EdgeType.VISUAL or not self.visual_robust_enabled:
            return base_noise

        robust_type = self.visual_robust_type
        delta = self.visual_robust_delta

        try:
            if robust_type == "huber":
                m_estimator = gtsam.noiseModel.mEstimator.Huber.Create(delta)
            elif robust_type == "cauchy":
                m_estimator = gtsam.noiseModel.mEstimator.Cauchy.Create(delta)
            elif robust_type == "tukey":
                m_estimator = gtsam.noiseModel.mEstimator.Tukey.Create(delta)
            else:
                logger.warning(
                    f"PGO: unknown visual robust type '{robust_type}', "
                    "falling back to non-robust noise model."
                )
                return base_noise

            return gtsam.noiseModel.Robust.Create(m_estimator, base_noise)
        except Exception as e:
            logger.warning(
                f"PGO: failed to create robust noise model "
                f"for visual factor (type={robust_type}, delta={delta}): {e}"
            )
            return base_noise
    
    # ------------------------------------------------------------------ factor inputs (cached per factor)
    def _measurement(self, factor) -> 'gtsam.Pose3':
        """GTSAM measurement of a factor (from its cached numpy copy of the mean), reused while that copy is the same."""
        m = factor.mean_np if hasattr(factor, "mean_np") else factor.mean
        entry = _measurement_cache.get(factor)
        if entry is not None and entry[0] is m:
            return entry[1]
        measurement = pypose_to_gtsam_pose3(m)
        _measurement_cache[factor] = (m, measurement)
        return measurement

    def _noise_cache_key(self):
        """Everything a between factor's noise model depends on besides the factor itself, or None (no caching: a
        noise function without a `cache_key` may depend on state the cache cannot see)."""
        if self.noise_fn is not None and not hasattr(self.noise_fn, "cache_key"):
            return None
        robust = (getattr(self, "visual_robust_enabled", None), getattr(self, "visual_robust_type", None),
                  getattr(self, "visual_robust_delta", None))
        return (self.noise_fn.cache_key() if self.noise_fn is not None else None,
                tuple(sorted((k.name, float(v)) for k, v in self.uncertainty_scales.items())),
                bool(getattr(self, "scale_by_multiplicity", True)), robust)

    def _between_noise(self, factor, num_visual_edges: int, key) -> 'gtsam.noiseModel.Base':
        """Noise model of a between factor; cached per factor while its inputs (measurement copy, noise metadata,
        multiplicity and `key`) are unchanged.  Same model as _compute_between_noise."""
        if key is None:
            return self._compute_between_noise(factor, num_visual_edges)
        modelled = self.noise_fn is not None and factor.type in getattr(self.noise_fn, "cache_types", ())
        m = factor.mean_np if hasattr(factor, "mean_np") else factor.mean
        sd = None if modelled else (factor.std_np if hasattr(factor, "std_np") else factor.std)
        fields = (factor.type, getattr(factor, "noise_scale", None), getattr(factor, "noise_scale_along", None),
                  getattr(factor, "noise_scale_rot", None), getattr(factor, "n_frames", None),
                  getattr(factor, "odom_fault", None), num_visual_edges)
        entry = _noise_cache.get(factor)
        if entry is not None and entry[0] is m and entry[1] is sd and entry[2] == fields and entry[3] == key:
            return entry[4]
        model = self._compute_between_noise(factor, num_visual_edges)
        _noise_cache[factor] = (m, sd, fields, key, model)
        return model

    def _compute_between_noise(self, factor, num_visual_edges: int) -> 'gtsam.noiseModel.Base':
        """Noise model of a between factor: the calibrated model (noise_fn) or the factor's own std, scaled by the
        uncertainty scale of its type and, for visual factors, by the number of visual factors of its edge."""
        # Convert diagonal std from pypose to gtsam noise model (calibrated model if available)
        pypose_stds = None
        gtsam_cov = None
        if self.noise_fn is not None:
            s = self.noise_fn(factor)
            if s is not None:
                s = np.asarray(s, dtype=np.float64)
                if s.ndim == 2:
                    gtsam_cov = s.copy()
                else:
                    pypose_stds = s.copy()
        if gtsam_cov is not None:
            k = float(self.uncertainty_scales.get(factor.type, 1.0))
            if factor.type == EdgeType.VISUAL and num_visual_edges > 0 and getattr(self, "scale_by_multiplicity", True):
                k *= num_visual_edges
            gtsam_cov = gtsam_cov * k ** 2 + np.eye(6) * 1e-18
            return self._make_between_noise_model(factor, None, gtsam_cov)
        if pypose_stds is None:
            pypose_stds = (factor.std_np if hasattr(factor, "std_np") else factor.std.tensor().cpu().numpy().flatten()).astype(np.float64).copy()

        # Apply uncertainty scaling factor
        if factor.type in self.uncertainty_scales:
            pypose_stds *= self.uncertainty_scales[factor.type]

        # Scale std by number of visual edges
        if factor.type == EdgeType.VISUAL and num_visual_edges > 0 and getattr(self, "scale_by_multiplicity", True):
            pypose_stds *= num_visual_edges

        # Ensure non-negative stds
        pypose_stds[pypose_stds < 0] = 0.0

        # Reorder from pypose [vx, vy, vz, wx, wy, wz] to gtsam [wx, wy, wz, vx, vy, vz]
        gtsam_sigmas = np.array([
            pypose_stds[3],  # wx
            pypose_stds[4],  # wy
            pypose_stds[5],  # wz
            pypose_stds[0],  # vx
            pypose_stds[1],  # vy
            pypose_stds[2],  # vz
        ])
        gtsam_sigmas += 1e-9  # Add epsilon for stability

        return self._make_between_noise_model(factor, gtsam_sigmas)

    def _expand_odom_nodes(
        self,
        target_node_id: int,
        hypothesis_id: int = 0,
        depth: int = 100,
        with_vertices: bool = True,
    ) -> Tuple[Set[int], List[Vertex]]:
        """
        Expand the odom edges by depth and create vertices for the specified hypothesis.
        
        Args:
            target_node_id: The central node to expand from
            hypothesis_id: Which hypothesis component to use for pose
            depth: Number of keyframes to expand in each direction
            
        Returns:
            Tuple[Set[int], List[Vertex]]: Node IDs and corresponding vertices
        """
        odom_node_ids = set()
        # Get all nodes sorted by ID for efficient consecutive checking
        all_node_ids = sorted(self.nodes.keys())
        target_idx = all_node_ids.index(target_node_id)

        min_idx = max(0, target_idx - depth)
        max_idx = min(len(all_node_ids), target_idx + depth)
        odom_node_ids.update(all_node_ids[min_idx:max_idx])
        
        odom_nodes = [self._vertex(node_id, hypothesis_id) for node_id in odom_node_ids] if with_vertices else []

        return odom_node_ids, odom_nodes

    def _vertex(self, node_id: int, hypothesis_id: int, vertex_id: Optional[int] = None) -> Vertex:
        """Vertex of keyframe `node_id` in hypothesis `hypothesis_id` (its belief row, read lazily)."""
        kf = self.nodes[node_id]
        return Vertex(id=node_id if vertex_id is None else vertex_id, original_kf_id=node_id,
                      original_comp_id=hypothesis_id, temporary=kf.temporary,
                      src=(kf.pose_mu, kf.pose_std, hypothesis_id))
    
    def _expand_visual_edges(
        self,
        target_node_ids: Set[int],
        k_hop: int,
        hypothesis_id: int,
        from_hypothesis_id: int,
        hypothesis_visual_edges: Dict,
        hypothesis_visual_adjacency: Dict,
    ) -> Tuple[Set[int], List[Tuple[int, int, List]], List[Vertex]]:
        """
        Extract the visual edges and expand the subgraph via BFS.
        
        It will expand all nodes in target_node_ids by k_hop through visual edges.
        Then collect the visual edges between the expanded nodes in hypothesis_id.
        We only consider edges either: 
        1) both vertexes comp ids are from the same hypothesis_id 
        2) from_comp_id is from_hypothesis_id and to_comp_id is hypothesis_id
        
        For case 2, we will add an edge from from_hypothesis_id to hypothesis_id, 
        but don't further expand the neighborhood.
        
        Args:
            target_node_ids: Initial set of node IDs to expand from
            k_hop: Number of hops to expand
            hypothesis_id: The hypothesis to extract edges for
            from_hypothesis_id: Source hypothesis for cross-hypothesis edges
            hypothesis_visual_edges: Visual edges for the hypothesis
            hypothesis_visual_adjacency: Visual adjacency list for the hypothesis
        
        Returns:
            Tuple containing:
            - Set of expanded node IDs
            - List of edges (u, v, edge_factors)
            - List of Vertex objects
        """
        # Use BFS to expand k-hop from the target nodes
        visual_node_ids = target_node_ids.copy()
        q = collections.deque([(node_id, 0) for node_id in target_node_ids])
        visited = target_node_ids.copy()
        
        while q:
            curr_id, hop_depth = q.popleft()
            if hop_depth >= k_hop:
                continue
            
            # Check all neighbors in visual adjacency
            for neighbor_id in hypothesis_visual_adjacency.get(curr_id, set()):
                if neighbor_id in visited:
                    continue
                
                # Get the edge factors for this edge (could be in either direction)
                edge_key_1 = (curr_id, neighbor_id)
                edge_key_2 = (neighbor_id, curr_id)
                
                edge_factors_1 = hypothesis_visual_edges.get(edge_key_1, [])
                edge_factors_2 = hypothesis_visual_edges.get(edge_key_2, [])
                
                # Check if any factor satisfies case 1 or case 2
                has_case_1 = False
                has_case_2 = False
                
                for factor in edge_factors_1 + edge_factors_2:
                    if factor.from_comp_id == hypothesis_id and factor.to_comp_id == hypothesis_id:
                        has_case_1 = True
                    elif factor.from_comp_id == from_hypothesis_id and factor.to_comp_id == hypothesis_id:
                        has_case_2 = True
                
                # Case 1: expand through this neighbor
                if has_case_1:
                    visited.add(neighbor_id)
                    visual_node_ids.add(neighbor_id)
                    q.append((neighbor_id, hop_depth + 1))
                # Case 2: add to nodes but don't expand
                elif has_case_2:
                    visited.add(neighbor_id)
                    visual_node_ids.add(neighbor_id)
        
        # Collect all edges between the expanded nodes that satisfy case 1 or case 2
        # iterate over all edges in the hypothesis to avoid re-adding the same edge
        edges = []
        for (u, v), edge_factors in hypothesis_visual_edges.items():
            if u in visual_node_ids and v in visual_node_ids:
                relevant_factors = [
                    factor for factor in edge_factors 
                    if (factor.from_comp_id == hypothesis_id and factor.to_comp_id == hypothesis_id) or
                       (factor.from_comp_id == from_hypothesis_id and factor.to_comp_id == hypothesis_id)
                ]
                if relevant_factors:
                    edges.append((u, v, relevant_factors))
        
        nodes = [self._vertex(node_id, hypothesis_id) for node_id in visual_node_ids]
        return visual_node_ids, edges, nodes

    def _expand_odom_edges(
        self,
        target_node_ids: Set[int],
    ) -> List[Tuple[int, int, List]]:
        """
        Collect odometry edges between consecutive nodes in the target set.

        TODO: this seems not correct. We want to connect visual edges between two nodes
        with odom edges if they're connected by odom edges. the odom edge can be n-hop, not just consecutive nodes.
        
        Args:
            target_node_ids: Set of node IDs to collect edges from
            
        Returns:
            List of edges (u, v, [edge])
        """
        edges = []
        sorted_node_ids = sorted(target_node_ids)
        for i in range(len(sorted_node_ids) - 1):
            u = sorted_node_ids[i]
            v = sorted_node_ids[i + 1]
            if (u, v) in self.odom_edges:
                edges.append((u, v, [self.odom_edges[(u, v)]]))

        return edges

    @timeit
    def construct_for_loop_closure(
        self,
        target_node_id: int,
        other_hypothesis_id: int = 0,
    ) -> None:
        """
        Construct a pose graph for loop closure between two hypotheses.
        
        For hypothesis 0 (current active), we use original keyframe IDs.
        For the other hypothesis, we create temporary vertex IDs to avoid conflicts,
        and add loop closure edges connecting the same keyframe across hypotheses.
        
        The constructed graph is stored in self.vertices and self.edges.
        
        Args:
            target_node_id: The central node to expand from
            other_hypothesis_id: The hypothesis to merge with (0 means no merge)
        """

        depth = self.depth
        k_hop = self.k_hop
        hypotheses = self.hypothesis_manager.hypotheses
        # --- Step 1: Expand odometry nodes by depth for hypothesis 0 ---
        odom_node_ids, odom_nodes = self._expand_odom_nodes(
            target_node_id, hypothesis_id=0, depth=depth, with_vertices=False
        )
        
        # --- Step 2: Expand visual edges by k-hop for hypothesis 0 ---
        hypothesis_0 = hypotheses[0]
        visual_node_ids, visual_edges, vertices = self._expand_visual_edges(
            odom_node_ids, 
            k_hop, 
            hypothesis_id=0, 
            from_hypothesis_id=0,
            hypothesis_visual_edges=hypothesis_0.visual_edges,
            hypothesis_visual_adjacency=hypothesis_0.visual_adjacency,
        )

        # --- Step 3: Collect odometry edges for hypothesis 0 ---
        odom_edges = self._expand_odom_edges(visual_node_ids)
        
        edges = visual_edges + odom_edges

        # --- Step 4: Merge with the other hypothesis if needed ---
        if other_hypothesis_id != 0 and other_hypothesis_id in hypotheses:
            other_hypothesis = hypotheses[other_hypothesis_id]
            
            # Map from original kf_id to temp vertex ID for the other hypothesis
            kf_to_temp_vertex = {}
            
            # Get all keyframes that belong to the other hypothesis
            # (from start_idx onwards)
            sorted_kf_ids = sorted(self.nodes.keys())
            start_idx = other_hypothesis.start_idx
            
            # Find keyframes that exist in both hypotheses (for LC edges)
            # These are keyframes >= start_idx that are also in visual_node_ids
            common_kf_ids = [kf_id for kf_id in sorted_kf_ids if kf_id >= start_idx]
            if self.chart_aware:
                common_kf_ids = [i for i in common_kf_ids
                                if int(self.nodes[i].pose_charts[other_hypothesis_id]) == self.source_component_charts[other_hypothesis_id]
                                and self.nodes[i].pose_weights[other_hypothesis_id] > 0]
            
            # --- Step 4.1: Create temp vertices for the other hypothesis ---
            previous_temp_id = None
            previous_kf_id = None
            # Important: temp vertex ids start from the last keyframe id in the nodes
            # to avoid conflicts with the original keyframe ids
            # (must not collide with any keyframe id: ids are not dense once temporary keyframes are
            # pruned or a map is loaded, so len(nodes) can be far below the maximum id)
            temp_vertex_id = max(self.nodes.keys()) + 1
            
            for kf_id in common_kf_ids:
                kf = self.nodes[kf_id]
                # keyframes created before the hypothesis carried any weight have no pose in it
                # (identity placeholder): twinning them would tie the chain to the origin
                if float(kf.pose_weights[other_hypothesis_id]) <= 0.0:
                    previous_temp_id = None
                    previous_kf_id = None
                    continue
                
                # Create a temp vertex for this keyframe in the other hypothesis
                kf_to_temp_vertex[kf_id] = temp_vertex_id
                
                v = self._vertex(kf_id, other_hypothesis_id, vertex_id=temp_vertex_id)
                vertices.append(v)
                
                # --- Step 4.2: Add odometry edge between consecutive temp vertices ---
                if previous_kf_id is not None and (previous_kf_id, kf_id) in self.odom_edges:
                    odom_edge = self.odom_edges[(previous_kf_id, kf_id)]
                    edges.append((previous_temp_id, temp_vertex_id, [odom_edge]))
                
                # --- Step 4.3: Add LC edge between original and temp vertex ---
                # LC edge is identity because they represent the same physical location
                # but only add if this keyframe exists in hypothesis 0's subgraph
                if kf_id in visual_node_ids:
                    lc_edge = Edge(
                        mean=pp.identity_SE3(device=self.device),
                        std=pp.se3(torch.full((6,), 0.01, device=self.device)),  # Small uncertainty
                        type=EdgeType.LOOP_CLOSURE,
                    )
                    edges.append((kf_id, temp_vertex_id, [lc_edge]))
                
                previous_temp_id = temp_vertex_id
                temp_vertex_id += 1
                previous_kf_id = kf_id
            
            # --- Step 4.4: Add visual edges for the other hypothesis ---
            # We need to remap vertex IDs from original to temp IDs
            for (u, v), edge_factors in other_hypothesis.visual_edges.items():
                cross_hypo = False
                # u can be either from current hypothesis (0) or from the other hypothesis
                # if u is from 0, then the from_comp_id should also be 0
                if u not in kf_to_temp_vertex and v in kf_to_temp_vertex:
                    temp_u = u
                    temp_v = kf_to_temp_vertex[v]
                    cross_hypo = True

                elif u in kf_to_temp_vertex and v in kf_to_temp_vertex:
                    # both u and v are from the other hypothesis
                    # then the both from and to comp ids should be other_hypothesis_id
                    temp_u = kf_to_temp_vertex[u]
                    temp_v = kf_to_temp_vertex[v]
                else:
                    continue   # an endpoint has no twin (keyframe predates the hypothesis): skip the edge

                # Filter factors that belong to this hypothesis
                # Accept edges where:
                # 1. Both from_comp_id and to_comp_id are from other_hypothesis_id
                # 2. from_comp_id is from current hypothesis (0) and to_comp_id is other_hypothesis_id
                relevant_factors = [
                    factor for factor in edge_factors 
                    if (not cross_hypo and factor.from_comp_id == other_hypothesis_id and factor.to_comp_id == other_hypothesis_id) or
                        (cross_hypo and factor.from_comp_id == 0 and factor.to_comp_id == other_hypothesis_id)
                ]
                # this can happen and not necessarily an error, i just want to check when it happens
                # assert len(relevant_factors) == len(edge_factors), f"Edge factors length mismatch for edge ({u}, {v})"
                
                edges.append((temp_u, temp_v, relevant_factors))
        
        # Store constructed graph
        # Initialize the new-session vertices of hypothesis 0 (kf_id >= start_idx) from their twins in the
        # other hypothesis: their hypothesis-0 poses form an odometry chain hanging off the new atlas
        # centre, possibly hundreds of metres from the map, and LM does not recover from that; the
        # tracked hypothesis already encodes the loop-closure alignment (identity LC edges make this exact).
        # (chart-aware graphs keep each chart's own coordinates; the join transports them after the solve)
        if other_hypothesis_id != 0 and other_hypothesis_id in hypotheses and not self.chart_aware:
            start_idx = hypotheses[other_hypothesis_id].start_idx
            # every keyframe of the current session (including the initial one created before the
            # hypothesis was born) is re-initialised, otherwise the odometry edge from the session's
            # initial keyframe (still at the atlas centre) drags the merged chain
            sess_start = getattr(getattr(self.hypothesis_manager, "system", None), "_session_start_kf_id", 0) or start_idx
            start_idx = min(start_idx, sess_start) if sess_start > 0 else start_idx
            twin_of = {v.original_kf_id: v for v in vertices if v.id not in self.nodes}
            successor = None
            # session keyframes without a twin (created before the hypothesis) are re-initialised by
            # composing the nearest twinned keyframe with the odometry chain
            for v in vertices:
                if v.original_comp_id == 0 and v.id in self.nodes and v.id >= start_idx:
                    if v.id in twin_of:
                        v.pose = twin_of[v.id].pose
            for v in vertices:
                if v.original_comp_id == 0 and v.id in self.nodes and v.id >= start_idx and v.id not in twin_of:
                    # walk forward along odometry edges to the first twinned keyframe
                    chain, cur, ok = [], v.id, False
                    if successor is None:      # first odometry edge out of each keyframe (edge insertion order)
                        successor = {}
                        for (a, b) in self.odom_edges:
                            successor.setdefault(a, b)
                    for _ in range(10000):
                        nxt = successor.get(cur)
                        if nxt is None:
                            break
                        chain.append((cur, nxt))
                        cur = nxt
                        if cur in twin_of:
                            ok = True
                            break
                    if ok:
                        T = twin_of[cur].pose
                        for (a, b) in reversed(chain):
                            T = T @ self.odom_edges[(a, b)].mean.Inv()
                        v.pose = T
        self.vertices = vertices
        self.edges = edges
        self.vertex_map = {v.id: v for v in vertices}
        if self.chart_aware:
            self._retain_connected_chart_graph(target_node_id, other_hypothesis_id)
        if self.hypothesis_manager.source_states is not None:
            from cross.core.conditional_pgo import prepare
            prepare(self,other_hypothesis_id)

    def construct_window(self, first_free_id: int) -> Set[int]:
        """Hypothesis-0 graph for a windowed loop-closure optimisation: the keyframes with id >= first_free_id (free)
        and the older keyframes that share a hypothesis-0 factor with them (returned: the fixed boundary).  Built from
        the adjacency of the window only, so its cost does not grow with the size of the map."""
        h0 = self.hypothesis_manager.hypotheses[0]
        ids = sorted(self.nodes.keys())
        start = bisect.bisect_left(ids, first_free_id)
        free = ids[start:]
        free_set = set(free)
        boundary: Set[int] = set()
        edges = []
        seen = set()
        for u in free:
            for v in h0.visual_adjacency.get(u, ()):
                if v not in self.nodes:
                    continue
                for key in ((u, v), (v, u)):
                    if key in seen:
                        continue
                    seen.add(key)
                    factors = [f for f in h0.visual_edges.get(key, ()) if f.from_comp_id == 0 and f.to_comp_id == 0]
                    if factors:
                        edges.append((key[0], key[1], factors))
                        if v not in free_set:
                            boundary.add(v)
        chain = ([ids[start - 1]] if start > 0 else []) + free     # the odometry chain into and through the window
        for a, b in zip(chain[:-1], chain[1:]):
            if (a, b) in self.odom_edges:
                edges.append((a, b, [self.odom_edges[(a, b)]]))
                if a not in free_set:
                    boundary.add(a)
        self.vertices = [self._vertex(i, 0) for i in free] + [self._vertex(i, 0) for i in sorted(boundary)]
        self.edges = edges
        self.vertex_map = {v.id: v for v in self.vertices}
        return boundary

    def _retain_connected_chart_graph(self, target_node_id, other_hypothesis_id=0):
        """Do not optimize unrelated saved sessions merely due to adjacent IDs."""
        neighbors = collections.defaultdict(set)
        for a, b, factors in self.edges:
            if factors and a in self.vertex_map and b in self.vertex_map:
                neighbors[a].add(b)
                neighbors[b].add(a)
        connected, pending = set(), [target_node_id]
        while pending:
            node = pending.pop()
            if node not in connected:
                connected.add(node)
                pending.extend(neighbors[node] - connected)
        self.vertices = [v for v in self.vertices if v.id in connected]
        self.edges = [(a, b, f) for a, b, f in self.edges if f and a in connected and b in connected]
        self.vertex_map = {v.id: v for v in self.vertices}
        originals = [v for v in self.vertices if v.original_comp_id == 0]
        self.source_node_charts = {v.id: int(self.nodes[v.id].pose_charts[0]) for v in originals}
        self.source_node_poses = {v.id: v.pose.clone() for v in originals}
        reference_chart = self.source_component_charts[other_hypothesis_id]
        anchors = [v.id for v in originals if self.source_node_charts[v.id] == reference_chart]
        if not anchors:
            raise ValueError("Graph does not connect to its proposed reference chart")
        self.preferred_fixed_node = min(anchors)

    def construct_for_local_smoothing(
        self,
        target_node_id: int,
        window_kfs: int = 30,
        k_hop: int = 1,
    ) -> None:
        """
        Construct a local pose graph within hypothesis 0 around recent keyframes.
        Expands odometry nodes within a window and includes visual edges up to k-hop.
        """
        # Step 1: Expand odometry nodes around target
        odom_node_ids, _odom_nodes = self._expand_odom_nodes(
            target_node_id, hypothesis_id=0, depth=window_kfs, with_vertices=False
        )

        # Step 2: Expand visual edges based on these nodes
        hypothesis_0 = self.hypothesis_manager.hypotheses[0]
        visual_node_ids, visual_edges, nodes = self._expand_visual_edges(
            target_node_ids=odom_node_ids,
            k_hop=k_hop,
            hypothesis_id=0,
            from_hypothesis_id=0,
            hypothesis_visual_edges=hypothesis_0.visual_edges,
            hypothesis_visual_adjacency=hypothesis_0.visual_adjacency,
        )

        # Step 3: Collect odometry edges among the expanded node set
        odom_edges = self._expand_odom_edges(visual_node_ids)

        # Store
        self.vertices = nodes
        self.edges = visual_edges + odom_edges
        self.vertex_map = {v.id: v for v in self.vertices}
        if self.chart_aware:
            self._retain_connected_chart_graph(target_node_id)
        if self.hypothesis_manager.source_states is not None:
            from cross.core.conditional_pgo import prepare
            prepare(self,0)

    def validate_edge_uncertainties(
        self,
        expected_ranges: Optional[Dict[EdgeType, Tuple[float, float]]] = None,
        clamp: bool = False,
    ) -> Dict[str, any]:
        """Validate edge uncertainties are in expected ranges.

        Args:
            expected_ranges: Dict mapping EdgeType to (min_std, max_std) tuples
            clamp: Whether to clamp values to expected ranges (modifies edges in-place)

        Returns:
            Statistics about edge uncertainties including warnings
        """
        if expected_ranges is None:
            expected_ranges = {
                EdgeType.LOOP_CLOSURE: (0.005, 0.05),
                EdgeType.ODOMETRY: (0.01, 0.1),
                EdgeType.VISUAL: (0.05, 0.5),
            }

        stats = {}
        warnings = []

        for (id1, id2, factors) in self.edges:
            for factor in factors:
                std_vals = factor.std.tensor().cpu().numpy().flatten()
                edge_type = factor.type

                if edge_type not in expected_ranges:
                    continue

                min_expected, max_expected = expected_ranges[edge_type]

                # Check if any std values are outside expected range
                mean_std = std_vals.mean()
                min_std = std_vals.min()
                max_std = std_vals.max()

                if edge_type.name not in stats:
                    stats[edge_type.name] = {
                        'count': 0,
                        'mean_std': [],
                        'min_std': [],
                        'max_std': [],
                        'out_of_range': 0,
                    }

                stats[edge_type.name]['count'] += 1
                stats[edge_type.name]['mean_std'].append(mean_std)
                stats[edge_type.name]['min_std'].append(min_std)
                stats[edge_type.name]['max_std'].append(max_std)

                if min_std < min_expected or max_std > max_expected:
                    stats[edge_type.name]['out_of_range'] += 1
                    warnings.append(
                        f"{edge_type.name} edge ({id1}, {id2}): "
                        f"std range [{min_std:.4f}, {max_std:.4f}] "
                        f"outside expected [{min_expected:.4f}, {max_expected:.4f}]"
                    )

                    if clamp:
                        # Clamp the values
                        clamped_std = np.clip(std_vals, min_expected, max_expected)
                        factor.std = pp.se3(torch.from_numpy(clamped_std).to(
                            dtype=torch.float32, device=self.device
                        ))

        # Aggregate statistics
        for edge_type_name in stats:
            stats[edge_type_name]['mean_std'] = np.mean(stats[edge_type_name]['mean_std'])
            stats[edge_type_name]['min_std'] = np.min(stats[edge_type_name]['min_std'])
            stats[edge_type_name]['max_std'] = np.max(stats[edge_type_name]['max_std'])

        return {'stats': stats, 'warnings': warnings}

    def inspect_edge_uncertainties(self, max_edges_per_type: int = 5) -> None:
        """Print detailed edge uncertainty information for debugging.

        Args:
            max_edges_per_type: Maximum number of edges to display per type
        """
        edge_data = collections.defaultdict(list)

        for (id1, id2, factors) in self.edges:
            for factor in factors:
                std_vals = factor.std.tensor().cpu().numpy().flatten()
                edge_data[factor.type.name].append({
                    'edge': (id1, id2),
                    'std_full': std_vals,
                    'std_mean': std_vals.mean(),
                    'std_trans': std_vals[:3].mean(),
                    'std_rot': std_vals[3:].mean(),
                })

        logger.debug("=" * 80)
        logger.debug("EDGE UNCERTAINTY INSPECTION")
        logger.debug("=" * 80)

        for edge_type, data in sorted(edge_data.items()):
            logger.debug(f"{edge_type} Edges: {len(data)} total")
            logger.debug("-" * 80)

            if data:
                # Show summary statistics
                all_means = [item['std_mean'] for item in data]
                all_trans = [item['std_trans'] for item in data]
                all_rot = [item['std_rot'] for item in data]

                logger.debug(f"  Overall mean std: {np.mean(all_means):.6f}")
                logger.debug(f"  Translation mean std: {np.mean(all_trans):.6f}")
                logger.debug(f"  Rotation mean std: {np.mean(all_rot):.6f}")
                logger.debug(f"  Min/Max mean std: {np.min(all_means):.6f} / {np.max(all_means):.6f}")
                logger.debug(f"  Sample edges (first {min(max_edges_per_type, len(data))}):")

                for i, item in enumerate(data[:max_edges_per_type]):
                    logger.debug(f"    Edge {item['edge']}: "
                                 f"mean={item['std_mean']:.6f}, "
                                 f"trans={item['std_trans']:.6f}, "
                                 f"rot={item['std_rot']:.6f}")
                    logger.debug(f"      Full std: {item['std_full']}")

        logger.debug("=" * 80)

    @timeit
    def solve(
        self,
        optim_node_ids: Set[int],
        fixed_node_ids: Set[int],
    ) -> None:
        """
        Optimizes the constructed pose graph using GTSAM.
        
        Results are stored in self.optimized_poses and self.optimization_cost.
        
        Args:
            optim_node_ids: Set of vertex IDs to optimize
            fixed_node_ids: Set of vertex IDs to fix
        """
        assert self.vertices and self.edges, "No graph constructed. Call construct_for_loop_closure() first."
        if self.chart_aware:
            charts = {self.source_node_charts[i] for i in fixed_node_ids}
            if len(charts) != 1:
                raise ValueError("Fixed vertices must share one coordinate chart")
            self.output_chart = charts.pop()
        
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        all_node_ids = optim_node_ids.union(fixed_node_ids)
        
        # 1. Prepare initial estimates for all nodes (one device transfer for all vertices)
        ids_list = [i for i in all_node_ids]
        for node_id in ids_list:
            assert node_id in self.vertex_map, f"Node {node_id} not found in vertex map"
        poses_np = torch.stack([self.vertex_map[i].pose_row().reshape(-1) for i in ids_list]).detach().cpu().numpy().astype(np.float64) if ids_list else np.zeros((0, 7))
        for node_id, p in zip(ids_list, poses_np):
            initial.insert(node_id, pypose_to_gtsam_pose3(p))
        
        # 2. Add prior factors for fixed nodes
        used = set()                       # keys of the factors added (variables without any factor are dropped below)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-9))
        for node_id in fixed_node_ids:
            assert node_id in self.vertex_map, f"Node {node_id} not found in vertex map"
            fixed_pose_gtsam = initial.atPose3(node_id)
            graph.add(gtsam.PriorFactorPose3(node_id, fixed_pose_gtsam, prior_noise))
            used.add(node_id)
        
        # 3. Add between factors for all edges
        has_edges = False
        conditional_factors = []
        record_conditional = hasattr(self, 'conditional_belief')
        noise_key = self._noise_cache_key()
        self.n_factors = {"used_visual": 0, "skipped_visual": 0, "odometry": 0, "loop_closure": 0}
        skip_fn, counts = self.skip_fn, self.n_factors
        visual_type, odometry_type = EdgeType.VISUAL, EdgeType.ODOMETRY
        for (id1, id2, factors) in self.edges:
            if id1 in all_node_ids and id2 in all_node_ids:
                # one pass: the factors the information criterion keeps, and the counts by type
                kept, num_visual_edges = [], 0
                for f in factors:
                    if skip_fn is not None and skip_fn(f):
                        if f.type == visual_type:
                            counts["skipped_visual"] += 1
                        continue
                    kept.append(f)
                    if f.type == visual_type:
                        num_visual_edges += 1
                        counts["used_visual"] += 1
                    elif f.type == odometry_type:
                        counts["odometry"] += 1
                    else:
                        counts["loop_closure"] += 1
                if not kept:
                    continue
                for factor in kept:
                    has_edges = True
                    nonlinear = gtsam.BetweenFactorPose3(id1, id2, self._measurement(factor),
                                                         self._between_noise(factor, num_visual_edges, noise_key))
                    graph.add(nonlinear)
                    used.add(id1)
                    used.add(id2)
                    if record_conditional:
                        conditional_factors.append((nonlinear,factor))

        # Log edge statistics for debugging
        if has_edges and logger.level("DEBUG").no >= 100:   # disabled: costs a device sync per factor
            edge_stats = {'LOOP_CLOSURE': [], 'ODOMETRY': [], 'VISUAL': []}
            for (id1, id2, factors) in self.edges:
                if id1 in all_node_ids and id2 in all_node_ids:
                    for factor in factors:
                        stds = factor.std.tensor().cpu().numpy().flatten().copy()
                        # Apply scaling factor if applicable
                        if factor.type in self.uncertainty_scales:
                            stds *= self.uncertainty_scales[factor.type]
                        edge_stats[factor.type.name].append(stds)

            for edge_type, std_list in edge_stats.items():
                if std_list:
                    std_array = np.array(std_list)
                    logger.debug(f"{edge_type} edges: count={len(std_list)}, "
                                f"mean_std={std_array.mean(axis=0)}, "
                                f"min_std={std_array.min(axis=0)}, "
                                f"max_std={std_array.max(axis=0)}")

        if not has_edges and len(optim_node_ids) > 0:
            logger.warning("No edges found in the pose graph to optimize.")
            self.optimization_cost = 0.0
            self.optimized_poses = {}
            return
        
        # 4. Setup and run optimizer
        # a vertex without any factor (all its edges skipped or quarantined) is not part of the elimination ordering
        # and makes GTSAM abort ("inconsistent arguments"): drop such variables, they keep their current pose
        unused = [int(k) for k in initial.keys() if int(k) not in used]
        for k in unused:
            initial.erase(k)
        if unused:
            logger.debug(f"PGO: {len(unused)} unconstrained vertices dropped ({unused[:8]})")
            optim_node_ids = [n for n in optim_node_ids if n not in set(unused)]
        params = gtsam.LevenbergMarquardtParams()
        self.initial_cost = graph.error(initial)
        if getattr(self, "eval_only", False):          # diagnostics: cost at the initial values, no optimisation
            self.lm_iterations = 0
            self.optimization_cost = self.initial_cost
            self.optimized_poses = {}
            # per-factor costs (largest first) for the diagnostics
            self.factor_costs = sorted(((float(graph.at(i).error(initial)), str(graph.at(i).keys())) for i in range(graph.size())), reverse=True)[:5]
            self._graph, self._initial = graph, initial
            return
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        with timeblock("GTSAM PGO"):
            result = optimizer.optimize()
        self.lm_iterations = int(optimizer.iterations())
        self._graph, self._result, self._initial = graph, result, initial     # diagnostics
        
        # 5. Extract results
        self.optimization_cost = graph.error(result)
        
        self.optimized_poses = {}
        rows, kept_ids = [], []
        for node_id in optim_node_ids:
            if not result.exists(node_id):
                continue
            rows.append(_pypose_row(result.atPose3(node_id)))
            kept_ids.append(node_id)
        if kept_ids:          # one conversion and device transfer for all poses (as gtsam_to_pypose_pose3 per pose)
            poses = torch.from_numpy(np.stack(rows)).to(dtype=torch.float32, device=self.device)
            with _plain_tensor_ops():
                for i, node_id in enumerate(kept_ids):
                    self.optimized_poses[node_id] = as_se3(poses[i].clone())
        if hasattr(self,'conditional_belief'):
            from cross.core.conditional_pgo import solve_responses
            solve_responses(self,result,conditional_factors,optim_node_ids,fixed_node_ids)
            # Fixed nodes also need their bias-centered pose/model persisted.
            for node_id in fixed_node_ids:
                self.optimized_poses[node_id] = gtsam_to_pypose_pose3(result.atPose3(node_id),self.device)
