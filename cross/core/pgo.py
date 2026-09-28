import collections
import numpy as np
import torch
import pypose as pp
import gtsam
from typing import Tuple, List, Dict, Set, Optional
from dataclasses import dataclass
from loguru import logger

from cross.core.types import Edge, EdgeType
from cross.utils.profile import timeit, timeblock


def pypose_to_gtsam_pose3(pose: pp.LieTensor) -> 'gtsam.Pose3':
    """Converts a pypose SE3 LieTensor to a gtsam.Pose3 object."""
    pose_tensor = pose.tensor().cpu().numpy().astype(np.float64)
    # pypose quat: [qx, qy, qz, qw]
    # gtsam quat: [w, x, y, z]
    rot = gtsam.Rot3(pose_tensor[6], pose_tensor[3], pose_tensor[4], pose_tensor[5])
    trans = gtsam.Point3(pose_tensor[0], pose_tensor[1], pose_tensor[2])
    
    return gtsam.Pose3(rot, trans)


def gtsam_to_pypose_pose3(pose: 'gtsam.Pose3', device: str) -> pp.LieTensor:
    """Converts a gtsam.Pose3 object to a pypose SE3 LieTensor."""
    rot_quat = pose.rotation().toQuaternion().coeffs()  # [x, y, z, w]
    trans = pose.translation()  # [x, y, z]
    # pypose quat: [qx, qy, qz, qw]
    pypose_tensor = np.concatenate([trans, rot_quat])
    return pp.SE3(torch.from_numpy(pypose_tensor).to(dtype=torch.float32, device=device))


@dataclass
class Vertex:
    """Represents a vertex (node) in the pose graph."""
    id: int  # Original keyframe ID (or temp vertex ID for multi-hypothesis)
    pose: pp.LieTensor
    std: pp.LieTensor
    original_kf_id: int = 0  # Original keyframe ID
    original_comp_id: int = 0  # Component/hypothesis ID
    temporary: bool = False  # Whether the vertex is a temp kf 


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
        ):
        """
        Args:
            nodes: Dictionary of keyframe ID to Keyframe objects
            odom_edges: Dictionary of odometry edges
            depth: Odometry expansion depth
            k_hop: Visual edge expansion hops
            device: Device for tensor operations
            uncertainty_scales: Optional scaling factors for edge uncertainties by type
        """
        self.nodes = hypothesis_manager.nodes
        self.odom_edges = hypothesis_manager.odom_edges
        self.hypothesis_manager = hypothesis_manager
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
    
    def _make_between_noise_model(self, factor, gtsam_sigmas: np.ndarray) -> 'gtsam.noiseModel.Base':
        """
        Create a GTSAM noise model for a between factor, optionally wrapping visual
        factors in a robust kernel to downweight outliers.
        """
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
    
    def _expand_odom_nodes(
        self,
        target_node_id: int,
        hypothesis_id: int = 0,
        depth: int = 100,
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
        
        odom_nodes = []
        for node_id in odom_node_ids:
            v = Vertex(
                id=node_id,
                pose=self.nodes[node_id].pose_mu[hypothesis_id],
                std=self.nodes[node_id].pose_std[hypothesis_id],
                original_kf_id=node_id,
                original_comp_id=hypothesis_id,
                temporary=self.nodes[node_id].temporary,
            )
            odom_nodes.append(v)

        return odom_node_ids, odom_nodes
    
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
        
        nodes = []
        for node_id in visual_node_ids:
            v = Vertex(
                id=node_id,
                pose=self.nodes[node_id].pose_mu[hypothesis_id],
                std=self.nodes[node_id].pose_std[hypothesis_id],
                original_kf_id=node_id,
                original_comp_id=hypothesis_id,
                temporary=self.nodes[node_id].temporary,
            )
            nodes.append(v)
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
            target_node_id, hypothesis_id=0, depth=depth
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
            
            # --- Step 4.1: Create temp vertices for the other hypothesis ---
            previous_temp_id = None
            previous_kf_id = None
            # Important: temp vertex ids start from the last keyframe id in the nodes
            # to avoid conflicts with the original keyframe ids
            temp_vertex_id = len(self.nodes)
            
            for kf_id in common_kf_ids:
                kf = self.nodes[kf_id]
                
                # Create a temp vertex for this keyframe in the other hypothesis
                kf_to_temp_vertex[kf_id] = temp_vertex_id
                
                v = Vertex(
                    id=temp_vertex_id,
                    pose=kf.pose_mu[other_hypothesis_id],
                    std=kf.pose_std[other_hypothesis_id],
                    original_kf_id=kf_id,
                    original_comp_id=other_hypothesis_id,
                    temporary=kf.temporary,
                )
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

                else:
                    # both u and v are from the other hypothesis
                    # then the both from and to comp ids should be other_hypothesis_id
                    temp_u = kf_to_temp_vertex[u]
                    temp_v = kf_to_temp_vertex[v]

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
        self.vertices = vertices
        self.edges = edges
        self.vertex_map = {v.id: v for v in vertices}

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
            target_node_id, hypothesis_id=0, depth=window_kfs
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
        
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        all_node_ids = optim_node_ids.union(fixed_node_ids)
        
        # 1. Prepare initial estimates for all nodes
        for node_id in all_node_ids:
            assert node_id in self.vertex_map, f"Node {node_id} not found in vertex map"
            vertex = self.vertex_map[node_id]
            initial_pose_gtsam = pypose_to_gtsam_pose3(vertex.pose)
            initial.insert(node_id, initial_pose_gtsam)
        
        # 2. Add prior factors for fixed nodes
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-9))
        for node_id in fixed_node_ids:
            assert node_id in self.vertex_map, f"Node {node_id} not found in vertex map"
            fixed_pose_gtsam = initial.atPose3(node_id)
            graph.add(gtsam.PriorFactorPose3(node_id, fixed_pose_gtsam, prior_noise))
        
        # 3. Add between factors for all edges
        has_edges = False
        for (id1, id2, factors) in self.edges:
            if id1 in all_node_ids and id2 in all_node_ids:
                num_visual_edges = sum(1 for f in factors if f.type == EdgeType.VISUAL)
                for factor in factors:
                    has_edges = True
                    # Convert measurement from pypose to gtsam
                    measurement_gtsam = pypose_to_gtsam_pose3(factor.mean)
                    
                    # Convert diagonal std from pypose to gtsam noise model
                    pypose_stds = factor.std.tensor().cpu().numpy().flatten()

                    # Apply uncertainty scaling factor
                    if factor.type in self.uncertainty_scales:
                        pypose_stds *= self.uncertainty_scales[factor.type]

                    # Scale std by number of visual edges
                    if factor.type == EdgeType.VISUAL and num_visual_edges > 0:
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

                    noise_model = self._make_between_noise_model(factor, gtsam_sigmas)
                    graph.add(gtsam.BetweenFactorPose3(id1, id2, measurement_gtsam, noise_model))

        # Log edge statistics for debugging
        if has_edges:
            edge_stats = {'LOOP_CLOSURE': [], 'ODOMETRY': [], 'VISUAL': []}
            for (id1, id2, factors) in self.edges:
                if id1 in all_node_ids and id2 in all_node_ids:
                    for factor in factors:
                        stds = factor.std.tensor().cpu().numpy().flatten()
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
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        with timeblock("GTSAM PGO"):
            result = optimizer.optimize()
        
        # 5. Extract results
        self.optimization_cost = graph.error(result)
        
        self.optimized_poses = {}
        for node_id in optim_node_ids:
            optimized_pose_gtsam = result.atPose3(node_id)
            optimized_pose_pypose = gtsam_to_pypose_pose3(optimized_pose_gtsam, self.device)
            self.optimized_poses[node_id] = optimized_pose_pypose
