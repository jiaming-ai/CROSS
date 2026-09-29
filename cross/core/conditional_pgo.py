"""Conditional graph solve at an already selected hypothesis's bias belief.

The graph does not observe the bias a second time. It solves x at that belief's
mean and propagates first-order responses through the final robust Gaussian
linearization. This is a Gauss-Newton response, not an exact derivative of a
nonzero-residual robust optimum. Conditional node covariances are retained;
reusing graph edges is not grounds to halve the online filter's uncertainty.
"""
import copy

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .conditional import SourceFactor
from .conditional_pose import ConditionalPose, adjoint, inverse, normalize_mean


def _matrix(pose):
    """Normalize near-unit stored quaternions before double-precision geometry.

    GTSAM's quaternion constructor and our transpose inverse assume SO(3).
    Passing float32 norm roundoff through repeated graph joins can amplify it;
    normalizing after forming the matrix is already too late.
    """
    return normalize_mean(pose.double()).matrix().cpu().numpy()


def prepare(pg, component):
    """Snapshot conditional factors and collapse copies of one physical node.

    CROSS's two-branch graph contains copies linked by identity constraints.
    At commitment these are the same physical pose: collapsing them avoids
    counting the same odometry measurement twice in the conditional solve.
    Source IDs remain distinct across acquisitions after the chart join.
    """
    import pypose as pp
    import torch
    hm = pg.hypothesis_manager
    if hm.source_states is None:
        return
    belief = hm.source_states[component].copy()
    id_map = {v.id:v.original_kf_id for v in pg.vertices}
    models = {}
    required_factors = []
    for vertex in pg.vertices:
        kf = hm.nodes[vertex.original_kf_id]
        if kf.conditional_poses is None or kf.conditional_poses[vertex.original_comp_id] is None:
            raise ValueError('Conditional PGO requires a model for every active node')
        model = kf.conditional_poses[vertex.original_comp_id]
        required_factors.append(model.factor)
        models[vertex.id] = model
    edges, seen = [],set()
    for a,b,factors in pg.edges:
        u,v = id_map[a],id_map[b]
        if u == v:
            continue  # structural identity between copies, not a sensor factor
        selected = []
        for edge in factors:
            if id(edge) in seen:
                continue
            seen.add(id(edge))
            if edge.conditional_pose is None:
                raise ValueError('Conditional PGO cannot use a legacy edge without source provenance')
            required_factors.append(edge.conditional_pose.factor)
            selected.append(edge)
        if selected:
            edges.append((u,v,selected))
    # Expand the complete source set once before evaluating node/edge models.
    belief = belief.expand_many(required_factors)
    chosen = {}
    pg.conditional_source_nodes = {}
    for vertex in pg.vertices:
        pose,model = models[vertex.id].at_known(_matrix(vertex.pose),belief)
        if vertex.original_comp_id == 0:
            pg.conditional_source_nodes[vertex.original_kf_id] = (pose,model)
        # A realized alternative supplies the initialization of duplicated
        # query vertices, already expressed in the proposed reference chart.
        if vertex.original_kf_id not in chosen or vertex.original_comp_id == component:
            item = copy.copy(vertex)
            item.id = vertex.original_kf_id
            item.pose = normalize_mean(pp.from_matrix(torch.as_tensor(pose,device=pg.device,dtype=torch.float64),pp.SE3_type))
            item.conditional_pose = model
            chosen[item.id] = item
    conditioned_edges = []
    for a,b,factors in edges:
        conditioned = []
        for edge in factors:
            if hm.schmidt_map_geometry and hm.map_geometry_basis == "factor":
                from .factor_geometry import factor_identity
                geometry_identity = factor_identity(edge,a,b)
            pose,model = edge.conditional_pose.at_known(_matrix(edge.mean),belief)
            item = copy.copy(edge)
            if hm.schmidt_map_geometry and hm.map_geometry_basis == "factor":
                item.geometry_factor_identity = geometry_identity
            item.mean = normalize_mean(pp.from_matrix(torch.as_tensor(pose,device=pg.device,dtype=torch.float64),pp.SE3_type))
            item.std = pp.se3(torch.as_tensor(model.geometry_covariance.diagonal().copy(),
                                             device=pg.device,dtype=edge.std.dtype).clip(1e-12).sqrt())
            item.conditional_pose = model
            conditioned.append(item)
        conditioned_edges.append((a,b,conditioned))
    pg.vertices,pg.edges = list(chosen.values()),conditioned_edges
    pg.vertex_map = chosen
    pg.conditional_belief = belief
    pg.conditional_component = component
    pg.conditional_tracking_snapshot = tuple(value.clone() for value in hm.dist)


def solve_responses(pg, result, factors, free_ids, fixed_ids):
    """Reuse GTSAM's whitened robust Jacobians; solve A dx/db = -dr/db."""
    permutation = [3,4,5,0,1,2]  # columns: GTSAM rotation-first -> CROSS translation-first
    free_ids = sorted(free_ids)
    positions = {key:6*i for i,key in enumerate(free_ids)}
    belief = pg.conditional_belief
    n = len(belief.keys)
    A = sparse.lil_matrix((6*len(factors),6*len(free_ids)))
    B = np.zeros((6*len(factors),n))
    fixed_responses = {i:pg.vertex_map[i].conditional_pose.factor.jacobian for i in fixed_ids}
    for row,(nonlinear,edge) in enumerate(factors):
        jacobian = nonlinear.linearize(result)
        keys = list(jacobian.keys())
        dense = jacobian.getA()
        a,b = keys
        Ai,Aj = dense[:,:6][:,permutation],dense[:,6:][:,permutation]
        Xi,Xj = result.atPose3(a).matrix(),result.atPose3(b).matrix()
        Z = _matrix(edge.mean)
        error_transform = inverse(Z)@inverse(Xi)@Xj
        Jz = edge.conditional_pose.factor.jacobian
        sl = slice(6*row,6*row+6)
        # The same robust whitening multiplies node and measurement responses.
        B[sl] = -Aj@adjoint(inverse(error_transform))@Jz
        for key,block in ((a,Ai),(b,Aj)):
            if key in positions:
                A[sl,positions[key]:positions[key]+6] = block
            elif key in fixed_responses:
                B[sl] += block@fixed_responses[key]
            else:
                raise ValueError('A graph response has neither a free nor fixed endpoint')
    A = A.tocsc()
    responses = dict(fixed_responses)
    if free_ids:
        normal = (A.T@A).tocsc()
        solver = splu(normal)  # Reject an unanchored/singular graph, not a fake zero response.
        solution = solver.solve(np.asarray(-A.T@B)) if n else np.empty((6*len(free_ids),0))
        responses.update({key:solution[6*i:6*i+6] for i,key in enumerate(free_ids)})
    pg.optimized_conditional_poses = {}
    for key,J in responses.items():
        previous = pg.vertex_map[key].conditional_pose
        pg.optimized_conditional_poses[key] = ConditionalPose(previous.geometry_covariance,
            SourceFactor(belief.keys,J,belief.prior_variances,belief.mean))
    pg.conditional_response_diagnostics = dict(sources=n,vertices=len(responses),edges=len(factors),
        approximation='fixed-robust-weight-gauss-newton',bias_updated=False,geometry_covariance_reused=True)
    if pg.hypothesis_manager.schmidt_map_geometry:
        from .map_geometry import prepare_refresh
        pg.joint_geometry = prepare_refresh(pg,result,factors,set(free_ids),fixed_ids)


def apply_result(hm, pg, optimized_poses, other):
    """Apply a synchronous conditional solve, including poses outside its graph."""
    import pypose as pp
    import torch
    if pg is None or not hasattr(pg,'optimized_conditional_poses'):
        raise ValueError('Conditional PGO result lacks source responses')
    if not hm.chart_aware:
        raise ValueError('Conditional graph joins require coordinate-chart provenance')
    if any(not torch.equal(old,new) for old,new in zip(pg.conditional_tracking_snapshot,hm.dist)):
        return dict(success=False,message='Conditional PGO tracking snapshot is stale')
    for component in {0,other}:
        if (pg.source_component_charts[component] != int(hm.component_charts[component]) or
                pg.source_component_generations[component] != hm.component_generations[component]):
            return dict(success=False,message='Conditional PGO refers to a retired chart or hypothesis')
    belief = pg.conditional_belief
    anchored_reference = other != 0 and hm.reference_audit(other)['unanchored_reference_candidate']
    optimized = {key:pose for key,pose in optimized_poses.items() if key in pg.conditional_source_nodes}
    if not optimized:
        return dict(success=False,message='Conditional PGO has no original node results')

    def correction(old_pose, old_model, new_pose, new_model):
        # Right response of C(b)=X_new(b) X_old(b)^-1. It must not be
        # discarded by treating a source-dependent chart join as a constant.
        # Factor-basis refresh may append new graph-noise coordinates after
        # the old-node snapshot was made. Align columns before subtraction;
        # a one-column bias Jacobian would otherwise broadcast silently.
        if old_model.factor.keys != belief.keys:
            old_pose,old_model = old_model.at_known(old_pose,belief)
        if new_model.factor.keys != belief.keys:
            new_pose,new_model = new_model.at_known(new_pose,belief)
        C = new_pose@inverse(old_pose)
        J = adjoint(old_pose)@(new_model.factor.jacobian-old_model.factor.jacobian)
        return C,J

    def transport(pose,model,C,J):
        if set(model.factor.keys).issubset(belief.keys):
            pose,model = model.at_known(pose,belief)
        else:
            pose,model,_ = model.at(pose,belief)
        response = adjoint(inverse(pose))@J+model.factor.jacobian
        moved = ConditionalPose(model.geometry_covariance,
                                SourceFactor(belief.keys,response,belief.prior_variances,belief.mean))
        return C@pose,moved

    transforms = {}
    for key in sorted(optimized):
        chart = pg.source_node_charts[key]
        old_pose,old_model = pg.conditional_source_nodes[key]
        transforms[chart] = correction(old_pose,old_model,_matrix(optimized[key]),
                                       pg.optimized_conditional_poses[key])

    # Keep the selected live pose's displacement beyond the latest graph node.
    selected_state = hm.source_states[other]
    selected_pose = _matrix(hm.dist[0][other])
    current_model = ConditionalPose.from_state(selected_state)
    eligible = [key for key in optimized if hm.nodes[key].conditional_poses[other] is not None]
    if not eligible:
        return dict(success=False,message='Conditional PGO has no anchor for the selected live hypothesis')
    latest = max(eligible)
    kf = hm.nodes[latest]
    old_pose,old_model = kf.conditional_poses[other].at_known(_matrix(kf.pose_mu[other]),belief)
    C,J = correction(old_pose,old_model,_matrix(optimized[latest]),
                     pg.optimized_conditional_poses[latest])
    current_pose,current_model = transport(selected_pose,current_model,C,J)
    current_state,_ = belief.with_pose(current_model)

    pending = []
    for node in hm.nodes.values():
        for component,model in enumerate(node.conditional_poses or []):
            if model is None:
                continue
            chart = int(node.pose_charts[component])
            if component == 0 and node.id in optimized:
                pose = _matrix(optimized[node.id])
                moved = pg.optimized_conditional_poses[node.id]
            elif chart in transforms:
                pose,moved = transport(_matrix(node.pose_mu[component]),model,*transforms[chart])
            else:
                continue
            if set(moved.factor.keys).issubset(belief.keys):
                response,_ = belief.factor_response(moved.factor)
                covariance = moved.geometry_covariance+response@belief.covariance@response.T
                marginal = np.sqrt(covariance.diagonal().clip(0))
            else:
                state,_ = belief.with_pose(moved)
                marginal = np.sqrt(state.marginal_covariance().diagonal().clip(0))
            value = pp.from_matrix(torch.as_tensor(pose,device=node.pose_mu.device,dtype=node.pose_mu.dtype),pp.SE3_type)
            std = pp.se3(torch.as_tensor(marginal,device=node.pose_std.device,dtype=node.pose_std.dtype))
            pending.append((node,component,value,moved,std))
    # Retain unrelated tracking modes at their own bias posterior. Construct
    # their conversions too before publishing any node or tracking state.
    tracking_pending = []
    for component,state in enumerate(hm.source_states):
        if state is None or component in {0,other}:
            continue
        chart = int(hm.component_charts[component])
        if chart in transforms:
            pose,model = transport(_matrix(hm.dist[0][component]),
                                   ConditionalPose.from_state(state),*transforms[chart])
            pose,model,own_belief = model.at(pose,state)
            updated,_ = own_belief.with_pose(model)
            value = pp.from_matrix(torch.as_tensor(pose,device=hm.device,dtype=hm.dist[0].dtype),pp.SE3_type)
            std = pp.se3(torch.as_tensor(np.sqrt(updated.marginal_covariance().diagonal().clip(0)),
                                         device=hm.device,dtype=hm.dist[1].dtype))
            tracking_pending.append((component,updated,value,std))
    current_value = pp.from_matrix(torch.as_tensor(current_pose,device=hm.device,dtype=hm.dist[0].dtype),pp.SE3_type)
    current_std = pp.se3(torch.as_tensor(np.sqrt(current_state.marginal_covariance().diagonal().clip(0)),
                                       device=hm.device,dtype=hm.dist[1].dtype))
    if hm.schmidt_map_geometry:
        from .map_geometry import stage_refresh
        pending,current_state,current_std,diagnostics = stage_refresh(hm,pg,pending,current_pose,current_state)
        pg.conditional_response_diagnostics.update(geometry_covariance_reused=False,
                                                   schmidt_map_geometry=diagnostics)
    # All potentially failing pose/covariance conversions have now succeeded.
    affected = set()
    for node,component,pose,model,std in pending:
        node.pose_mu[component] = pose
        node.pose_std[component] = std
        node.conditional_poses[component] = model
        node.pose_charts[component] = pg.output_chart
        node.last_pgo_step = hm.step_counter
        affected.add(node.id)
    for component,state,pose,std in tracking_pending:
        hm.source_states[component] = state
        hm.dist[0][component] = pose
        hm.dist[1][component] = std
        hm.component_charts[component] = pg.output_chart
    if other != 0:
        hm.merge_hypotheses(other,conditional_transport_done=True)
        if anchored_reference:
            hm.reference_support.mark_anchored()
    hm.source_states[0] = current_state
    hm.dist[0][0] = current_value
    hm.dist[1][0] = current_std
    hm.dist[2][0] = 1.
    hm.component_charts[0] = pg.output_chart
    if hm.system.topo_map is not None:
        hm.system.topo_map.update_after_pgo(affected)
    return dict(success=True,pose_graph=pg,optimized_poses=optimized,other_hypothesis_id=other,
                cost=pg.optimization_cost,conditional_response=pg.conditional_response_diagnostics)
