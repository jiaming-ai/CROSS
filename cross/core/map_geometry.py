"""Dense conditional map uncertainty, refreshed at a committed graph solve.

Raw graph factors occur once. Retired map variables are marginalized, metric
source inference is retained, and new map errors share one covariance. This
experimental implementation requires the entire connected map and one surviving
tracking mode. Newly inserted node residuals are independent until a refresh.
"""
import time
import uuid

import numpy as np

from .conditional import SourceFactor,SourceState
from .conditional_pose import ConditionalPose,inverse,log


def is_geometry_source(key):
    return key.startswith('geometry:')


def validate_load_mode(saved,enabled,basis='epoch'):
    """Reject silently changing the inference semantics of a persisted map."""
    if basis == 'factor':
        from .factor_geometry import validate_load_mode as validate_factors
        return validate_factors(saved,enabled)
    if basis != 'epoch':
        raise ValueError('Unknown shared map geometry basis')
    version=saved.get('schmidt_map_geometry_version',0)
    if version not in (0,1):
        raise ValueError('Unsupported Schmidt map geometry version')
    # Detect older diagnostic maps as well as explicitly versioned maps.
    belief=saved.get('hypo_data',{}).get('source_belief') or {}
    if any(k.startswith('geometry:factor:') for k in belief.get('keys',())):
        raise ValueError('Persistent factor maps require the factor geometry basis')
    if (version or any(is_geometry_source(k) for k in belief.get('keys',()))) and not enabled:
        raise ValueError('This map contains shared geometry; enable schmidt_map_geometry to load it')


def prepare_refresh(pg,result,factors,free_ids,fixed_ids):
    """Recover the graph's conditional covariance without publishing state."""
    import gtsam
    start=time.perf_counter()
    hm=pg.hypothesis_manager
    if hm.map_geometry_basis == 'factor':
        from .factor_geometry import prepare_refresh as prepare_factors
        return prepare_factors(pg,result,factors,free_ids,fixed_ids)
    if set(pg.vertex_map) != set(hm.nodes):
        raise ValueError('Schmidt map geometry requires the complete connected map')
    if any(s is not None for i,s in enumerate(hm.source_states)
           if i not in {0,pg.conditional_component}):
        raise ValueError('Schmidt map geometry requires one surviving mode after commitment')
    if len(fixed_ids) != 1:
        raise ValueError('Schmidt map geometry requires one exact gauge anchor')
    anchor=next(iter(fixed_ids))
    fixed=pg.vertex_map[anchor].conditional_pose
    if (np.max(np.abs(fixed.geometry_covariance)) > 1e-12 or
            np.max(np.abs(fixed.factor.jacobian),initial=0.) > 1e-12):
        raise ValueError('Schmidt map geometry requires a deterministic gauge')
    geometry=[i for i,k in enumerate(pg.conditional_belief.keys) if is_geometry_source(k)]
    graph=gtsam.NonlinearFactorGraph()
    for factor,edge in factors:
        if np.max(np.abs(edge.conditional_pose.factor.jacobian[:,geometry]),initial=0.) > 1e-12:
            raise ValueError('Raw factors cannot re-observe a derived map posterior')
        graph.add(factor)
    graph.add(gtsam.PriorFactorPose3(anchor,result.atPose3(anchor),
        gtsam.noiseModel.Diagonal.Sigmas(np.full(6,1e-9))))
    keys=sorted(free_ids)
    marginal=gtsam.Marginals(graph,result).jointMarginalCovariance(keys)
    permutation=[3,4,5,0,1,2]
    C=np.block([[marginal.at(i,j)[np.ix_(permutation,permutation)] for j in keys] for i in keys])
    C=(C+C.T)/2
    np.linalg.cholesky(C)
    return dict(keys=keys,covariance=C,seconds=time.perf_counter()-start)


def stage_refresh(hm,pg,pending,current_pose,current_state):
    """Build replacement node/current messages before any live mutation."""
    if hm.map_geometry_basis == 'factor':
        from .factor_geometry import stage_refresh as stage_factors
        return stage_factors(hm,pg,pending,current_pose,current_state)
    import pypose as pp
    import torch
    from .conditional_pgo import _matrix

    data=pg.joint_geometry
    C,free=data['covariance'],data['keys']
    old=current_state
    b=[i for i,key in enumerate(old.keys) if not is_geometry_source(key)]
    epoch=uuid.uuid4().hex
    geometry_keys=tuple(f'geometry:{epoch}:{key}:{d}' for key in free for d in range(6))
    keys=tuple(old.keys[i] for i in b)+geometry_keys
    mean=np.r_[old.mean[b],np.zeros(len(geometry_keys))]
    V=np.zeros((len(keys),len(keys)))
    V[:len(b),:len(b)]=old.covariance[np.ix_(b,b)]
    V[len(b):,len(b):]=C
    variances=np.r_[old.prior_variances[b],C.diagonal()]
    belief=SourceState(np.zeros((6,6)),keys,mean,V,np.zeros((6,len(keys))),
                        variances,old.seen_factors)
    old_locations={key:i for i,key in enumerate(pg.conditional_belief.keys)}
    selected_b=[old_locations[old.keys[i]] for i in b]
    positions={key:i for i,key in enumerate(free)}
    latest=max(hm.nodes)
    replacements=[]
    current_model=current_std=None
    for node,component,pose,model,std in pending:
        if component == 0:
            previous=pg.optimized_conditional_poses[node.id]
            J=np.zeros((6,len(keys)))
            J[:,:len(b)]=previous.factor.jacobian[:,selected_b]
            if node.id in positions:
                offset=len(b)+6*positions[node.id]
                J[:,offset:offset+6]=np.eye(6)
            model=ConditionalPose(np.zeros((6,6)),SourceFactor(keys,J,variances,mean))
            marginal=J@V@J.T
            std=pp.se3(torch.as_tensor(np.sqrt(marginal.diagonal().clip(0)),
                device=node.pose_std.device,dtype=node.pose_std.dtype))
            if node.id == latest:
                displacement=log(inverse(_matrix(pose))@current_pose)
                if np.linalg.norm(displacement) > 1e-5:
                    raise ValueError('Schmidt refresh requires the current pose to be a graph node')
                current_model,current_std=model,std
        replacements.append((node,component,pose,model,std))
    if current_model is None:
        raise ValueError('Schmidt refresh lacks the current graph node')
    updated,_=belief.with_pose(current_model)
    diagnostics=dict(geometry_variables=len(geometry_keys),metric_variables=len(b),
        nodes=len(hm.nodes),seconds=data['seconds'],covariance_bytes=C.nbytes,
        approximation='dense-full-graph-conditional-gaussian; new-node residual reuse not lifted',
        metric_belief_updated=False,retired_geometry_marginalized=True)
    return replacements,updated,current_std,diagnostics
