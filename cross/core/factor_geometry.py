"""Experimental persistent graph-factor coordinates across CROSS hypotheses.

At a frozen robust linearization, A dx + epsilon = 0 gives the map response
dx = -(A.T A)^-1 A.T epsilon. Each raw graph factor retains six standard-normal
coordinates across solves. Other live modes keep their source posteriors and
are transported by conditional_pgo; their old geometry is never discarded.

Fixed robust weights and independent whitened factor noises are approximations.
This is established Gaussian sensitivity algebra, with unbounded source
responses; the fixed nuisance covariance is stored diagonally.
Temporary node pruning is disabled until composed-edge noise lineage exists.
"""
import hashlib
import json
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .conditional import SourceFactor
from .conditional_pose import ConditionalPose, inverse, log

PREFIX = 'geometry:factor:'


def is_geometry_source(key):
    return key.startswith('geometry:')


def validate_load_mode(saved, enabled):
    version = saved.get('schmidt_map_geometry_version', 0)
    if version not in (0, 2):
        raise ValueError('Factor geometry cannot load epoch-node geometry maps')
    belief = saved.get('hypo_data', {}).get('source_belief') or {}
    geometry = [k for k in belief.get('keys', ()) if is_geometry_source(k)]
    if (version or geometry) and not enabled:
        raise ValueError('This map contains shared geometry; enable schmidt_map_geometry to load it')
    if any(not k.startswith(PREFIX) for k in geometry):
        raise ValueError('Cannot reinterpret previous geometry coordinates')


def factor_identity(edge, first, second):
    """Identify the immutable raw edge BEFORE copying or bias recentering.

    Ignore zero response columns so unrelated priors cannot rename a factor.
    The raw serialized measurement reproduces the identity after map loading.
    """
    model = edge.conditional_pose
    fac = model.factor
    keep = np.flatnonzero(np.any(fac.jacobian != 0, axis=0))
    record = dict(version=1, endpoints=[int(first), int(second)], kind=edge.type.name,
        mean=edge.mean.tensor().detach().double().cpu().numpy().tolist(),
        std=edge.std.tensor().detach().double().cpu().numpy().tolist(),
        covariance=model.geometry_covariance.tolist(), factor_id=fac.factor_id,
        sources=[dict(key=fac.keys[i], response=fac.jacobian[:, i].tolist(),
                      prior=float(fac.prior_variances[i]), center=float(fac.center[i])) for i in keep],
        log_depth_scale=fac.log_depth_scale)
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prepare_refresh(pg, result, factors, free_ids, fixed_ids):
    """Add persistent factor responses without mutating the live manager."""
    start = time.perf_counter()
    hm = pg.hypothesis_manager
    if set(pg.vertex_map) != set(hm.nodes):
        raise ValueError('Factor-basis geometry requires the complete connected map')
    if len(fixed_ids) != 1:
        raise ValueError('Factor-basis geometry requires one exact gauge anchor')
    anchor = next(iter(fixed_ids))
    fixed = pg.vertex_map[anchor].conditional_pose
    if (np.max(np.abs(fixed.geometry_covariance)) > 1e-12 or
            np.max(np.abs(fixed.factor.jacobian), initial=0.) > 1e-12):
        raise ValueError('Factor-basis geometry requires a deterministic gauge')
    if any(is_geometry_source(k) and not k.startswith(PREFIX) for k in pg.conditional_belief.keys):
        raise ValueError('Cannot reuse epoch-node geometry as persistent factor noise')
    geometry = [i for i, k in enumerate(pg.conditional_belief.keys) if is_geometry_source(k)]
    free = sorted(free_ids)
    positions = {key: 6 * i for i, key in enumerate(free)}
    permutation = [3, 4, 5, 0, 1, 2]
    A = sparse.lil_matrix((6 * len(factors), 6 * len(free)))
    factor_keys = []
    for row, (nonlinear, edge) in enumerate(factors):
        if np.max(np.abs(edge.conditional_pose.factor.jacobian[:, geometry]), initial=0.) > 1e-12:
            raise ValueError('Raw factors cannot re-observe a derived map posterior')
        factor_keys.extend(f'{PREFIX}{edge.geometry_factor_identity}:{d}' for d in range(6))
        jac = nonlinear.linearize(result)
        # Use the base-factor API; newer bindings need not downcast the
        # result of linearize() to JacobianFactor. Noise is already included.
        dense, _ = jac.jacobian()
        for column, key in enumerate(jac.keys()):
            if key in positions:
                offset = positions[key]
                A[6*row:6*row+6, offset:offset+6] = dense[:, 6*column:6*column+6][:, permutation]
            elif key not in fixed_ids:
                raise ValueError('Graph factor has an unspecified endpoint')
    if len(set(factor_keys)) != len(factor_keys):
        raise ValueError('One physical graph factor occurs twice in the committed graph')
    A = A.tocsc()
    H = -splu((A.T @ A).tocsc()).solve(A.T.toarray())
    assert np.isfinite(H).all()
    belief, _, _ = pg.conditional_belief.expand(
        SourceFactor(tuple(factor_keys), np.zeros((6, len(factor_keys))), np.ones(len(factor_keys))))
    locations = {key: i for i, key in enumerate(belief.keys)}
    columns = [locations[k] for k in factor_keys]
    old_geometry = [i for i, k in enumerate(belief.keys) if is_geometry_source(k)]
    for key, previous in list(pg.optimized_conditional_poses.items()):
        J, offset = belief.factor_response(previous.factor)
        assert np.max(np.abs(offset), initial=0.) < 1e-12
        J[:, old_geometry] = 0
        if key in positions:
            J[:, columns] = H[positions[key]:positions[key]+6]
        pg.optimized_conditional_poses[key] = ConditionalPose(np.zeros((6, 6)),
            SourceFactor(belief.keys, J, belief.prior_variances, belief.mean))
    pg.conditional_belief = belief
    return dict(keys=free, factor_keys=factor_keys, response=H,
                seconds=time.perf_counter()-start, factors=len(factors))


def stage_refresh(hm, pg, pending, current_pose, current_state):
    """Attach the optimized current-node model and preserve other branches."""
    import pypose as pp
    import torch
    from .conditional_pgo import _matrix

    latest = max(hm.nodes)
    match = next((item for item in pending if item[0].id == latest and item[1] == 0), None)
    if match is None:
        raise ValueError('Factor-basis refresh lacks the current graph node')
    _, _, pose, model, _ = match
    if np.linalg.norm(log(inverse(_matrix(pose)) @ current_pose)) > 1e-5:
        raise ValueError('Factor-basis refresh requires the current pose to be a graph node')
    updated, _ = current_state.with_pose(model)
    std = pp.se3(torch.as_tensor(np.sqrt(updated.marginal_covariance().diagonal().clip(0)),
        device=hm.device, dtype=hm.dist[1].dtype))
    data = pg.joint_geometry
    diagnostics = dict(geometry_variables=sum(is_geometry_source(k) for k in updated.keys),
        metric_variables=sum(not is_geometry_source(k) for k in updated.keys), nodes=len(hm.nodes),
        factors=data['factors'], seconds=data['seconds'], response_bytes=data['response'].nbytes,
        geometry_basis='persistent-whitened-raw-factor-v1',
        approximation='fixed-robust-linearization; independent raw factors; no temporary-node pruning',
        metric_belief_updated=False, retired_geometry_marginalized=False,
        other_modes_preserved=[i for i, s in enumerate(hm.source_states)
                               if s is not None and i not in {0, pg.conditional_component}])
    return pending, updated, std, diagnostics
