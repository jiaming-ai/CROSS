"""Saved H(x|b) messages and SE3 composition, without a second bias prior.

The map retains conditional geometry and a linear pose response at a declared
bias center. Bias means/correlations belong to the live hypothesis (and one
committed map belief). Conditional pixel/map errors are still approximated as
independent; this representation only addresses the declared shared sources.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .conditional import (SourceFactor, SourceState, ConditionalProduct,
                          conditional_product, transport_covariance)


def skew(v):
    x,y,z = v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def adjoint(T):
    A = np.zeros((6,6))
    A[:3,:3] = A[3:,3:] = T[:3,:3]
    A[:3,3:] = skew(T[:3,3]) @ T[:3,:3]
    return A


def inverse(T):
    result = np.eye(4)
    result[:3,:3] = T[:3,:3].T
    result[:3,3] = -result[:3,:3] @ T[:3,3]
    return result


def normalize_mean(pose):
    """Keep float32 group products on SE3 before matrix/log-chart conversion.

    Repeated quaternion -> matrix -> quaternion cycles otherwise amplify
    roundoff. Only normalize an already finite, near-unit quaternion; reject
    a materially invalid pose instead of projecting arbitrary matrices.
    """
    import pypose as pp
    import torch
    data = pose.tensor().clone()
    norm = torch.linalg.vector_norm(data[...,3:],dim=-1,keepdim=True)
    if not torch.isfinite(data).all() or torch.any((norm-1).abs()>1e-3):
        raise ValueError('Conditional pose mean has a non-unit or non-finite quaternion')
    data[...,3:] /= norm
    return pp.SE3(data)


def right_jacobian(twist):
    """d Log(Exp(x)^-1 Exp(x+dx))/d dx, [translation,rotation] order."""
    twist = np.asarray(twist, dtype=np.float64)
    ad = np.zeros((6,6))
    ad[:3,:3] = ad[3:,3:] = skew(twist[3:])
    ad[:3,3:] = skew(twist[:3])
    J, term = np.eye(6), np.eye(6)
    for n in range(1,60):
        term = term @ (-ad)/(n+1)
        J += term
        if np.max(np.abs(term)) < 1e-15:
            return J
    raise ValueError('SE3 response exceeds the local linearization domain')


def retract_frozen_response(keys, response, offset, prior_response=None):
    """Retain a frozen map action through a finite right-tangent correction.

    For zero-mean frozen geometry g, use the local conditional pose family
    P Exp(J0 g) Exp(d + (J1-J0) g). Its response at P Exp(d) is
    Ad(Exp(-d)) J0 + Jr(d) (J1-J0). This preserves a common rigid map action
    when an observation has no information about it (J1 == J0), including
    when a saved message is evaluated at an updated metric-bias mean.

    ``response`` is J1; ``prior_response`` is J0 and defaults to J1 for saved
    message evaluation. Active metric sources keep their additive-tangent
    response. This is a first-order conditional model, not a global Gaussian
    approximation for arbitrarily large map uncertainty.
    """
    T = right_jacobian(offset)
    result = T @ response
    frozen = np.array([k.startswith('geometry:') for k in keys])
    if frozen.any():
        before = response if prior_response is None else prior_response
        result[:, frozen] += (adjoint(exp(-np.asarray(offset))) - T) @ before[:, frozen]
    return result


def exp(twist):
    twist = np.asarray(twist, dtype=np.float64)
    result = np.eye(4)
    result[:3,:3] = Rotation.from_rotvec(twist[3:]).as_matrix()
    result[:3,3] = right_jacobian(-twist)[:3,:3] @ twist[:3]
    return result


def log(T):
    rotation = Rotation.from_matrix(T[:3,:3]).as_rotvec()
    v = np.linalg.solve(right_jacobian(-np.r_[np.zeros(3),rotation])[:3,:3], T[:3,3])
    return np.r_[v,rotation]


def residual_product(prior, residual, observation_covariance, factor, process_covariance=None,
                     frozen_keys=()):
    """Condition a right-tangent pose on a relative SE3 residual.

    Call after evaluating the observation model at ``prior.mean``. For
    r=Log(P^-1 O), the residual Jacobians are -Jl(r)^-1 and Jr(r)^-1.
    Transport BOTH uncertain poses to that residual chart, condition there,
    then return the pose response in P's right tangent. Using the identity
    for the prior Jacobian can spuriously observe a shared rigid map shift.
    This is first-order Gaussian conditioning, not an exact Lie-group density.
    """
    if factor.factor_id is not None and factor.factor_id in prior.seen_factors:
        return ConditionalProduct(prior.copy(),np.zeros(6),None,None,duplicate=True)
    if (factor.keys != prior.keys or not np.array_equal(factor.center,prior.mean)
            or factor.log_depth_scale):
        raise ValueError('Evaluate and align the observation at the current source belief first')
    residual = np.asarray(residual,dtype=np.float64)
    back = right_jacobian(-residual)
    F = np.linalg.inv(back)
    G = np.linalg.inv(right_jacobian(residual))
    common_prior = SourceState(transport_covariance(prior.geometry_covariance,F),
        prior.keys,prior.mean,prior.covariance,F@prior.jacobian,
        prior.prior_variances,prior.seen_factors)
    common_factor = SourceFactor(factor.keys,G@factor.jacobian,factor.prior_variances,
                                 factor.center,factor.factor_id)
    process = None if process_covariance is None else transport_covariance(process_covariance,F)
    if frozen_keys:
        from .schmidt import schmidt_product
        result = schmidt_product(common_prior,residual,
            transport_covariance(observation_covariance,G),common_factor,process,frozen_keys)
    else:
        result = conditional_product(common_prior,residual,
            transport_covariance(observation_covariance,G),common_factor,process)
    result.state.geometry_covariance = transport_covariance(result.state.geometry_covariance,back)
    result.state.jacobian = back@result.state.jacobian
    result.pose_offset = back@result.pose_offset
    return result


@dataclass
class ConditionalPose:
    """Right-tangent pose model; its mean is stored by its node or edge.

    S contains conditional geometry, not marginal source uncertainty. There
    is deliberately no bias covariance here: saved nodes must not each supply
    another copy of the same metric prior.
    """
    geometry_covariance: np.ndarray
    factor: SourceFactor

    def __post_init__(self):
        # Reuse covariance validation without inventing a bias belief.
        self.geometry_covariance = SourceState(self.geometry_covariance).geometry_covariance
        self.factor = SourceFactor.from_record(self.factor.record())

    @classmethod
    def from_state(cls, state):
        return cls(state.geometry_covariance,
                   SourceFactor(state.keys,state.jacobian,state.prior_variances,state.mean))

    def record(self):
        return dict(version=1, geometry_covariance=self.geometry_covariance.tolist(), factor=self.factor.record())

    @classmethod
    def from_record(cls, record):
        if record.get('version') != 1:
            raise ValueError('Unsupported conditional pose version')
        return cls(record['geometry_covariance'], SourceFactor.from_record(record['factor']))

    def at(self, pose, belief):
        """Evaluate the frozen local model at a live belief's bias mean.

        Rebase both J and S into the returned right tangent. This is a local
        response, conditional on the selected geometric fit; no RANSAC rerun.
        """
        state, J, offset = belief.expand(self.factor)
        transport = right_jacobian(offset)
        factor = SourceFactor(state.keys,retract_frozen_response(state.keys,J,offset),state.prior_variances,state.mean,self.factor.factor_id)
        return pose @ exp(offset), ConditionalPose(transport_covariance(self.geometry_covariance,transport),factor), state

    def at_known(self, pose, belief):
        """Evaluate a pose without copying an already complete source belief.

        Graph preparation expands the union of its raw priors first. No source
        mean/covariance is returned or shared with a mutable tracking state.
        """
        J, offset = belief.factor_response(self.factor)
        transport = right_jacobian(offset)
        factor = SourceFactor(belief.keys, retract_frozen_response(belief.keys,J,offset), belief.prior_variances,
                              belief.mean, self.factor.factor_id)
        return pose @ exp(offset), ConditionalPose(
            transport_covariance(self.geometry_covariance, transport), factor)


def compose(first_pose, first, second_pose, second, belief):
    """Convolve a saved node's conditional message with a relative-pose fit."""
    # Expand once for both factors, so their source columns align and shared
    # sources add with signs before any marginalization.
    belief,_,_ = belief.expand(first.factor)
    belief,_,_ = belief.expand(second.factor)
    a,first,_ = first.at(first_pose,belief)
    b,second,_ = second.at(second_pose,belief)
    A = adjoint(inverse(b))
    factor = SourceFactor(belief.keys,A@first.factor.jacobian+second.factor.jacobian,
                          belief.prior_variances,belief.mean,second.factor.factor_id)
    S = transport_covariance(first.geometry_covariance,A)+second.geometry_covariance
    return a@b, ConditionalPose(S,factor)


def records(models):
    return None if models is None else [m.record() if m is not None else None for m in models]


def restore(records):
    return None if records is None else [ConditionalPose.from_record(m) if m is not None else None for m in records]
