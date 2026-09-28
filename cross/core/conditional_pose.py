"""Saved H(x|b) messages and SE3 composition, without a second bias prior.

The map retains conditional geometry and a linear pose response at a declared
bias center. Bias means/correlations belong to the live hypothesis (and one
committed map belief). Conditional pixel/map errors are still approximated as
independent; this representation only addresses the declared shared sources.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .conditional import SourceFactor, SourceState, transport_covariance


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
        factor = SourceFactor(state.keys,transport@J,state.prior_variances,state.mean,self.factor.factor_id)
        return pose @ exp(offset), ConditionalPose(transport_covariance(self.geometry_covariance,transport),factor), state


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
