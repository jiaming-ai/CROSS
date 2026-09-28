"""Schmidt conditioning for declared persistent nuisance variables.

This is the standard consider-filter approximation: camera/metric variables
remain active, while mature map means and their covariance block stay fixed.
Cross-covariances must still change. The dense factorization here is a reference
implementation, not a bounded-map or novel filtering algorithm.
"""
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .conditional import SourceState, ConditionalProduct, conditional_product


def _response(covariance, cross):
    """Factor a joint Gaussian, including deterministic source coordinates."""
    active = np.flatnonzero(covariance.diagonal() > 0)
    result = np.zeros_like(cross)
    if not len(active):
        if np.max(np.abs(cross),initial=0.) > 1e-12:
            raise ValueError('A deterministic source cannot have pose cross-covariance')
        return result
    V = covariance[np.ix_(active,active)]
    C = cross[:,active]
    try:
        result[:,active] = cho_solve(cho_factor(V,lower=True,check_finite=False),
                                     C.T,check_finite=False).T
    except np.linalg.LinAlgError:
        # An exactly shared/deterministic source prior can be semidefinite.
        # Conditional response outside its support is immaterial; do not add
        # a made-up independent prior or jitter to make it invertible.
        values,vectors = np.linalg.eigh(V)
        tolerance = np.finfo(float).eps*max(1.,float(np.max(np.abs(values))))*len(values)*10
        if values.min() < -tolerance:
            raise ValueError('Schmidt source covariance is not positive semidefinite')
        positive = values > tolerance
        result[:,active] = ((C@vectors[:,positive])/values[positive])@vectors[:,positive].T
    if not np.allclose(result@covariance,cross,rtol=1e-8,atol=1e-10):
        raise ValueError('Pose/source cross-covariance lies outside the source support')
    return result


def schmidt_product(prior, residual, observation_covariance, factor,
                    process_covariance=None, frozen_keys=()):
    """Condition in one local chart with zero gain for named nuisance states.

    Return the same conditional pose/source representation as
    ``conditional_product``. Freezing a mean alone while shrinking its covariance
    is inconsistent; use the full Joseph covariance expression and retain all
    active/nuisance cross-covariances. An empty frozen set is the full update.
    """
    frozen_keys = frozenset(frozen_keys)
    if not frozen_keys:
        return conditional_product(prior,residual,observation_covariance,factor,process_covariance)
    if factor.factor_id is not None and factor.factor_id in prior.seen_factors:
        return ConditionalProduct(prior.copy(),np.zeros(6),None,None,duplicate=True)
    prior,A,offset = prior.expand(factor)
    if not frozen_keys.issubset(prior.keys):
        raise ValueError('Every frozen source must exist in the expanded belief')
    frozen = np.array([key in frozen_keys for key in prior.keys])
    residual = np.asarray(residual,dtype=np.float64)+offset
    S = prior.geometry_covariance.copy()
    if process_covariance is not None:
        S += np.asarray(process_covariance,dtype=np.float64)
    R = np.asarray(observation_covariance,dtype=np.float64)
    if (residual.shape != (6,) or R.shape != (6,6) or
            not np.isfinite(residual).all() or not np.isfinite(R).all()):
        raise ValueError('A Schmidt pose factor requires finite 6D geometry')
    if not np.allclose(R,R.T,rtol=0,atol=1e-12) or np.linalg.eigvalsh(R).min() < -1e-12:
        raise ValueError('Observation covariance must be symmetric positive semidefinite')
    D = prior.jacobian-A
    U = prior.covariance@D.T
    innovation = S+R+D@U
    innovation = (innovation+innovation.T)/2
    chol = np.linalg.cholesky(innovation)
    source_gain = np.linalg.solve(innovation,U.T).T
    source_gain[frozen] = 0
    pose_cross = S+prior.jacobian@U
    pose_gain = np.linalg.solve(innovation,pose_cross.T).T
    mean = prior.mean+source_gain@residual
    V = (prior.covariance-source_gain@U.T-U@source_gain.T
         +source_gain@innovation@source_gain.T)
    V = (V+V.T)/2
    pose_covariance = S+prior.jacobian@prior.covariance@prior.jacobian.T-pose_gain@pose_cross.T
    pose_covariance = (pose_covariance+pose_covariance.T)/2
    cross = prior.jacobian@prior.covariance-pose_gain@U.T
    J = _response(V,cross)
    conditional = pose_covariance-J@cross.T
    conditional = (conditional+conditional.T)/2
    seen = prior.seen_factors|({factor.factor_id} if factor.factor_id is not None else set())
    posterior = SourceState(conditional,prior.keys,mean,V,J,prior.prior_variances,seen)
    whitened = np.linalg.solve(chol,residual)
    overlap = -.5*(float(whitened@whitened)+2*float(np.log(np.diag(chol)).sum())+6*math.log(2*math.pi))
    return ConditionalProduct(posterior,pose_gain@residual,overlap,innovation)
