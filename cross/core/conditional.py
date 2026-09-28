"""Conditional Gaussian pose/source factors for CROSS's mixture update.

All six-dimensional quantities are expressed in one declared local tangent.
The caller owns SE3 retraction, source identity, geometric verification and
association. A factor is conditional on the source variables; its prior is
introduced once in SourceState, never multiplied in again with a saved node.
This module models shared scale errors, not all map/pixel correlations.
"""

from dataclasses import dataclass, field
import math

import numpy as np


def _symmetric(value):
    return (value + value.T) * .5


@dataclass
class SourceFactor:
    keys: tuple[str, ...]
    jacobian: np.ndarray
    prior_variances: np.ndarray
    center: np.ndarray | None = None
    factor_id: str | None = None
    log_depth_scale: bool = False

    def __post_init__(self):
        self.keys = tuple(self.keys)
        self.jacobian = np.array(self.jacobian, dtype=np.float64, copy=True)
        self.prior_variances = np.array(self.prior_variances, dtype=np.float64, copy=True)
        self.center = np.zeros(len(self.keys)) if self.center is None else np.array(self.center, dtype=np.float64, copy=True)
        if len(set(self.keys)) != len(self.keys) or any(not isinstance(k, str) or not k for k in self.keys):
            raise ValueError("Source keys must be unique nonempty strings")
        if (self.jacobian.shape != (6, len(self.keys)) or self.prior_variances.shape != (len(self.keys),)
                or self.center.shape != (len(self.keys),)):
            raise ValueError("Source factor dimensions do not match its keys")
        if (not all(np.isfinite(v).all() for v in (self.jacobian, self.prior_variances, self.center))
                or np.any(self.prior_variances < 0)):
            raise ValueError("Source factor values must be finite with nonnegative prior variances")
        if self.factor_id is not None and (not isinstance(self.factor_id, str) or not self.factor_id):
            raise ValueError("A geometric factor identity must be a nonempty string")
        if self.log_depth_scale and np.any(self.jacobian[3:] != 0):
            raise ValueError('An exact log-depth response can only scale translation')

    def record(self):
        return dict(keys=list(self.keys), jacobian=self.jacobian.tolist(),
                    prior_variances=self.prior_variances.tolist(), center=self.center.tolist(),
                    factor_id=self.factor_id,log_depth_scale=self.log_depth_scale)

    @classmethod
    def from_record(cls, record):
        return cls(tuple(record['keys']), np.asarray(record['jacobian']).reshape(6,-1),
                   record['prior_variances'], record['center'], record.get('factor_id'),record.get('log_depth_scale',False))


@dataclass
class SourceState:
    """b~N(mean,covariance), pose|b~N(mu+J(b-mean),geometry_covariance).

    The pose mean lives in HypothesisManager.dist. Full conditional covariance
    and pose/source cross-covariance stay here; exporting a diagonal pose
    marginal to legacy consumers must not discard these sufficient statistics.
    Dense bias covariance is an experimental reference representation, with
    quadratic memory. It is not a bounded-map implementation.
    """
    geometry_covariance: np.ndarray
    keys: tuple[str, ...] = ()
    mean: np.ndarray = field(default_factory=lambda: np.empty(0))
    covariance: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    jacobian: np.ndarray = field(default_factory=lambda: np.empty((6, 0)))
    prior_variances: np.ndarray = field(default_factory=lambda: np.empty(0))
    seen_factors: frozenset[str] = frozenset()

    def __post_init__(self):
        self.keys = tuple(self.keys)
        self.seen_factors = frozenset(self.seen_factors)
        for name in ("geometry_covariance", "mean", "covariance", "jacobian", "prior_variances"):
            setattr(self, name, np.array(getattr(self, name), dtype=np.float64, copy=True))
        n = len(self.keys)
        if (self.geometry_covariance.shape != (6, 6) or self.mean.shape != (n,)
                or self.covariance.shape != (n, n) or self.jacobian.shape != (6, n)
                or self.prior_variances.shape != (n,) or len(set(self.keys)) != n):
            raise ValueError("Conditional state dimensions do not match its keys")
        if not all(np.isfinite(getattr(self, name)).all() for name in
                   ("geometry_covariance", "mean", "covariance", "jacobian", "prior_variances")):
            raise ValueError("Conditional state must be finite")
        if np.any(self.prior_variances < 0):
            raise ValueError("Source priors cannot have negative variance")
        if not np.allclose(self.geometry_covariance, self.geometry_covariance.T, rtol=0, atol=1e-12):
            raise ValueError("Conditional pose covariance must be symmetric")
        if np.linalg.eigvalsh(self.geometry_covariance).min() < -1e-12:
            raise ValueError("Conditional pose covariance must be positive semidefinite")
        if not np.allclose(self.covariance, self.covariance.T, rtol=0, atol=1e-12):
            raise ValueError("Source covariance must be symmetric")
        if np.any(self.covariance.diagonal() < -1e-12):
            raise ValueError("Source covariance has a negative marginal variance")

    def copy(self):
        return SourceState(self.geometry_covariance, self.keys, self.mean, self.covariance,
                           self.jacobian, self.prior_variances, self.seen_factors)

    def record(self):
        return dict(version=1, geometry_covariance=self.geometry_covariance.tolist(),
                    keys=list(self.keys), mean=self.mean.tolist(), covariance=self.covariance.tolist(),
                    jacobian=self.jacobian.tolist(), prior_variances=self.prior_variances.tolist(),
                    seen_factors=sorted(self.seen_factors))

    @classmethod
    def from_record(cls, record):
        if record.get('version') != 1:
            raise ValueError('Unsupported conditional source state version')
        n = len(record['keys'])
        state = cls(record['geometry_covariance'], tuple(record['keys']), record['mean'],
                    np.asarray(record['covariance']).reshape(n,n),
                    np.asarray(record['jacobian']).reshape(6,n), record['prior_variances'],
                    frozenset(record['seen_factors']))
        if n and np.linalg.eigvalsh(state.covariance).min() < -1e-10:
            raise ValueError('Saved source covariance is not positive semidefinite')
        return state

    def with_pose(self, model):
        """Attach a conditional pose to this bias belief, without new evidence."""
        result, observed_J, offset = self.expand(model.factor)
        result.geometry_covariance = model.geometry_covariance.copy()
        result.jacobian = observed_J
        return result, offset

    def expand(self, factor):
        """Append previously unseen priors, preserving every old correlation."""
        existing = set(self.keys)
        keys = self.keys + tuple(k for k in factor.keys if k not in existing)
        locations = {k: i for i, k in enumerate(keys)}
        n, old = len(keys), len(self.keys)
        mean, covariance, jacobian, variances = np.zeros(n), np.zeros((n,n)), np.zeros((6,n)), np.zeros(n)
        mean[:old], covariance[:old,:old] = self.mean, self.covariance
        jacobian[:,:old], variances[:old] = self.jacobian, self.prior_variances
        for key, variance in zip(factor.keys, factor.prior_variances):
            i = locations[key]
            if i < old:
                if not np.isclose(variances[i], variance, rtol=1e-12, atol=1e-15):
                    raise ValueError(f"A reused source cannot acquire a different prior: {key}")
            else:
                covariance[i,i] = variances[i] = variance
        result = SourceState(self.geometry_covariance, keys, mean, covariance, jacobian, variances, self.seen_factors)
        observed_J = np.zeros((6,n))
        centered_offset = np.zeros(6)
        for j, key in enumerate(factor.keys):
            i = locations[key]
            displacement = mean[i]-factor.center[j]
            multiplier = np.exp(-displacement) if factor.log_depth_scale else 1.
            observed_J[:,i] = factor.jacobian[:,j]*multiplier
            centered_offset += factor.jacobian[:,j] * ((1.-multiplier) if factor.log_depth_scale else displacement)
        return result, observed_J, centered_offset

    def marginal_covariance(self):
        return _symmetric(self.geometry_covariance + self.jacobian @ self.covariance @ self.jacobian.T)

    def seed(self, factor, geometry_covariance):
        """Initialize a newborn pose from H(x|b), without observing b twice."""
        result, observed_J, offset = self.expand(factor)
        result.geometry_covariance = np.array(geometry_covariance, dtype=np.float64, copy=True)
        result.jacobian = observed_J
        if factor.factor_id is not None:
            result.seen_factors = result.seen_factors | {factor.factor_id}
        return result, offset


@dataclass
class ConditionalProduct:
    state: SourceState
    pose_offset: np.ndarray
    log_overlap: float | None
    innovation_covariance: np.ndarray | None
    duplicate: bool = False


def conditional_product(prior, residual, observation_covariance, factor, process_covariance=None):
    """Integrate the shared bias once and retain its posterior cross-covariance.

    residual is Log(mu_prior^-1 mu_observation), with observation mean at
    factor.center. This is a local Gaussian factor on a fixed geometric fit,
    not a derivative through RANSAC or a declaration of independent pixels.
    Replaying an identified identical factor contributes no information.
    """
    if factor.factor_id is not None and factor.factor_id in prior.seen_factors:
        return ConditionalProduct(prior.copy(), np.zeros(6), None, None, duplicate=True)
    prior, observed_J, offset = prior.expand(factor)
    residual = np.asarray(residual, dtype=np.float64) + offset
    S = prior.geometry_covariance.copy()
    if process_covariance is not None:
        S += np.asarray(process_covariance, dtype=np.float64)
    R = np.asarray(observation_covariance, dtype=np.float64)
    if residual.shape != (6,) or R.shape != (6,6) or not np.isfinite(residual).all() or not np.isfinite(R).all():
        raise ValueError("A conditional pose factor requires finite 6D geometry")
    if not np.allclose(R,R.T,rtol=0,atol=1e-12) or np.linalg.eigvalsh(R).min() < -1e-12:
        raise ValueError("Observation covariance must be symmetric positive semidefinite")
    base_covariance = _symmetric(S+R)
    difference = prior.jacobian-observed_J
    innovation = _symmetric(base_covariance+difference@prior.covariance@difference.T)
    # Cholesky rejects invalid covariance instead of silently dropping a source
    # correlation or treating a singular factor as arbitrarily strong evidence.
    chol = np.linalg.cholesky(innovation)
    np.linalg.cholesky(base_covariance)
    bias_gain = np.linalg.solve(innovation, difference@prior.covariance).T
    updated_mean = prior.mean+bias_gain@residual
    updated_covariance = _symmetric(prior.covariance-bias_gain@difference@prior.covariance)
    conditional_gain = np.linalg.solve(base_covariance, S.T).T
    updated_J = prior.jacobian+conditional_gain@(observed_J-prior.jacobian)
    updated_S = _symmetric(S-conditional_gain@S)
    pose_offset = conditional_gain@residual+updated_J@(updated_mean-prior.mean)
    whitened = np.linalg.solve(chol,residual)
    overlap = -.5*(float(whitened@whitened)+2*float(np.log(np.diag(chol)).sum())+6*math.log(2*math.pi))
    seen = prior.seen_factors | ({factor.factor_id} if factor.factor_id is not None else set())
    posterior = SourceState(updated_S,prior.keys,updated_mean,updated_covariance,updated_J,
                            prior.prior_variances,seen)
    return ConditionalProduct(posterior,pose_offset,overlap,innovation)
