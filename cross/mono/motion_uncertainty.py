"""Conservative motion bounds with unknown correlation between input frames.

For n zero-mean errors with arbitrary joint covariance, Cov(sum e_i) is at
most n * sum Cov(e_i), in PSD order (Cauchy--Schwarz in every direction).
Transport the full moments to the receiving camera before making a diagonal
envelope for CROSS's inherited motion interface. Dropping world-frame cross
terms earlier introduces an unnecessary dependence on the chart origin.

This is a first-order bound on declared error marginals, not their calibration.
Metric teacher bias is represented separately by the identified source state.
"""
import numpy as np


def diagonal_envelope(covariance):
    """A diagonal PSD upper bound, retaining off-diagonal mass conservatively."""
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (6, 6) or not np.isfinite(covariance).all():
        raise ValueError('Motion covariance must be a finite 6x6 matrix')
    covariance = (covariance + covariance.T) / 2
    # Apply diagonal dominance in correlation coordinates, then restore units.
    # A plain row sum of a mixed metre/radian covariance would make this bound
    # depend on the numerical choice of translation units.
    if np.linalg.eigvalsh(covariance).min() < -1e-10 * max(1., np.linalg.norm(covariance)):
        raise ValueError('Motion covariance lost positive semidefiniteness')
    variances = covariance.diagonal()
    if np.any(variances < 0):
        raise ValueError('Motion covariance has negative marginal variance')
    support = variances > 0
    if np.any(covariance[~support] != 0):
        raise ValueError('Deterministic motion coordinates have nonzero cross covariance')
    std = np.sqrt(variances[support])
    bound = np.zeros(6)
    bound[support] = std * (np.abs(covariance[np.ix_(support, support)]) @ (1/std))
    return np.diag(bound)


def correlated_interval_bound(previous_prefix, current_prefix, world_to_current, steps):
    """Bound all intervening increments, including replaced mapping images.

    Prefixes contain sums of full world-tangent covariance matrices. Their
    difference is transported with the current camera's inverse adjoint.
    `steps` is the number of camera increments, not the number of mapper jobs.
    """
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('A motion interval needs a positive camera-step count')
    previous, current, transport = map(lambda x: np.asarray(x, dtype=np.float64),
                                       (previous_prefix, current_prefix, world_to_current))
    if any(x.shape != (6, 6) or not np.isfinite(x).all() for x in (previous, current, transport)):
        raise ValueError('Motion prefixes and transport must be finite 6x6 matrices')
    full = steps * (transport @ (current - previous) @ transport.T)
    return diagonal_envelope(full)
