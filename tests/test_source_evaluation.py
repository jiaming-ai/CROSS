"""Read-only graph evaluation must retain owned state and identical inference."""
import numpy as np
import pytest

from cross.core.conditional import SourceFactor, SourceState
from cross.core.conditional_pose import ConditionalPose, exp


@pytest.mark.parametrize('log_depth_scale', [False, True])
def test_known_factor_matches_expansion_without_copying_or_exposing_belief(log_depth_scale):
    rng = np.random.default_rng(513)
    for _ in range(15):
        matrix = rng.normal(size=(8,8))*.03
        V = matrix@matrix.T+np.eye(8)*.012
        state = SourceState(np.eye(6)*.1, tuple(f'image:{i}' for i in range(8)),
            rng.normal(size=8)*.02, V, rng.normal(size=(6,8))*.1, np.ones(8)*.0144)
        ids = [5,2,7]
        response = rng.normal(size=(6,3))*.05
        if log_depth_scale:
            response[3:] = 0
        factor = SourceFactor(tuple(state.keys[i] for i in ids), response,
            state.prior_variances[ids], rng.normal(size=3)*.02, 'one-physical-pair', log_depth_scale)
        original = state.record()
        _, reference_J, reference_offset = state.expand(factor)
        J, offset = state.factor_response(factor)
        np.testing.assert_array_equal(J, reference_J)
        np.testing.assert_array_equal(offset, reference_offset)
        model = ConditionalPose(np.eye(6)*.04, factor)
        pose = exp(rng.normal(size=6)*.05)
        expected_pose, expected_model, _ = model.at(pose, state)
        actual_pose, actual_model = model.at_known(pose, state)
        np.testing.assert_array_equal(actual_pose, expected_pose)
        assert actual_model.record() == expected_model.record()
        # Mutating evaluation results must never mutate the live source belief.
        J[:] = 100;offset[:] = 100
        actual_model.factor.jacobian[:] = 100
        actual_model.factor.center[:] = 100
        assert state.record() == original


def test_batch_extension_matches_sequential_prior_order_covariance_and_ownership():
    state = SourceState(np.eye(6), ('a','b'), np.array([.1,.2]),
        np.array([[1.,.3],[.3,2.]]), np.arange(12).reshape(6,2)*.01, np.array([1.,2.]),
        frozenset(['old-measurement']))
    factors = [SourceFactor(('b','z','a'), np.ones((6,3)), np.array([2.,3.,1.])),
               SourceFactor(('w','z'), np.ones((6,2)), np.array([4.,3.]))]
    expected = state.copy()
    for factor in factors:
        expected, _, _ = expected.expand(factor)
    actual = state.expand_many(iter(factors))
    assert actual.record() == expected.record()
    assert actual.keys == ('a','b','z','w')
    actual.covariance[0,0] = 100
    assert state.covariance[0,0] == 1.
    empty = state.expand_many([])
    assert empty.record() == state.record()
    empty.mean[0] = 100
    assert state.mean[0] == .1


def test_readonly_and_batch_operations_reject_missing_or_conflicting_priors_atomically():
    state = SourceState(np.eye(6), ('a',), np.zeros(1), np.ones((1,1)), np.ones((6,1)), np.ones(1))
    original = state.record()
    missing = SourceFactor(('new',), np.ones((6,1)), np.ones(1))
    conflict = SourceFactor(('a',), np.ones((6,1)), np.array([2.]))
    with pytest.raises(ValueError, match='all source priors'):
        state.factor_response(missing)
    with pytest.raises(ValueError, match='different prior'):
        state.factor_response(conflict)
    with pytest.raises(ValueError, match='different prior'):
        state.expand_many([missing, conflict])
    changed_new = SourceFactor(('new',), np.ones((6,1)), np.array([2.]))
    with pytest.raises(ValueError, match='different prior'):
        state.expand_many([missing, changed_new])
    assert state.record() == original
