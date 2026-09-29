"""Actual graph solves and multi-mode transport in the optional factor basis."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import pypose as pp

from cross.core.conditional import SourceFactor
from cross.core.conditional_pose import ConditionalPose, inverse, adjoint
from cross.core.conditional_pgo import _matrix
from cross.core.map_geometry import validate_load_mode as validate_geometry_mode
from cross.core.schmidt import schmidt_product
from cross.core.types import EdgeType
from test_conditional_graph import chain, solve, apply_graph, manager


def validate_load_mode(saved, enabled):
    return validate_geometry_mode(saved, enabled, basis='factor')


def dense(covariance):
    return covariance.to_dense() if hasattr(covariance, "to_dense") else covariance


def make_chain():
    hm = chain()
    hm.schmidt_map_geometry = True
    hm.map_geometry_basis = "factor"
    hm.source_states[1] = None
    hm.nodes[0].conditional_poses[0].geometry_covariance[:] = 0
    return hm


def test_factor_response_matches_independent_odometry_covariance():
    hm = make_chain()
    pg = solve(hm)
    rows = []
    for i in range(1, 11):
        row = np.zeros((6, 60))
        for j in range(1, i + 1):
            after = inverse(_matrix(pg.optimized_poses[j])) @ _matrix(pg.optimized_poses[i])
            row[:, 6*(j-1):6*j] = adjoint(inverse(after))
        rows.append(row)
    response = np.vstack(rows)
    Q = np.kron(np.eye(10), hm.odom_edges[0, 1].conditional_pose.geometry_covariance)
    actual = pg.joint_geometry['response']
    np.testing.assert_allclose(actual @ actual.T, response @ Q @ response.T, rtol=1e-5, atol=1e-9)
    assert apply_graph(hm, pg)['success']


def test_repeated_solve_reuses_noise_priors_and_keeps_metric_cross_covariance():
    hm = make_chain()
    assert apply_graph(hm, solve(hm))['success']
    state = hm.source_states[0]
    node = hm.nodes[3].conditional_poses[0]
    fac = SourceFactor(state.keys, node.factor.jacobian, state.prior_variances, state.mean, 'new-query')
    hm.source_states[0] = schmidt_product(state, np.zeros(6), np.eye(6)*.001, fac,
        frozen_keys=state.keys[1:]).state
    before = hm.source_states[0].copy()
    assert np.linalg.norm(dense(before.covariance)[0, 1:]) > 1e-7
    assert apply_graph(hm, solve(hm))['success']
    after = hm.source_states[0]
    assert after.keys == before.keys and after.seen_factors == before.seen_factors
    np.testing.assert_array_equal(after.mean, before.mean)
    np.testing.assert_array_equal(dense(after.covariance), dense(before.covariance))
    np.testing.assert_array_equal(dense(after.covariance)[1:, 1:], np.eye(60))


def test_unrealized_competing_mode_survives_a_committed_graph_refresh():
    hm = make_chain()
    assert apply_graph(hm, solve(hm))['success']
    hm.source_states[1] = hm.source_states[0].copy()
    hm.source_states[1].mean[0] = .06
    hm.source_states[1].covariance.active[0, 0] = .007
    hm.dist[0][1] = hm.dist[0][0].clone()
    hm.dist[0][1, 1] += .2
    hm.dist[2][:] = torch.tensor([.7, .3])
    hm.component_charts[1] = hm.component_charts[0]
    hm.ttl[1] = 7
    hm.llr_hist[1, 2] = .4
    before = hm.source_states[1].copy()
    weight, ttl, history = hm.dist[2][1].clone(), hm.ttl[1].clone(), hm.llr_hist[1].clone()
    pose = hm.dist[0][1].clone()
    pg = solve(hm)
    info = apply_graph(hm, pg)
    assert info['success']
    assert info['conditional_response']['schmidt_map_geometry']['other_modes_preserved'] == [1]
    after = hm.source_states[1]
    assert after.keys == before.keys
    np.testing.assert_array_equal(after.mean, before.mean)
    np.testing.assert_array_equal(dense(after.covariance), dense(before.covariance))
    np.testing.assert_allclose(after.marginal_covariance(), before.marginal_covariance(), atol=1e-8)
    torch.testing.assert_close(hm.dist[0][1], pose, atol=1e-5, rtol=0)
    assert hm.dist[2][1] == weight and hm.ttl[1] == ttl
    torch.testing.assert_close(hm.llr_hist[1], history, rtol=0, atol=0)


def test_first_refresh_aligns_source_columns_before_transporting_other_mode():
    hm = make_chain()
    hm.source_states[1] = hm.source_states[0].copy()
    hm.dist[0][1] = hm.dist[0][0].clone()
    hm.dist[2][:] = torch.tensor([.7, .3])
    hm.component_charts[1] = hm.component_charts[0]
    before = hm.source_states[1].copy()
    pg = solve(hm)
    assert apply_graph(hm, pg)['success']
    # Both poses are equal before the graph transport, so the newly shared
    # geometry response must be exactly the solved current node's response.
    expected = pg.optimized_conditional_poses[10].factor
    assert expected.keys == hm.source_states[1].keys
    np.testing.assert_allclose(hm.source_states[1].jacobian[:, 1:], expected.jacobian[:, 1:], atol=1e-7)
    np.testing.assert_array_equal(hm.source_states[1].geometry_covariance, before.geometry_covariance)


def test_new_graph_factor_appends_only_six_independent_coordinates():
    hm = make_chain()
    assert apply_graph(hm, solve(hm))['success']
    old = hm.source_states[0].copy()
    measurement = hm.nodes[0].pose_mu[0].Inv() @ hm.nodes[10].pose_mu[0]
    model = ConditionalPose(np.eye(6)*.04,
        SourceFactor((), np.zeros((6, 0)), np.empty(0), factor_id='independent-loop'))
    hm.add_edge(0, 10, measurement, pp.se3(torch.full((6,), .2)), EdgeType.VISUAL, conditional_pose=model)
    assert apply_graph(hm, solve(hm))['success']
    new = hm.source_states[0]
    assert new.keys[:len(old.keys)] == old.keys and len(new.keys) == len(old.keys) + 6
    np.testing.assert_array_equal(dense(new.covariance)[:len(old.keys), :len(old.keys)], dense(old.covariance))
    np.testing.assert_array_equal(dense(new.covariance)[-6:, -6:], np.eye(6))
    np.testing.assert_array_equal(dense(new.covariance)[:-6, -6:], np.zeros((len(old.keys), 6)))


def test_saved_raw_edges_reproduce_factor_identity_after_loading():
    hm = make_chain()
    assert apply_graph(hm, solve(hm))['success']
    old = hm.source_states[0].copy()
    saved = copy.deepcopy(hm.save_state())
    other = manager()
    other.schmidt_map_geometry = True
    other.map_geometry_basis = "factor"
    other.load_state(saved, SimpleNamespace(), 'cpu', 'cpu', {})
    other.initialize_source_filter()
    node = other.nodes[10]
    other.dist = tuple(v.clone() for v in (node.pose_mu, node.pose_std, node.pose_weights))
    other.component_charts[:] = node.pose_charts
    other.source_states[0], _ = other.saved_source_belief.with_pose(node.conditional_poses[0])
    other.source_states[1] = None
    other.system.last_added_kf_id = 10
    assert apply_graph(other, solve(other))['success']
    assert other.source_states[0].keys == old.keys
    np.testing.assert_array_equal(dense(other.source_states[0].covariance), dense(old.covariance))
    validate_load_mode(dict(schmidt_map_geometry_version=2, hypo_data=saved), True)
    with pytest.raises(ValueError, match='epoch-node'):
        validate_load_mode(dict(schmidt_map_geometry_version=1), True)


def test_persistent_geometry_contract_requires_explicit_mode_and_known_coordinates():
    with pytest.raises(ValueError,match='enable schmidt_map_geometry'):
        validate_load_mode(dict(schmidt_map_geometry_version=2),False)
    validate_load_mode(dict(schmidt_map_geometry_version=2),True)
    with pytest.raises(ValueError,match='previous geometry'):
        validate_load_mode(dict(hypo_data=dict(source_belief=dict(keys=['geometry:old:1:0']))),True)


def test_persistent_geometry_still_rejects_a_stochastic_gauge():
    hm=make_chain()
    hm.nodes[0].conditional_poses[0].geometry_covariance[:]=np.eye(6)*.01
    with pytest.raises(ValueError,match='deterministic gauge'):
        solve(hm)


def test_public_configuration_requires_explicit_schmidt_mode():
    from cross.core.config import HypothesisConfig
    from cross.core.hypothesis import HypothesisManager
    from cross.mono.config import MonoConfig

    assert MonoConfig().map_geometry_basis == 'epoch'
    assert HypothesisConfig().map_geometry_basis == 'epoch'
    with pytest.raises(ValueError, match='requires Schmidt'):
        MonoConfig(map_geometry_basis='factor')
    system = manager().system
    with pytest.raises(ValueError, match='requires Schmidt'):
        HypothesisManager(system, 2, HypothesisConfig(map_geometry_basis='factor'))
    common = dict(conditional_sources=True, schmidt_map_geometry=True,
                  chart_aware=True, session_recovery=True, map_geometry_basis='factor')
    mono = MonoConfig(frontend='streaming_pnp', retrieval_pose='metric_two_view',
                      retrieval_matcher='superpoint_lightglue', **common)
    core = HypothesisManager(system, 2, HypothesisConfig(**common))
    assert mono.map_geometry_basis == core.map_geometry_basis == 'factor'


def test_public_loader_rejects_mixing_geometry_bases_even_without_version_marker():
    for saved in (dict(schmidt_map_geometry_version=2),
                  dict(hypo_data=dict(source_belief=dict(keys=['geometry:factor:edge:0'])))):
        with pytest.raises(ValueError, match='version|factor geometry basis'):
            validate_geometry_mode(saved, True, basis='epoch')
        validate_geometry_mode(saved, True, basis='factor')
    for saved in (dict(schmidt_map_geometry_version=1),
                  dict(hypo_data=dict(source_belief=dict(keys=['geometry:epoch:1:0'])))):
        with pytest.raises(ValueError, match='epoch-node|previous geometry'):
            validate_geometry_mode(saved, True, basis='factor')
        validate_geometry_mode(saved, True, basis='epoch')


def test_factor_refresh_rejects_live_displacement_without_mutating_other_modes():
    hm = make_chain()
    hm.source_states[1] = hm.source_states[0].copy()
    hm.dist[0][0, 0] += .03
    pg = solve(hm)
    sources = [s.record() for s in hm.source_states]
    poses = [n.pose_mu.clone() for n in hm.nodes.values()]
    with pytest.raises(ValueError, match='current pose to be a graph node'):
        apply_graph(hm, pg)
    assert [s.record() for s in hm.source_states] == sources
    for node, pose in zip(hm.nodes.values(), poses):
        torch.testing.assert_close(node.pose_mu, pose, atol=0, rtol=0)
