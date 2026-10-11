import math

import numpy as np

from cross_nav.plan import Field, risk_cost
from cross_nav.planability import evaluate_goals


def test_soft_diagonal_accounts_for_high_risk_side_cell():
    gt = np.ones((5, 5), np.uint8)
    gt[1, 1] = 2
    probability = np.zeros((5, 5), np.float32)
    probability[1, 1] = .99
    maps = {'risk': np.zeros_like(gt, bool)}
    costs = {'risk': risk_cost(probability, 1., robot_radius=0., weight=64)}
    kwargs = dict(robot_radius=0., costs=costs)
    old = evaluate_goals(gt, maps, (2, 1), [(1, 2)], 1., **kwargs)[0]
    new = evaluate_goals(gt, maps, (2, 1), [(1, 2)], 1., diagonal_cost=True, **kwargs)[0]
    assert old['feasibility'] == new['feasibility'] == 'planable'
    assert old['gt_path'] == new['gt_path']
    assert old['methods']['risk']['status'] == 'collision'
    assert new['methods']['risk']['status'] == 'success'
    assert new['methods']['risk']['path'] == [(2, 1), (2, 2), (1, 2)]


def test_uniform_cost_keeps_diagonal_distance():
    ok = np.ones((5, 5), bool)
    field = Field(ok, (1, 1), np.ones((5, 5)), allow_corner_cutting=False, diagonal_cost=True)
    assert field.path((2, 2)) == [(1, 1), (2, 2)]
    assert math.isclose(field.dist[2, 2], math.sqrt(2))


def test_side_risk_does_not_change_straight_edge_cost():
    ok = np.ones((5, 5), bool)
    cost = np.ones((5, 5))
    cost[1, 1] = 100
    field = Field(ok, (2, 1), cost, allow_corner_cutting=False, diagonal_cost=True)
    assert field.path((2, 2)) == [(2, 1), (2, 2)]
    assert field.dist[2, 2] == 1.


def test_hard_corner_constraint_still_blocks_diagonal():
    ok = np.eye(3, dtype=bool)
    field = Field(ok, (0, 0), diagonal_cost=True, allow_corner_cutting=False)
    assert field.path((1, 1)) is None
