"""CPU checks of the BEV planner policy's frame conventions, goal handling and path resampling (no checkpoints)."""
import math

import numpy as np
import pytest

from cross_nav.bev import BEVSpec, points_to_cells
from cross_nav.local_plan import AdaptiveVisibilityPlanner
from cross_nav.policy.bev_policy import (FROZEN_PLANNER, DEFAULTS, blind_zone_mask, STATUS_GOAL, STATUS_NONE, STATUS_WAYPOINT,
                                         adaptive_plan, cell_to_robot, cells_to_path, clip_goal, crop_box,
                                         plan_window, resolve, robot_to_cell, turn_path)
from cross_nav.policy.sim.geom import control


def test_robot_frame_matches_bev_frame():
    spec = BEVSpec(400, .05)
    fl = np.array([[3.02, 0.], [0., 2.01], [-1.5, -4.33], [9.97, -9.97]])
    # BEV frame: x = right = -left, y = forward
    bev = np.stack((-fl[:, 1], fl[:, 0], np.zeros(len(fl))), -1)
    row, col, ok = points_to_cells(bev, spec)
    assert ok.all()
    np.testing.assert_array_equal(robot_to_cell(fl, 400, .05), np.stack((row, col), -1))
    # forward is up the map (smaller rows), left is toward column 0
    r0, c0 = robot_to_cell([0., 0.], 400, .05)
    rf, cf = robot_to_cell([1., 0.], 400, .05)
    rl, cl = robot_to_cell([0., 1.], 400, .05)
    assert rf < r0 and cf == c0 and rl == r0 and cl < c0
    # cell centres round-trip, and cell_xy agrees with cell_to_robot
    x, y = spec.cell_xy()
    rc = np.array([[0, 0], [17, 391], [200, 200]])
    np.testing.assert_allclose(cell_to_robot(rc, 400, .05), np.stack((y[rc[:, 0], rc[:, 1]], -x[rc[:, 0], rc[:, 1]]), -1))
    np.testing.assert_array_equal(robot_to_cell(cell_to_robot(rc, 400, .05), 400, .05), rc)


def test_goal_clipping_keeps_direction():
    g = clip_goal([25., -5.], 10., .25)
    assert np.abs(g).max() == pytest.approx(9.75)
    assert math.atan2(g[1], g[0]) == pytest.approx(math.atan2(-5., 25.))
    np.testing.assert_array_equal(clip_goal([3., 4.], 10., .25), [3., 4.])
    rc = robot_to_cell(clip_goal([0., -40.], 10., .25), 400, .05)
    assert 0 <= rc.min() and rc.max() < 400


def test_cells_to_path_spacing_and_padding():
    # straight forward route of 3 m from the start cell (200, 200)
    cells = [(200 - k, 200) for k in range(61)]
    p = cells_to_path(cells, 400, .05)
    assert p.shape == (16, 2) and p.dtype == np.float32
    np.testing.assert_allclose(p[:6, 0], np.arange(1, 7) * .5, atol=1e-6)
    np.testing.assert_allclose(p[6:], np.tile([3., 0.], (10, 1)), atol=1e-6)
    np.testing.assert_allclose(p[:, 1], 0, atol=1e-6)
    # route to the left: positive left, columns decreasing
    p = cells_to_path([(200, 200 - k) for k in range(41)], 400, .05)
    np.testing.assert_allclose(p[-1], [0., 2.], atol=1e-6)
    np.testing.assert_allclose(p[0], [0., .5], atol=1e-6)


def test_turn_path_turns_in_place():
    for mode in ('rotate', 'rotate_goal'):
        turn, fwd = control(turn_path([1., -1.], mode))
    assert fwd == 0. and turn > 0 and abs(turn) <= math.radians(45) + 1e-9
    assert control(turn_path([1., -1.], 'rotate'))[0] > 0           # fixed direction: left
    assert control(turn_path([1., -1.], 'rotate_goal'))[0] < 0      # toward the goal side
    np.testing.assert_array_equal(turn_path([1., 0.], 'stop'), 0)


def test_crop_box_matches_training_crop():
    # 4:3 input at the training size 448 x 592: about 2.3 px are cut on each side
    assert crop_box(384, 512, (448, 592)) == (0, 384, 2, 509)
    assert crop_box(384, 512, (384, 512)) == (0, 384, 0, 512)


def _scene(n=80):
    """Free floor, observed forward wedge, a wall across the forward direction with a gap on the left."""
    p = np.full((n, n), .02, np.float32)
    v = np.zeros((n, n), np.float32)
    r = np.arange(n)[:, None] - n // 2
    c = np.arange(n)[None, :] - n // 2
    v[(r <= 0) & (r >= -25) & (np.abs(c) <= -r + 2)] = .9
    p[n // 2 - 12, :] = .98
    p[n // 2 - 12, n // 2 - 14:n // 2 - 8] = .02
    return p, v


def test_lazy_adaptive_plan_equals_frozen_planner():
    p, v = _scene()
    kw = {k: FROZEN_PLANNER[k] for k in ('radius', 'risk_weight', 'visibility_weight', 'min_travel', 'min_progress')}
    res = .1
    for goal in [(10, 40), (30, 38), (5, 70), (60, 40)]:
        ref = AdaptiveVisibilityPlanner(p, v, (40, 40), res, thresholds=FROZEN_PLANNER['thresholds'], **kw).plan(goal)
        got = adaptive_plan(p, v, goal, res, FROZEN_PLANNER)
        assert got['visibility_threshold'] == ref['visibility_threshold']
        assert got['path'] == ref['path'] and got['target'] == ref['target']
        assert got['stop_for_reobserve'] == ref['stop_for_reobserve']


def test_plan_window_routes_through_gap_and_handles_far_and_hidden_goals():
    p, v = _scene()
    cfg = resolve({})
    res = .1
    # visible goal beyond the wall: the route uses the gap on the left
    path, rec = plan_window(p, v, np.array([2.0, 0.]), res, cfg)
    assert rec['status'] in (STATUS_GOAL, STATUS_WAYPOINT)
    assert path.shape == (16, 2)
    assert np.all(np.linalg.norm(np.diff(np.vstack(([0, 0], path)), axis=0), axis=1) <= .5 + 1e-5)
    cells = adaptive_plan(p, v, robot_to_cell([2.0, 0.], 80, res), res, cfg['planner'])['path']
    crossing = [c for r, c in cells if r == 28]                # the wall row: only through the gap (columns 26-31)
    assert crossing and all(26 <= c <= 31 for c in crossing)
    assert path[:, 1].max() > .7                               # the returned path detours to the left
    # far goal straight ahead: projected onto the map, a local waypoint making progress forward
    path, rec = plan_window(p, v, np.array([25., 0.]), res, cfg)
    assert np.abs(rec['goal_used']).max() == pytest.approx(4. - cfg['goal_margin_m'])
    assert rec['status'] == STATUS_WAYPOINT and rec['target'][0] >= .5
    # goal behind the robot: nothing observed there -> turn in place
    path, rec = plan_window(p, v, np.array([-3., 0.]), res, cfg)
    assert rec['status'] == STATUS_NONE and np.isnan(rec['target']).all()
    assert control(path)[1] == 0.


def test_config_paths_expand_env(monkeypatch):
    monkeypatch.setenv('CROSS_NAV_ROOT', '/x/root')
    cfg = resolve({'occupancy': '${CROSS_NAV_ROOT}/occ.pt', 'planner': {'visibility_weight': 0.}})
    assert cfg['occupancy'] == '/x/root/occ.pt'
    assert cfg['planner']['risk_weight'] == 128. and cfg['planner']['visibility_weight'] == 0.
    assert tuple(cfg['history_m']) == tuple(DEFAULTS['history_m'])
    monkeypatch.delenv('CROSS_NAV_ROOT')
    with pytest.raises(ValueError):
        resolve({'occupancy': '${CROSS_NAV_ROOT}/occ.pt'})


def test_blind_zone_fallback_only_when_frozen_planner_is_stuck():
    n, res = 80, .1
    p = np.full((n, n), .02, np.float32)
    v = np.zeros((n, n), np.float32)
    f = (n // 2 - np.arange(n) - .5)[:, None] * res + 0 * np.arange(n)[None]
    v[(f > 1.1) & (f < 3.5)] = .9                 # observed floor starts beyond the camera's blind zone
    goal = np.array([3., 0.])
    _, rec = plan_window(p, v, goal, res, resolve({'blind_zone': 'off'}))
    assert rec['status'] == STATUS_NONE           # frozen rule: the unobserved band ahead disconnects the robot
    path, rec = plan_window(p, v, goal, res, resolve({}))
    assert rec['blind_zone'] and rec['status'] == STATUS_GOAL
    assert path[-1, 0] == pytest.approx(3., abs=.11) and abs(path[-1, 1]) < .11
    # with observed floor everywhere ahead the fallback is not used and the result is the frozen one
    v2 = np.where(f > 0, .9, 0.).astype(np.float32)
    a, ra = plan_window(p, v2, goal, res, resolve({'blind_zone': 'off'}))
    b, rb = plan_window(p, v2, goal, res, resolve({}))
    assert not rb['blind_zone'] and ra['status'] == rb['status'] and np.array_equal(a, b)
    z = blind_zone_mask(n, res, 1.3, 90., .2)
    fl = cell_to_robot(np.argwhere(z), n, res)
    assert fl[:, 0].max() <= 1.3 and fl[:, 0].min() >= -.2 and np.all(np.abs(fl[:, 1]) <= np.maximum(fl[:, 0], .3) + 1e-9)
