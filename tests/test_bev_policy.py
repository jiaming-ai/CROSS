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
        assert fwd == 0. and 0 < abs(turn) <= math.radians(45) + 1e-9
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


# ------------------------------------------------------------------------------------------------ closed-loop rules
def _open(n=80, p_free=.02):
    return np.full((n, n), p_free, np.float32), np.full((n, n), .9, np.float32)


def test_odometry_pose_matches_sim_robot_frame():
    from cross_nav.policy.bev_policy import to_odom_2d, to_robot_2d
    from cross_nav.policy.sim.geom import to_robot
    from cross_nav.policy.sim.rollout import odom_pose
    rng = np.random.default_rng(0)
    for _ in range(5):
        pos, yaw, goal = rng.normal(size=3), rng.uniform(-3, 3), rng.normal(size=3) * 4
        ref = to_robot(goal[[0, 2]], pos[[0, 2]], yaw)
        pose = odom_pose(pos, yaw)
        got = to_robot_2d(odom_pose(goal, 0.)[:2], pose)
        np.testing.assert_allclose(got, ref, atol=1e-9)
        np.testing.assert_allclose(to_robot_2d(to_odom_2d(ref, pose), pose), ref, atol=1e-9)


def test_occ_block_forbids_crossing_a_predicted_wall():
    p, v = _open()
    p[40 - 12, :] = .97                                  # wall across the whole map 1.2 m ahead, everything observed
    goal = np.array([2.5, 0.])
    path, rec = plan_window(p, v, goal, .1, resolve({}))
    assert rec['status'] == STATUS_GOAL                  # frozen: occupancy is only a cost, the route crosses it
    path, rec = plan_window(p, v, goal, .1, resolve({'occ_block': .6}))
    assert rec['status'] != STATUS_GOAL
    if rec['status'] != STATUS_NONE:
        assert path[:, 0].max() < 1.2 - .2


def test_occ_block_lets_a_robot_touching_an_obstacle_leave():
    p, v = _open()
    p[40 - 2, 35:46] = .97                                # obstacle 0.15 m ahead, inside the robot's footprint
    path, rec = plan_window(p, v, np.array([-3., 0.]), .1, resolve({'occ_block': .6}))
    assert rec['status'] == STATUS_GOAL and path[-1, 0] < -2.5


def test_near_goal_approach():
    from cross_nav.policy.bev_policy import STATUS_APPROACH
    p, v = _open()
    v[:] = 0.                                            # nothing observed: the frozen planner finds no route
    goal = np.array([.3, .15])
    _, rec = plan_window(p, v, goal, .1, resolve({}))
    assert rec['status'] == STATUS_NONE
    cfg = resolve({'near_goal_m': 1.})
    path, rec = plan_window(p, v, goal, .1, cfg)
    assert rec['status'] == STATUS_APPROACH
    np.testing.assert_allclose(path[-1], goal, atol=1e-6)
    turn, fwd = control(path)
    assert fwd > 0 and turn > 0
    p[40 - 3:40 - 2, 38:43] = .9                         # obstacle on the segment: no straight approach
    _, rec = plan_window(p, v, goal, .1, cfg)
    assert rec['status'] != STATUS_APPROACH


def test_rotate_hold_keeps_the_turn_direction_of_a_streak():
    from cross_nav.policy.bev_policy import Episode
    p, v = _open()
    v[:] = 0.
    cfg = resolve({'no_path': 'rotate_hold', 'blind_zone': 'off'})
    st = Episode()
    sides = []
    for goal in ([-3., .5], [-3., -.5], [-3., .5], [-3., -.5]):
        path, rec = plan_window(p, v, np.array(goal), .1, cfg, st)
        assert rec['status'] == STATUS_NONE
        sides.append(np.sign(control(path)[0]))
    assert sides == [1, 1, 1, 1] and st.no_route == 4
    # stateless 'rotate_goal' flips with the goal side (the 2-cycle the hold prevents)
    sides = [np.sign(control(plan_window(p, v, np.array(g), .1, resolve({'no_path': 'rotate_goal', 'blind_zone': 'off'}))[0])[0])
             for g in ([-3., .5], [-3., -.5])]
    assert sides == [1, -1]


def test_memory_follows_the_last_route_in_the_odometry_frame():
    from cross_nav.policy.bev_policy import STATUS_MEMORY, Episode
    p, v = _open()
    cfg = resolve({'memory_steps': 3, 'occ_block': .6, 'blind_zone': 'off'})
    st = Episode()
    st.observe((1., 2., 0.), cfg)
    path, rec = plan_window(p, v, np.array([3., 1.]), .1, cfg, st)
    assert rec['status'] == STATUS_GOAL and st.route is not None
    # the robot moved 0.5 m forward and turned; the new window shows nothing: follow the remembered route
    st.observe((1.5, 2., .3), cfg)
    v0 = np.zeros_like(v)
    path2, rec = plan_window(p, v0, np.array([3., 1.]), .1, cfg, st)
    assert rec['status'] == STATUS_MEMORY
    from cross_nav.policy.bev_policy import to_odom_2d
    end = to_odom_2d(path2[-1], (1.5, 2., .3))
    np.testing.assert_allclose(end, [1. + path[-1, 0], 2. + path[-1, 1]], atol=.08)
    # a predicted obstacle on it, or age, ends the memory
    for _ in range(3):
        _, rec = plan_window(p, v0, np.array([3., 1.]), .1, cfg, st)
    assert rec['status'] == STATUS_NONE


def test_bump_marks_an_obstacle_where_commanded_motion_failed():
    from cross_nav.policy.bev_policy import Episode
    cfg = resolve({'bump': True})
    st = Episode()
    st.observe((0., 0., 0.), cfg)
    st.expect = .25
    st.observe((.02, 0., 0.), cfg)                        # moved 2 cm of 25: blocked
    assert st.n_bumps == 1
    p, _ = _open()
    q = st.add_bumps(p, .1, cfg)
    r, c = robot_to_cell([.3, 0.], 80, .1)
    assert q[r, c] == 1. and p[r, c] < .1
    st.expect = .25
    st.observe((.25, 0., 0.), cfg)                        # executed: no new bump
    assert st.n_bumps == 1


def test_relaxed_planner_after_a_no_route_streak():
    from cross_nav.policy.bev_policy import STATUS_RELAXED, Episode
    p, v = _open()
    v[:] = 0.
    cfg = resolve({'relax_after': 2, 'occ_block': .6, 'blind_zone': 'off'})
    st = Episode()
    goal = np.array([3., 0.])
    assert plan_window(p, v, goal, .1, cfg, st)[1]['status'] == STATUS_NONE
    assert plan_window(p, v, goal, .1, cfg, st)[1]['status'] == STATUS_NONE
    path, rec = plan_window(p, v, goal, .1, cfg, st)
    assert rec['status'] == STATUS_RELAXED and path[-1, 0] > 2.5
    assert st.no_route == 0


def test_astar_matches_dijkstra_length():
    from cross_nav.plan import Field
    from cross_nav.policy.bev_policy import astar
    rng = np.random.default_rng(1)
    ok = rng.random((60, 60)) > .25
    ok[5, 5] = ok[50, 52] = True
    cost = 1 + rng.random((60, 60)) * 3
    f = Field(ok, (5, 5), cost, allow_corner_cutting=False)
    path = astar(ok, cost, (5, 5), (50, 52))

    def length(pp):
        pp = np.asarray(pp)
        d = np.abs(np.diff(pp, axis=0)).sum(1)
        return sum((1.4142135623730951 if k == 2 else 1.) * .5 * (cost[tuple(a)] + cost[tuple(b)]) for k, a, b in zip(d, pp[:-1], pp[1:]))
    if np.isfinite(f.dist[50, 52]):
        assert len(path) and length(path) == pytest.approx(f.dist[50, 52], rel=1e-9)
        assert tuple(path[0]) == (5, 5) and tuple(path[-1]) == (50, 52)
    else:
        assert not len(path)


def test_world_map_fuses_in_the_odometry_frame_and_routes_around_a_seen_wall():
    from cross_nav.policy.bev_policy import STATUS_MAP, Episode, WorldMap, to_odom_2d
    n, res = 80, .1
    cfg = resolve({'map': {'res': .1, 'size': 200}, 'blind_zone': 'off'})
    pose = np.array([2., 1., math.radians(30)])
    p = np.full((n, n), .02, np.float32)
    v = np.full((n, n), .9, np.float32)
    # a wall 1.5 m ahead, 3 m wide (robot frame)
    i = (n / 2 - np.arange(n) - .5) * res
    F, L = np.meshgrid(i, i, indexing='ij')
    wall = (np.abs(F - 1.5) < .1) & (np.abs(L) < 1.5)
    p[wall] = .97
    wm = WorldMap(pose, cfg['map'])
    wm.fuse(p, v, pose, res)
    # the wall lands at the right odometry-frame place
    xy = to_odom_2d(np.array([1.5, 0.]), pose)
    vv, uu = wm.cell(xy)
    assert wm.num[vv, uu] / wm.den[vv, uu] > .8
    xy = to_odom_2d(np.array([.7, 0.]), pose)
    vv, uu = wm.cell(xy)
    assert wm.num[vv, uu] / wm.den[vv, uu] < .1
    # plan: around the wall to a goal 3 m ahead; a later window that sees nothing keeps the wall in memory
    st = Episode()
    st.observe(pose, cfg)
    st.world = wm
    wm.fuse(np.full((n, n), .3, np.float32), np.zeros((n, n), np.float32), pose, res)
    path, rec = plan_window(np.full((n, n), .3, np.float32), np.zeros((n, n), np.float32), np.array([3., 0.]), res, cfg, st)
    assert rec['status'] == STATUS_MAP
    fl = np.vstack(([0, 0], path))
    near_wall = np.abs(fl[:, 0] - 1.5) < .25
    assert near_wall.any() and np.abs(fl[near_wall, 1]).min() > 1.5


def test_empty_map_config_enables_the_map_with_defaults():
    from cross_nav.policy.bev_policy import MAP_DEFAULTS
    assert resolve({'map': {}})['map'] == MAP_DEFAULTS
    assert resolve({})['map'] is None


def test_world_map_unobserved_completions_never_block():
    from cross_nav.policy.bev_policy import WorldMap
    n, res = 80, .1
    cfg = resolve({'map': {'res': .1, 'size': 200}})['map']
    wm = WorldMap((0., 0., 0.), cfg)
    for _ in range(40):                                   # spinning in place, nothing observed, everything "occupied"
        wm.fuse(np.full((n, n), .8, np.float32), np.zeros((n, n), np.float32), (0., 0., 0.), res)
    route = wm.plan((0., 0., 0.), np.array([2., 1.]))
    assert route is not None and np.linalg.norm(route[-1] - [2., 1.]) < .15


def test_world_map_clearance_cost_keeps_away_from_walls():
    from cross_nav.policy.bev_policy import WorldMap
    n, res = 80, .1
    i = (n / 2 - np.arange(n) - .5) * res
    F, L = np.meshgrid(i, i, indexing='ij')
    p = np.where(np.abs(L + .45) < .1, .97, .02).astype(np.float32)     # a wall 0.45 m to the right, along the route
    v = np.full((n, n), .9, np.float32)
    dist = {}
    for cw in (0., 8.):
        wm = WorldMap((0., 0., 0.), resolve({'map': {'res': .1, 'size': 200, 'clear_m': .3, 'clear_weight': cw}})['map'])
        wm.fuse(p, v, (0., 0., 0.), res)
        route = wm.plan((0., 0., 0.), np.array([3., -.1]))
        mid = route[(route[:, 0] > .8) & (route[:, 0] < 2.2)]
        dist[cw] = float((mid[:, 1] + .45).min())
    assert dist[8.] > dist[0.]


def test_world_map_retry_searches_the_whole_grid_then_relaxes_blocking():
    from cross_nav.policy.bev_policy import WorldMap
    n, res = 80, .1
    i = (n / 2 - np.arange(n) - .5) * res
    F, L = np.meshgrid(i, i, indexing='ij')
    # a long wall between the robot and a goal 1 m ahead; its end is 3.5 m to the left (outside a 1 m search box)
    p = np.where((np.abs(F - .5) < .1) & (L < 3.5), .97, .02).astype(np.float32)
    v = np.full((n, n), .9, np.float32)
    for retry in (False, True):
        wm = WorldMap((0., 0., 0.), resolve({'map': {'res': .1, 'size': 200, 'margin_m': 1., 'retry': retry}})['map'])
        wm.fuse(p, v, (0., 0., 0.), res)
        route = wm.plan((0., 0., 0.), np.array([1., 0.]))
        assert (route is not None) == retry
    assert route[:, 1].max() > 3.5                         # around the wall's end
    # a closed wall: no route at block_p, a route when only P > retry_block_p blocks (here .8 < .85)
    p = np.where(np.abs(F - .5) < .1, .8, .02).astype(np.float32)
    wm = WorldMap((0., 0., 0.), resolve({'map': {'res': .1, 'size': 200, 'retry': True}})['map'])
    wm.fuse(p, v, (0., 0., 0.), res)
    assert wm.plan((0., 0., 0.), np.array([1., 0.])) is not None
    wm.c['retry'] = False
    assert wm.plan((0., 0., 0.), np.array([1., 0.])) is None
