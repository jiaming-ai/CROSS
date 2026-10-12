import math
from multiprocessing import Pipe

import numpy as np
import torch

from cross_nav.policy.head import TrajectoryHead
from cross_nav.policy.sim import wire
from cross_nav.policy.sim.geom import axes, control, history_ids, resample, to_robot, to_world


def test_axes_match_habitat_yaw():
    # yaw 0 faces -Z with left -X; +90 deg faces -X with left +Z (rotation about +Y)
    f, l = axes(0.)
    np.testing.assert_allclose(f, [0, -1], atol=1e-12); np.testing.assert_allclose(l, [-1, 0], atol=1e-12)
    f, l = axes(math.pi / 2)
    np.testing.assert_allclose(f, [-1, 0], atol=1e-12); np.testing.assert_allclose(l, [0, 1], atol=1e-12)
    pos, yaw = np.array([1., -2.]), .7
    pts = np.random.default_rng(0).normal(size=(5, 2))
    np.testing.assert_allclose(to_robot(to_world(pts, pos, yaw), pos, yaw), pts, atol=1e-12)


def test_resample_pads_with_end():
    p = resample(np.array([[0, 0], [3, 0], [3, 4]]), 16, .5)
    np.testing.assert_allclose(p[5], [3, 0]); np.testing.assert_allclose(p[7], [3, 1])
    np.testing.assert_allclose(p[13:], [[3, 4]] * 3)
    np.testing.assert_allclose(resample(np.array([[0., 0.]]), 4), 0)


def test_history_is_causal_by_travel():
    c = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.25]
    assert history_ids(c) == [9, 7, 5, 1]
    assert history_ids([0.]) == [0, 0, 0, 0]
    # turning in place adds frames without travel: the 0.5 m view stays the last frame 0.5 m back
    assert history_ids([0, .5, .5, .5]) == [3, 0, 0, 0]


def test_controller_turns_then_moves_without_overshoot():
    assert control(np.zeros((16, 2))) == (0., 0.)
    turn, fwd = control(np.array([[0., .5]] * 16))           # aim 90 deg left: turn in place
    assert turn > 0 and fwd == 0
    turn, fwd = control(resample(np.array([[0, 0], [8, 0]])))
    assert turn == 0 and math.isclose(fwd, .25)
    turn, fwd = control(np.array([[.1, 0.]] * 16))            # goal 10 cm ahead: step only 10 cm
    assert math.isclose(fwd, .1)


def test_wire_roundtrip_arrays():
    a, b = Pipe()
    msg = {'frame': np.arange(12, dtype=np.uint8).reshape(2, 2, 3), 'goal': np.array([1.5, -2], np.float32),
           'window': [3, 2, 1, 0], 'label': None, 'meta': {'x': np.float32(2.5)}}
    wire.send(a, msg)
    got = wire.recv(b)
    np.testing.assert_array_equal(got['frame'], msg['frame']); assert got['frame'].dtype == np.uint8
    np.testing.assert_array_equal(got['goal'], msg['goal']); assert got['window'] == [3, 2, 1, 0]
    assert got['label'] is None and got['meta']['x'] == 2.5


def test_polar_goal_far_goals_stay_bounded():
    torch.manual_seed(0)
    m = TrajectoryHead(token_dim=16, width=24, heads=4, layers=1, source='last', horizon=16, action_scale=8.,
                       goal_encoding='polar')
    e = m.encode_goal(torch.tensor([[25., 0.], [0., 0.]]))
    assert torch.isfinite(e).all() and e[0, 3] == 1 and e.abs().max() < 4
    x = torch.randn(2, 1, 4, 3, 4, 16)
    loss = m.loss(x, torch.tensor([[25., 1.], [2., -1.]]), torch.randn(2, 16, 2))
    loss.backward()
    assert torch.isfinite(loss) and m.goal[0].weight.grad.abs().sum() > 0


def test_keyframe_window_includes_turn_views():
    from cross_nav.policy.sim.geom import window_ids
    cum = [0, 0, 0, 0, .25, .5, .75, 1.]
    yaw = [0, .6, 1.2, 1.8, 1.8, 1.8, 1.8, 1.8]          # three 34 deg turns in place, then straight
    rule = {'type': 'keyframe', 'trans': .5, 'rot_deg': 30, 'lags': [0, 1, 2, 3, 4, 8]}
    ids = window_ids(cum, yaw, rule)
    assert ids[0] == 7 and ids[1] == 5 and ids[2:5] == [3, 2, 1] and ids[5] == 0
    assert window_ids(cum, yaw, {'type': 'distance', 'm': [0, .5, 1, 2]}) == [7, 5, 3, 0]


def test_trail_points_and_head_tokens():
    from cross_nav.policy.sim.geom import trail
    cum = [0., 1., 2., 3.]
    plane = [[0., 0.], [0., -1.], [0., -2.], [0., -3.]]        # driving along -z (yaw 0 faces -z)
    t = trail(cum, plane, np.array([0., -3.]), 0.)
    np.testing.assert_allclose(t[0], [-1, 0, 1], atol=1e-6)     # 1 m behind
    np.testing.assert_allclose(t[1], [-2, 0, 1], atol=1e-6)
    assert t[3, 2] == 0 and t[4, 2] == 0                         # 8 m / 16 m not travelled yet
    m = TrajectoryHead(token_dim=16, width=24, heads=4, layers=1, source='last', horizon=16, action_scale=8.,
                       goal_encoding='polar', trail=5)
    x = torch.randn(2, 1, 4, 3, 4, 16)
    loss = m.loss(x, torch.tensor([[5., 1.], [2., -1.]]), torch.randn(2, 16, 2), trail=torch.from_numpy(np.stack([t, t])))
    loss.backward()
    assert m.trail_embed[0].weight.grad.abs().sum() > 0
    assert m.memory(x).shape[1] == 4 * 3 * 4 + 5


def test_medoid_sampling_follows_previous_plan():
    from cross_nav.policy.server import Policy
    torch.manual_seed(0)
    pol = object.__new__(Policy)
    pol.dev, pol.scale, pol.head_pool, pol.steps = torch.device('cpu'), 'mono', 1, 2
    pol.gen = torch.Generator().manual_seed(0)
    pol.head = TrajectoryHead(token_dim=16, width=24, heads=4, layers=1, source='last', horizon=16, action_scale=8.,
                              goal_encoding='polar', mode='flow').eval()
    feat = {'tokens': torch.randn(2, 4, 3, 4, 16), 'xyz': torch.randn(2, 4, 3, 4, 3), 'log_scale': torch.zeros(2),
            'pred_chord': torch.ones(2, 4)}
    goal, hist = torch.tensor([[5., 1.], [2., -1.]]), torch.zeros(2, 4)
    one = pol.predict(feat, goal, hist, samples=6)
    assert one.shape == (2, 16, 2) and torch.isfinite(one).all()
    prev = torch.full((2, 16, 2), float('nan')); prev[0] = 50.          # far plan: commitment cannot pick a closer sample
    two = pol.predict(feat, goal, hist, samples=6, prev=prev, commit=1.)
    assert two.shape == (2, 16, 2) and torch.isfinite(two).all()
