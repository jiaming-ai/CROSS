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


def test_robot_agent_keeps_long_history_at_camera_rate():
    from cross_nav.policy.robot import NavigationAgent
    seen = {}

    def planner(images, goal, hist, trail=None):
        seen['hist'], seen['trail'], seen['n'] = hist, trail, len(images)
        return np.array([[.5 * (i + 1), 0.] for i in range(16)])

    agent = NavigationAgent(planner)
    agent.set_goal((100., 0.))
    img = np.zeros((4, 4, 3), np.uint8)
    for k in range(400):                                   # 20 m straight at 0.05 m per camera frame (10 Hz, 0.5 m/s)
        out = agent.step(img, (.05 * k, 0., 0.))
    np.testing.assert_allclose(seen['hist'], [0, .5, 1, 2], atol=.11)
    assert seen['n'] == 4 and seen['trail'][:, 2].all()   # the 16 m trail point is available
    np.testing.assert_allclose(seen['trail'][4, :2], [-16, 0], atol=.11)
    assert len(agent.frames) < 250 and out['cmd'][0] > 0


def test_robot_agent_replans_every_k_frames_and_follows_the_plan_between():
    from cross_nav.policy.robot import NavigationAgent
    calls = []

    def planner(images, goal, hist, trail=None):
        calls.append(1)
        return np.array([[.5 * (i + 1), 0.] for i in range(16)])

    agent = NavigationAgent(planner, replan_every=3)
    agent.set_goal((100., 0.))
    img = np.zeros((4, 4, 3), np.uint8)
    outs = [agent.step(img, (.05 * k, 0., 0.)) for k in range(9)]
    assert len(calls) == 3 and [o['replanned'] for o in outs[:4]] == [True, False, False, True]
    assert outs[1]['cmd'][0] > 0 and abs(outs[1]['cmd'][1]) < 1e-9          # follows the stored straight plan
    np.testing.assert_allclose(outs[2]['path'][0], [.4, 0.], atol=1e-9)     # the plan's points still ahead


def _fake_server(conn, log):
    """The policy server's frame buffer rules (cross_nav.policy.server): store each step's frame, drop frames older
    than the window; a predicting step needs every window frame."""
    buf = {}
    while True:
        m = wire.recv(conn)
        if m['cmd'] == 'end':
            wire.send(conn, {}); return
        buf[m['frame_idx']] = m['frame']
        for k in [k for k in buf if k < min(m['window'])]:
            del buf[k]
        if m['predict']:
            assert all(j in buf for j in m['window']), (m['window'], sorted(buf))
            log.append(len(m['window']))
            g = m['goal'] / max(np.linalg.norm(m['goal']), 1e-6)
            wire.send(conn, {'path': np.outer(np.arange(1, 17) * .5, g), 'debug': None})
        else:
            log.append(0)
            wire.send(conn, {'path': None, 'debug': None})


def test_socket_planner_drives_the_robot_agent_over_the_wire():
    import threading
    from cross_nav.policy.robot import NavigationAgent
    from cross_nav.policy.sim.rollout import SocketPlanner, odometry_step
    a, b = Pipe()
    log = []
    t = threading.Thread(target=_fake_server, args=(b, log)); t.start()
    info = {'history_m': [0., .5, 1., 2.], 'history_rule': None, 'bev': None}
    planner = SocketPlanner(a, 'ep', info, {})
    agent = NavigationAgent(planner, v_max=.5, w_max=1.)
    agent.set_goal((3., 2.))
    odom, rng = np.zeros(3), np.random.default_rng(0)
    for k in range(200):
        img = np.full((4, 4, 3), k % 256, np.uint8)                       # distinct frames
        out = agent.step(img, tuple(odom))
        if out['arrived']:
            break
        v, w = out['cmd']
        odom = odometry_step(odom, (v * .1, 0.), w * .1, rng, 0., np.zeros(2))
    planner.end(); t.join()
    assert out['arrived']
    assert 4 in log and log.count(0) < len(log) // 2                       # uploads only for window frames not yet held


def test_odometry_step_integrates_and_drifts():
    from cross_nav.policy.sim.rollout import odometry_step
    rng = np.random.default_rng(0)
    o = np.zeros(3)
    o = odometry_step(o, (1., 0.), math.pi / 2, rng, 0., np.zeros(2))       # 1 m forward, then facing left
    np.testing.assert_allclose(o, [1., 0., math.pi / 2], atol=1e-12)
    o = odometry_step(o, (.5, .2), 0., rng, 0., np.zeros(2))                 # 0.5 m forward (+y), 0.2 m slide left (-x)
    np.testing.assert_allclose(o, [.8, .5, math.pi / 2], atol=1e-12)
    o2 = odometry_step(np.zeros(3), (1., 0.), 0., rng, .05, np.array([.05, 0.]))
    assert not np.allclose(o2[:2], [1., 0.])


def test_stuck_recovery_turns_in_place_then_returns_control():
    from cross_nav.policy.recovery import StuckRecovery
    rec, path = StuckRecovery(window=5, min_disp=.3, turn_deg=90.), np.array([[.5 * (i + 1), 0.] for i in range(16)])
    for k in range(4):                                       # moving normally: the policy path passes through
        assert rec((.25 * k, 0., 0.), path) is path
    for _ in range(5):                                       # stuck at one spot
        out = rec((1., 0., 0.), path)
    assert out[0, 0] == 0 and out[0, 1] > 0                  # turn left in place
    assert rec((1., 0., math.radians(45)), path)[0, 1] > 0   # still turning
    assert rec((1., 0., math.radians(90)), path) is path     # turned: control back to the policy
    for _ in range(5):
        out = rec((1., 0., math.radians(90)), path)
    assert out[0, 1] < 0                                     # second recovery turns the other way


def test_robot_agent_reports_commanded_motion_to_the_bev_planner():
    from cross_nav.policy.robot import NavigationAgent

    class Planner:
        uses_odometry, history_m, history_rule = True, (0., .5, 1., 2.), None

        def __init__(self): self.expect = []

        def __call__(self, images, goal, hist, odom=None, window_odom=None, expect=None):
            self.expect.append(expect)
            return np.array([[.5 * (i + 1), 0.] for i in range(16)])

    pl = Planner()
    agent = NavigationAgent(pl, v_max=.5, replan_every=2)
    agent.set_goal((100., 0.))
    img = np.zeros((4, 4, 3), np.uint8)
    for k in range(5):                                      # blocked: commands forward, odometry stays put
        agent.step(img, (0., 0., 0.), .1 * k)
    assert pl.expect[0] == 0. and np.allclose(pl.expect[1:], [.1, .1])   # 2 ticks x 0.1 s x 0.5 m/s per plan
    agent.reset(); pl.expect.clear()
    agent.set_goal((100., 0.)); agent.step(img, (0., 0., 0.))
    assert pl.expect == [None]                              # no stamps: the planner keeps its own rule


def test_ai2thor_dataset_switch(tmp_path):
    from cross_nav.policy.sim import hab
    (tmp_path / 'ai2thor-hab.scene_dataset_config.json').write_text('{}')
    d = tmp_path / 'configs/scenes/ProcTHOR/1'
    d.mkdir(parents=True)
    for n in ('ProcTHOR-Train-2', 'ProcTHOR-Train-10'):
        (d / f'{n}.scene_instance.json').write_text('{}')
    assert hab.dataset_kind(str(tmp_path)) == 'ai2thor' and hab.dataset_name(str(tmp_path)) == 'ai2thor_sim'
    assert hab.scene_config(str(tmp_path)).endswith('ai2thor-hab.scene_dataset_config.json')
    assert hab.ai2thor_scenes(str(tmp_path)) == ['ProcTHOR-Train-10', 'ProcTHOR-Train-2']
    rooms = hab.Rooms(str(tmp_path), 'ProcTHOR-Train-2')
    assert rooms.indoor((0., .15, 0.)) and rooms.indoor((100., .1, -5.))
    assert not rooms.indoor((0., 2.75, 0.)) and not rooms.indoor((0., .8, 0.))   # ceiling top, furniture top
    assert hab.is_ai2thor_door('Doorway_Double_3_:0000') and hab.is_ai2thor_door('ArchitecTHOR-Val-00-Doorway_39f930c1_:0001')
    assert not hab.is_ai2thor_door('Doorframe_2_:0000') and not hab.is_ai2thor_door('Sofa_3_:0000')
    hssd = tmp_path / 'hssd'
    hssd.mkdir()
    assert hab.dataset_kind(str(hssd)) == 'hssd' and hab.dataset_name(str(hssd)) == 'hssd_sim'
    assert hab.scene_config(str(hssd)).endswith('hssd-hab.scene_dataset_config.json')


def test_real_cache_cast_splits_follow_the_gnm_recordings():
    from cross_nav.policy.real_cache import split_of
    assert split_of('cast', 'jackal_x', 'scand') is None and split_of('cast', 'tartan_20210826', 'tartan_drive') is None
    for g in [f'no{k}vc' for k in range(1, 40)]:            # GNM noNvcF is the same recording group as noNvc
        assert split_of('cast', g + 'F', 'go_stanford') == split_of('go_stanford', g)
    assert split_of('cast', 'Dec-09-2022-bww8', 'sac_son') == split_of('sac_son', 'Dec-09-2022-bww8')
    assert split_of('cast', 'anything', 'recon') == 'test' == split_of('recon', 'anything')


def test_real_cache_skips_incomplete_episodes(tmp_path):
    import json as js
    from PIL import Image
    from cross_nav.policy.real_cache import windows
    def episode(name, frames):
        d = tmp_path / name; d.mkdir()
        n = 40
        np.savez(d / 'trajectory.npz', position=np.stack([np.arange(n) * .5, np.zeros(n)], -1), yaw=np.zeros(n))
        (d / 'source.json').write_text(js.dumps({'group': 'g'}))
        for i in range(frames):
            Image.new('RGB', (8, 6)).save(d / f'{i}.jpg')
        return d
    rng = np.random.default_rng(0)
    assert len(windows('x', episode('full', 40), rng, 400)) > 0
    assert windows('x', episode('cut', 30), rng, 400) == []           # frames missing
    bad = episode('badnpz', 40); (bad / 'trajectory.npz').write_bytes(b'xx')
    assert windows('x', bad, rng, 400) == []


def test_stuck_recovery_time_window_ignores_planning_rate():
    from cross_nav.policy.recovery import StuckRecovery
    path = np.array([[.5 * (i + 1), 0.] for i in range(16)])
    rec = StuckRecovery(window=100, window_s=10.)
    for k in range(95):                                      # 9.4 s turning in place at 10 Hz: not yet stuck
        assert rec((0., 0., .05 * k), path, stamp=.1 * k) is path
    for k in range(95, 101):
        out = rec((0., 0., 0.), path, stamp=.1 * k)
    assert out is not path and out[0, 0] == 0                # 10 s without displacement: turn in place
