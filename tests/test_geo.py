"""Geo anchoring: geodesy, map<->ENU anchor, GNSS quality control, compass."""
import math

import numpy as np
import pytest

from cross.geo.geodesy import LocalFrame, ecef_to_lla, lla_to_ecef
from cross.geo.anchor import Anchor, AnchorConfig, rot_z, rotation_between
from cross.geo.gnss import GnssFix, GnssGate, GnssGateConfig, GnssNoiseModel, FIX_3D, FIX_NONE
from cross.geo.compass import Compass, CompassConfig, tilt_compensated_yaw


def test_geodesy_round_trip():
    rng = np.random.default_rng(0)
    lla = np.stack([rng.uniform(-80, 80, 200), rng.uniform(-180, 180, 200), rng.uniform(-100, 3000, 200)], 1)
    back = ecef_to_lla(lla_to_ecef(lla[:, 0], lla[:, 1], lla[:, 2]))
    assert np.abs(back[:, :2] - lla[:, :2]).max() < 1e-9
    assert np.abs(back[:, 2] - lla[:, 2]).max() < 1e-4
    f = LocalFrame(42.29, -83.71, 270.0)
    enu = f.to_enu(lla[:5, 0] * 0 + 42.30, lla[:5, 1] * 0 - 83.70, 280.0)
    # ~1.11 km north, ~0.82 km east of the origin
    assert abs(enu[0, 1] - 1112) < 5 and abs(enu[0, 0] - 823) < 5
    assert np.abs(f.to_lla(enu) - np.array([42.30, -83.70, 280.0])).max() < 1e-7


def _camera_track(n=300, seed=0):
    """A ground robot's camera positions in a camera-convention map frame (y down, z forward at the start)."""
    rng = np.random.default_rng(seed)
    yaw = np.cumsum(rng.normal(0, 0.03, n))
    step = 1.0
    xy = np.cumsum(np.stack([np.cos(yaw), np.sin(yaw)], 1) * step, 0)
    # levelled coordinates (x east-ish, y north-ish, z up) -> map frame (camera: x right, y down, z forward)
    lev = np.stack([xy[:, 0], xy[:, 1], rng.normal(0, 0.05, n)], 1)
    R_map_from_lev = rotation_between(np.array([0, 0, 1.0]), np.array([0, -1.0, 0]))
    return lev @ R_map_from_lev.T, lev


def test_anchor_4dof_with_outliers():
    p_map, lev = _camera_track()
    yaw_true, t_true = math.radians(37.0), np.array([120.0, -45.0, 3.0])
    a = Anchor(AnchorConfig(dof=4), vertical_map=np.array([0, -1.0, 0]))
    p_enu_true = (rot_z(yaw_true) @ (a.R_level @ p_map.T)).T + t_true
    rng = np.random.default_rng(1)
    p_enu = p_enu_true + rng.normal(0, 3.0, p_enu_true.shape) * np.array([1, 1, 2])
    bad = rng.choice(len(p_enu), 30, replace=False)
    p_enu[bad, :2] += rng.normal(0, 60.0, (30, 2))          # 10 % gross outliers (multipath)
    assert a.fit(p_map, p_enu, np.full(len(p_map), 3.0))
    assert abs(math.degrees(a.yaw - yaw_true)) < 1.0
    err = np.linalg.norm(a.to_enu(p_map)[:, :2] - p_enu_true[:, :2], axis=1)
    assert np.median(err) < 1.0
    assert np.allclose(a.to_map(a.to_enu(p_map)), p_map)


def test_anchor_unobservable_without_spread_but_compass_fixes_yaw():
    p_map, _ = _camera_track(n=3)
    p_map = p_map * 0.01                                     # robot standing: positions within centimetres
    a = Anchor(AnchorConfig(dof=4))
    p_enu = np.zeros_like(p_map)
    assert not a.fit(p_map, p_enu, np.full(3, 4.0))          # yaw not observable from positions
    ok = a.fit(p_map, p_enu, np.full(3, 4.0), heading_map=np.array([0.1]), heading_enu=np.array([0.5]),
               heading_sigma=np.array([math.radians(2.0)]))
    assert ok and abs(a.yaw - 0.4) < 1e-3


def test_anchor_6dof():
    p_map, _ = _camera_track(seed=3)
    R = rotation_between(np.array([0.2, -1, 0.1]), np.array([0, 0, 1.0]))
    t = np.array([10.0, 20.0, 30.0])
    p_enu = p_map @ R.T + t + np.random.default_rng(2).normal(0, 0.5, p_map.shape)
    a = Anchor(AnchorConfig(dof=6))
    assert a.fit(p_map, p_enu, np.full(len(p_map), 0.5), np.full(len(p_map), 0.5))
    assert np.abs(a.R - R).max() < 0.01


def _gate_stream(gate, track, enu, t0=0.0, dt=1.0, pred=None):
    out = []
    for i, (tr, e) in enumerate(zip(track, enu)):
        fix = GnssFix(t=t0 + i * dt, lat=0.0, lon=0.0, mode=FIX_3D)
        p = None if pred is None else (pred[i], np.eye(2) * 4.0)
        out.append(gate.process(fix, np.array([e[0], e[1], 0.0]), tr, p))
    return out


def test_gate_rejects_jumps_and_frozen_fixes_and_holds_off():
    rng = np.random.default_rng(0)
    n = 120
    track = np.stack([np.arange(n) * 1.2, np.zeros(n)], 1)          # 1.2 m/s straight, 1 Hz fixes
    enu = track @ rot_z(0.3)[:2, :2].T + np.array([50.0, 20.0]) + rng.normal(0, 2.0, (n, 2))
    enu[30] += np.array([40.0, -30.0])                               # a multipath jump
    enu[60:110] = enu[59]                                            # receiver holds its last fix (indoors) while moving
    gate = GnssGate(GnssNoiseModel(), GnssGateConfig())
    d = _gate_stream(gate, track, enu)
    assert [x.reason for x in d[:10]] == ["holdoff"] * 10           # reacquisition hold-off at the start
    assert sum(x.used for x in d[73:110]) == 0                       # stale stretch once detected
    assert not d[30].used and d[30].reason == "relative"
    assert sum(x.used for x in d[85:110]) <= 2                       # a stale position fails the relative test
    assert all(x.reason in ("holdoff", "unverified") for x in d[:15])  # used once the window verifies the motion
    assert all(x.used for x in d[16:30])


def test_gate_gap_starts_new_epoch_and_absolute_gate():
    n = 40
    track = np.stack([np.arange(n) * 1.0, np.zeros(n)], 1)
    enu = track.copy() + np.random.default_rng(1).normal(0, 1.0, (n, 2))
    gate = GnssGate(GnssNoiseModel(), GnssGateConfig())
    for i in range(20):
        gate.process(GnssFix(t=float(i), lat=0, lon=0, mode=FIX_3D), np.r_[enu[i], 0], track[i], (track[i], np.eye(2)))
    dd = gate.process(GnssFix(t=100.0, lat=0, lon=0, mode=FIX_3D), np.r_[enu[20], 0], track[20], (track[20], np.eye(2)))
    assert dd.reason == "holdoff"                                    # after a 80 s gap
    # a biased stream (prediction off by 60 m, constant) is accepted only after the robot travelled further than the
    # offset, with every fix consistent
    gate = GnssGate(GnssNoiseModel(), GnssGateConfig())
    n = 200
    track = np.stack([np.arange(n) * 2.0, np.zeros(n)], 1)
    enu = track + np.random.default_rng(2).normal(0, 1.0, (n, 2))
    pred = track + np.array([60.0, 0.0])
    reasons = [gate.process(GnssFix(t=float(i), lat=0, lon=0, mode=FIX_3D), np.r_[enu[i], 0], track[i],
                            (pred[i], np.eye(2) * 4)).reason for i in range(n)]
    assert "drift" in reasons
    first = reasons.index("drift")
    assert first >= 10 and 2.0 * first >= 60.0 * 0.5


def test_gate_no_fix_and_noise_model():
    nm = GnssNoiseModel()
    assert nm.sigma(GnssFix(0, 1, 1, mode=FIX_NONE)) is None
    assert nm.sigma(GnssFix(0, 1, 1, sigma_h=1.5))[0] == pytest.approx(1.5)
    assert nm.sigma(GnssFix(0, 1, 1, hdop=2.0))[0] == pytest.approx(6.0)
    s4 = nm.sigma(GnssFix(0, 1, 1, mode=FIX_3D, num_sats=4))[0]
    s8 = nm.sigma(GnssFix(0, 1, 1, mode=FIX_3D, num_sats=8))[0]
    assert s4 > s8
    sc = nm.update_posterior(np.random.default_rng(0).chisquare(2, 500) * 4.0)   # residuals 2x the model
    assert 1.2 < sc < 2.1


def test_compass_tilt_and_offset_and_disturbance():
    B_ned = np.array([20.0, -2.0, 45.0])
    for bearing in (0.0, 45.0, 170.0, -100.0):
        for pitch in (0.0, 0.15):
            b = math.radians(bearing)
            Rz = np.array([[math.cos(b), -math.sin(b), 0], [math.sin(b), math.cos(b), 0], [0, 0, 1]])
            Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
            R_nb = Rz @ Ry                                        # body (FRD) -> NED
            m = R_nb.T @ B_ned
            a = R_nb.T @ np.array([0, 0, -9.81])
            decl = math.atan2(B_ned[1], B_ned[0])                 # the field points slightly west of north
            yaw = tilt_compensated_yaw(m, a)
            expect = math.pi / 2 - (b - decl)
            assert abs(((yaw - expect + math.pi) % (2 * math.pi)) - math.pi) < 1e-6
    c = Compass(CompassConfig(min_offset_samples=10))
    for i in range(40):
        c.add_offset_sample(0.3 + 0.01 * np.sin(i), 0.1)
    assert abs(c.offset - 0.2) < 0.02
    a = np.array([0, 0, -9.81])
    for i in range(100):
        assert not c.disturbed(B_ned * (1 + 0.005 * np.sin(i)), a, outdoor_ok=True) or i < 20
    assert c.disturbed(B_ned * 1.5, a)                            # a steel structure nearby


def test_manager_factors_correct_a_drifting_chain_in_the_pose_graph():
    """GeoManager + PoseGraph: a camera-convention odometry chain that drifts in heading is pulled back to the GNSS
    track by decimated GNSS factors with the soft gauge; indoor (no fix) and stale-receiver stretches are ignored."""
    import types
    import pypose as pp
    import torch
    from cross.core.config import GeoConfig
    from cross.core.pgo import PoseGraph
    from cross.core.types import Edge, EdgeType, Keyframe
    from cross.geo.manager import GeoManager

    rng = np.random.default_rng(3)
    n = 900                                       # keyframes 1 m apart, one per second
    yaw_rate = np.where((np.arange(n) // 150) % 2 == 0, 0.0, 0.02)
    # true camera poses (map = first camera frame: x right, y down, z forward); turning = rotation about -y
    def cam_pose(yaw, p):
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
        return T
    yaw = np.cumsum(yaw_rate)
    pos = np.zeros((n, 3))
    for i in range(1, n):
        pos[i] = pos[i - 1] + np.array([math.sin(yaw[i - 1]), 0, math.cos(yaw[i - 1])])
    T_true = [cam_pose(yaw[i], pos[i]) for i in range(n)]
    # odometry with a heading bias (0.05 deg per metre)
    T_odo = [np.eye(4)]
    deltas = []
    for i in range(1, n):
        d = np.linalg.inv(T_true[i - 1]) @ T_true[i]
        d = d @ cam_pose(0.0009, np.zeros(3))
        deltas.append(d)
        T_odo.append(T_odo[-1] @ d)
    # GNSS in ENU: map->ENU = level (up = -y) then yaw 30 deg, then translation
    a_true = Anchor(AnchorConfig(), vertical_map=np.array([0, -1.0, 0]))
    R_true = rot_z(math.radians(30)) @ a_true.R_level
    t_true = np.array([500.0, -200.0, 10.0])
    frame = LocalFrame(42.29, -83.71, 270.0)
    enu_true = pos @ R_true.T + t_true
    # correlated receiver error (Gauss-Markov, tau 30 s, 3 m) + indoor gap + stale stretch
    err = np.zeros((n, 2))
    for i in range(1, n):
        err[i] = math.exp(-1 / 30) * err[i - 1] + math.sqrt(1 - math.exp(-2 / 30)) * rng.normal(0, 3.0, 2)
    lla = frame.to_lla(np.c_[enu_true[:, :2] + err, enu_true[:, 2]])
    indoor = (np.arange(n) >= 400) & (np.arange(n) < 480)
    stale = (np.arange(n) >= 600) & (np.arange(n) < 660)
    cfg = GeoConfig(enabled=True)
    geo = GeoManager(cfg)
    geo.set_vertical(np.array([0, 1.0, 0]))
    nodes, odom_edges = {}, {}
    Keyframe._next_id = 0
    hm = types.SimpleNamespace(nodes=nodes, odom_edges=odom_edges, chart_aware=False, source_states=None,
                               component_charts=torch.zeros(1), component_generations=[0], system=None,
                               hypotheses={0: types.SimpleNamespace(visual_edges={}, visual_adjacency={})})

    def solve():
        pg = PoseGraph(hm, depth=10 ** 7, device="cpu")
        pg.construct_for_loop_closure(target_node_id=max(nodes), other_hypothesis_id=0)
        pg.unary_position_factors = geo.pgo_factors(nodes, set(nodes))
        pg.unary_robust_c = geo.robust_c
        pg.soft_priors = {0: geo.soft_gauge_cov(nodes[0].pose_mu[0].matrix().numpy()[:3, :3].astype(float))}
        pg.solve(optim_node_ids=set(nodes), fixed_node_ids=set())
        for i, p_ in pg.optimized_poses.items():
            nodes[i].pose_mu[0] = p_
        return pg

    belief = np.eye(4)
    n_opt = 0
    for i in range(n):
        if i > 0:
            geo.on_motion(deltas[i - 1])
            belief = belief @ deltas[i - 1]
        if not indoor[i]:
            la = lla[599] if stale[i] else lla[i]
            geo.observe(float(i), {"t": float(i), "lat": la[0], "lon": la[1], "alt": la[2]}, None, belief, True,
                        last_kf_id=i - 1 if i > 0 else None, nodes=nodes)
        q = pp.mat2SE3(torch.tensor(belief, dtype=torch.float32))
        kf = Keyframe(pose_mu=pp.SE3(q.tensor()[None]), pose_std=pp.se3(torch.full((1, 6), 0.1)),
                      pose_weights=torch.ones(1))
        nodes[kf.id] = kf
        if i > 0:
            odom_edges[(i - 1, i)] = Edge(pp.mat2SE3(torch.tensor(deltas[i - 1], dtype=torch.float32)),
                                          pp.se3(torch.tensor([0.02] * 3 + [0.002] * 3)), EdgeType.ODOMETRY)
        geo.on_keyframe(kf.id)
        if geo.should_optimize(nodes, geo.n_kf):
            solve()
            geo.after_optimize(nodes, geo.n_kf)
            belief = nodes[kf.id].pose_mu[0].matrix().numpy().astype(float)
            n_opt += 1
    assert geo.anchored and n_opt >= 2
    assert geo.gate.stats["stale"] >= 1
    pg = solve()
    geo.after_optimize(nodes, geo.n_kf)
    n_fac = len(pg.unary_position_factors)
    P_opt = np.array([nodes[i].pose_mu[0].matrix().numpy()[:3, 3] for i in range(n)])
    P_odo = np.array([T[:3, 3] for T in T_odo])
    truth = geo.frame.to_enu(*frame.to_lla(enu_true).T)            # ground truth in the manager's ENU frame
    e_opt = np.linalg.norm(geo.anchor.to_enu(P_opt)[:, :2] - truth[:, :2], axis=1)
    a_odo = Anchor(AnchorConfig(), vertical_map=np.array([0, -1.0, 0]))
    a_odo.fit(P_odo, truth + np.c_[err, np.zeros(n)], np.full(n, 3.0))   # best anchor of the odometry chain
    e_odo = np.linalg.norm(a_odo.to_enu(P_odo)[:, :2] - truth[:, :2], axis=1)
    print("rmse opt", np.sqrt((e_opt ** 2).mean()), "odo", np.sqrt((e_odo ** 2).mean()), "factors", n_fac, "opt", n_opt,
          "tau", geo.err.tau, "scale", geo.noise.scale, geo.gate.stats)
    assert 15 <= n_fac <= 120                     # decimated: about one per correlation time
    # (prior noise 5 m against a 3 m receiver error: the default without posterior rescaling, see GeoConfig)
    assert np.sqrt((e_opt ** 2).mean()) < 0.35 * np.sqrt((e_odo ** 2).mean())
    assert np.sqrt((e_opt ** 2).mean()) < 8.0
    assert 5.0 < geo.err.tau < 60.0                # the receiver error's correlation time (30 s) from the residuals


def test_geo_stream_nclt_layout(tmp_path):
    """gnss.txt with NCLT's two rows per fix ('msg' column, satellites not reported), imu.txt with the magnetometer,
    times.txt: one fix per frame on the loader's clock, unknown mode, compass sample; degradation."""
    from cross.dataloader.geo import GeoStream, parse_degrade
    t = 1000.0 + np.arange(0, 20, 0.2)
    rows = []
    for ti in t:
        rows.append([ti, 42.29, -83.71, np.nan, 2, 0, 0.1, 1.0])
        rows.append([ti, 42.29, -83.71, 270.0, 3, 0, 0.1, 1.0])
    np.savetxt(tmp_path / "gnss.txt", np.array(rows), header="t_s lat_deg lon_deg alt_m msg num_sats track_rad speed_mps")
    imu = np.c_[t, np.tile([0.2, 0.0, 0.45, 0.0, 0.0, -9.81, 0, 0, 0], (len(t), 1))]
    np.savetxt(tmp_path / "imu.txt", imu, header="t_s mag_x mag_y mag_z acc_x acc_y acc_z gyro_x gyro_y gyro_z")
    times = 1000.0 + np.arange(0, 19, 0.5)
    np.savetxt(tmp_path / "times.txt", times)
    gs = GeoStream.load(tmp_path, len(times), 2.0)
    assert len(gs.fix) == len(t) and np.isfinite(gs.fix[:, 3]).all()        # one row per fix, with altitude
    w = gs.window(3, 4, timestamp=2.0)
    g = w["gnss"]
    assert "mode" not in g and "num_sats" not in g                           # NCLT: quality not reported
    assert abs(g["t"] - 2.0) < 1e-6 and abs(g["alt"] - 270.0) < 1e-9
    assert "mag" in w["compass"]
    gd = GeoStream.load(tmp_path, len(times), 2.0, degrade=parse_degrade("bias=20,outage=2:6"))
    assert len(gd.fix) < len(gs.fix)
    assert gd.window(5, 6, timestamp=3.0).get("gnss") is None                 # inside the outage


def test_geo_state_columns_round_trip():
    """The geo state stored with a map (format v2 keeps per-keyframe data as numpy columns) loads back, and the
    version-1 dict form still loads."""
    import pickle
    from cross.core.config import GeoConfig
    from cross.geo.manager import GeoManager
    g = GeoManager(GeoConfig(enabled=True))
    g.frame = LocalFrame(42.29, -83.71, 270.0)
    g.anchor.R, g.anchor.t, g.anchor.yaw = np.eye(3), np.zeros(3), 0.0
    g.anchor.cov = np.eye(4) * 1e-6
    g.factors = {5: {"enu": np.array([1.0, 2.0, np.nan]), "sh": 5.0, "sv": 10.0, "delta": np.array([0.1, 0, 0]), "t": 3.0},
                 9: {"enu": np.array([4.0, 5.0, 6.0]), "sh": 4.0, "sv": 8.0, "delta": np.zeros(3), "t": 9.0}}
    s = g.state()
    assert isinstance(s["factors"]["enu"], np.ndarray) and s["factors"]["enu"].shape == (2, 3)
    s["keyframe_lla"] = GeoManager.lla_columns({3: [42.0, -83.0, 270.0], 1: [42.1, -83.1, 271.0]})
    s2 = pickle.loads(pickle.dumps(s))
    h = GeoManager(GeoConfig(enabled=True))
    h.load_state(s2)
    assert h.anchored and set(h.map_factors) == {5, 9}
    assert np.isnan(h.map_factors[5]["enu"][2]) and h.map_factors[9]["sv"] == 8.0
    assert GeoManager.lla_records(s2["keyframe_lla"]) == {1: [42.1, -83.1, 271.0], 3: [42.0, -83.0, 270.0]}
    v1 = dict(s2, factors={7: {"enu": [1, 2, 3], "sh": 5.0, "sv": 10.0, "delta": [0, 0, 0], "t": 1.0}})
    h.load_state(v1)
    assert set(h.map_factors) == {7}
    assert GeoManager.lla_records({3: [1, 2, 3]}) == {3: [1, 2, 3]}


def test_geo_anchor_kept_when_a_map_is_saved_again_without_gnss(tmp_path, monkeypatch):
    """geo.enabled by default: a geo-anchored map loaded and saved again in a session without GNSS input keeps its
    anchor, its GNSS factors and every keyframe's latitude / longitude (map format v2, System.save_map / load_map)."""
    import pypose as pp
    import torch
    from cross.core.config import HypothesisConfig, PoseEstType, SystemConfig
    from cross.core.hypothesis import HypothesisManager
    from cross.core.system import System
    from cross.core.types import Edge, EdgeType, Keyframe
    from cross.db import store
    from cross.geo.manager import GeoManager
    from test_map_store import _db, _rng_image

    def stub(monkeypatch):
        cfg = SystemConfig()
        assert cfg.geo.enabled
        s = System.__new__(System)
        s.config, s._lc_verifier, s.topo_map, s.visualize = cfg, None, None, False
        s.state_device = s.storage_device = s.device = "cpu"
        s.pose_est_type, s.pose_est = PoseEstType.PNP, None
        s._anchor_pending, s._contra_pending, s.loaded_node_ids = [], [], frozenset()
        s._last_retrieved_results, s._projection_n_nodes = None, -1
        s.db = _db(monkeypatch)
        s.hypothesis_manager = HypothesisManager(s, 3, HypothesisConfig())
        s._geo = GeoManager(cfg.geo)
        return s

    rng = np.random.default_rng(0)
    s1 = stub(monkeypatch)
    hm, db = s1.hypothesis_manager, s1.db
    s1.current_atlas = db.create_atlas()
    hm.dist = (pp.identity_SE3(3), pp.se3(torch.ones(3, 6) * .1), torch.tensor([1., 0., 0.]))
    hm.create_hypothesis_branch(0, 0)
    prev = None
    for i in range(5):
        mu = pp.SE3(torch.tensor([[i * 3.0, 0.0, i * 1.0, 0, 0, 0, 1.0]] * 3))
        kf = db.insert(i, _rng_image(rng), None, mu=mu, sigma=pp.se3(torch.rand(3, 6)), weights=torch.tensor([1., 0., 0.]),
                       atlas=s1.current_atlas, timestamp=float(i))
        hm.nodes[kf.id] = kf
        if prev is not None:
            hm.odom_edges[(prev, kf.id)] = Edge(pp.randn_SE3(), pp.se3(torch.rand(6)), EdgeType.ODOMETRY)
        prev = kf.id
    g = s1._geo
    g.frame = LocalFrame(42.29, -83.71, 270.0)
    g.anchor.R, g.anchor.t, g.anchor.yaw = rot_z(0.3) @ g.anchor.R_level, np.array([10.0, 20.0, 0.0]), 0.3
    g.anchor.cov = np.eye(4) * 1e-6
    k0 = min(hm.nodes)
    g.factors = {k0: {"enu": np.array([10.0, 20.0, 1.0]), "sh": 5.0, "sv": 20.0, "delta": np.zeros(3), "t": 0.0}}
    lla1 = g.keyframe_lla(hm.nodes)
    s1.save_map(str(tmp_path / "a" / "map.pkl"))

    s2 = stub(monkeypatch)                                     # a session without GNSS input
    s2.load_map(str(tmp_path / "a" / "map.pkl"))
    assert s2._geo.anchored and s2._geo.frame.lat0 == 42.29
    s2.hypothesis_manager.dist = (pp.identity_SE3(3), pp.se3(torch.ones(3, 6) * .1), torch.tensor([1., 0., 0.]))
    s2.save_map(str(tmp_path / "b" / "map.pkl"))
    d = store.read_map(str(tmp_path / "b" / "map.pkl"))
    geo = d["geo"]
    assert abs(geo["anchor"]["yaw"] - 0.3) < 1e-12 and geo["frame"]["lon0"] == -83.71
    lla2 = GeoManager.lla_records(geo["keyframe_lla"])
    assert set(lla2) == set(lla1)
    assert max(abs(np.array(lla2[k]) - np.array(lla1[k])).max() for k in lla1) < 1e-9
    assert set(GeoManager.factor_records(geo["factors"])) == {k0}


def _anchored_manager(**cfg_kw):
    from cross.core.config import GeoConfig
    from cross.geo.manager import GeoManager
    m = GeoManager(GeoConfig(**cfg_kw))
    m.frame = LocalFrame(42.29, -83.71, 270.0)
    # map = camera frame of a level camera (up = -y); ENU = levelled map, no yaw, offset
    rng = np.random.default_rng(0)
    p_map = np.c_[rng.uniform(-200, 200, 200), np.zeros(200), rng.uniform(-200, 200, 200)]
    p_enu = p_map @ m.anchor.R_level.T + np.array([10.0, 20.0, 0.0])
    assert m.anchor.fit(p_map, p_enu, np.full(200, 1.0))
    return m


def test_manager_focus_region_follows_the_trusted_fix():
    m = _anchored_manager(retrieval_focus=True, focus_margin_m=5.0, gate=False)
    assert m.focus_region() is None                                  # no fix yet
    e = m.anchor.to_enu(np.array([30.0, 0.0, 40.0]))
    lat, lon, alt = m.frame.to_lla(np.asarray(e, float))
    m.observe(0.0, {"t": 0.0, "lat": lat, "lon": lon, "alt": alt, "sigma_h": 2.0}, None, None, False)
    center, r, up = m.focus_region()
    assert np.linalg.norm((center - np.array([30.0, 0.0, 40.0]))[[0, 2]]) < 0.5
    assert 5.0 + 2.0 * 3.0 < r < 5.0 + 2.0 * 6.0                     # sqrt(chi2_2) x sigma + margin
    m.tick(10.0)
    assert m.focus_region() is None                                  # older than fix_max_age_s
