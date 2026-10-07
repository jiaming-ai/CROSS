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
