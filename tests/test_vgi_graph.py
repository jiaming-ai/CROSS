"""The local pose graph of VGGT + IMU (cross/imu/vgi_graph.py): the per-factor Jacobian assembly gives the same
estimates as the dense-autodiff implementation it replaced (commit 0207e1b), on synthetic windows with every factor
type and option, through several marginalizations; and it recovers the scale of a simulated drive."""

import importlib.util
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from cross.imu.preintegration import preintegrate      # noqa: E402
from cross.imu.simulate import simulate_imu            # noqa: E402
from cross.imu.vgi_graph import GraphConfig, VgiGraph   # noqa: E402
from test_imu_scale import robot_path                   # noqa: E402

REFERENCE_COMMIT = "0207e1b"


def _reference_class():
    """VgiGraph as of REFERENCE_COMMIT (dense Jacobians by autodiff over the whole window)."""
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{REFERENCE_COMMIT}:cross/imu/vgi_graph.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"reference implementation {REFERENCE_COMMIT} not available")
    path = Path(tempfile.mkdtemp()) / "vgi_graph_reference.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("cross.imu._vgi_graph_reference", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                       # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod.VgiGraph


def _window(samples, td, a0, a1):
    """The raw samples covering camera times a0..a1, on the camera clock at time offset td (camera = IMU clock + td)."""
    tc = samples[:, 0] + td
    lo = max(int(np.searchsorted(tc, a0, "right")) - 1, 0)
    hi = min(int(np.searchsorted(tc, a1, "left")) + 1, len(tc))
    out = samples[lo:hi].copy()
    out[:, 0] = tc[lo:hi]
    return out


def _drive(graphs, cfg_kw, seed=0, seconds=24.0, every=3, K=6, outliers=True, check=None, need_std=True,
           link_bias=0.01, offset=0.0, rot_noise=0.003, turn_period=23.0, stereo_std=None, truth=None, depth=None,
           learned=None):
    """Feed the same synthetic measurements to every graph; check(graphs, stage) after each solve / marginalize.
    offset: the IMU's stamps are this early (camera = IMU clock + offset); a graph estimating the offset gets the
    samples at its current estimate, preintegrated again when it moves (as the frontend does).  stereo_std: each pass's
    scale observed by a stereo pair with this log noise instead of learned depth (every 7th one off by log 2 when
    outliers).  truth: filled with the true log scale of each node's pass.  depth: per graph, whether it gets the
    learned depth (default all).  learned: filled with each node's learned-depth observation."""
    fps = 10.0
    poses = robot_path(seconds, fps, seed, 0.6, turn_period)
    rng = np.random.default_rng(seed + 11)
    t, w, a = simulate_imu(poses, fps, seed)
    samples = np.concatenate([t[:, None] - offset, w, a], 1)
    used = {id(g): 0.0 for g in graphs}             # the offset each graph's samples were taken at
    frames = list(range(0, len(poses), every))
    node, scale = {}, {}
    for n, f in enumerate(frames):
        s_true = float(np.exp(rng.normal(0, 0.3)))
        tt = f / fps
        da3 = np.log(s_true * 1.5) + rng.normal(0, 0.1)
        if n == 0:
            acc = a[(t >= tt) & (t <= tt + 0.2)].mean(0)
            ids = [g.start(poses[f, :3, :3], acc, da3, tt) for g in graphs]
        else:
            fp = frames[n - 1]
            ids = []
            for g in graphs:
                part = _window(samples, used[id(g)], fp / fps, tt)
                if g.cfg.time_offset:
                    pre, jac = g.preintegrate(part, fp / fps, tt, used[id(g)])
                elif g is not graphs[0]:
                    # the same factor data for every graph (the reference took its gyro-bias Jacobians by finite
                    # differences): the comparison is of the solvers
                    pre, jac = shared
                else:
                    pre, jac = shared = g.preintegrate(part, fp / fps, tt)
                ids.append(g.add_node(pre, jac, g.lam[g.ids[-1]], tt))
            views = [frames[n - 1]] + ([frames[n - K]] if n - K >= 0 else [])
            noise = [(Rotation.from_rotvec(rng.normal(0, rot_noise, 3)).as_matrix(), rng.normal(0, 0.02, 3))
                     for _ in views]
            bad = outliers and n % 5 == 0
            for g, i in zip(graphs, ids):
                for (Rn, tn), fa in zip(noise, views):
                    T = np.linalg.inv(poses[fa]) @ poses[f]
                    Rm = T[:3, :3] @ Rn
                    if bad:
                        Rm = Rm @ Rotation.from_rotvec([0.0, 0.1, 0.0]).as_matrix()      # an outlier pass (Huber)
                    g.add_relative(node[fa], i, i, Rm, (T[:3, 3] * (1 + tn[0]) + tn) / s_true)
                if len(views) == 2:
                    T = np.linalg.inv(poses[views[1]]) @ poses[views[0]]
                    g.add_relative(node[views[1]], node[views[0]], i, T[:3, :3], T[:3, 3] / s_true)
                Tr = np.linalg.inv(poses[fp]) @ poses[f]
                g.add_rotation(node[fp], i, Tr[:3, :3] @ noise[0][0], rot_noise)
                g.add_link(i, node[fp], np.log(s_true / scale[fp]) + link_bias)
        for k, (g, i) in enumerate(zip(graphs, ids)):
            if stereo_std is None:
                if depth is None or depth[k]:
                    g.add_depth(i, da3, 0.15)
            else:
                e = rng.normal(0, stereo_std) + (np.log(2.0) if outliers and n % 7 == 3 else 0.0)
                g.add_stereo(i, np.log(s_true) + e, stereo_std)
        if truth is not None:
            truth[ids[0]] = np.log(s_true)
        if learned is not None:
            learned[ids[0]] = da3
        node[f], scale[f] = ids[0], s_true
        if n > 0:
            for g in graphs:
                g.solve(need_std=need_std)
            if check:
                check(graphs, f"solve {n}")
            for g in graphs:
                g.marginalize()
                if g.cfg.time_offset and abs(g.td - used[id(g)]) > 0.004:
                    used[id(g)] = float(g.td)
                    g.repreintegrate(lambda a0, a1, g=g: _window(samples, used[id(g)], a0, a1), used[id(g)])
            if check:
                check(graphs, f"marginalize {n}")
    return poses, node, frames


def _state(g):
    out = {"g": g.g, "bg": g.bg, "ba": g.ba, "beta": np.array([g.beta]), "kappa": np.array([g.kappa])}
    for i in g.ids:
        out[f"R{i}"], out[f"p{i}"], out[f"v{i}"], out[f"lam{i}"] = g.R[i], g.p[i], g.v[i], np.array([g.lam[i]])
    return out


CASES = [
    dict(depth_bias=False, rot_scale=False, rot_rel=0.0),
    dict(depth_bias=True, rot_scale=False, rot_rel=0.05),
    dict(depth_bias=False, rot_scale=True, rot_rel=0.05),
    dict(depth_bias=True, rot_scale=True, rot_rel=0.05, gyro_dt_noise=0.1, accel_dt_noise=1.0),
]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_same_estimates_as_dense_reference(case):
    Ref = _reference_class()
    # the reference predates the robustness options (translation noise from the prediction, Huber gauge links)
    cfg = replace(GraphConfig(), window=6, trans_sigma_predicted=False, robust_links=False, **CASES[case])
    T_ci = np.eye(4)
    T_ci[:3, :3] = Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix()
    T_ci[:3, 3] = [0.05, -0.02, 0.1]
    new, ref = VgiGraph(cfg, T_ci, 1.1e-3, 1.2e-2), Ref(cfg, T_ci, 1.1e-3, 1.2e-2)
    worst = {"state": 0.0, "cost": 0.0}

    def check(graphs, stage):
        g1, g0 = graphs
        assert g1.ids == g0.ids, stage
        s1, s0 = _state(g1), _state(g0)
        for k in s0:
            err = np.max(np.abs(s1[k] - s0[k]) / (1e-6 + np.abs(s0[k])))
            worst["state"] = max(worst["state"], float(np.max(np.abs(s1[k] - s0[k]))))
            assert np.allclose(s1[k], s0[k], rtol=1e-6, atol=1e-7), f"{stage}: {k} differs ({err:.2e})"
        if stage.startswith("solve"):
            c1, c0 = g1.last_info["cost"], g0.last_info["cost"]
            worst["cost"] = max(worst["cost"], abs(c1 - c0) / max(abs(c0), 1e-12))
            assert abs(c1 - c0) <= 1e-6 * max(abs(c0), 1.0), f"{stage}: cost {c1} vs {c0}"
            assert g1.last_info["residuals"] == g0.last_info["residuals"], stage
            l1, l0 = g1.last_info["lam_std"], g0.last_info["lam_std"]
            assert (np.isnan(l1) and np.isnan(l0)) or abs(l1 - l0) <= 1e-6 * max(abs(l0), 1e-3), stage
            # the normal equations themselves (analytic Jacobians vs autodiff over the whole window)
            H1, gr1, _, _, _ = g1._system(g1._prepare())
            J0, r0 = g0._system(g0._rel_weights())
            H0, gr0 = J0.T @ J0, J0.T @ r0
            scale = np.abs(H0).max()
            worst["H"] = max(worst.get("H", 0.0), float(np.abs(H1 - H0).max() / scale))
            assert np.abs(H1 - H0).max() <= 1e-7 * scale, f"{stage}: H differs"
            assert np.abs(gr1 - gr0).max() <= 1e-7 * max(np.abs(gr0).max(), 1e-6) + 1e-6 * scale ** 0.5, stage
        if g0.prior is not None:
            (L1, i1, x1), (L0, i0, x0) = g1.prior, g0.prior
            assert i1 == i0, stage
            assert np.allclose(L1 @ L1.T, L0 @ L0.T, rtol=1e-6, atol=1e-6 * np.abs(L0 @ L0.T).max()), stage
            for k in x0:
                assert np.allclose(x1[k], x0[k], rtol=1e-6, atol=1e-7), f"{stage}: prior {k}"

    _drive([new, ref], CASES[case], check=check, need_std=case != 3)
    assert new.prior is not None                                      # several window slides happened


def test_huber_and_all_factor_types_exercised():
    cfg = replace(GraphConfig(), window=6, depth_bias=True, rot_scale=True, rot_rel=0.05)
    g = VgiGraph(cfg, np.eye(4), 1.1e-3, 1.2e-2)
    seen = {}

    def check(graphs, stage):
        if stage.startswith("solve"):
            blocks = graphs[0]._blocks(graphs[0]._prepare(), jac=False)
            for kind, w in graphs[0]._huber(blocks).items():
                seen[kind] = min(seen.get(kind, 1.0), float(w.min()))
            for _, _, _, kind in blocks:
                seen.setdefault(kind, 1.0)

    _drive([g], {}, check=check)
    assert {"imu", "rel", "rots", "links", "depth", "gnorm", "prior"} <= set(seen)
    assert seen["rel"] < 1.0                                          # the outlier passes were down-weighted


def test_scale_of_a_simulated_drive():
    """Learned depth 1.5x off with its bias estimated, unbiased gauge links: the newest node within 10 % of the path
    length of its true position (a gross-error check; the gauge is fixed at the first node)."""
    cfg = replace(GraphConfig(), window=20, depth_bias=True)
    g = VgiGraph(cfg, np.eye(4), 1.1e-3, 1.2e-2)
    poses, node, frames = _drive([g], {}, seconds=30.0, outliers=False, link_bias=0.0)
    f = frames[-1]
    path = float(np.linalg.norm(np.diff(poses[:f + 1, :3, 3], axis=0), axis=1).sum())
    assert np.linalg.norm(g.p[node[f]] - (poses[f, :3, 3] - poses[0, :3, 3])) < 0.1 * path


def test_twin_without_learned_depth_measures_its_bias():
    """IMU-arbitrated learned-depth calibration (vgio_depth_calib): a twin graph given every factor but the learned
    depth (1.5x off) has the true scale from the IMU and the passes, so learned depth minus the twin's scale, where the
    twin is certain, is the learned depth's bias (log 1.5)."""
    main = VgiGraph(replace(GraphConfig(), window=20, depth_bias=True, depth_bias_std=0.05), np.eye(4), 1.1e-3, 1.2e-2)
    twin = VgiGraph(replace(GraphConfig(), window=20, depth_bias=False), np.eye(4), 1.1e-3, 1.2e-2)
    truth, learned, samples = {}, {}, []

    def check(graphs, stage):
        if stage.startswith("solve"):
            j = twin.ids[-1]
            std = twin.last_info.get("lam_std", np.inf)
            if np.isfinite(std) and std < 0.1:
                samples.append((learned[j] - twin.lam[j], twin.lam[j] - truth[j]))
    _drive([main, twin], {}, seconds=40.0, outliers=False, link_bias=0.0, truth=truth, learned=learned,
           depth=[True, False], check=check)
    s = np.array(samples)
    assert len(s) >= 20                                          # the IMU observes the scale on this path
    assert abs(np.median(s[:, 0]) - np.log(1.5)) < 0.05          # the learned-depth bias
    assert np.median(np.abs(s[:, 1])) < 0.1                      # the twin's scale where it is certain


@pytest.mark.parametrize("offset", [0.0, 0.03, -0.02])
def test_time_offset_estimated(offset):
    """The IMU's stamps off by a constant, the graph started at zero offset: the offset, a variable of the graph, found
    within 5 ms.  It is seen mainly through the rotations: an offset t_d changes an interval's rotation by about
    (w(t1) - w(t0)) t_d, which a gyro bias mimics while the angular acceleration stays constant over the window; the
    turn rate here changes within the window (period 4 s; with 23 s the estimate is biased by ~8 ms)."""
    cfg = replace(GraphConfig(), window=20, depth_bias=True, time_offset=True, rot_std=0.003)
    g = VgiGraph(cfg, np.eye(4), 1.1e-3, 1.2e-2)
    _drive([g], {}, seconds=30.0, outliers=False, link_bias=0.0, offset=offset, turn_period=4.0)
    assert abs(g.td - offset) < 0.005, g.td


def test_gyro_bias_jacobians_match_finite_differences():
    """The analytic gyro-bias Jacobians of the preintegration against central differences, on a turning interval."""
    poses = robot_path(10.0, 10.0, 3, 0.6)
    t, w, a = simulate_imu(poses, 10.0, 3)
    samples = np.concatenate([t[:, None], w, a], 1)
    t0, t1 = 4.0, 4.3
    base = preintegrate(samples, t0, t1, 1e-3, 1e-2)
    eps = 1e-4
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        plus, minus = samples.copy(), samples.copy()
        plus[:, 1:4] -= d
        minus[:, 1:4] += d
        p, m = preintegrate(plus, t0, t1, 0, 0, full=False), preintegrate(minus, t0, t1, 0, 0, full=False)
        JR = Rotation.from_matrix(m.dR.T @ p.dR).as_rotvec() / (2 * eps)
        assert np.allclose(base.J_Rg[:, k], JR, rtol=1e-4, atol=1e-8)
        assert np.allclose(base.J_vg[:, k], (p.dv - m.dv) / (2 * eps), rtol=1e-4, atol=1e-7)
        assert np.allclose(base.J_pg[:, k], (p.dp - m.dp) / (2 * eps), rtol=1e-4, atol=1e-8)


def test_rotations_stay_on_so3():
    """A slightly non-orthonormal extrinsic (as printed in calibration files) and hundreds of nodes: every stored
    rotation stays a rotation (the reported trajectory is a running product of them)."""
    T_ci = np.eye(4)
    T_ci[:3, :3] = Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix() * (1 + 1e-6)
    g = VgiGraph(replace(GraphConfig(), window=6), T_ci, 1.1e-3, 1.2e-2)
    _drive([g], {}, seconds=40.0, outliers=False)
    for R in list(g.R.values()) + [g.R_cb, g.prior[2]["R"]]:
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and abs(np.linalg.det(R) - 1) < 1e-12


def test_imu_calibration_rotation_is_orthonormalized():
    from cross.dataloader.imu import ImuCalibration
    R = Rotation.from_rotvec([0.3, 0.1, -0.2]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = np.round(R, 4)                     # four printed digits
    T[:3, 3] = [0.1, 0.2, 0.3]
    c = ImuCalibration.from_dict({"T_cam_imu": T.tolist()})
    Rc = c.T_cam_imu[:3, :3]
    assert np.allclose(Rc @ Rc.T, np.eye(3), atol=1e-12) and np.allclose(Rc, R, atol=1e-4)
    assert np.allclose(c.T_cam_imu[:3, 3], [0.1, 0.2, 0.3])


def test_translation_gate():
    """The IMU test of a pass's translation: a pass reporting a fraction of the motion at highway speed is rejected
    while the gyro agrees with the passes' rotations; an agreeing one is accepted; an IMU velocity that ran away loses
    within seconds (its uncertainty grows with the time since the last accepted translation); an IMU whose gyro
    disagrees with the passes (dropout, clock jump) outvotes nothing; never more than the maximum gap."""
    import types
    from cross.mono.config import MonoConfig
    from cross.mono.vggt_imu_frontend import VggtImuFrontend
    fe = VggtImuFrontend(np.eye(3), MonoConfig(), device="cpu")
    fe.graph = types.SimpleNamespace(cfg=GraphConfig(), R={0: np.eye(3), 1: np.eye(3)}, p={0: np.zeros(3), 1: np.zeros(3)})
    fe.scale_filter = types.SimpleNamespace(initialized=True, lam_std=0.1)

    def gate(v_pred, v_meas, last_ok=99.7, t=100.0, rot_ok=True):
        fe.m = {"timestamp": t - 0.3}
        fe._last_trans_ok = last_ok
        fe._rot_checks = [True] * 4 + [rot_ok]
        fe.graph.p[1] = np.array([0.0, 0.0, v_pred * 0.3])
        return fe._translation_gate(0, 1, 0.0, np.array([0.0, 0.0, v_meas * 0.3]), t)["ok"]

    assert not gate(25.7, 3.0)                     # KITTI 01: VGGT-Omega reports a fraction of the motion
    assert not gate(25.7, 3.0, last_ok=92.0)       # ... and still loses to the IMU 8 s later at that speed
    assert gate(25.7, 24.0)                        # agreement
    assert not gate(1.5, 0.4)                      # a runaway IMU velocity wins right after a good update ...
    assert gate(1.5, 0.4, last_ok=96.0)            # ... but not for long
    assert gate(25.7, 3.0, rot_ok=False)           # a gyro inconsistent with the passes outvotes nothing
    assert gate(25.7, 3.0, last_ok=80.0)           # beyond the maximum gap the pass is accepted whatever it says
    fe.scale_filter.initialized = False
    assert fe._translation_gate(0, 1, 0.0, np.array([0.0, 0.0, 0.1]), 100.0) is None


def test_imu_dropout_inflates_the_factor():
    """A 1.2 s hole in the IMU stream inside an interval: the factor's covariance grows with the hole (the bridged
    rotation and velocity are guesses), an interval without a hole keeps the sensor noise."""
    poses = robot_path(10.0, 10.0, 5, 0.6)
    t, w, a = simulate_imu(poses, 10.0, 5)
    samples = np.concatenate([t[:, None], w, a], 1)
    g = VgiGraph(replace(GraphConfig(), window=6), np.eye(4), 1.1e-3, 1.2e-2)
    for t0 in np.arange(0.0, 3.0, 0.3):            # the IMU's motion level from normal intervals
        g.preintegrate(_window(samples, 0.0, t0, t0 + 0.3), t0, t0 + 0.3)
    full, _ = g.preintegrate(_window(samples, 0.0, 4.0, 5.5), 4.0, 5.5)
    holed = samples[(samples[:, 0] < 4.1) | (samples[:, 0] > 5.3)]
    gap, _ = g.preintegrate(_window(holed, 0.0, 4.0, 5.5), 4.0, 5.5)
    assert getattr(gap, "dropout", 0.0) > 1.0 and getattr(full, "dropout", 0.0) == 0.0
    assert np.trace(gap.cov[0:3, 0:3]) > 100 * np.trace(full.cov[0:3, 0:3])
    assert np.trace(gap.cov[3:6, 3:6]) > 100 * np.trace(full.cov[3:6, 3:6])


def test_zero_rate_update():
    """A gyro mean over an interval at rest measures the gyro bias directly: given early in the drive with an offset
    the visual rotations cannot see, the bias follows it and keeps it after those nodes are marginalized."""
    cfg = replace(GraphConfig(), window=10, gyro_bias_walk=1e-5)
    ref = VgiGraph(cfg, np.eye(4), 1.1e-3, 1.2e-2)
    _drive([ref], {}, seconds=30.0, outliers=False)              # the bias the visual rotations give
    offset = np.array([0.0, 0.004, 0.0])                         # rad/s, far beyond what they resolve
    g = VgiGraph(cfg, np.eye(4), 1.1e-3, 1.2e-2)

    def check(graphs, stage):
        if stage.startswith("solve") and int(stage.split()[1]) <= 15:
            graphs[0].add_zero_rate(graphs[0].ids[-1], ref.bg + offset, np.full(3, 1e-4))
    _drive([g], {}, seconds=30.0, outliers=False, check=check)
    assert np.linalg.norm(g.bg - ref.bg - offset) < 0.3 * np.linalg.norm(offset), (g.bg - ref.bg, offset)



def test_stereo_scale_of_a_simulated_drive():
    """Stereo + IMU: each pass's scale observed by the stereo pair in it (no learned depth, no bias state).  With outlier
    passes (rotations 0.1 rad off, stereo scales log 2 off) the window's pass scales stay within a few percent and the
    stereo outliers are down-weighted; without them the newest node is within 3 % of the path length of its true
    position (learned depth 1.5x off with its bias state: 6 %)."""
    cfg = replace(GraphConfig(), window=20, depth_bias=False)
    g = VgiGraph(cfg, np.eye(4), 1.1e-3, 1.2e-2)
    truth, seen = {}, {}

    def check(graphs, stage):
        if stage.startswith("solve"):
            blocks = graphs[0]._blocks(graphs[0]._prepare(), jac=False)
            for kind, w in graphs[0]._huber(blocks).items():
                seen[kind] = min(seen.get(kind, 1.0), float(w.min()))
    _drive([g], {}, seconds=30.0, link_bias=0.0, stereo_std=0.03, truth=truth, check=check)
    err = np.array([g.lam[i] - truth[i] for i in g.ids])
    assert np.median(np.abs(err)) < 0.03, err
    assert seen.get("stereo", 1.0) < 1.0                           # the log-2 stereo outliers were down-weighted
    assert not g.depth and len(g.stereo) == len(g.ids)            # marginalized nodes take their stereo factor along
    g = VgiGraph(cfg, np.eye(4), 1.1e-3, 1.2e-2)
    poses, node, frames = _drive([g], {}, seconds=30.0, link_bias=0.0, stereo_std=0.03, outliers=False)
    f = frames[-1]
    path = float(np.linalg.norm(np.diff(poses[:f + 1, :3, 3], axis=0), axis=1).sum())
    assert np.linalg.norm(g.p[node[f]] - (poses[f, :3, 3] - poses[0, :3, 3])) < 0.03 * path


def test_stereo_depth_scale_observation():
    """The stereo-depth scale of a pass (VgioPassService._stereo_depth_scale): a textured slanted plane rendered as a
    rectified pair (0.1 m baseline), the pass's depth at half the true depth: SGBM depth against it gives log 2."""
    import cv2
    import torch
    from cross.mono.config import MonoConfig
    from cross.mono.vggt_imu_frontend import VggtImuFrontend
    h, w, f, b = 240, 320, 300.0, 0.1
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    rng = np.random.default_rng(0)
    tex = cv2.resize(rng.integers(0, 255, (h // 4, w // 2)).astype(np.uint8), (2 * w, h), interpolation=cv2.INTER_NEAREST)
    tex = cv2.GaussianBlur(tex, (3, 3), 0)
    depth = 2.0 + 1.0 * np.linspace(0, 1, w)[None, :].repeat(h, 0)          # 2-3 m, slanted
    xs = np.arange(w)[None, :].repeat(h, 0).astype(np.float32)
    left = cv2.cvtColor(tex[:, w // 2:w // 2 + w], cv2.COLOR_GRAY2RGB)
    # right image: the left pixel x sees the plane point that the right camera images at x - f b / depth
    disp = (f * b / depth).astype(np.float32)
    mapx = xs + disp                                                        # right(x) = left(x + d)
    right = cv2.remap(left, mapx, np.arange(h)[:, None].repeat(w, 1).astype(np.float32), cv2.INTER_LINEAR)
    T_rl = np.eye(4)
    T_rl[0, 3] = b
    fe = VggtImuFrontend(K, MonoConfig(), device="cpu", depth_transform=lambda x: x, T_right_in_left=T_rl)
    obs, info, sgbm = fe.service._stereo_depth_scale(torch.from_numpy((depth / 2.0).astype(np.float32)), left, right)
    assert obs is not None and sgbm.shape == left.shape[:2], info
    assert abs(obs[0] - np.log(2.0)) < 0.03 and obs[1] < 0.1, (obs, info)


def test_metric_relative_factor_jacobians():
    """The metric relative-pose factor (stereo-tracked corners): analytic Jacobians against central differences of the
    whitened residual in the graph's tangent space (right rotation perturbations, positions additive)."""
    rng = np.random.default_rng(4)
    g = VgiGraph(replace(GraphConfig(), window=6), np.eye(4), 1.1e-3, 1.2e-2)
    for i in range(2):
        g._new_node(Rotation.from_rotvec(rng.normal(0, 0.4, 3)).as_matrix(), rng.normal(0, 2, 3), np.zeros(3), 0.0, i)
    g.gauge = (g.R[0].copy(), g.p[0].copy())
    A = rng.normal(0, 1, (6, 6))
    g.add_metric_relative(0, 1, Rotation.from_rotvec(rng.normal(0, 0.3, 3)).as_matrix(), rng.normal(0, 1, 3),
                          A @ A.T * 1e-3 + 1e-4 * np.eye(6))
    P = g._prepare()
    (cols, r0, J, kind), = [b for b in g._blocks(P) if b[3] == "met"]
    eps = 1e-6
    for k, c in enumerate(cols[0]):
        d = np.zeros(P["D"])
        d[c] = eps
        saved = g._save()
        g._apply(d)
        rp = [b for b in g._blocks(P, jac=False) if b[3] == "met"][0][1]
        g._restore(saved)
        g._apply(-d)
        rm = [b for b in g._blocks(P, jac=False) if b[3] == "met"][0][1]
        g._restore(saved)
        assert np.allclose((rp - rm) / (2 * eps), J[:, :, k], atol=1e-5, rtol=1e-4), (k, (rp - rm) / (2 * eps), J[:, :, k])


def test_stereo_pnp_motion():
    """The metric motion of tracked corners with the stereo depth of the earlier frame (VggtImuFrontend._stereo_pnp, the
    depth sampled at the corners by VgioPassService.corner_depth): a known motion is recovered, its covariance shrinks
    with the pixel noise, and a rotation that disagrees with the gyro is not used."""
    from cross.mono.config import MonoConfig
    from cross.mono.vggt_imu_frontend import VggtImuFrontend
    rng = np.random.default_rng(1)
    h, w, f = 480, 640, 400.0
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    T_rl = np.eye(4)
    T_rl[0, 3] = 0.12
    R_mb = Rotation.from_rotvec([0.01, 0.05, -0.02]).as_matrix()
    t_mb = np.array([0.1, -0.02, 0.6])
    uv = np.stack([rng.uniform(20, w - 20, 200), rng.uniform(20, h - 20, 200)], 1).round()
    z = rng.uniform(2.0, 8.0, 200)
    X = np.concatenate([(uv - K[:2, 2]) / f, np.ones((200, 1))], 1) * z[:, None]
    Xb = (X - t_mb) @ R_mb                                        # x_b = R_mb^T (x_m - t_mb)
    ub = Xb[:, :2] / Xb[:, 2:] * f + K[:2, 2] + rng.normal(0, 0.3, (200, 2))
    depth = np.zeros((h, w))
    depth[uv[:, 1].astype(int), uv[:, 0].astype(int)] = z
    fe = VggtImuFrontend(K, MonoConfig(), device="cpu", T_right_in_left=T_rl)
    fe.service._last_sgbm = (7, depth)
    corner_z = fe.service.corner_depth(7, uv.astype(np.float32))
    keep = np.arange(200)[::2]                                    # the corners still tracked (ids among the detected)
    tracks = (uv.astype(np.float32)[keep, None], ub.astype(np.float32)[keep, None], keep, [], 7)
    out, info = fe._stereo_pnp(R_mb, 0.1, tracks, corner_z)
    assert out is not None, info
    R, t, cov = out
    assert np.degrees(np.linalg.norm(Rotation.from_matrix(R.T @ R_mb).as_rotvec())) < 0.2
    assert np.linalg.norm(t - t_mb) < 0.03 and np.all(np.linalg.eigvalsh(cov) > 0)
    assert np.sqrt(np.trace(cov[3:6, 3:6])) < 0.05, cov
    out, info = fe._stereo_pnp(Rotation.from_rotvec([0.0, 0.2, 0.0]).as_matrix() @ R_mb, 0.1, tracks, corner_z)
    assert out is None and info["reason"] == "gyro"


def test_stereo_gate():
    """The stereo translation tests with the IMU as the arbiter: a pass and a corner motion that agree with the IMU are
    accepted; corners locked on vehicles alongside (~1 m/s at 27 m/s) and a pass reporting half the speed are rejected
    while the IMU is trustworthy, even when both visual cues agree with each other; with an untrustworthy IMU the pass
    is kept and a corner motion that disagrees with it is dropped."""
    import types
    from cross.mono.config import MonoConfig
    from cross.mono.vggt_imu_frontend import VggtImuFrontend
    fe = VggtImuFrontend(np.eye(3), MonoConfig(), device="cpu", T_right_in_left=np.eye(4))
    fe.graph = types.SimpleNamespace(cfg=GraphConfig(), R={0: np.eye(3), 1: np.eye(3)}, p={0: np.zeros(3), 1: np.zeros(3)},
                                     v_std=0.1)
    fe.scale_filter = types.SimpleNamespace(initialized=True, lam_std=0.03)

    def gate(v_pred, v_pass, v_pnp=None, rot_ok=True, last_ok=99.7, t=100.0):
        fe.m = {"timestamp": t - 0.3}
        fe._last_trans_ok = last_ok
        fe._rot_checks = [True] * 4 + [rot_ok]
        fe.graph.p[1] = np.array([0.0, 0.0, v_pred * 0.3])
        pnp = None if v_pnp is None else (np.eye(3), np.array([0.0, 0.0, v_pnp * 0.3]), np.eye(6) * 0.02 ** 2)
        ok, pnp_ok, _ = fe._stereo_gate(0, 1, 0.0, np.array([0.0, 0.0, v_pass * 0.3]), True, pnp, t)
        return ok, pnp_ok

    assert gate(27.0, 26.5, 27.2) == (True, True)
    assert gate(27.0, 26.5, 1.0) == (True, False)          # corners on vehicles alongside
    assert gate(27.0, 13.0, 27.0) == (False, True)         # a pass that under-reports the motion
    assert gate(27.0, 6.0, 6.1) == (False, False)          # both visual cues on moving vehicles: the IMU wins
    assert gate(27.0, 6.0, 6.1, rot_ok=False) == (True, True)    # an IMU inconsistent with the passes outvotes nothing
    assert gate(27.0, 6.0, 20.0, rot_ok=False) == (True, False)
    assert gate(27.0, 6.0, last_ok=80.0) == (True, True)   # beyond the maximum gap the pass is accepted
    fe.graph.v_std = np.inf
    assert gate(27.0, 6.0, 6.1) == (True, True)            # a velocity the graph does not know yet vetoes nothing
    fe.graph.v_std = 0.1
    fe.scale_filter.initialized = False
    assert gate(27.0, 1.0, 1.0) == (True, True)            # no test before the scale is known


def test_stereo_gate_failure_model():
    """The stereo tests' failure model: while a cue keeps failing below the IMU's prediction (a collapse), one a little
    below (a partial collapse) is rejected although within 4 sigma, one as far above is accepted; a cue failing on both
    sides (noise) is held to the same tolerance on both; with every recent test passed the test is (nearly) the 4
    sigma one."""
    import types
    from cross.mono.config import MonoConfig
    from cross.mono.vggt_imu_frontend import VggtImuFrontend

    def frontend():
        fe = VggtImuFrontend(np.eye(3), MonoConfig(), device="cpu", T_right_in_left=np.eye(4))
        fe.graph = types.SimpleNamespace(cfg=GraphConfig(), R={0: np.eye(3), 1: np.eye(3)},
                                         p={0: np.zeros(3), 1: np.zeros(3)}, v_std=0.6)
        fe.scale_filter = types.SimpleNamespace(initialized=True, lam_std=0.01)
        return fe

    def gate(fe, v_pred, v_pass, v_pnp=None, t=100.0):
        fe.m = {"timestamp": t - 0.3}
        fe._last_trans_ok = t - 0.3
        fe._rot_checks = [True] * 5
        fe.graph.p[1] = np.array([0.0, 0.0, v_pred * 0.3])
        pnp = None if v_pnp is None else (np.eye(3), np.array([0.0, 0.0, v_pnp * 0.3]), np.eye(6) * 0.05 ** 2)
        ok, pnp_ok, _ = fe._stereo_gate(0, 1, 0.0, np.array([0.0, 0.0, v_pass * 0.3]), True, pnp, t)
        return ok, pnp_ok

    fe = frontend()
    for _ in range(20):
        assert gate(fe, 25.0, 25.3, 25.1) == (True, True)
    assert gate(fe, 25.0, 22.4)[0]                         # 2.6 below (2.9 sigma) with every recent test passed
    assert gate(fe, 25.0, 27.6)[0]
    for _ in range(12):                                    # a collapse: passes and corners far below the prediction
        assert gate(fe, 25.0, 9.0, 6.0) == (False, False)
    assert not gate(fe, 25.0, 22.4)[0]                     # now a partial collapse is the likelier explanation
    assert gate(fe, 25.0, 27.6)[0]                         # as far above: no collapse reports more motion
    fe = frontend()
    for k in range(12):                                    # noisy passes, failing on both sides
        assert not gate(fe, 25.0, 35.0 if k % 2 else 15.0)[0]
    assert not gate(fe, 25.0, 22.4)[0] and not gate(fe, 25.0, 27.6)[0]
    assert gate(fe, 25.0, 24.2)[0] and gate(fe, 25.0, 25.8)[0]
