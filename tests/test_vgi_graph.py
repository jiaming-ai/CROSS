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
           link_bias=0.01, offset=0.0, rot_noise=0.003, turn_period=23.0):
    """Feed the same synthetic measurements to every graph; check(graphs, stage) after each solve / marginalize.
    offset: the IMU's stamps are this early (camera = IMU clock + offset); a graph estimating the offset gets the
    samples at its current estimate, preintegrated again when it moves (as the frontend does)."""
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
        for g, i in zip(graphs, ids):
            g.add_depth(i, da3, 0.15)
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
    cfg = replace(GraphConfig(), window=6, **CASES[case])
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
