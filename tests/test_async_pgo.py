"""Background pose-graph optimisation (cross/core/async_pgo.py): the forked job, its equality with the synchronous
optimisation, and the transport of a result to the state of the moment it is applied (CPU only)."""
import os
import types

import gtsam
import numpy as np
import pypose as pp
import pytest
import torch

from cross.core import async_pgo as ap_mod
from cross.core.async_pgo import AsyncPgo, ForkedJob, JobError
from cross.core.config import SystemConfig
from cross.core.lc_verify import LoopClosureVerifier
from cross.core.system import System
from cross.core.types import Edge, EdgeType, Keyframe, VisualEdge
from test_pgo_scale import _lie, make_manager

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")


# ------------------------------------------------------------------------------------------------ the forked job
def test_forked_job_returns_the_result_and_runs_in_another_process():
    job = ForkedJob(lambda: (os.getpid(), np.arange(10))).start()
    pid, arr = job.result()
    assert pid != os.getpid() and np.array_equal(arr, np.arange(10))


def test_forked_job_sees_a_snapshot_not_later_changes():
    state = {"x": 1}
    import time
    job = ForkedJob(lambda: (time.sleep(0.2), state["x"])[1]).start()
    state["x"] = 2                                           # the parent goes on; the child keeps its copy
    assert job.result() == 1 and state["x"] == 2


def test_forked_job_large_result_and_ready():
    import time
    job = ForkedJob(lambda: (time.sleep(0.2), np.zeros(5_000_000))[1]).start()
    assert not job.ready()
    deadline = time.time() + 20
    while not job.ready() and time.time() < deadline:
        time.sleep(0.02)
    assert job.ready() and job.result().shape == (5_000_000,)


def test_forked_job_failures_do_not_hang_or_leak():
    def boom():
        raise ValueError("no")
    with pytest.raises(JobError, match="ValueError"):
        ForkedJob(boom).start().result()
    with pytest.raises(JobError, match="without a result"):
        ForkedJob(lambda: os._exit(3)).start().result()
    job = ForkedJob(lambda: __import__("time").sleep(30)).start()
    job.cancel()
    import time
    deadline = time.time() + 5
    while ap_mod._live_children and time.time() < deadline:      # children that finished are reaped lazily
        ap_mod.reap_finished()
        time.sleep(0.01)
    assert not ap_mod._live_children


# ----------------------------------------------------------------------------------- equality with the synchronous path
METHODS = ("_verified_lc_optimise", "_submit_async_pgo", "_poll_async_pgo", "_apply_async_pgo", "_resubmit_pending_pgo",
           "_drain_async_pgo", "_cancel_async_pgo", "_apply_async_poses", "_apply_async_geo", "_geo_maybe_optimize", "note_anchor")


def make_system(test_before_apply=False, outlier=True, async_lag=None, n=240, seed=0):
    """A hypothesis-0 graph with the real verifier and the System's methods of the verified loop closure bound to it."""
    hm = make_manager(n=n, loops=((10, 200), (40, 230)), seed=seed)
    hm.device = "cpu"
    sys_ = hm.system
    sys_.config.mapping.loop_closure.test_before_apply = test_before_apply
    sys_.config.mapping.loop_closure.async_pgo = async_lag is not None
    sys_.state_device = "cpu"
    sys_._processed_frame_num = 100
    sys_._last_pgo_step = -1
    sys_.use_odometry = True
    v = LoopClosureVerifier(sys_, sys_.config.mapping.loop_closure)
    v.stats["pgo_time"] = 0.0
    sys_._lc_verifier = v
    sys_.hypothesis_manager = hm
    sys_._apgo = AsyncPgo(sys_, lag_steps=async_lag) if async_lag is not None else None
    for name in METHODS:
        setattr(sys_, name, types.MethodType(System.__dict__[name], sys_))
    # the new loop edges of this step: a consistent revisit and, optionally, a wrong place
    new = [(60, 230)]
    if outlier:
        new.append((100, 215))
    rng = np.random.default_rng(1)
    for a, b in new:
        wrong = (a, b) == (100, 215)
        T = gtsam.Pose3.Expmap(np.array([0, 0, 0, 3.0, -2.0, 0.5]))
        # measurement: truth between the *current estimates* (so the consistent one fits and the wrong one does not)
        pa = pp.SE3(hm.nodes[a].pose_mu[0].tensor()).matrix().numpy().astype(np.float64)
        pb = pp.SE3(hm.nodes[b].pose_mu[0].tensor()).matrix().numpy().astype(np.float64)
        rel = gtsam.Pose3(np.linalg.inv(pa) @ pb)
        # the revisit disagrees with the (drifted) estimate by a metre so that the optimisation moves things
        m = rel.compose(T) if wrong else rel.compose(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.6, -0.4, 0.0)))
        f = VisualEdge(_lie(m), pp.se3(torch.full((6,), 0.05)), EdgeType.VISUAL, 0, 0)
        f.noise_scale, f.noise_scale_along, f.noise_scale_rot, f.informative = 1.0, 1.5, None, True
        hm.hypotheses[0].visual_edges.setdefault((a, b), []).append(f)
        hm.hypotheses[0].visual_adjacency.setdefault(a, set()).add(b)
        hm.hypotheses[0].visual_adjacency.setdefault(b, set()).add(a)
    sys_.new_keys = new
    return sys_


def poses(hm):
    return {k: kf.pose_mu.tensor()[0].detach().numpy().copy() for k, kf in hm.nodes.items()}


@pytest.mark.parametrize("test_before_apply", [False, True])
@pytest.mark.parametrize("outlier", [False, True])
def test_zero_lag_is_bit_identical_to_the_synchronous_optimisation(test_before_apply, outlier):
    s1 = make_system(test_before_apply, outlier)
    s2 = make_system(test_before_apply, outlier, async_lag=0)
    assert all(np.array_equal(a, b) for a, b in zip(poses(s1.hypothesis_manager).values(), poses(s2.hypothesis_manager).values()))
    keys, ref = s1.new_keys, min(a for a, b in s1.new_keys)
    ret1, ret2 = {}, {}
    assert s1._verified_lc_optimise(ret1, list(keys), ref)
    assert s2._submit_async_pgo(ret2, list(keys), ref, list(keys)) is True
    assert not s2._apgo.busy and s2._apgo.stats["applied"] == 1
    p1, p2 = poses(s1.hypothesis_manager), poses(s2.hypothesis_manager)
    assert any(not np.array_equal(p1[k], poses(make_system(test_before_apply, outlier).hypothesis_manager)[k]) for k in p1)  # it moved
    for k in p1:
        assert np.array_equal(p1[k], p2[k]), k
    q1 = [(a, b) for (a, b, f) in s1._lc_verifier.quarantine]
    q2 = [(a, b) for (a, b, f) in s2._lc_verifier.quarantine]
    assert q1 == q2 and (len(q1) == 1) == outlier
    for k in ("pgo", "posterior_flagged", "posterior_rejected"):
        assert s1._lc_verifier.stats.get(k, 0) == s2._lc_verifier.stats.get(k, 0), k
    assert s2.hypothesis_manager.pose_epoch >= 1            # (the synchronous run bumps it once per applied solution)


# ------------------------------------------------------------------------------------------- a result that arrives late
def add_tail(hm, n_new, start_dist=True):
    """Extend the chain by n_new keyframes (odometry from the last one), as the front end does while the job runs."""
    last = max(hm.nodes)
    for j in range(1, n_new + 1):
        i = last + j
        step = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.25, 0, 0))
        prev = pp.SE3(hm.nodes[i - 1].pose_mu.tensor()[0]).matrix().numpy().astype(np.float64)
        T = gtsam.Pose3(prev @ step.matrix())
        mu = torch.stack([_lie(T).tensor()] * 3)
        kf = Keyframe(pp.SE3(mu), pp.se3(torch.full((3, 6), 0.05)), torch.tensor([1.0, 0.0, 0.0]), None, None)
        kf.id = i
        hm.nodes[i] = kf
        e = Edge(_lie(step), pp.se3(torch.full((6,), 0.05)), EdgeType.ODOMETRY)
        e.n_frames = 3
        hm.odom_edges[(i - 1, i)] = e


def rel(hm, a, b):
    pa = pp.SE3(hm.nodes[a].pose_mu.tensor()[0]).matrix().numpy().astype(np.float64)
    pb = pp.SE3(hm.nodes[b].pose_mu.tensor()[0]).matrix().numpy().astype(np.float64)
    return np.linalg.inv(pa) @ pb


@pytest.mark.parametrize("lag", [2, -1])
def test_late_result_moves_the_keyframes_added_meanwhile_with_the_latest_one(lag):
    sync = make_system(False, False)
    late = make_system(False, False, async_lag=lag)
    late._apgo.sec_per_vertex = 1e-3                          # (free-running: a long job, so that it goes to the background)
    hm = late.hypothesis_manager
    keys, ref = late.new_keys, min(a for a, b in late.new_keys)
    sync._verified_lc_optimise({}, list(keys), ref)
    hm.dist = (pp.SE3(hm.nodes[239].pose_mu.tensor().clone()), pp.se3(torch.zeros(3, 6)), torch.tensor([1.0, 0.0, 0.0]))
    d_before = hm.dist[0][0].tensor().clone()
    late._submit_async_pgo({}, list(keys), ref, list(keys))
    assert late._apgo.busy
    add_tail(hm, 5)                                          # the front end goes on
    tail_before = {i: poses(hm)[i] for i in range(240, 245)}
    late._processed_frame_num += 2
    if lag < 0:
        import time
        while not late._apgo.job.ready():
            time.sleep(0.01)
    assert late._poll_async_pgo({}) is True
    ps, pl = poses(sync.hypothesis_manager), poses(hm)
    for k in range(240):                                     # keyframes of the fork: the optimised poses of the sync run
        assert np.array_equal(ps[k], pl[k]), k
    # the tail: odometry-consistent (rigidly moved)
    for i in range(240, 245):
        assert np.allclose(rel(hm, i - 1, i), np.array(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.25, 0, 0)).matrix()), atol=1e-4)
    moved = np.linalg.norm(ps[239][:3] - make_system(False, False).hypothesis_manager.nodes[239].pose_mu.tensor()[0].numpy()[:3])
    assert moved > 1e-3
    # the tracked pose follows the latest keyframe's correction
    d_after = hm.dist[0][0].tensor()
    assert np.allclose(d_after[:3].numpy() - d_before[:3].numpy(), ps[239][:3] - poses(make_system(False, False).hypothesis_manager)[239][:3], atol=1e-3)
    assert late._apgo.stats["stale_steps"] == 2 and late._apgo.stats["applied"] == 1


def test_triggers_during_a_job_are_rechecked_after_it():
    s = make_system(False, False, async_lag=3)
    keys, ref = s.new_keys, 60
    s._submit_async_pgo({}, list(keys), ref, list(keys))
    assert s._apgo.busy
    s._submit_async_pgo({}, [(70, 222)], 70, [(70, 222)])                  # another trigger while the job runs
    assert (70, 222) in s._apgo.pending and s._apgo.stats["submitted"] == 1
    s._processed_frame_num += 3
    s._poll_async_pgo({})
    assert not s._apgo.pending                                # re-checked: consistent edges are not optimised again
    assert s._apgo.stats["applied"] == 1


def test_a_restructured_graph_discards_the_result():
    s = make_system(False, True, async_lag=1)
    s._submit_async_pgo({}, list(s.new_keys), 60, list(s.new_keys))
    before = poses(s.hypothesis_manager)
    s.hypothesis_manager.graph_epoch += 1                     # a merge / promoted hypothesis meanwhile
    s._processed_frame_num += 1
    assert s._poll_async_pgo({}) is False
    assert all(np.array_equal(before[k], v) for k, v in poses(s.hypothesis_manager).items())
    assert s._apgo.stats["discarded"] == 1
    assert s._apgo.busy and s._apgo.stats["resubmitted"] == 1      # the wrong edge is still inconsistent: optimised again
    s._apgo.cancel(requeue=False)


def test_failed_jobs_are_requeued_and_switch_the_background_mode_off(monkeypatch):
    s = make_system(False, False, async_lag=0)
    monkeypatch.setattr(ap_mod, "verified_pgo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    for _ in range(3):
        s._apgo.pending.clear()
        s._submit_async_pgo({}, list(s.new_keys), 60, list(s.new_keys))
    assert s._apgo.disabled and not s._apgo.usable() and not s._apgo.busy


def test_drain_applies_the_job_and_forgets_the_triggers():
    s = make_system(False, False, async_lag=50)
    before = poses(s.hypothesis_manager)
    s._submit_async_pgo({}, list(s.new_keys), 60, list(s.new_keys))
    s._apgo.note_pending([(1, 2)])
    s._drain_async_pgo()
    assert not s._apgo.busy and not s._apgo.pending
    after = poses(s.hypothesis_manager)
    assert any(not np.array_equal(before[k], after[k]) for k in before)


def test_gnss_triggered_optimisation_in_the_background_keeps_later_factors_pending():
    s = make_system(False, False, async_lag=2)
    calls = []
    geo = types.SimpleNamespace(pending=[5, 6], n_kf=7, window_ref_kf=None, robust_c=1.0, last_opt_kf=0,
                                noise=types.SimpleNamespace(scale=1.0), err=types.SimpleNamespace(tau=1.0),
                                pgo_factors=lambda nodes, ids: [], should_optimize=lambda nodes, n: True)
    geo.after_optimize = lambda nodes, n_kf: (calls.append(n_kf), setattr(geo, "pending", []))
    s._geo = geo
    s._geo_maybe_optimize({})
    assert s._apgo.busy and s._apgo.job_meta["kind"] == "geo"
    s._geo_maybe_optimize({})                                  # one job at a time: nothing more is forked
    assert s._apgo.stats["submitted"] == 1
    geo.pending.append(9)                                      # a GNSS factor arrives while the job runs
    add_tail(s.hypothesis_manager, 3)
    s._processed_frame_num += 2
    assert s._poll_async_pgo({}) is True
    assert calls == [7] and geo.pending == [9]
    assert s._apgo.stats["applied"] == 1


def test_short_jobs_stay_in_the_front_end_in_free_running_mode():
    s = make_system(False, False, async_lag=-1)
    hm = s.hypothesis_manager
    assert hm.pgo_vertex_count(60) == 240                       # (whole session: the map is below pgo_window_min_nodes)
    before = poses(hm)
    s._submit_async_pgo({}, list(s.new_keys), 60, list(s.new_keys))
    assert not s._apgo.busy and s._apgo.stats["submitted"] == 0 and s._apgo.sec_per_vertex != 5e-5     # ran in the front end
    assert any(not np.array_equal(before[k], v) for k, v in poses(hm).items())
    s._apgo.sec_per_vertex = 1e-3                                                                      # a slow machine: 0.24 s
    s._submit_async_pgo({}, list(s.new_keys), 60, list(s.new_keys))
    assert s._apgo.busy
    s._apgo.cancel(requeue=False)


def test_vertex_count_of_a_windowed_optimisation():
    hm = make_manager(n=240, window_min_nodes=100)
    hm.system.config.mapping.loop_closure.pgo_window_margin = 10
    assert hm.pgo_vertex_count(None) == 240
    assert hm.pgo_vertex_count(200) == 240 - (200 - 10)         # the window from keyframe 190
    assert hm.pgo_vertex_count(5) == 240                        # a loop back to the start: everything



def test_a_hung_worker_is_killed_after_the_timeout():
    job = ForkedJob(lambda: __import__("time").sleep(60)).start()
    with pytest.raises(JobError, match="killed"):
        job.result(timeout=0.3)
    assert not ap_mod._live_children


def test_keyframes_moved_by_something_else_meanwhile_get_the_correction_added_to_their_present_pose():
    sync = make_system(False, False)
    late = make_system(False, False, async_lag=2)
    hm = late.hypothesis_manager
    keys, ref = late.new_keys, min(a for a, b in late.new_keys)
    sync._verified_lc_optimise({}, list(keys), ref)
    late._submit_async_pgo({}, list(keys), ref, list(keys))
    # something else (a local smoothing, another optimisation) shifts keyframes 100..110 by 10 cm in the meantime
    shift = pp.SE3(torch.tensor([0.1, 0.0, 0.0, 0, 0, 0, 1.0]))
    for i in range(100, 111):
        hm.nodes[i].pose_mu[0] = shift @ hm.nodes[i].pose_mu[0]
    hm.pose_epoch += 1
    base = poses(make_system(False, False).hypothesis_manager)
    before = {i: poses(hm)[i] for i in range(100, 111)}
    late._processed_frame_num += 2
    assert late._poll_async_pgo({}) is True
    ps, pl = poses(sync.hypothesis_manager), poses(hm)
    for k in range(240):
        if 100 <= k <= 110:                                  # (optimised correction) o (the shift)
            corr = np.array(pp.SE3(torch.from_numpy(ps[k])).matrix() @ np.linalg.inv(np.array(pp.SE3(torch.from_numpy(base[k])).matrix())))
            expect = corr @ np.array(pp.SE3(torch.from_numpy(before[k])).matrix())
            assert np.allclose(np.array(pp.SE3(torch.from_numpy(pl[k])).matrix()), expect, atol=2e-5), k
        else:
            assert np.array_equal(ps[k], pl[k]), k
