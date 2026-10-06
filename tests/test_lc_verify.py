"""Unit tests of the verified loop closure (cross/core/lc_verify.py) on synthetic graphs (CPU only)."""
import math
import types

import gtsam
import numpy as np
import pypose as pp
import pytest
import torch

from cross.core.config import LoopClosureConfig, NoiseModelConfig
from cross.core.lc_verify import (ChainPredictor, LoopClosureVerifier, NoiseModel, chi2_of, chi2_threshold, logmap,
                                  residual, to_gtsam, transport)
from cross.core.types import Edge, EdgeType, Keyframe


def _lie(T: gtsam.Pose3) -> pp.LieTensor:
    q = T.rotation().toQuaternion().coeffs()      # x y z w
    t = T.translation()
    return pp.SE3(torch.tensor([t[0], t[1], t[2], q[0], q[1], q[2], q[3]], dtype=torch.float32))


class FakeHM:
    def __init__(self):
        self.nodes = {}
        self.odom_edges = {}
        self.hypotheses = {0: types.SimpleNamespace(visual_edges={}, visual_adjacency={})}


class FakeSystem:
    def __init__(self, hm, session_start=0):
        self.hypothesis_manager = hm
        self._session_start_kf_id = session_start


def make_chain(n=120, step=0.25, turn_every=30, rng=None, k_t=0.06, k_r=0.07):
    """Square-ish trajectory: keyframe poses (gtsam), noisy odometry edges with the per-unit-motion noise model."""
    rng = rng or np.random.default_rng(0)
    gt = [gtsam.Pose3()]
    edges = {}
    for i in range(1, n):
        d = gtsam.Pose3(gtsam.Rot3.Ypr(math.pi / 2 if i % turn_every == 0 else 0.0, 0, 0), gtsam.Point3(step, 0, 0))
        gt.append(gt[-1].compose(d))
        L = float(np.linalg.norm(d.translation())); th = float(np.linalg.norm(logmap(d)[:3]))
        s = np.array([k_r * th + 1e-3] * 3 + [k_t * L + 2e-3] * 3)
        noisy = d.compose(gtsam.Pose3.Expmap(rng.normal(0, s)))
        e = Edge(_lie(noisy), pp.se3(torch.full((6,), 0.1)), EdgeType.ODOMETRY)
        e.n_frames = 1
        edges[(i - 1, i)] = e
    return gt, edges


def test_chain_covariance_matches_monte_carlo():
    hm = FakeHM()
    gt, hm.odom_edges = make_chain(n=100)
    noise = NoiseModel(NoiseModelConfig())
    cp = ChainPredictor(hm, noise)
    T, cov = cp.predict(0, 99)
    assert T is not None
    # Monte Carlo with the same per-edge sigmas
    rng = np.random.default_rng(1)
    S = []
    for _ in range(400):
        acc = gtsam.Pose3()
        for k in range(99):
            e = hm.odom_edges[(k, k + 1)]
            s = noise.odom_from_factor(e)
            acc = acc.compose(to_gtsam(e.mean)).compose(gtsam.Pose3.Expmap(rng.normal(0, s)))
        S.append(logmap(T.between(acc)))
    emp = np.cov(np.array(S).T)
    ratio = np.sqrt(np.diag(emp) / np.diag(cov))
    assert np.all(ratio > 0.75) and np.all(ratio < 1.35), ratio
    # reverse order gives the inverse transform and a consistent covariance
    Ti, covi = cp.predict(99, 0)
    assert np.allclose(Ti.matrix(), T.inverse().matrix(), atol=1e-9)
    assert np.allclose(covi, transport(cov, T.inverse()), atol=1e-9)


def test_prior_gate_accepts_true_and_rejects_aliased():
    hm = FakeHM()
    gt, hm.odom_edges = make_chain(n=120)
    for i, g in enumerate(gt):
        kf = Keyframe(pose_mu=_lie(g).unsqueeze(0), pose_std=pp.se3(torch.zeros(1, 6)), pose_weights=torch.ones(1))
        kf.id = i
        kf.step_created = i
        hm.nodes[i] = kf
    cfg = LoopClosureConfig()
    v = LoopClosureVerifier(FakeSystem(hm), cfg)
    last = 119
    # current pose = last keyframe (no motion since)
    refs = [hm.nodes[5], hm.nodes[40], hm.nodes[110]]
    true_meas = [_lie(gt[r.id].between(gt[last])) for r in refs]
    ok, chis = v.prior_gate(refs, true_meas, last, gtsam.Pose3(), 1)
    assert all(o is True for o in ok), (ok, chis)
    # aliased measurement: same rotation, 8 m off
    off = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(8.0, 0, 0))
    bad = [_lie(gt[r.id].between(gt[last]).compose(off)) for r in refs]
    ok_b, chis_b = v.prior_gate(refs, bad, last, gtsam.Pose3(), 1)
    assert ok_b[2] is False, (ok_b, chis_b)     # short chain: certainly inconsistent
    assert all(c > 0 for c in chis_b)


def test_inpass_gate_drops_wrong_reference():
    hm = FakeHM()
    gt, hm.odom_edges = make_chain(n=60)
    for i, g in enumerate(gt):
        kf = Keyframe(pose_mu=_lie(g).unsqueeze(0), pose_std=pp.se3(torch.zeros(1, 6)), pose_weights=torch.ones(1))
        kf.id = i
        hm.nodes[i] = kf
    v = LoopClosureVerifier(FakeSystem(hm), GUARD_ON)
    refs = [hm.nodes[10], hm.nodes[11], hm.nodes[12], hm.nodes[40]]
    # pass poses: current at gt[13]; correct references at their true poses, reference 40 placed as if it were keyframe 12
    cur = gt[13]
    c2w = [cur.matrix(), gt[10].matrix(), gt[11].matrix(), gt[12].matrix(), gt[12].compose(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.05, 0, 0))).matrix()]
    keep = v.inpass_gate(np.asarray(c2w), refs, np.array([True, True, True, True]), covis=[0.5, 0.5, 0.5, 0.9])
    assert keep.tolist() == [True, True, True, False], keep
    # all consistent -> nothing dropped
    c2w_ok = [cur.matrix()] + [gt[r.id].matrix() for r in refs]
    keep2 = v.inpass_gate(np.asarray(c2w_ok), refs, np.array([True, True, True, True]))
    assert keep2.all()


def test_chi2_threshold_and_residual():
    assert abs(chi2_threshold(0.999) - 22.4577) < 1e-2
    T = gtsam.Pose3(gtsam.Rot3.Yaw(0.1), gtsam.Point3(1, 2, 3))
    r = residual(T, T)
    assert np.allclose(r, 0)
    assert chi2_of(np.ones(6), np.eye(6)) == pytest.approx(6.0)


def test_chain_cache_follows_edge_replacement():
    """A removed temporary keyframe (bridged) plus a new keyframe leaves the edge count unchanged: the chain
    predictor must still see the new edge (the stale cache made every reference untestable and 'informative')."""
    import types, yaml, torch, pypose as pp
    from cross.core.hypothesis import HypothesisManager
    from cross.core.config import HypothesisConfig, LoopClosureConfig
    from cross.core.types import EdgeType, Keyframe
    from cross.core.lc_verify import LoopClosureVerifier, NoiseModel
    sys_ = types.SimpleNamespace(_lc_verifier=None, topo_map=None, _session_start_kf_id=0)
    hm = HypothesisManager(sys_, n_components=3, config=HypothesisConfig()); sys_.hypothesis_manager = hm
    sd = pp.se3(torch.full((6,), 0.1)); step = pp.SE3(torch.tensor([0.3, 0, 0, 0, 0, 0, 1.0]))

    def add(i):
        k = Keyframe(pp.SE3(torch.tensor([[0.3 * i, 0, 0, 0, 0, 0, 1.0]] * 3)), pp.identity_se3(3), torch.ones(3) / 3, None, None)
        k.id = i; hm.add_node(k)
        if i > 0:
            hm.add_edge(id1=i - 1, id2=i, rel_pose_mean=step, rel_pose_std=sd, type=EdgeType.ODOMETRY, meta={"n_frames": 3})
    for i in range(4):
        add(i)
    v = LoopClosureVerifier(sys_, LoopClosureConfig()); sys_._lc_verifier = v
    assert v.chain.predict(0, 3)[0] is not None
    # remove keyframe 2 (bridge 1->3) and add keyframe 4: the count of odometry edges is unchanged
    hm.nodes[2].temporary = True
    hm.odom_edges[(1, 3)] = hm.odom_edges[(1, 2)]; del hm.odom_edges[(1, 2)]; del hm.odom_edges[(2, 3)]
    hm.odom_edges_version += 1
    add(4)
    T, cov = v.chain.predict(0, 4)
    assert T is not None and abs(T.translation()[0] - 0.9) < 1e-6   # 0->1, 1->3 (one step: the bridge kept edge 1->2), 3->4
    # floors: a zero intercept never yields a zero sigma
    v.noise.cfg.visual_t_a = 0.0; v.noise.cfg.visual_t_b = 0.1
    assert v.noise.visual(0.0)[3] == NoiseModel.FLOOR_T and v.noise.visual(1.0)[3] == 0.1


def test_chain_prefix_matches_walk():
    """The prefix-product chain predictor equals edge-by-edge compounding (both orders, inflation, appends, bridging)."""
    import types, torch, pypose as pp, numpy as np, gtsam
    from cross.core.hypothesis import HypothesisManager
    from cross.core.config import HypothesisConfig, LoopClosureConfig
    from cross.core.types import EdgeType, Keyframe
    from cross.core.lc_verify import LoopClosureVerifier
    rng = np.random.default_rng(0)
    sys_ = types.SimpleNamespace(_lc_verifier=None, topo_map=None, _session_start_kf_id=0)
    hm = HypothesisManager(sys_, n_components=3, config=HypothesisConfig()); sys_.hypothesis_manager = hm
    sd = pp.se3(torch.full((6,), 0.1))

    def add(i):
        k = Keyframe(pp.identity_SE3(3), pp.identity_se3(3), torch.ones(3) / 3, None, None); k.id = i; hm.add_node(k)
        if i > 0:
            w = rng.normal(0, 0.3, 3); t = rng.normal(0, 0.5, 3)
            T = gtsam.Pose3(gtsam.Rot3.Expmap(w), t); q = T.rotation().toQuaternion(); tt = T.translation()
            hm.add_edge(id1=i - 1, id2=i, rel_pose_mean=pp.SE3(torch.tensor([tt[0], tt[1], tt[2], q.x(), q.y(), q.z(), q.w()])),
                        rel_pose_std=sd, type=EdgeType.ODOMETRY, meta={"n_frames": int(rng.integers(1, 5))})
    for i in range(40):
        add(i)
    v = LoopClosureVerifier(sys_, LoopClosureConfig()); c = v.chain
    A = gtsam.Pose3(gtsam.Rot3.Expmap([0.2, -0.1, 0.3]), [1.0, -2.0, 0.5])
    assert np.allclose(c._adjoint(A.matrix()[None])[0], A.AdjointMap())
    for a, b in [(0, 39), (39, 0), (5, 6), (6, 5), (12, 30), (30, 12), (3, 3)]:
        for infl in (1.0, 2.0):
            T1, C1 = c.predict(a, b, infl); T2, C2 = c.predict_walk(a, b, infl)
            assert np.allclose(T1.matrix(), T2.matrix(), atol=1e-9) and np.allclose(C1, C2, rtol=1e-6, atol=1e-10), (a, b)
    add(40)                                     # O(1) append path
    T1, C1 = c.predict(2, 40, 2.0); T2, C2 = c.predict_walk(2, 40, 2.0)
    assert np.allclose(T1.matrix(), T2.matrix(), atol=1e-9) and np.allclose(C1, C2, rtol=1e-6, atol=1e-10)
    # bridging (two edges replaced by one) + append: rebuild path
    e12, e23 = hm.odom_edges[(20, 21)], hm.odom_edges[(21, 22)]
    from cross.core.types import Edge
    br = Edge(e12.mean @ e23.mean, sd, EdgeType.ODOMETRY); br.n_frames = 5
    hm.odom_edges[(20, 22)] = br; del hm.odom_edges[(20, 21)]; del hm.odom_edges[(21, 22)]; hm.odom_edges_version += 1
    add(41)
    T1, C1 = c.predict(10, 41); T2, C2 = c.predict_walk(10, 41)
    assert np.allclose(T1.matrix(), T2.matrix(), atol=1e-9) and np.allclose(C1, C2, rtol=1e-6, atol=1e-10)
    assert c.predict(21, 41)[0] is None          # the removed node is off the chain


def test_online_metric_scale():
    """Measurements 1.2x longer than the odometry chain: the verifier's scale ratio converges to 1.2."""
    import types, torch, pypose as pp, numpy as np
    from cross.core.hypothesis import HypothesisManager
    from cross.core.config import HypothesisConfig, LoopClosureConfig
    from cross.core.types import EdgeType, Keyframe
    from cross.core.lc_verify import LoopClosureVerifier
    sys_ = types.SimpleNamespace(_lc_verifier=None, topo_map=None, _session_start_kf_id=0)
    hm = HypothesisManager(sys_, n_components=3, config=HypothesisConfig()); sys_.hypothesis_manager = hm
    sd = pp.se3(torch.full((6,), 0.1)); step = pp.SE3(torch.tensor([0.3, 0, 0, 0, 0, 0, 1.0]))
    for i in range(60):
        k = Keyframe(pp.SE3(torch.tensor([[0.3 * i, 0, 0, 0, 0, 0, 1.0]] * 3)), pp.identity_se3(3), torch.ones(3) / 3, None, None)
        k.id = i; hm.add_node(k)
        if i > 0:
            hm.add_edge(id1=i - 1, id2=i, rel_pose_mean=step, rel_pose_std=sd, type=EdgeType.ODOMETRY, meta={"n_frames": 3})
    v = LoopClosureVerifier(sys_, LoopClosureConfig())
    for last in range(5, 60):
        refs = [hm.nodes[last - 4]]                                   # 4 edges back: chain 1.2 m
        meas = [pp.SE3(torch.tensor([1.2 * 1.2, 0, 0, 0, 0, 0, 1.0]))]  # measured 1.2x too long
        v.prior_gate(refs, meas, last, pp.identity_SE3(), 1)
    assert abs(v.scale_ratio - 1.2) < 1e-6


def test_pgo_result_keeps_keyframe_std():
    """Applying an optimisation result must not shrink the keyframe std (repeated halving underflowed to zero)."""
    import types, torch, pypose as pp
    from cross.core.hypothesis import HypothesisManager
    from cross.core.config import HypothesisConfig
    from cross.core.types import Keyframe
    sys_ = types.SimpleNamespace(_lc_verifier=None, topo_map=None, _session_start_kf_id=0, last_added_kf_id=None)
    hm = HypothesisManager(sys_, n_components=3, config=HypothesisConfig()); sys_.hypothesis_manager = hm
    k = Keyframe(pp.identity_SE3(3), pp.se3(torch.full((3, 6), 0.05)), torch.ones(3) / 3, None, None); k.id = 0; hm.add_node(k)
    for _ in range(200):
        hm.apply_pgo_result({"optimized_poses": {0: pp.SE3(torch.tensor([1.0, 0, 0, 0, 0, 0, 1.0]))}, "other_hypothesis_id": 0})
    assert float(hm.nodes[0].pose_std[0].tensor().min()) >= 0.05 - 1e-6


def test_pose_conversion_round_trip_stays_unit():
    """pypose -> gtsam -> pypose round trips (one per optimisation) must not de-normalise the quaternion: gtsam
    builds a scaled, non-orthonormal rotation from a non-unit quaternion and the error compounded to |q| = 1.67
    after ~100 optimisations of an HSSD map."""
    import numpy as np, torch, pypose as pp
    from cross.core.pgo import pypose_to_gtsam_pose3, gtsam_to_pypose_pose3
    q = np.array([0.1, 0.2, -0.3, 0.9]); q = q / np.linalg.norm(q) * 1.02      # slightly off unit, as float32 storage can leave it
    p = pp.SE3(torch.tensor([1.0, -2.0, 0.5, *q], dtype=torch.float32))
    P0 = pypose_to_gtsam_pose3(p)
    for _ in range(1000):
        p = gtsam_to_pypose_pose3(pypose_to_gtsam_pose3(p), "cpu")
    v = p.tensor().numpy()
    assert abs(np.linalg.norm(v[3:7]) - 1.0) < 1e-5
    P1 = pypose_to_gtsam_pose3(p)
    assert np.linalg.norm(P0.between(P1).translation()) < 1e-4 and np.linalg.norm(P0.rotation().between(P1.rotation()).xyz()) < 1e-4


def test_anchored_chi2_corroborates_map_edges_through_the_chain():
    """Relocalization session: a map edge from an earlier observation, carried through the session's odometry chain,
    predicts a later map measurement; the true one passes, an aliased one (8 m off) does not."""
    hm = FakeHM()
    gt, edges = make_chain(n=120)
    start = 50
    for i, g in enumerate(gt):
        # session keyframes hold the (kidnapped) belief: arbitrary frame
        pose = g if i < start else gtsam.Pose3(gtsam.Rot3.Ypr(1.0, 0, 0), gtsam.Point3(30, -12, 0)).compose(g)
        kf = Keyframe(pose_mu=_lie(pose).unsqueeze(0), pose_std=pp.se3(torch.zeros(1, 6)), pose_weights=torch.ones(1))
        kf.id = i
        kf.step_created = i
        hm.nodes[i] = kf
    hm.odom_edges = {k: e for k, e in edges.items() if k[0] >= start}
    v = LoopClosureVerifier(FakeSystem(hm, session_start=start), LoopClosureConfig())
    assert v.anchor is None
    anchor = v.make_anchor(45, 55, _lie(gt[45].between(gt[55])))
    true_meas = _lie(gt[48].between(gt[60]))
    c2 = v.anchored_chi2(anchor, 48, true_meas, 60, gtsam.Pose3(), 1)
    assert c2 is not None and c2 <= v.thr, c2
    off = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(8.0, 0, 0))
    c2_bad = v.anchored_chi2(anchor, 48, _lie(gt[48].between(gt[60]).compose(off)), 60, gtsam.Pose3(), 1)
    assert c2_bad > v.thr, c2_bad
    assert v.anchor is None      # the candidate is only borrowed for the test


def test_contradict_anchor_drops_wrong_anchor_on_separated_consensus():
    """A session anchored to a wrong place: true map measurements of three separated places, all rejected, agree with
    each other through the odometry chain and drop the anchor; without enough separation they do not."""
    from collections import deque
    from cross.core.system import System
    hm = FakeHM()
    gt, edges = make_chain(n=120)
    start = 50
    for i, g in enumerate(gt):
        pose = g if i < start else gtsam.Pose3(gtsam.Rot3.Ypr(1.0, 0, 0), gtsam.Point3(30, -12, 0)).compose(g)
        kf = Keyframe(pose_mu=_lie(pose).unsqueeze(0), pose_std=pp.se3(torch.zeros(1, 6)), pose_weights=torch.ones(1))
        kf.id = i
        kf.step_created = i
        hm.nodes[i] = kf
    hm.odom_edges = {k: e for k, e in edges.items() if k[0] >= start}

    def run(sep):
        cfg = LoopClosureConfig(anchor_contradict_min=2, anchor_contradict_window=100, anchor_min_separation=sep,
                                anchor_contradict_min_offset=2.0)
        v = LoopClosureVerifier(FakeSystem(hm, session_start=start), cfg)
        off = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(8.0, 0, 0))
        v.update_anchor(40, 52, _lie(gt[40].between(gt[52]).compose(off)))      # wrong anchor
        fake = types.SimpleNamespace(_lc_verifier=v, hypothesis_manager=hm, last_added_kf_id=90, _session_start_kf_id=start,
                                     _processed_frame_num=90, _contra_pending=deque(), _anchor_pending=deque(),
                                     config=types.SimpleNamespace(mapping=types.SimpleNamespace(loop_closure=cfg)))
        fake._kf_position = types.MethodType(System._kf_position, fake)
        fake._separated = types.MethodType(System._separated, fake)
        for kf_s, map_kf in ((60, 44), (70, 45), (80, 46)):
            fake._contra_pending.append((kf_s, map_kf, v.make_anchor(map_kf, kf_s, _lie(gt[map_kf].between(gt[kf_s])))))
        h0_ok = [False]
        System._contradict_anchor(fake, [hm.nodes[48]], [_lie(gt[48].between(gt[90]))], h0_ok, pp.identity_SE3(), 1)
        return v.anchor, h0_ok

    anchor, ok = run(sep=0.0)
    assert anchor is None and ok == [None]
    anchor, ok = run(sep=1000.0)            # no two places that far apart: no consensus
    assert anchor is not None and ok == [False]


def test_corroborate_anchor_needs_a_separated_earlier_map_edge():
    from collections import deque
    from cross.core.system import System
    hm = FakeHM()
    gt, edges = make_chain(n=120)
    start = 50
    for i, g in enumerate(gt):
        pose = g if i < start else gtsam.Pose3(gtsam.Rot3.Ypr(1.0, 0, 0), gtsam.Point3(30, -12, 0)).compose(g)
        kf = Keyframe(pose_mu=_lie(pose).unsqueeze(0), pose_std=pp.se3(torch.zeros(1, 6)), pose_weights=torch.ones(1))
        kf.id = i
        kf.step_created = i
        hm.nodes[i] = kf
    hm.odom_edges = {k: e for k, e in edges.items() if k[0] >= start}

    def run(sep):
        cfg = LoopClosureConfig(anchor_corroborate_window=100, anchor_min_separation=sep)
        v = LoopClosureVerifier(FakeSystem(hm, session_start=start), cfg)
        fake = types.SimpleNamespace(_lc_verifier=v, hypothesis_manager=hm, last_added_kf_id=70, _session_start_kf_id=start,
                                     _processed_frame_num=70, _anchor_pending=deque(),
                                     config=types.SimpleNamespace(mapping=types.SimpleNamespace(loop_closure=cfg)))
        fake._kf_position = types.MethodType(System._kf_position, fake)
        fake._separated = types.MethodType(System._separated, fake)
        fake._anchor_pending.append((58, 45, v.make_anchor(45, 58, _lie(gt[45].between(gt[58])))))
        h0_ok = [None]
        System._corroborate_anchor(fake, [hm.nodes[48]], [_lie(gt[48].between(gt[70]))], h0_ok, pp.identity_SE3(), 1)
        return h0_ok

    assert run(0.0) == [True]
    assert run(1000.0) == [None]


def _chain_system(n=120, session_start=0):
    hm = FakeHM()
    gt, hm.odom_edges = make_chain(n=n)
    for i, g in enumerate(gt):
        kf = Keyframe(pose_mu=_lie(g).unsqueeze(0), pose_std=pp.se3(torch.zeros(1, 6)), pose_weights=torch.ones(1))
        kf.id = i
        kf.step_created = i
        hm.nodes[i] = kf
    return hm, gt


def test_split_covariance_is_along_the_bearing():
    noise = NoiseModel(NoiseModelConfig())
    T = gtsam.Pose3(gtsam.Rot3.Ypr(0.3, 0.0, 0.0), gtsam.Point3(4.0, 1.0, 0.0))
    C = noise.split_cov(0.01, 0.5, 0.05, T)
    u = np.asarray(T.rotation().matrix()).T @ (np.asarray(T.translation()) / np.linalg.norm(T.translation()))
    assert math.isclose(float(u @ C[3:, 3:] @ u), 0.25, rel_tol=1e-9)
    w = np.cross(u, [0, 0, 1.0]); w /= np.linalg.norm(w)
    assert math.isclose(float(w @ C[3:, 3:] @ w), 0.0025, rel_tol=1e-9)
    assert np.allclose(C[:3, :3], np.eye(3) * 1e-4) and np.allclose(C[:3, 3:], 0)
    # isotropic when both scales agree: equals the diagonal model
    s = noise.visual(float(np.linalg.norm(T.translation())))
    assert np.allclose(noise.visual_cov(T, 1.0, 1.0), np.diag(s ** 2))


def test_margin_keeps_neighbours_out_and_loops_in():
    """Information criterion with a margin: a measurement of a keyframe one edge back is not informative, one of a
    keyframe 100 edges back (a revisit after drift) is."""
    hm, gt = _chain_system()
    cfg = LoopClosureConfig()
    assert cfg.informative_margin == 2.0 and cfg.anisotropic
    v = LoopClosureVerifier(FakeSystem(hm), cfg)
    last = 119
    refs = [hm.nodes[118], hm.nodes[19]]
    meas = [_lie(gt[r.id].between(gt[last])) for r in refs]
    ok, _ = v.prior_gate(refs, meas, last, gtsam.Pose3(), 1)
    assert ok == [True, True]
    assert v.last_loop_flags == [False, True]
    # without the margin and with the base model, the neighbour of a 0.25 m odometry step would also count
    cfg1 = LoopClosureConfig(); cfg1.informative_margin = 1.0
    v1 = LoopClosureVerifier(FakeSystem(hm), cfg1)
    v1.prior_gate(refs, meas, last, gtsam.Pose3(), 1)
    assert v1.last_loop_flags[1] is True


def test_anisotropic_scales_follow_the_innovation():
    """Measurements whose length is 20 % off (direction exact) inflate the along-track scale, not the across one."""
    hm, gt = _chain_system(n=200)
    v = LoopClosureVerifier(FakeSystem(hm), GUARD_ON)
    rng = np.random.default_rng(1)
    for last in range(30, 199):
        refs = [hm.nodes[last - 3]]
        T = gt[last - 3].between(gt[last])
        T = gtsam.Pose3(T.rotation(), gtsam.Point3(*(np.asarray(T.translation()) * (1 + 0.2 * rng.standard_normal()))))
        v.prior_gate(refs, [_lie(T)], last, gtsam.Pose3(), 1)
    assert v.scales_along["sess"] > 1.5, v.scales_along
    assert v.scales["sess"] < 1.2, v.scales


def test_split_test_rejects_a_reversed_view():
    """A loop measurement with a plausible translation but the view reversed (180 deg) passes a translation-only test
    after a long chain and fails the split test; the true measurement passes both."""
    hm, gt = _chain_system()
    last = 119
    ref = hm.nodes[10]
    T_true = gt[10].between(gt[last])
    flip = gtsam.Pose3(gtsam.Rot3.Ypr(math.pi, 0.0, 0.0), gtsam.Point3(0.0, 0.0, 0.0))
    T_flip = gtsam.Pose3(T_true.rotation().compose(flip.rotation()), T_true.translation())
    cfg_t = LoopClosureConfig(); cfg_t.test_dof = "translation"
    ok_t, _ = LoopClosureVerifier(FakeSystem(hm), cfg_t).prior_gate([ref, ref], [_lie(T_true), _lie(T_flip)], last, gtsam.Pose3(), 1)
    assert ok_t == [True, True]
    cfg_s = LoopClosureConfig()
    assert cfg_s.test_dof == "split"
    ok_s, _ = LoopClosureVerifier(FakeSystem(hm), cfg_s).prior_gate([ref, ref], [_lie(T_true), _lie(T_flip)], last, gtsam.Pose3(), 1)
    assert ok_s == [True, False]


def test_loop_closure_skips_a_graph_that_cannot_be_built(monkeypatch):
    """A proposal whose pose graph cannot be built (construct_for_loop_closure raises ValueError, e.g. a chart-aware
    graph that does not reach the proposed reference chart) is skipped (success False), not raised to the session."""
    import threading
    import cross.core.hypothesis as hyp

    class _Graph:
        def __init__(self, *a, **kw):
            self.vertices, self.edges = [], []

        def construct_for_loop_closure(self, **kw):
            raise ValueError("Graph does not connect to its proposed reference chart")

    monkeypatch.setattr(hyp, "PoseGraph", _Graph)
    hm = hyp.HypothesisManager.__new__(hyp.HypothesisManager)
    hm.hypotheses, hm.nodes, hm.no_pgo_for_lc = {0: object(), 1: object()}, {0: object(), 1: object()}, False
    hm.graph_lock, hm.device = threading.RLock(), "cpu"
    hm.system = types.SimpleNamespace(config=types.SimpleNamespace(mapping=types.SimpleNamespace(loop_closure=LoopClosureConfig())),
                                      _session_start_kf_id=0)
    hm.pgo_noise_fn = lambda: None
    hm.pgo_skip_fn = lambda: None
    res = hm.handle_loop_closure(1)
    assert res["success"] is False and "reference chart" in res["message"]


# ----------------------------------------------------------------------------- odometry scale guard: fault response
GUARD_ON = LoopClosureConfig(odom_guard_inflate=True, odom_guard_map=True, odom_guard_attribute=True,
                             odom_guard_per_observation=True)

def test_guard_fault_inflates_only_the_translation():
    nm = NoiseModel(NoiseModelConfig())
    s = nm.odom(2.0, 0.1, 3)
    assert np.array_equal(nm.with_fault(s, 2.0, 0.0), s)
    f = nm.with_fault(s, 2.0, 0.5)
    assert np.allclose(f[:3], s[:3]) and np.allclose(f[3:], np.sqrt(s[3:] ** 2 + 1.0))


def _fire(v, g, first_kf, n=60):
    """Two windows of departing samples g over spans starting at first_kf, first_kf + 1, ..."""
    for i in range(n):
        v._guard_push(g, first_kf + i // 2)


def test_guard_fault_inflates_the_departing_stretch_and_fades():
    """A runaway from keyframe 60 on: the guard fires after two windows, rescales the odometry, and the odometry edges
    from the first departing window's earliest span on get the measured error as translation sigma (the chain's
    prediction across them loosens, the stretch before keeps its covariance); healthy windows let the error fade."""
    hm, _ = _chain_system(n=120)
    v = LoopClosureVerifier(FakeSystem(hm), GUARD_ON)
    _, cov_run = v.chain.predict(60, 100)
    _, cov_early = v.chain.predict(10, 40)
    for i in range(60):
        assert v._guard_push(1.0 + 0.01 * (-1) ** i, 10)
    assert v.odom_scale == 1.0 and v.odom_fault == 0.0
    _fire(v, 0.25, 60)
    assert abs(v.odom_scale - 0.25) < 1e-12 and abs(v.odom_fault - 0.75) < 1e-12
    assert all(getattr(e, "odom_fault", 0.0) == 0.75 for (a, _), e in hm.odom_edges.items() if a >= 60)
    assert not any(getattr(e, "odom_fault", 0.0) for (a, _), e in hm.odom_edges.items() if a < 60)
    _, cov_run2 = v.chain.predict(60, 100)
    _, cov_early2 = v.chain.predict(10, 40)
    assert np.trace(cov_run2[3:, 3:]) > 10 * np.trace(cov_run[3:, 3:])
    assert np.allclose(cov_early2, cov_early)
    for i in range(30):                                   # healthy again (relative to the corrected odometry)
        v._guard_push(1.02, 100)
    assert abs(v.odom_fault - 0.02) < 1e-9
    assert all(getattr(e, "odom_fault", 0.0) == 0.75 for (a, _), e in hm.odom_edges.items() if a >= 60)


def test_guard_without_inflation_only_rescales():
    hm, _ = _chain_system(n=120)
    v = LoopClosureVerifier(FakeSystem(hm), LoopClosureConfig(odom_guard_inflate=False, odom_guard_per_observation=True))
    _fire(v, 0.25, 60)
    assert abs(v.odom_scale - 0.25) < 1e-12 and v.odom_fault == 0.0
    assert not any(getattr(e, "odom_fault", 0.0) for e in hm.odom_edges.values())


def _reloc_session(odom_factor, session_start=50, n=120):
    """Map keyframes 0..session_start-1 at the true poses; session keyframes after them linked by odometry edges whose
    translations are odom_factor times the true ones."""
    hm = FakeHM()
    gt, _ = make_chain(n=n)
    for i, g in enumerate(gt):
        kf = Keyframe(pose_mu=_lie(g).unsqueeze(0), pose_std=pp.se3(torch.zeros(1, 6)), pose_weights=torch.ones(1))
        kf.id = i
        kf.step_created = i
        hm.nodes[i] = kf
    for i in range(session_start + 1, n):
        d = gt[i - 1].between(gt[i])
        d = gtsam.Pose3(d.rotation(), gtsam.Point3(*(np.asarray(d.translation()) * odom_factor)))
        e = Edge(_lie(d), pp.se3(torch.full((6,), 0.1)), EdgeType.ODOMETRY)
        e.n_frames = 1
        hm.odom_edges[(i - 1, i)] = e
    v = LoopClosureVerifier(FakeSystem(hm, session_start=session_start), GUARD_ON)
    v.guard_metric = True                     # stereo / depth: the map measurements are metric on their own
    return hm, gt, v


def _localize(v, gt, refs, last):
    """One observation of session keyframe `last` against map keyframes `refs` (true measurements)."""
    hm = v.system.hypothesis_manager
    return v.prior_gate([hm.nodes[r] for r in refs], [_lie(gt[r].between(gt[last])) for r in refs], last, gtsam.Pose3(), 1)


@pytest.mark.parametrize("factor", [3.0, 1.0])
def test_map_guard_samples_detect_a_runaway_in_a_relocalization_session(factor):
    """Relocalization session: true map measurements (three agreeing references per observation) against an odometry
    chain that claims 3x the motion fire the guard (odometry scale -> 1/3, the chain over the sampled spans is
    inflated); with a healthy chain nothing happens.  No session anchor is needed."""
    hm, gt, v = _reloc_session(factor)
    v.update_anchor(47, 52, _lie(gt[47].between(gt[52])))
    for last in range(53, 120):
        _localize(v, gt, (40, 45, 48), last)
    if factor == 1.0:
        assert v.odom_scale == 1.0 and v.odom_fault == 0.0
        return
    assert v.stats.get("odom_guard_updates", 0) >= 1
    assert abs(v.odom_scale - 1.0 / 3.0) < 0.05, v.odom_scale
    assert any(getattr(e, "odom_fault", 0.0) > 0.5 for e in hm.odom_edges.values())


def test_map_guard_samples_off():
    hm, gt, v = _reloc_session(3.0)
    v.guard_map = False
    v.update_anchor(47, 52, _lie(gt[47].between(gt[52])))
    for last in range(53, 120):
        _localize(v, gt, (40, 45, 48), last)
    assert v.odom_scale == 1.0


def test_chain_fault_error_grows_with_the_displacement():
    """A straight stretch of 40 edges with fault 0.5: the prediction's variance along the direction of travel gains
    (0.5 x displacement)^2 (shared scale error), not 40 independent (0.5 x 0.25)^2."""
    hm, _ = _chain_system(n=30)
    hm.odom_edges = {}
    for i in range(1, 41):
        e = Edge(_lie(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.25, 0, 0))), pp.se3(torch.full((6,), 0.1)), EdgeType.ODOMETRY)
        e.n_frames = 1
        hm.odom_edges[(i - 1, i)] = e
    cp = ChainPredictor(hm, NoiseModel(NoiseModelConfig()))
    _, cov0 = cp.predict(0, 40)
    for e in hm.odom_edges.values():
        e.odom_fault = 0.5
    cp = ChainPredictor(hm, NoiseModel(NoiseModelConfig()))
    _, cov1 = cp.predict(0, 40)
    indep = 40 * (0.5 * 0.25) ** 2                       # the edges' own (independent) inflation
    assert abs(cov1[3, 3] - cov0[3, 3] - indep - (0.5 * 10.0) ** 2) < 1e-6
    assert abs(cov1[4, 4] - cov0[4, 4] - indep) < 1e-6   # across the direction: independent share only
    _, cov_half0 = ChainPredictor(hm, NoiseModel(NoiseModelConfig())).predict(20, 40)
    for e in hm.odom_edges.values():
        e.odom_fault = 0.0
    _, cov_half_ok = ChainPredictor(hm, NoiseModel(NoiseModelConfig())).predict(20, 40)
    # half the stretch: (0.5 x 5 m)^2 shared plus 20 independent shares
    assert abs(cov_half0[3, 3] - cov_half_ok[3, 3] - 20 * (0.5 * 0.25) ** 2 - (0.5 * 5.0) ** 2) < 1e-6
    for e in hm.odom_edges.values():
        e.odom_fault = 0.5
    _, covr = ChainPredictor(hm, NoiseModel(NoiseModelConfig())).predict(40, 0)   # reverse order keeps the shared term
    assert covr[3, 3] > (0.5 * 10.0) ** 2


def test_map_guard_samples_need_a_metric_estimator():
    """Mono (guard_metric False): the map measurements' scale follows the odometry, so they are no guard samples."""
    hm, gt, v = _reloc_session(3.0)
    v.guard_metric = False
    v.update_anchor(47, 52, _lie(gt[47].between(gt[52])))
    for last in range(53, 120):
        _localize(v, gt, (40, 45, 48), last)
    assert v.odom_scale == 1.0


def test_guard_counts_observations_not_references():
    """A failed estimate of one frame gives agreeing samples for all its references: 10 frames with 8 departing samples
    each are 10 window samples (no firing); 60 such frames fire."""
    hm, _ = _chain_system(n=120)
    v = LoopClosureVerifier(FakeSystem(hm), GUARD_ON)
    v.guard_metric = True
    for _ in range(10):
        for _ in range(8):
            v._guard_sample(0.25, 60)
        v._guard_flush()
    assert v.odom_scale == 1.0
    for _ in range(60):
        for _ in range(8):
            v._guard_sample(0.25, 60)
        v._guard_flush()
    assert abs(v.odom_scale - 0.25) < 1e-9


def test_map_fix_needs_two_agreeing_references():
    """One reference per observation, or references that disagree (an aliased one 20 m off): no map fix, no sample."""
    hm, gt, v = _reloc_session(3.0)
    for last in range(53, 120):
        _localize(v, gt, (45,), last)
    assert v.odom_scale == 1.0 and not v._fix_hist
    hm2 = v.system.hypothesis_manager
    off = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(20.0, 0, 0))
    for last in range(53, 120):
        v.prior_gate([hm2.nodes[45], hm2.nodes[30]], [_lie(gt[45].between(gt[last])), _lie(gt[30].between(gt[last]).compose(off))],
                     last, gtsam.Pose3(), 1)
    assert v.odom_scale == 1.0 and not v._fix_hist


def _attrib_verifier():
    hm, _ = _chain_system(n=120)
    v = LoopClosureVerifier(FakeSystem(hm), GUARD_ON)
    v.guard_metric = True
    for _ in range(30):                                   # one healthy window: the odometry's pace, 0.05 m per frame
        v._guard_push(1.0, 10, 0.05)
    return hm, v


def test_guard_holds_when_the_odometry_kept_its_pace():
    """The measured / odometry ratio collapses (a failing visual estimator) while the odometry's speed is unchanged:
    the departure is the visual estimator's, the odometry is kept (KITTI 01, PnP at highway speed)."""
    hm, v = _attrib_verifier()
    for _ in range(60):
        v._guard_push(0.05, 60, 0.05)
    assert v.odom_scale == 1.0 and v.odom_fault == 0.0 and v.stats.get("odom_guard_held", 0) == 1
    assert not any(getattr(e, "odom_fault", 0.0) for e in hm.odom_edges.values())


def test_guard_fires_when_the_odometry_sped_up():
    """The ratio drops to 0.25 while the odometry's speed rose 4x (a diverging VIO): the guard fires; a second departure
    after the rescaling is judged on the cumulative departure and the raw speed."""
    hm, v = _attrib_verifier()
    for _ in range(60):
        v._guard_push(0.25, 60, 0.2)
    assert abs(v.odom_scale - 0.25) < 1e-12
    for _ in range(60):                                   # the VIO keeps accelerating: raw 0.6 m / frame, ratio 0.4 more
        v._guard_push(0.4, 100, 0.6)
    assert abs(v.odom_scale - 0.1) < 1e-12


def test_guard_needs_a_healthy_reference_pace():
    hm, _ = _chain_system(n=120)
    v = LoopClosureVerifier(FakeSystem(hm), GUARD_ON)
    for _ in range(60):
        v._guard_push(0.25, 60, 0.2)
    assert v.odom_scale == 1.0 and v.stats.get("odom_guard_held", 0) == 1


def test_guard_options_off_is_the_old_guard():
    """Default config: per-measurement windows, no attribution, no inflation, no epoch, no map fixes."""
    hm, gt, _ = _reloc_session(3.0)
    v = LoopClosureVerifier(FakeSystem(hm, session_start=50), LoopClosureConfig())
    v.guard_metric = True
    assert not (v.guard_inflate or v.guard_map or v.guard_attribute or v.guard_per_obs)
    for _ in range(60):
        v._guard_sample(0.25, 60, 0.05)                 # 60 measurements of one departure: the old guard fires
    assert abs(v.odom_scale - 0.25) < 1e-12 and v.odom_fault == 0.0 and v._guard_epoch_kf is None
    for last in range(53, 120):
        _localize(v, gt, (40, 45, 48), last)
    assert not v._fix_hist
