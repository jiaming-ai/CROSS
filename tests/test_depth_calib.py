"""IMU-arbitrated learned-depth calibration of the VGGT-inertial odometry (imu.vgio_depth_calib): the offset from the
twin graph's samples, and the offset kept with the map for later sessions."""
import collections
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _frontend(store=None, **imu):
    from cross.mono.config import MonoConfig
    from cross.mono.vggt_imu_frontend import VggtImuFrontend
    mc = MonoConfig()
    mc.imu.vgio_depth_calib = True
    for k, v in imu.items():
        setattr(mc.imu, k, v)
    fe = VggtImuFrontend(np.eye(3), mc, device="cpu")
    fe.calib_store = store
    return fe


def _calib():
    return types.SimpleNamespace(T_cam_imu=np.eye(4), gyro_noise_density=1e-3, accel_noise_density=1e-2,
                                 gyro_random_walk=1e-5)


def test_offset_from_certain_samples_and_kept_for_the_next_session():
    """Samples are taken only where the twin's scale is certain; the median of enough of them over a long enough
    time is the offset, written to the store (the back end, saved with the map); a new session of the same learned
    depth starts from it, one of another learned depth does not."""
    store = types.SimpleNamespace(odometry_calib={})
    fe = _frontend(store, vgio_depth_calib_apply="session")
    fe._start(_calib())
    twin = types.SimpleNamespace(lam={}, std=0.05)
    twin.solve = lambda need_std=True: {"lam_std": twin.std}
    twin.marginalize = lambda: None
    fe.twin = twin
    rng = np.random.default_rng(0)
    for k in range(60):
        twin.lam[k] = float(rng.normal(0.0, 0.3))
        twin.std = 0.5 if k % 2 else 0.05                      # every other node uncertain: no sample
        out = fe._depth_calibrate(k, (twin.lam[k] + 0.4 + rng.normal(0, 0.05), 0.2), timestamp=float(k))
        assert out["calib_used"] == (k % 2 == 0)
        if k < 2 * fe.config.imu.vgio_depth_calib_min_samples - 2:
            assert fe.depth_calib == 0.0                      # not enough samples yet
    assert abs(fe.depth_calib - 0.4) < 0.05
    stored = store.odometry_calib["learned_depth"]
    assert abs(stored["offset"] - 0.4) < 0.05 and stored["samples"] == 30 and stored["source"].startswith("da3:")

    fe2 = _frontend(store)                                      # "stored": the map's offset, fixed in the session
    fe2._start(_calib())
    assert fe2.depth_calib == stored["offset"]
    fe2.twin = twin
    twin.std = 0.05
    for k in range(60, 120):
        twin.lam[k] = 0.0
        fe2._depth_calibrate(k, (0.9, 0.2), timestamp=float(k))   # this session sees another offset
    assert fe2.depth_calib == stored["offset"] and abs(store.odometry_calib["learned_depth"]["offset"] - 0.9) < 1e-6
    fe3 = _frontend(store, depth_prior_source="head")
    fe3._start(_calib())
    assert fe3.depth_calib == 0.0


def test_offset_bounded_by_the_scale_band():
    fe = _frontend(vgio_depth_calib_apply="session")
    fe._start(_calib())
    fe.twin = types.SimpleNamespace(lam={0: 0.0}, solve=lambda need_std=True: {"lam_std": 0.01},
                                    marginalize=lambda: None)
    fe._calib_samples = collections.deque([(float(t), 3.0) for t in range(40)], maxlen=300)
    fe._depth_calibrate(0, (3.0, 0.2), timestamp=40.0)
    assert abs(fe.depth_calib - np.log(fe.config.imu.scale_band)) < 1e-9


def test_odometry_calibration_saved_and_loaded_with_the_map(tmp_path, monkeypatch):
    """System.save_map / load_map (format v2) keep the odometry source's calibration, also when the map is saved
    again by a later session."""
    import pypose as pp
    import torch
    from cross.core.config import HypothesisConfig, PoseEstType, SystemConfig
    from cross.core.hypothesis import HypothesisManager
    from cross.core.system import System
    from cross.db import store
    from test_map_store import _db, _rng_image

    def stub():
        s = System.__new__(System)
        s.config, s._lc_verifier, s.topo_map, s.visualize = SystemConfig(), None, None, False
        s.state_device = s.storage_device = s.device = "cpu"
        s.pose_est_type, s.pose_est = PoseEstType.PNP, None
        s._anchor_pending, s._contra_pending, s.loaded_node_ids = [], [], frozenset()
        s._last_retrieved_results, s._projection_n_nodes = None, -1
        s.db = _db(monkeypatch)
        s.hypothesis_manager = HypothesisManager(s, 3, HypothesisConfig())
        s._geo = None
        s.odometry_calib = {}
        return s

    s1 = stub()
    hm, db = s1.hypothesis_manager, s1.db
    s1.current_atlas = db.create_atlas()
    hm.dist = (pp.identity_SE3(3), pp.se3(torch.ones(3, 6) * .1), torch.tensor([1., 0., 0.]))
    hm.create_hypothesis_branch(0, 0)
    rng = np.random.default_rng(0)
    for i in range(3):
        mu = pp.SE3(torch.tensor([[i * 1.0, 0.0, 0.0, 0, 0, 0, 1.0]] * 3))
        kf = db.insert(i, _rng_image(rng), None, mu=mu, sigma=pp.se3(torch.rand(3, 6)), weights=torch.tensor([1., 0., 0.]),
                       atlas=s1.current_atlas, timestamp=float(i))
        hm.nodes[kf.id] = kf
    calib = {"learned_depth": {"offset": 0.41, "source": "da3:x", "samples": 30, "span_s": 25.0}}
    s1.odometry_calib = dict(calib)
    s1.save_map(str(tmp_path / "a" / "map.pkl"))
    s2 = stub()
    s2.load_map(str(tmp_path / "a" / "map.pkl"))
    assert s2.odometry_calib == calib
    s2.hypothesis_manager.dist = (pp.identity_SE3(3), pp.se3(torch.ones(3, 6) * .1), torch.tensor([1., 0., 0.]))
    s2.save_map(str(tmp_path / "b" / "map.pkl"))
    assert store.read_map(str(tmp_path / "b" / "map.pkl"))["odometry_calib"] == calib
