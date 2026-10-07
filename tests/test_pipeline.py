"""cross/pipeline.py: mode inputs, observation cadence, map-frame output between observations, sessions."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pp = pytest.importorskip("pypose")

from cross.mono.frontend import MonoEstimate  # noqa: E402
from cross.pipeline import OdometryFrontend, Pipeline, restrict_inputs  # noqa: E402


def translation(x):
    T = np.eye(4)
    T[0, 3] = x
    return T


class FakeMapper:
    """Records the observations; its belief is the frontend pose shifted by `offset` (a fixed map alignment)."""

    def __init__(self, offset=10.0):
        self.calls, self.loaded, self.offset = [], [], offset
        self.pose = np.eye(4)
        self.last_added_kf_id = None
        self.hypothesis_manager = SimpleNamespace(dist=None)

    def step(self, obs, data=None):
        self.calls.append(obs)
        if obs.get("delta_pose") is not None:
            self.pose = self.pose @ obs["delta_pose"]
        if obs.get("rgb") is not None:
            mu = pp.mat2SE3(torch.tensor(translation(self.offset) @ self.pose, dtype=torch.float32)).unsqueeze(0)
            self.hypothesis_manager.dist = (mu, None, torch.ones(1))

    def get_current_pose(self):
        return self.hypothesis_manager.dist[0][0]

    def load_map(self, path):
        self.loaded.append(path)


class FakeFrontend:
    def __init__(self, valid_from=0):
        self.index, self.pose, self.valid_from = 0, np.eye(4), valid_from

    def track(self, frame):
        delta = translation(1.0) if self.index else np.eye(4)
        self.pose = self.pose @ delta
        self.index += 1
        return MonoEstimate(frame["timestamp"], self.pose.copy(), delta, np.eye(6) * 1e-4, None,
                            dict(valid=self.index - 1 >= self.valid_from))


def frame(i, **extra):
    return dict(rgb=np.zeros((4, 4, 3), np.uint8), timestamp=float(i), depth=np.ones((4, 4), np.float32),
                rgb_right=np.zeros((4, 4, 3), np.uint8), delta_pose=translation(0.5), **extra)


def to_mat(p):
    return p.matrix().detach().cpu().numpy().astype(np.float64)


def test_inputs_of_each_mode():
    f = frame(0)
    mono = restrict_inputs(f, "mono", "visual")
    assert mono["depth"] is None and mono["rgb_right"] is None and mono["delta_pose"] is None
    rgbd = restrict_inputs(f, "rgbd", "external")
    assert rgbd["depth"] is not None and rgbd["rgb_right"] is None and rgbd["delta_pose"] is not None
    stereo = restrict_inputs(f, "stereo", "visual")
    assert stereo["rgb_right"] is not None and stereo["delta_pose"] is None
    assert f["depth"] is not None          # the loader's frame is not modified


def test_external_odometry_passes_frames_unchanged():
    mapper = FakeMapper()
    session = Pipeline(mapper, None, mode="rgbd", odometry="external")
    f = frame(0)
    session.process(f)
    assert mapper.calls == [f] and session.mapped_now


def test_observation_cadence_and_map_frame_poses_between_observations():
    mapper = FakeMapper(offset=10.0)
    session = Pipeline(mapper, FakeFrontend(valid_from=2), mapping_interval=3, mode="stereo", odometry="visual")
    for i in range(9):
        session.process(frame(i))
        c0, best, _ = session.belief(to_mat)
        if session.initialized:
            # map frame = frontend pose + the mapper's fixed offset, observed or not
            assert np.allclose(c0[:3, 3], [10.0 + i, 0, 0], atol=1e-4)
            assert np.allclose(best, c0, atol=1e-4)
    observed = [i for i, c in enumerate(mapper.calls) if c["rgb"] is not None]
    assert observed == [2, 3, 6]            # first valid frame, then every third frame
    assert all(c["delta_pose"] is not None for c in mapper.calls)       # motion reaches the back end every frame
    assert mapper.calls[2]["initial_chart_pose"] is not None and mapper.calls[3]["initial_chart_pose"] is None
    assert mapper.calls[2]["rgb_right"] is not None and mapper.calls[4]["rgb_right"] is None
    assert np.allclose(mapper.calls[3]["depth"], 1.0)                  # the frame's own depth (stereo mode)


def test_mono_mode_predicts_depth_for_observations():
    calls = []

    class Depth:
        def predict_metric(self, rgb, K, shape):
            calls.append(shape)
            return np.full(shape, 2.0, np.float32)
    mapper = FakeMapper()
    session = Pipeline(mapper, FakeFrontend(), mapping_interval=2, mode="mono", odometry="visual",
                       depth_model=Depth(), K=np.eye(3))
    for i in range(4):
        session.process(frame(i))
    assert len(calls) == 2 and all(np.allclose(c["depth"], 2.0) for c in mapper.calls if c["rgb"] is not None)


def test_load_map_starts_a_fresh_frontend():
    mapper = FakeMapper()
    session = Pipeline(mapper, FakeFrontend(), 1, "rgbd", "visual", frontend_factory=FakeFrontend)
    session.process(frame(0))
    first = session.frontend
    session.load_map("map.pkl")
    assert session.frontend is not first and session.frontend.index == 0 and not session.initialized
    assert mapper.loaded == ["map.pkl"]


def test_odometry_frontend_composes_dataset_motion():
    frontend = OdometryFrontend()
    poses = [frontend.track(frame(i)).pose for i in range(3)]
    assert np.allclose([p[0, 3] for p in poses], [0.0, 0.5, 1.0])      # the first frame's motion is ignored


def test_mode_config_files_and_left_image():
    """The shipped configuration of each mode and odometry: the stereo mode with external odometry observes on the left
    image (no stereo anchors) and half as often; its VGGT-inertial and visual odometries keep the pair and the cadence."""
    from cross.core.config import load_config
    from cross.pipeline import apply_right_image, mode_config_files
    ext = apply_right_image(load_config(*mode_config_files("stereo", "external")))
    own = apply_right_image(load_config(*mode_config_files("stereo", "vgio")))
    assert mode_config_files("rgbd", "external") == [] and mode_config_files("mono", "external") == []
    assert [p.rsplit("/", 1)[-1] for p in mode_config_files("stereo", "visual", fast=True)] == ["stereo.yaml", "stereo_fast.yaml"]
    pe = ext.pose_est
    assert (pe.obs_min_translation, pe.obs_min_rotation, pe.obs_max_interval_steps) == (0.6, 0.3, 6)
    assert pe.ff.right_image == "left" and not pe.ff.use_curr_anchor and pe.ff.n_ref_anchors == 0
    assert pe.ff.use_odom_anchor and pe.ff.use_map_anchors and not pe.ff.store_right_images
    pe = own.pose_est
    assert (pe.obs_min_translation, pe.obs_min_rotation, pe.obs_max_interval_steps) == (0.3, 0.15, 3)
    assert pe.ff.right_image == "pair" and pe.ff.use_curr_anchor and pe.ff.n_ref_anchors == 2
    # the fast preset keeps the pair (its only scale anchor) and the stereo.yaml cadence with external odometry too
    pe = apply_right_image(load_config(*mode_config_files("stereo", "external", fast=True))).pose_est
    assert (pe.obs_min_translation, pe.obs_max_interval_steps, pe.ff.right_image, pe.ff.max_refs) == (0.3, 3, "pair", 4)
    assert pe.ff.use_curr_anchor and pe.ff.n_ref_anchors == 0


def test_left_image_session_drops_the_right_image():
    mapper = FakeMapper()
    mapper.config = SimpleNamespace(pose_est=SimpleNamespace(ff=SimpleNamespace(right_image="left")))
    session = Pipeline(mapper, None, mode="stereo", odometry="external")
    f = frame(0)
    session.process(f)
    assert mapper.calls[0]["rgb_right"] is None and mapper.calls[0]["rgb"] is f["rgb"]
    assert f["rgb_right"] is not None      # the loader's frame is not modified


def test_vgio_align_by_mode():
    from cross.mono.config import MonoConfig
    from cross.pipeline import _vgio_config
    stereo = _vgio_config(MonoConfig(), "stereo", np.eye(4))
    mono = _vgio_config(MonoConfig(), "mono")
    assert stereo.imu.vgio_align is True and mono.imu.vgio_align is False
    explicit = MonoConfig()
    explicit.imu.vgio_align = False
    assert _vgio_config(explicit, "stereo", np.eye(4)).imu.vgio_align is False
