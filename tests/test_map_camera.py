"""A loaded map's camera against the session's (System._use_map_camera): mismatches fail, a session without a rig
takes the map's for its stored right images."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pypose")

from cross.core.config import PoseEstType, SystemConfig  # noqa: E402
from cross.core.system import System  # noqa: E402
from cross.core.types import Camera  # noqa: E402

K = np.array([[320.0, 0, 255.5], [0, 320.0, 191.5], [0, 0, 1]])
RIG = np.eye(4)
RIG[0, 3] = 0.064


class _PoseEst:
    T_right_in_left = None

    def set_stereo_calibration(self, T):
        self.T_right_in_left = T


def _session(rig=None, K_s=K, size=(512, 384), n_ref_anchors=2, right_images=True):
    cfg = SystemConfig()
    cfg.pose_est.type = PoseEstType.FF
    cfg.pose_est.ff.n_ref_anchors = n_ref_anchors
    kf = SimpleNamespace(has_image=lambda name: right_images and name == "raw_rgb_right")
    s = SimpleNamespace(camera=Camera(K_s.copy(), *size), T_right_in_left=rig, pose_est_type=PoseEstType.FF, config=cfg,
                        db=SimpleNamespace(get_all_keyframes=lambda: [kf]), pose_est=_PoseEst())
    return s


def _record(rig=RIG, K_m=K, size=(512, 384)):
    return {"K": K_m.tolist(), "width": size[0], "height": size[1],
            "T_right_in_left": None if rig is None else rig.tolist()}


def test_mono_session_takes_the_map_rig():
    s = _session()
    System._use_map_camera(s, _record())
    np.testing.assert_allclose(s.pose_est.T_right_in_left, RIG)


def test_mono_session_without_right_images_or_anchors_gets_no_rig():
    for kw in ({"right_images": False}, {"n_ref_anchors": 0}):
        s = _session(**kw)
        s.pose_est.T_right_in_left = RIG       # left over from an earlier map: cleared
        System._use_map_camera(s, _record())
        assert s.pose_est.T_right_in_left is None
    s = _session()
    System._use_map_camera(s, _record(rig=None))     # a mono map
    assert s.pose_est.T_right_in_left is None


def test_stereo_session_keeps_its_rig():
    s = _session(rig=RIG)
    System._use_map_camera(s, _record())
    assert s.pose_est.T_right_in_left is None       # untouched: the stereo session set its own at construction


def test_other_camera_fails():
    K2 = K.copy()
    K2[0, 0] = K2[1, 1] = 300.0
    with pytest.raises(ValueError, match="another camera"):
        System._use_map_camera(_session(), _record(K_m=K2))
    with pytest.raises(ValueError, match="another camera"):
        System._use_map_camera(_session(size=(512, 288)), _record())


def test_other_rig_fails_for_a_stereo_session():
    rig2 = RIG.copy()
    rig2[0, 3] = 0.12
    with pytest.raises(ValueError, match="another stereo rig"):
        System._use_map_camera(_session(rig=rig2), _record())


def test_old_map_without_record():
    s = _session()
    System._use_map_camera(s, None)
    assert s.pose_est.T_right_in_left is None
