"""A loaded map's camera against the session's (System._use_map_camera): mismatches fail, a mono session in a stereo
map passes."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pypose")

from cross.core.system import System  # noqa: E402
from cross.core.types import Camera  # noqa: E402

K = np.array([[320.0, 0, 255.5], [0, 320.0, 191.5], [0, 0, 1]])
RIG = np.eye(4)
RIG[0, 3] = 0.064


def _session(rig=None, K_s=K, size=(512, 384)):
    return SimpleNamespace(camera=Camera(K_s.copy(), *size), T_right_in_left=rig)


def _record(rig=RIG, K_m=K, size=(512, 384)):
    return {"K": K_m.tolist(), "width": size[0], "height": size[1],
            "T_right_in_left": None if rig is None else rig.tolist()}


def test_same_camera_passes():
    System._use_map_camera(_session(), _record())               # mono session, stereo map
    System._use_map_camera(_session(rig=RIG), _record())        # stereo session, same rig
    System._use_map_camera(_session(), _record(rig=None))       # mono map


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
    System._use_map_camera(_session(), None)
