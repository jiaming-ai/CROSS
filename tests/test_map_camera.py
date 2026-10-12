"""Cameras of a map (System.cameras, Keyframe.cam_id): a session relocalizes in and extends a map made with another
camera (System._use_map_camera); the feed-forward estimator corrects every view with its own camera
(PoseEstFeedForward.set_cameras)."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pypose")

from cross.core.config import FeedForwardConfig, PoseEstType  # noqa: E402
from cross.core.system import System  # noqa: E402
from cross.core.types import Camera, Keyframe  # noqa: E402
from cross.cv.pose_est_ff import FFPrediction, PoseEstFeedForward, principal_point_rotation  # noqa: E402
from cross.utils.camera import get_transforms_ff  # noqa: E402

K = np.array([[320.0, 0, 255.5], [0, 320.0, 191.5], [0, 0, 1]])
RIG = np.eye(4)
RIG[0, 3] = 0.064


class _Est:
    def __init__(self):
        self.cameras = None

    def set_camera(self, K, w, h):
        self.camera = (np.asarray(K).copy(), w, h)

    def set_cameras(self, table, sid):
        self.cameras = (copy.deepcopy(table), sid)


def _session(rig=None, K_s=K, size=(512, 384), est=PoseEstType.FF, nodes=None):
    """A System with only what _use_map_camera reads; the input camera is the processed one unless given."""
    s = System.__new__(System)
    s.camera = Camera(np.asarray(K_s, dtype=np.float64).copy(), *size)
    s._input_camera = copy.deepcopy(s.camera)
    s.T_right_in_left = rig
    s.pose_est_type = est
    s.pose_est = _Est()
    s._kf_quality = None
    s.transforms_shared = False
    s.hypothesis_manager = SimpleNamespace(nodes=nodes or {})
    s.cameras, s.cam_id = [s._camera_record()], 0
    return s


def _record(rig=RIG, K_m=K, size=(512, 384)):
    return {"K": np.asarray(K_m).tolist(), "width": size[0], "height": size[1],
            "T_right_in_left": None if rig is None else rig.tolist()}


def test_same_camera_is_camera_0():
    for sess, rec in ((_session(), _record()),                  # mono session, stereo map
                      (_session(rig=RIG), _record()),           # stereo session, same rig
                      (_session(rig=RIG), _record(rig=None))):  # stereo session, mono map: the map entry takes the rig
        sess._use_map_camera(rec)
        assert sess.cam_id == 0 and len(sess.cameras) == 1
    assert sess.cameras[0]["T_right_in_left"] == RIG.tolist()


def test_other_camera_gets_a_new_id():
    K2 = K.copy()
    K2[0, 0] = K2[1, 1] = 300.0
    s = _session(K_s=K2)
    s._use_map_camera(_record())
    assert s.cam_id == 1 and len(s.cameras) == 2 and s.pose_est.cameras[1] == 1
    rig2 = RIG.copy()
    rig2[0, 3] = 0.12
    s = _session(rig=rig2)                                      # same left camera, another rig: another camera
    s._use_map_camera(_record())
    assert s.cam_id == 1
    with pytest.raises(ValueError, match="another camera"):     # PnP uses the session's intrinsics for every view
        _session(K_s=K2, est=PoseEstType.PNP)._use_map_camera(_record())


def test_other_image_size_takes_the_map_canvas():
    """A 16:9 camera with an off-centre principal point in a map of 512 x 384 images: centred 4:3 crop, resize."""
    Kb = np.array([[553.0, 0, 448.0], [0, 553.0, 228.0], [0, 0, 1]])     # 848 x 480
    s = _session(K_s=Kb, size=(848, 480))
    s._use_map_camera(_record())
    assert (s.camera.frame_width, s.camera.frame_height) == (512, 384)
    sc = 512 / 640                                                      # 640 x 480 crop, 104 px off each side
    assert np.allclose(s.camera.K[0], [553.0 * sc, 0, (448.0 - 104) * sc])
    assert np.allclose(s.camera.K[1], [0, 553.0 * sc, 228.0 * sc])
    img = np.zeros((480, 848, 3), dtype=np.uint8)
    img[:, 104 + 320] = 255                                             # crop centre column -> canvas centre
    out = s.rgb_transform(img)
    assert out.shape == (3, 384, 512) and int(out[0, 192].argmax()) in (255, 256)
    assert s.cam_id == 1 and s.pose_est.camera[1:] == (512, 384)
    s = _session(K_s=Kb, size=(848, 480))
    s.transforms_shared = True                                          # vgio: transforms are shared
    with pytest.raises(ValueError, match="cannot change its image size"):
        s._use_map_camera(_record())


def test_canvas_transform_taller_camera():
    cam = Camera(np.array([[400.0, 0, 319.5], [0, 400.0, 319.5], [0, 0, 1]]), 640, 640)
    rgb_tf, depth_tf = get_transforms_ff(cam, canvas=(512, 384))
    assert rgb_tf(np.zeros((640, 640, 3), dtype=np.uint8)).shape == (3, 384, 512)
    assert depth_tf(torch.zeros(1, 640, 640)).shape == (1, 384, 512)
    assert np.isclose(cam.fx, 400.0 * 0.8) and np.isclose(cam.py, (319.5 - 80) * 0.8)


def test_legacy_maps_and_keyframe_ids():
    s = _session()
    s._use_map_camera(None)                                    # saved before the camera record
    assert s.cam_id == 0 and len(s.cameras) == 1
    K2 = K.copy()
    K2[0, 2] += 20
    kfs = {i: Keyframe.__new__(Keyframe) for i in range(3)}
    for i, kf in kfs.items():
        kf.id = i
    s = _session(nodes=kfs)
    s._use_map_camera(_record(), {"table": [_record(), _record(K_m=K2)], "kf": np.array([2]), "cam": np.array([1])})
    assert [kfs[i].cam_id for i in range(3)] == [0, 0, 1] and s.cam_id == 0
    s.db = SimpleNamespace(get_all_keyframes=lambda: list(kfs.values()))
    cols = s._camera_columns()
    assert cols["kf"].tolist() == [2] and cols["cam"].tolist() == [1] and len(cols["table"]) == 2


def _pass_estimator(monkeypatch, K_s, K_r, rig_r, n_ref_anchors=1):
    est = PoseEstFeedForward.__new__(PoseEstFeedForward)
    est.device = "cpu"
    est.config = FeedForwardConfig(covis_source="head", min_covis=0.0, n_ref_anchors=n_ref_anchors, scale_method="mean")
    est.backend = SimpleNamespace(covis_head=object())
    est._R_model_to_cam = None
    est.T_right_in_left = None                                  # a session without a rig (left image / mono)
    est.set_camera(K_s, 512, 384)
    est.set_cameras([_record(rig=None, K_m=K_s), _record(rig=rig_r, K_m=K_r)], 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    return est


def test_mixed_pass_uses_each_views_camera(monkeypatch):
    """Ground truth in calibrated frames: the current camera at the origin, a reference of camera 1 1 m to its right,
    both looking ahead, the reference's stored right image 0.12 m further.  The model predicts each view in its own
    centred-principal-point frame (X_model = X_cal M^-1) at half the metric scale; the reference's stereo pair (camera
    1's rig) gives the scale, the per-view corrections the relative pose."""
    K_s = np.array([[320.0, 0, 275.5], [0, 320.0, 181.5], [0, 0, 1]])      # off-centre by (+20, -10) px
    K_r = np.array([[280.0, 0, 235.5], [0, 280.0, 201.5], [0, 0, 1]])      # off-centre by (-20, +10) px
    rig_r = np.eye(4)
    rig_r[0, 3] = 0.12
    est = _pass_estimator(monkeypatch, K_s, K_r, rig_r)
    M = {0: np.eye(4), 1: np.eye(4)}
    M[0][:3, :3] = principal_point_rotation(K_s, 512, 384).T
    M[1][:3, :3] = principal_point_rotation(K_r, 512, 384).T
    X_cur = np.eye(4)
    X_ref = np.eye(4)
    X_ref[0, 3] = 1.0
    X_refR = X_ref @ rig_r
    model = [X_cur @ np.linalg.inv(M[0]), X_ref @ np.linalg.inv(M[1]), X_refR @ np.linalg.inv(M[1])]
    c2w = np.stack(model)
    c2w[:, :3, 3] *= 0.5

    def infer(images, n_depth=None, focal=None):
        n = images.shape[0]                                     # current, reference (, reference right)
        return FFPrediction(c2w=c2w[:n], K=np.stack([K_s] * n), depth=torch.zeros(0, 384, 512), depth_conf=None,
                            hw=(384, 512), covis=np.ones((n, n)))
    est.backend.infer = infer
    img = torch.rand(3, 384, 512)
    poses, valid, _ = est.estimate_pose(img[None], None, img, None, ref_images_right=[img], ref_cam_ids=[1])
    assert valid.tolist() == [True]
    assert abs(est.last_scale.scale - 2.0) < 1e-6
    T = poses.matrix()[0].double().numpy()
    assert np.allclose(T, np.linalg.inv(X_ref) @ X_cur, atol=1e-5)
    # a reference of the session's camera has no right-image anchor without a session rig: no scale
    poses, valid, _ = est.estimate_pose(img[None], None, img, None, ref_images_right=[img], ref_cam_ids=[0])
    assert not valid.any()
