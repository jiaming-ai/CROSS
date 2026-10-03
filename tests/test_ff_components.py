import numpy as np
import torch

from cross.core.types import Camera
from cross.cv.pose_est_ff import FFPrediction, covisibility_scores
from cross.utils.camera import get_transforms_ff


def test_transform_ff_kitti_like():
    K = np.array([[700.0, 0, 620.0], [0, 700.0, 190.0], [0, 0, 1.0]])
    cam = Camera(K=K.copy(), frame_width=1242, frame_height=375)
    rgb_tf, depth_tf = get_transforms_ff(cam, 512)
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    out = rgb_tf(img)
    assert out.shape == (3, cam.frame_height, cam.frame_width)
    assert cam.frame_width == 512 and cam.frame_height % 16 == 0
    # aspect crop to 0.5 -> 750 px wide crop -> scale 512/750
    s = 512 / 750
    assert abs(cam.fx - 700 * s) < 1e-6
    assert abs(cam.px - (620 - (1242 - 750) / 2) * s) < 1e-6
    d = depth_tf(torch.zeros(1, 375, 1242))
    assert d.shape == (1, cam.frame_height, cam.frame_width)


def test_transform_ff_square():
    cam = Camera(K=np.array([[320.0, 0, 319.5], [0, 320.0, 319.5], [0, 0, 1.0]]), frame_width=640, frame_height=640)
    rgb_tf, _ = get_transforms_ff(cam, 512)
    assert rgb_tf(np.zeros((640, 640, 3), dtype=np.uint8)).shape == (3, 512, 512)
    assert abs(cam.fx - 256.0) < 1e-6


def _synthetic_pred(device="cpu"):
    H, W = 64, 96
    K = np.array([[80.0, 0, W / 2], [0, 80.0, H / 2], [0, 0, 1]])
    # a fronto-parallel plane at depth 5 seen from two cameras displaced by 0.5 m in x
    depth0 = torch.full((H, W), 5.0)
    depth1 = torch.full((H, W), 5.0)
    c2w = np.repeat(np.eye(4)[None], 3, 0)
    c2w[1, 0, 3] = 0.5
    c2w[2, 2, 3] = 50.0            # camera far beyond the plane: no overlap
    depth2 = torch.full((H, W), 5.0)
    return FFPrediction(c2w=c2w, K=np.stack([K, K, K]), depth=torch.stack([depth0, depth1, depth2]),
                        depth_conf=None, hw=(H, W))


def test_covisibility_overlap_vs_none():
    pred = _synthetic_pred()
    s = covisibility_scores(pred, [1, 2], 0, grid=32)
    assert s[0] > 0.8          # small lateral shift of a plane: nearly all points consistent
    assert s[1] < 0.05         # camera behind the plane: nothing projects consistently


def test_principal_point_rotation():
    from cross.cv.pose_est_ff import principal_point_rotation
    K_centred = np.array([[320.0, 0, 319.5], [0, 320.0, 239.5], [0, 0, 1]])
    assert np.allclose(principal_point_rotation(K_centred, 640, 480), np.eye(3))
    K = np.array([[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1]])        # KITTI 00-02
    R = principal_point_rotation(K, 1241, 376)
    ray = np.array([(620.0 - 607.1928) / 718.856, (187.5 - 185.2157) / 718.856, 1.0])
    assert np.allclose(R @ [0, 0, 1.0], ray / np.linalg.norm(ray))           # model optical axis -> image-centre ray
    assert abs(np.degrees(np.arccos((np.trace(R) - 1) / 2)) - 1.04) < 0.01
    assert np.allclose(R @ R.T, np.eye(3))
