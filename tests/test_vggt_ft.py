"""Checks of the fine-tuning package that need no data or GPU (depth codec, crops, GT covisibility, normalisation,
losses, scene cache round trip)."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vggt_ft.dataio.depthcodec import decode, encode
from vggt_ft.dataio.transforms import centred_crop_resize, target_shape
from vggt_ft.geometry import connected_to_ref, gt_covisibility, normalize_window


def test_depth_codec_round_trip():
    d = np.array([[0.0, 0.01, 0.5], [3.0, 120.0, 1e4]], np.float32)
    r = decode(encode(d))
    assert r[0, 0] == 0 and r[1, 2] == 0                      # invalid and out-of-range stay invalid
    ok = d[(d > 0) & (d <= 1000)]
    assert np.allclose(r[(d > 0) & (d <= 1000)], ok, rtol=2e-4)


def test_target_shape_balanced():
    h, w = target_shape(0.75)
    assert h % 16 == 0 and w % 16 == 0 and abs(h * w - 512 * 512) / (512 * 512) < 0.05 and abs(h / w - 0.75) < 0.05


def test_crop_keeps_principal_point_centred():
    img = np.zeros((480, 640, 3), np.uint8)
    dep = np.ones((480, 640), np.float32)
    K = np.array([[500.0, 0, 300.0], [0, 500.0, 250.0], [0, 0, 1]])
    for zoom in (1.0, 0.7):
        im2, d2, K2 = centred_crop_resize(img, dep, K, (336, 448), zoom)
        assert im2.shape[:2] == (336, 448) and d2.shape == (336, 448)
        assert abs(K2[0, 2] - (448 - 1) / 2) < 1.0 and abs(K2[1, 2] - (336 - 1) / 2) < 1.0


def _window(S=3, H=24, W=32, shift=0.1):
    K = torch.tensor([[30.0, 0, (W - 1) / 2], [0, 30.0, (H - 1) / 2], [0, 0, 1]]).expand(1, S, 3, 3).clone()
    E = torch.eye(4).expand(1, S, 4, 4).clone()
    for s in range(S):
        E[0, s, 0, 3] = -shift * s                         # cameras moving along +x
    depth = torch.full((1, S, H, W), 2.0)
    return depth, depth > 0, E, K


def test_covisibility_and_connectivity():
    depth, mask, E, K = _window()
    cov = gt_covisibility(depth, mask, E, K)
    assert torch.allclose(cov.diagonal(dim1=1, dim2=2), torch.ones(1, 3))
    assert cov[0, 0, 1] > 0.8 and cov[0, 0, 2] > 0.7
    far = E.clone()
    far[0, 2, 0, 3] = -100.0
    cov2 = gt_covisibility(depth, mask, far, K)
    assert cov2[0, 0, 2] < 0.01
    assert connected_to_ref(cov2).tolist() == [[True, True, False]]
    wid = torch.tensor([[1, 1, 2]])
    assert gt_covisibility(depth, mask, E, K, wid)[0, 0, 2] == 0


def test_normalize_window_unit_scale():
    depth, mask, E, K = _window()
    E_n, d_n, pts, scale = normalize_window(E, depth, mask, K)
    assert torch.allclose(E_n[0, 0], torch.eye(4), atol=1e-6)
    assert abs(float(pts[mask].norm(dim=-1).mean()) - 1.0) < 1e-4
    assert torch.allclose(d_n * scale[:, None, None, None], depth)


def test_losses_zero_at_ground_truth():
    from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding
    from vggt_ft.losses import camera_loss, covis_loss
    depth, mask, E, K = _window()
    E_n, d_n, pts, scale = normalize_window(E, depth, mask, K)
    enc = extri_intri_to_pose_encoding(E_n[:, :, :3], K, (24, 32))
    cl = camera_loss(enc, E_n, K, (24, 32))
    assert float(cl["loss_camera"]) < 1e-5
    enc2 = enc.clone()
    enc2[..., 3:7] *= -1                                     # q and -q are the same rotation
    assert float(camera_loss(enc2, E_n, K, (24, 32))["loss_R"]) < 1e-5
    tgt = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert float(covis_loss(torch.tensor([[[9.0, -9.0], [-9.0, 9.0]]]), tgt)["loss_covis"]) < 1e-3


def test_scene_cache_round_trip(tmp_path):
    from vggt_ft.dataio.scene import load_scene
    from vggt_ft.prep.common import SceneWriter
    sw = SceneWriter(tmp_path, "toy", "s0", metric=True)
    q = sw.sequence("a")
    K = np.array([[100.0, 0, 63.5], [0, 100.0, 47.5], [0, 0, 1]])
    for i in range(3):
        E = np.eye(4)
        E[0, 3] = -0.1 * i
        q.add(f"{i:03d}", np.full((96, 128, 3), 80 + i, np.uint8), np.full((96, 128), 2.5, np.float32), K, E)
    assert sw.close() == 3
    sc = load_scene(tmp_path / "toy" / "s0")
    img, dep, K2, E2 = sc.seq("a").load(2)
    assert img.shape == (96, 128, 3) and np.allclose(dep, 2.5, rtol=1e-3)
    assert np.allclose(K2, K) and np.allclose(E2[0, 3], -0.2)


def test_scale_distill_target():
    import math
    import torch
    from vggt_ft.train import scale_distill_target
    d_s = torch.rand(2, 3, 8, 8, 1) + 0.5
    steach = {"depth": 2.0 * d_s, "log_scale": torch.full((2,), math.log(3.0))}
    tgt = scale_distill_target(steach, {"depth": d_s})
    assert torch.allclose(tgt, torch.full((2,), math.log(6.0)), atol=1e-5)


def test_pseudo_metric_scale(tmp_path):
    import json
    from vggt_ft.dataio.windows import DatasetPool
    from vggt_ft.prep.common import SceneWriter
    sw = SceneWriter(tmp_path, "toy", "s0", metric=False)
    q = sw.sequence("a")
    K = np.array([[100.0, 0, 63.5], [0, 100.0, 47.5], [0, 0, 1]])
    for i in range(3):
        E = np.eye(4)
        E[0, 3] = -0.1 * i
        q.add(f"{i:03d}", np.full((96, 128, 3), 80, np.uint8), np.full((96, 128), 2.0, np.float32), K, E)
    sw.close()
    json.dump({"s0": {"n": 3, "k": {"ens": 3.0}, "iqr": {"ens": [2.9, 3.1]}}},
              open(tmp_path / "toy" / "pseudo_metric.json", "w"))
    pool = DatasetPool(str(tmp_path), "toy", {"pseudo_metric": True})
    sc = pool.scenes[0]
    assert sc.metric and sc.pseudo_metric and sc.scale == 3.0
    _, dep, _, E2 = sc.seq("a").load(2)
    assert np.allclose(dep, 6.0, rtol=1e-3) and np.allclose(E2[0, 3], -0.6)
    assert np.allclose(sc.seq("a").centres[2], [0.6, 0, 0])


def test_multilayer_scale_head_shapes():
    import torch
    from vggt_ft.heads import build_scale_head
    head = build_scale_head({"arch": "multilayer", "layers": [-1, 4, 23], "grid": 4})
    B, S, h, w = 2, 3, 6, 8
    tokens = [None] * 24
    for l in (4, 23):
        tokens[l] = torch.randn(B, S, 17 + h * w, 2048)
    out = head(tokens, 17, (h, w), torch.randn(B, S, h * w, 1024))
    assert out.shape == (B,) and torch.allclose(out, torch.full((B,), math.log(2.0)), atol=1e-5)


def test_invariant_alignment_ignores_global_rescale():
    """scale_align: invariant -> the geometric losses do not change when every output is scaled by the same factor."""
    from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding
    from vggt_ft.train import compute_losses
    depth, mask, E, K = _window()
    S, H, W = depth.shape[1:]
    E_n, d_n, _, _ = normalize_window(E, depth, mask, K)
    enc = extri_intri_to_pose_encoding(E_n[:, :, :3], K, (H, W))
    noise = 1 + 0.05 * torch.randn(1, S, H, W, generator=torch.Generator().manual_seed(0))
    batch = {"depths": depth, "masks": mask, "extrinsics": E, "intrinsics": K, "world_id": torch.zeros(1, S, dtype=torch.long),
             "same_session": torch.ones(1, S, dtype=torch.bool), "is_neg": torch.zeros(1, S, dtype=torch.bool),
             "metric": torch.ones(1, dtype=torch.bool)}
    lc = {"scale_align": "invariant", "w_scale": 0.0, "w_covis": 0.0}

    def total(c):
        e = enc.clone()
        e[..., :3] = e[..., :3] * c
        pred = {"pose_enc": e, "depth": (d_n * noise * c)[..., None], "depth_conf": torch.ones(1, S, H, W)}
        return float(compute_losses(pred, batch, lc, (H, W))["loss"])
    assert abs(total(1.0) - total(3.0)) < 1e-4 * max(1.0, abs(total(1.0)))
