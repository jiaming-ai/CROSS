"""CROSS World (cross_world/) and the frame capture of the mapping (cross/core/world_capture.py): the parts that run
without a GPU rasterizer."""
import gzip
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cross.core.config import WorldCaptureConfig
from cross.core.world_capture import WorldCapture, capture_dir
from cross_world.export import spz_bytes, view_rotation
from cross_world.gaussians import matrix_to_quat, quat_mul
from cross_world.map_views import MapViews, SourceSequence, View, load_capture_views
from cross_world.partition import assign_views, make_partition
from cross_world.world import ChunkModel, World, compute_anchors


def _pose(yaw=0.0, t=(0, 0, 0)):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]           # turn about the camera's y (vertical) axis
    T[:3, 3] = t
    return T


def test_partition_balanced_cells_tile_the_region():
    rng = np.random.default_rng(0)
    centers = np.c_[rng.uniform(0, 100, 500), rng.normal(0, 0.1, 500), rng.uniform(0, 20, 500)]
    ids = list(range(500))
    part = make_partition(ids, centers, np.array([0, -1.0, 0]), max_views=100, margin=5.0)
    assert all(len(c.core_ids) <= 100 for c in part.chunks)
    assert sorted(i for c in part.chunks for i in c.core_ids) == ids
    pts = np.c_[rng.uniform(-4, 104, 2000), np.zeros(2000), rng.uniform(-4, 24, 2000)]
    cell = part.cell_of(pts)
    assert (cell >= 0).all()                                   # the cells cover the region without gaps
    samples = {i: centers[i][None] + np.array([[0, 0, 1.0]]) for i in ids}
    assign_views(part, ids, centers, samples, min_visible=0.5, min_points=1)
    for c in part.chunks:
        assert set(c.core_ids) <= set(c.train_ids)


def test_spz_header_and_layout():
    n = 10
    rng = np.random.default_rng(1)
    q = rng.normal(size=(n, 4))
    b = spz_bytes(rng.normal(size=(n, 3)), q, np.full((n, 3), -3.0), rng.uniform(size=n), rng.normal(size=(n, 3)),
                  rng.normal(scale=0.1, size=(n, 15, 3)), sh_degree=1)
    raw = gzip.decompress(b)
    magic, version, count, deg, frac, flags, _ = struct.unpack("<IIIBBBB", raw[:16])
    assert (magic, version, count, deg, frac) == (0x5053474E, 2, n, 1, 12)
    assert len(raw) == 16 + n * (9 + 1 + 3 + 3 + 3 + 9)       # centres, alpha, colour, scale, rotation, SH band 1
    scale = raw[16 + n * 13: 16 + n * 16]
    assert set(scale) == {round((-3.0 + 10) * 16)}


def test_view_rotation_maps_up_to_y():
    up = np.array([0.1, -0.99, 0.05])
    up /= np.linalg.norm(up)
    R = view_rotation(up, np.array([[1.0, 0, 0], [0, 0, 1.0]]))
    assert np.allclose(R @ up, [0, 1, 0], atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_quaternions():
    R = torch.from_numpy(_pose(0.3)[:3, :3]).float()
    q = matrix_to_quat(R[None])[0]
    q2 = quat_mul(q, q)
    R2 = torch.from_numpy(_pose(0.6)[:3, :3]).float()
    assert torch.allclose(matrix_to_quat(R2[None])[0], q2, atol=1e-5)


def _world(kf_poses, means):
    from cross_world.partition import Chunk, Partition
    part = Partition(np.zeros(3), np.array([[1.0, 0, 0], [0, 0, 1.0]]), np.array([0, -1.0, 0]), np.array([-1e3, -1e3]),
                     np.array([1e3, 1e3]), [Chunk(0, np.array([-1e3, -1e3]), np.array([1e3, 1e3]), list(kf_poses))], 1.0)
    n = len(means)
    sp = {"means": means.float(), "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1), "scales": torch.zeros(n, 3),
          "opacities": torch.zeros(n), "sh0": torch.zeros(n, 1, 3), "shN": torch.zeros(n, 0, 3)}
    ids = sorted(kf_poses)
    ai, aw = compute_anchors(sp["means"], ids, np.stack([kf_poses[k][:3, 3] for k in ids]), 2)
    return World(part, [ChunkModel(0, {"near": sp}, {"near": ai}, {"near": aw})], dict(kf_poses), {}, {}, {"sh_degree": 0})


def test_repose_identity_and_rigid_motion():
    kf = {1: _pose(0, (0, 0, 0)), 2: _pose(0, (2, 0, 0)), 3: _pose(0, (4, 0, 0))}
    means = torch.tensor([[0.5, 0.2, 3.0], [3.0, -0.1, 5.0]])
    w = _world(kf, means)
    w.repose(dict(kf))
    assert torch.allclose(w.chunks[0].layers["near"]["means"], means, atol=1e-6)
    D = _pose(0.2, (1.0, 0.0, -2.0))                          # the whole map moves rigidly: the Gaussians follow exactly
    w.repose({k: D @ T for k, T in kf.items()})
    expect = (torch.from_numpy(D[:3, :3]).float() @ means.T).T + torch.from_numpy(D[:3, 3]).float()
    assert torch.allclose(w.chunks[0].layers["near"]["means"], expect, atol=1e-5)
    q = w.chunks[0].layers["near"]["quats"][0]
    assert torch.allclose(q, matrix_to_quat(torch.from_numpy(D[:3, :3]).float()[None])[0], atol=1e-5)


def test_source_clock_calibration(tmp_path):
    (tmp_path / "left").mkdir()
    for i in range(20):
        (tmp_path / "left" / f"{i:06d}.png").write_bytes(b"")
    (tmp_path / "calib.json").write_text(json.dumps({"K": np.eye(3).tolist(), "width": 4, "height": 4, "fps": 10.0}))
    t = 1.3e9 + np.arange(20) * 0.1037                        # absolute, drifting clock in times.txt
    np.savetxt(tmp_path / "times.txt", t, fmt="%.6f")
    src = SourceSequence(tmp_path)
    src.calibrate([0.0, 1.2, 1.5])                            # the map uses frame index / fps
    assert src.index_of(1.2) == 12 and src.index_of(1.5) == 15
    src.calibrate([t[3], t[7]])                               # the map uses times.txt
    assert src.index_of(t[7]) == 7


def test_capture_frames_follow_their_anchor_keyframes(tmp_path):
    cfg = WorldCaptureConfig(enabled=True, min_translation=0.0, min_rotation_deg=0.0)
    cap = WorldCapture(cfg, np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]]), 64, 48)
    kf = {7: _pose(0.0, (0, 0, 0)), 9: _pose(0.1, (1, 0, 0))}
    T = _pose(0.05, (0.5, 0, 0.2))
    img = (np.random.default_rng(0).uniform(size=(48, 64, 3)) * 255).astype(np.uint8)
    cap.add(T, [(7, kf[7], 0.5), (9, kf[9], 0.6)], {"rgb": img, "depth": np.full((48, 64), 2.0, np.float32)}, 3.0)
    map_path = tmp_path / "map.pkl"
    cap.save(map_path)
    assert (capture_dir(map_path) / "capture.json").exists()
    views = [View(id=k, T_wc=P, K=np.eye(3), width=64, height=48, _image=lambda: img) for k, P in kf.items()]
    mv = MapViews(views=views, edges=[], T_right_in_left=None, mode="rgbd", map_path=str(map_path))
    cv = load_capture_views(mv)
    assert len(cv) == 1 and np.allclose(cv[0].T_wc, T, atol=1e-9)
    assert cv[0].image().shape == (48, 64, 3)
    assert np.allclose(cv[0].depth(), 2.0, atol=1e-3)
    D = _pose(0.3, (5, 0, 1))                                 # the map moves rigidly: the frame moves with it
    for v in views:
        v.T_wc = D @ v.T_wc
    assert np.allclose(load_capture_views(mv)[0].T_wc, D @ T, atol=1e-9)
    assert load_capture_views(mv, exclude_times=np.array([3.05]))  == []
    cap.cleanup()


def test_keyframes_take_their_captured_frames(tmp_path):
    from cross_world.map_views import _keyframes_from_capture
    cfg = WorldCaptureConfig(enabled=True, min_translation=0.0, min_rotation_deg=0.0)
    cap = WorldCapture(cfg, np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]]), 64, 48)
    kf = {7: _pose(0.0, (0, 0, 0)), 9: _pose(0.1, (1, 0, 0))}
    big = np.full((48, 64, 3), 200, np.uint8)
    for t, k in ((1.0, 7), (2.0, 9)):
        cap.add(kf[k], [(k, kf[k], 0.1)], {"rgb": big}, t)
    map_path = tmp_path / "map.pkl"
    cap.save(map_path)
    small = np.zeros((24, 32, 3), np.uint8)
    views = [View(id=k, T_wc=kf[k], K=np.eye(3), width=32, height=24, timestamp=t, _image=lambda: small)
             for t, k in ((1.0, 7), (2.0, 9))]
    assert _keyframes_from_capture(views, map_path, None) == 2
    assert views[0].width == 64 and views[0].image().shape == (48, 64, 3) and views[0].K[0, 0] == 100.0
    views.append(View(id=11, T_wc=np.eye(4), K=np.eye(3), width=32, height=24, timestamp=3.0, _image=lambda: small))
    with pytest.warns(UserWarning):
        assert _keyframes_from_capture(views[2:], map_path, None) == 0
    cap.cleanup()


def test_sky_texture_directions_fill_and_shell():
    from cross_world.sky import SkyModel, pixel_dirs, push_pull, sky_frame, sky_splats
    up = np.array([0.0, -1.0, 0.0])                            # OpenCV-style map: y down
    R = sky_frame(up)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.allclose(R[2], up)
    m = SkyModel(R, res=16)
    with torch.no_grad():                                      # a texture that is red above the horizon, blue below
        m.tex[:] = -6
        m.tex[0, 0, :8] = 6
        m.tex[0, 2, 8:] = 6
    d = torch.tensor([[0.0, -1.0, 0.2], [0.0, 1.0, 0.2]])      # up-ish, down-ish
    c = m(torch.nn.functional.normalize(d, dim=-1))
    assert c[0, 0] > 0.9 and c[0, 2] < 0.1 and c[1, 2] > 0.9 and c[1, 0] < 0.1
    # pixel directions: the principal point looks along the camera's z axis
    K = torch.tensor([[[100.0, 0, 31.5], [0, 100.0, 23.5], [0, 0, 1]]])
    dirs = pixel_dirs(torch.eye(4)[None], K, 64, 48)
    assert dirs.shape == (1, 48, 64, 3) and torch.allclose(dirs[0, 23, 31], torch.tensor([0.0, 0, 1]), atol=0.01)
    # seen texels keep their colour, unseen ones get a smooth fill from them
    img = torch.zeros(3, 8, 16)
    img[1, :4] = 1.0
    w = torch.zeros(8, 16)
    w[:4] = 1
    f = push_pull(img, w)
    assert torch.allclose(f[:, :4], img[:, :4]) and float(f[1, 4:].min()) > 0.2
    # the splat shell: the splat of a texel sits in the direction that samples that texel
    st = {"tex": torch.sigmoid(m.tex[0]).detach(), "R": torch.from_numpy(R).float()}
    sp = sky_splats(st, np.zeros(3), 100.0, res=16, sh_coeffs=3)
    assert sp["means"].shape == (512, 3) and sp["shN"].shape == (512, 3, 3)
    assert torch.allclose(sp["means"].norm(dim=1), torch.full((512,), 100.0), atol=1e-3)
    col = m(torch.nn.functional.normalize(sp["means"], dim=-1))
    from cross_world.gaussians import sh_to_rgb
    assert float((col - sh_to_rgb(sp["sh0"][:, 0])).abs().max()) < 0.05


def test_capture_keeps_keyframes_and_frames_between_only_on_request():
    K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
    T0, T1 = _pose(0.0, (0, 0, 0)), _pose(0.0, (1, 0, 0))
    kf_only = WorldCapture(WorldCaptureConfig(enabled=True), K, 64, 48)
    assert kf_only.wants(T0, new_keyframe=True) and not kf_only.wants(T0) and not kf_only.wants(T1)
    between = WorldCapture(WorldCaptureConfig(enabled=True, between=True, min_translation=0.5), K, 64, 48)
    assert between.wants(T0) and between.wants(T1, new_keyframe=True)
    kf_only.cleanup()
    between.cleanup()
