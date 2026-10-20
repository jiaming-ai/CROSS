"""The folder formats a user's own data goes through (docs/GUIDE.md "Using your own data"): the posed RGB-D folder
(cross/dataloader/posed_rgbd.py) and the prepared stereo folder (cross/dataloader/stereo_loader.py), and run.py's
layout detection.  Synthetic folders of a few frames; no models."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cross.dataloader.posed_rgbd import PosedRGBDLoader          # noqa: E402
from cross.dataloader.stereo_loader import StereoSequenceLoader  # noqa: E402

W, H, N = 64, 48, 5


def _path(n: int, step: float = 0.1) -> np.ndarray:
    """Camera-to-world poses of a camera moving along its x axis (OpenCV axes), with a small turn per frame."""
    T = np.repeat(np.eye(4)[None], n, 0)
    for i in range(n):
        a = 0.02 * i
        T[i, :3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        T[i, 0, 3] = step * i
    return T


def _images(folder: Path, n: int, seed: int):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        cv2.imwrite(str(folder / f"{i:06d}.png"), rng.integers(0, 255, (H, W, 3), dtype=np.uint8))


def _calib(root: Path, **extra):
    K = [[50.0, 0.0, W / 2], [0.0, 50.0, H / 2], [0.0, 0.0, 1.0]]
    (root / "calib.json").write_text(json.dumps({"K": K, "width": W, "height": H, "fps": 5.0, **extra}))


def _posed_folder(root: Path, depth=True, odom=None) -> Path:
    _images(root / "rgb", N, 0)
    if depth:
        (root / "depth").mkdir()
        for i in range(N):
            cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), np.full((H, W), 1500 + i, np.uint16))   # millimetres
    _calib(root)
    np.savetxt(root / "poses_left.txt", _path(N).reshape(N, 16))
    if odom is not None:
        np.savetxt(root / "odom_left.txt", odom.reshape(N, 16))
    return root


def _stereo_folder(root: Path, baseline=0.12, right_dirs=None) -> Path:
    _images(root / "left", N, 1)
    T_rl = np.eye(4)
    T_rl[0, 3] = baseline
    for d in ["right"] + list((right_dirs or {}).values()):
        _images(root / d, N, 2)
    _calib(root, T_right_in_left=T_rl.tolist(), baseline=baseline, **({"right_dirs": right_dirs} if right_dirs else {}))
    np.savetxt(root / "poses_left.txt", _path(N).reshape(N, 16))
    return root


def _rel(a, b):
    return np.linalg.inv(a) @ b


def test_posed_folder_reads_images_depth_and_ground_truth_odometry(tmp_path):
    ds = PosedRGBDLoader(str(_posed_folder(tmp_path)))
    assert len(ds) == N and (ds.rgb_width, ds.rgb_height) == (W, H)
    np.testing.assert_allclose(ds.rgb_K[0, 0], 50.0)
    frames = list(ds.replay_data())
    assert len(frames) == N
    assert frames[0]["rgb"].shape == (H, W, 3) and frames[0]["rgb"].dtype == np.uint8
    np.testing.assert_allclose(frames[2]["depth"], 1.502, atol=1e-6)            # uint16 mm -> metres
    np.testing.assert_allclose(frames[0]["delta_pose"], np.eye(4))              # the first frame has no motion
    gt = _path(N)
    for i in range(1, N):                     # without odom_left.txt (and no snr): the ground truth's own steps
        np.testing.assert_allclose(frames[i]["delta_pose"], _rel(gt[i - 1], gt[i]), atol=1e-9)
        np.testing.assert_allclose(frames[i]["timestamp"], i / 5.0)


def test_posed_folder_prefers_the_robots_own_odometry(tmp_path):
    odom = _path(N, step=0.11)                  # the robot's odometry overestimates the distance by 10 %
    ds = PosedRGBDLoader(str(_posed_folder(tmp_path, odom=odom)), snr=10.0)
    frames = list(ds.replay_data())
    for i in range(1, N):                     # real odometry: used as is, no simulated noise
        np.testing.assert_allclose(frames[i]["delta_pose"], _rel(odom[i - 1], odom[i]), atol=1e-9)
        np.testing.assert_allclose(frames[i]["world_pose"], _path(N)[i])     # ground truth only for evaluation


def test_odom_file_selects_another_odometry_file(tmp_path):
    root = _posed_folder(tmp_path, odom=_path(N, step=0.11))
    vio = _path(N, step=0.09)
    np.savetxt(root / "odom_vio.txt", vio.reshape(N, 16))
    frames = list(PosedRGBDLoader(str(root), odom_file="odom_vio.txt").replay_data())
    np.testing.assert_allclose(frames[1]["delta_pose"], _rel(vio[0], vio[1]), atol=1e-9)
    # a missing file falls back to odom_left.txt
    odom = _path(N, step=0.11)
    frames = list(PosedRGBDLoader(str(root), odom_file="missing.txt").replay_data())
    np.testing.assert_allclose(frames[1]["delta_pose"], _rel(odom[0], odom[1]), atol=1e-9)


def test_posed_folder_with_stride_composes_the_skipped_steps(tmp_path):
    ds = PosedRGBDLoader(str(_posed_folder(tmp_path)))
    frames = list(ds.replay_data(stride=2))
    assert [f["frame_idx"] for f in frames] == [0, 2, 4]
    gt = _path(N)
    np.testing.assert_allclose(frames[1]["delta_pose"], _rel(gt[0], gt[2]), atol=1e-9)


def test_posed_folder_without_depth_gives_no_depth(tmp_path):
    ds = PosedRGBDLoader(str(_posed_folder(tmp_path, depth=False)))
    assert next(iter(ds.replay_data()))["depth"] is None


def test_stereo_folder_reads_the_pair_and_the_rig(tmp_path):
    ds = StereoSequenceLoader(str(_stereo_folder(tmp_path, baseline=0.12)))
    assert ds.dataset_type == "simchange" and len(ds) == N
    np.testing.assert_allclose(ds.baseline, 0.12)
    np.testing.assert_allclose(ds.T_right_in_left[0, 3], 0.12)
    f = list(ds.replay_data())[1]
    assert f["rgb"].shape == f["rgb_right"].shape == (H, W, 3)
    assert f["depth"] is None                                   # depth_source "none" by default
    gt = _path(N)
    np.testing.assert_allclose(f["delta_pose"], _rel(gt[0], gt[1]), atol=1e-9)


def test_stereo_folder_picks_a_rendered_baseline(tmp_path):
    root = _stereo_folder(tmp_path, baseline=0.12, right_dirs={"0.30": "right_0.30"})
    ds = StereoSequenceLoader(str(root), baseline=0.3)
    np.testing.assert_allclose(ds.baseline, 0.3)
    assert all(p.parent.name == "right_0.30" for p in ds.right_paths)
    with pytest.raises(ValueError, match="not rendered"):
        StereoSequenceLoader(str(root), baseline=0.5)


def test_stereo_loader_reads_a_posed_rgbd_folder_without_a_right_camera(tmp_path):
    ds = StereoSequenceLoader(str(_posed_folder(tmp_path)))
    assert ds.dataset_type == "posed_rgbd" and ds.T_right_in_left is None and ds.baseline == 0.0
    f = next(iter(ds.replay_data()))
    assert f["rgb_right"] is None


def test_unknown_layout_is_an_error(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(ValueError, match="Could not infer"):
        StereoSequenceLoader(str(tmp_path))


def test_layout_detection(tmp_path):
    from cross.dataloader.auto import detect_loader, load_dataset
    posed, stereo, empty = tmp_path / "posed", tmp_path / "stereo", tmp_path / "empty"
    _posed_folder(posed), _stereo_folder(stereo), empty.mkdir()
    assert detect_loader(str(posed)) == "posed" and isinstance(load_dataset(str(posed)), PosedRGBDLoader)
    assert detect_loader(str(stereo)) == "stereo" and isinstance(load_dataset(str(stereo)), StereoSequenceLoader)
    with pytest.raises(ValueError, match="Could not detect"):
        detect_loader(str(empty))
    with pytest.raises(ValueError, match="Unknown loader"):
        load_dataset(str(posed), loader="nope")
