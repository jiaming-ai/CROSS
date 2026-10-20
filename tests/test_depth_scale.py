"""RGB-D mode with the feed-forward estimator: the metric scale of a pass from measured depth (CPU only)."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cross.core.config import FeedForwardConfig, PoseEstType, load_config
from cross.cv.depth_scale import depth_scale, view_log_ratios
from cross.cv.pose_est_ff import FFPrediction, PoseEstFeedForward


def _scene(h=48, w=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 1.0 + 6.0 * torch.rand(h, w, generator=g)          # metric depth 1-7 m


def test_depth_scale_recovers_the_gauge_scale():
    true = _scene()
    pred = true / 2.5                                          # model gauge: 2.5x smaller than metric
    meas = true.clone()
    meas[:, :10] = 0.0                                         # missing depth (sparse LiDAR, holes)
    meas[:5, 10:] = 50.0                                       # beyond max_depth: ignored
    meas[20:24, 20:40] *= 3.0                                  # wrong depth on a few pixels: the median ignores it
    est = depth_scale([pred], [meas], [torch.ones_like(pred)], max_depth=10.0, min_pixels=100)
    assert est.valid and est.method == "depth" and est.n_used == 1
    assert abs(est.scale - 2.5) < 1e-4
    assert est.log_std >= 0.02                                 # floor


def test_depth_scale_needs_enough_trusted_pixels():
    pred = _scene()
    meas = torch.zeros_like(pred)
    meas[0, :50] = 3.0
    assert not depth_scale([pred], [meas], min_pixels=200).valid
    assert depth_scale([pred], [meas], min_pixels=20).valid
    assert not depth_scale([pred], [None]).valid               # no measured depth at all


def test_confidence_quantile_keeps_the_confident_half():
    true = _scene()
    pred = true / 2.0
    conf = torch.ones_like(pred)
    conf[:, :32] = 0.1                                         # unconfident half predicts the wrong depth
    pred[:, :32] = true[:, :32]
    r = view_log_ratios(pred, true, conf, conf_quantile=0.5)
    assert torch.allclose(r, torch.full_like(r, float(np.log(2.0))), atol=1e-5)


def test_depth_scale_over_several_views_and_resized_depth():
    true = _scene(seed=1)
    views = [true / 2.0, _scene(seed=2) / 2.0, _scene(seed=3) / 2.0]
    meas = [true[None], None, _scene(seed=3)[::2, ::2].repeat_interleave(2, 0).repeat_interleave(2, 1)]
    meas[2] = meas[2][::2, ::2]                                # a coarser depth map: resized to the prediction's grid
    est = depth_scale(views, meas, min_pixels=50)
    assert est.valid and est.n_anchors == 2 and est.n_used == 2
    assert abs(est.scale - 2.0) < 0.05


def _estimator(monkeypatch, **cfg):
    est = PoseEstFeedForward.__new__(PoseEstFeedForward)
    est.device = "cpu"
    est.config = FeedForwardConfig(covis_source="head", min_covis=0.0, n_ref_anchors=0, use_curr_anchor=False, **cfg)
    est.backend = SimpleNamespace(covis_head=object())
    est._R_model_to_cam = None
    est.T_right_in_left = None
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    return est


def test_estimator_scales_the_pass_by_the_measured_depth(monkeypatch):
    """Current camera at the origin, one reference 1 m to its right.  The model predicts poses and depth at 0.4x the
    metric scale; the current frame's measured depth makes the relative pose metric."""
    est = _estimator(monkeypatch, depth_scale=True)
    X_ref = np.eye(4)
    X_ref[0, 3] = 1.0
    c2w = np.stack([np.eye(4), X_ref])
    c2w[:, :3, 3] *= 0.4
    true = _scene(384, 512)
    calls = []

    def infer(images, n_depth=None, focal=None):
        calls.append(n_depth)
        n = images.shape[0]
        k = n if n_depth is None else n_depth
        return FFPrediction(c2w=c2w[:n], K=np.stack([np.eye(3)] * n), depth=(true * 0.4).expand(k, -1, -1),
                            depth_conf=torch.ones(k, 384, 512), hw=(384, 512), covis=np.ones((n, n)))
    est.backend.infer = infer
    img = torch.rand(3, 384, 512)
    poses, valid, _ = est.estimate_pose(img[None], None, img, true[None])
    assert valid.tolist() == [True] and calls[-1] == 1         # the depth head runs for the current view only
    assert est.last_scale.method == "depth" and abs(est.last_scale.scale - 2.5) < 1e-4
    T = poses.matrix()[0].double().numpy()
    assert np.allclose(T, np.linalg.inv(X_ref), atol=1e-4)
    # references' stored depth too ("all"); a reference without depth is skipped
    est.config.depth_scale_views = "all"
    poses, valid, _ = est.estimate_pose(img[None], [None], img, true[None])
    assert valid.tolist() == [True] and calls[-1] == 1 and est.last_scale.n_used == 1
    poses, valid, _ = est.estimate_pose(img[None], true[None, None], img, true[None])
    assert calls[-1] == 2 and est.last_scale.n_used == 2
    # without depth (or with the option off) the pass has no scale anchor: no observation
    poses, valid, _ = est.estimate_pose(img[None], None, img, None)
    assert not valid.any()
    est.config.depth_scale = False
    poses, valid, _ = est.estimate_pose(img[None], None, img, true[None])
    assert not valid.any() and calls[-1] == 0


def test_rgbd_ff_mode_files_and_arguments():
    import argparse

    from cross.pipeline import CONFIG_DIR, add_session_args, finish_session_args, mode_file_name, shipped_config_files
    for odo in ("wheel", "basalt"):
        cfg = load_config(os.path.join(CONFIG_DIR, "modes", f"rgbd_ff_{odo}.yaml"))
        ff = cfg.pose_est.ff
        assert cfg.pose_est.type == PoseEstType.FF and ff.depth_scale and ff.depth_scale_views == "current"
        assert not ff.use_curr_anchor and ff.n_ref_anchors == 0 and not ff.use_odom_anchor and not ff.use_map_anchors
        assert cfg.pose_est.obs_min_translation == 0.3 and ff.max_refs == 6
    # the PnP RGB-D mode and the stereo mode are unchanged by the option
    assert load_config(os.path.join(CONFIG_DIR, "modes", "rgbd_wheel.yaml")).pose_est.type == PoseEstType.PNP
    assert not load_config(os.path.join(CONFIG_DIR, "modes", "stereo_wheel.yaml")).pose_est.ff.depth_scale
    ap = argparse.ArgumentParser()
    ap.add_argument("--odom-file", default=None)
    add_session_args(ap)
    a = finish_session_args(ap.parse_args(["--mode", "rgbd_ff_wheel"]))
    assert (a.mode, a.odometry, a.rgbd_estimator) == ("rgbd", "external", "ff")
    assert shipped_config_files(a)[0].endswith("modes/rgbd_ff_wheel.yaml")
    a = finish_session_args(ap.parse_args(["--mode", "rgbd", "--rgbd-estimator", "ff", "--odometry", "basalt"]))
    assert shipped_config_files(a)[0].endswith("modes/rgbd_ff_basalt.yaml")
    a = finish_session_args(ap.parse_args(["--mode", "rgbd_wheel"]))
    assert a.rgbd_estimator == "pnp" and shipped_config_files(a)[0].endswith("modes/rgbd_wheel.yaml")
    with pytest.raises(ValueError):
        finish_session_args(ap.parse_args(["--mode", "rgbd_ff_wheel", "--rgbd-estimator", "pnp"]))
    with pytest.raises(ValueError):
        mode_file_name("rgbd", "visual", rgbd_estimator="ff")


def test_map_and_reloc_picks_the_estimator_and_depth():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
    import map_and_reloc as mr
    assert mr.estimator_of("rgbd") == "pnp" and mr.estimator_of("rgbd", "ff") == "ff" and mr.estimator_of("stereo") == "ff"
    args = SimpleNamespace(mode="rgbd", estimator="ff", pnp_depth="gt", odometry="external", depth_source="none")
    assert mr.depth_source(args) == "gt"
