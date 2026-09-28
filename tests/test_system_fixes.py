"""Unit tests of generic system fixes (odometry drift model, VPR buffer growth on map load, uint8 storage, relocalization evidence)."""
import numpy as np
import pypose as pp
import torch

from cross.dataloader.posed_rgbd import PosedRGBDLoader as StereoSequenceLoader


def _pose(x, y, z):
    T = pp.identity_SE3(1)
    T.tensor()[0, :3] = torch.tensor([x, y, z], dtype=T.dtype)
    return T


def test_odometry_drift_model():
    """Scale bias scales the step translation; heading drift rotates about the world vertical by rate * |t|."""
    ld = StereoSequenceLoader.__new__(StereoSequenceLoader)
    ld.odom_scale_bias, ld.odom_yaw_drift, ld.odom_vertical_world = 0.02, 1.0, np.array([0.0, 0.0, 1.0])
    delta = np.eye(4); delta[:3, 3] = [0.5, 0.0, 0.0]
    out = ld._bias_delta(delta, np.eye(4))
    assert np.isclose(np.linalg.norm(out[:3, 3]), 0.51)
    ang = np.degrees(np.arccos(np.clip((np.trace(out[:3, :3]) - 1) / 2, -1, 1)))
    assert np.isclose(ang, 0.5, atol=1e-6)                   # 1 deg per metre over 0.5 m
    axis = np.array([out[2, 1] - out[1, 2], out[0, 2] - out[2, 0], out[1, 0] - out[0, 1]])
    assert abs(axis[2]) > 0.99 * np.linalg.norm(axis)         # about the vertical


def test_vpr_buffer_grows_on_map_load():
    """load_state sets the size to the stored count before growing the buffer: the copy must use existing rows only."""
    from cross.db.db import KeyframeDatabase
    db = KeyframeDatabase.__new__(KeyframeDatabase)
    db.device = "cpu"
    db._embedding_buffer = torch.ones(10, 4)
    db._current_size = 25                                     # as set by load_state for a 25-keyframe map
    db._extend_buffer(25)
    assert db._embedding_buffer.shape[0] >= 25 and torch.all(db._embedding_buffer[:10] == 1)


def test_uint8_keyframe_storage_roundtrip():
    from cross.db.db import to_uint8_image, as_float_image
    x = torch.rand(3, 8, 8)
    y = as_float_image(to_uint8_image(x))
    assert y.dtype == torch.float32 and float((y - x).abs().max()) <= 0.5 / 255 + 1e-6
    d = torch.rand(1, 8, 8) * 5
    assert as_float_image(d.half()).dtype == torch.float32


def _reloc_manager(unique: bool):
    """Relocalization session (map loaded, hypothesis 0 not anchored) with three places 10 m apart."""
    import types
    from cross.core.config import HypothesisConfig
    from cross.core.hypothesis import HypothesisManager
    cfg = HypothesisConfig(unmatched_evidence="miss", reloc_unique_evidence=unique)
    sys_ = types.SimpleNamespace(_lc_verifier=None, topo_map=None, _session_start_kf_id=100)
    hm = HypothesisManager(sys_, n_components=3, config=cfg); sys_.hypothesis_manager = hm
    d = hm.device
    mu = pp.SE3(torch.tensor([[0., 0, 0, 0, 0, 0, 1], [10., 0, 0, 0, 0, 0, 1], [20., 0, 0, 0, 0, 0, 1]], device=d))
    std = pp.se3(torch.full((3, 6), 0.2, device=d))
    hm.dist = (mu, std, torch.tensor([0.1, 0.45, 0.45], device=d))
    hm.ttl[:] = 10
    hm.realized[:] = True
    return hm, mu, std



def test_reloc_unique_evidence_counts_only_unrivalled_support():
    """Against a lost hypothesis 0 every supported place gains evidence; against the best other place, two places
    supported by the same observation gain none, and a place supported alone gains the miss margin."""
    for supported, expect_c, expect_u in (((1, 2), [0.0, 2.0, 2.0], [-2.0, 0.0, 0.0]),
                                         ((1,), [0.0, 2.0, 0.0], [-2.0, 2.0, -2.0])):
        hm, mu, std = _reloc_manager(unique=True)
        conf = torch.zeros(3, device=hm.device)
        conf[list(supported)] = 1.0
        hm.gmm_filtering(mu, std, torch.ones(3, device=hm.device) / 3, conf)
        p = (hm.llr_hist_ptr - 1) % hm.llr_hist_length
        np.testing.assert_allclose(hm.log_c_hist[:, p].cpu().numpy(), expect_c, atol=1e-4)
        np.testing.assert_allclose(hm.log_u_hist[:, p].cpu().numpy(), expect_u, atol=1e-4)



def test_relocalization_options_default_off():
    from cross.core.config import HypothesisConfig, LoopClosureConfig
    lc = LoopClosureConfig()
    assert lc.anchor_corroborate_window == 0 and lc.anchor_contradict_min == 0
    cfg = HypothesisConfig()
    assert not cfg.reloc_unique_evidence and cfg.reloc_min_frames == 0
