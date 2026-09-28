"""Unit tests of generic system fixes (projection axis, odometry drift model, VPR buffer growth on map load)."""
import numpy as np
import pypose as pp
import torch

from cross.utils import lie_tensor
from cross.dataloader.posed_rgbd import PosedRGBDLoader as StereoSequenceLoader


def _pose(x, y, z):
    T = pp.identity_SE3(1)
    T.tensor()[0, :3] = torch.tensor([x, y, z], dtype=T.dtype)
    return T


def test_projection_vertical_axis_down_looking():
    """With vertical_axis=2 (down-looking camera), a displacement along y separates poses; one along z (altitude)
    does not.  The legacy projection (x, z) does the opposite."""
    a, b, c = _pose(0, 0, 0), _pose(0, 3, 0), _pose(0, 0, 3)
    try:
        lie_tensor.set_projection_vertical_axis(2)
        pa, pb, pc = (lie_tensor.project_SE3(p) for p in (a, b, c))
        assert torch.norm(pa - pb) > 2.9 and torch.norm(pa - pc) < 1e-6
        lie_tensor.set_projection_vertical_axis(-1)
        pa, pb, pc = (lie_tensor.project_SE3(p) for p in (a, b, c))
        assert torch.norm(pa - pb) < 1e-6 and torch.norm(pa - pc) > 2.9
    finally:
        lie_tensor.set_projection_vertical_axis(-1)


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


def test_auv_noise_keeps_roll_pitch_and_depth():
    """AUV odometry mode: the noisy step has the same gravity direction and the same depth change as the true step."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    ld = StereoSequenceLoader.__new__(StereoSequenceLoader)
    ld.odom_noise_mode, ld.snr, ld.noise_rng = "auv", 2.0, rng
    ld.odom_vertical_world = np.array([0.0, 0.0, 1.0])
    for _ in range(20):
        c2w_start = np.eye(4); c2w_start[:3, :3] = Rotation.random(random_state=rng.integers(1e6)).as_matrix()
        delta = np.eye(4); delta[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.3, 3)).as_matrix(); delta[:3, 3] = rng.normal(0, 0.5, 3)
        c2w_end = c2w_start @ delta
        noisy = ld._noise_delta(delta, c2w_end)
        end_noisy = c2w_start @ noisy
        v = ld.odom_vertical_world
        assert np.allclose(end_noisy[:3, :3].T @ v, c2w_end[:3, :3].T @ v, atol=1e-6)     # roll / pitch unchanged
        assert np.isclose(end_noisy[:3, 3] @ v, c2w_end[:3, 3] @ v, atol=1e-6)          # depth unchanged


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
