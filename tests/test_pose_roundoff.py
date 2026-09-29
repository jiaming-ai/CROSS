"""Long float32 motion chains must remain on SE3 in the scalar mapper."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

torch = pytest.importorskip('torch')
pp = pytest.importorskip('pypose')

from cross.core.hypothesis import HypothesisManager
from cross.core.odom_accum import OdomAccumulator
from cross.utils.lie_tensor import normalize_se3


def motion():
    value = pp.se3(torch.tensor([.002, -.001, .003, .004, -.006, .002])).Exp()
    # A one-ulp norm error is an ordinary float32 input, not a bad rotation.
    data = value.tensor().clone()
    data[3:] *= 1 + torch.finfo(torch.float32).eps
    return pp.SE3(data)


def assert_group(pose):
    q = pose.tensor()[..., 3:]
    assert float((torch.linalg.vector_norm(q, dim=-1) - 1).abs().max()) < 2e-7
    matrix = pose.matrix().double().numpy()
    np.testing.assert_allclose(matrix[..., :3, :3].swapaxes(-1, -2) @ matrix[..., :3, :3],
                               np.broadcast_to(np.eye(3), matrix[..., :3, :3].shape), atol=1e-6)
    np.testing.assert_allclose((pose.Inv() @ pose).matrix().double().numpy(),
                               np.broadcast_to(np.eye(4), matrix.shape), atol=2e-6)


def test_scalar_accumulation_and_consumers_remain_rigid_over_long_chain():
    accum = OdomAccumulator(device='cpu')
    accum.register_item('frame')
    accum.register_item('keyframe')
    hm = HypothesisManager(SimpleNamespace(device='cpu', topo_map=None), 2)
    hm.dist = (pp.identity_SE3(2), pp.se3(torch.zeros(2, 6)), torch.tensor([1., 0.]))
    delta = motion()
    expected = np.eye(4)
    exact_delta = np.eye(4)
    exact_delta[:3, :3] = Rotation.from_quat(delta.tensor()[3:].double().numpy()).as_matrix()
    exact_delta[:3, 3] = delta.tensor()[:3].double().numpy()
    for i in range(2048):
        accum.update_odom(delta)
        step, std = accum.get_since_last_reading('frame')
        hm.motion_update(step, std)
        expected = expected @ exact_delta
        if i % 31 == 0:
            segment, _ = accum.get_since_last_reading('keyframe')
            assert_group(segment)
    assert_group(accum._accumulated_odom)
    assert_group(hm.dist[0])
    np.testing.assert_allclose(accum._accumulated_odom.matrix().double().numpy(), expected, atol=6e-5)
    np.testing.assert_allclose(hm.dist[0][0].matrix().double().numpy(), expected, atol=6e-5)
    # Inactive slots and the inherited weight policy are untouched.
    torch.testing.assert_close(hm.dist[0][1], pp.identity_SE3(), rtol=0, atol=0)
    torch.testing.assert_close(hm.dist[2], torch.tensor([1., 0.]), rtol=0, atol=0)


def test_scalar_filter_normalizes_roundoff_in_stored_mean():
    hm = HypothesisManager(SimpleNamespace(device='cpu', topo_map=None), 1)
    data = motion().tensor().unsqueeze(0)
    data[:, 3:] *= 1 + 5e-5
    hm.dist = (pp.SE3(data), pp.se3(torch.full((1, 6), .1)), torch.ones(1))
    for _ in range(12):
        hm.gmm_filtering(pp.identity_SE3(1), pp.se3(torch.full((1, 6), .1)),
                         torch.ones(1), torch.ones(1))
        assert_group(hm.dist[0])


@pytest.mark.parametrize('kind', ['zero', 'nonunit', 'nan', 'infinite_position'])
def test_invalid_poses_are_rejected_without_mutating_accumulator(kind):
    accum = OdomAccumulator(device='cpu')
    data = pp.identity_SE3().tensor().clone()
    if kind == 'zero':
        data[3:] = 0
    elif kind == 'nonunit':
        data[6] = 1.01
    elif kind == 'nan':
        data[3] = float('nan')
    else:
        data[0] = float('inf')
    with pytest.raises(ValueError, match='non-unit or non-finite'):
        accum.update_odom(pp.SE3(data))
    torch.testing.assert_close(accum._accumulated_odom, pp.identity_SE3(), rtol=0, atol=0)


def test_normalization_preserves_translation_and_does_not_mutate_input():
    pose = motion()
    before = pose.tensor().clone()
    actual = normalize_se3(pose)
    torch.testing.assert_close(actual.tensor()[:3], before[:3], rtol=0, atol=0)
    torch.testing.assert_close(pose.tensor(), before, rtol=0, atol=0)
    assert_group(actual)
