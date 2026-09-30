import numpy as np
import pytest

from cross.mono.config import MonoConfig


def test_streaming_rotation_requires_explicit_frontend_and_checkpoint():
    with pytest.raises(ValueError,match='streaming_pnp'):
        MonoConfig(rotation_tracker='dpvo',dpvo_checkpoint='weights.pth')
    with pytest.raises(ValueError,match='checkpoint'):
        MonoConfig(frontend='streaming_pnp',rotation_tracker='dpvo')
    config=MonoConfig(frontend='streaming_pnp',rotation_tracker='dpvo',dpvo_checkpoint='weights.pth',
                      conditional_sources=True,chart_aware=True,session_recovery=True,retrieval_pose='metric_pnp')
    assert config.trace_metric_sources and config.conditional_sources


def test_shared_patch_mask_uses_the_current_padded_boxes_once():
    torch=pytest.importorskip('torch')
    from cross.mono.background_patches import BackgroundPatchifier,background_indices
    patchifier=BackgroundPatchifier(torch.nn.Identity(),device='cpu',external_masks=True)
    boxes=np.array([[10.,20.,30.,40.]],np.float32)
    patchifier.observe(np.zeros((48,64,3),np.uint8),boxes)
    np.testing.assert_array_equal(patchifier.current_boxes.numpy(),boxes)
    assert patchifier.detector is None
    selected,count=background_indices(torch.tensor([[9.,19.],[15.,25.],[31.,41.]]),patchifier.current_boxes,2)
    assert selected.tolist()==[0,2] and count==2
    with pytest.raises(ValueError,match='current frame'):
        patchifier.observe(np.zeros((48,64,3),np.uint8))
    patchifier.observe(np.zeros((48,64,3),np.uint8),np.empty((0,4),np.float32))
    assert patchifier.current_boxes.shape==(0,4)
