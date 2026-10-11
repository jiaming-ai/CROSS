import pytest
import torch

from cross_nav.uncertainty import attenuated_bce
from cross_nav.train import bev_losses


def test_attenuation_branch_backpropagates_through_decoder():
    from cross_nav.head import DirectBEVHead
    head = DirectBEVHead(size=16, depth=1, dim_agg=32, dim_dino=16, feat=16, dim=32, heads=4,
                         uncertainty=True, uncertainty_mode="attenuated")
    torch.nn.init.normal_(head.error_head[-1].weight, std=.1)
    tl = [None] * 24
    for layer in (4, 11, 17, 23):
        tl[layer] = torch.randn(1, 1, 21, 32)
    out = head(tl, 17, (2, 2), torch.randn(1, 1, 4, 16), None, None, None, None,
               torch.zeros(1), torch.tensor([[0., -1., 0.]]), return_aux=True)
    out["error_logits"].sum().backward()
    assert head.up3.net[0].weight.grad.abs().sum() > 0
    assert head.out.weight.grad is None


def test_attenuation_reduces_occupancy_gradient_and_masks_unknown():
    z = torch.tensor([[[0., 0., 0.]]], requires_grad=True)
    raw = torch.tensor([[[-3., 3., 0.]]], requires_grad=True)
    labels = torch.tensor([[[2, 2, 0]]])
    loss, _ = attenuated_bce(z, raw, labels)
    loss.backward()
    assert abs(z.grad[0, 0, 1]) < abs(z.grad[0, 0, 0]) / 10
    assert z.grad[0, 0, 2] == 0 and raw.grad[0, 0, 2] == 0


def test_regularizer_prevents_all_uncertain_and_loss_is_nonnegative():
    labels = torch.ones(1, 2, 2, dtype=torch.long)
    z = torch.full((1, 2, 2), -10.)
    raw = torch.full_like(z, 20., requires_grad=True)
    loss, _ = attenuated_bce(z, raw, labels, beta=.3)
    loss.backward()
    assert loss > 5 and (raw.grad > 0).all()
    small, _ = attenuated_bce(z, torch.full_like(z, -30.), labels)
    assert small >= 0
    with pytest.raises(ValueError):
        attenuated_bce(z, raw, labels, beta=0)


def test_dice_gradient_survives_high_uncertainty():
    labels = torch.full((1, 2, 2), 2)
    z = torch.zeros(1, 2, 2, requires_grad=True)
    raw = torch.full_like(z, 30., requires_grad=True)
    loss, _ = bev_losses(z, labels, w_dice=.5, raw_uncertainty=raw)
    loss.backward()
    assert (z.grad < -.01).all()


def test_hard_cells_increase_uncertainty_easy_cells_reduce_it():
    z = torch.tensor([[[-5., 5.]]])
    raw = torch.zeros_like(z, requires_grad=True)
    loss, _ = attenuated_bce(z, raw, torch.full_like(z, 2), beta=.3)
    loss.backward()
    assert raw.grad[0, 0, 0] < 0 and raw.grad[0, 0, 1] > 0
