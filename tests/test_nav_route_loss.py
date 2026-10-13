import numpy as np
import torch

from cross_nav.route_loss import critical_obstacles, route_bce


def test_mines_only_actual_obstacles_on_prediction_only_route():
    gt = np.ones((9, 9), np.uint8)
    gt[4, 6] = 2
    mask, n = critical_obstacles(np.zeros((9, 9)), gt, res=1., radius=0., goals=[(4, 7)])
    assert n == 1
    assert np.array_equal(np.argwhere(mask), [[4, 6]])


def test_unknown_route_cells_are_not_obstacle_targets():
    gt = np.ones((9, 9), np.uint8)
    gt[4, 6] = 0
    mask, n = critical_obstacles(np.zeros((9, 9)), gt, res=1., radius=0., goals=[(4, 7)])
    assert n == 1  # A known-free alternative goes around the unknown cell.
    assert not mask.any()


def test_unplanable_start_returns_no_targets_without_snapping():
    gt = np.ones((9, 9), np.uint8)
    gt[4, 4] = 2
    mask, n = critical_obstacles(np.zeros((9, 9)), gt, res=1., radius=0.)
    assert n == 0 and not mask.any()


def test_loss_gradient_only_updates_mined_occupied_cells():
    gt = torch.ones((1, 9, 9), dtype=torch.uint8)
    gt[0, 4, 6] = 2
    logits = torch.full((1, 9, 9), -5., requires_grad=True)
    loss, parts = route_bce(logits, gt, res=1., radius=0., max_goals=81)
    loss.backward()
    assert parts['route_pixels'] == 1
    assert logits.grad[0, 4, 6] < 0
    assert torch.count_nonzero(logits.grad).item() == 1


def test_empty_mining_has_finite_zero_loss_and_gradient():
    gt = torch.zeros((1, 9, 9), dtype=torch.uint8)
    logits = torch.zeros((1, 9, 9), requires_grad=True)
    loss, parts = route_bce(logits, gt, res=1., radius=0.)
    loss.backward()
    assert loss.item() == 0 and parts['route_pixels'] == 0
    assert torch.count_nonzero(logits.grad).item() == 0
