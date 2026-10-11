import torch
import torch.nn.functional as F

from cross_nav.footprint_loss import footprint_bce
from cross_nav.plan import inflate


def test_zero_radius_matches_known_cell_bce():
    x = torch.randn(2, 7, 9, requires_grad=True)
    y = torch.randint(0, 3, x.shape)
    mask = y != 0
    expected = ((F.binary_cross_entropy_with_logits(x, (y == 2).float(), reduction="none") * mask)
                .sum((1, 2)) / mask.sum((1, 2))).mean()
    actual, _ = footprint_bce(x, y, 0)
    torch.testing.assert_close(actual, expected)


def test_disk_target_matches_planner_and_excludes_boundary():
    y = torch.ones(1, 11, 11, dtype=torch.long)
    y[0, 5, 5] = 2
    x = torch.full(y.shape, -2.0, requires_grad=True)
    got, fraction = footprint_bce(x, y, 2)
    target = torch.from_numpy(inflate((y[0] == 2).numpy(), 2)).float()
    expected = F.binary_cross_entropy_with_logits(x[0, 2:-2, 2:-2], target[2:-2, 2:-2])
    torch.testing.assert_close(got, expected)
    assert abs(float(fraction) - 49 / 121) < 1e-6


def test_unknown_has_no_gradient_and_empty_labels_are_finite():
    x = torch.randn(1, 9, 9, requires_grad=True)
    y = torch.ones_like(x, dtype=torch.long)
    y[0, 4, 4] = 0
    loss, _ = footprint_bce(x, y, 2)
    loss.backward()
    assert x.grad[0, 4, 4] == 0
    x.grad = None
    loss, fraction = footprint_bce(x, torch.zeros_like(y), 2)
    loss.backward()
    assert loss == 0 and fraction == 0 and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) == 0


def test_missed_obstacle_receives_corrective_gradient():
    x = torch.full((1, 9, 9), -3.0, requires_grad=True)
    y = torch.ones_like(x, dtype=torch.long)
    y[0, 4, 4] = 2
    loss, _ = footprint_bce(x, y, 1)
    loss.backward()
    assert x.grad[0, 4, 4] < 0
