import numpy as np
import pytest
import torch

from cross_nav.plan import Field, evaluate_planning, risk_cost, summarize
from cross_nav.uncertainty import error_loss


def test_required_dataset_cannot_silently_disappear(tmp_path):
    from cross_nav.data import NavStream
    stream = NavStream({"root": str(tmp_path), "bev_root": str(tmp_path),
                        "datasets": {"missing": {"weight": 1}}, "require_all_datasets": True})
    with pytest.raises(RuntimeError, match="required dataset missing unavailable"):
        stream._build()


def test_uncertainty_head_preserves_occupancy_and_detaches_features():
    from cross_nav.head import DirectBEVHead
    cfg = dict(size=16, depth=1, dim_agg=32, dim_dino=16, feat=16, dim=32, heads=4)
    old = DirectBEVHead(**cfg)
    head = DirectBEVHead(**cfg, uncertainty=True)
    missing, unexpected = head.load_state_dict(old.state_dict(), strict=False)
    assert missing and all(k.startswith("error_head.") for k in missing) and not unexpected
    tl = [None] * 24
    for layer in (4, 11, 17, 23):
        tl[layer] = torch.randn(1, 1, 21, 32)
    args = (tl, 17, (2, 2), torch.randn(1, 1, 4, 16), None, None, None, None,
            torch.zeros(1), torch.tensor([[0., -1., 0.]]))
    aux = head(*args, return_aux=True)
    assert torch.equal(old(*args), aux["logits"])
    assert aux["error_logits"].shape == (1, 16, 16)
    error_loss(aux["error_logits"], aux["logits"], torch.ones(1, 16, 16)).backward()
    assert head.out.weight.grad is None
    assert head.up3.net[0].weight.grad is None
    assert head.error_head[-1].bias.grad is not None


def test_error_targets_mask_unknown_and_detach_occupancy():
    logits = torch.tensor([[[2., -2., 2., -2.]]], requires_grad=True)
    labels = torch.tensor([[[2, 1, 1, 0]]])
    error = torch.zeros_like(logits, requires_grad=True)
    error_loss(error, logits, labels).backward()
    assert logits.grad is None
    assert error.grad[0, 0, 0] > 0 and error.grad[0, 0, 1] > 0
    assert error.grad[0, 0, 2] < 0 and error.grad[0, 0, 3] == 0


def test_soft_cost_recovers_blurry_door():
    gt = np.ones((80, 80), np.uint8)
    gt[:, 38:42] = 2
    gt[28:42, 38:42] = 1
    p = np.where(gt == 2, .99, .01).astype(np.float32)
    p[28:42, 38:42] = .55
    maps = {"hard": p > .5, "soft": np.zeros_like(gt, bool)}
    rec = evaluate_planning(gt, maps, .05, (60, 20), n_goals=40, dist_range=(1., 4.), seed=1,
                            costs={"soft": risk_cost(p, .05)})
    assert summarize(rec, "soft")["success"] > summarize(rec, "hard")["success"]


def test_weighted_path_avoids_risk_and_unit_cost_matches():
    ok = np.ones((15, 15), bool)
    p = np.zeros_like(ok, dtype=float)
    p[7, 3:12] = .9
    start, goal = (7, 0), (7, 14)
    unit = Field(ok, start).path(goal)
    assert unit == Field(ok, start, np.ones_like(p)).path(goal)
    path = Field(ok, start, risk_cost(p, .05, robot_radius=0)).path(goal)
    assert not any(p[r, c] > 0 for r, c in path)
    assert np.isfinite(risk_cost(np.ones_like(p), .05)).all()


def test_unknown_ground_truth_is_not_counted_as_verified_success():
    gt = np.ones((40, 40), np.uint8)
    gt[10:, 19:21] = 0
    maps = {"blind": np.zeros_like(gt, bool), "known": gt == 0}
    recs = evaluate_planning(gt, maps, .1, (30, 8), n_goals=60,
                             dist_range=(1., 4.), robot_radius=0, check_radius=0, seed=2)
    blind, known = summarize(recs, "blind"), summarize(recs, "known")
    assert blind["unknown_path_rate"] > 0
    assert blind["known_success"] < blind["success"]
    assert known["unknown_path_rate"] == 0
    assert known["known_success"] == known["success"]
