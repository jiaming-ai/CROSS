import numpy as np
import pytest

from cross_nav.observed_plan import ObservedAudit


def test_unseen_obstacle_is_not_scored_as_collision_after_reobserve_boundary():
    gt = np.ones((5, 7), np.uint8); gt[2, 5] = 2
    seen = np.zeros_like(gt, bool); seen[:, :4] = True
    audit = ObservedAudit(gt, seen, (2, 1), 1., radius=0.)
    assert audit.opportunity((2, 6))
    assert audit.score([(2, c) for c in range(1, 7)], (2, 6))["status"] == "safe_to_reobserve"


def test_visible_obstacle_remains_failure():
    gt = np.ones((5, 7), np.uint8); gt[2, 3] = 2
    audit = ObservedAudit(gt, np.ones_like(gt, bool), (2, 1), 1., radius=0.)
    assert audit.score([(2, c) for c in range(1, 7)], (2, 6))["status"] == "visible_collision"


def test_zero_length_safe_prefix_cannot_be_success():
    gt = np.ones((5, 7), np.uint8); seen = np.zeros_like(gt, bool)
    audit = ObservedAudit(gt, seen, (2, 1), 1., radius=0.)
    assert not audit.opportunity((2, 6))
    assert audit.score([(2, c) for c in range(1, 7)], (2, 6))["status"] == "no_progress"


def test_seen_diagonal_side_collision_is_detected():
    gt = np.ones((5, 5), np.uint8); gt[1, 2] = 2
    audit = ObservedAudit(gt, np.ones_like(gt, bool), (1, 1), 1., radius=0.)
    assert audit.score([(1, 1), (2, 2)], (2, 2))["status"] == "visible_collision"


def test_seen_goal_can_be_reached_and_no_path_still_reported():
    gt = np.ones((5, 5), np.uint8)
    audit = ObservedAudit(gt, np.ones_like(gt, bool), (2, 1), 1., radius=0.)
    assert audit.score([(2, 1), (2, 2), (2, 3)], (2, 3))["status"] == "goal_reached"
    assert audit.score(None, (2, 3))["status"] == "no_path"
    with pytest.raises(ValueError, match="unchanged start"):
        audit.score([(2, 2), (2, 3)], (2, 3))


def test_full_footprint_stops_before_unknown_side_cells():
    gt = np.ones((9, 9), np.uint8)
    seen = np.zeros_like(gt, bool); seen[3:6, 1:7] = True
    audit = ObservedAudit(gt, seen, (4, 2), 1., radius=1.)
    result = audit.score([(4, c) for c in range(2, 8)], (4, 7))
    assert result["status"] == "safe_to_reobserve"
    assert result["last_observed_cell"] == [4, 5]


def test_a_safe_detour_with_no_goal_progress_is_not_success():
    gt = np.ones((7, 7), np.uint8)
    seen = np.zeros_like(gt, bool); seen[1:5, :3] = True
    audit = ObservedAudit(gt, seen, (3, 1), 1., radius=0.)
    result = audit.score([(3, 1), (2, 1), (1, 1), (0, 1)], (3, 6))
    assert result["travel"] == 2.
    assert result["status"] == "no_progress"


def test_explicit_reobservation_stop_requires_safe_progress():
    gt = np.ones((7, 9), np.uint8)
    audit = ObservedAudit(gt, np.ones_like(gt, bool), (3, 1), 1., radius=0.)
    path = [(3, 1), (3, 2), (3, 3)]
    assert audit.score(path, (3, 7))['status'] == 'incomplete_path'
    assert audit.score(path, (3, 7), stop_for_reobserve=True)['status'] == 'safe_to_reobserve'
    assert audit.score(path[:1], (3, 7), stop_for_reobserve=True)['status'] == 'incomplete_path'
    assert audit.score([(3, 1), (2, 1)], (3, 7), stop_for_reobserve=True)['status'] == 'incomplete_path'
    gt[3, 2] = 2
    blocked = ObservedAudit(gt, np.ones_like(gt, bool), (3, 1), 1., radius=0.)
    assert blocked.score(path, (3, 7), stop_for_reobserve=True)['status'] == 'visible_collision'
