import numpy as np

from cross_nav.local_plan import LocalWaypointPlanner
from cross_nav.observed_plan import ObservedAudit


def test_local_waypoint_stays_visible_and_moves_toward_goal():
    p = np.zeros((9, 12), np.float32)
    v = np.zeros_like(p); v[:, :6] = 1
    planner = LocalWaypointPlanner(p, v, (4, 1), 1., radius=0.)
    result = planner.plan((4, 10))
    assert result['stop_for_reobserve']
    assert result['target'] != (4, 10)
    assert all(v[cell] for cell in result['path'])
    audit = ObservedAudit(np.ones_like(p, np.uint8), v.astype(bool), (4, 1), 1., radius=0.)
    assert audit.score(result['path'], (4, 10), stop_for_reobserve=True)['status'] == 'safe_to_reobserve'


def test_visible_goal_and_insufficient_predicted_horizon():
    p = np.zeros((9, 12), np.float32)
    visible = LocalWaypointPlanner(p, np.ones_like(p), (4, 1), 1., radius=0.)
    assert visible.plan((4, 10))['target'] == (4, 10)
    assert not visible.plan((4, 10))['stop_for_reobserve']
    unknown = LocalWaypointPlanner(p, np.zeros_like(p), (4, 1), 1., radius=0.)
    assert unknown.plan((4, 10))['path'] is None
