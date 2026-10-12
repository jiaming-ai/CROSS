"""Choose a local waypoint using predicted occupancy and input observability."""
import numpy as np

from .plan import Field, disk, risk_cost
from .planability import footprint_blocked
import cv2


class LocalWaypointPlanner:
    def __init__(self, occupancy, observability, start, res, *, threshold=.5,
                 radius=.2, risk_weight=64., visibility_weight=0., min_travel=1., min_progress=.5,
                 blocked=None):
        """blocked: optional (H, W) bool hard obstacles (footprint-inflated like unobserved cells); cells inside
        the robot's current footprint are never blocked, so a robot touching an obstacle can still leave."""
        p, v = np.asarray(occupancy), np.asarray(observability)
        if v.shape != p.shape or not np.isfinite(v).all() or (v < 0).any() or (v > 1).any():
            raise ValueError('observability must be a matching probability map')
        if not 0 <= threshold <= 1 or min_travel <= 0 or min_progress <= 0:
            raise ValueError('invalid visibility or progress threshold')
        if not np.isfinite(visibility_weight) or visibility_weight < 0:
            raise ValueError('visibility weight must be finite and nonnegative')
        self.start, self.res = tuple(start), res
        self.min_travel, self.min_progress = min_travel, min_progress
        cost = risk_cost(p, res, robot_radius=radius, weight=risk_weight)
        visible = v >= threshold
        # The current physical footprint is available without GT scene labels.
        current = np.zeros_like(p, np.uint8)
        current[self.start] = 1
        current = cv2.dilate(current, disk(int(np.ceil(radius/res))),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        visible |= current
        if visibility_weight:
            confidence = np.asarray(v, np.float32).copy()
            confidence[current] = 1.
            least = -cv2.dilate(-confidence, disk(int(np.ceil(radius/res))))
            # A conservative score; correlated cells do not give calibrated
            # probabilities for complete paths or full-footprint observability.
            cost += visibility_weight * -np.log(np.maximum(least, 1e-6))
        impassable = ~visible
        if blocked is not None:
            impassable |= np.asarray(blocked, bool) & ~current
        allowed = ~footprint_blocked(impassable, res, radius)
        self.field = Field(allowed, self.start, cost, allow_corner_cutting=False, diagonal_cost=True)
        self.coords = np.argwhere(np.isfinite(self.field.dist)) if self.field.valid else np.empty((0, 2), int)
        self.travel = np.linalg.norm(self.coords-self.start, axis=1)*res
        self.costs = self.field.dist[tuple(self.coords.T)] if len(self.coords) else np.empty(0)

    def plan(self, goal):
        goal = tuple(goal)
        if self.field.valid and np.isfinite(self.field.dist[goal]):
            return {'path':self.field.path(goal), 'target':goal, 'stop_for_reobserve':False}
        progress = (np.linalg.norm(np.asarray(goal)-self.start)
                    -np.linalg.norm(self.coords-goal, axis=1))*self.res
        eligible = np.flatnonzero((self.travel >= self.min_travel) & (progress >= self.min_progress))
        if not len(eligible):
            return {'path':None, 'target':None, 'stop_for_reobserve':True}
        target = tuple(self.coords[eligible[np.argmin(self.costs[eligible])]])
        return {'path':self.field.path(target), 'target':target, 'stop_for_reobserve':True}


class AdaptiveVisibilityPlanner:
    """Prefer the highest predicted visibility threshold allowing local progress.

    Threshold selection uses only predicted maps and the requested goal. Success
    still needs an external execution audit; visibility is not calibrated safety.
    """
    def __init__(self, occupancy, observability, start, res, *,
                 thresholds=(.75, .625, .5, .375, .25), **kwargs):
        self.thresholds = tuple(thresholds)
        if (not self.thresholds or
                any(not np.isfinite(t) or not 0 <= t <= 1 for t in self.thresholds) or
                any(a <= b for a, b in zip(self.thresholds, self.thresholds[1:]))):
            raise ValueError('thresholds must be finite, strictly decreasing probabilities')
        self.planners = [LocalWaypointPlanner(occupancy, observability, start, res,
                                              threshold=t, **kwargs)
                         for t in self.thresholds]

    def plan(self, goal):
        for threshold, planner in zip(self.thresholds, self.planners):
            result = planner.plan(goal)
            if result['path'] is not None:
                break
        return {**result, 'visibility_threshold': threshold}
