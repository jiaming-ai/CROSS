"""Choose a local waypoint using predicted occupancy and input observability."""
import numpy as np

from .plan import Field, disk, risk_cost
from .planability import footprint_blocked
import cv2


class LocalWaypointPlanner:
    def __init__(self, occupancy, observability, start, res, *, threshold=.5,
                 radius=.2, risk_weight=64., min_travel=1., min_progress=.5):
        p, v = np.asarray(occupancy), np.asarray(observability)
        if v.shape != p.shape or not np.isfinite(v).all() or (v < 0).any() or (v > 1).any():
            raise ValueError('observability must be a matching probability map')
        if not 0 <= threshold <= 1 or min_travel <= 0 or min_progress <= 0:
            raise ValueError('invalid visibility or progress threshold')
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
        allowed = ~footprint_blocked(~visible, res, radius)
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
