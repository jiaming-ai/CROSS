"""Receding-horizon path audit; stop evaluation at the observed-space boundary.

This measures a proposed local execution segment, not final-goal completion or
closed-loop navigation. Observation masks are GT diagnostics, never planner inputs.
"""
import cv2
import numpy as np

from .plan import Field, disk
from .planability import footprint_blocked


class ObservedAudit:
    def __init__(self, gt, observed, start, res, radius=.2, min_travel=1., min_progress=.5):
        self.gt = np.asarray(gt)
        self.observed = np.asarray(observed, bool).copy()
        if self.observed.shape != self.gt.shape or res <= 0 or radius < 0:
            raise ValueError("invalid observation grid or geometry")
        if min_travel <= 0 or min_progress <= 0:
            raise ValueError("progress thresholds must be positive")
        self.start, self.res = tuple(start), res
        self.min_travel, self.min_progress = min_travel, min_progress
        # Only the currently occupied robot footprint has proprioceptive support.
        current = np.zeros_like(self.gt, np.uint8)
        current[self.start] = 1
        current = cv2.dilate(current, disk(int(np.ceil(radius / res))),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        self.observed |= current & (self.gt == 1)
        self.observed &= self.gt != 0
        self.inside = ~footprint_blocked(~self.observed, res, radius)
        self.free = ~footprint_blocked((self.gt != 1) | ~self.observed, res, radius)
        field = Field(self.free, self.start, allow_corner_cutting=False)
        self.reachable = np.argwhere(np.isfinite(field.dist)) if field.valid else np.empty((0, 2), int)

    def opportunity(self, goal):
        """Can GT support a nontrivial visible segment toward this goal?"""
        q = self.reachable
        if not len(q):
            return False
        if np.any(np.all(q == goal, axis=1)):
            return True
        distance = np.linalg.norm(np.asarray(goal) - self.start) * self.res
        travel = np.linalg.norm(q - self.start, axis=1) * self.res
        progress = distance - np.linalg.norm(q - goal, axis=1) * self.res
        return bool(np.any((travel >= self.min_travel) & (progress >= self.min_progress)))

    def score(self, path, goal, *, stop_for_reobserve=False):
        """Audit a segment; an explicit local stop must still make useful progress.

        ``stop_for_reobserve`` declares a planner action, not an inferred success
        for every truncated path. Report its use as observed-local-stop-v2.
        """
        if path is None:
            return {"status": "no_path", "travel": 0., "progress": 0.}
        if not len(path) or tuple(path[0]) != self.start:
            raise ValueError("path must begin at the unchanged start")
        previous, travel = self.start, 0.
        initial = np.linalg.norm(np.asarray(goal) - self.start) * self.res

        def result(status):
            progress = initial - np.linalg.norm(np.asarray(goal) - previous) * self.res
            return {"status": status, "travel": float(travel), "progress": float(progress),
                    "last_observed_cell": list(previous)}

        for k, cell in enumerate(path):
            cell = tuple(cell)
            if k and max(abs(cell[0]-previous[0]), abs(cell[1]-previous[1])) > 1:
                raise ValueError("path has a non-adjacent step")
            checked = [cell]
            if k and cell[0] != previous[0] and cell[1] != previous[1]:
                checked += [(cell[0], previous[1]), (previous[0], cell[1])]
            if any(not (0 <= r < self.gt.shape[0] and 0 <= c < self.gt.shape[1]) for r, c in checked):
                raise ValueError("path leaves grid")
            # No motion into a footprint with unobserved cells is scored as executed.
            if not all(self.inside[x] for x in checked):
                out = result("no_progress")
                if out["travel"] >= self.min_travel and out["progress"] >= self.min_progress:
                    out["status"] = "safe_to_reobserve"
                return out
            if not all(self.free[x] for x in checked):
                return result("visible_collision")
            travel += np.linalg.norm(np.asarray(cell) - previous) * self.res
            previous = cell
            if cell == tuple(goal):
                return result("goal_reached")
        out = result("incomplete_path")
        if stop_for_reobserve and out["travel"] >= self.min_travel and out["progress"] >= self.min_progress:
            out["status"] = "safe_to_reobserve"
        return out
