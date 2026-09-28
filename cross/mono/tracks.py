"""Causal sparse image tracks tied to an uncertain metric-depth anchor."""

import cv2
import numpy as np

from .geometry import inverse


class MetricAnchorTracks:
    def __init__(self, K):
        self.K = np.asarray(K, dtype=np.float64)
        self.gray = None
        self.pixels = np.empty((0, 2), np.float32)
        self.points = np.empty((0, 3), np.float64)
        self.last_candidates = 0

    def reset(self, gray, pixels, depth):
        self.gray = gray
        pixels = np.ascontiguousarray(pixels, np.float32).reshape(-1, 2)
        z = cv2.remap(depth.astype(np.float32), pixels[:, 0:1], pixels[:, 1:2], cv2.INTER_LINEAR).ravel()
        h, w = gray.shape
        valid = np.isfinite(z) & (z > .01)
        valid &= ((pixels >= [10, 10]) & (pixels < [w-10, h-10])).all(axis=1)
        self.pixels = pixels[valid]
        self.points = np.c_[self.pixels, np.ones(valid.sum())] @ np.linalg.inv(self.K).T * z[valid, None]

    def track(self, gray, exclusion_boxes=()):
        self.last_candidates = 0
        if self.gray is None or len(self.pixels) < 20:
            return None
        old = self.pixels.reshape(-1, 1, 2)
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        current, forward_status, error = cv2.calcOpticalFlowPyrLK(self.gray, gray, old, None, **options)
        if current is None:
            return None
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, current, None, **options)
        if backward is None:
            return None
        current = current[:, 0]
        good = forward_status.ravel().astype(bool) & backward_status.ravel().astype(bool)
        good &= np.linalg.norm(backward[:, 0] - self.pixels, axis=1) <= 1.
        good &= np.isfinite(current).all(axis=1) & (error.ravel() <= 20.)
        height, width = gray.shape
        good &= ((current >= [10, 10]) & (current < [width-10, height-10])).all(axis=1)
        for box in exclusion_boxes:
            good &= ~((current >= box[:2]) & (current <= box[2:])).all(axis=1)
        points, pixels = self.points[good], current[good]
        self.last_candidates = len(points)
        if len(points) < 20:
            return None
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points, pixels, self.K, None, iterationsCount=150, reprojectionError=3.,
            confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if not success or inliers is None or len(inliers) < 20 or len(inliers) / len(points) < .3:
            return None
        inliers = inliers.ravel()
        tiles = np.floor(pixels[inliers] / [width, height] * 4).astype(int)
        if len(np.unique(tiles, axis=0)) < 4:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(points[inliers], pixels[inliers], self.K, None, rvec, tvec)
        projected = cv2.projectPoints(points[inliers], rvec, tvec, self.K, None)[0].reshape(-1, 2)
        residual = float(np.median(np.linalg.norm(projected - pixels[inliers], axis=1)))
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
        if not np.isfinite(pose).all() or residual > 3.:
            return None
        self.gray, self.pixels, self.points = gray, pixels[inliers], points[inliers]
        return inverse(pose), len(inliers), residual
