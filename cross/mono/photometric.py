"""Subpixel alignment of descriptor correspondences, without pose priors."""

import cv2
import numpy as np


def refine_correspondences(reference, current, reference_pixels, current_pixels):
    """Align local patches and reject inconsistent/large descriptor corrections.

    Descriptor matching initializes pyramidal Lucas--Kanade. Tracking back
    must recover the original reference coordinate within one pixel. These
    are image-space checks: no trajectory, motion filter or ground truth is
    used to decide whether a correspondence should be retained.
    """
    first = np.ascontiguousarray(reference_pixels, dtype=np.float32).reshape(-1, 1, 2)
    initial = np.ascontiguousarray(current_pixels, dtype=np.float32).reshape(-1, 1, 2)
    if len(first) == 0:
        return first.reshape(-1, 2), initial.reshape(-1, 2)
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                   flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=1e-4)
    forward, good_forward, error = cv2.calcOpticalFlowPyrLK(reference, current, first, initial.copy(), **options)
    if forward is None:
        return np.empty((0, 2)), np.empty((0, 2))
    backward, good_backward, _ = cv2.calcOpticalFlowPyrLK(current, reference, forward, first.copy(), **options)
    if backward is None:
        return np.empty((0, 2)), np.empty((0, 2))
    good = good_forward.ravel().astype(bool) & good_backward.ravel().astype(bool)
    good &= np.linalg.norm(backward - first, axis=(1, 2)) <= 1.0
    good &= np.linalg.norm(forward - initial, axis=(1, 2)) <= 4.0
    good &= np.isfinite(forward).all(axis=(1, 2)) & (error.ravel() <= 20.)
    h, w = current.shape[:2]
    good &= ((forward[:, 0] >= [10, 10]) & (forward[:, 0] < [w-10, h-10])).all(axis=1)
    return first[good, 0], forward[good, 0]
