"""Small NumPy helpers for geometric baselines."""
import numpy as np


def lift_np(depth, K, E_c_from_0, A, scale=1.0):
    """Lift every second pixel into metric BEV coordinates.

    Depth and camera translation must share input units. Apply metric scale to
    the complete camera-to-reference transform, including its translation.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and positive')
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:2, 0:W:2]
    d = depth[ys, xs]
    ok = d > 0
    P = np.stack([(xs[ok] + .5-K[0, 2])/K[0, 0]*d[ok],
                  (ys[ok] + .5-K[1, 2])/K[1, 1]*d[ok], d[ok]], -1)
    P0 = ((P-E_c_from_0[:3, 3]) @ E_c_from_0[:3, :3])*scale
    return P0 @ A.T, ok
