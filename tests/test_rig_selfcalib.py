"""cross/rig/selfcalib.py on synthetic recordings: a rig with a known extrinsic and clock offset moves along a smooth
walk with turns; the clock offset, the extrinsic (hand-eye) and the pair geometry are recovered (CPU, no images)."""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from cross.rig.recording import Recording
from cross.rig.selfcalib import estimate_clock_offset, handeye, pair_frames, pair_pose

R_TRUE = Rotation.from_euler("xyz", [0.4, -0.8, 1.5], degrees=True).as_matrix()
T_TRUE = np.array([0.013, -0.193, 0.011])          # x_B = R x_A + T
D_TRUE = 0.037                                      # B clock = A clock + d


def walk(t):
    """Camera B poses (camera-to-world, OpenCV axes) of a walk with turns and small head motion."""
    yaw = 0.6 * np.sin(0.21 * t) + 0.25 * t + 0.15 * np.sin(1.3 * t)
    pitch = 0.08 * np.sin(0.9 * t + 1.0)
    roll = 0.05 * np.sin(1.7 * t)
    R = Rotation.from_euler("yxz", np.c_[yaw, pitch, roll]).as_matrix()
    p = np.c_[np.cos(0.1 * t) * 20, 0.03 * np.sin(2.0 * t), np.sin(0.1 * t) * 20]
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = R, p
    return T


def make_recs(n=1500, fps=10.0, seed=0):
    rng = np.random.default_rng(seed)
    tB = 1000.0 + np.arange(n) / fps                    # B clock
    tA = 1000.0 + np.arange(n) / fps + 0.013            # A exposures, on A's clock (a phase shift)
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R_TRUE, T_TRUE
    TB = walk(tB - 1000.0)
    TA = walk(tA + D_TRUE - 1000.0) @ X                 # A at the same physical instant as B clock tA + d
    Y = np.eye(4)                                       # A's ARKit world differs from B's
    Y[:3, :3] = Rotation.from_euler("y", 40, degrees=True).as_matrix()
    Y[:3, 3] = [3.0, 0.0, -2.0]
    TA = Y @ TA
    TA[:, :3, 3] += rng.normal(0, 0.002, (n, 3))
    TB[:, :3, 3] += rng.normal(0, 0.002, (n, 3))
    return Recording(None, pd.DataFrame(), tA, TA), Recording(None, pd.DataFrame(), tB, TB)


def test_clock_offset_and_handeye_recovered():
    recA, recB = make_recs()
    d, corr = estimate_clock_offset(recA, recB, search=0.3, step=0.001)
    assert corr > 0.9 and abs(d - D_TRUE) < 0.004, (d, corr)
    R, T, st = handeye(recA, recB, d)
    assert np.degrees(np.linalg.norm(Rotation.from_matrix(R @ R_TRUE.T).as_rotvec())) < 0.2
    assert np.linalg.norm(T - T_TRUE) < 0.01, (T, T_TRUE)


def test_pair_geometry_compensates_the_timing():
    recA, recB = make_recs()
    iB, jA, dt = pair_frames(recA, recB, D_TRUE)
    assert len(iB) > 1000 and np.all(np.abs(dt) <= 0.06)
    P = pair_pose(recB, R_TRUE, T_TRUE, D_TRUE, recB.t[iB[:50]], recA.t[jA[:50]])
    # the true relative pose of the two exposures, from the generating walk
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R_TRUE, T_TRUE
    TB = walk(recB.t[iB[:50]] - 1000.0)
    TA = walk(recA.t[jA[:50]] + D_TRUE - 1000.0) @ X
    true = np.linalg.inv(TB) @ TA
    assert np.abs(P[:, :3, 3] - true[:, :3, 3]).max() < 0.01
    ang = [np.degrees(np.linalg.norm(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).as_rotvec())) for a, b in zip(P, true)]
    assert max(ang) < 0.1
