"""Rolling-shutter rectification maps of scripts/rig/prepare_iphone_stereo.py."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_spec = importlib.util.spec_from_file_location(
    "prepare_iphone_stereo", Path(__file__).resolve().parents[1] / "scripts/rig/prepare_iphone_stereo.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)

K_RAW = np.array([[1340.0, 0, 959.5], [0, 1340.0, 719.5], [0, 0, 1]])
K_OUT = np.array([[470.0, 0, 239.5], [0, 470.0, 319.5], [0, 0, 1]])
R_RECT = Rotation.from_rotvec([0.01, -0.02, 0.005]).as_matrix()


def test_no_readout_matches_opencv():
    mx, my = prep.rolling_shutter_map(K_RAW, R_RECT.T, K_OUT, (480, 640), np.array([0.3, -0.5, 0.2]), 0.0, 1440)
    cx, cy = cv2.initUndistortRectifyMap(K_RAW, np.zeros(5), R_RECT, K_OUT, (480, 640), cv2.CV_32FC1)
    assert np.abs(mx - cx).max() < 1e-2 and np.abs(my - cy).max() < 1e-2


def test_readout_shifts_rows_by_their_time():
    w = np.array([0.0, 1.0, 0.0])                  # turning about the camera y axis: rows shift in u
    readout = 0.03
    mx0, my0 = prep.rolling_shutter_map(K_RAW, np.eye(3), K_OUT, (480, 640), w, 0.0, 1440)
    mx, my = prep.rolling_shutter_map(K_RAW, np.eye(3), K_OUT, (480, 640), w, readout, 1440)
    mid = np.argmin(np.abs(my0[:, 240] - 719.5))
    assert abs(mx[mid, 240] - mx0[mid, 240]) < 0.05          # the middle row is the frame time
    tau = (my0[0, 240] - 719.5) / 1440 * readout             # first output row: exposed tau before the middle
    # exp(-w tau) x shifts u by about -f * w_y * tau (x' = x - w_y tau z)
    assert abs((mx[0, 240] - mx0[0, 240]) - (-1340.0 * 1.0 * tau)) < 0.5
