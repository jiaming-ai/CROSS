import numpy as np
import pytest

from cross_nav.geometry import lift_np


def test_metric_scale_includes_camera_translation():
    K = np.array([[1.,0,.5],[0,1.,.5],[0,0,1.]])
    E = np.eye(4); E[0,3] = -1.
    points, valid = lift_np(np.array([[2.]]), K, E, np.eye(3), scale=3.)
    # Camera is at x=1 in reconstruction units, hence x=3 metres.
    np.testing.assert_allclose(points, [[3.,0.,6.]])
    assert valid.all()


def test_already_metric_geometry_keeps_unit_scale_and_rotates():
    K = np.array([[1.,0,.5],[0,1.,.5],[0,0,1.]])
    E = np.eye(4); E[0,3] = -3.
    A = np.array([[0,0,1],[1,0,0],[0,1,0.]])
    points, _ = lift_np(np.array([[6.]]), K, E, A)
    np.testing.assert_allclose(points, [[6.,3.,0.]])
    with pytest.raises(ValueError):lift_np(np.array([[6.]]), K, E, A, scale=0.)
