import numpy as np

from cross_nav.bev import BEVSpec
from cross_nav.observation import observed_cells


def scene(depth=1.):
    E = np.eye(4); E[:3, :3] = np.diag([1., -1., -1.]); E[2, 3] = 1.
    relative = E.copy(); relative[:3, 3] = 0
    K = np.array([[4., 0., 2.], [0., 4., 2.], [0., 0., 1.]])
    world = dict(ground=np.zeros((20, 20), np.float32), origin=np.array([-1., -1.]),
                 res=.1, R_g=np.eye(3), label_cfg={"h_lo": .1, "h_hi": 1.8})
    return dict(depths=np.full((1, 5, 5), depth), extrinsics=relative[None], intrinsics=K[None],
                up_cam=np.array([0., 0., -1.]), bev=np.ones((8, 8), np.uint8), world=world,
                anchor_E=E, spec=BEVSpec(8, .1))


def test_ground_visible_with_correct_camera_origin():
    assert observed_cells(**scene()).all()


def test_foreground_depth_occludes_ground_even_inside_frustum():
    assert not observed_cells(**scene(.2)).any()


def test_missing_depth_and_excluded_frame_cannot_establish_observation():
    assert not observed_cells(**scene(0.)).any()
    assert not observed_cells(**scene(), frame_valid=[False]).any()


def test_unknown_labels_are_not_observation_targets():
    args = scene(); args["bev"][:] = 0
    assert not observed_cells(**args).any()


def test_floor_hits_do_not_mark_occupied_columns_as_observed_obstacles():
    args = scene(); args["bev"][:] = 2
    assert not observed_cells(**args).any()
    args["depths"][:] = .8  # Surface 0.2 m above local ground.
    assert observed_cells(**args).any()
