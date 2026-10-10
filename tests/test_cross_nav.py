"""CROSS Nav: the BEV frame of the labels (world crop) and of the head (lifting from the camera) agree."""
import math

import numpy as np
import torch

from cross_nav.bev import BEVSpec, bev_axes, bev_axes_np, camera_bev_pose, crop_world, points_to_cells
from cross_nav.head import BEVHead, DirectBEVHead, GravityHead


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _camera(yaw=30.0, pitch=-20.0, roll=3.0, centre=(2.0, -1.0, 1.2)):
    """Camera-from-world E in a z-up world: level camera looking along +x, then yaw / pitch / roll."""
    R_wc0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)        # camera x right, y down, z forward -> world
    R_wc = _rot("z", yaw) @ R_wc0 @ _rot("x", pitch) @ _rot("z", roll)  # pitch about camera x (negative = down)
    E = np.eye(4)
    E[:3, :3] = R_wc.T
    E[:3, 3] = -R_wc.T @ np.asarray(centre)
    return E


def test_axes_batched_matches_numpy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = rng.normal(size=3)
        u[1] = -abs(u[1]) - 0.5
        u /= np.linalg.norm(u)
        np.testing.assert_allclose(bev_axes(torch.tensor(u)[None])[0].numpy(), bev_axes_np(u), atol=1e-6)


def test_label_crop_and_head_lift_agree():
    spec = BEVSpec(200, 0.05)
    E = _camera()
    R_g = np.eye(3)
    p, right, fwd, up_cam = camera_bev_pose(E, R_g)
    assert abs(p[2] - 1.2) < 1e-9
    # world BEV with one occupied cell at a known world point
    res_w, origin = 0.05, np.array([-10.0, -10.0])
    world = np.zeros((400, 400), np.uint8)
    X = np.array([4.0, 0.5, 0.0])                                        # a floor point in front of the camera
    world[int((X[1] - origin[1]) / res_w), int((X[0] - origin[0]) / res_w)] = 2
    lab = crop_world(world, origin, res_w, p, right, fwd, spec)
    r_lab, c_lab = np.argwhere(lab == 2)[0]
    # head: the same point seen by the camera, lifted with the camera's up vector
    Pc = E[:3, :3] @ X + E[:3, 3]
    assert Pc[2] > 0
    head = BEVHead(size=spec.size, res=spec.res, depth=1)
    K = torch.tensor([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    u, v = K[0, 0] * Pc[0] / Pc[2] + K[0, 2], K[1, 1] * Pc[1] / Pc[2] + K[1, 2]
    depth = torch.zeros(1, 1, 80, 100)
    iu, iv = int(math.floor(u)), int(math.floor(v))
    depth[0, 0, iv, iu] = Pc[2]
    K2 = K.clone()
    K2[0, 2] -= (u - iu - 0.5)                                           # put the pixel centre exactly on the point
    K2[1, 2] -= (v - iv - 0.5)
    P = head.lift(depth, None, torch.eye(4)[:3][None, None], K2[None, None], torch.zeros(1),
                  torch.tensor(up_cam, dtype=torch.float32)[None])[0, 0, iv, iu].numpy()
    assert abs(P[2] - (X[2] - p[2])) < 1e-3                              # height relative to the camera
    r, c, ok = points_to_cells(P[None].astype(np.float64), spec)
    assert ok[0] and abs(r[0] - r_lab) <= 1 and abs(c[0] - c_lab) <= 1


def test_head_shapes():
    B, S, h, w = 1, 2, 4, 5
    tl = [None] * 24
    for layer in (4, 11, 17, 23):
        tl[layer] = torch.randn(B, S, 17 + h * w, 2048)
    head = BEVHead(size=64, depth=1)
    out = head(tl, 17, (h, w), torch.randn(B, S, h * w, 1024), torch.rand(B, S, 64, 80) + 0.5,
               torch.rand(B, S, 64, 80) + 1, torch.eye(4)[:3].expand(B, S, 3, 4),
               torch.tensor([[50.0, 0, 40], [0, 50.0, 32], [0, 0, 1]]).expand(B, S, 3, 3), torch.zeros(B),
               torch.tensor([[0.0, -1.0, 0.0]]))
    assert out.shape == (B, 64, 64)
    g = GravityHead()(tl, 17)
    assert torch.allclose(g, torch.tensor([[0.0, -1.0, 0.0]]))


def test_direct_head_shapes():
    B, S, h, w = 2, 3, 4, 5
    tl = [None] * 24
    for layer in (4, 11, 17, 23):
        tl[layer] = torch.randn(B, S, 17 + h * w, 2048)
    head = DirectBEVHead(size=64, depth=1)
    out = head(tl, 17, (h, w), torch.randn(B, S, h * w, 1024), None, None, None, None, torch.zeros(B),
               torch.tensor([[0.0, -1.0, 0.0]] * B), torch.tensor([[True, True, False]] * B))
    assert out.shape == (B, 64, 64)
    out.sum().backward()


def test_planning_on_a_corridor():
    from cross_nav.plan import evaluate_planning, summarize
    gt = np.ones((80, 80), np.uint8)                 # free room 4 x 4 m
    gt[:, 38:42] = 2                                 # wall with a door
    gt[30:40, 38:42] = 1
    start = (60, 20)
    perfect = gt == 2
    blind = np.zeros_like(perfect)                   # sees no wall: straight paths collide
    closed = perfect.copy()
    closed[30:40, 38:42] = True                      # misses the door: no path to the far side
    recs = evaluate_planning(gt, {"perfect": perfect, "blind": blind, "closed": closed}, 0.05, start, n_goals=20,
                             dist_range=(1.0, 4.0), seed=1)
    assert recs
    sp, sb, sc = summarize(recs, "perfect"), summarize(recs, "blind"), summarize(recs, "closed")
    assert sp["success"] == 1.0
    assert sb["collision"] > 0 and sb["success"] < 1.0
    assert sc["no_path"] > 0
