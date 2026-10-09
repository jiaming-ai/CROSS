"""Camera trajectories of a simulated scene: ground-robot / handheld-like routes through the free space of the mesh BEV.

Per trajectory (random, seeded): camera height in [0.25, 1.6] m, base pitch mostly in [-30, +10] deg with slow
variation, small roll, horizontal FoV in [60, 100] deg, a clearance radius, a frame spacing in [0.15, 0.3] m and a
rotation threshold of 8-10 deg; handheld-style trajectories also look around (slow yaw oscillation).  The route visits
random waypoints spread over the free space inside the annotated rooms (farthest-point sampling), joined by SimChange's
clearance-aware planner (trajectory.plan_route); SimChange's unicycle model (trajectory.robot_motion) drives it at a
fine time step, and frames are kept whenever the camera moved `step` metres or turned `turn` degrees since the last
kept frame.

    trajs = make_trajectories(bev_out, aux, n=3, seed=0)    # list of dicts (c2w (N,4,4) OpenCV axes, K, W, H, ...)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy import ndimage

FREE = 1


def _simchange():
    root = os.environ.get("SIMCHANGE_ROOT", "/data2/projects/cross_v2/SimChange")
    p = os.path.join(root, "simchange")
    if p not in sys.path:
        sys.path.insert(0, p)
    import trajectory as T                                  # noqa: E402  (SimChange, no Blender dependency)
    return T


def smooth_noise(n: int, rng, scale_frames: float, amp: float) -> np.ndarray:
    """Smooth random signal (Gaussian-filtered white noise normalised to std `amp`)."""
    if n < 2 or amp == 0:
        return np.zeros(n)
    x = ndimage.gaussian_filter1d(rng.standard_normal(n + int(6 * scale_frames)), scale_frames, mode="wrap")
    x = x[int(3 * scale_frames): int(3 * scale_frames) + n]
    return x / max(x.std(), 1e-9) * amp


def look_rotation(yaw, pitch, roll) -> np.ndarray:
    """Camera-to-world rotation (OpenCV axes: x right, y down, z forward) for a z-up world."""
    f = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    cr, sr = math.cos(roll), math.sin(roll)
    r, d = cr * r + sr * d, -sr * r + cr * d
    return np.stack([r, d, f], 1)


def sample_params(rng) -> dict:
    style = "robot" if rng.random() < 0.5 else "handheld"
    if style == "robot":
        h = float(rng.uniform(0.25, 1.0))
        pitch = float(np.clip(rng.normal(-5.0, 8.0), -30, 10))
        yaw_amp, roll_amp, bob = 0.0, 0.7, 0.0
    else:
        h = float(rng.uniform(0.8, 1.6))
        pitch = float(np.clip(rng.normal(-12.0, 9.0), -30, 10))
        yaw_amp, roll_amp, bob = float(rng.uniform(5, 25)), 2.0, 0.015
    return dict(style=style, height=h, pitch=pitch, pitch_amp=float(rng.uniform(1.5, 5.0)), yaw_amp=yaw_amp,
                roll_amp=roll_amp, bob=bob, hfov=float(rng.uniform(60, 100)), step=float(rng.uniform(0.15, 0.3)),
                turn=float(rng.uniform(8.0, 10.0)), radius=float(rng.uniform(0.25, 0.4) if style == "handheld" else rng.uniform(0.22, 0.32)),
                frames=int(rng.integers(150, 251)))


def plan_mask(bev, rmask, res, radius):
    """Largest 8-connected component of the free cells farther than `radius` from anything not free (inside the rooms)."""
    free = bev == FREE
    d = ndimage.distance_transform_edt(free) * res
    m = d >= radius
    if rmask is not None:
        m &= rmask
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def farthest_points(ys, xs, k, rng):
    i0 = int(rng.integers(len(ys)))
    pts = [i0]
    d = (ys - ys[i0]) ** 2 + (xs - xs[i0]) ** 2
    for _ in range(k - 1):
        j = int(np.argmax(d))
        pts.append(j)
        d = np.minimum(d, (ys - ys[j]) ** 2 + (xs - xs[j]) ** 2)
    return pts


def make_trajectory(bev, rmask, origin, ground, res, rng, p: dict, z_floor_default: float = 0.0):
    T = _simchange()
    mask = plan_mask(bev, rmask, res, p["radius"])
    if mask is None or mask.sum() * res * res < 4.0:
        return None
    ys, xs = np.nonzero(mask)
    area = mask.sum() * res * res
    target_len = p["frames"] * p["step"] * 0.9
    k = int(np.clip(area / 6.0, 4, 14))
    idx = farthest_points(ys, xs, k, rng)
    # nearest-neighbour tour from a random start
    order = [idx.pop(int(rng.integers(len(idx))))]
    while idx:
        last = order[-1]
        j = int(np.argmin([(ys[i] - ys[last]) ** 2 + (xs[i] - xs[last]) ** 2 for i in idx]))
        order.append(idx.pop(j))
    wp = [(origin[0] + (xs[i] + 0.5) * res, origin[1] + (ys[i] + 0.5) * res) for i in order]
    occ = {"grid": np.where(mask, 0, 255).astype(np.uint8), "x0": origin[0] + 0.5 * res, "y0": origin[1] + 0.5 * res,
           "res": res}
    route = T.plan_route(wp, occ, radius=0.01, step=0.02, smooth_m=0.8)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))])
    if s[-1] > target_len:
        route = route[s <= target_len]
    if len(route) < 20:
        return None
    lim = T.MotionLimits(v_max=0.03, a_max=0.003, w_max_deg=2.0, alpha_max_deg=0.4)
    xy, heading, _ = T.robot_motion(route, lim)
    n = len(xy)
    pitch = np.radians(p["pitch"] + smooth_noise(n, rng, 60, p["pitch_amp"]))
    yaw = heading + np.radians(smooth_noise(n, rng, 80, p["yaw_amp"]))
    roll = np.radians(smooth_noise(n, rng, 40, p["roll_amp"]))
    hz = p["height"] + smooth_noise(n, rng, 15, p["bob"])
    # floor height under each station (the planning ground; default when unknown)
    ix = np.clip(np.floor((xy[:, 0] - origin[0]) / res).astype(int), 0, bev.shape[1] - 1)
    iy = np.clip(np.floor((xy[:, 1] - origin[1]) / res).astype(int), 0, bev.shape[0] - 1)
    zf = ground[iy, ix].astype(np.float64)
    zf = np.where(np.isfinite(zf), zf, z_floor_default)
    zf = ndimage.median_filter(zf, 9, mode="nearest")
    keep = [0]
    Rk = look_rotation(yaw[0], pitch[0], roll[0])
    for i in range(1, n):
        R = look_rotation(yaw[i], pitch[i], roll[i])
        ang = math.degrees(math.acos(np.clip((np.trace(R.T @ Rk) - 1) / 2, -1, 1)))
        if np.linalg.norm(xy[i] - xy[keep[-1]]) >= p["step"] or ang >= p["turn"]:
            keep.append(i)
            Rk = R
        if len(keep) >= p["frames"]:
            break
    c2w = np.zeros((len(keep), 4, 4))
    for j, i in enumerate(keep):
        c2w[j, :3, :3] = look_rotation(yaw[i], pitch[i], roll[i])
        c2w[j, :3, 3] = [xy[i, 0], xy[i, 1], zf[i] + hz[i]]
        c2w[j, 3, 3] = 1
    return {"c2w": c2w, "h": hz[keep], "floor": zf[keep], "xy_fine": xy, "length_m": float(min(s[-1], target_len))}


def make_trajectories(out: dict, aux: dict, n: int, seed: int, width=640, height=480, presets=None):
    rng = np.random.default_rng(seed)
    bev, origin, res = out["bev"], out["origin"], float(out["res"])
    ground = aux["g_int"]
    rmask = aux.get("rmask")
    if rmask is not None:
        k = int(round(0.4 / res)) * 2 + 1
        rmask = ndimage.binary_dilation(rmask, np.ones((k, k), bool))
    presets = presets or ["noon", "afternoon", "morning", "overcast", "evening", "night"]
    pw = np.array([0.22, 0.2, 0.18, 0.18, 0.12, 0.1])[: len(presets)]
    trajs = []
    for t in range(n):
        for attempt in range(4):
            p = sample_params(rng)
            tr = make_trajectory(bev, rmask, origin, ground, res, rng, p)
            if tr is not None and len(tr["c2w"]) >= 60:
                break
        else:
            continue
        fx = width / 2.0 / math.tan(math.radians(p["hfov"]) / 2.0)
        K = np.array([[fx, 0, width / 2.0 - 0.5], [0, fx, height / 2.0 - 0.5], [0, 0, 1.0]])
        trajs.append({"name": f"t{t:02d}", "params": p, "preset": str(rng.choice(presets, p=pw / pw.sum())),
                      "K": K, "width": width, "height": height, **tr})
    return trajs
