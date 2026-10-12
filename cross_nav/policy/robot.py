"""Online navigation agent for a real robot: RGB + odometry + metric goal -> local path and velocity command.

The agent runs exactly the loop evaluated in simulation (cross_nav.policy.sim.rollout): it keeps recent frames
indexed by travelled distance, feeds the causal window (current frame plus the frames at the planner's history
distances) and the robot-frame goal to a planner, and follows the returned path. Two planners share the interface
`planner(images uint8 (S,H,W,3), goal (2,) metres forward/left, hist_chord (S,) metres) -> path (16,2)`:

* `TrajectoryPlanner`  the direct trajectory head (cross_nav.policy.head) on the frozen VGGT backbone
* `BEVPlanner`         the BEV occupancy + observability heads and the adaptive local planner

    agent = NavigationAgent(TrajectoryPlanner(head_ckpt, backbone_ckpt))
    agent.set_goal(goal_xy)                                   # odometry frame, metres
    out = agent.step(rgb, (x, y, yaw))                         # per camera frame
    publish(out['cmd'])                                        # (v m/s, omega rad/s); out['path_odom'] for display

Odometry frame: x forward / y left at yaw 0, yaw counter-clockwise (ROS REP-103). Images must match the planner's
training camera (512x384, ~90 deg horizontal FoV, level, ~0.8 m high) or be validated separately. The agent is a local
planner: it does not check collisions itself; keep the robot's own safety layer (bumpers, emergency stop, speed limits).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .sim.geom import window_ids


def to_robot_2d(points, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(points, np.float64) - (x, y)
    return np.stack((d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c), -1)


def to_odom_2d(points, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(points, np.float64)
    return np.stack((x + p[..., 0] * c - p[..., 1] * s, y + p[..., 0] * s + p[..., 1] * c), -1)


def pure_pursuit(path, v_max=.5, w_max=1., lookahead=.5, turn_only=math.radians(60)):
    """(v, omega) toward the path point `lookahead` metres along the path; rotate in place when it is far off-heading."""
    p = np.vstack(([0., 0.], np.asarray(path, np.float64)))
    seg = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    total = float(seg.sum())
    if total < 1e-3:
        return 0., 0.
    arc = np.r_[0., np.cumsum(seg)]
    aim = np.array([np.interp(min(lookahead, total), arc, p[:, i]) for i in range(2)])
    theta = math.atan2(aim[1], aim[0])
    if abs(theta) > turn_only:
        return 0., math.copysign(w_max, theta)
    dist = float(np.hypot(*aim))
    v = v_max * min(1., total / lookahead) * max(0., math.cos(theta))
    w = float(np.clip(v * 2 * aim[1] / max(dist ** 2, 1e-6), -w_max, w_max))
    return v, w


class NavigationAgent:
    def __init__(self, planner, history_m=None, arrive_m=.2, v_max=.5, w_max=1., lookahead=.5, max_frames=256,
                 camera_height=.8):
        self.planner = planner
        self.history_m = tuple(history_m or getattr(planner, 'history_m', (0., .5, 1., 2.)))
        self.history_rule = getattr(planner, 'history_rule', None) or {'type': 'distance', 'm': list(self.history_m)}
        self.arrive_m, self.v_max, self.w_max, self.lookahead = arrive_m, v_max, w_max, lookahead
        self.frames = deque(maxlen=max_frames)       # (image, pose, cumulative travel)
        self.goal, self.cum, self.camera_height = None, 0., camera_height

    def set_goal(self, goal_xy):
        self.goal = np.asarray(goal_xy, np.float64)

    def reset(self):
        self.frames.clear(); self.cum = 0.

    def step(self, image, pose):
        """image uint8 HxWx3 RGB; pose (x, y, yaw) in the odometry frame. Returns path, command and arrival flag."""
        if self.goal is None:
            raise RuntimeError('set_goal first')
        if self.frames:
            self.cum += float(np.hypot(pose[0] - self.frames[-1][1][0], pose[1] - self.frames[-1][1][1]))
        self.frames.append((np.asarray(image), tuple(pose), self.cum))
        goal_rel = to_robot_2d(self.goal, pose)
        if float(np.hypot(*goal_rel)) < self.arrive_m:
            return {'arrived': True, 'cmd': (0., 0.), 'path': np.zeros((0, 2)), 'path_odom': np.zeros((0, 2)), 'goal_rel': goal_rel}
        ids = window_ids([f[2] for f in self.frames], [f[1][2] for f in self.frames], self.history_rule)
        imgs = np.stack([self.frames[i][0] for i in ids])
        hist = np.array([math.hypot(pose[0] - self.frames[i][1][0], pose[1] - self.frames[i][1][1]) for i in ids], np.float32)
        path = np.asarray(self.planner(imgs, goal_rel.astype(np.float32), hist), np.float64)
        v, w = pure_pursuit(path, self.v_max, self.w_max, self.lookahead)
        return {'arrived': False, 'cmd': (v, w), 'path': path, 'path_odom': to_odom_2d(path, pose), 'goal_rel': goal_rel,
                'window': ids}


class TrajectoryPlanner:
    """Direct trajectory head + frozen backbone in this process (CUDA)."""
    def __init__(self, head_checkpoint, backbone_checkpoint, scale='odometry', grid=(12, 16), device='cuda'):
        import torch
        from types import SimpleNamespace
        from .server import Policy
        self.torch = torch
        self.policy = Policy(SimpleNamespace(backbone=backbone_checkpoint, head=head_checkpoint, grid=grid, scale=scale,
                                             flow_steps=10, seed=0))
        self.history_m = (0., .5, 1., 2.)

    def __call__(self, images, goal, hist_chord):
        t = self.torch
        imgs = t.from_numpy(np.ascontiguousarray(images))[None].cuda().permute(0, 1, 4, 2, 3).float().div_(255)
        hist = t.from_numpy(np.asarray(hist_chord, np.float32))[None].cuda()
        feat = self.policy(imgs, hist)
        return self.policy.predict(feat, t.from_numpy(np.asarray(goal, np.float32))[None].cuda(), hist)[0].cpu().numpy()


class BEVPlanner:
    """BEV occupancy/observability heads + adaptive local planner (cross_nav.policy.bev_policy)."""
    def __init__(self, config, backbone_checkpoint=None):
        import json
        import torch
        from pathlib import Path
        from .bev_policy import BEVPlannerPolicy
        cfg = json.loads(Path(config).read_text()) if isinstance(config, (str, bytes)) or hasattr(config, 'read_text') else config
        backbone = None
        if backbone_checkpoint:
            from .backbone import FrozenMetricTokens
            backbone = FrozenMetricTokens(backbone_checkpoint).cuda().eval()
        self.torch, self.bev = torch, BEVPlannerPolicy(cfg, backbone=backbone)
        self.history_m = tuple(self.bev.history_m)
        self.last_debug = None

    def __call__(self, images, goal, hist_chord):
        t = self.torch
        imgs = t.from_numpy(np.ascontiguousarray(images))[None].cuda().permute(0, 1, 4, 2, 3).float().div_(255)
        paths, debug = self.bev(imgs, t.from_numpy(np.asarray(goal, np.float32))[None].cuda(),
                                t.from_numpy(np.asarray(hist_chord, np.float32))[None].cuda(), want_debug=[True])
        self.last_debug = debug[0]
        return paths[0]
