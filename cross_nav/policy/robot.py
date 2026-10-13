"""Online navigation agent for a real robot: RGB + odometry + metric goal -> local path and velocity command.

The agent runs exactly the loop evaluated in simulation (cross_nav.policy.sim.rollout): it keeps recent frames
indexed by travelled distance, feeds the causal window (current frame plus the frames at the planner's history
distances) and the robot-frame goal to a planner, and follows the returned path. Two planners share the interface
`planner(images uint8 (S,H,W,3), goal (2,) metres forward/left, hist_chord (S,) metres) -> path (16,2)`:

* `TrajectoryPlanner`  the direct trajectory head (cross_nav.policy.head) on the frozen VGGT backbone
* `BEVPlanner`         the BEV occupancy + observability heads and the adaptive local planner
* `HybridPlanner`      both on one backbone, switched by cross_nav.policy.hybrid (the BEV planner drives, the
                       trajectory head takes over on a stall or too little goal progress)

    agent = NavigationAgent(TrajectoryPlanner(head_ckpt, backbone_ckpt))
    agent.set_goal(goal_xy)                                   # odometry frame, metres
    out = agent.step(rgb, (x, y, yaw), t)                      # per camera frame; t: time in seconds
    publish(out['cmd'])                                        # (v m/s, omega rad/s); out['path_odom'] for display

Odometry frame: x forward / y left at yaw 0, yaw counter-clockwise (ROS REP-103). Images must match the planner's
training camera (512x384, ~90 deg horizontal FoV, level, ~0.8 m high) or be validated separately. Validated in
Habitat at 10 Hz with v_max 0.5 m/s (rollout --control robot; outputs/2026-10-12_sim_nav/robot_control.md): pass
timestamps (the BEV planner's bump rule needs the commanded motion) and use replan_every for a planner slower than
the camera. The agent is a local
planner: it does not check collisions itself; keep the robot's own safety layer (bumpers, emergency stop, speed limits).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .sim.geom import TRAIL_M, window_ids


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


def pure_pursuit(path, v_max=.5, w_max=1., lookahead=.4, turn_only=math.radians(60)):
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
    def __init__(self, planner, history_m=None, arrive_m=.2, v_max=.5, w_max=1., lookahead=.4, max_frames=1024,
                 camera_height=.8, store_m=.1, store_deg=10., replan_every=1, recovery='auto'):
        self.planner = planner
        if recovery == 'auto':      # stuck recovery for the trajectory planner (the BEV planner has its own rules):
            from .recovery import StuckRecovery    # < 0.3 m displacement in 10 s (100 planning frames without stamps)
            recovery = None if getattr(planner, 'uses_odometry', False) else StuckRecovery(window=100, window_s=10., max_turn_steps=50)
        self.recovery = recovery
        self.history_m = tuple(history_m or getattr(planner, 'history_m', (0., .5, 1., 2.)))
        self.history_rule = getattr(planner, 'history_rule', None) or {'type': 'distance', 'm': list(self.history_m)}
        self.arrive_m, self.v_max, self.w_max, self.lookahead = arrive_m, v_max, w_max, lookahead
        # Past frames are kept after store_m of travel or store_deg of rotation since the last kept one (the simulator
        # moves up to 0.25 m per step; a camera at 10-30 Hz would otherwise fill the buffer within a few metres and
        # cut the 16 m odometry trail and the BEV planner's keyframe history short). The current frame always leads.
        self.store_m, self.store_rad = store_m, math.radians(store_deg)
        self.frames = deque(maxlen=max_frames)       # kept frames: (image, pose, cumulative travel)
        self.goal, self.cum, self.last, self.camera_height = None, 0., None, camera_height
        # Plan on every replan_every-th frame (a planner slower than the camera); in between, follow the rest of the
        # last plan, kept in the odometry frame.
        self.replan_every, self.tick, self.plan = max(1, int(replan_every)), 0, None
        self.stamp, self.v_cmd, self.commanded = None, 0., 0.   # forward motion commanded since the last plan

    def set_goal(self, goal_xy):
        self.goal = np.asarray(goal_xy, np.float64)

    def reset(self):
        self.frames.clear(); self.cum = 0.; self.last = None; self.tick = 0; self.plan = None
        self.stamp, self.v_cmd, self.commanded = None, 0., 0.
        if self.recovery is not None:
            self.recovery.reset()
        if hasattr(self.planner, 'reset'):
            self.planner.reset()

    def trail(self, pose, frames=None):
        """Past positions TRAIL_M metres of travel ago in the current robot frame, with availability flags."""
        frames = list(self.frames) if frames is None else frames
        out = np.zeros((len(TRAIL_M), 3), np.float32)
        cum = [f[2] for f in frames]
        for k, h in enumerate(TRAIL_M):
            past = [f for f, c in zip(frames, cum) if c <= cum[-1] - h]
            if past and cum[-1] - cum[0] >= h:
                out[k] = (*to_robot_2d(np.asarray(past[-1][1][:2]), pose), 1.)
        return out

    def step(self, image, pose, stamp=None):
        """image uint8 HxWx3 RGB; pose (x, y, yaw) in the odometry frame; stamp: time in seconds (optional: with it,
        the forward motion commanded since the last plan goes to the BEV planner's bump rule, which then remembers an
        obstacle ahead when the odometry shows that motion was not executed). Returns path, command and arrival flag."""
        if self.goal is None:
            raise RuntimeError('set_goal first')
        if stamp is not None and self.stamp is not None:
            self.commanded += self.v_cmd * max(0., stamp - self.stamp)
        self.stamp = stamp
        if self.last is not None:
            self.cum += float(np.hypot(pose[0] - self.last[0], pose[1] - self.last[1]))
        self.last = tuple(pose)
        cur = (np.asarray(image), tuple(pose), self.cum)
        if not self.frames or self.cum - self.frames[-1][2] >= self.store_m or \
                abs((pose[2] - self.frames[-1][1][2] + math.pi) % (2 * math.pi) - math.pi) >= self.store_rad:
            self.frames.append(cur)
        frames = list(self.frames) if self.frames[-1] is cur else list(self.frames) + [cur]
        goal_rel = to_robot_2d(self.goal, pose)
        if float(np.hypot(*goal_rel)) < self.arrive_m:
            return {'arrived': True, 'cmd': (0., 0.), 'path': np.zeros((0, 2)), 'path_odom': np.zeros((0, 2)), 'goal_rel': goal_rel}
        self.tick += 1
        if self.plan is not None and (self.tick - 1) % self.replan_every:
            path = self.remaining(pose)
            v, w = pure_pursuit(path, self.v_max, self.w_max, self.lookahead)
            self.v_cmd = v
            return {'arrived': False, 'cmd': (v, w), 'path': path, 'path_odom': to_odom_2d(path, pose), 'goal_rel': goal_rel,
                    'replanned': False}
        if getattr(self.planner, 'hybrid', False):
            path = self.hybrid_plan(frames, pose, goal_rel, stamp)
            self.plan = np.vstack((np.asarray(pose[:2], np.float64)[None], to_odom_2d(path, pose)))
            v, w = pure_pursuit(path, self.v_max, self.w_max, self.lookahead)
            self.v_cmd = v
            return {'arrived': False, 'cmd': (v, w), 'path': path, 'path_odom': self.plan[1:], 'goal_rel': goal_rel,
                    'replanned': True, 'controller': self.planner.controller}
        ids = window_ids([f[2] for f in frames], [f[1][2] for f in frames], self.history_rule)
        imgs = np.stack([frames[i][0] for i in ids])
        hist = np.array([math.hypot(pose[0] - frames[i][1][0], pose[1] - frames[i][1][1]) for i in ids], np.float32)
        if getattr(self.planner, 'uses_odometry', False):   # per-episode rules (BEV planner): poses of the window
            path = self.planner(imgs, goal_rel.astype(np.float32), hist, odom=np.asarray(pose, np.float64),
                                window_odom=np.array([frames[i][1] for i in ids], np.float64),
                                expect=self.commanded if stamp is not None else None)
            self.commanded = 0.
        else:
            path = self.planner(imgs, goal_rel.astype(np.float32), hist, trail=self.trail(pose, frames))
            if self.recovery is not None:
                path = self.recovery(pose, path, stamp)
        path = np.asarray(path, np.float64)
        self.plan = np.vstack((np.asarray(pose[:2], np.float64)[None], to_odom_2d(path, pose)))
        v, w = pure_pursuit(path, self.v_max, self.w_max, self.lookahead)
        self.v_cmd = v
        return {'arrived': False, 'cmd': (v, w), 'path': path, 'path_odom': self.plan[1:], 'goal_rel': goal_rel,
                'window': ids, 'replanned': True}

    def hybrid_plan(self, frames, pose, goal_rel, stamp):
        """HybridPlanner: the BEV planner runs on every plan (its map and bump memory stay current); the switch picks its
        path or the trajectory head's (with its own stuck recovery, reset at each hand-over)."""
        hp, cum, yaw = self.planner, [f[2] for f in frames], [f[1][2] for f in frames]
        hist = lambda ids: np.array([math.hypot(pose[0] - frames[i][1][0], pose[1] - frames[i][1][1]) for i in ids], np.float32)
        ids = window_ids(cum, yaw, hp.bev.history_rule)
        path = hp.bev(np.stack([frames[i][0] for i in ids]), goal_rel.astype(np.float32), hist(ids), odom=np.asarray(pose, np.float64),
                      window_odom=np.array([frames[i][1] for i in ids], np.float64), expect=self.commanded if stamp is not None else None)
        self.commanded = 0.
        st = hp.bev.bev.episodes.get('robot')
        hp.controller = hp.switch(pose, goal_rel, st.no_route if st is not None else 0, stamp)
        if hp.controller == 'traj':
            if hp.switch.fresh:
                hp.recovery.reset()
            ids = window_ids(cum, yaw, {'type': 'distance', 'm': list(hp.traj.history_m)})
            path = hp.traj(np.stack([frames[i][0] for i in ids]), goal_rel.astype(np.float32), hist(ids), trail=self.trail(pose, frames))
            path = hp.recovery(pose, path, stamp)
        return np.asarray(path, np.float64)

    def remaining(self, pose):
        """The last plan after its point nearest to the robot, in the current robot frame."""
        k = int(np.argmin(np.linalg.norm(self.plan - np.asarray(pose[:2]), axis=-1)))
        return to_robot_2d(self.plan[min(k + 1, len(self.plan) - 1):], pose)


class TrajectoryPlanner:
    """Direct trajectory head + frozen backbone in this process (CUDA)."""
    def __init__(self, head_checkpoint, backbone_checkpoint, scale='odometry', grid=(12, 16), device='cuda', flow_temp=.5):
        import torch
        from types import SimpleNamespace
        from .server import Policy
        self.torch = torch
        self.policy = Policy(SimpleNamespace(backbone=backbone_checkpoint, head=head_checkpoint, grid=grid, scale=scale,
                                             flow_steps=10, seed=0, flow_temp=flow_temp))
        self.history_m = (0., .5, 1., 2.)

    def __call__(self, images, goal, hist_chord, trail=None):
        t = self.torch
        imgs = t.from_numpy(np.ascontiguousarray(images))[None].cuda().permute(0, 1, 4, 2, 3).float().div_(255)
        hist = t.from_numpy(np.asarray(hist_chord, np.float32))[None].cuda()
        feat = self.policy(imgs, hist)
        tr = None if trail is None else t.from_numpy(np.asarray(trail, np.float32))[None].cuda()
        return self.policy.predict(feat, t.from_numpy(np.asarray(goal, np.float32))[None].cuda(), hist, tr)[0].cpu().numpy()


class BEVPlanner:
    """BEV occupancy/observability heads + adaptive local planner (cross_nav.policy.bev_policy)."""
    def __init__(self, config, backbone_checkpoint=None, backbone=None):
        import json
        import torch
        from pathlib import Path
        from .bev_policy import BEVPlannerPolicy
        cfg = json.loads(Path(config).read_text()) if isinstance(config, (str, bytes)) or hasattr(config, 'read_text') else config
        if backbone is None and backbone_checkpoint:
            from .backbone import FrozenMetricTokens
            backbone = FrozenMetricTokens(backbone_checkpoint).cuda().eval()
        self.torch, self.bev = torch, BEVPlannerPolicy(cfg, backbone=backbone)
        self.history_m = tuple(self.bev.history_m)
        self.history_rule = self.bev.history_rule
        self.uses_odometry = True
        self.last_debug = None

    def reset(self):
        self.bev.end_episode('robot')

    def __call__(self, images, goal, hist_chord, odom=None, window_odom=None, expect=None):
        """odom: (x, y, yaw) of the current frame and window_odom (S, 3) of the window's frames, for the per-episode
        rules (memory, bump, scale). expect: forward motion commanded since the previous call (NavigationAgent passes
        it when step() gets timestamps); the bump rule needs it on a robot (without it, it assumes the simulator's
        step controller: keep 'bump' off then)."""
        t = self.torch
        imgs = t.from_numpy(np.ascontiguousarray(images))[None].cuda().permute(0, 1, 4, 2, 3).float().div_(255)
        ep = [{'ep': 'robot', 'odom': odom, 'window_odom': window_odom, 'expect': expect}] if odom is not None else None
        paths, debug = self.bev(imgs, t.from_numpy(np.asarray(goal, np.float32))[None].cuda(),
                                t.from_numpy(np.asarray(hist_chord, np.float32))[None].cuda(), want_debug=[True], episodes=ep)
        self.last_debug = debug[0]
        return paths[0]


class HybridPlanner:
    """BEV planner + trajectory head on one shared backbone, switched by cross_nav.policy.hybrid.Switch
    (configs/policy/hybrid_best.json: BEV drives; the trajectory head takes over for block_s seconds after a stall or
    too little goal progress). Use with NavigationAgent and time stamps; out['controller'] says which one drove."""
    hybrid = True

    def __init__(self, bev_config, head_checkpoint, backbone_checkpoint, hybrid_config, flow_temp=.5):
        import json
        from pathlib import Path
        from .hybrid import Switch
        from .recovery import StuckRecovery
        self.traj = TrajectoryPlanner(head_checkpoint, backbone_checkpoint, flow_temp=flow_temp)
        self.bev = BEVPlanner(bev_config, backbone=self.traj.policy.backbone)
        cfg = json.loads(Path(hybrid_config).read_text()) if isinstance(hybrid_config, (str, bytes)) or hasattr(hybrid_config, 'read_text') else hybrid_config
        self.cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
        self.switch, self.recovery = Switch(self.cfg), StuckRecovery(window=100, window_s=10., max_turn_steps=50)
        self.history_m, self.history_rule, self.uses_odometry, self.controller = self.bev.history_m, self.bev.history_rule, True, None

    def reset(self):
        from .hybrid import Switch
        self.bev.reset(); self.recovery.reset(); self.switch = Switch(self.cfg); self.controller = None
