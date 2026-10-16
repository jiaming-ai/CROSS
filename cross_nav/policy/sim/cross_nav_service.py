"""CROSS with navigation in one feed-forward pass, as a policy server for one simulator worker (CROSS environment).

    python -m cross_nav.policy.sim.cross_nav_service --socket /tmp/cn0.sock --bev cross_nav/configs/policy/bev_planner_best.json \
        [--extra-views K] [--shadow 0 1 2 3 4] [--set section.key=value ...]

The worker (cross_nav.policy.sim.rollout, step control) talks to it as to cross_nav.policy.server: one 'info', then per
episode 'step' messages with the newest frame, the robot-frame goal and the odometry pose, and an 'end'. Each episode
is a fresh CROSS session (mono + wheel odometry: --mode mono_wheel's configuration, synchronous pose-graph
optimisation, an observation at every frame). The BEV occupancy + observability heads run on CROSS's own VGGT-Omega
pass (pose_est.ff.nav_bev_head): the current view, its retrieved references, the previous observation and K extra
nearby keyframes chosen for coverage (pose_est.ff.nav_extra_views); the BEV planner's closed-loop rules
(cross_nav.policy.bev_policy.BEVPlannerPolicy.plan_maps) plan on those maps. No second backbone pass.

--shadow K...: diagnostic passes per step with K extra keyframes each (same CROSS state, same views otherwise); their
BEVs are planned without episode memory (the per-window local planner) and the paths returned in reply['local'], which
the worker scores against the navmesh (local success, rollout --local-eval). The robot drives on the main pass's plan.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from multiprocessing.connection import Listener

import numpy as np

from . import wire


def camera_delta(fwd, left, th):
    """4x4 camera-frame (OpenCV) motion T_prev_cur of a planar robot motion (forward, left, turn) of a level camera
    at the robot's rotation centre."""
    c, s = math.cos(th), math.sin(th)
    t = np.eye(4)
    t[:3, :3] = [[c, 0., -s], [0., 1., 0.], [s, 0., c]]      # rotation about the camera's -y (up) axis by th
    t[:3, 3] = [-left, 0., fwd]
    return t


def odom_delta(prev, cur):
    """(forward, left, turn) of odometry pose cur (x, y, yaw; REP-103) in the frame of prev."""
    dx, dy = cur[0] - prev[0], cur[1] - prev[1]
    c, s = math.cos(prev[2]), math.sin(prev[2])
    th = (cur[2] - prev[2] + math.pi) % (2 * math.pi) - math.pi
    return c * dx + s * dy, -s * dx + c * dy, th


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--socket', required=True)
    p.add_argument('--bev', required=True, help='BEV planner JSON (occupancy / visibility / visibility_gravity heads '
                   'and the closed-loop rules; its backbone entry is not used: CROSS runs its own checkpoint)')
    p.add_argument('--extra-views', type=int, default=0, help='pose_est.ff.nav_extra_views of the driving pass')
    p.add_argument('--shadow', type=int, nargs='*', default=[], help='extra-view counts of diagnostic passes')
    p.add_argument('--mode', default='mono_wheel', choices=['mono_wheel'])
    p.add_argument('--set', nargs='*', default=[], help='CROSS config overrides section.key=value')
    p.add_argument('--timestep', type=float, default=.5, help='seconds per simulator step (frame time stamps)')
    a = p.parse_args()
    import torch
    from cross.core.config import load_config
    from cross.core.system import System
    from cross.core.types import Camera
    from cross.pipeline import Pipeline, mode_config_files
    from ..bev_policy import BEVPlannerPolicy, Episode, plan_window, resolve

    bev_cfg = resolve(json.loads(open(a.bev).read()))
    sets = ['pose_est.obs_max_interval_steps=1',                    # an observation (and a BEV) at every frame
            f'pose_est.ff.nav_bev_head={bev_cfg["occupancy"]}', f'pose_est.ff.nav_bev_visibility={bev_cfg["visibility"]}',
            f'pose_est.ff.nav_bev_visibility_gravity={bev_cfg["visibility_gravity"]}',
            f'pose_est.ff.nav_extra_views={a.extra_views}'] + list(a.set)
    base = load_config(*mode_config_files('mono', 'external', 'ff'), sets=sets)
    base.mapping.loop_closure.async_pgo = False
    listener = Listener(a.socket, family='AF_UNIX')
    print(json.dumps({'listening': a.socket, 'extra_views': a.extra_views, 'shadow': a.shadow,
                      'checkpoint': base.pose_est.ff.checkpoint}), flush=True)
    conn = listener.accept()
    info = {'history_m': [0.], 'history_rule': {'type': 'distance', 'm': [0.]}, 'bev': a.bev, 'cross': True,
            'extra_views': a.extra_views, 'shadow': a.shadow}
    pose_est, pipe, policy, uid = None, None, None, None
    state, last_odom, near = None, None, None
    stats = {'steps': 0, 'no_bev': 0, 'nav_only': 0, 'head_scale': 0, 'seconds': 0., 'bev_s': 0.}
    t_start = time.time()
    while True:
        m = wire.recv(conn)
        cmd = m['cmd']
        if cmd == 'info':
            wire.send(conn, info)
        elif cmd == 'end':
            if pipe is not None:
                pipe.mapper.shutdown()
            pipe, uid, state = None, None, None
            wire.send(conn, {})
        elif cmd == 'quit':
            wire.send(conn, {})
            break
        elif cmd == 'step':
            if m['ep'] != uid:                    # a new episode: a new CROSS session (the model is reused)
                if pipe is not None:
                    pipe.mapper.shutdown()
                H, W = m['frame'].shape[:2]
                hfov = float(m['meta']['camera']['hfov'])
                f = W / 2 / math.tan(math.radians(hfov) / 2)
                K = np.array([[f, 0., W / 2], [0., f, H / 2], [0., 0., 1.]])
                system = System(visualize=False, debug=False, camera=Camera(K, int(W), int(H)),
                                config=copy.deepcopy(base), pose_estimator=pose_est)
                if pose_est is None:
                    pose_est = system.pose_est
                    pose_est.nav_shadow = list(a.shadow)
                    policy = BEVPlannerPolicy(bev_cfg, predictor=pose_est.backend.nav)
                pipe = Pipeline(system, None, 1, 'mono', 'external')
                uid, state, last_odom = m['ep'], Episode(), None
            t0 = time.perf_counter()
            odom = np.asarray(m['odom'], np.float64)
            delta = None if last_odom is None else camera_delta(*odom_delta(last_odom, odom))
            last_odom = odom
            # a level camera: gravity is the camera's -y (as the standalone planner's gravity 'imu')
            pose_est.set_gravity(np.array([0., -1., 0.]))
            n0 = pose_est._n_passes
            pipe.process({'rgb': np.ascontiguousarray(m['frame']), 'timestamp': float(m['step']) * a.timestep,
                          'delta_pose': delta})
            fresh = pose_est._n_passes != n0 and pose_est.last_bev is not None
            bev = pose_est.last_bev if fresh else None
            t1 = time.perf_counter()
            goal = np.asarray(m['goal'], np.float32)
            N = policy.size
            if bev is not None:
                obs = bev['observability'] if bev.get('observability') is not None else np.ones((N, N), np.float16)
                err = bev['uncertainty'] if bev.get('uncertainty') is not None else np.zeros((N, N), np.float16)
                cat = {'prob': bev['prob'].astype(np.float32)[None], 'obs': obs.astype(np.float32)[None],
                       'err': err.astype(np.float32)[None], 'log_scale': np.array([bev['log_scale']], np.float32),
                       'odometry': np.array([bev.get('scale_source') == 'anchors']),
                       'up_head_deg': np.array([bev['gravity_head_deg']], np.float32),
                       'n_views': np.array([bev['n_views']]), 'fresh': np.array([True])}
                stats['head_scale'] += bev.get('scale_source') == 'head'
                stats['nav_only'] += bool(pose_est.last_info.get('nav_only'))
            else:                                 # no pass this frame: the episode map and the last route only
                stats['no_bev'] += 1
                cat = {'prob': np.full((1, N, N), .5, np.float32), 'obs': np.zeros((1, N, N), np.float32),
                       'err': np.zeros((1, N, N), np.float32), 'log_scale': np.array([np.nan], np.float32),
                       'odometry': np.array([False]), 'up_head_deg': np.zeros(1, np.float32), 'n_views': np.zeros(1),
                       'fresh': np.array([False])}
            ep = {'ep': uid, 'odom': odom, 'expect': m.get('expect')}
            paths, debug = policy.plan_maps(cat, goal[None], [state], [ep], [bool(m.get('debug'))])
            reply = {'path': paths[0], 'debug': debug[0],
                     'cross': {'seconds': t1 - t0, 'fresh': bool(fresh), 'n_views': int(bev['n_views']) if bev else 0,
                               'n_extra': int(bev.get('n_extra', 0)) if bev else 0,
                               'scale': bev.get('scale_source') if bev else None,
                               'pose': np.asarray(pipe.mapper.get_current_pose().tensor().detach().cpu().numpy()
                                                  if hasattr(pipe.mapper.get_current_pose(), 'tensor') else
                                                  pipe.mapper.get_current_pose(), np.float64).reshape(-1)[:7]}}
            if a.shadow and fresh:
                local = {}
                if near is None or near.shape[0] != N:      # cells within 5 m of the robot
                    yy, xx = np.mgrid[:N, :N]
                    near = np.hypot((xx + .5 - N / 2) * policy.res, (N / 2 - yy - .5) * policy.res) <= 5.
                for k, b in sorted(pose_est.last_bev_shadow.items()):
                    if b is None:
                        continue
                    o = b['observability'] if b.get('observability') is not None else np.ones((N, N), np.float16)
                    path, rec = plan_window(np.clip(b['prob'].astype(np.float32), 0, 1), np.clip(o.astype(np.float32), 0, 1),
                                            goal, policy.res, policy.cfg, None)
                    # obs_area: m^2 within 5 m predicted observed (P > 0.5)
                    local[str(k)] = {'path': path, 'status': int(rec['status']), 'n_views': int(b['n_views']),
                                     'n_extra': int(b.get('n_extra', 0)), 'scale': b.get('scale_source'),
                                     'obs_area': float((o.astype(np.float32)[near] > .5).sum() * policy.res ** 2)}
                reply['local'] = local
            stats['steps'] += 1
            stats['seconds'] += t1 - t0
            wire.send(conn, reply)
            if stats['steps'] % 500 == 0:
                print(json.dumps({**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()},
                                  'mean_step_s': round(stats['seconds'] / stats['steps'], 3),
                                  'wall_s': round(time.time() - t_start)}), flush=True)
    conn.close()


if __name__ == '__main__':
    main()
