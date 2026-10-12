"""Closed-loop point-goal rollouts: expert data collection, policy evaluation and DAgger (habitat environment).

    python -m cross_nav.policy.sim.rollout --hssd ROOT --episodes eps.json --socket /tmp/nav0.sock \
        --mode expert|policy|dagger --out results.jsonl [--shard i n] [--record] [--beta 0.3] [--gpu 1]

Each step renders the current view, sends it with the causal window (frames at 0 / 0.5 / 1 / 2 m of past travel),
the robot-frame goal and the expert label (navmesh shortest path from the current position, 16 points at 0.5 m) to
the policy server, then executes one command of the waypoint controller on the followed path:

* expert  the expert path, with per-episode DART noise (turn/forward noise, random excursions); labels stay clean
* policy  the predicted path
* dagger  the predicted path, the expert path with probability beta per step

The robot slides along the navmesh boundary (Habitat's try_step); a step whose result differs from the commanded one
by more than 2 cm counts as a collision. Success: within 0.2 m of the goal (the goal vector is known, so arrival
stops the robot) within --max-steps. Results: one JSON line per episode (success, SPL, collisions, path).
"""
from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from . import wire
from .geom import control, resample, spl, to_robot, trail, window_ids
from .hab import camera_sample, dataset_name, expert_path, geodesic, make_sim, render, set_hfov

SUCCESS_M = .2


def noise_profile(rng, mode):
    if mode != 'expert':
        return {'turn': 0., 'fwd': 0., 'excursion': 0.}
    level = rng.choice([0., .1, .2, .35])
    return {'turn': float(level), 'fwd': float(level * .3), 'excursion': float(rng.choice([0., .02, .05]))}


def odom_pose(pos, yaw):
    """Odometry pose (x, y, yaw) of a robot at Habitat position pos (Y-up) and heading yaw: x forward / y left at
    yaw 0, yaw counter-clockwise (REP-103), i.e. x = -z, y = -x of the world, as cross_nav.policy.robot expects."""
    return np.array([-pos[2], -pos[0], yaw], np.float64)


def run_episode(sim, conn, ep, a, rng, viz=None):
    pf = sim.pathfinder
    pos = np.asarray(ep['start'], np.float64)
    yaw = float(ep['yaw'])
    goal = np.asarray(ep['goal'], np.float64)
    uid = f"{a.tag}:{ep['id']}"
    noise = noise_profile(rng, a.mode)
    camera = camera_sample(rng, a.camera)
    set_hfov(sim, camera['hfov'])
    frames_cum, yaws, cams, plane, traj = [], [], [], [], [pos[[0, 2]].tolist()]
    cum, travelled, collisions, excursion = 0., 0., 0, 0
    success, steps, min_geo = False, 0, ep['geodesic']
    for step in range(a.max_steps):
        rgb = np.ascontiguousarray(render(sim, pos, yaw, camera=camera))
        frames_cum.append(cum); cams.append(pos + np.array([0., camera['height'], 0.]))
        yaws.append(yaw); plane.append(pos[[0, 2]].copy())
        ids = window_ids(frames_cum, yaws, a.history_rule)
        goal_rel = to_robot(goal[[0, 2]], pos[[0, 2]], yaw)
        d_geo, pts = expert_path(sim, pos, goal)
        label = None
        if pts is not None:
            label = resample(to_robot(pts[:, [0, 2]], pos[[0, 2]], yaw), a.horizon, a.spacing)
            min_geo = min(min_geo, d_geo)
        use_expert = a.mode == 'expert' or (a.mode == 'dagger' and rng.random() < a.beta)
        record = a.record and label is not None and rng.random() < a.record_prob
        hist = np.array([np.linalg.norm(cams[-1] - cams[i]) for i in ids], np.float32)
        msg = {'cmd': 'step', 'ep': uid, 'frame_idx': len(frames_cum) - 1, 'frame': rgb, 'window': ids,
               'goal': goal_rel.astype(np.float32), 'hist_chord': hist, 'trail': trail(frames_cum, plane, pos[[0, 2]], yaw), 'label': None if label is None else label.astype(np.float32),
               'record': record, 'predict': not use_expert, 'step': step, 'expert_dist': d_geo, 'debug': viz is not None,
               'odom': odom_pose(pos, yaw),
               'meta': {'ep': ep['id'], 'scene': ep['scene'], 'mode': a.mode, 'tag': a.tag, 'expert_step': bool(use_expert),
                        'camera': camera, 'dataset': dataset_name(a.hssd)}}
        wire.send(conn, msg)
        reply = wire.recv(conn)
        if use_expert:
            if label is None:
                break
            path = label
        else:
            path = reply['path']
        if viz is not None:
            viz.append({'rgb': rgb, 'pos': pos.copy(), 'yaw': yaw, 'goal_rel': goal_rel, 'path': np.asarray(path, np.float32),
                        'expert': label, 'window': ids, 'expert_step': bool(use_expert), 'debug': reply.get('debug')})
        turn, fwd = control(path, a.lookahead, a.step_m)
        if a.mode == 'expert':
            if excursion == 0 and noise['excursion'] and rng.random() < noise['excursion']:
                # off-route excursion: one sharp turn, then a few straight steps; the labels keep the expert path
                excursion, turn = int(rng.integers(3, 9)), float(rng.choice([-1, 1]) * rng.uniform(.3, 1.2))
                fwd = 0.
            elif excursion > 0:
                excursion -= 1
                turn, fwd = 0., a.step_m
            turn += rng.normal(0, noise['turn']); fwd = max(0., fwd + rng.normal(0, noise['fwd']) * a.step_m)
        yaw = (yaw + turn + math.pi) % (2 * math.pi) - math.pi
        if fwd > 0:
            f = np.array([-math.sin(yaw), 0., -math.cos(yaw)])
            want = pos + fwd * f
            new = np.asarray(pf.try_step(pos.astype(np.float32), want.astype(np.float32)), np.float64)
            if np.linalg.norm((new - want)[[0, 2]]) > .02:
                collisions += 1
            moved = float(np.linalg.norm((new - pos)[[0, 2]]))
            travelled += moved; cum += moved; pos = new
            traj.append(pos[[0, 2]].round(3).tolist())
        steps = step + 1
        if np.linalg.norm((goal - pos)[[0, 2]]) < SUCCESS_M and abs(goal[1] - pos[1]) < .5:
            success = True
            break
    wire.send(conn, {'cmd': 'end', 'ep': uid}); wire.recv(conn)
    final_geo, _ = geodesic(pf, pos, goal)
    return {'id': ep['id'], 'scene': ep['scene'], 'geodesic': ep['geodesic'], 'euclidean': ep['euclidean'], 'success': success,
            'spl': spl(success, ep['geodesic'], travelled), 'steps': steps, 'travelled': travelled, 'collisions': collisions,
            'collision_free': success and collisions == 0, 'final_geo': final_geo, 'min_geo': min_geo, 'mode': a.mode,
            'noise': noise, 'camera': camera, 'trajectory': traj[::2] + [traj[-1]]}


class SocketPlanner:
    """cross_nav.policy.robot planner interface over the policy server's wire protocol, so --control robot runs the
    robot's own NavigationAgent. Window frames the server does not hold yet are uploaded first as non-predicting step
    messages (frames are recognised by content), then the current frame asks for a path."""
    def __init__(self, conn, uid, info, meta):
        self.conn, self.uid, self.meta = conn, uid, meta
        self.history_m = tuple(info['history_m'])
        self.history_rule = info.get('history_rule') or {'type': 'distance', 'm': list(info['history_m'])}
        self.uses_odometry = bool(info.get('bev'))
        self.sent, self.next_idx, self.calls, self.debug, self.last_debug = {}, 0, 0, False, None
        self.on_server = set()       # keys of frames the server holds (it drops those older than the latest window)

    def _send(self, msg):
        wire.send(self.conn, msg)
        return wire.recv(self.conn)

    def __call__(self, images, goal, hist, trail=None, odom=None, window_odom=None, expect=None):
        keys = [zlib.crc32(np.ascontiguousarray(im).data) for im in images]
        for key in keys:
            if key not in self.sent:
                self.sent[key] = self.next_idx; self.next_idx += 1
        window = [self.sent[k] for k in keys]
        base = {'cmd': 'step', 'ep': self.uid, 'window': window, 'label': None, 'record': False, 'step': self.calls,
                'meta': self.meta}
        for j in range(len(images) - 1, 0, -1):          # older window frames the server has not seen, oldest first
            if keys[j] not in keys[:j] and keys[j] not in self.on_server:
                self._send({**base, 'frame_idx': window[j], 'frame': images[j], 'predict': False, 'debug': False,
                            'odom': None if window_odom is None else window_odom[j]})
                self.on_server.add(keys[j])
        reply = self._send({**base, 'frame_idx': window[0], 'frame': images[0], 'predict': True, 'goal': goal,
                            'hist_chord': hist, 'trail': trail if trail is not None else np.zeros((5, 3), np.float32),
                            'odom': odom, 'expect': expect, 'debug': self.debug, 'expert_dist': np.nan,
                            'recover': False})                   # the agent applies its own stuck recovery
        self.on_server.add(keys[0])
        lo = min(window)                                   # the server drops frames older than the window
        self.on_server = {k for k in self.on_server if self.sent[k] >= lo}
        self.calls += 1
        self.last_debug = reply.get('debug')
        return reply['path']

    def end(self):
        self._send({'cmd': 'end', 'ep': self.uid})


def odometry_step(odom, delta, turn, rng, noise, bias):
    """Dead-reckoned odometry pose after a planar displacement delta = (forward, left) metres in the previous robot
    frame (sliding included) and a turn of turn radians; with noise > 0, the displacement and the turn are scaled by
    (1 + per-episode bias + per-tick N(0, noise)). Noise-free odometry is the ground-truth pose."""
    delta = np.asarray(delta, np.float64)
    if noise > 0:
        delta = delta * (1. + bias[0] + rng.normal(0, noise))
        turn *= 1. + bias[1] + rng.normal(0, noise)
    x, y, th = odom
    c, s = math.cos(th), math.sin(th)
    return np.array([x + delta[0] * c - delta[1] * s, y + delta[0] * s + delta[1] * c,
                     (th + turn + math.pi) % (2 * math.pi) - math.pi])


def run_episode_robot(sim, conn, ep, a, rng, viz=None):
    """Camera-rate closed loop through cross_nav.policy.robot.NavigationAgent: every 1/rate s render a frame, give it
    with the odometry pose to the agent (it keeps frames, picks the window, plans every replan_every frames and returns
    a pure-pursuit (v, omega)), and integrate (v, omega) for 1/rate s with navmesh sliding. A tick whose result
    differs from the commanded motion by more than 2 cm x (commanded / 0.25 m) is a contact (the step controller's
    2 cm per 0.25 m step, scaled to the tick); contact_steps = commanded distance in contact / 0.25 m is comparable
    with the step controller's contact count. Time limit: a.time_limit seconds."""
    from ..robot import NavigationAgent
    pf = sim.pathfinder
    pos = np.asarray(ep['start'], np.float64)
    yaw = float(ep['yaw'])
    goal = np.asarray(ep['goal'], np.float64)
    uid = f"{a.tag}:{ep['id']}"
    camera = camera_sample(rng, a.camera)
    set_hfov(sim, camera['hfov'])
    meta = {'ep': ep['id'], 'scene': ep['scene'], 'mode': a.mode, 'tag': a.tag, 'expert_step': False, 'camera': camera,
            'dataset': dataset_name(a.hssd)}
    planner = SocketPlanner(conn, uid, a.info, meta)
    agent = NavigationAgent(planner, arrive_m=SUCCESS_M, v_max=a.v_max, w_max=a.w_max, lookahead=a.lookahead,
                            replan_every=a.replan_every)
    odom = odom_pose(pos, yaw)
    agent.set_goal(odom_pose(goal, 0.)[:2])          # the goal in the odometry frame (exact at the start)
    bias = rng.normal(0, a.odom_noise, 2) if a.odom_noise > 0 else np.zeros(2)
    dt = 1. / a.rate
    traj, travelled, collisions, success, ticks, plans = [pos[[0, 2]].tolist()], 0., 0, False, 0, 0
    contact_m = 0.                                         # commanded forward motion during contact ticks
    for tick in range(int(round(a.time_limit * a.rate))):
        rgb = np.ascontiguousarray(render(sim, pos, yaw, camera=camera))
        planner.debug = viz is not None
        out = agent.step(rgb, tuple(odom), tick * dt if a.stamps else None)
        ticks = tick + 1
        if out['arrived']:
            break
        plans += bool(out.get('replanned'))
        if viz is not None and out.get('replanned'):
            viz.append({'rgb': rgb, 'pos': pos.copy(), 'yaw': yaw, 'goal_rel': to_robot(goal[[0, 2]], pos[[0, 2]], yaw),
                        'path': np.asarray(out['path'], np.float32), 'expert': None, 'window': out['window'],
                        'expert_step': False, 'debug': planner.last_debug})
        v, w = out['cmd']
        turn, fwd = w * dt, v * dt
        mid = yaw + turn / 2
        new = pos
        if fwd > 0:
            want = pos + fwd * np.array([-math.sin(mid), 0., -math.cos(mid)])
            new = np.asarray(pf.try_step(pos.astype(np.float32), want.astype(np.float32)), np.float64)
            if np.linalg.norm((new - want)[[0, 2]]) > .02 * fwd / a.step_m:
                collisions += 1; contact_m += fwd
        moved = float(np.linalg.norm((new - pos)[[0, 2]]))
        odom = odometry_step(odom, to_robot((new - pos)[[0, 2]], np.zeros(2), yaw), turn, rng, a.odom_noise, bias)
        travelled += moved; pos = new
        yaw = (yaw + turn + math.pi) % (2 * math.pi) - math.pi
        if moved > 0 and tick % 5 == 0:
            traj.append(pos[[0, 2]].round(3).tolist())
        if np.linalg.norm((goal - pos)[[0, 2]]) < SUCCESS_M and abs(goal[1] - pos[1]) < .5:
            success = True
            break
    planner.end()
    traj.append(pos[[0, 2]].round(3).tolist())
    final_geo, _ = geodesic(pf, pos, goal)
    return {'id': ep['id'], 'scene': ep['scene'], 'geodesic': ep['geodesic'], 'euclidean': ep['euclidean'], 'success': success,
            'spl': spl(success, ep['geodesic'], travelled), 'steps': ticks, 'seconds': ticks * dt, 'plans': plans,
            'travelled': travelled, 'collisions': collisions, 'contact_s': collisions * dt,
            'contact_steps': contact_m / a.step_m,
            'collision_free': success and collisions == 0, 'final_geo': final_geo, 'mode': a.mode, 'control': 'robot',
            'rate': a.rate, 'replan_every': a.replan_every, 'odom_noise': a.odom_noise,
            'odom_error_m': float(np.hypot(*(odom[:2] - odom_pose(pos, yaw)[:2]))), 'camera': camera, 'trajectory': traj}


def save_viz(path, ep, result, steps, sim=None):
    """Per-step record of one episode for the viewer: JPEG frames, poses, followed / expert paths, policy debug maps."""
    import io
    from PIL import Image
    jpg = np.empty(len(steps), dtype=object)   # filled one by one: equal-length byte arrays must not stack to 2-D
    for i, s in enumerate(steps):
        b = io.BytesIO(); Image.fromarray(s['rgb']).save(b, 'JPEG', quality=85); jpg[i] = np.frombuffer(b.getvalue(), np.uint8)
    debug = {}
    first = next((s['debug'] for s in steps if s['debug']), None) or {}
    for k, v0 in first.items():   # steps without debug (DAgger expert steps): NaN, -1 or False
        v0 = np.asarray(v0)
        fill = np.full(v0.shape, np.nan if v0.dtype.kind == 'f' else -1 if v0.dtype.kind == 'i' else 0, v0.dtype)
        debug['debug_' + k] = np.stack([np.asarray(s['debug'][k]) if s['debug'] else fill for s in steps])
    np.savez_compressed(path, jpg=np.array(jpg, dtype=object), pos=np.stack([s['pos'] for s in steps]),
                        yaw=np.array([s['yaw'] for s in steps]), goal_rel=np.stack([s['goal_rel'] for s in steps]),
                        path=np.stack([s['path'] for s in steps]),
                        expert=np.stack([s['expert'] if s['expert'] is not None else np.full_like(s['path'], np.nan) for s in steps]),
                        window=np.array([s['window'] for s in steps]), expert_step=np.array([s['expert_step'] for s in steps]),
                        goal=np.asarray(ep['goal']), start=np.asarray(ep['start']), result=json.dumps(result), **debug, **topdown(sim, ep))


def topdown(sim, ep, mpp=.05):
    """Navigable top-down map at the start height: rows follow +z from bounds lower z, columns +x from lower x."""
    if sim is None:
        return {}
    lo, _ = sim.pathfinder.get_bounds()
    grid = np.asarray(sim.pathfinder.get_topdown_view(mpp, float(ep['start'][1])), dtype=bool)
    return {'topdown': np.packbits(grid, axis=1), 'topdown_shape': np.array(grid.shape), 'topdown_origin_xz': np.array([lo[0], lo[2]]),
            'topdown_mpp': np.float32(mpp)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hssd', required=True); p.add_argument('--episodes', required=True); p.add_argument('--socket', required=True)
    p.add_argument('--mode', choices=['expert', 'policy', 'dagger'], required=True); p.add_argument('--out', required=True)
    p.add_argument('--shard', type=int, nargs=2, default=(0, 1)); p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--navmesh-dir'); p.add_argument('--record', action='store_true'); p.add_argument('--record-prob', type=float, default=1.)
    p.add_argument('--beta', type=float, default=.3); p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--horizon', type=int, default=16); p.add_argument('--spacing', type=float, default=.5)
    p.add_argument('--lookahead', type=float, default=.4); p.add_argument('--step-m', type=float, default=.25)
    p.add_argument('--tag', default='run'); p.add_argument('--seed', type=int, default=0); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--camera', choices=['nominal', 'random'], default='nominal')
    p.add_argument('--viz-dir'); p.add_argument('--viz-every', type=int, default=0, help='save every k-th episode for the viewer')
    p.add_argument('--control', choices=['step', 'robot'], default='step',
                   help='step: one waypoint-controller command per frame; robot: camera-rate (v, omega) loop of '
                        'cross_nav.policy.robot.NavigationAgent (policy mode only)')
    p.add_argument('--rate', type=float, default=10., help='robot control: camera / control rate (Hz)')
    p.add_argument('--replan-every', type=int, default=1, help='robot control: plan on every k-th frame')
    p.add_argument('--v-max', type=float, default=.5); p.add_argument('--w-max', type=float, default=1.)
    p.add_argument('--time-limit', type=float, default=0., help='robot control: seconds (default 2 x max_steps x step_m / v_max)')
    p.add_argument('--odom-noise', type=float, default=0., help='robot control: relative odometry noise (bias + per tick)')
    p.add_argument('--stamps', action='store_true', help='robot control: give the agent frame times (commanded motion '
                                                         'for the BEV bump rule)')
    a = p.parse_args()
    if a.control == 'robot':
        if a.mode != 'policy':
            raise SystemExit('--control robot supports --mode policy only')
        a.time_limit = a.time_limit or 2 * a.max_steps * a.step_m / a.v_max
    eps = json.loads(Path(a.episodes).read_text())['episodes']
    i, n = a.shard
    eps = [e for k, e in enumerate(eps) if k % n == i]
    if a.limit:
        eps = eps[:a.limit]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)['id'] for l in out.read_text().splitlines() if l.strip()}
    eps = sorted((e for e in eps if e['id'] not in done), key=lambda e: e['scene'])
    conn = Client(a.socket, family='AF_UNIX')
    wire.send(conn, {'cmd': 'info'})
    info = wire.recv(conn)
    a.history_rule = info.get('history_rule') or {'type': 'distance', 'm': list(info['history_m'])}
    a.info = info
    sim, scene, t0 = None, None, time.time()
    with out.open('a') as f:
        for k, ep in enumerate(eps):
            if ep['scene'] != scene:
                if sim is not None:
                    sim.close()
                sim, scene = make_sim(a.hssd, ep['scene'], a.gpu, a.navmesh_dir), ep['scene']
            rng = np.random.default_rng((zlib.crc32(f"{a.tag}:{ep['id']}".encode()) + a.seed) % 2**31)
            viz = [] if a.viz_dir and a.viz_every and zlib.crc32(ep['id'].encode()) % a.viz_every == 0 else None
            r = (run_episode_robot if a.control == 'robot' else run_episode)(sim, conn, ep, a, rng, viz)
            if viz:
                Path(a.viz_dir).mkdir(parents=True, exist_ok=True)
                save_viz(Path(a.viz_dir) / f"{ep['id'].replace(':', '_')}.npz", ep, r, viz, sim)
            f.write(json.dumps(r) + '\n'); f.flush()
            if k % 10 == 0:
                print(json.dumps({'done': k + 1, 'of': len(eps), 'seconds': round(time.time() - t0), 'success': r['success']}), flush=True)
    if sim is not None:
        sim.close()
    conn.close()


if __name__ == '__main__':
    main()
