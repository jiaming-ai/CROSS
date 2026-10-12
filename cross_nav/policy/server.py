"""GPU policy server for closed-loop simulation: one frozen VGGT pass per causal window, batched over workers.

    python -m cross_nav.policy.server --socket /tmp/nav0.sock --backbone ckpt.pt [--head best.pt] \
        [--record DIR] [--grid 12 16] [--scale mono|odometry] [--max-batch 32]

Workers (cross_nav.policy.sim.rollout) send the newest rendered frame of an episode and the indices of the causal
window; the server keeps each episode's recent frames, so a step transfers one image. With --head, it returns the
predicted (forward, left) path; with --record, it stores last-layer tokens (pooled to --grid), metric geometry, goal,
and the expert label sent by the worker, in shards of --shard windows.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from multiprocessing.connection import Listener, wait
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .backbone import FrozenMetricTokens, sha256
from .head import TrajectoryHead
from .sim import wire


class ShardWriter:
    def __init__(self, root, tag, size):
        self.root, self.tag, self.size = Path(root), tag, size
        self.root.mkdir(parents=True, exist_ok=True)
        self.k = len(list(self.root.glob(f'{tag}_*.npz')))
        self.buf = []

    def add(self, row):
        self.buf.append(row)
        if len(self.buf) >= self.size:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        name = self.root / f'{self.tag}_{self.k:05d}'
        tok = torch.stack([r.pop('tokens') for r in self.buf]).to(torch.bfloat16).view(torch.int16).numpy()   # bf16 bits
        np.save(str(name) + '.tok.tmp.npy', tok)
        small = {k: np.stack([r[k] for r in self.buf]) for k in ('xyz', 'log_scale', 'pred_chord', 'hist_chord', 'goal',
                                                                 'label', 'step', 'expert_dist')}
        small['meta'] = np.array([json.dumps(r['meta']) for r in self.buf])
        np.savez(str(name) + '.tmp.npz', **small)
        os.replace(str(name) + '.tok.tmp.npy', str(name) + '.tok.npy')
        os.replace(str(name) + '.tmp.npz', str(name) + '.npz')  # the .npz marks a complete shard
        self.k += 1
        self.buf = []


def odometry_or_mono(pred_chord, hist_chord, mono):
    """Median metric/predicted camera-centre baseline ratio when a baseline exceeds 0.2 m, else the scale head."""
    valid = (pred_chord > 1e-3) & (hist_chord > .2)
    ratio = torch.where(valid, hist_chord / pred_chord.clamp_min(1e-3), torch.nan)
    odo = ratio.nanmedian(dim=1).values.log()
    return torch.where(valid.any(1) & torch.isfinite(odo), odo, mono)


class Policy:
    def __init__(self, a):
        self.dev = torch.device('cuda')
        self.backbone = FrozenMetricTokens(a.backbone).to(self.dev).eval()
        self.grid = tuple(a.grid)
        self.head = None
        self.scale = a.scale
        if a.head:
            ck = torch.load(a.head, map_location='cpu', weights_only=True)
            self.head = TrajectoryHead(**ck['head_cfg']).to(self.dev).eval()
            self.head.load_state_dict(ck['model'])
            self.head_pool = int(ck['train_cfg'].get('token_pool', 1))
            self.steps = a.flow_steps
            self.gen = torch.Generator(device=self.dev).manual_seed(a.seed)
            self.head_sha = sha256(a.head)

    @torch.no_grad()
    def __call__(self, images, hist_chord):
        f = self.backbone(images.to(self.dev, non_blocking=True), pool=self.grid)
        tokens = f['tokens'][:, -1]                                          # B,S,h,w,C last layer
        out = {'tokens': tokens, 'xyz': f['xyz_unit'], 'log_scale': f['log_scale'], 'pred_chord': f['pred_history_chord']}
        return out

    @torch.no_grad()
    def predict(self, feat, goal, hist_chord):
        tok, xyz = feat['tokens'], feat['xyz']
        if self.head_pool > 1:
            k = self.head_pool
            B, S, h, w, C = tok.shape
            tok = F.avg_pool2d(tok.permute(0, 1, 4, 2, 3).reshape(B * S, C, h, w).float(), k).to(tok.dtype)
            tok = tok.reshape(B, S, C, h // k, w // k).permute(0, 1, 3, 4, 2)
            xyz = F.avg_pool2d(xyz.permute(0, 1, 4, 2, 3).reshape(B * S, 3, h, w), k).reshape(B, S, 3, h // k, w // k).permute(0, 1, 3, 4, 2)
        log_scale = feat['log_scale']
        if self.scale == 'odometry':
            log_scale = odometry_or_mono(feat['pred_chord'], hist_chord, log_scale)
        geometry = {'log_scale': log_scale, 'xyz_unit': xyz} if self.head.geometry != 'none' else None
        noise = torch.randn(goal.shape[0], self.head.horizon, 2, device=self.dev, generator=self.gen)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.head.predict(tok[:, None].contiguous(), goal, steps=self.steps, noise=noise, geometry=geometry).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--socket', required=True); p.add_argument('--backbone', required=True); p.add_argument('--head')
    p.add_argument('--bev', help='JSON config of the BEV head + planner policy (cross_nav.policy.bev_policy); replaces --head')
    p.add_argument('--record'); p.add_argument('--tag', default='rec'); p.add_argument('--shard', type=int, default=512)
    p.add_argument('--grid', type=int, nargs=2, default=(12, 16)); p.add_argument('--scale', choices=['mono', 'odometry'], default='mono')
    p.add_argument('--max-batch', type=int, default=32); p.add_argument('--flow-steps', type=int, default=10)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--workers', type=int, default=0,
                                                                  help='exit after this many workers have connected and left')
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    policy = Policy(a)
    bev = None
    if a.bev:
        from .bev_policy import BEVPlannerPolicy
        bev = BEVPlannerPolicy(json.loads(Path(a.bev).read_text()), backbone=policy.backbone)
    writer = ShardWriter(a.record, a.tag, a.shard) if a.record else None
    if os.path.exists(a.socket):
        os.unlink(a.socket)
    listener = Listener(a.socket, family='AF_UNIX')
    listener._listener._socket.setblocking(False)
    conns, frames, odoms, seen, done = [], {}, {}, 0, 0
    info = {'head': a.head, 'head_cfg': policy.head.cfg if policy.head else None, 'grid': list(policy.grid), 'scale': a.scale,
            'history_m': list(bev.history_m) if bev else [0., .5, 1., 2.], 'bev': a.bev,
            'history_rule': getattr(bev, 'history_rule', None) if bev else None}
    t0, n_windows, n_batches, t_bev = time.time(), 0, 0, {}
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))      # flush recorded shards, then exit
    print(json.dumps({'listening': a.socket, **info}), flush=True)
    while not stop:
        try:
            while True:
                c = listener.accept(); conns.append(c); seen += 1
        except (BlockingIOError, OSError):
            pass
        if not conns:
            if a.workers and seen >= a.workers:
                break
            time.sleep(.01); continue
        ready = wait(conns, timeout=.05)
        batch = []
        for c in ready:
            try:
                m = wire.recv(c)
            except (EOFError, OSError):
                conns.remove(c); done += 1; continue
            cmd = m['cmd']
            if cmd == 'info':
                wire.send(c, info)
            elif cmd == 'end':
                frames.pop(m['ep'], None); odoms.pop(m['ep'], None)
                if bev is not None:
                    bev.end_episode(m['ep'])
                wire.send(c, {})
            elif cmd == 'step':
                buf = frames.setdefault(m['ep'], {})
                buf[m['frame_idx']] = m['frame']
                for k in [k for k in buf if k < min(m['window'])]:
                    del buf[k]
                if m.get('odom') is not None:     # odometry pose (x, y, yaw) of each frame, for the BEV policy's rules
                    ob = odoms.setdefault(m['ep'], {})
                    ob[m['frame_idx']] = np.asarray(m['odom'], np.float64)
                    for k in [k for k in ob if k < min(m['window'])]:
                        del ob[k]
                if m.get('predict') or (m.get('record') and writer is not None):
                    batch.append((c, m))
                else:
                    wire.send(c, {'path': None, 'debug': None})   # expert step, not recorded: no backbone pass
        for i in range(0, len(batch), a.max_batch):
            part = batch[i:i + a.max_batch]
            imgs = np.stack([np.stack([frames[m['ep']][j] for j in m['window']]) for _, m in part])
            imgs = torch.from_numpy(imgs).pin_memory().to('cuda', non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255)
            hist = torch.from_numpy(np.stack([m['hist_chord'] for _, m in part]).astype(np.float32)).cuda()
            goal = torch.from_numpy(np.stack([m['goal'] for _, m in part]).astype(np.float32)).cuda()
            debug = [None] * len(part)
            if bev is not None:
                feat = None
                eps = []
                for _, m in part:
                    ob = odoms.get(m['ep'], {})
                    wo = np.stack([ob[j] for j in m['window']]) if all(j in ob for j in m['window']) else None
                    eps.append({'ep': m['ep'], 'step': m.get('step'), 'odom': ob.get(m['frame_idx']), 'window_odom': wo})
                paths, debug = bev(imgs, goal, hist, want_debug=[bool(m.get('debug')) for _, m in part], episodes=eps)
                for k, v in bev.timing.items():
                    t_bev[k] = t_bev.get(k, 0.) + v
            else:
                feat = policy(imgs, hist)
                paths = policy.predict(feat, goal, hist).cpu().numpy() if policy.head is not None and any(m['predict'] for _, m in part) else None
            if writer is not None and feat is not None:
                cpu = {k: v.cpu() for k, v in feat.items()}
                for j, (_, m) in enumerate(part):
                    if m.get('record') and m.get('label') is not None:
                        writer.add({'tokens': cpu['tokens'][j].to(torch.bfloat16), 'xyz': cpu['xyz'][j].half().numpy(),
                                    'log_scale': cpu['log_scale'][j].float().numpy(), 'pred_chord': cpu['pred_chord'][j].float().numpy(),
                                    'hist_chord': m['hist_chord'].astype(np.float32), 'goal': m['goal'].astype(np.float32),
                                    'label': m['label'].astype(np.float32), 'step': np.int32(m['step']),
                                    'expert_dist': np.float32(m.get('expert_dist', np.nan)), 'meta': m['meta']})
            for j, (c, m) in enumerate(part):
                wire.send(c, {'path': paths[j] if paths is not None else None, 'debug': debug[j]})
            n_windows += len(part); n_batches += 1
            if n_batches % 200 == 0:
                print(json.dumps({'windows': n_windows, 'batches': n_batches, 'mean_batch': round(n_windows / n_batches, 1),
                                  'windows_per_s': round(n_windows / (time.time() - t0), 1), 'clients': len(conns),
                                  **{k: round(v / n_windows, 3) for k, v in t_bev.items()}}), flush=True)
    if writer is not None:
        writer.flush()
    print(json.dumps({'exit': True, 'windows': n_windows, 'seconds': round(time.time() - t0)}), flush=True)


if __name__ == '__main__':
    main()
