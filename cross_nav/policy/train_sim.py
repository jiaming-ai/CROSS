"""Train the trajectory head on recorded closed-loop windows (cross_nav.policy.server --record shards).

    python -m cross_nav.policy.train_sim --train DIR [DIR ...] --val DIR [DIR ...] --out RUN \
        [--mode regression|flow] [--geometry none|scalar|metric] [--token-pool 1|2] [--steps 20000] [--batch 128]

Shards hold last-layer VGGT tokens (N,S,h,w,C bf16), metric geometry, the robot-frame goal and the expert label (navmesh
shortest path from the visited state, so DAgger states carry corrective labels). Checkpoints are selected on offline
validation ADE of held-out scenes; the decisive metric is closed-loop success (cross_nav.policy.sim.rollout).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .head import TrajectoryHead
from .server import odometry_or_mono

SMALL = ('xyz', 'log_scale', 'pred_chord', 'hist_chord', 'goal', 'label')


def shard_names(dirs):
    names = []
    for d in dirs:
        names += sorted(str(p)[:-4] for p in Path(d).glob('*.npz') if Path(str(p)[:-4] + '.tok.npy').exists())
    if not names:
        raise ValueError(f'no shards in {dirs}')
    return names


def pool_tokens(tok, k):
    """tok N,S,h,w,C -> k x k average pooling of the grid (float32 math, input dtype kept)."""
    if k == 1:
        return tok
    N, S, h, w, C = tok.shape
    x = F.avg_pool2d(tok.permute(0, 1, 4, 2, 3).reshape(N * S, C, h, w).float(), k)
    return x.reshape(N, S, C, h // k, w // k).permute(0, 1, 3, 4, 2).to(tok.dtype)


def load(dirs, pool, device, max_windows=0, stream=False, token_device=None):
    """Windows of the shards. Small arrays live on `device`; tokens are pooled and resident on `device`, or, with
    stream=True, left in memory-mapped shards (TokenRows) for datasets larger than the GPU."""
    names = shard_names(dirs)
    toks, small, meta, rows = [], {k: [] for k in SMALL}, [], []
    n = 0
    for k, name in enumerate(names):
        z = np.load(name + '.npz')
        m = np.load(name + '.tok.npy', mmap_mode='r')
        if stream:
            rows += [(k, i) for i in range(len(m))]
        else:
            t = torch.from_numpy(m[:]).view(torch.bfloat16)
            for i in range(0, len(t), 256):
                toks.append(pool_tokens(t[i:i + 256].to(device), pool).to(token_device or device))
        for key in SMALL:
            small[key].append(torch.from_numpy(z[key].astype(np.float32)))
        small.setdefault('trail', []).append(torch.from_numpy(z['trail'].astype(np.float32)) if 'trail' in z.files
                                              else torch.zeros(len(m), 5, 3))       # older shards: trail unavailable
        meta += [json.loads(x) for x in z['meta']]
        n += len(m)
        if max_windows and n >= max_windows:
            break
    X = TokenRows(names, rows) if stream else torch.cat(toks)
    S = {k: torch.cat(v).to(device) for k, v in small.items()}
    if pool > 1:
        N, V, h, w, _ = S['xyz'].shape
        S['xyz'] = F.avg_pool2d(S['xyz'].permute(0, 1, 4, 2, 3).reshape(N * V, 3, h, w), pool).reshape(N, V, 3, h // pool, w // pool).permute(0, 1, 3, 4, 2)
    return X, S, meta, names


class TokenRows(torch.utils.data.Dataset):
    """Token rows read from memory-mapped shards (opened lazily in each loader worker)."""
    def __init__(self, names, rows):
        self.names, self.rows, self.maps = names, rows, {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        k, r = self.rows[i]
        if k not in self.maps:
            self.maps[k] = np.load(self.names[k] + '.tok.npy', mmap_mode='r')
        return torch.from_numpy(np.array(self.maps[k][r])), i


def sample_weights(meta, weights, default):
    """Per-window sampling probability: each dataset's weight split evenly over its windows."""
    ds = [m.get('dataset', 'hssd_sim') for m in meta]
    names = sorted(set(ds))
    w = {d: weights.get(d, default) for d in names}
    total = sum(w.values())
    count = {d: ds.count(d) for d in names}
    return torch.tensor([w[d] / total / count[d] for d in ds], dtype=torch.double), {d: round(w[d] / total, 4) for d in names}, count


def geometry_of(S, idx, how, rng_noise=0.):
    log_s = S['log_scale'][idx]
    if how == 'odometry':
        log_s = odometry_or_mono(S['pred_chord'][idx], S['hist_chord'][idx], log_s)
    elif how == 'mixed':
        odo = odometry_or_mono(S['pred_chord'][idx], S['hist_chord'][idx], log_s)
        log_s = torch.where(torch.rand_like(log_s) < .5, odo, log_s)
    if rng_noise:
        log_s = log_s + rng_noise * torch.randn_like(log_s)
    return {'log_scale': log_s, 'xyz_unit': S['xyz'][idx]}


@torch.no_grad()
def evaluate(model, X, S, meta, scale, batch=256):
    model.eval()
    dev = S['goal'].device
    gen = torch.Generator(device=dev).manual_seed(2026)
    preds = []
    for i in range(0, len(X), batch):
        idx = torch.arange(i, min(i + batch, len(X)), device=dev)
        noise = torch.randn(len(idx), model.horizon, 2, device=dev, generator=gen)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            preds.append(model.predict(X[i:i + len(idx)].to(dev, non_blocking=True)[:, None], S['goal'][idx], noise=noise,
                                       geometry=geometry_of(S, idx, scale) if model.geometry != 'none' else None,
                                       trail=S['trail'][idx] if model.trail else None).float())
    pred = torch.cat(preds)
    err = (pred - S['label']).norm(dim=-1)
    dagger = torch.tensor([not m.get('expert_step', True) for m in meta], device=dev)
    out = {'n': len(X), 'ADE': float(err.mean()), 'FDE': float(err[:, -1].mean()), 'E1m': float(err[:, 1].mean()),
           'E2m': float(err[:, 3].mean())}
    if dagger.any():
        out['ADE_policy_states'] = float(err[dagger].mean())
    ds = np.array([m.get('dataset', 'hssd_sim') for m in meta])
    out['by_dataset'] = {str(d): round(float(err[torch.from_numpy(ds == d).to(err.device)].mean()), 4) for d in sorted(set(ds))}
    out['macro_ADE'] = float(np.mean([float(v) for v in out['by_dataset'].values()]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', nargs='+', required=True); p.add_argument('--val', nargs='+', required=True); p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=['regression', 'flow'], default='regression')
    p.add_argument('--geometry', choices=['none', 'scalar', 'metric'], default='none')
    p.add_argument('--scale', choices=['mono', 'odometry', 'mixed'], default='mixed')
    p.add_argument('--token-pool', type=int, default=1); p.add_argument('--width', type=int, default=256)
    p.add_argument('--layers', type=int, default=4); p.add_argument('--heads', type=int, default=8)
    p.add_argument('--compression', type=int, default=0); p.add_argument('--goal-encoding', default='polar')
    p.add_argument('--steps', type=int, default=20000); p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--wd', type=float, default=.01)
    p.add_argument('--seed', type=int, default=17); p.add_argument('--init'); p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--max-train', type=int, default=0); p.add_argument('--token-dropout', type=float, default=0.)
    p.add_argument('--trail', action='store_true', help='odometry trail input (past positions at 1-16 m of travel)')
    p.add_argument('--trail-dropout', type=float, default=.3)
    p.add_argument('--stream', action='store_true', help='read training tokens from memory-mapped shards (datasets larger than the GPU)')
    p.add_argument('--weights', nargs='*', default=[], help='dataset=weight sampling weights (meta dataset names)')
    p.add_argument('--default-weight', type=float, default=1.); p.add_argument('--loader-workers', type=int, default=12)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = vars(a); (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    t0 = time.time()
    X, S, meta, names = load(a.train, a.token_pool, 'cuda', a.max_train, a.stream)
    Xv, Sv, metav, vnames = load(a.val, a.token_pool, 'cuda', token_device='cpu')   # host memory: shared GPUs
    train_scenes, val_scenes = {m['scene'] for m in meta}, {m['scene'] for m in metav}
    assert not train_scenes & val_scenes, 'scene leakage between train and val'
    horizon = S['label'].shape[1]
    token_dim = np.load(names[0] + '.tok.npy', mmap_mode='r').shape[-1]
    probs, mix, counts = sample_weights(meta, dict((k, float(v)) for k, v in (w.split('=') for w in a.weights)), a.default_weight)
    model = TrajectoryHead(token_dim=token_dim, width=a.width, layers=a.layers, heads=a.heads, horizon=horizon, source='last',
                           compression=a.compression, mode=a.mode, geometry=a.geometry, action_scale=horizon * .5,
                           goal_encoding=a.goal_encoding, trail=5 if a.trail else 0).cuda()
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location='cpu', weights_only=True)['model'])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    prov = {'train_windows': len(X), 'val_windows': len(Xv), 'train_scenes': len(train_scenes), 'val_scenes': len(val_scenes),
            'train_shards': names, 'val_shards': vnames, 'mix': mix, 'windows_per_dataset': counts,
            'parameters': sum(q.numel() for q in model.parameters()),
            'load_seconds': round(time.time() - t0)}
    (out / 'provenance.json').write_text(json.dumps(prov, indent=2))
    print(json.dumps({k: v for k, v in prov.items() if 'shards' not in k}), flush=True)
    best, log = math.inf, []
    sampler = torch.utils.data.WeightedRandomSampler(probs, a.batch * (a.steps + 1), replacement=True,
                                                     generator=torch.Generator().manual_seed(a.seed))
    if a.stream:
        batches = iter(torch.utils.data.DataLoader(X, batch_size=a.batch, sampler=sampler, num_workers=a.loader_workers,
                                                   pin_memory=True, persistent_workers=True, prefetch_factor=4))
    else:
        order = torch.tensor(list(sampler), device='cuda').view(-1, a.batch)
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(model, Xv, Sv, metav, 'odometry' if a.scale != 'mono' else 'mono')
            r.update(step=step, seconds=round(time.time() - t0)); log.append(r)
            print(json.dumps(r), flush=True)
            (out / 'validation.json').write_text(json.dumps(log, indent=2))
            ck = {'model': model.state_dict(), 'head_cfg': model.cfg, 'step': step, 'validation': r,
                  'train_cfg': {**cfg, 'raw': False}, 'provenance': {k: v for k, v in prov.items() if 'shards' not in k}}
            if r['macro_ADE'] < best:
                best = r['macro_ADE']; torch.save(ck, out / 'best.pt')
            if step == a.steps:
                torch.save(ck, out / 'final.pt'); break
        model.train()
        if a.stream:
            tok, idx = next(batches)
            idx = idx.cuda(non_blocking=True)
            tok = pool_tokens(tok.cuda(non_blocking=True).view(torch.bfloat16), a.token_pool)
        else:
            idx = order[step]
            tok = X[idx]
        tok = tok[:, None]
        lr = a.lr * min(1., (step + 1) / 500) * .5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g['lr'] = lr
        if a.token_dropout:
            keep = (torch.rand(tok.shape[:-1], device='cuda') > a.token_dropout).to(tok.dtype)[..., None]
            tok = tok * keep
        geom = geometry_of(S, idx, a.scale, .1) if a.geometry != 'none' else None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tr = None
            if a.trail:
                tr = S['trail'][idx].clone()
                tr[torch.rand(len(idx), device='cuda') < a.trail_dropout] = 0.      # robust to a missing trail
            loss = model.loss(tok, S['goal'][idx], S['label'][idx], geometry=geom, trail=tr)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss at {step}')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        if (step + 1) % 200 == 0:
            print(json.dumps({'step': step + 1, 'loss': round(float(loss), 5), 'lr': lr, 'seconds': round(time.time() - t0)}), flush=True)
    (out / 'DONE.json').write_text(json.dumps({'best_macro_ADE': best, 'final': log[-1]}, indent=2))


if __name__ == '__main__':
    main()
