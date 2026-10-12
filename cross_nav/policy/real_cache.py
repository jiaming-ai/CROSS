"""Frozen-backbone shards of real robot demonstrations, in the simulator recording format (server.ShardWriter).

    python -m cross_nav.policy.real_cache --root REALNAV --datasets sac_son go_stanford ... --out SHARDS \
        --backbone ckpt.pt [--per-dataset 20000] [--split train|val|test]

Windows match closed-loop simulation: views at 0 / 0.5 / 1 / 2 m of past travel (the first frame repeats near an
episode start), a goal 2-25 m further along the demonstration (its end when shorter), and the label = the next 8 m of
the demonstrated path at 0.5 m arc spacing, ending at the goal. Images are letterboxed into 512x384 (full field of
view kept). Whole recordings/sessions (source.json "group") go to one split: val when crc32(dataset:group) % 10 == 0;
RECON is test-only.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .backbone import FrozenMetricTokens, sha256
from .server import ShardWriter
from .sim.geom import HISTORY_M, history_ids

W, H = 512, 384
TEST_ONLY = {'recon'}


def split_of(dataset, group):
    if dataset in TEST_ONLY:
        return 'test'
    return 'val' if zlib.crc32(f'{dataset}:{group}'.encode()) % 10 == 0 else 'train'


def windows(dataset, ep_dir, rng, max_per_episode, horizon=16, spacing=.5):
    with np.load(ep_dir / 'trajectory.npz') as z:
        pos, yaw = z['position'].astype(np.float64), z['yaw'].astype(np.float64)
    src = json.loads((ep_dir / 'source.json').read_text())
    n = len(pos)
    if n < 5 or not (np.isfinite(pos).all() and np.isfinite(yaw).all()):
        return []
    edges = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
    if np.any(edges > 1.5):
        return []                     # converters split episodes at discontinuities; skip any that remain
    arc = np.r_[0., np.cumsum(edges)]
    out = []
    for a in rng.permutation(n):
        if len(out) >= max_per_episode:
            break
        remain = arc[-1] - arc[a]
        if remain < 1.:
            continue
        g = rng.uniform(2., 12.) if rng.random() < .6 else rng.uniform(12., 25.)
        g = min(g, remain)
        f = np.array([np.cos(yaw[a]), np.sin(yaw[a])]); l = np.array([-f[1], f[0]])
        local = (pos - pos[a]) @ np.stack((f, l), 1)
        s = arc[a] + np.minimum(np.arange(1, horizon + 1) * spacing, g)
        label = np.stack([np.interp(s, arc, local[:, i]) for i in range(2)], -1)
        goal = np.array([np.interp(arc[a] + g, arc, local[:, i]) for i in range(2)])
        if np.linalg.norm(goal) < .5:
            continue                  # loops back to the start: no usable goal direction
        ids = history_ids(arc[:a + 1], HISTORY_M)
        out.append({'images': [str(ep_dir / f'{i}.jpg') for i in ids], 'goal': goal.astype(np.float32),
                    'label': label.astype(np.float32), 'hist_chord': np.linalg.norm(pos[ids] - pos[a], axis=-1).astype(np.float32),
                    'step': int(a), 'meta': {'ep': f'{dataset}:{ep_dir.name}', 'scene': f"{dataset}:{src['group']}", 'dataset': dataset,
                                             'tag': 'real', 'expert_step': True, 'label_source': src.get('label_source', 'odometry'),
                                             'goal_arc': float(g)}})
    return out


def letterbox(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        r = min(W / im.width, H / im.height)
        im = im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (W, H), (123, 116, 103))
        canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
        return np.asarray(canvas)


class Windows(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return torch.from_numpy(np.stack([letterbox(p) for p in self.rows[i]['images']])), i


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--datasets', nargs='+', required=True); p.add_argument('--out', required=True)
    p.add_argument('--backbone', required=True); p.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    p.add_argument('--per-dataset', type=int, default=20000); p.add_argument('--per-episode', type=int, default=40)
    p.add_argument('--batch', type=int, default=16); p.add_argument('--workers', type=int, default=8); p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    model = FrozenMetricTokens(a.backbone).cuda().eval()
    out = Path(a.out) / a.split
    receipt = {'backbone_sha256': sha256(a.backbone), 'split': a.split, 'datasets': {}}
    t0 = time.time()
    for ds in a.datasets:
        if (out / f'{ds}.DONE.json').exists():
            continue
        rng = np.random.default_rng(zlib.crc32(ds.encode()) + a.seed)
        eps = sorted(d for d in (Path(a.root) / ds).iterdir() if (d / 'trajectory.npz').exists())
        eps = [d for d in eps if split_of(ds, json.loads((d / 'source.json').read_text())['group']) == a.split]
        rows = []
        for d in rng.permutation(len(eps)):
            rows += windows(ds, eps[d], rng, a.per_episode)
        if len(rows) > a.per_dataset:
            rows = [rows[i] for i in sorted(rng.choice(len(rows), a.per_dataset, replace=False))]
        writer = ShardWriter(out, ds, 512)
        if writer.k:
            raise RuntimeError(f'partial shards of {ds} exist in {out}; remove them first')
        loader = DataLoader(Windows(rows), batch_size=a.batch, num_workers=a.workers, pin_memory=True)
        for imgs, idx in loader:
            f = model(imgs.cuda(non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255), pool=(12, 16))
            for j, i in enumerate(idx.tolist()):
                r = rows[i]
                writer.add({'tokens': f['tokens'][j, -1].cpu(), 'xyz': f['xyz_unit'][j].half().cpu().numpy(),
                            'log_scale': f['log_scale'][j].float().cpu().numpy(), 'pred_chord': f['pred_history_chord'][j].float().cpu().numpy(),
                            'hist_chord': r['hist_chord'], 'goal': r['goal'], 'label': r['label'], 'step': np.int32(r['step']),
                            'expert_dist': np.float32(np.nan), 'meta': r['meta']})
        writer.flush()
        info = {'windows': len(rows), 'episodes': len(eps), 'groups': len({r['meta']['scene'] for r in rows}), 'seconds': round(time.time() - t0)}
        (out / f'{ds}.DONE.json').write_text(json.dumps(info, indent=2))
        receipt['datasets'][ds] = info
        print(json.dumps({ds: info}), flush=True)
    (out / f'receipt_{int(time.time())}.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
