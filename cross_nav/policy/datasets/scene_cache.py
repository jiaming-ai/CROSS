"""NCLT / GrandTour / TartanGround scene caches (vggt_ft format) -> common trajectory format, images symlinked.

    python -m cross_nav.policy.datasets.scene_cache --cache CACHE --bev-root BEV --dataset nclt --out REALNAV

Camera centres are projected onto the gravity plane of each world (gravity from its BEV label, as in
cross_nav.policy.data.scene_records): x along the first frame's projected optical axis, y to its left; yaw is the
projected optical axis. Sequences are split at > 1.5 m jumps. TartanGround keeps differential-drive sequences and
excludes the reserved test environments.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ..data import scene_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True); p.add_argument('--bev-root', required=True)
    p.add_argument('--dataset', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out) / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    stats = {'episodes': 0, 'frames': 0, 'groups': set(), 'arc': [], 'spacing': []}
    for rec in scene_records(a.cache, a.bev_root, a.dataset):
        c, f, l = rec['position'], rec['forward'], rec['left']
        e1, e2 = f[0], l[0]
        pos = np.stack(((c - c[0]) @ e1, (c - c[0]) @ e2), -1)
        yaw = np.arctan2(f @ e2, f @ e1)
        jumps = np.nonzero(np.linalg.norm(np.diff(pos, axis=0), axis=-1) > 1.5)[0]
        for k, (s, e) in enumerate(zip(np.r_[0, jumps + 1], np.r_[jumps + 1, len(pos)])):
            if e - s < 20:
                continue
            dest = out / f"{rec['sequence'].replace('/', '__')}__{k}"
            dest.mkdir(exist_ok=True)
            for i, src in enumerate(rec['images'][s:e]):
                link = dest / f'{i}.jpg'
                if not link.exists():
                    os.symlink(os.path.realpath(src), link)
            np.savez_compressed(dest / 'trajectory.npz', position=pos[s:e], yaw=yaw[s:e])
            (dest / 'source.json').write_text(json.dumps({'original': rec['sequence'], 'segment': int(k), 'group': str(rec['group']), 'metric': True,
                                                          'label_source': 'dataset_ground_truth_poses', 'camera': {'note': 'see vggt_ft scene.json'}}))
            d = np.linalg.norm(np.diff(pos[s:e], axis=0), axis=-1)
            stats['episodes'] += 1; stats['frames'] += e - s; stats['groups'].add(rec['group'])
            stats['arc'].append(float(d.sum())); stats['spacing'] += d.tolist()
    ready = {'dataset': a.dataset, 'episodes': int(stats['episodes']), 'frames': int(stats['frames']), 'groups': len(stats['groups']),
             'arc_quantiles': np.quantile(stats['arc'], [0, .25, .5, .75, 1]).tolist(),
             'spacing_quantiles': np.quantile(stats['spacing'], [.05, .25, .5, .75, .95]).tolist(), 'label_source': 'dataset_ground_truth_poses'}
    (out / 'READY.json').write_text(json.dumps(ready, indent=2))
    print(json.dumps(ready), flush=True)


if __name__ == '__main__':
    main()
