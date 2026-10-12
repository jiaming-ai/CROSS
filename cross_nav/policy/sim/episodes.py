"""Point-goal episodes on the largest navmesh island of HSSD scenes (habitat environment).

    python -m cross_nav.policy.sim.episodes --hssd ROOT --split test --per-scene 25 --out test.json \
        [--min-geo 1.5] [--max-geo 20] [--navmesh-dir DIR] [--scenes ...] [--scenes-file F] [--exclude ...]

--hssd may also be an hssd/ai2thor-hab root (scenes from --scenes / --scenes-file; every navigable point is indoor).

Habitat PointNav-style sampling on the island with the largest indoor area: indoor start and goal on one floor (height difference < 0.3 m), geodesic distance in
[min, max], and an episode whose geodesic/Euclidean ratio is below 1.1 kept only with probability 0.05, so most need a
detour. Starts at least 0.1 m from obstacles, random start heading. Seeded by scene name, so splits are reproducible.
"""
from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import numpy as np

from .hab import Rooms, geodesic, indoor_island, make_sim


def split_scenes(root, split):
    """Scene ids of an official split from scene_splits.yaml (top-level keys with '- id' lists; no yaml dependency)."""
    out, cur = {}, None
    for line in (Path(root) / 'scene_splits.yaml').read_text().splitlines():
        if line and not line.startswith((' ', '-')) and line.rstrip().endswith(':'):
            cur = line.strip()[:-1]; out[cur] = []
        elif line.strip().startswith('- ') and cur is not None:
            out[cur].append(line.strip()[2:].strip().strip("'\""))
    return out[split]


def sample(sim, rooms, scene, n, min_geo, max_geo, seed, tries=20000):
    pf = sim.pathfinder
    pf.seed(seed)
    rng = np.random.default_rng(seed)
    isl, area = indoor_island(pf, rooms)
    out = []
    for _ in range(tries):
        if len(out) >= n:
            break
        s = np.asarray(pf.get_random_navigable_point(island_index=isl))
        g = np.asarray(pf.get_random_navigable_point(island_index=isl))
        if not (np.isfinite(s).all() and np.isfinite(g).all()) or abs(s[1] - g[1]) > .3:
            continue
        if not (rooms.indoor(s) and rooms.indoor(g)):
            continue
        if pf.distance_to_closest_obstacle(s, .5) < .1:
            continue
        d, _ = geodesic(pf, s, g)
        e = float(np.linalg.norm((s - g)[[0, 2]]))
        if not (min_geo <= d <= max_geo) or (d / max(e, 1e-6) < 1.1 and rng.random() > .05):
            continue
        out.append({'id': f'{scene}:{len(out):03d}', 'scene': scene, 'start': s.tolist(), 'yaw': float(rng.uniform(-math.pi, math.pi)),
                    'goal': g.tolist(), 'geodesic': d, 'euclidean': e, 'island': int(isl), 'indoor_area': round(float(area), 1)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hssd', required=True); p.add_argument('--split', required=True); p.add_argument('--out', required=True)
    p.add_argument('--per-scene', type=int, default=25); p.add_argument('--min-geo', type=float, default=1.5)
    p.add_argument('--max-geo', type=float, default=20.); p.add_argument('--navmesh-dir'); p.add_argument('--scenes', nargs='*')
    p.add_argument('--gpu', type=int, default=0); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exclude', nargs='*', default=[], help='scenes to leave out (e.g. the held-out test scenes)')
    p.add_argument('--scenes-file', help='scene ids, one per line (AI2-THOR houses have no official split file)')
    a = p.parse_args()
    if a.scenes_file:
        a.scenes = (a.scenes or []) + [l.strip() for l in Path(a.scenes_file).read_text().splitlines() if l.strip()]
    scenes = [s for s in (a.scenes or split_scenes(a.hssd, a.split)) if s not in set(a.exclude)]
    eps, skipped = [], []
    for scene in scenes:
        try:
            sim = make_sim(a.hssd, scene, a.gpu, a.navmesh_dir)
        except Exception as e:  # noqa: BLE001  (record broken scenes, keep going)
            skipped.append({'scene': scene, 'error': repr(e)}); continue
        try:
            got = sample(sim, Rooms(a.hssd, scene), scene, a.per_scene, a.min_geo, a.max_geo, (zlib.crc32(scene.encode()) + a.seed) % 2**31)
        finally:
            sim.close()
        if len(got) < a.per_scene:
            skipped.append({'scene': scene, 'error': f'only {len(got)} episodes'})
        eps += got
        print(json.dumps({'scene': scene, 'episodes': len(got)}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({'split': a.split, 'min_geo': a.min_geo, 'max_geo': a.max_geo, 'seed': a.seed,
                                       'skipped': skipped, 'episodes': eps}, indent=1))
    print(json.dumps({'episodes': len(eps), 'skipped': len(skipped)}), flush=True)


if __name__ == '__main__':
    main()
