"""Static web viewer of closed-loop episodes: runs side by side, top-down map, BEV overlay, camera frames.

    python -m cross_nav.policy.sim.viewer --runs "trajectory policy=DIR_A" "BEV + planner=DIR_B" --out OUTDIR \
        [--max-episodes 12] [--episodes ID ...] [--frame-width 320] [--jpeg-quality 75]

Each DIR holds results_*.jsonl (cross_nav.policy.sim.rollout --out) and viz/*.npz (rollout --viz-dir, see save_viz).
The summary tables use every results line; the episode browser shows up to --max-episodes episodes that have a viz
file, episodes recorded by more runs first. Episodes with the same id are shown side by side, synchronised by step.

Output (open OUTDIR/index.html directly, file:// works; no network access):
    index.html viewer.css viewer.js   the page (copied from viewer_web/)
    data/index.js                     runs, summaries, episode list
    data/ep_NNN.js                    one episode, all runs (loaded on demand with a <script> tag)
    frames/ep_NNN/<run>/SSSS.jpg      camera frames, --frame-width pixels wide

Frames in the page: world plane (x, z) with x to the right and z downward (the topdown bitmap's rows / columns); robot
paths are (forward, left) metres, forward = (-sin yaw, -cos yaw), left = (-cos yaw, sin yaw) (geom.py). BEV debug maps
(bev_policy.py): row 0 = far forward, column 0 = far left, the robot at the map centre.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import shutil
import zlib
from pathlib import Path

import numpy as np

WEB = Path(__file__).with_name('viewer_web')
BINS = ((0., 5.), (5., 10.), (10., 20.), (20., math.inf))
LEVELS = 63                 # BEV map quantisation (64 levels: ample for a translucent overlay, half the size of 256)
BEV_SCALARS = ('status', 'blind_zone', 'vis_threshold', 'log_scale', 'odometry_scale', 'n_views', 'up_head_deg')


# ------------------------------------------------------------------------------------------------ results
def load_results(run_dir):
    """id -> result over results_*.jsonl (a later line of the same id wins; truncated lines are skipped)."""
    out = {}
    for f in sorted(Path(run_dir).glob('results_*.jsonl')):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r['id']] = r
    return out


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {'n': 0}
    mean = lambda k: float(np.mean([float(r[k]) for r in rows]))
    fin = [r['final_geo'] for r in rows if r.get('final_geo') is not None and math.isfinite(r['final_geo'])]
    return {'n': n, 'success': mean('success'), 'spl': mean('spl'), 'collision_free': mean('collision_free'),
            'collisions': mean('collisions'), 'steps': mean('steps'),
            'final_geo_median': float(np.median(fin)) if fin else None}


def summary_table(results, ids=None):
    """Per run: overall and per geodesic-distance bin; ids restricts to a common episode set."""
    out = {}
    for name, res in results.items():
        rows = [r for i, r in res.items() if ids is None or i in ids]
        bins = []
        for lo, hi in BINS:
            sub = [r for r in rows if lo <= r['geodesic'] < hi]
            if sub or hi != math.inf:
                bins.append({'lo': lo, 'hi': None if hi == math.inf else hi, **summarize(sub)})
        out[name] = {'all': summarize(rows), 'bins': bins}
    return out


# ------------------------------------------------------------------------------------------------ episodes
def jpeg_list(z):
    """The per-step JPEG byte arrays of a viz file (an object array; equal-length frames may have stacked to 2-D)."""
    j = z['jpg']
    return [np.asarray(x, np.uint8) for x in (list(j) if j.ndim == 1 else [row.astype(np.uint8) for row in j])]


def b64z(a):
    return base64.b64encode(zlib.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()


def rnd(a, k=3):
    """Nested lists rounded to k decimals, NaN -> None (JSON null)."""
    a = np.asarray(a, np.float64)
    out = np.round(a, k).astype(object)
    out[~np.isfinite(a)] = None
    return out.tolist()


def write_frames(jpgs, out_dir, width, quality):
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    size = None
    for k, b in enumerate(jpgs):
        im = Image.open(io.BytesIO(b.tobytes())).convert('RGB')
        if width and im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        im.save(out_dir / f'{k:04d}.jpg', 'JPEG', quality=quality)
        size = im.size
    return size


def load_run_episode(npz, frame_dir, rel_dir, width, quality):
    z = np.load(npz, allow_pickle=True)
    result = json.loads(str(z['result']))
    jpgs = jpeg_list(z)
    T = len(jpgs)
    fw, fh = write_frames(jpgs, frame_dir, width, quality)
    pos = np.asarray(z['pos'], np.float64)
    traj = result.get('trajectory') or []
    final = traj[-1] if traj else pos[-1, [0, 2]].tolist()
    ep = {'T': T, 'result': {k: v for k, v in result.items() if k != 'trajectory'}, 'trajectory': rnd(traj),
          'final': rnd(final), 'pos': rnd(pos[:, [0, 2]]), 'yaw': rnd(z['yaw'], 4), 'path': rnd(z['path']),
          'expert': rnd(z['expert']), 'goal_rel': rnd(z['goal_rel']), 'window': np.asarray(z['window']).tolist(),
          'expert_step': np.asarray(z['expert_step']).astype(bool).tolist(),
          'frames': {'dir': rel_dir, 'w': fw, 'h': fh, 'src_w': None},
          'camera': result.get('camera')}
    from PIL import Image
    with Image.open(io.BytesIO(jpgs[0].tobytes())) as im0:
        ep['frames']['src_w'], ep['frames']['src_h'] = im0.size
    if 'topdown' in z.files:
        ep['topdown'] = {'bits': base64.b64encode(np.asarray(z['topdown'], np.uint8).tobytes()).decode(),
                         'shape': np.asarray(z['topdown_shape']).tolist(), 'origin': rnd(z['topdown_origin_xz'], 4),
                         'mpp': float(z['topdown_mpp'])}
    if 'debug_occ' in z.files:
        ep['bev'] = bev_record(z)
    return ep, z


def bev_record(z):
    """BEV debug maps quantised to LEVELS + 1 values (P * LEVELS), zlib + base64; per-step scalars as lists."""
    occ = np.asarray(z['debug_occ'], np.float32)
    T, n, _ = occ.shape
    rec = {'n': int(n), 'res': float(np.asarray(z['debug_map_res']).reshape(-1)[0]),
           'half': float(np.asarray(z['debug_map_half']).reshape(-1)[0]), 'maps': {}}
    for k in ('occ', 'obs', 'err'):
        if 'debug_' + k not in z.files:
            continue
        m = np.nan_to_num(np.asarray(z['debug_' + k], np.float32), nan=0.)
        scale = 1.
        if k == 'err':
            scale = float(max(np.percentile(m, 99.5), 1e-6))   # error score: not a probability, normalised to [0, 1]
        q = np.clip(np.round(m / scale * LEVELS), 0, LEVELS).astype(np.uint8)
        rec['maps'][k] = {'data': b64z(q), 'scale': scale, 'levels': LEVELS}
    for k in BEV_SCALARS:
        if 'debug_' + k in z.files:
            v = np.asarray(z['debug_' + k]).reshape(T)
            rec[k] = v.astype(bool).tolist() if v.dtype == bool else rnd(v, 4)
    for k in ('target', 'goal_used', 'path'):
        if 'debug_' + k in z.files:
            rec[k] = rnd(z['debug_' + k])
    return rec


def world_bounds(z, ep):
    """(x0, z0, x1, z1) covering start, goal, the trajectory and the expert paths of a run (world metres)."""
    pts = [np.asarray(z['start'])[[0, 2]][None], np.asarray(z['goal'])[[0, 2]][None], np.asarray(z['pos'])[:, [0, 2]]]
    if ep['trajectory']:
        pts.append(np.asarray(ep['trajectory'], np.float64))
    yaw = np.asarray(z['yaw'], np.float64)
    for key in ('path', 'expert'):
        p = np.asarray(z[key], np.float64)
        f = np.stack((-np.sin(yaw), -np.cos(yaw)), -1)[:, None]
        l = np.stack((-np.cos(yaw), np.sin(yaw)), -1)[:, None]
        w = np.asarray(z['pos'])[:, None, [0, 2]] + p[..., :1] * f + p[..., 1:] * l
        pts.append(w.reshape(-1, 2))
    p = np.concatenate(pts)
    p = p[np.isfinite(p).all(1)]
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]


def pick_episodes(viz, max_n, wanted):
    ids = sorted({i for v in viz.values() for i in v})
    if wanted:
        ids = [i for i in wanted if i in ids]
    else:
        ids.sort(key=lambda i: -sum(i in v for v in viz.values()))
    return ids[:max_n] if max_n else ids


def slug(s):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s).strip('_') or 'run'


def build(runs, out, max_episodes=12, episodes=None, frame_width=320, quality=75):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for d in ('data', 'frames'):
        shutil.rmtree(out / d, ignore_errors=True)
    (out / 'data').mkdir()
    results = {name: load_results(d) for name, d in runs}
    viz = {}
    for name, d in runs:
        files = {}
        for f in sorted((Path(d) / 'viz').glob('*.npz')):
            try:
                with np.load(f, allow_pickle=True) as z:
                    files[json.loads(str(z['result']))['id']] = f
            except Exception as e:   # a file being written, or an older format
                print(f'skip {f}: {e}')
        viz[name] = files
    id_sets = [set(r) for r in results.values()]
    common = set.intersection(*id_sets) if id_sets else set()
    differ = any(s != id_sets[0] for s in id_sets)
    index = {'runs': [{'name': name, 'dir': str(d), 'slug': slug(name), 'n_results': len(results[name]),
                       'n_viz': len(viz[name])} for name, d in runs],
             'summary': summary_table(results), 'common': summary_table(results, common) if differ else None,
             'n_common': len(common), 'episodes': []}
    ids = pick_episodes(viz, max_episodes, episodes)
    for k, eid in enumerate(ids):
        key = f'ep_{k:03d}'
        ep = {'id': eid, 'key': key, 'runs': {}}
        bounds = []
        for ri, (name, _) in enumerate(runs):
            f = viz[name].get(eid)
            if f is None:
                continue
            rel = f'frames/{key}/{ri}'
            rec, z = load_run_episode(f, out / rel, rel, frame_width, quality)
            ep['runs'][name] = rec
            bounds.append(world_bounds(z, rec))
            ep.setdefault('scene', rec['result'].get('scene'))
            ep.setdefault('geodesic', rec['result'].get('geodesic'))
            ep.setdefault('start', rnd(np.asarray(z['start'])[[0, 2]]))
            ep.setdefault('goal', rnd(np.asarray(z['goal'])[[0, 2]]))
            z.close()
        b = np.asarray(bounds)
        margin = 1.5
        ep['view'] = [float(b[:, 0].min()) - margin, float(b[:, 1].min()) - margin,
                      float(b[:, 2].max()) + margin, float(b[:, 3].max()) + margin]
        # topdown maps of the same scene and start height are identical across runs: keep one copy
        tds = [r.pop('topdown') for r in ep['runs'].values() if 'topdown' in r]
        ep['topdown'] = tds[0] if tds else None
        (out / 'data' / f'{key}.js').write_text(f'window.VIEWER.addEpisode({json.dumps(ep, separators=(",", ":"))});\n')
        row = {'id': eid, 'key': key, 'scene': ep.get('scene'), 'geodesic': ep.get('geodesic'), 'runs': {}}
        for name in results:
            r = results[name].get(eid)
            row['runs'][name] = None if r is None else {'success': r['success'], 'collisions': r['collisions'],
                                                         'steps': r['steps'], 'spl': r['spl'], 'viz': eid in viz[name]}
        index['episodes'].append(row)
        print(f'{key} {eid}: ' + ', '.join(f'{n} T={ep["runs"][n]["T"]}' for n in ep['runs']), flush=True)
    (out / 'data' / 'index.js').write_text(f'window.VIEWER.setIndex({json.dumps(index, separators=(",", ":"))});\n')
    for f in ('index.html', 'viewer.css', 'viewer.js'):
        shutil.copy(WEB / f, out / f)
    total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    print(json.dumps({'out': str(out / 'index.html'), 'episodes': len(ids), 'bytes': total}))
    return out / 'index.html'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--runs', nargs='+', required=True, metavar='NAME=DIR', help='run name and its output directory')
    p.add_argument('--out', required=True)
    p.add_argument('--max-episodes', type=int, default=12, help='episodes in the browser (0: all with viz files)')
    p.add_argument('--episodes', nargs='*', help='episode ids to show (default: recorded by most runs first)')
    p.add_argument('--frame-width', type=int, default=320)
    p.add_argument('--jpeg-quality', type=int, default=75)
    a = p.parse_args()
    runs = []
    for s in a.runs:
        name, sep, d = s.partition('=')
        if not sep or not name or not d:
            p.error(f'--runs expects NAME=DIR, got {s!r}')
        if not Path(d).is_dir():
            p.error(f'not a directory: {d}')
        runs.append((name, d))
    if len({n for n, _ in runs}) != len(runs):
        p.error('run names must be unique')
    build(runs, a.out, a.max_episodes, a.episodes, a.frame_width, a.jpeg_quality)


if __name__ == '__main__':
    main()
