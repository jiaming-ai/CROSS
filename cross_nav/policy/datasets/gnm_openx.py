"""Convert the Open X-Embodiment releases of GNM datasets (RLDS TFRecords) to realnav episodes.

    python -m cross_nav.policy.datasets.gnm_openx --dataset cory_hall --raw RLDS_ROOT --out REALNAV [--workers 12]

RLDS_ROOT holds robotics/berkeley_gnm_<dataset>/0.1.0/*.tfrecord-* (gs://gresearch/robotics, fetched and
checksum-verified with cross_nav.policy.datasets.fetch). Each RLDS episode is one GNM trajectory folder of the
original release (episode_metadata/file_path); its per-step observation/position (x, y, metres, odometry frame),
observation/yaw (rad) and observation/image are written as one or more realnav episodes: frames are subsampled by
travelled distance (realnav_common.subsample), and episodes are cut at pose jumps (> 1.5 m) and backward motion.
Only numeric arrays and encoded images are read (TFRecord/protobuf parsing, no pickles).
RECON is this project's reserved test set: its READY.json says "split": "test_only".
"""
import argparse, io, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from .realnav_common import subsample, split_at_jumps, write_ready, previews, wrap

INFO = {
    'cory_hall': dict(
        group=lambda ep: '_'.join(ep.split('_')[:2]),  # building floor + day, e.g. cory5_aug17 (= data.gnm_group)
        camera={'model': 'forward-facing monocular wide-angle camera on a 1/10-scale RC car (Kahn et al., ICRA 2018, GCG)',
                'fisheye': 'wide-angle with strong barrel distortion (fisheye-like, see preview)', 'resolution_released': [85, 64],
                'height_m': 'about 0.15 (RC car, approximate)', 'intrinsics': None},
        label='odometry', license='Open X-Embodiment release (CC BY 4.0); original data: Kahn et al. 2018',
        caveats=['images are 85x64 px as released (the only public version)',
                 'each RLDS episode is one short collision-terminated rollout re-zeroed at its start; '
                 'rollouts cannot be chained, so most are far shorter than one 8 m training window',
                 'self-supervised RL exploration data (GCG): many rollouts end by driving into a wall, '
                 'so the last ~1 m of an episode is a collision approach rather than a goal-directed demonstration']),
    'sac_son': dict(
        group=lambda ep: ep.split('_')[0],  # date + robot/building tag, e.g. Dec-06-2022-bww8 (= data.gnm_group)
        camera={'model': 'Ricoh Theta S 360 camera, front fisheye half (/fisheye_image)', 'fisheye': True,
                'robot': 'iRobot Create-based robot (SACSoN/HuRoN, Hirose et al. 2023)', 'resolution_released': [160, 120],
                'intrinsics': None, 'note': 'the raw HuRoN bags record the fisheye stream at 160x120 too'},
        label='odometry', license='Open X-Embodiment release (CC BY 4.0); original data: SACSoN/HuRoN, Hirose et al. 2023',
        caveats=['images are 160x120 px (native resolution of the recorded fisheye stream)']),
    'recon': dict(
        group=lambda ep: ep[:17],  # robot + recording date (= data.gnm_group)
        camera={'model': "Clearpath Jackal front camera, RECON hdf5 'images/rgb_left' (RECON, Shah et al. 2021)",
                'fisheye': True, 'note': 'circular fisheye image boundary visible in the frames',
                'resolution_released': [160, 120], 'intrinsics': None},
        label='odometry', license='Open X-Embodiment release (CC BY 4.0); original data: RECON, Shah et al. 2021',
        caveats=['RESERVED TEST-ONLY dataset in this project: never train or tune on it',
                 'converted subset of the 256 Open-X train shards (see READY.json shards)',
                 'images are 160x120 px as released by GNM/Open-X'],
        split='test_only'),
}
MIN_ARC = 1.0  # keep short pieces (Cory Hall rollouts) but drop near-static fragments


def pieces(pos, yaw, max_jump=1.5):
    """Subsample by distance, cut at jumps and backward motion. Returns lists of source indices."""
    out = []
    for a, b in split_at_jumps(pos, max_jump=max_jump):
        keep = a + subsample(pos[a:b], yaw[a:b])
        cur = [keep[0]]
        for i, j in zip(keep[:-1], keep[1:]):
            d = pos[j] - pos[i]
            if np.hypot(*d) >= 0.05 and math.cos(math.atan2(d[1], d[0]) - yaw[i]) < -0.5:
                out.append(cur); cur = [j]  # moving > 120 deg away from the heading: backward driving
            else:
                cur.append(j)
        out.append(cur)
    return [np.asarray(p) for p in out
            if len(p) >= 3 and np.linalg.norm(np.diff(pos[p], axis=0), axis=1).sum() >= MIN_ARC]


def write_pieces(out, key, pos, yaw, load_image, meta, t=None):
    """Write the pieces of one source trajectory; load_image(i) returns a PIL image. Returns [(name, n, arc)]."""
    res = []
    for k, idx in enumerate(pieces(pos, yaw)):
        name = key if k == 0 else f'{key}__{k}'
        d = Path(out) / name; d.mkdir(parents=True, exist_ok=True)
        size = None
        for n, i in enumerate(idx):
            im = load_image(int(i)).convert('RGB')
            if max(im.size) > 1024:
                s = 1024 / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
            im.save(d / f'{n}.jpg', quality=95); size = im.size
        arrays = dict(position=pos[idx].astype(np.float64), yaw=wrap(yaw[idx]).astype(np.float64))
        if t is not None: arrays['t'] = (t[idx] - t[idx[0]]).astype(np.float64)
        np.savez(d / 'trajectory.npz', **arrays)
        (d / 'source.json').write_text(json.dumps(dict(meta, source_indices=[int(idx[0]), int(idx[-1])],
                                                       image_size=list(size)), indent=1))
        res.append((name, len(idx), float(np.linalg.norm(np.diff(arrays['position'], axis=0), axis=1).sum())))
    return res


def convert_shard(path, out, dataset):
    from tfrecord.reader import tfrecord_loader
    info = INFO[dataset]; stats = dict(shard=Path(path).name, episodes_in=0, frames_in=0, written=0, frames=0, arc_m=0.0)
    for rec in tfrecord_loader(str(path), None):
        original = rec['episode_metadata/file_path'].decode()
        ep = Path(original).name
        pos = np.asarray(rec['steps/observation/position'], np.float64).reshape(-1, 2)
        yaw = np.asarray(rec['steps/observation/yaw'], np.float64).reshape(-1)
        imgs = rec['steps/observation/image']
        if isinstance(imgs, bytes): imgs = [imgs]
        assert len(pos) == len(yaw) == len(imgs), original
        stats['episodes_in'] += 1; stats['frames_in'] += len(pos)
        if not (np.isfinite(pos).all() and np.isfinite(yaw).all()): continue
        meta = dict(original=original, group=info['group'](ep), metric=True, label_source=info['label'],
                    camera=info['camera'], license=info['license'], dataset=dataset,
                    source=f'gs://gresearch/robotics/berkeley_gnm_{dataset}/0.1.0/{Path(path).name}',
                    fps=4.0 if dataset != 'cory_hall' else None,
                    note='GNM-processed trajectory (positions/yaw from the robot odometry, GNM sampling rate)')
        if info.get('split'): meta['split'] = info['split']
        for name, n, arc in write_pieces(out, ep, pos, yaw, lambda i: Image.open(io.BytesIO(imgs[i])), meta):
            stats['written'] += 1; stats['frames'] += n; stats['arc_m'] += arc
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(INFO)); ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--ready-only', action='store_true', help='only rewrite READY.json/previews of a finished conversion')
    a = ap.parse_args()
    out = Path(a.out) / a.dataset; out.mkdir(parents=True, exist_ok=True)
    shards = sorted((Path(a.raw) / f'robotics/berkeley_gnm_{a.dataset}/0.1.0').glob('*.tfrecord-*'))
    shards = [s for s in shards if not s.name.endswith(('.ok', '.json'))]
    log_path = out / 'conversion_log.jsonl'
    if a.ready_only:
        logs = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    else:
        logs = []
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(convert_shard, s, out, a.dataset) for s in shards]
            for f in as_completed(futs):
                logs.append(f.result()); print(json.dumps(logs[-1]), flush=True)
        log_path.write_text(''.join(json.dumps(l) + '\n' for l in logs))
    info = INFO[a.dataset]
    eps = list(out.glob('*/trajectory.npz'))
    windows = 0
    for p in eps:  # episodes long enough for one causal 8 m window (2 m history + 6.5 m future)
        with np.load(p) as z: windows += np.linalg.norm(np.diff(z['position'], axis=0), axis=1).sum() >= 8.5
    extra = dict(dataset=a.dataset, label_source=info['label'], license=info['license'],
                 source=f'Open X-Embodiment berkeley_gnm_{a.dataset} 0.1.0 (gs://gresearch/robotics)',
                 shards=sorted(l['shard'] for l in logs), source_episodes=sum(l['episodes_in'] for l in logs),
                 source_frames=sum(l['frames_in'] for l in logs), episodes_with_arc_ge_8_5m=int(windows),
                 caveats=info['caveats'], camera=info['camera'],
                 converter=f'python -m cross_nav.policy.datasets.gnm_openx --dataset {a.dataset}')
    if info.get('split'): extra['split'] = info['split']
    ready = write_ready(out, extra)
    previews(out)
    print(json.dumps({k: v for k, v in ready.items() if k not in ('shards',)}, indent=1), flush=True)


if __name__ == '__main__':
    main()
