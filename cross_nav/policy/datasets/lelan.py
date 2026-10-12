"""Convert LeLaN (NHirose/LeLaN_dataset_NoMaD_traj) into single-view pseudo-labelled samples.

LeLaN has no executed metric trajectory: each frame comes with language prompts for a detected
object, the object's estimated position on the robot frame, and an 8-step collision-avoidance
trajectory toward the object that the NoMaD policy generated (model output, not executed motion).
So it is not written in the realnav trajectory format; instead:

<out>/images/<subset>/<sequence>/<frame>.jpg   front fisheye view (224x224)
<out>/records.jsonl                            one line per (frame, object) sample
<out>/README.json                              format, counts, caveats

python -m cross_nav.policy.datasets.lelan --raw <dir containing dataset_LeLaN_v2> --out <out> [--workers 12]

The per-frame labels are Python pickles from the official dataset repo (the only format they are
released in). They are read with an unpickler that only admits numpy array reconstruction.
"""
import argparse, collections, json, os, pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

DENORM = 0.12  # metres per normalized unit (LeLaN train_utils: "0.12 is de-normalization")
SUBSETS = {  # subset -> (source, camera)
    'gs2': ('GO Stanford 2 (TurtleBot2, Ricoh Theta S 360 camera)', 'dual'),
    'gs4': ('GO Stanford 4 (TurtleBot2, Ricoh Theta S 360 camera)', 'dual'),
    'sacson': ('SACSoN / HuRoN (Ricoh Theta 360 camera)', 'dual'),
    'humanwalk': ('hand-held Ricoh Theta 360 camera, human walking', 'front'),
}


class NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split('.')[0] == 'numpy':
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'refusing {module}.{name}')


def load_labels(path):
    if os.path.getsize(path) == 0: return []
    with open(path, 'rb') as f: obj = NumpyOnly(f).load()
    return obj if isinstance(obj, list) else []


def sequences(root):
    """Yield (subset, sequence key, group, image dir, pickle dir)."""
    root = Path(root)
    for subset in SUBSETS:
        base = root / f'dataset_LeLaN_{subset}'
        if subset == 'gs4':
            yield subset, 'gs4', None, base / 'image', base / 'pickle_nomad'; continue
        if subset == 'sacson':
            for d in sorted(p for p in base.glob('*/*') if (p / 'image').is_dir()):
                yield subset, f'{d.parent.name}_{d.name}', d.parent.name, d / 'image', d / 'pickle_nomad'
            continue
        for d in sorted(p for p in base.iterdir() if (p / 'image').is_dir()):
            yield subset, d.name, d.name, d / 'image', d / 'pickle_nomad'


def convert_sequence(args):
    subset, seq, group, image_dir, pickle_dir, out = args
    out = Path(out)
    images = sorted(image_dir.glob('*.jpg'))
    rows, stats = [], collections.Counter()
    for img in images:
        idx = int(img.stem)
        pk = pickle_dir / f'{idx:08d}.pkl'
        if not pk.is_file(): stats['no_label'] += 1; continue
        try: labels = load_labels(pk)
        except Exception: stats['bad_label'] += 1; continue
        labels = [l for l in labels if isinstance(l, dict) and 'nomad_traj_norm' in l and l.get('prompt')]
        if not labels: stats['no_object'] += 1; continue
        rel = Path('images') / subset / seq / f'{idx}.jpg'
        dst = out / rel
        if not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(img) as im:
                im = im.convert('RGB')
                if SUBSETS[subset][1] == 'dual': im = im.crop((0, 0, im.height, im.height))  # front lens
                im.save(dst, quality=95)
        g = group if group is not None else f'gs4_block{idx // 2000:03d}'
        for k, l in enumerate(labels):
            traj = np.asarray(l['nomad_traj_norm'], np.float64).reshape(-1, 4)
            prompts = [p[0] if isinstance(p, (tuple, list)) else p for p in l['prompt']]
            prompts = [p for p in prompts if isinstance(p, str) and p]
            if not prompts or not np.isfinite(traj).all(): stats['bad_object'] += 1; continue
            obj = np.asarray(l['pose_median'], np.float64).reshape(-1)[:2]
            rows.append(dict(
                id=f'{subset}:{seq}:{idx}:{k}', subset=subset, sequence=seq, group=g, frame=idx, image=str(rel),
                prompts=prompts, object_xy_m=[round(float(v), 4) for v in obj],
                bbox_raw=[int(v) for v in np.asarray(l.get('bbox', [-1] * 4)).reshape(-1)[:4]],
                nomad_xy_m=np.round(traj[:, :2] * DENORM, 4).tolist(),
                nomad_yaw=np.round(np.arctan2(traj[:, 3], traj[:, 2]), 4).tolist()))
            stats['samples'] += 1
        stats['frames'] += 1
    return rows, stats, len(images)


def preview(out, n=4, seed=0):
    """Contact sheets: sample images (bbox drawn as [row_min, row_max, col_min, col_max]) + NoMaD path and object."""
    import random
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    lines = (out / 'records.jsonl').read_text().splitlines()
    rng = random.Random(seed)
    (out / 'preview').mkdir(exist_ok=True)
    for sheet in range(n):
        rows = [json.loads(lines[rng.randrange(len(lines))]) for _ in range(4)]
        fig, axs = plt.subplots(2, 4, figsize=(13, 6.5))
        for r, a0, a1 in zip(rows, axs[0], axs[1]):
            with Image.open(out / r['image']) as im: a0.imshow(im)
            y0, y1, x0, x1 = r['bbox_raw']
            a0.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color='y', lw=2))
            a0.set_title(f"{r['id']}\n{r['prompts'][0]}", fontsize=7); a0.axis('off')
            xy = np.asarray([[0, 0]] + r['nomad_xy_m'])
            a1.plot(-xy[:, 1], xy[:, 0], 'b.-', label='NoMaD path')
            a1.plot(-r['object_xy_m'][1], r['object_xy_m'][0], 'r*', ms=12, label='object')
            a1.set_aspect('equal'); a1.set_xlabel('right (m)'); a1.set_ylabel('forward (m)'); a1.grid(alpha=.3)
            lim = max(1.0, np.abs(xy).max(), *np.abs(r['object_xy_m'])) + .2
            a1.set_xlim(-lim, lim); a1.set_ylim(-.2, lim)
        axs[1][0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / 'preview' / f'samples_{sheet}.png', dpi=80); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', required=True, help='directory that contains dataset_LeLaN_v2/ (or is it)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--preview_only', action='store_true', help='only render preview/ sheets from records.jsonl')
    a = ap.parse_args()
    if a.preview_only: preview(a.out); return
    raw = Path(a.raw)
    if (raw / 'dataset_LeLaN_v2').is_dir(): raw = raw / 'dataset_LeLaN_v2'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [(*s, out) for s in sequences(raw)]
    stats, per_subset, nimg = collections.Counter(), collections.defaultdict(collections.Counter), 0
    tmp = out / 'records.jsonl.tmp'
    with open(tmp, 'w') as f, ProcessPoolExecutor(a.workers) as ex:
        for job, (rows, st, n) in zip(jobs, ex.map(convert_sequence, jobs, chunksize=1)):
            for r in rows: f.write(json.dumps(r) + '\n')
            stats.update(st); per_subset[job[0]].update(st); per_subset[job[0]]['sequences'] += 1
            per_subset[job[0]]['raw_images'] += n; nimg += n
    tmp.rename(out / 'records.jsonl')
    # Summaries over the written records.
    final, dist, groups, nprompt, agree = [], collections.defaultdict(list), collections.defaultdict(set), [], []
    with open(out / 'records.jsonl') as f:
        for line in f:
            r = json.loads(line)
            xy = np.asarray(r['nomad_xy_m'])
            dist[r['subset']].append(float(np.linalg.norm(xy[-1])))
            final.append(float(np.linalg.norm(r['object_xy_m'])))
            groups[r['subset']].add(r['group']); nprompt.append(len(r['prompts']))
            o = np.asarray(r['object_xy_m'])
            if np.linalg.norm(o) > .3 and np.linalg.norm(xy[-1]) > .2:  # NoMaD end direction vs object direction
                agree.append(abs(np.degrees(np.angle(complex(*xy[-1]) / complex(*o)))))
    q = lambda v: {f'q{int(p*100)}': round(float(np.quantile(v, p)), 3) for p in (.05, .25, .5, .75, .95)} if len(v) else {}
    youtube = raw / 'dataset_LeLaN_youtube'
    n_yt = sum(1 for _ in youtube.glob('*/pickle_nomad/*.pkl')) if youtube.is_dir() else 0
    readme = {
        'dataset': 'LeLaN (Learning Language-conditioned Navigation), NoMaD-trajectory release',
        'source': 'https://huggingface.co/datasets/NHirose/LeLaN_dataset_NoMaD_traj (dataset_LeLaN_v2.zip); code https://github.com/NHirose/learning-language-navigation',
        'license': 'MIT (dataset card and LeLaN code). Underlying recordings: GO Stanford 2/4 and SACSoN (Hirose et al.), human-walking videos by the LeLaN authors.',
        'why_not_trajectory_format': 'No executed metric trajectory or odometry is released. Frames are temporally ordered within a sequence, '
            'but the only motion labels are model outputs: an 8-step NoMaD (diffusion policy) trajectory toward the detected object, '
            'generated from the image and an object crop. So these are single-view pseudo-labelled samples.',
        'format': {
            'images/<subset>/<sequence>/<frame>.jpg': 'front fisheye lens view, 224x224 RGB (for 360-camera subsets the left half of the 448x224 dual-fisheye frame; the back lens is dropped). JPEG q95.',
            'records.jsonl': 'one JSON object per (frame, object) sample',
            'fields': {
                'id': 'subset:sequence:frame:object_index',
                'image': 'path relative to this directory',
                'sequence, frame': 'frames of a sequence are in temporal order (frame index = original file index); spacing in metres is unknown',
                'group': 'recording/video/session for train/val splitting (gs4 is one long recording: split into blocks of 2000 frames)',
                'prompts': 'alternative language names of the target object (from a VLM labeller)',
                'object_xy_m': 'estimated object position on the robot frame, metres [forward, left] (LeLaN pose_median; estimated from the image, not measured)',
                'bbox_raw': 'object bounding box as stored in LeLaN: [row_min, row_max, col_min, col_max] pixels in the 224x224 front view (checked visually on previews)',
                'nomad_xy_m': '8 waypoints [forward, left] in metres = nomad_traj_norm[:, :2] * 0.12 (de-normalization used by LeLaN training)',
                'nomad_yaw': '8 headings (rad) from nomad_traj_norm cos/sin columns',
            },
        },
        'label_source': 'nomad (model-generated, not executed); object positions estimated by the LeLaN labelling pipeline',
        'counts': {'samples': stats['samples'], 'frames_with_labels': stats['frames'], 'raw_images': nimg,
                   'frames_skipped': {k: stats[k] for k in ('no_label', 'bad_label', 'no_object')}, 'bad_objects': stats['bad_object'],
                   'per_subset': {k: dict(v, groups=len(groups[k])) for k, v in per_subset.items()}},
        'nomad_final_waypoint_distance_m': {k: q(v) for k, v in dist.items()},
        'object_distance_m': q(final), 'prompts_per_sample': q(nprompt),
        'nomad_end_vs_object_direction_deg': dict(q(agree), frac_below_30=round(float(np.mean(np.asarray(agree) < 30)), 3),
                                                  frac_above_90=round(float(np.mean(np.asarray(agree) > 90)), 3)),
        'camera': {'gs2/gs4/sacson': 'Ricoh Theta 360 camera on a small ground robot (height not released); front fisheye, ~180 deg FOV, circular image',
                   'humanwalk': 'hand-held Ricoh Theta (height not released, roughly chest height), front fisheye ~180 deg FOV',
                   'note': 'strongly distorted circular fisheye, unlike a pinhole robot camera; no intrinsics released'},
        'skipped': {'youtube': f'{n_yt} label pickles without images: images must be extracted from YouTube videos with '
                    'learning-language-navigation/download_youtube/save_youtube_image.py (copyrighted videos, not redistributed); not done.'},
        'caveats': [
            'Horizon is short: the NoMaD trajectory covers ~1 m (8 steps) versus our 16 x 0.5 m = 8 m targets.',
            'Trajectories are NoMaD outputs conditioned on an object crop, i.e. pseudo-labels; they were never executed. '
            'Their end direction is within 30 deg of the object direction for ~77% of samples and opposite (>90 deg) for ~2-3%.',
            'Frames of gs2/gs4/sacson are the same recordings as GNM go_stanford / SACSoN data (with different labels); keep their groups consistent with the GNM split if both are used.',
            'OmniVLA uses only the front 224x224 lens of the dual-fisheye frames, as done here.',
        ],
    }
    (out / 'README.json').write_text(json.dumps(readme, indent=2))
    preview(out)
    print(json.dumps(readme['counts'], indent=1))


if __name__ == '__main__':
    main()
