"""Convert CAST (catglossop/CAST-dataset) into realnav episodes, keeping counterfactual labels apart.

CAST is built from the GNM mixture: language-annotated chunks of the original GNM trajectories
(cast_filtered_dataset), counterfactual records (cast_counterfactual_dataset: the first steps are real
frames of a GNM trajectory, the remaining steps are synthetic actions/positions with white placeholder
images), and atomic-skill chunks (atomic_dataset). Images are 128x128 PNG; positions are the GNM
trajectories' metric odometry in each trajectory's own frame (chunk names give the original frame index).

This converter reads the RLDS tfrecords straight out of the release tarballs (no extraction), merges
all REAL frames of each original trajectory across the three files (they share frame indices and
positions), drops RECON trajectories (test-only for us), and writes per episode:
  {i}.jpg, trajectory.npz (position, yaw; frame of the original GNM trajectory), source.json,
  language.json   language instructions of the chunks covering the episode (episode-local frame ranges)
  counterfactual.npz  synthetic continuations branching from a real frame (never mixed into positions)

python -m cross_nav.policy.datasets.cast --tars <dir with *_dataset.tar.gz> --out <out> [--workers 12] [--parts filtered,counterfactual]

atomic_dataset (39 GB) is not used by default: its records are 3-11-step snippets named <traj>_chunk_<k>
without the original frame index, so they cannot be placed on the trajectories by name; they also
re-use the same GNM trajectories (RECON included).
"""
import argparse, collections, io, json, re, struct, tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from cross_nav.policy.datasets import realnav_common as rc

TARS = {'filtered': 'cast_filtered_dataset.tar.gz', 'counterfactual': 'cast_counterfactual_dataset.tar.gz', 'atomic': 'atomic_dataset.tar.gz'}
NAME = re.compile(r'^(?P<traj>.+?)_chunk_(?P<chunk>\d+)_start_(?P<start>\d+)_end_(?P<end>\d+)(?:_cf_(?P<cf>\d+))?$')
LICENSE = ('CAST release: no license file on the Hugging Face dataset card (paper/code: cast-vla.github.io, github.com/catglossop/CAST); '
           'underlying images come from the GNM datasets under their own licenses (SCAND CC-BY-4.0, TartanDrive BSD-3-Clause, '
           'SACSoN/GO Stanford/Seattle/Cory Hall research-use releases). Treat as research-only.')
CAMERA = {
    'cory_hall': 'Cory Hall (Berkeley) indoor RC car, forward pinhole camera',
    'go_stanford': 'GO Stanford 4 (TurtleBot2, Ricoh Theta 360 camera; GNM uses the front fisheye view)',
    'go_stanford2': 'GO Stanford 2 / GONet 2017 (TurtleBot2, Ricoh Theta 360 camera, front fisheye view)',
    'sac_son': 'SACSoN/HuRoN (TurtleBot2-class robot, Ricoh Theta 360 camera, front fisheye view)',
    'seattle': 'Seattle off-road (UW, Clearpath Warthog-class vehicle, forward camera)',
    'scand': 'SCAND (Jackal or Spot, forward camera), social navigation on UT Austin campus',
    'tartan_drive': 'TartanDrive (Yamaha ATV off-road, forward stereo camera)',
}


def source_of(traj, norm):
    """Original GNM dataset of a trajectory from its name pattern and CAST normalization factor."""
    if traj.startswith('cory'): return 'cory_hall'
    if traj.startswith('tartan_'): return 'tartan_drive'
    if traj.startswith('uw_'): return 'seattle'
    if re.match(r'^(jackal|spot)_[0-9a-z]{32}', traj): return 'scand'
    if traj.startswith('jackal_20'): return 'recon'
    if re.match(r'^[A-Z][a-z]{2}-\d{2}-\d{4}', traj): return 'sac_son'
    if re.match(r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', traj): return 'go_stanford2'
    if re.match(r'^(no\d+vc|sim\d+h)', traj): return 'go_stanford'
    return {0.06: 'cory_hall', 0.12: 'go_stanford', 0.25: 'recon', 0.255: 'sac_son', 0.35: 'seattle', 0.38: 'scand', 0.72: 'tartan_drive'}.get(round(norm, 3), 'unknown')


def group_of(dataset, traj):
    """Session-level group, matching cross_nav.policy.data.gnm_group where that function defines one."""
    p = traj.split('_')
    if dataset == 'cory_hall': return '_'.join(p[:2])
    if dataset == 'sac_son': return p[0]
    if dataset in ('seattle', 'scand', 'tartan_drive'): return '_'.join(p[:2])
    if dataset == 'go_stanford': return p[0]
    return traj


# ---------------------------------------------------------------- tar / tfrecord reading
def shard_members(tar_path):
    out = []
    with tarfile.open(tar_path, 'r:') as t:
        for m in t:
            if m.isfile() and '.tfrecord' in m.name and not Path(m.name).name.startswith('._'):
                out.append((m.name, m.offset_data, m.size))
    return out


def records(tar_path, offset, size):
    with open(tar_path, 'rb') as f:
        f.seek(offset); end = offset + size
        while f.tell() < end:
            head = f.read(12)
            if len(head) < 12: break
            (n,) = struct.unpack('<Q', head[:8])
            data = f.read(n); f.read(4)
            yield data


def parse(data):
    from tfrecord import example_pb2
    ex = example_pb2.Example(); ex.ParseFromString(data)
    F = ex.features.feature
    fl = lambda k: np.asarray(F[k].float_list.value, np.float64)
    path = F['episode_metadata/file_path'].bytes_list.value[0].decode()
    n = len(F['steps/is_first'].int64_list.value)
    lang = [s.decode() for s in F['steps/language_instruction'].bytes_list.value]
    lang = np.asarray(lang, object).reshape(n, -1) if lang else np.empty((n, 0), object)
    return dict(path=path, norm=float(F['episode_metadata/normalization_factor'].float_list.value[0]),
                pos=fl('steps/observation/position').reshape(n, 2), yaw=fl('steps/observation/yaw').reshape(n),
                images=list(F['steps/observation/image'].bytes_list.value), lang=lang)


def is_placeholder(png):
    if len(png) > 2000: return False
    a = np.asarray(Image.open(io.BytesIO(png)))
    return bool(a.min() >= 250)


# ---------------------------------------------------------------- pass 1: metadata
def scan_shard(args):
    """Per record: its real frames (frame index, pose, image pointer) and its language / counterfactual payload."""
    kind, tar_path, name, offset, size = args
    recs, unparsed = [], 0
    for r, data in enumerate(records(tar_path, offset, size)):
        ex = parse(data)
        parts = ex['path'].rstrip('/').split('/')
        m = NAME.match(parts[-1])
        if not m: unparsed += 1; continue
        traj, start = m['traj'], int(m['start'])
        real = [not is_placeholder(b) for b in ex['images']]
        nreal = len(real) if all(real) else real.index(False)
        frames = [(start + i, ex['pos'][i, 0], ex['pos'][i, 1], ex['yaw'][i], name, r, i) for i in range(nreal)]
        instr = sorted({s for s in ex['lang'][0] if s}) if ex['lang'].size else []
        rec = dict(traj=traj, start=start, kind=kind, version=parts[-2], record=parts[-1], norm=ex['norm'], frames=frames, lang=instr)
        if m['cf'] is not None:
            if nreal == len(real): continue  # no synthetic part
            rec['cf'] = dict(anchor=start + nreal - 1, pos=ex['pos'][nreal:].tolist(), yaw=ex['yaw'][nreal:].tolist())
        recs.append(rec)
    return recs, unparsed, kind


# ---------------------------------------------------------------- pass 2: images
def write_shard(args):
    kind, tar_path, name, offset, size, assign = args  # assign: {(rec, step): path}
    by_rec = collections.defaultdict(list)
    for (rr, s), p in assign.items(): by_rec[rr].append((s, p))
    n = 0
    for r, data in enumerate(records(tar_path, offset, size)):
        need = by_rec.get(r)
        if not need: continue
        ex = parse(data)
        for s, p in need:
            with Image.open(io.BytesIO(ex['images'][s])) as im: rc.save_jpg(im, p)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tars', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--parts', default='filtered,counterfactual')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for kind in a.parts.split(','):
        tp = Path(a.tars) / TARS[kind]
        if not tp.is_file(): print('missing', tp); continue
        jobs += [(kind, str(tp), nm, o, s) for nm, o, s in shard_members(tp)]
    print(f'{len(jobs)} shards', flush=True)
    recs, stats = [], collections.Counter()
    with ProcessPoolExecutor(a.workers) as ex:
        for rr, unp, kind in ex.map(scan_shard, jobs, chunksize=1):
            stats[f'unparsed_{kind}'] += unp; recs += rr
    # Merge real frames per trajectory. Different CAST versions of the same trajectory are not always in the same
    # coordinate frame / frame indexing (seen mostly on GO Stanford 2: ~0.1 m offsets, some re-centred records). Records
    # are grouped into consistent components: a record joins (and may connect) the components whose frames it overlaps
    # if every overlapping frame agrees within TOL (filtered chunks also join adjacent filtered chunks); a record that disagrees with any overlapping frame is dropped; a record
    # without overlap starts a new component. Episodes are built per component, so frames of different components
    # (possibly different coordinate frames) are never mixed.
    TOL = 0.02
    comps = collections.defaultdict(list)  # traj -> [frame -> (x, y, yaw, shard, rec, step)]
    owner = collections.defaultdict(dict)  # traj -> frame -> component index
    norms, cfs, langs, mism = {}, [], [], []
    filtered_frames = collections.defaultdict(set)
    order = sorted(recs, key=lambda r: (r['kind'] != 'filtered', r['version'], r['traj'], r['start']))
    for r in order:
        t = r['traj']; own = owner[t]; C = comps[t]
        hit = {own[f] for f, *_ in r['frames'] if f in own}
        if r['kind'] == 'filtered' and r['frames']:
            # Filtered chunks of one trajectory are mutually consistent (checked: no disagreement on overlaps), so
            # adjacent chunks are joined too; counterfactual records join only through verified overlap.
            for f in (r['frames'][0][0] - 1, r['frames'][-1][0] + 1):
                if f in own and f in filtered_frames[t]: hit.add(own[f])
        ov = [np.hypot(C[own[f]][f][0] - x, C[own[f]][f][1] - y) for f, x, y, *_ in r['frames'] if f in own]
        if ov and max(ov) > TOL:
            stats[f"rejected_inconsistent_{r['kind']}"] += 1; mism.append(max(ov)); continue
        if hit:
            k = min(hit)
            for j in sorted(hit - {k}):  # connect components through this record
                for f, v in C[j].items(): C[k][f] = v; own[f] = k
                C[j] = {}
        else:
            C.append({}); k = len(C) - 1
        for f, x, y, yw, nm, rr, i in r['frames']:
            if f not in own: C[k][f] = (x, y, yw, nm, rr, i); own[f] = k
            if r['kind'] == 'filtered': filtered_frames[t].add(f)
        stats[f"accepted_{r['kind']}"] += 1; stats[f"accepted_{r['kind']}_{'overlap' if hit else 'new_component'}"] += 1
        norms.setdefault(t, r['norm'])
        if 'cf' in r: cfs.append(dict(traj=t, record=r['record'], lang=r['lang'], norm=r['norm'], **r['cf']))
        elif r['lang']: langs.append(dict(traj=t, first=r['start'], last=r['start'] + len(r['frames']) - 1, lang=r['lang'], kind=r['kind']))
    for r in recs: stats[f"real_frames_{r['kind']}"] += len(r['frames'])
    stats.update({f"records_{k}": v for k, v in collections.Counter(r['kind'] for r in recs).items()})
    mismatch = np.asarray(mism)
    print(json.dumps(dict(stats)), flush=True)
    # Build episodes.
    assign = collections.defaultdict(dict)  # shard name -> {(rec, step): path}
    shard_of = {nm: (k, tp, o, s) for k, tp, nm, o, s in jobs}
    ep_index = collections.defaultdict(list)  # traj -> [(episode key, orig frames)]
    per_source = collections.defaultdict(collections.Counter)
    pieces = [(traj, ci, d) for traj, C in sorted(comps.items()) for ci, d in enumerate(C) if d]
    for traj, ci, d in pieces:
        src = source_of(traj, norms.get(traj, 0))
        if src == 'recon': per_source['recon']['dropped_frames'] += len(d); continue
        k = len(ep_index[traj])
        fr = np.array(sorted(d)); P = np.array([d[f][:2] for f in fr]); Y = rc.wrap(np.array([d[f][2] for f in fr]))
        # Split at missing frames that leave a jump > 1.0 m or more than 2 missing frames, and at pose jumps > 1.5 m.
        step = np.linalg.norm(np.diff(P, axis=0), axis=1); gapf = np.diff(fr)
        cut = (gapf > 3) | ((gapf > 1) & (step > 1.0)) | (step > 1.5)
        edges = [0] + (np.flatnonzero(cut) + 1).tolist() + [len(fr)]
        for s, e in zip(edges[:-1], edges[1:]):
            if e - s < 8: per_source[src]['short_pieces'] += 1; continue
            keep = rc.subsample(P[s:e], Y[s:e]) + s
            if len(keep) < 8: per_source[src]['short_pieces'] += 1; continue
            key = f'{traj}_{k}'; k += 1
            dd = out / key; dd.mkdir(parents=True, exist_ok=True)
            for i, j in enumerate(keep):
                x, y, yw, nm, r, st = d[fr[j]]
                assign[nm][(r, st)] = str(dd / f'{i}.jpg')
            np.savez(dd / 'trajectory.npz', position=P[keep], yaw=Y[keep], frame=fr[keep])
            (dd / 'source.json').write_text(json.dumps(dict(
                original=f'CAST:{traj} (GNM {src} trajectory), original frames {int(fr[keep[0]])}..{int(fr[keep[-1]])}',
                group=group_of(src, traj), source_dataset=src, gnm_trajectory=traj, metric=True, label_source='odometry',
                label_note='positions/yaw as stored by CAST from the GNM-processed trajectory (robot odometry / GPS-fused odometry per source dataset)',
                camera=dict(description=CAMERA.get(src, 'unknown'), stored='128x128 RGB (resized by CAST/GNM preprocessing)'),
                license=LICENSE, fps=None, frame_spacing_note=f'GNM sampling; nominal spacing {norms.get(traj, 0):.3g} m (CAST normalization factor)',
                cast_shards=len({d[fr[j]][3] for j in keep}))))
            ep_index[traj].append((key, fr[keep], ci))
            per_source[src]['episodes'] += 1; per_source[src]['frames'] += len(keep)
    # Language and counterfactual side files.
    lang_by, cf_by = collections.defaultdict(list), collections.defaultdict(list)
    for l in langs:
        cl = owner[l['traj']].get(l['first'])
        for key, frs, cj in ep_index.get(l['traj'], []):
            if cj != cl: continue
            i0, i1 = np.searchsorted(frs, l['first']), np.searchsorted(frs, l['last'], side='right') - 1
            if i1 >= i0 and l['lang']:
                lang_by[key].append(dict(first=int(i0), last=int(i1), instructions=l['lang'], source=l['kind']))
    cf_unmatched = 0
    for c in cfs:
        hit = None
        ci = owner[c['traj']].get(c['anchor'])
        for key, frs, cj in ep_index.get(c['traj'], []):
            if cj == ci and frs[0] <= c['anchor'] <= frs[-1]: hit = (key, frs); break
        if hit is None: cf_unmatched += 1; continue
        key, frs = hit
        j = int(np.searchsorted(frs, c['anchor'], side='right') - 1)  # last kept frame at or before the branch frame
        cf_by[key].append((j, c))
    for key, items in lang_by.items(): (out / key / 'language.json').write_text(json.dumps(items))
    for key, items in cf_by.items():
        M = max(len(c['pos']) for _, c in items)
        pos = np.full((len(items), M, 2), np.nan); yaw = np.full((len(items), M), np.nan)
        for q, (_, c) in enumerate(items):
            pos[q, :len(c['pos'])] = c['pos']; yaw[q, :len(c['yaw'])] = rc.wrap(np.asarray(c['yaw']))
        L = max(1, max(len(c['lang']) for _, c in items))
        lang = np.array([(c['lang'] + [''] * L)[:L] for _, c in items], dtype=str)
        with np.load(out / key / 'trajectory.npz') as z: kept = z['frame']
        np.savez(out / key / 'counterfactual.npz', anchor=np.array([j for j, _ in items], np.int32),
                 anchor_original_frame=np.array([c['anchor'] for _, c in items], np.int32),
                 anchor_is_kept=np.array([kept[j] == c['anchor'] for j, c in items]),
                 position=pos, yaw=yaw, length=np.array([len(c['pos']) for _, c in items], np.int32), language=lang,
                 record=np.array([c['record'] for _, c in items], dtype=str))
    # Pass 2: write images.
    ijobs = []
    for nm, asg in assign.items():
        kind, tp, o, sz = shard_of[nm]
        ijobs.append((kind, tp, nm, o, sz, asg))
    with ProcessPoolExecutor(a.workers) as ex:
        nimg = sum(ex.map(write_shard, ijobs, chunksize=1))
    print('images written', nimg, flush=True)
    cf_len = [len(c['pos']) for c in cfs]
    extra = dict(
        dataset='CAST (Counterfactual Augmentation with Synthetic Trajectories), real frames only',
        source='https://huggingface.co/datasets/catglossop/CAST-dataset (' + ', '.join(TARS[p] for p in a.parts.split(',')) + ')',
        label_source='odometry', metric=True, license=LICENSE,
        per_source_dataset={k: dict(v) for k, v in per_source.items()},
        raw_stats=dict(stats),
        consistency=dict(tolerance_m=TOL, rejected_records=int(len(mismatch)), rejected_max_disagreement_m=rc.quantiles(mismatch),
                         note='records whose real frames disagree (> tolerance) with overlapping accepted frames of the same trajectory (another CAST '
                              'version: ~0.1 m offsets or re-centred frames, mostly GO Stanford 2) are dropped; non-overlapping records form separate '
                              'components and episodes never mix components'),
        counterfactual=dict(records=len(cfs), unmatched_anchor=cf_unmatched, episodes_with=len(cf_by), steps=rc.quantiles(cf_len),
                            file='counterfactual.npz: anchor (K,) episode frame index of the last real frame at/before the branch, anchor_original_frame, '
                                 'anchor_is_kept, position (K,M,2) and yaw (K,M) of the synthetic continuation in the episode trajectory frame '
                                 '(NaN padded; length (K,)), language (K,L) instruction(s) the counterfactual was generated for, record (K,) CAST record name',
                            note='Synthetic: actions/positions produced by CAST counterfactual generation (VLM + policy), never executed; no images exist for them.'),
        language='language.json: [{first, last (episode-local frame indices, inclusive), instructions, source}] from filtered/atomic chunks',
        caveats=['All CAST frames are re-uses of GNM trajectories (Cory Hall, GO Stanford 2/4, SACSoN, Seattle, SCAND, TartanDrive); if the GNM datasets are '
                 'converted too, these episodes duplicate them at 128x128. source.json has source_dataset and gnm_trajectory/group so splits can follow the GNM split.',
                 'RECON trajectories (jackal_2019-*, normalization 0.25) are dropped.',
                 'Images are 128x128 (downscaled by CAST); fisheye for GO Stanford/SACSoN (360 camera front view).',
                 'Frame spacing follows the GNM sampling of each source (Cory Hall ~0.06 m ... TartanDrive ~0.7 m); no timestamps are released.',
                 'CAST chunks cover only parts of each trajectory; episodes are the contiguous covered stretches (many are short).',
                 'atomic_dataset.tar.gz (39 GB) not downloaded/used: 3-11-step snippets without original frame indices, same GNM trajectories.'])
    ready = rc.write_ready(out, extra)
    rc.previews(out)
    print(json.dumps(ready, indent=1))


if __name__ == '__main__':
    main()
