"""Convert a subset of BDD-with-MBRA (NHirose/BDD_OmniVLA) into realnav episodes, streaming from the remote zip.

The release has, per BDD100K clip (~40 s): 30 fps dash-cam frames downscaled to 96x96, the phone GPS
log at 1 Hz (gps.json: latitude, longitude, speed m/s, course deg, timestamp ms, img_id of the matching
frame), and (in BDD_dataset_pickle.zip) one MBRA action chunk per GPS sample. The executed car motion
comes from the GPS log; MBRA chunks are model re-annotations for a robot embodiment, stored apart.

Steps (all resumable):
  index : parse the zip central directory into index.npz; get the directory (4.1 GB, bytes 152832815075..156922106409 of
          the 156922106409-byte zip, from its zip64 end record) with
          python -m cross_nav.policy.datasets.parallel_download <zip url> <raw>/BDD_dataset.zip.central_directory --start 152832815075 --end 156922106409
  convert: pick --videos clips (seeded, at most --per_group per session prefix) and write episodes:
      <out>/<clip>_<k>/{i}.jpg            original JPEG bytes (96x96, not re-encoded)
      <out>/<clip>_<k>/trajectory.npz     position/yaw/t from GPS (cubic Hermite in time, speed+course tangents)
      <out>/<clip>_<k>/mbra.npz           MBRA chunks at the GPS frames: frame index, raw (8,4) [x,y,cos,sin]
      <out>/<clip>_<k>/source.json
python -m cross_nav.policy.datasets.bdd_mbra index --cd <raw>/BDD_dataset.zip.central_directory --out <raw>/index.npz
python -m cross_nav.policy.datasets.bdd_mbra convert --index <raw>/index.npz --pickles <raw>/BDD_dataset_pickle.zip --out <out> --videos 4000

The MBRA pickles are numpy arrays from the official dataset repo; they are read with an unpickler that
only admits numpy reconstruction.
"""
import argparse, collections, io, json, math, pickle, random, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cross_nav.policy.datasets import realnav_common as rc
from cross_nav.policy.datasets.remote_zip import fetch_span, member_bytes, parse_central_directory

URL = 'https://huggingface.co/datasets/NHirose/BDD_OmniVLA/resolve/main/BDD_dataset.zip'
ZIP_SIZE = 156922106409
LICENSE = ('BDD images: BDD / BAIR Open Research Commons license (UC Berkeley; research and not-for-profit use, '
           'copyright notice must be kept; see the LICENSE file of NHirose/BDD_OmniVLA). MBRA annotations: MIT.')


class NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split('.')[0] == 'numpy': return super().find_class(module, name)
        raise pickle.UnpicklingError(f'refusing {module}.{name}')


# ---------------------------------------------------------------- index
def build_index(cd_path, out):
    buf = open(cd_path, 'rb').read()
    vids, gps = {}, {}
    vi, iid, off, cs, me = [], [], [], [], []
    for name, o, c, u, m in parse_central_directory(buf):
        p = name.split('/')
        if len(p) == 3 and p[2] == 'gps.json':
            gps[p[1]] = (o, c, m)
        elif len(p) == 4 and p[2] == 'img' and p[3].endswith('.jpg'):
            k = vids.setdefault(p[1], len(vids))
            vi.append(k); iid.append(int(p[3][:-4])); off.append(o); cs.append(c); me.append(m)
    names = np.array(sorted(vids, key=vids.get))
    g = np.array([gps.get(n, (-1, 0, 0)) for n in names], np.int64)
    np.savez(out, names=names, gps=g, vid=np.array(vi, np.int32), img_id=np.array(iid, np.int32),
             offset=np.array(off, np.int64), csize=np.array(cs, np.int64), method=np.array(me, np.int8))
    print(json.dumps({'videos': len(names), 'images': len(vi), 'with_gps': int((g[:, 0] >= 0).sum())}))


# ---------------------------------------------------------------- geometry
def latlon_to_local(lat, lon):
    """Local tangent-plane metres (x east, y north) around the first fix (accurate to <0.1% over a few km)."""
    R = 6378137.0
    lat0, lon0 = math.radians(lat[0]), math.radians(lon[0])
    la, lo = np.radians(lat), np.radians(lon)
    e2 = 6.69437999014e-3
    rn = R / np.sqrt(1 - e2 * np.sin(lat0) ** 2)            # prime vertical radius
    rm = R * (1 - e2) / (1 - e2 * np.sin(lat0) ** 2) ** 1.5  # meridian radius
    return np.stack(((lo - lon0) * rn * np.cos(lat0), (la - lat0) * rm), -1)


def hermite(t, t0, t1, p0, p1, v0, v1):
    h = t1 - t0; s = (t - t0) / h
    h00, h10, h01, h11 = 2*s**3 - 3*s**2 + 1, s**3 - 2*s**2 + s, -2*s**3 + 3*s**2, s**3 - s**2
    pos = h00[:, None] * p0 + h10[:, None] * h * v0 + h01[:, None] * p1 + h11[:, None] * h * v1
    d00, d10, d01, d11 = 6*s**2 - 6*s, 3*s**2 - 4*s + 1, -6*s**2 + 6*s, 3*s**2 - 2*s
    vel = (d00[:, None] * p0 + d10[:, None] * h * v0 + d01[:, None] * p1 + d11[:, None] * h * v1) / h
    return pos, vel


def gps_track(gps):
    """Per-frame position/yaw/time from the 1 Hz GPS log. Returns frame ids, pos, yaw, t, segment ranges, diagnostics."""
    gps = sorted((g for g in gps if all(k in g for k in ('latitude', 'longitude', 'timestamp', 'img_id'))), key=lambda g: g['img_id'])
    if len(gps) < 3: return None
    lat = np.array([g['latitude'] for g in gps]); lon = np.array([g['longitude'] for g in gps])
    ts = np.array([g['timestamp'] for g in gps], np.float64) / 1000.0
    fid = np.array([g['img_id'] for g in gps])
    spd = np.array([g.get('speed', -1.0) if g.get('speed') is not None else -1.0 for g in gps], np.float64)
    crs = np.array([g.get('course', -1.0) if g.get('course') is not None else -1.0 for g in gps], np.float64)
    P = latlon_to_local(lat, lon)
    # Displacement-based heading/speed as fallback where course/speed are missing.
    dP = np.gradient(P, ts, axis=0) if len(ts) > 1 else np.zeros_like(P)
    yaw_c = np.where(crs >= 0, np.pi / 2 - np.radians(crs), np.arctan2(dP[:, 1], dP[:, 0]))
    v = np.where(spd >= 0, spd, np.linalg.norm(dP, axis=1))
    V = v[:, None] * np.stack((np.cos(yaw_c), np.sin(yaw_c)), -1)
    # Segment boundaries: GPS time gaps, non-monotonic ids, implausible jumps (>60 m/s).
    dt = np.diff(ts); dd = np.linalg.norm(np.diff(P, axis=0), axis=1)
    bad = (dt <= 0) | (dt > 2.5) | (np.diff(fid) <= 0) | (dd / np.maximum(dt, 1e-3) > 60)
    frames, pos, yaw, tt, seg = [], [], [], [], []
    for i in range(len(gps) - 1):
        if bad[i]:
            seg.append(len(frames)); continue
        f = np.arange(fid[i], fid[i + 1])
        tf = ts[i] + (f - fid[i]) / (fid[i + 1] - fid[i]) * dt[i]
        p, vel = hermite(tf, ts[i], ts[i + 1], P[i], P[i + 1], V[i], V[i + 1])
        sp = np.linalg.norm(vel, axis=1)
        # Heading: motion direction when moving, else interpolated course.
        dy = math.atan2(math.sin(yaw_c[i + 1] - yaw_c[i]), math.cos(yaw_c[i + 1] - yaw_c[i]))
        y_course = yaw_c[i] + (tf - ts[i]) / dt[i] * dy
        y = np.where(sp > 1.0, np.arctan2(vel[:, 1], vel[:, 0]), y_course)
        frames += f.tolist(); pos += p.tolist(); yaw += y.tolist(); tt += tf.tolist()
    seg.append(len(frames))
    # Independent heading check: GPS course vs direction of displacement between consecutive fixes (moving > 3 m/s).
    mv = (~bad) & (v[:-1] > 3) & (crs[:-1] >= 0)
    ang = np.degrees(np.abs(rc.wrap(np.arctan2(np.diff(P[:, 1]), np.diff(P[:, 0])) - (yaw_c[:-1] + rc.wrap(yaw_c[1:] - yaw_c[:-1]) / 2))))[mv]
    diag = dict(n_gps=len(gps), course_vs_displacement_deg=ang.tolist(), gps_speed_vs_displacement=(np.median(dd[~bad] / dt[~bad]) / max(np.median(v[:-1][~bad]), 1e-3)) if (~bad).any() and np.median(v) > 1 else None,
                accuracy_m=float(np.median([g.get('accuracy', np.nan) for g in gps])))
    return np.array(frames), np.array(pos).reshape(-1, 2), rc.wrap(np.array(yaw)), np.array(tt), sorted(set(seg)), diag, fid


# ---------------------------------------------------------------- convert
_IDX = None


def _load_index(path):
    global _IDX
    if _IDX is None:
        z = np.load(path)
        _IDX = {k: z[k] for k in z.files}
        _IDX['start'] = np.searchsorted(_IDX['vid'], np.arange(len(_IDX['names']) + 1))
    return _IDX


_PKZ = {}


def _pickle_zip(path):
    if path not in _PKZ: _PKZ[path] = zipfile.ZipFile(path)  # parse the 1.1M-entry directory once per worker
    return _PKZ[path]


def convert_video(args):
    try:
        return _convert_video(args)
    except Exception as e:  # network failure etc.: not logged as done, so a rerun retries the clip
        return dict(video=str(_load_index(args[0])['names'][args[1]]), status='error', error=repr(e)[:300])


def _convert_video(args):
    index_path, k, pickles, out, raw_dir = args
    I = _load_index(index_path)
    name = str(I['names'][k])
    a, b = I['start'][k], I['start'][k + 1]
    img_id, off, cs, me = I['img_id'][a:b], I['offset'][a:b], I['csize'][a:b], I['method'][a:b]
    go, gc, gm = I['gps'][k]
    if go < 0: return dict(video=name, status='no_gps')
    offs = np.r_[off, go]; css = np.r_[cs, gc]
    span, s0 = fetch_span(URL, offs, css, ZIP_SIZE)
    gps = json.loads(member_bytes(span, s0, int(go), int(gc), int(gm)))
    (Path(raw_dir) / f'{name}.gps.json').write_text(json.dumps(gps))
    tr = gps_track(gps)
    if tr is None: return dict(video=name, status='short_gps')
    frames, pos, yaw, t, segs, diag, gfid = tr
    have = dict(zip(img_id.tolist(), range(len(img_id))))
    # MBRA chunks keyed by the GPS frame id.
    mbra = {}
    z = _pickle_zip(pickles)
    for f in gfid:
        try:
            data = z.read(f'BDD_dataset_pickle/{name}/pickle/{int(f)}.pkl')
        except KeyError:
            continue
        if data: mbra[int(f)] = np.asarray(NumpyOnly(io.BytesIO(data)).load(), np.float64).reshape(-1, 4)
    eps, bounds = [], [0] + segs
    for s, e in zip(bounds[:-1], bounds[1:]):
        f, p, y, tt = frames[s:e], pos[s:e], yaw[s:e], t[s:e]
        ok = np.array([int(x) in have for x in f], bool)
        f, p, y, tt = f[ok], p[ok], y[ok], tt[ok]
        for a2, b2 in rc.split_at_jumps(p, tt, max_jump=1.5, max_gap=0.5):
            if b2 - a2 < 10: continue
            keep = rc.subsample(p[a2:b2], y[a2:b2]) + a2
            if len(keep) < 10: continue
            eps.append((f[keep], p[keep], y[keep], tt[keep]))
    written = []
    for j, (f, p, y, tt) in enumerate(eps):
        key = f'{name}_{j}'
        d = Path(out) / key; d.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(f):
            q = have[int(fr)]
            (d / f'{i}.jpg').write_bytes(member_bytes(span, s0, int(off[q]), int(cs[q]), int(me[q])))
        np.savez(d / 'trajectory.npz', position=p - p[0], yaw=y, t=tt - tt[0])
        m_frames = [i for i, fr in enumerate(f) if int(fr) in mbra]
        if m_frames:
            np.savez(d / 'mbra.npz', frame=np.array(m_frames, np.int32), img_id=np.array([int(f[i]) for i in m_frames], np.int32),
                     raw=np.stack([mbra[int(f[i])] for i in m_frames]))
        (d / 'source.json').write_text(json.dumps(dict(
            original=f'BDD_dataset.zip:BDD_dataset/{name} (BDD100K clip {name}), frames img/{int(f[0])}..{int(f[-1])}.jpg',
            group=name.split('-')[0], source_dataset='bdd100k', clip=name, metric=True, label_source='gps',
            label_note='phone GPS at 1 Hz (lat/lon, speed, course); cubic Hermite interpolation to 30 fps frames using speed+course tangents; '
                       'yaw = motion direction when > 1 m/s, else GPS course. MBRA action chunks (model re-annotation, not executed) are in mbra.npz.',
            gps_accuracy_m=diag['accuracy_m'],
            camera=dict(model='dash-cam / phone camera (BDD100K), mounted behind the windshield ~1.2-1.5 m high', native='1280x720 (16:9)',
                        stored='96x96: the full frame resized to a square (horizontally squeezed), as released by BDD_OmniVLA', fisheye=False),
            license=LICENSE, fps=30, embodiment='car')))
        written.append(dict(key=key, n=len(f)))
    return dict(video=name, status='ok' if written else 'no_episode', episodes=written, diag=diag)


def select_videos(index_path, n, per_group, seed):
    I = np.load(index_path)
    names = I['names']; g = I['gps']
    rng = random.Random(seed)
    order = list(range(len(names))); rng.shuffle(order)
    used, pick = collections.Counter(), []
    for k in order:
        if g[k, 0] < 0: continue
        grp = str(names[k]).split('-')[0]
        if used[grp] >= per_group: continue
        used[grp] += 1; pick.append(k)
        if len(pick) >= n: break
    return pick


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('index'); a1.add_argument('--cd', required=True); a1.add_argument('--out', required=True)
    a2 = sub.add_parser('convert')
    a2.add_argument('--index', required=True); a2.add_argument('--pickles', required=True); a2.add_argument('--out', required=True)
    a2.add_argument('--raw_gps', default=None, help='where to keep each clip gps.json (default <index dir>/gps)')
    a2.add_argument('--videos', type=int, default=4000); a2.add_argument('--per_group', type=int, default=2)
    a2.add_argument('--seed', type=int, default=0); a2.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    if a.cmd == 'index': return build_index(a.cd, a.out)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    raw_gps = Path(a.raw_gps or Path(a.index).parent / 'gps'); raw_gps.mkdir(parents=True, exist_ok=True)
    pick = select_videos(a.index, a.videos, a.per_group, a.seed)
    log = out / 'convert_log.jsonl'
    done = set()
    if log.exists():
        for line in log.read_text().splitlines(): done.add(json.loads(line)['video'])
    names = np.load(a.index)['names']
    jobs = [(a.index, k, a.pickles, str(out), str(raw_gps)) for k in pick if str(names[k]) not in done]
    print(f'selected {len(pick)} clips, {len(jobs)} to do', flush=True)
    status = collections.Counter()
    with open(log, 'a') as f, ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(convert_video, jobs, chunksize=4)):
            status[r['status']] += 1
            if r['status'] == 'error': print('error', r, flush=True); continue
            f.write(json.dumps(r) + '\n'); f.flush()
            if i % 200 == 0: print(i, dict(status), flush=True)
    finish(out, a)


def finish(out, a):
    rows = [json.loads(l) for l in (out / 'convert_log.jsonl').read_text().splitlines()]
    ratio = [r['diag']['gps_speed_vs_displacement'] for r in rows if r.get('diag') and r['diag'].get('gps_speed_vs_displacement')]
    course = [x for r in rows if r.get('diag') for x in r['diag'].get('course_vs_displacement_deg', [])]
    # MBRA vs executed motion: path length of the raw chunk vs the car's travel over the next second.
    mb_len, car_1s = [], []
    for ep in sorted(p.parent for p in out.glob('*/mbra.npz'))[:3000]:
        with np.load(ep / 'mbra.npz') as z: fr, raw = z['frame'], z['raw']
        with np.load(ep / 'trajectory.npz') as z: pos, t = z['position'], z['t']
        for i, r in zip(fr, raw):
            j = np.searchsorted(t, t[i] + 1.0)
            if j < len(t):
                mb_len.append(float(np.linalg.norm(np.diff(np.r_[[[0, 0]], r[:, :2]], axis=0), axis=1).sum()))
                car_1s.append(float(np.linalg.norm(np.diff(pos[i:j + 1], axis=0), axis=1).sum()))
    mb_len, car_1s = np.array(mb_len), np.array(car_1s)
    corr = float(np.corrcoef(mb_len, car_1s)[0, 1]) if len(mb_len) > 10 else None
    extra = dict(
        dataset='BDD100K clips with MBRA annotations (NHirose/BDD_OmniVLA), subset',
        source='https://huggingface.co/datasets/NHirose/BDD_OmniVLA (BDD_dataset.zip read by HTTP range requests; BDD_dataset_pickle.zip)',
        subset=f'{a.videos} clips drawn with seed {a.seed}, at most {a.per_group} per session prefix, out of 28641 clips; extend with a larger --videos (same seed keeps the existing picks)',
        label_source='gps', metric=True, license=LICENSE,
        clip_status=dict(collections.Counter(r['status'] for r in rows)),
        gps_displacement_speed_over_reported_speed=rc.quantiles(ratio),
        gps_course_vs_displacement_direction_deg=rc.quantiles(course),
        heading_note='heading_vs_motion_deg is ~0 by construction (yaw = interpolated motion direction when moving); the independent check is gps_course_vs_displacement_direction_deg',
        mbra=dict(file='mbra.npz per episode: frame (index into the episode frames), img_id, raw (K,8,4) [x, y, cos, sin] normalized MBRA action chunk',
                  note='MBRA (model-based re-annotation) actions generated by a model re-trained on BDD for a small-robot embodiment; not executed motion. '
                       'OmniVLA uses them as robot action labels with 0.25 m normalization (unverified for BDD).',
                  raw_path_length_units=rc.quantiles(mb_len), car_travel_next_1s_m=rc.quantiles(car_1s), corr_raw_length_vs_car_speed=corr),
        camera='BDD dash-cam, ~1.2-1.5 m high behind the windshield, released at 96x96 (full 16:9 frame squeezed to square); JPEG bytes copied unchanged',
        caveats=['Car embodiment: road scenes, 5-30 m/s, camera much higher than a ground robot.',
                 'Images are only 96x96 with the aspect ratio squeezed from 16:9; upsampling for the backbone will be blurry.',
                 'Positions come from 1 Hz phone GPS (reported accuracy ~5-10 m absolute); relative motion is smoother than that but lateral detail inside 1 s is interpolated.',
                 'Frame spacing exceeds the 0.3 m target above ~9 m/s (30 fps source); slow/stationary frames are dropped.',
                 'Group = first 8 hex chars of the BDD clip name (session prefix).'])
    ready = rc.write_ready(out, extra)
    rc.previews(out)
    print(json.dumps(ready, indent=1))


if __name__ == '__main__':
    main()
