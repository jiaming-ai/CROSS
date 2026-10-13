"""FrodoBots-2K (Earth Rover sidewalk robots, teleoperated in many cities) -> realnav episodes.

Source: https://huggingface.co/datasets/BitRobot/FrodoBots-2K (formerly frodobots/FrodoBots-2K; CC-BY-SA-4.0).
The raw data are 24 zip files on S3 (~898 GB, listed in complete-dataset.csv). Each ride directory
``output_rides_<p>/ride_<id>_<YYYYmmddHHMMSS>/`` holds control_data (10 Hz commands + 4 wheel RPMs),
gps_data (1 Hz), imu_data (100 Hz accelerometer, ~1 Hz gyroscope and magnetometer), camera frame
timestamps and HLS video segments (front camera = ``uid_s_1000`` 1024x576 ~20 fps, rear = ``uid_s_1001``).

We never download whole zips: members are read with HTTP range requests (the zip central directory
gives each member's offset), so only the CSVs and the front-camera segments recorded while the robot
moves are fetched.

Pipeline (each step is resumable):
  index    central directories of the 24 zips -> <raw>/zip_listing.json
  fetch    --kind meta  : control/gps/front-timestamp CSVs of every ride (small)
  survey   per-ride motion/GPS/bytes summary -> <raw>/survey.json
  select   rides spread uniformly over zip parts and geographic cells under a byte budget -> <raw>/selection.json
  fetch    --kind imu|video for the selected rides (video: only front segments that overlap motion)
  convert  poses + decoded/subsampled frames -> <out>/<episode>/
  ready    READY.json + preview contact sheets

Poses: wheel odometry (mean of the 4 wheel RPMs times a wheel circumference calibrated against GPS over
long windows) for travelled distance, yaw from the differential wheel speed scaled per ride against the
gyroscope yaw rate, with the slowly varying heading drift corrected by GPS course (see ``estimate_poses``).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import struct
import threading
import time
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

BASE_URL = 'https://frodobots-2k-dataset.s3.ap-southeast-1.amazonaws.com/output_rides_{}.zip'
N_PARTS = 24
FRONT = '1000'
# The wheel RPM stream is ~0.3 s early relative to the camera and the IMU: on pilot rides, rotation-induced horizontal
# image shift (phase correlation) lags the wheel yaw rate by a median 0.3 s (0.2-0.45 s on well-correlated rides) and
# the gyroscope lags it by 0.2-0.5 s. Control timestamps are shifted by this offset.
CONTROL_TIME_OFFSET_S = 0.3
# The IMU (gyroscope, magnetometer) clock is ~0.5 s early relative to the camera: gyroscope yaw rate leads the image
# rotation by a median 0.45 s and the (shifted) wheel yaw rate by 0.65 s on the same rides.
IMU_TIME_OFFSET_S = 0.5
MAX_SEGMENTS = 50  # ~14 min of front video per ride at most
LICENSE = 'CC-BY-SA-4.0 (FrodoBots-2K, FrodoBots Lab)'
CAMERA = {'model': 'Earth Rover (FrodoBots, 2024 fleet) front camera, HLS H.264 stream', 'resolution': [1024, 576], 'fps_nominal': 20,
          'fov': 'not published; wide-angle. Rough estimate from rotation-induced image shift vs yaw rate on pilot rides: '
                 'fx ~ 370-400 px at 1024x576 (HFOV ~ 100-105 deg near the centre)',
          'fisheye': 'no, but visible barrel distortion', 'intrinsics': None,
          'height_above_ground_m': 'not published (small sidewalk rover; parts of the rover body are visible at the bottom of the image)'}


# --------------------------------------------------------------------------------------------- HTTP
def _session():
    import requests
    s = requests.Session()
    s.mount('https://', requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=64, max_retries=3))
    return s


_tls = threading.local()


def _sess():
    if not hasattr(_tls, 's'): _tls.s = _session()
    return _tls.s


def get_range(url, start, end, retries=10):
    """Bytes [start, end) of url."""
    out = bytearray()
    for k in range(retries):
        try:
            r = _sess().get(url, headers={'Range': f'bytes={start + len(out)}-{end - 1}'}, timeout=(30, 120), stream=True)
            if r.status_code not in (200, 206): raise IOError(f'HTTP {r.status_code}')
            for b in r.iter_content(1 << 20): out += b
            if len(out) >= end - start: return bytes(out[:end - start])
        except Exception:  # noqa: BLE001 - network hiccup, retry from where we stopped
            if k == retries - 1: raise
            time.sleep(min(60, 2 + 4 * k))
    raise IOError(f'range {start}-{end} of {url} incomplete')


def get_range_parallel(url, start, end, chunk=4 << 20, workers=16):
    parts = [(s, min(s + chunk, end)) for s in range(start, end, chunk)]
    with ThreadPoolExecutor(workers) as ex:
        return b''.join(ex.map(lambda se: get_range(url, *se), parts))


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile to parse the central directory)."""

    def __init__(self, url):
        self.url = url
        self.size = int(_sess().head(url, timeout=60).headers['Content-Length'])
        self.pos = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0: return b''
        data = get_range_parallel(self.url, self.pos, self.pos + n) if n > (8 << 20) else get_range(self.url, self.pos, self.pos + n)
        self.pos += len(data)
        return data


def fetch_member(url, m):
    """Download and decompress one zip member described by a listing row (name, size, csize, ctype, offset, crc)."""
    name, size, csize, ctype, off, crc = m
    want = 30 + len(name.encode()) + 256 + csize
    buf = get_range(url, off, off + want)
    sig, _, _, _, _, _, _, _, _, nlen, elen = struct.unpack('<IHHHHHIIIHH', buf[:30])
    if sig != 0x04034b50: raise IOError(f'bad local header for {name}')
    a = 30 + nlen + elen
    if a + csize > len(buf): buf += get_range(url, off + len(buf), off + a + csize)
    raw = buf[a:a + csize]
    data = zlib.decompressobj(-15).decompress(raw) if ctype == 8 else raw
    if len(data) != size or (zlib.crc32(data) & 0xffffffff) != crc: raise IOError(f'size/crc mismatch for {name}')
    return data


# ------------------------------------------------------------------------------------------ listing
def cmd_index(a):
    raw = Path(a.raw); raw.mkdir(parents=True, exist_ok=True)
    path = raw / 'zip_listing.json'
    listing = json.loads(path.read_text()) if path.exists() else {}
    for p in range(N_PARTS):
        if str(p) in listing: continue
        t0 = time.time()
        with zipfile.ZipFile(RangeFile(BASE_URL.format(p))) as z:
            listing[str(p)] = [[i.filename, i.file_size, i.compress_size, i.compress_type, i.header_offset, i.CRC]
                               for i in z.infolist() if not i.filename.endswith('/')]
        path.write_text(json.dumps(listing))
        print(f'part {p}: {len(listing[str(p)])} members ({time.time() - t0:.0f} s)', flush=True)


def rides_from_listing(listing):
    """{ride_dir: {'part': p, 'members': {kind: [...]}, 'front': [(t_start, member), ...]}}"""
    rides = {}
    for p, members in listing.items():
        for m in members:
            parts = m[0].split('/')
            if len(parts) < 3 or not parts[1].startswith('ride_'): continue
            r = rides.setdefault(parts[1], dict(part=int(p), csv={}, front=[], playlist=None, prefix=None))
            fn = parts[-1]
            if len(parts) == 3 and fn.endswith('.csv'):
                r['csv'][re.sub(r'_\d+\.csv$', '', fn)] = m
            elif f'uid_s_{FRONT}__uid_e_video' in fn:
                if fn.endswith('.m3u8'): r['playlist'] = m
                elif fn.endswith('.ts'):
                    ts = fn.rsplit('_', 1)[-1][:-3]  # YYYYmmddHHMMSSfff (UTC)
                    t = datetime.strptime(ts[:14], '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc).timestamp() + int(ts[14:]) / 1000
                    r['front'].append((t, m))
                    r['prefix'] = fn.split('_ride_')[0]
    for r in rides.values(): r['front'].sort(key=lambda x: x[0])
    return rides


def ride_dir(raw, name, part):
    return Path(raw) / 'rides' / f'output_rides_{part}' / name


# ------------------------------------------------------------------------------------------ fetching
def _write_member(url, m, dest):
    if dest.exists() and dest.stat().st_size == m[1]: return 0
    data = fetch_member(url, m)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.part'); tmp.write_bytes(data); tmp.rename(dest)
    return len(data)


def moving_intervals(rd, pad=3.0, rpm_thr=3.0):
    """Time intervals (unix s) where any wheel turns, padded, from control_data.csv."""
    import pandas as pd
    c = pd.read_csv(next(rd.glob('control_data*.csv')))
    t = c.timestamp.values.astype(float)
    mv = np.abs(c[['rpm_1', 'rpm_2', 'rpm_3', 'rpm_4']].values).max(1) > rpm_thr
    iv = []
    for i in np.flatnonzero(mv):
        if iv and t[i] - pad <= iv[-1][1]: iv[-1][1] = t[i] + pad
        else: iv.append([t[i] - pad, t[i] + pad])
    return iv


def front_segments(r, rd=None, moving_only=True, max_segments=MAX_SEGMENTS, name=''):
    """Front-camera segment members (with start times) to download.

    moving_only: only segments overlapping wheel motion (from control_data). max_segments: cap per ride (a
    contiguous run of the moving segments at a deterministic pseudo-random offset), so the byte budget is spread
    over many rides instead of a few long ones (segments are ~16.5 s)."""
    segs = r['front']
    if moving_only and rd is not None:
        iv = moving_intervals(rd)
        out = []
        for k, (t, m) in enumerate(segs):
            e = segs[k + 1][0] if k + 1 < len(segs) else t + 20.0
            if any(a < e and b > t for a, b in iv): out.append((t, m))
        segs = out
    if max_segments and len(segs) > max_segments:
        o = zlib.crc32(name.encode()) % (len(segs) - max_segments + 1)
        segs = segs[o:o + max_segments]
    return segs


def cmd_fetch(a):
    raw = Path(a.raw)
    rides = rides_from_listing(json.loads((raw / 'zip_listing.json').read_text()))
    names = sorted(rides)
    if a.kind in ('imu', 'video'):
        names = json.loads((raw / 'selection.json').read_text())['rides']
    if a.limit: names = names[:a.limit]
    jobs = []
    for n in names:
        r = rides[n]; rd = ride_dir(raw, n, r['part']); url = BASE_URL.format(r['part'])
        if a.kind == 'meta':
            for k in ('control_data', 'gps_data', 'front_camera_timestamps'):
                if k in r['csv']: jobs.append((url, r['csv'][k], rd / r['csv'][k][0].split('/')[-1]))
            if r['playlist']: jobs.append((url, r['playlist'], rd / 'recordings' / r['playlist'][0].split('/')[-1]))
        elif a.kind == 'imu':
            if 'imu_data' in r['csv']: jobs.append((url, r['csv']['imu_data'], rd / r['csv']['imu_data'][0].split('/')[-1]))
        else:
            for t, m in front_segments(r, rd, moving_only=not a.all_segments, name=n):
                jobs.append((url, m, rd / 'recordings' / m[0].split('/')[-1]))
    total = sum(j[1][2] for j in jobs)
    print(f'{a.kind}: {len(names)} rides, {len(jobs)} members, {total / 1e9:.2f} GB compressed', flush=True)
    done = 0; nbytes = 0; fails = 0; t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_write_member, *j): j for j in jobs}
        for f in as_completed(futs):
            done += 1
            try: nbytes += f.result()
            except Exception as e:  # noqa: BLE001
                fails += 1; print('FAIL', futs[f][1][0], e, flush=True)
            if done % 200 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f'{done}/{len(jobs)} members, {nbytes / 1e9:.2f} GB new, {nbytes / 1e6 / max(el, 1):.1f} MB/s, {fails} failed', flush=True)
    print('FETCH_DONE', a.kind, flush=True)


# -------------------------------------------------------------------------------------------- survey
def latlon_to_xy(lat, lon, lat0, lon0):
    R = 6378137.0
    return np.stack([np.deg2rad(lon - lon0) * R * math.cos(math.radians(lat0)), np.deg2rad(lat - lat0) * R], -1)


def load_gps(rd):
    import pandas as pd
    g = pd.read_csv(next(rd.glob('gps_data*.csv')))
    t = g.timestamp.values.astype(float) / 1000.0
    lat, lon = g.latitude.values.astype(float), g.longitude.values.astype(float)
    ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1e-6) & (np.abs(lon) > 1e-6)
    return t[ok], lat[ok], lon[ok]


def survey_ride(name, r, raw):
    import pandas as pd
    rd = ride_dir(raw, name, r['part'])
    out = dict(ride=name, part=r['part'], prefix=r['prefix'], n_front_segments=len(r['front']),
               front_bytes=sum(m[2] for _, m in r['front']))
    try:
        c = pd.read_csv(next(rd.glob('control_data*.csv')))
        t = c.timestamp.values.astype(float)
        rpm = c[['rpm_1', 'rpm_2', 'rpm_3', 'rpm_4']].values.astype(float)
        dt = np.clip(np.diff(t, prepend=t[0]), 0, 0.5)
        mv = np.abs(rpm).max(1) > 3
        out.update(duration=float(t[-1] - t[0]) if len(t) else 0.0, moving_s=float(dt[mv].sum()),
                   wheel_revs=float(np.sum(np.abs(rpm.mean(1)) / 60 * dt)))
        segs = front_segments(r, rd, max_segments=0)
        out['moving_front_bytes'] = sum(m[2] for _, m in segs)
        out['moving_front_segments'] = len(segs)
        segs = front_segments(r, rd, name=name)
        out['capped_front_bytes'] = sum(m[2] for _, m in segs)
    except (StopIteration, Exception) as e:  # noqa: BLE001
        out['error'] = f'control: {e}'
    try:
        gt, lat, lon = load_gps(rd)
        if len(lat):
            out.update(lat=float(np.median(lat)), lon=float(np.median(lon)), gps_n=int(len(lat)))
            xy = latlon_to_xy(lat, lon, lat[0], lon[0])
            out['gps_path_m'] = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    except (StopIteration, Exception) as e:  # noqa: BLE001
        out['gps_error'] = str(e)
    return out


def geo_cell(lat, lon, deg=0.5):
    if lat is None or lon is None: return 'unknown'
    return f'{math.floor(lat / deg) * deg:+.1f}_{math.floor(lon / deg) * deg:+.1f}'


def cmd_survey(a):
    raw = Path(a.raw)
    rides = rides_from_listing(json.loads((raw / 'zip_listing.json').read_text()))
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda n: survey_ride(n, rides[n], raw), sorted(rides)))
    for row in rows: row['cell'] = geo_cell(row.get('lat'), row.get('lon'))
    (raw / 'survey.json').write_text(json.dumps(rows, indent=0))
    mv = [r for r in rows if r.get('moving_s', 0) > 0]
    print(f'{len(rows)} rides; {len(mv)} with motion; moving hours {sum(r["moving_s"] for r in mv) / 3600:.0f}; '
          f'front GB all {sum(r["front_bytes"] for r in rows) / 1e9:.0f}, moving-only {sum(r.get("moving_front_bytes", 0) for r in rows) / 1e9:.0f}')
    cells = defaultdict(lambda: [0, 0.0])
    for r in mv: cells[r['cell']][0] += 1; cells[r['cell']][1] += r['moving_s'] / 3600
    for c, (n, h) in sorted(cells.items(), key=lambda x: -x[1][1]): print(f'  cell {c}: {n} rides, {h:.1f} moving h')


# -------------------------------------------------------------------------------------------- select
def cmd_select(a):
    """Round-robin over geographic cells (and zip parts inside a cell), random rides, until the byte budget."""
    raw = Path(a.raw)
    rows = json.loads((raw / 'survey.json').read_text())
    ok = [r for r in rows if r.get('moving_s', 0) >= a.min_moving_s and r.get('gps_n', 0) >= 10 and 'error' not in r
          and r.get('capped_front_bytes', 0) > 0 and abs(r.get('lat', 999)) <= 90 and abs(r.get('lon', 999)) <= 180]
    rng = random.Random(a.seed)
    by_cell = defaultdict(list)
    for r in ok: by_cell[r['cell']].append(r)
    for c in by_cell: rng.shuffle(by_cell[c])
    chosen, total = [], 0
    budget = a.budget_gb * 1e9
    cells = sorted(by_cell)
    while total < budget and any(by_cell.values()):
        for c in cells:
            if not by_cell[c]: continue
            r = by_cell[c].pop()
            chosen.append(r); total += r['capped_front_bytes']
            if total >= budget: break
    sel = dict(seed=a.seed, budget_gb=a.budget_gb, min_moving_s=a.min_moving_s, total_gb=total / 1e9,
               rides=[r['ride'] for r in chosen], n_cells=len({r['cell'] for r in chosen}),
               moving_hours=sum(r['moving_s'] for r in chosen) / 3600,
               note='round-robin over 0.5-degree GPS cells (random ride per cell per round) until budget; front segments overlapping motion only, at most MAX_SEGMENTS per ride')
    (raw / 'selection.json').write_text(json.dumps(sel, indent=0))
    print(f"selected {len(chosen)} rides, {total / 1e9:.1f} GB, {sel['moving_hours']:.0f} moving h, {sel['n_cells']} cells")


# ------------------------------------------------------------------------------------- pose estimate
def parse_imu(rd):
    """compass (N,4), gyroscope (N,4), accelerometer (N,4): x, y, z, unix t."""
    import pandas as pd
    imu = pd.read_csv(next(rd.glob('imu_data*.csv')))
    num = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
    out = {}
    for col in ('compass', 'gyroscope', 'accelerometer'):
        arr = [np.array(num.findall(s), float).reshape(-1, 4) for s in imu[col].astype(str) if s and s != 'nan']
        arr = [x for x in arr if x.size]
        a = np.concatenate(arr) if arr else np.zeros((0, 4))
        a = a[np.argsort(a[:, 3], kind='stable')]
        _, u = np.unique(a[:, 3], return_index=True)
        a = a[u]
        a[:, 3] += IMU_TIME_OFFSET_S
        out[col] = a
    return out


def load_control(rd):
    import pandas as pd
    c = pd.read_csv(next(rd.glob('control_data*.csv')))
    c = c.sort_values('timestamp').drop_duplicates('timestamp')
    return c.timestamp.values.astype(float) + CONTROL_TIME_OFFSET_S, c[['rpm_1', 'rpm_2', 'rpm_3', 'rpm_4']].values.astype(float), \
        c[['linear', 'angular']].values.astype(float)


def wrap(x): return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def smooth(x, k):
    if k <= 1 or len(x) < k: return x
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode='edge'), np.ones(k) / k, mode='valid')


def gyro_yaw_rate(rd, t, rpm, imu=None):
    """Gyroscope yaw-rate samples (rad/s, bias removed) as (time, rate, bias deg/s); x axis = yaw per the dataset notes.

    The bias is the median rate while all wheels are still (it is several deg/s on some robots)."""
    g = (imu or parse_imu(rd))['gyroscope']
    if len(g) < 10: return np.zeros(0), np.zeros(0), None
    still = np.interp(g[:, 3], t, (np.abs(rpm).max(1) < 1).astype(float)) > 0.99
    bias = float(np.median(g[still, 0])) if still.sum() >= 5 else float(np.median(g[:, 0]))
    return g[:, 3], np.deg2rad(g[:, 0] - bias), bias


def compass_heading(imu, t0, t1):
    """Magnetometer heading (rad, up to a constant): atan2(-z, y) after centring the horizontal axes of the ride
    (hard iron; y = west, z = south per the dataset notes). Returns (times, heading)."""
    m = imu['compass']
    m = m[(m[:, 3] > t0) & (m[:, 3] < t1)]
    if len(m) < 10: return np.zeros(0), np.zeros(0)
    cy = (np.percentile(m[:, 1], 2) + np.percentile(m[:, 1], 98)) / 2
    cz = (np.percentile(m[:, 2], 2) + np.percentile(m[:, 2], 98)) / 2
    return m[:, 3], np.arctan2(-(m[:, 2] - cz), m[:, 1] - cy)


def circ_movmean(tm, o, win):
    """Circular moving mean of angles o (sorted times tm) over a centred window of win seconds."""
    c, s = np.r_[0, np.cumsum(np.cos(o))], np.r_[0, np.cumsum(np.sin(o))]
    lo = np.searchsorted(tm, tm - win / 2); hi = np.searchsorted(tm, tm + win / 2)
    return np.arctan2(s[hi] - s[lo], c[hi] - c[lo])


def fit_yaw_scale(tm, h, t, D, win=30.0, grid=np.arange(0.004, 0.0302, 0.0002)):
    """Scale k (rad per rpm*s of differential) that makes compass heading minus k*D(t) slowly varying.

    Returns (k, median deviation deg of the offsets from their win-second circular moving mean)."""
    Dm = np.interp(tm, t, D)
    best = (np.inf, None)
    for k in grid:
        o = wrap(h - k * Dm)
        dev = float(np.median(np.abs(wrap(o - circ_movmean(tm, o, win)))))
        if dev < best[0]: best = (dev, float(k))
    return best[1], math.degrees(best[0])


def wheel_health(rpm):
    """Relative disagreement of the two wheels on each side while moving (a faulty wheel or encoder gives > 0.15)."""
    mv = np.abs(rpm).max(1) > 20
    if mv.sum() < 50: return 0.0
    r = rpm[mv]; s = np.maximum(np.abs(r).mean(1), 1)
    return float(max(np.median(np.abs(r[:, 0] - r[:, 2]) / s), np.median(np.abs(r[:, 1] - r[:, 3]) / s)))


def estimate_poses(rd, circumference, rpm_clip=300.0, dt_max=0.5, drift_window_s=30.0, gps_course_min_speed=0.5):
    """Planar trajectory of a ride on the ~10 Hz control timeline.

    distance: wheel odometry, v = circumference * mean(rpm_1..4) / 60 (circumference calibrated against GPS).
    yaw:      k * D(t) + b(t), D = integral of the differential wheel speed (rpm_2 + rpm_4 - rpm_1 - rpm_3) / 2
              (right minus left; rpm_1 FL, rpm_2 FR, rpm_3 RL, rpm_4 RR). The per-ride scale k (absorbs skid-steer
              slip) and the slowly varying offset b(t) (circular mean over drift_window_s) come from the 1 Hz
              magnetometer heading (hard-iron centred per ride). If the compass is unusable, k comes from the
              gyroscope (or the population median) and b(t) from the GPS course (60 s windows).
              The absolute heading (x east, y north) is fixed by one constant fitted to the GPS course.
    The 1 Hz gyroscope is left as an independent check (info['gyro_r2']: fit of gyro yaw rate to the yaw rate).
    Returns dict(t, xy, yaw, v, w, gps_t, gps_xy, latlon0, info).
    """
    from scipy.ndimage import median_filter
    t, rpm, cmd = load_control(rd)
    bad = (np.abs(rpm) > rpm_clip).any(1)
    rpm[bad] = np.nan
    for j in range(4):
        x = rpm[:, j]; ok = np.isfinite(x)
        rpm[:, j] = np.interp(t, t[ok], x[ok]) if ok.any() else 0.0
    rpm = median_filter(rpm, size=(3, 1), mode='nearest')
    dt = np.clip(np.diff(t, prepend=t[0]), 0, dt_max)
    v = circumference * rpm.mean(1) / 60.0
    diff = (rpm[:, 1] + rpm[:, 3] - rpm[:, 0] - rpm[:, 2]) / 2.0
    D = np.cumsum(diff * dt)
    info = dict(n_control=int(len(t)), rpm_outliers=int(bad.sum()), control_gaps_gt1s=int((np.diff(t) > 1).sum()),
                wheel_disagreement=round(wheel_health(rpm), 3))
    imu = parse_imu(rd)
    tm, hc = compass_heading(imu, t[0], t[-1])
    k, dev = (fit_yaw_scale(tm, hc, t, D) if len(tm) >= 30 else (None, None))
    info.update(compass_samples=int(len(tm)), compass_yaw_scale=k, compass_dev_deg=None if dev is None else round(dev, 2))
    compass_ok = k is not None and dev < 8.0 and 0.006 <= k <= 0.026
    gt_, gw, bias = gyro_yaw_rate(rd, t, rpm, imu)
    kg, r2g = None, None
    if len(gt_) >= 10:
        d = np.interp(gt_ + GYRO_LAG_S, t, diff); m = np.abs(d) > 5
        if m.sum() >= 10:
            kg = float(np.sum(gw[m] * d[m]) / np.sum(d[m] ** 2))
            r2g = float(1 - np.mean((gw[m] - kg * d[m]) ** 2) / max(np.mean(gw[m] ** 2), 1e-12))
    info.update(gyro_bias_dps=bias, gyro_samples=int(len(gt_)), gyro_yaw_scale=kg, gyro_fit_r2=None if r2g is None else round(r2g, 3))
    if compass_ok: src = 'compass'
    elif kg is not None and r2g >= 0.4 and kg > 0: k, src = kg, 'gyro'
    else: k, src = DEFAULT_YAW_SCALE, 'population'
    info.update(yaw_scale=k, yaw_scale_source=src)
    w = k * diff
    yaw_dr = k * D
    gt, lat, lon = load_gps(rd)
    lat0, lon0 = (float(np.median(lat)), float(np.median(lon))) if len(lat) else (0.0, 0.0)
    gxy = latlon_to_xy(lat, lon, lat0, lon0) if len(lat) else np.zeros((0, 2))
    off = np.zeros_like(t)
    if compass_ok:
        o = wrap(hc - np.interp(tm, t, yaw_dr))
        off = np.interp(t, tm, np.unwrap(circ_movmean(tm, o, drift_window_s)))
    # GPS course: absolute constant (compass case) or slow drift (fallback)
    info['gps_course_samples'] = 0
    if len(gt) > 5:
        dg = np.diff(gxy, axis=0); gdt = np.diff(gt); mid = (gt[1:] + gt[:-1]) / 2
        spd = np.linalg.norm(dg, axis=1) / np.maximum(gdt, 1e-3)
        crs = np.arctan2(dg[:, 1], dg[:, 0])
        vw = np.interp(mid, t, smooth(v, 10)); ww = np.interp(mid, t, smooth(np.abs(w), 10))
        use = (spd > gps_course_min_speed) & (spd < 4) & (vw > 0.3) & (ww < 0.15) & (gdt < 2.5) & (mid > t[0]) & (mid < t[-1])
        info['gps_course_samples'] = int(use.sum())
        if use.sum() >= 5:
            o = wrap(crs[use] - np.interp(mid[use], t, yaw_dr + off))
            if compass_ok:
                c = math.atan2(np.sin(o).mean(), np.cos(o).mean())
                off = off + c
                info['gps_course_resid_deg_median'] = round(math.degrees(float(np.median(np.abs(wrap(o - c))))), 2)
            else:
                oo = np.unwrap(circ_movmean(mid[use], o, 60.0))
                off = np.interp(t, mid[use], oo)
                info['gps_course_resid_deg_median'] = round(math.degrees(float(np.median(np.abs(wrap(o - oo))))), 2)
    yaw = wrap(yaw_dr + off)
    xy = np.cumsum(np.stack([v * np.cos(yaw), v * np.sin(yaw)], -1) * dt[:, None], axis=0)
    if len(gt) > 5:
        m = (gt > t[0]) & (gt < t[-1])
        if m.any():
            gi = np.stack([np.interp(gt[m], t, xy[:, 0]), np.interp(gt[m], t, xy[:, 1])], -1)
            xy = xy + np.median(gxy[m] - gi, axis=0)
    # recheck gyro against the final yaw rate (independent of the compass/GPS fit)
    if len(gt_) >= 10:
        wr = np.interp(gt_ + GYRO_LAG_S, t, np.gradient(np.unwrap(yaw), t)); m = np.abs(wr) > 0.05
        if m.sum() >= 10:
            info['gyro_r2_final'] = round(float(1 - np.mean((gw[m] - wr[m]) ** 2) / max(np.mean(gw[m] ** 2), 1e-12)), 3)
    return dict(t=t, xy=xy, yaw=yaw, v=v, w=w, cmd=cmd, gps_t=gt, gps_xy=gxy, latlon0=(lat0, lon0), info=info)


GYRO_R2_MIN = 0.0  # reject rides whose yaw rate explains the gyroscope worse than zero (broken wheel/compass data)
WHEEL_FAULT = 0.15  # side-partner wheel disagreement above which a ride is rejected
GYRO_LAG_S = 0.0  # after CONTROL_TIME_OFFSET_S the gyroscope and the wheel differential are aligned


DEFAULT_YAW_SCALE = 0.0  # rad/s per rpm of differential; set from calibration.json at runtime


def gps_consistency(p, window_m=10.0, max_dt=60.0):
    """Odometry vs GPS displacement over windows of >= window_m odometry arc length (GPS samples inside the ride).

    Returns rows (|d_odo|, |d_gps|, |d_odo - d_gps|, angle(d_odo, d_gps) deg, odometry arc)."""
    t, xy, gt, gxy = p['t'], p['xy'], p['gps_t'], p['gps_xy']
    m = (gt > t[0]) & (gt < t[-1])
    gt, gxy = gt[m], gxy[m]
    if len(gt) < 5: return []
    oxy = np.stack([np.interp(gt, t, xy[:, 0]), np.interp(gt, t, xy[:, 1])], -1)
    arc = np.interp(gt, t, np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    rows, j, i = [], 0, 0
    while i < len(gt):
        j = max(j, i + 1)
        while j < len(gt) and arc[j] - arc[i] < window_m: j += 1
        if j >= len(gt): break
        if gt[j] - gt[i] <= max_dt:
            do, dg = oxy[j] - oxy[i], gxy[j] - gxy[i]
            ang = abs(math.degrees(wrap(math.atan2(do[1], do[0]) - math.atan2(dg[1], dg[0]))))
            rows.append((float(np.linalg.norm(do)), float(np.linalg.norm(dg)), float(np.linalg.norm(do - dg)), ang, float(arc[j] - arc[i])))
        i = j  # non-overlapping windows
    return rows


def cmd_qc(a):
    """GPS consistency of the odometry labels on converted rides -> <raw>/qc.json."""
    global DEFAULT_YAW_SCALE
    raw, out = Path(a.raw), Path(a.out)
    cal = json.loads((raw / 'calibration.json').read_text()); DEFAULT_YAW_SCALE = cal['yaw_scale']
    rides = rides_from_listing(json.loads((raw / 'zip_listing.json').read_text()))
    ok = []
    for line in (out / 'convert_log.jsonl').read_text().splitlines():
        d = json.loads(line)
        if d['status'] == 'ok': ok.append(d['ride'])
    rng = random.Random(0); names = rng.sample(ok, min(a.n, len(ok)))

    def one(n):
        p = estimate_poses(ride_dir(raw, n, rides[n]['part']), cal['circumference_m'])
        return p['info'], {w: gps_consistency(p, w) for w in (10.0, 30.0)}
    with ThreadPoolExecutor(a.workers) as ex: res = list(ex.map(one, names))
    q = lambda v, qs=(.1, .25, .5, .75, .9): [round(float(np.quantile(v, x)), 3) for x in qs] if len(v) else None
    rep = dict(n_rides=len(res), quantiles='q10,q25,q50,q75,q90')
    for w in (10.0, 30.0):
        rows = np.array([r for _, d in res for r in d[w]])
        if not len(rows): continue
        big = rows[:, 1] > 0.5 * w
        rep[f'window_{int(w)}m'] = dict(n=int(len(rows)), odo_len=q(rows[:, 0]), gps_len=q(rows[:, 1]),
                                        len_ratio_odo_over_gps=q(rows[big, 0] / rows[big, 1]),
                                        disp_error_m=q(rows[:, 2]), direction_error_deg=q(rows[big, 3]))
    infos = [i for i, _ in res]
    rep['gyro_r2_final'] = q([i['gyro_r2_final'] for i in infos if i.get('gyro_r2_final') is not None])
    rep['compass_dev_deg'] = q([i['compass_dev_deg'] for i in infos if i.get('compass_dev_deg') is not None])
    rep['yaw_scale'] = q([i['yaw_scale'] for i in infos])
    rep['yaw_scale_source'] = {k: sum(i['yaw_scale_source'] == k for i in infos) for k in ('compass', 'gyro', 'population')}
    rep['gps_course_resid_deg_median'] = q([i['gps_course_resid_deg_median'] for i in infos if i.get('gps_course_resid_deg_median') is not None])
    rep['note'] = ('non-overlapping windows of >= 10 / 30 m odometry arc; GPS (consumer, 1 Hz) is noisy at the few-metre level, '
                   'so 10 m windows mostly measure GPS noise. GPS enters the labels only through one global wheel circumference '
                   'and, for compass rides, one heading constant per ride (GPS-course drift correction only for non-compass rides), '
                   'so length ratios and direction errors are close to independent checks.')
    (raw / 'qc.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def _circ_windows(rd, win=30):
    """(GPS chord m, wheel revolutions) over win-second windows of straight forward driving."""
    t, rpm, _ = load_control(rd); gt, lat, lon = load_gps(rd)
    if len(gt) < win + 1: return []
    rpm = np.where(np.abs(rpm) > 300, 0, rpm)
    dt = np.clip(np.diff(t, prepend=t[0]), 0, 0.5)
    revs = np.cumsum(rpm.mean(1) / 60 * dt)
    diff = np.abs(rpm[:, 1] + rpm[:, 3] - rpm[:, 0] - rpm[:, 2]) / 2
    gxy = latlon_to_xy(lat, lon, lat[0], lon[0])
    out = []
    for i in range(0, len(gt) - win, win // 3):
        a0, a1 = gt[i], gt[i + win]
        if a1 - a0 > 1.3 * win: continue
        s = (t >= a0) & (t <= a1)
        if s.sum() < 5 * win or np.median(diff[s]) > 3 or np.mean(rpm[s].mean(1) > 5) < 0.95: continue
        rv = np.interp(a1, t, revs) - np.interp(a0, t, revs)
        if rv < 15: continue
        out.append((float(np.linalg.norm(gxy[i + win] - gxy[i])), float(rv)))
    return out


def cmd_calibrate(a):
    """Wheel circumference (m) from GPS chords of straight drives, and the population differential-to-yaw-rate scale."""
    global DEFAULT_YAW_SCALE
    raw = Path(a.raw)
    rides = rides_from_listing(json.loads((raw / 'zip_listing.json').read_text()))
    survey = [r for r in json.loads((raw / 'survey.json').read_text()) if r.get('moving_s', 0) > 120 and r.get('gps_n', 0) > 60]
    rng = random.Random(0); rng.shuffle(survey)
    names = [r['ride'] for r in survey[:a.n]]

    def one(n):
        try: return n, _circ_windows(ride_dir(raw, n, rides[n]['part']))
        except Exception:  # noqa: BLE001
            return n, []
    with ThreadPoolExecutor(a.workers) as ex: res = list(ex.map(one, names))
    ratios = np.array([d / rv for _, w in res for d, rv in w])
    per_ride = np.array([np.median([d / rv for d, rv in w]) for _, w in res if len(w) >= 3])
    circ = float(np.median(ratios))
    # yaw scale: per-ride gyro fits on rides whose IMU was downloaded
    imu_rides = [n for n in names + json.loads((raw / 'selection.json').read_text())['rides']
                 if any(ride_dir(raw, n, rides[n]['part']).glob('imu_data*.csv'))]
    imu_rides = list(dict.fromkeys(imu_rides))[:a.n]
    DEFAULT_YAW_SCALE = float('nan')

    def fit(n):
        try:
            info = estimate_poses(ride_dir(raw, n, rides[n]['part']), circ)['info']
            return info['compass_yaw_scale'], info['compass_dev_deg'], info['gyro_bias_dps'], info['wheel_disagreement']
        except Exception:  # noqa: BLE001
            return None
    with ThreadPoolExecutor(a.workers) as ex: fits = [f for f in ex.map(fit, imu_rides) if f and f[0] is not None]
    ks = np.array([k for k, dev, _, wd in fits if dev < 8 and wd < WHEEL_FAULT and 0.006 <= k <= 0.026])
    q = lambda v, qs=(.1, .25, .5, .75, .9): [round(float(np.quantile(v, x)), 5) for x in qs] if len(v) else None
    cal = dict(circumference_m=circ, wheel_diameter_m=circ / math.pi, circumference_window_q10_25_50_75_90=q(ratios),
               circumference_per_ride_q=q(per_ride), n_windows=int(len(ratios)), n_rides_with_windows=int(sum(1 for _, w in res if w)),
               yaw_scale=float(np.median(ks)) if len(ks) else None, yaw_scale_q=q(ks), n_yaw_fits=int(len(ks)),
               compass_dev_deg_q=q(np.array([d for _, d, _, _ in fits])), gyro_bias_dps_q=q(np.array([f[2] for f in fits if f[2] is not None])),
               wheel_fault_frac=float(np.mean([f[3] >= WHEEL_FAULT for f in fits])) if fits else None,
               method='circumference: median over 30 s straight forward-driving windows of GPS chord / wheel revolutions; '
                      'yaw_scale: median of per-ride compass fits (rad per rpm*s of differential; rides with compass deviation < 8 deg and healthy wheels)')
    (raw / 'calibration.json').write_text(json.dumps(cal, indent=1))
    print(json.dumps(cal, indent=1))


# --------------------------------------------------------------------------------------------- video
def segment_frames(path, t_start):
    """Yield (unix_time, av.VideoFrame) of one HLS .ts segment; time = segment start (from its name) + pts offset."""
    import av
    with av.open(str(path)) as c:
        p0 = min((p.pts for p in c.demux(c.streams.video[0]) if p.pts is not None), default=None)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        for f in c.decode(s):
            if f.pts is None or p0 is None: continue
            yield t_start + (f.pts - p0) * tb, f


def segment_times(path, t_start):
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        pts = sorted(p.pts for p in c.demux(s) if p.pts is not None)
    if not pts: return np.zeros(0)
    return t_start + (np.array(pts, float) - pts[0]) * tb


def convert_ride(name, r, raw, out, circumference, max_gap=1.0):
    from . import realnav_common as rc
    rd = ride_dir(raw, name, r['part'])
    segs = [(t, rd / 'recordings' / m[0].split('/')[-1]) for t, m in r['front']]
    segs = [(t, p) for t, p in segs if p.exists()]
    if not segs: return dict(ride=name, status='no video')
    p = estimate_poses(rd, circumference)
    if p['info']['wheel_disagreement'] >= WHEEL_FAULT: return dict(ride=name, status='rejected: wheel fault', info=p['info'])
    w30 = np.array(gps_consistency(p, 30.0))
    if len(w30) >= 3:
        big = w30[:, 1] > 15
        ratio = float(np.median(w30[big, 0] / w30[big, 1])) if big.sum() >= 3 else None
        p['info']['odo_gps_len_ratio_30m'] = ratio
        if ratio is not None and not 0.6 <= ratio <= 1.6: return dict(ride=name, status='rejected: odometry/GPS scale', info=p['info'])
    g2 = p['info'].get('gyro_r2_final')
    if g2 is not None and g2 < GYRO_R2_MIN: return dict(ride=name, status='rejected: yaw rate disagrees with gyroscope', info=p['info'])
    # frame times per segment (demux only)
    ft, fseg = [], []
    for k, (t0, path) in enumerate(segs):
        try: tt = segment_times(path, t0)
        except Exception:  # noqa: BLE001
            continue
        ft.append(tt); fseg.append(np.full(len(tt), k))
    if not ft: return dict(ride=name, status='undecodable')
    ft = np.concatenate(ft); fseg = np.concatenate(fseg)
    o = np.argsort(ft, kind='stable'); ft, fseg = ft[o], fseg[o]
    try:  # agreement of the segment-name + pts frame times with the dataset's frame timestamp file
        import pandas as pd
        ct_ = np.sort(pd.read_csv(next(rd.glob('front_camera_timestamps*.csv'))).timestamp.values.astype(float))
        j_ = np.clip(np.searchsorted(ct_, ft), 1, len(ct_) - 1)
        p['info']['frame_time_vs_csv_ms'] = round(float(np.median(np.minimum(np.abs(ft - ct_[j_]), np.abs(ft - ct_[j_ - 1])))) * 1000, 1)
    except Exception:  # noqa: BLE001
        pass
    # per-ride residual camera/wheel lag from rotation-induced horizontal image shift (falls back to 0)
    lag, lag_r = video_lag(segs, fseg, ft, p)
    p['info'].update(video_lag_s=lag, video_lag_r=lag_r)
    tq = ft - lag  # pose time of each frame
    inside = (tq >= p['t'][0]) & (tq <= p['t'][-1])
    ct = p['t']
    j = np.clip(np.searchsorted(ct, tq), 1, len(ct) - 1)
    gap_ok = (ct[j] - ct[j - 1]) <= max_gap
    valid = inside & gap_ok
    pos = np.stack([np.interp(tq, ct, p['xy'][:, 0]), np.interp(tq, ct, p['xy'][:, 1])], -1)
    yu = np.unwrap(p['yaw']); yaw = rc.wrap(np.interp(tq, ct, yu))
    # pieces: contiguous valid frames without frame time gaps
    cut = np.r_[True, (np.diff(ft) > max_gap) | ~valid[1:] | ~valid[:-1]]
    starts = np.flatnonzero(cut)
    pieces = [(a, b) for a, b in zip(starts, np.r_[starts[1:], len(ft)]) if valid[a]]
    keep_all = []
    for a, b in pieces:
        if b - a < 20: continue
        for c0, c1 in rc.split_at_jumps(pos[a:b], ft[a:b], max_jump=1.5, max_gap=max_gap):
            kk = rc.subsample(pos[a + c0:a + c1], yaw[a + c0:a + c1], min_step=MIN_STEP_M) + a + c0
            arc = np.linalg.norm(np.diff(pos[kk], axis=0), axis=1).sum() if len(kk) > 1 else 0
            if len(kk) >= 10 and arc >= a_min_arc: keep_all.append(kk)
    if not keep_all: return dict(ride=name, status='no motion episodes')
    want = {}
    for e, kk in enumerate(keep_all):
        for i in kk: want[int(i)] = e
    # decode needed frames segment by segment
    frames = {}
    need_seg = sorted({int(fseg[i]) for i in want})
    for k in need_seg:
        t0, path = segs[k]
        idx = np.flatnonzero(fseg == k)
        tset = {round(float(ft[i]), 6): int(i) for i in idx if int(i) in want}
        try:
            for tf, f in segment_frames(path, t0):
                i = tset.get(round(tf, 6))
                if i is not None:
                    im = f.to_image()
                    if corrupt_frame(im): frames[i] = None; continue
                    if max(im.size) > 1024: im = im.resize((1024, round(im.height * 1024 / im.width)), Image.BICUBIC)
                    b = io.BytesIO(); im.save(b, 'JPEG', quality=95); frames[i] = (b.getvalue(), im.size)
        except Exception:  # noqa: BLE001
            pass
    lat0, lon0 = p['latlon0']
    bad = {i for i, v in frames.items() if v is None}
    eps = []
    pieces = []
    for kk in keep_all:
        kk = np.array([int(i) for i in kk if int(i) in frames and int(i) not in bad])
        if len(kk) < 10: continue
        # decoding failures or dropped corrupt frames can open gaps: split again
        for c0, c1 in rc.split_at_jumps(pos[kk], ft[kk], max_jump=1.5, max_gap=max_gap):
            if c1 - c0 >= 10: pieces.append(kk[c0:c1])
    for e, kk in enumerate(pieces):
        P = pos[kk]
        if np.linalg.norm(np.diff(P, axis=0), axis=1).sum() < a_min_arc: continue
        key = f'{name}_{e:03d}'
        src = dict(original=f'output_rides_{r["part"]}/{name}', group=name, ride=name, session_hash=r.get('prefix'),
                   robot=None, geo_cell=geo_cell(lat0, lon0), lat=round(lat0, 3), lon=round(lon0, 3), part=r['part'],
                   metric=True, label_source='odometry', label_detail=LABEL_DETAIL,
                   pose_info=p['info'], circumference_m=circumference, control_time_offset_s=CONTROL_TIME_OFFSET_S,
                   imu_time_offset_s=IMU_TIME_OFFSET_S,
                   camera=CAMERA, license=LICENSE, fps=20, t0=float(ft[kk[0]]),
                   image_size=list(frames[int(kk[0])][1]))
        d = Path(out) / key; d.mkdir(parents=True, exist_ok=True)
        for q, i in enumerate(kk): (d / f'{q}.jpg').write_bytes(frames[int(i)][0])
        np.savez(d / 'trajectory.npz', position=P.astype(np.float64), yaw=yaw[kk].astype(np.float64), t=(ft[kk] - ft[kk[0]]).astype(np.float64))  # t: camera clock
        (d / 'source.json').write_text(json.dumps(src, indent=1))
        eps.append(key)
    p['info']['corrupt_frames'] = len(bad)
    return dict(ride=name, status='ok' if eps else 'no motion episodes', episodes=eps, info=p['info'])


def best_lag(ta, a, tb, b, rng=1.0, step=0.05):
    """(corr, L) maximising corr(a(t + L), b(t)), i.e. a happens L seconds after b."""
    tt = np.arange(max(ta[0], tb[0]) + rng, min(ta[-1], tb[-1]) - rng, step)
    if len(tt) < 200: return (float('nan'), 0.0)
    bb = np.interp(tt, tb, b); best = (-1.0, 0.0)
    for L in np.arange(-rng, rng + 1e-9, step):
        aa = np.interp(tt + L, ta, a)
        if aa.std() < 1e-9 or bb.std() < 1e-9: continue
        c = float(np.corrcoef(aa, bb)[0, 1])
        if c > best[0]: best = (c, float(L))
    return best


def video_lag(segs, fseg, ft, p, min_r=0.5):
    """Residual lag (s) of the camera behind the pose clock: horizontal image shift between consecutive frames
    (phase correlation on the upper image half) against the yaw rate. Returns (lag or 0 if unreliable, corr)."""
    import cv2
    shift = np.full(len(ft), np.nan)
    pos_of = {}
    for k in np.unique(fseg):
        idx = np.flatnonzero(fseg == k)
        for i in idx: pos_of[round(float(ft[i]), 6)] = int(i)
    for k, (t0, path) in enumerate(segs):
        prev = None
        try:
            for tf, f in segment_frames(path, t0):
                g = f.to_ndarray(format='gray')[: f.height // 2:4, ::4].astype(np.float32)
                i = pos_of.get(round(tf, 6))
                if prev is not None and i is not None: shift[i] = cv2.phaseCorrelate(prev, g)[0][0]
                prev = g
        except Exception:  # noqa: BLE001
            continue
    ok = np.isfinite(shift)
    if ok.sum() < 400: return 0.0, None
    w = np.gradient(np.unwrap(p['yaw']), p['t'])
    c, L = best_lag(ft[ok], shift[ok], p['t'], w)
    if not np.isfinite(c) or c < min_r: return 0.0, (None if not np.isfinite(c) else round(c, 3))
    return round(L, 3), round(c, 3)


def corrupt_frame(im):
    """Decoder garbage (e.g. a flat grey frame with a coloured bar before the first keyframe)."""
    g = np.asarray(im.convert('L').resize((64, 36)))
    return np.bincount(g.ravel(), minlength=256).max() > 0.5 * g.size or g.std() < 2


LABEL_DETAIL = ('wheel odometry: distance = mean of the 4 wheel RPMs x wheel circumference calibrated against GPS; '
                'yaw = per-ride scale x integrated differential wheel speed (scale fitted to the 1 Hz magnetometer heading) '
                '+ slowly varying offset (30 s circular mean of magnetometer minus wheel yaw); absolute heading constant from '
                'the GPS course. Frame: x east, y north (m), origin near the ride GPS median. Gyroscope unused (independent check).')
a_min_arc = 5.0  # drop episodes shorter than this (m)
MIN_STEP_M = 0.25  # frame spacing target: kept frames 0.25-0.3 m apart (or > 5 deg of turn)


def _convert_worker(args):
    name, r, raw, out, circ = args
    try:
        return convert_ride(name, r, raw, out, circ)
    except Exception as e:  # noqa: BLE001
        import traceback
        return dict(ride=name, status=f'error: {e}', tb=traceback.format_exc()[-800:])


def cmd_convert(a):
    global DEFAULT_YAW_SCALE
    raw, out = Path(a.raw), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cal = json.loads((raw / 'calibration.json').read_text())
    DEFAULT_YAW_SCALE = cal['yaw_scale']
    rides = rides_from_listing(json.loads((raw / 'zip_listing.json').read_text()))
    survey = {r['ride']: r for r in json.loads((raw / 'survey.json').read_text())}
    names = json.loads((raw / 'selection.json').read_text())['rides']
    if a.limit: names = names[:a.limit]
    log_path = out / 'convert_log.jsonl'
    done = set()
    if log_path.exists():
        for d in log_records(log_path):
            if d['status'] in ('ok', 'no motion episodes') or d['status'].startswith('rejected'): done.add(d['ride'])
    todo = []
    for n in names:
        if n in done: continue
        r = dict(rides[n]); r['cell'] = survey.get(n, {}).get('cell')
        rd = ride_dir(raw, n, r['part'])
        if not (rd / 'recordings').is_dir(): continue
        if a.require_complete:
            segs = front_segments(r, rd, name=n)
            if not segs or not all((rd / 'recordings' / m[0].split('/')[-1]).exists() for _, m in segs): continue
            if not any(rd.glob('imu_data*.csv')): continue
        todo.append((n, r, str(raw), str(out), cal['circumference_m']))
    print(f'converting {len(todo)} rides ({len(done)} already done)', flush=True)
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(cal['yaw_scale'],)) as pool, open(log_path, 'a') as log:
        for k, res in enumerate(pool.imap_unordered(_convert_worker, todo)):
            log.write(json.dumps(res) + '\n'); log.flush()
            if k % 20 == 0: print(k, res['ride'], res['status'], len(res.get('episodes', [])), flush=True)
    print('CONVERT_DONE', flush=True)


def _init_worker(k):
    global DEFAULT_YAW_SCALE
    DEFAULT_YAW_SCALE = k
    os.environ.setdefault('OMP_NUM_THREADS', '1')


# ------------------------------------------------------------------------------------------- ready
def log_records(path):
    """Parsed lines of convert_log.jsonl; a line cut by an interrupted write (e.g. a full disk) is skipped."""
    for line in Path(path).read_text().splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def status_counts(out):
    c = defaultdict(int)
    p = Path(out) / 'convert_log.jsonl'
    if p.exists():
        for d in log_records(p): c[d['status']] += 1
    return dict(c)


def cmd_ready(a):
    from . import realnav_common as rc
    raw, out = Path(a.raw), Path(a.out)
    cal = json.loads((raw / 'calibration.json').read_text())
    sel = json.loads((raw / 'selection.json').read_text())
    eps = sorted(p.parent for p in out.glob('*/trajectory.npz'))
    cells = defaultdict(set)
    for ep in eps:
        s = json.loads((ep / 'source.json').read_text()); cells[s.get('geo_cell')].add(s['group'])
    qc = json.loads((raw / 'qc.json').read_text()) if (raw / 'qc.json').exists() else None
    extra = dict(dataset='frodobots2k', source='FrodoBots-2K raw rides (S3 zips listed by https://huggingface.co/datasets/BitRobot/FrodoBots-2K)',
                 license=LICENSE, label_source='odometry', calibration=cal, geo_cells=len(cells),
                 rides_per_geo_cell={c: len(v) for c, v in sorted(cells.items(), key=lambda x: -len(x[1]))},
                 convert_status=status_counts(out),
                 rides_selected=len(sel['rides']), selection=dict((k, v) for k, v in sel.items() if k != 'rides'),
                 gps_consistency=qc, camera=CAMERA,
                 label_detail=LABEL_DETAIL,
                 caveats=['labels are wheel odometry with magnetometer-calibrated differential-drive yaw (not MBRA relabels; MBRA '
                          'actions exist only as model outputs in BitRobot/berkeley-frodobots-lerobot-7k at 224x128)',
                          'skid-steer slip: yaw scale fitted per ride to the 1 Hz magnetometer; residual turn-magnitude errors of '
                          '~10-20% are possible; the gyroscope (1 Hz) is only an independent check',
                          'heading_vs_motion_deg is ~0 by construction (positions are integrated along the yaw); see gps_consistency',
                          'clock offsets: control +0.3 s and IMU +0.5 s to the camera clock (global), plus a per-ride camera lag '
                          'fitted from rotation-induced image shift when reliable (source.json pose_info.video_lag_s)',
                          'front camera intrinsics are not published; wide-angle with visible barrel distortion',
                          'teleoperated by remote gamers with latency: paths include hesitations, stops and corrections',
                          'group = ride (session); no robot id in the data; geo_cell (0.5 deg GPS cell) allows geography-held-out splits'])
    ready = rc.write_ready(out, extra)
    paths = rc.previews(out, n=4, seed=a.seed)
    print(json.dumps({k: v for k, v in ready.items() if k not in ('calibration',)}, indent=1))
    print('previews', *paths)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('index'); s.add_argument('--raw', required=True)
    s = sub.add_parser('fetch'); s.add_argument('--raw', required=True); s.add_argument('--kind', choices=['meta', 'imu', 'video'], required=True)
    s.add_argument('--workers', type=int, default=32); s.add_argument('--limit', type=int, default=0)
    s.add_argument('--all_segments', action='store_true', help='video: also segments without motion')
    s = sub.add_parser('survey'); s.add_argument('--raw', required=True); s.add_argument('--workers', type=int, default=8)
    s = sub.add_parser('select'); s.add_argument('--raw', required=True); s.add_argument('--budget_gb', type=float, default=250)
    s.add_argument('--min_moving_s', type=float, default=120); s.add_argument('--seed', type=int, default=0)
    s = sub.add_parser('calibrate'); s.add_argument('--raw', required=True); s.add_argument('--n', type=int, default=400); s.add_argument('--workers', type=int, default=8)
    s = sub.add_parser('convert'); s.add_argument('--raw', required=True); s.add_argument('--out', required=True)
    s.add_argument('--workers', type=int, default=12); s.add_argument('--limit', type=int, default=0)
    s.add_argument('--require_complete', action='store_true', help='skip rides whose video/IMU download is incomplete')
    s = sub.add_parser('qc'); s.add_argument('--raw', required=True); s.add_argument('--out', required=True)
    s.add_argument('--n', type=int, default=300); s.add_argument('--workers', type=int, default=8)
    s = sub.add_parser('ready'); s.add_argument('--raw', required=True); s.add_argument('--out', required=True); s.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    {'index': cmd_index, 'fetch': cmd_fetch, 'survey': cmd_survey, 'select': cmd_select, 'calibrate': cmd_calibrate,
     'convert': cmd_convert, 'qc': cmd_qc, 'ready': cmd_ready}[a.cmd](a)


if __name__ == '__main__':
    main()
