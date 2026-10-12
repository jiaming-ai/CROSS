"""Shared helpers for writing realnav episodes: frame subsampling, episode splitting, READY.json, previews.

Episode format (read by cross_nav.policy.data.gnm_records):
<out>/<episode>/{0..N-1}.jpg, trajectory.npz (position (N,2) m, yaw (N,) rad, optional t (N,) s), source.json
"""
import json, math, random
from pathlib import Path

import numpy as np
from PIL import Image

MAX_STEP, MIN_STEP, MIN_TURN = 0.3, 0.05, math.radians(5)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def split_at_jumps(position, t=None, max_jump=1.5, max_gap=None):
    """Return [start, end) index ranges of pieces without pose jumps (> max_jump m) or time gaps (> max_gap s)."""
    n = len(position)
    if n == 0: return []
    d = np.linalg.norm(np.diff(position, axis=0), axis=1)
    cut = ~np.isfinite(d) | (d > max_jump)
    if t is not None and max_gap is not None: cut |= np.diff(t) > max_gap
    edges = [0] + (np.flatnonzero(cut) + 1).tolist() + [n]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def subsample(position, yaw, min_step=MIN_STEP, max_step=MAX_STEP, min_turn=MIN_TURN):
    """Greedy keep-list: drop frames closer than min_step to the last kept frame unless yaw changed > min_turn.

    Frames are never skipped when the next one would be > max_step from the last kept frame, so the
    kept spacing stays <= max_step wherever the source frames allow it.
    """
    keep = [0]
    for i in range(1, len(position)):
        last = keep[-1]
        moved = np.linalg.norm(position[i] - position[last])
        turned = abs(wrap(yaw[i] - yaw[last]))
        nxt = np.linalg.norm(position[i + 1] - position[last]) if i + 1 < len(position) else np.inf
        if moved >= min_step or turned > min_turn or (nxt > max_step and moved > 1e-3):
            keep.append(i)
    return np.asarray(keep)


def save_jpg(img, path, max_side=1024):
    """Save a PIL image as JPEG q95, downscaled only when the long side exceeds max_side."""
    img = img.convert('RGB')
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
    img.save(path, quality=95)
    return img.size


def write_episode(out_dir, images, position, yaw, source, t=None):
    """images: list of PIL images or callables returning one (loaded lazily)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    size = None
    for i, im in enumerate(images):
        im = im() if callable(im) else im
        size = save_jpg(im, out_dir / f'{i}.jpg')
    arrays = dict(position=np.asarray(position, np.float64).reshape(-1, 2), yaw=np.asarray(yaw, np.float64).reshape(-1))
    if t is not None: arrays['t'] = np.asarray(t, np.float64).reshape(-1)
    np.savez(out_dir / 'trajectory.npz', **arrays)
    (out_dir / 'source.json').write_text(json.dumps(source, indent=1))
    return size


def heading_motion_angle(position, yaw, min_disp=0.2):
    """Angles (deg) between the heading and the displacement to the next frame >= min_disp m away."""
    out = []
    j = 0
    for i in range(len(position)):
        j = max(j, i + 1)
        while j < len(position) and np.linalg.norm(position[j] - position[i]) < min_disp: j += 1
        if j >= len(position): break
        d = position[j] - position[i]
        out.append(abs(math.degrees(wrap(math.atan2(d[1], d[0]) - yaw[i]))))
    return out


def quantiles(v, ps=(.05, .25, .5, .75, .95)):
    v = np.asarray(v, np.float64)
    return {f'q{int(round(p * 100))}': round(float(np.quantile(v, p)), 4) for p in ps} if v.size else {}


def episode_stats(ep):
    with np.load(ep / 'trajectory.npz') as z:
        pos, yaw = z['position'], z['yaw']
        t = z['t'] if 't' in z else None
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    src = json.loads((ep / 'source.json').read_text())
    speed = None
    if t is not None and len(t) > 1:
        dt = np.diff(t); ok = dt > 0
        speed = (step[ok] / dt[ok]).tolist()
    return dict(n=len(pos), arc=float(step.sum()), step=step.tolist(), group=src.get('group'), speed=speed,
                angle=heading_motion_angle(pos, yaw), finite=bool(np.isfinite(pos).all() and np.isfinite(yaw).all()),
                duration=float(t[-1] - t[0]) if t is not None and len(t) else None)


def write_ready(out, extra, sample_steps=200000):
    """Scan all episodes under out and write READY.json (merging the dataset-specific fields in extra)."""
    out = Path(out)
    eps = sorted(p.parent for p in out.glob('*/trajectory.npz'))
    rng = random.Random(0)
    arcs, steps, speeds, angles, groups, n, dur, nonfinite = [], [], [], [], set(), 0, 0.0, 0
    sizes = {}
    for k, ep in enumerate(eps):
        s = episode_stats(ep)
        arcs.append(s['arc']); n += s['n']; groups.add(s['group']); nonfinite += not s['finite']
        steps += s['step']; angles += s['angle']
        if s['speed']: speeds += s['speed']
        if s['duration']: dur += s['duration']
        if k % max(1, len(eps) // 50) == 0:
            with Image.open(ep / '0.jpg') as im: sizes[str(im.size)] = sizes.get(str(im.size), 0) + 1
    if len(steps) > sample_steps: steps = rng.sample(steps, sample_steps)
    ready = dict(episodes=len(eps), frames=n, groups=len(groups), total_km=round(sum(arcs) / 1000, 2),
                 total_hours=round(dur / 3600, 2) if dur else None, episodes_nonfinite=nonfinite,
                 arc_length_m=quantiles(arcs), frame_spacing_m=quantiles(steps), speed_mps=quantiles(speeds),
                 heading_vs_motion_deg=quantiles(angles), image_sizes_sampled=sizes)
    ready.update(extra)
    (out / 'READY.json').write_text(json.dumps(ready, indent=1))
    return ready


def contact_sheet(ep, path, n_frames=6):
    """Save a PNG with a few frames of an episode and its top-down trajectory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ep = Path(ep)
    with np.load(ep / 'trajectory.npz') as z: pos, yaw = z['position'], z['yaw']
    src = json.loads((ep / 'source.json').read_text())
    idx = np.linspace(0, len(pos) - 1, n_frames).round().astype(int)
    fig = plt.figure(figsize=(3 * n_frames, 6.5))
    for k, i in enumerate(idx):
        ax = fig.add_subplot(2, n_frames, k + 1)
        with Image.open(ep / f'{i}.jpg') as im: ax.imshow(im)
        ax.set_title(f'frame {i}', fontsize=8); ax.axis('off')
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(pos[:, 0], pos[:, 1], '-', lw=1)
    step = max(1, len(pos) // 40)
    ax.quiver(pos[::step, 0], pos[::step, 1], np.cos(yaw[::step]), np.sin(yaw[::step]), angles='xy', scale=30, width=.003, color='r')
    ax.plot(pos[idx, 0], pos[idx, 1], 'ko', ms=4)
    for k, i in enumerate(idx): ax.annotate(str(i), pos[i], fontsize=7)
    ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    arc = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()
    ax.set_title(f"{ep.name}  N={len(pos)}  arc={arc:.1f} m  label={src.get('label_source')}  group={src.get('group')}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def previews(out, n=4, seed=0):
    out = Path(out)
    eps = sorted(p.parent for p in out.glob('*/trajectory.npz'))
    (out / 'preview').mkdir(exist_ok=True)
    paths = []
    for ep in random.Random(seed).sample(eps, min(n, len(eps))):
        p = out / 'preview' / f'{ep.name}.png'
        contact_sheet(ep, p); paths.append(p)
    return paths
