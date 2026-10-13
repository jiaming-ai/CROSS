"""Convert a dataset in the GNM/visualnav-transformer processed layout to realnav episodes (used for GoStanford2).

    python -m cross_nav.policy.datasets.gnm_processed --dataset go_stanford --raw DIR_OR_ZIP --out REALNAV [--workers 12]

The processed layout (README "Data-Wrangling" of robodhruv/visualnav-transformer) is one folder per trajectory with
0.jpg..T.jpg and traj_data.pkl = {"position": (T,2) xy, "yaw": (T,)}. --raw may be the folder holding the trajectory
folders or a .zip of it (read in place). traj_data.pkl is a Python pickle published by the dataset authors (the only
format these labels exist in); it is read with an unpickler that admits nothing but numpy array reconstruction, and
the result must be a dict of numeric arrays. Frames are subsampled and split as in gnm_openx.
"""
import argparse, io, json, pickle, re, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from .gnm_openx import write_pieces
from .realnav_common import write_ready, previews

INFO = {
    'go_stanford': dict(
        group=lambda ep: ep.split('_')[0],  # recording, e.g. no3vc (trajectory folders are <recording>_<seg>_<k>)
        # The release mixes real runs (noNvc_*), their horizontally mirrored copies (noNvcF_*: images flipped, yaw
        # negated) and simulator renderings (simNh_*, simNhF_*). Only the unmirrored real runs are kept.
        include=lambda ep: re.match(r'^no\d+vc_', ep) is not None,
        camera={'model': 'Ricoh Theta S 360 camera on a TurtleBot2, front fisheye view as processed by GNM',
                'fisheye': True, 'robot': 'TurtleBot2 (GO Stanford 2, Hirose et al. 2019)', 'intrinsics': None},
        label='odometry', fps=None,
        license='GO Stanford dataset terms (Stanford, Hirose et al. 2019); GNM "GoStanford2 (Modified)" release',
        source='https://huggingface.co/datasets/franciszzj/GoStanford2 go_stanford.zip (mirror of the GNM Google Drive '
               'folder 1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_)',
        caveats=['160x120 images as released by GNM', 'TurtleBot2 wheel odometry',
                 'excluded: mirrored augmentation copies (noNvcF_*) and simulator trajectories (simNh*_*)']),
}


class NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split('.')[0] == 'numpy' and name in ('_reconstruct', 'ndarray', 'dtype', 'scalar', '_frombuffer'):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'refusing {module}.{name}')


def load_traj(b):
    d = NumpyOnly(io.BytesIO(b)).load()
    if not isinstance(d, dict): raise ValueError('traj_data.pkl is not a dict')
    # GoStanford2 stores object arrays (position: np.float64 scalars, yaw: 1-element arrays)
    pos = np.array([[float(np.asarray(v).reshape(-1)[0]) for v in row] for row in d['position']], np.float64).reshape(-1, 2)
    yaw = np.array([float(np.asarray(v).reshape(-1)[0]) for v in d['yaw']], np.float64)
    return pos, yaw


class Source:
    def __init__(self, raw):
        self.raw = Path(raw)
        self.zip = zipfile.ZipFile(self.raw) if self.raw.suffix == '.zip' else None

    def trajectories(self):
        if self.zip:
            return sorted(str(Path(n).parent) for n in self.zip.namelist() if n.endswith('/traj_data.pkl'))
        return sorted(str(p.parent.relative_to(self.raw)) for p in self.raw.rglob('traj_data.pkl'))

    def read(self, rel):
        return self.zip.read(rel) if self.zip else (self.raw / rel).read_bytes()


def convert(raw, trajs, out, dataset):
    src = Source(raw); info = INFO[dataset]; res = []
    for rel in trajs:
        ep = Path(rel).name
        pos, yaw = load_traj(src.read(f'{rel}/traj_data.pkl'))
        if len(pos) != len(yaw) or not (np.isfinite(pos).all() and np.isfinite(yaw).all()):
            res.append((ep, 'bad arrays')); continue
        meta = dict(original=f'{Path(raw).name}:{rel}', group=info['group'](ep), metric=True, label_source=info['label'],
                    camera=info['camera'], license=info['license'], source=info['source'], dataset=dataset, fps=info['fps'],
                    note='GNM-processed trajectory (positions/yaw from robot odometry at the GNM sampling rate)')
        try:
            w = write_pieces(out, ep, pos, yaw, lambda i: Image.open(io.BytesIO(src.read(f'{rel}/{i}.jpg'))), meta)
        except KeyError as e:
            res.append((ep, f'missing image {e}')); continue
        res.append((ep, w))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(INFO)); ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out) / a.dataset; out.mkdir(parents=True, exist_ok=True)
    trajs = Source(a.raw).trajectories()
    include = INFO[a.dataset].get('include')
    n_all = len(trajs)
    if include: trajs = [t for t in trajs if include(Path(t).name)]
    chunks = [trajs[i::a.workers * 4] for i in range(a.workers * 4)]
    n_in, bad, written = len(trajs), [], 0
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(convert, a.raw, c, out, a.dataset) for c in chunks]):
            for ep, r in f.result():
                if isinstance(r, str): bad.append((ep, r))
                else: written += len(r)
    info = INFO[a.dataset]
    extra = dict(dataset=a.dataset, label_source=info['label'], license=info['license'], source=info['source'],
                 source_trajectories=n_all, source_trajectories_used=n_in, skipped=bad[:50], n_skipped=len(bad), caveats=info['caveats'], camera=info['camera'],
                 converter=f'python -m cross_nav.policy.datasets.gnm_processed --dataset {a.dataset}',
                 pickle_note='traj_data.pkl read with a numpy-only restricted unpickler (author-published labels)')
    ready = write_ready(out, extra)
    previews(out)
    print(json.dumps(ready, indent=1), flush=True)


if __name__ == '__main__':
    main()
