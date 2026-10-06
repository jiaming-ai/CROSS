"""CROSS benchmark sequences (OpenLORIS-Scene, KITTI, ROVER, SimChange) as held-out test scenes (never trained on).

Input: CROSS's benchmark folders (benchmark/datasets/prepare_*.py; the SimChange v2 renders), one per recording,
<bench_root>/<dataset>/<recording>/<setup>/ with poses_left.txt (camera-to-world of the left / colour camera, OpenCV
axes, metres, 16 values per row), times.txt, calib.json ("K" of the stored images) and depth/<k>.png (uint16 mm,
z-depth, 0 = invalid, aligned to the colour image).  One scene per place, its recordings as sessions in the place's
single ground-truth frame:

  openloris  office / corridor / home / cafe (rgbd/: D435i colour + aligned depth).  market is skipped unless
             --openloris-market: its ground truth is off in time (market1-1 leads the gyros by ~1 s) and jittery.
  kitti      one scene per sequence (stereo/left: cam2 colour); no depth (stored as invalid)
  rover      campus_large (rgbd/: D435i colour, undistorted, + registered depth); ground truth = prism position +
             heading of travel (no roll / pitch), every recording registered to the map recording (calib.json
             gt_alignment); recordings registered to different targets become separate scenes
  simchange  one scene per rendered scene, its change variants as sessions (one Blender world; left camera +
             rendered depth)

Each sequence keeps at most --max-frames frames, uniform in time.  scene.json carries "split": "test".

  python -m vggt_ft.prep.cross_bench <bench_root> <out_root> [--datasets openloris kitti rover simchange]
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from vggt_ft.prep.common import SceneWriter, write_index

SPECS = {   # setup folder, image / depth folders (depth None: none recorded), scene metadata
    "openloris": dict(setup="rgbd", img="rgb", depth="depth", kind="indoor", dynamic=True, synthetic=False,
                      depth_src="D435i aligned depth"),
    "kitti": dict(setup="stereo", img="left", depth=None, kind="driving", dynamic=True, synthetic=False,
                  depth_src="none (all invalid)"),
    "rover": dict(setup="rgbd", img="rgb", depth="depth", kind="outdoor", dynamic=False, synthetic=False,
                  depth_src="D435i depth registered to colour (outdoor, noisy at range)"),
    "simchange": dict(setup=".", img="left", depth="depth", kind="indoor", dynamic=False, synthetic=True,
                      depth_src="rendered"),
}


def scenes_of(dataset: str, base: Path, market: bool = False) -> dict[str, tuple[str, list[tuple[str, Path]]]]:
    """{scene: (world, [(sequence, folder)])} of one dataset's benchmark folder."""
    setup = SPECS[dataset]["setup"]
    if dataset == "simchange":
        return {sc.name: (sc.name, [(v.name, v) for v in sorted(sc.iterdir()) if (v / "poses_left.txt").is_file()])
                for sc in sorted(base.iterdir()) if (sc / "map" / "poses_left.txt").is_file()}
    groups = {}
    for p in sorted(base.iterdir()):
        f = p / setup
        if not (f / "calib.json").is_file():
            continue
        if dataset == "openloris":
            m = re.fullmatch(r"([a-z]+)\d+-\d+", p.name)                  # skips the dev clips (home1-1_f600, ...)
            if m and (m.group(1) != "market" or market):
                groups.setdefault((m.group(1), m.group(1)), []).append((p.name, f))
        elif dataset == "kitti":
            groups[(p.name, p.name)] = [(p.name, f)]
        else:                                                             # rover: <place>_<condition>_<date>
            place = "_".join(p.name.split("_")[:2])
            to = json.loads((f / "calib.json").read_text()).get("gt_alignment", {}).get("to", p.name)
            groups.setdefault((place, to), []).append((p.name[len(place) + 1:], f))
    out = {}
    for (place, world), seqs in groups.items():
        name = place if sum(k[0] == place for k in groups) == 1 else f"{place}_{world}"
        out[name] = (name, seqs)
    return out


def pick(t: np.ndarray, n: int) -> np.ndarray:
    """Indices of at most n frames, uniform in time (the first and the last frame included)."""
    if len(t) <= n:
        return np.arange(len(t))
    g = np.linspace(t[0], t[-1], n)
    i = np.clip(np.searchsorted(t, g), 1, len(t) - 1)
    return np.unique(np.where(np.abs(t[i - 1] - g) <= np.abs(t[i] - g), i - 1, i))


def frames_of(f: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """(camera-to-world (N,4,4), times (N,), calib) of a benchmark folder."""
    calib = json.loads((f / "calib.json").read_text())
    c2w = np.loadtxt(f / "poses_left.txt").reshape(-1, 4, 4)
    t = np.loadtxt(f / "times.txt").reshape(-1) if (f / "times.txt").is_file() \
        else np.arange(len(c2w)) / float(calib.get("fps", 10.0))
    assert len(t) == len(c2w), f"{f}: {len(t)} times, {len(c2w)} poses"
    return c2w, t, calib


def convert_scene(dataset: str, scene: str, world: str, seqs, out_root: str, max_frames: int, use_depth: bool) -> str:
    spec = SPECS[dataset]
    dsrc = spec["depth_src"] if use_depth else "none (all invalid)"
    sw = SceneWriter(out_root, dataset, scene, world=world, metric=True, synthetic=spec["synthetic"],
                     dynamic=spec["dynamic"], kind=spec["kind"], extra={"split": "test", "depth": dsrc,
                                                                         "source": "CROSS benchmark"})
    lines = []
    for name, f in seqs:
        c2w, t, calib = frames_of(f)
        K = np.asarray(calib["K"], np.float64)
        n = len(c2w)
        assert (f / spec["img"] / f"{n - 1:06d}.png").exists() and not (f / spec["img"] / f"{n:06d}.png").exists(), \
            f"{f}: images do not match the {n} poses"

        def load(i):
            nm = f"{i:06d}"
            bgr = cv2.imread(str(f / spec["img"] / f"{nm}.png"), cv2.IMREAD_COLOR)
            assert bgr is not None and bgr.shape[:2] == (calib["height"], calib["width"]), f"{f} {nm}"
            if not (use_depth and spec["depth"]):
                return i, nm, bgr, np.zeros(bgr.shape[:2], np.float32)
            dep = cv2.imread(str(f / spec["depth"] / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
            assert dep is not None and dep.shape == bgr.shape[:2], f"{f} depth {nm}"
            return i, nm, bgr, dep.astype(np.float32) / 1000.0

        q = sw.sequence(name, session=name)
        sel = pick(t, max_frames)
        with ThreadPoolExecutor(8) as io:                                 # reads from a network share: latency-bound
            for i, nm, bgr, dep in io.map(load, sel):
                q.add(nm, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dep, K, np.linalg.inv(c2w[i]), float(t[i]))
        lines.append(f"{name} {len(sel)}/{n}")
    return f"{dataset}/{scene}: {sw.close()} frames ({', '.join(lines)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_root", help="holds <dataset>/<recording>/<setup>/")
    ap.add_argument("out_root")
    ap.add_argument("--datasets", nargs="+", default=list(SPECS), choices=list(SPECS))
    ap.add_argument("--src", nargs="*", default=[], help="dataset=<folder> overrides <bench_root>/<dataset>")
    ap.add_argument("--max-frames", type=int, default=400, help="per sequence, uniform in time")
    ap.add_argument("--no-depth", nargs="*", default=[], help="datasets stored without depth")
    ap.add_argument("--openloris-market", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true", help="rewrite scenes that already have a scene.json")
    a = ap.parse_args()
    src = dict(s.split("=", 1) for s in a.src)
    jobs = []
    for ds in a.datasets:
        for scene, (world, seqs) in scenes_of(ds, Path(src.get(ds, Path(a.bench_root) / ds)), a.openloris_market).items():
            if a.overwrite or not (Path(a.out_root) / ds / scene / "scene.json").exists():
                jobs.append((ds, scene, world, seqs, a.out_root, a.max_frames, ds not in a.no_depth))
    jobs.sort(key=lambda j: -len(j[3]))                                   # the scenes with most sequences first
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(convert_scene, *zip(*jobs)) if jobs else []:
            print(msg, flush=True)
    for ds in a.datasets:
        print(ds, len(write_index(Path(a.out_root) / ds)), "scenes")


if __name__ == "__main__":
    main()
