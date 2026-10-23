"""Storage of the benchmark results: metrics in git, per-frame data on the Hugging Face Hub.

benchmark/results/results.json (in git) holds every cell's metrics, the T3 trial outcomes included.  The per-frame data
the page draws (FRAMES: trajectories and error curves) lives in benchmark/results/frames/<track>/<dataset>/<scene>.json.gz
(git-ignored), one file per scene, which `python benchmark/results_io.py push` uploads to the public dataset repo
FRAMES_REPO.  benchmark/results/frames.json (in git) pins the Hub revision and the sha256 of every file, so a checkout
gets exactly the per-frame data of its results: load_results(frames=True) downloads it when it is missing or stale.

  python benchmark/results_io.py split                  # move per-frame data out of a results.json that still has it
  python benchmark/results_io.py push [--message ...]   # upload benchmark/results/frames/, pin the revision
  python benchmark/results_io.py pull                   # download the pinned revision
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark/results/results.json"
FRAMES_DIR = ROOT / "benchmark/results/frames"
FRAMES_LOCK = ROOT / "benchmark/results/frames.json"
FRAMES_REPO = "Jamie-nus/CROSS-benchmark-runs"
FRAMES = ("traj_est", "traj_gt", "err_curve")      # per-frame data (downsampled by run.py), only the page reads it


def cell_key(r) -> str:
    """A cell (as collect.key): track, dataset, scene, system, setup, seed, sequence / map + query, odometry source."""
    return "|".join(str(x) for x in (r.get("track"), r.get("dataset"), r.get("scene"), r.get("system"), r.get("setup"),
                                     r.get("seed"), r.get("sequence"), r.get("map"), r.get("query"),
                                     r.get("odom") or "external"))


def shard(r) -> str:
    return f"{r.get('track')}/{r.get('dataset')}/{str(r.get('scene')).replace('/', '_')}.json.gz"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def add_scale_sim3(r):
    """Record the Sim(3) scale of a monocular T1 cell scored before metrics.ate wrote `scale_sim3`, so the tables'
    scale error does not need the per-frame data: `scale` of a Sim(3)-aligned run, or a fit to a CROSS run's stored
    trajectory (its traj_est / traj_gt are the same keyframes, downsampled alike, so they pair)."""
    if r.get("track") != "t1" or r.get("setup") != "mono" or "scale_sim3" in r or r.get("ate_rmse") is None:
        return
    if r.get("align") == "sim3":
        r["scale_sim3"] = r.get("scale")
    elif str(r.get("system", "")).startswith("cross"):
        e, g = r.get("traj_est") or [], r.get("traj_gt") or []
        if len(e) == len(g) >= 3:
            import sys

            import numpy as np
            sys.path.insert(0, str(ROOT / "benchmark/eval"))
            from metrics import umeyama
            r["scale_sim3"] = umeyama(np.asarray(e, float), np.asarray(g, float), with_scale=True)[0]


def save_results(rows, path: Path = RESULTS, frames_dir: Path = FRAMES_DIR, lock: Path = FRAMES_LOCK):
    """Write results.json without the per-frame data and the per-frame data of every cell to its scene's file (cells
    whose run has none keep the data already stored for them); update the lock's sha256 (its revision stays until the
    next push)."""
    stored = {}                                   # shard -> {cell: frames}
    for r in rows:
        add_scale_sim3(r)
        s = shard(r)
        if s not in stored:
            f = frames_dir / s
            stored[s] = json.loads(gzip.decompress(f.read_bytes())) if f.is_file() else {}
        fr = {k: r[k] for k in FRAMES if r.get(k) is not None}
        if fr:
            stored[s][cell_key(r)] = fr
    keep = {s: {cell_key(r) for r in rows if shard(r) == s} for s in stored}
    for s, d in stored.items():
        d = {k: v for k, v in sorted(d.items()) if k in keep[s]}
        f = frames_dir / s
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(gzip.compress(json.dumps(d, separators=(",", ":")).encode(), 9, mtime=0))
    slim = [{k: v for k, v in r.items() if k not in FRAMES} for r in rows]
    Path(path).write_text(json.dumps({"results": slim}, separators=(",", ":")))
    if Path(frames_dir).resolve() == FRAMES_DIR.resolve():
        old = json.loads(lock.read_text()) if lock.is_file() else {}
        files = {str(p.relative_to(frames_dir)): _sha(p) for p in sorted(frames_dir.rglob("*.json.gz"))}
        lock.write_text(json.dumps({"repo": old.get("repo", FRAMES_REPO), "revision": old.get("revision"),
                                    "files": files}, indent=1) + "\n")


def frames_ok(lock: Path = FRAMES_LOCK, frames_dir: Path = FRAMES_DIR) -> bool:
    if not lock.is_file():
        return True                               # nothing pinned: whatever is local
    files = json.loads(lock.read_text())["files"]
    return all((frames_dir / f).is_file() and _sha(frames_dir / f) == h for f, h in files.items())


def pull(lock: Path = FRAMES_LOCK, frames_dir: Path = FRAMES_DIR):
    """Download the pinned revision of the per-frame data into frames_dir and check it."""
    from huggingface_hub import snapshot_download
    pin = json.loads(lock.read_text())
    if not pin.get("revision"):
        raise SystemExit(f"{lock}: no revision pinned (run `python benchmark/results_io.py push` first)")
    snapshot_download(pin["repo"], repo_type="dataset", revision=pin["revision"], local_dir=frames_dir.parent,
                      allow_patterns=["frames/**"])
    if not frames_ok(lock, frames_dir):
        raise SystemExit(f"per-frame data of {pin['repo']}@{pin['revision']} does not match {lock}")
    print(f"per-frame data: {pin['repo']}@{pin['revision'][:10]} -> {frames_dir}")


def push(message: str, lock: Path = FRAMES_LOCK, frames_dir: Path = FRAMES_DIR):
    """Upload frames_dir (only what changed) to the Hub repo and pin the new revision in the lock."""
    from huggingface_hub import HfApi
    if not frames_ok(lock, frames_dir):
        raise SystemExit("benchmark/results/frames does not match frames.json: re-run collect.py or results_io.py split")
    pin = json.loads(lock.read_text())
    api = HfApi()
    api.create_repo(pin["repo"], repo_type="dataset", exist_ok=True)
    info = api.upload_folder(repo_id=pin["repo"], repo_type="dataset", folder_path=str(frames_dir),
                             path_in_repo="frames", commit_message=message, delete_patterns=["*.json.gz"])
    pin["revision"] = info.oid
    lock.write_text(json.dumps(pin, indent=1) + "\n")
    print(f"pushed {pin['repo']}@{info.oid[:10]}; commit {lock.relative_to(ROOT)}")


def load_results(path: Path = RESULTS, frames: bool = False, lock: Path = FRAMES_LOCK, frames_dir: Path = FRAMES_DIR):
    """The cells of results.json; with frames=True also their per-frame data (downloaded first when the local copy is
    missing or not the pinned one)."""
    rows = json.loads(Path(path).read_text())["results"]
    if not frames:
        return rows
    if Path(path).resolve() == RESULTS.resolve() and not frames_ok(lock, frames_dir):
        pull(lock, frames_dir)
    cache = {}
    for r in rows:
        s = shard(r)
        if s not in cache:
            f = frames_dir / s
            cache[s] = json.loads(gzip.decompress(f.read_bytes())) if f.is_file() else {}
        r.update(cache[s].get(cell_key(r), {}))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["split", "push", "pull"])
    ap.add_argument("--message", default=None, help="push: Hub commit message (default: the git commit)")
    a = ap.parse_args()
    if a.cmd == "split":
        save_results(json.loads(RESULTS.read_text())["results"])
        print(f"{RESULTS.relative_to(ROOT)}: {RESULTS.stat().st_size / 1e6:.1f} MB; per-frame data in "
              f"{FRAMES_DIR.relative_to(ROOT)} ({sum(p.stat().st_size for p in FRAMES_DIR.rglob('*.gz')) / 1e6:.1f} MB)")
    elif a.cmd == "pull":
        pull()
    else:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
        push(a.message or f"per-frame data of the benchmark results (CROSS {rev.strip()})")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()
