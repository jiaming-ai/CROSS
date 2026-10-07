#!/usr/bin/env python3
"""Development split of the benchmark (benchmark/DEV.md): run it, and compare two runs.

  # run (map jobs first, then the query jobs; one job per GPU slot); results go to <out>/<dataset>/<scene>/<system>[@variant]
  python benchmark/dev.py run --systems cross_stereo --tier quick --gpus 0 [--variant views7 --args "--max-refs 4"]
  # compare two runs cell by cell (a run is <system>[@<variant>]); T3 as paired trial flips
  python benchmark/dev.py compare cross_stereo cross_stereo@views7 [--tier quick]

Data and results: --data / $BENCH_DATA (the benchmark folders; clips by benchmark/datasets/make_dev.py) and
--out / $BENCH_DEV_RESULTS.  Jobs whose result exists are skipped (delete the run folder, or pass --force, to redo).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark" / "eval"))


def dev_cfg(datasets=None):
    ds = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())
    return {k: v for k, v in ds.items() if v.get("dev") and (not datasets or k in datasets)}


def scenes(cfg, tier):
    """(dataset, scene, map, queries) of a tier: quick = the scenes with a `quick` list, and those queries; full = every
    scene of the dev entries; val = the entries marked `tier: val` (a larger confirmation set)."""
    for dataset, d in cfg.items():
        # an entry with a `tier` of its own (val, occ) belongs to that tier only; the others form quick / full
        if d.get("tier") is not None and d.get("tier") != tier or d.get("tier") is None and tier not in ("quick", "full"):
            continue
        for scene, sc in d["scenes"].items():
            if tier == "quick":
                if "quick" not in sc:
                    continue
                yield dataset, scene, sc["map"], list(sc["quick"])
            else:
                yield dataset, scene, sc["map"], list(sc.get("queries", []))


def job_lists(a):
    cfg = dev_cfg(a.datasets)
    sy = yaml.safe_load((ROOT / "benchmark/configs/systems.yaml").read_text())
    maps, queries = [], []
    for system in a.systems:
        for setup in sy[system]["setups"]:
            for dataset, scene, m, qs in scenes(cfg, a.tier):
                if setup not in cfg[dataset]["setups"] or f"{cfg[dataset].get('data', dataset)}/{setup}" in sy[system].get("skip", []):
                    continue
                base = ["--dataset", dataset, "--scene", scene, "--system", system, "--setup", setup, "--seed", str(a.seed)]
                maps.append(base + ["--task", "map"])
                queries += [base + ["--task", "query", "--query", q] for q in qs]
    return maps, queries


def run_jobs(jobs, a, log_dir: Path):
    """Run benchmark/run.py jobs on the GPU slots (each slot one job at a time)."""
    slots = [g for g in a.gpus for _ in range(a.jobs_per_gpu)]
    todo = list(jobs)
    lock = threading.Lock()
    failed = []

    def worker(gpu):
        while True:
            with lock:
                if not todo:
                    return
                job = todo.pop(0)
            cmd = [sys.executable, str(ROOT / "benchmark/run.py"), *job, "--data", a.data, "--out", a.out, "--wait-for-map"]
            if a.variant:
                cmd += ["--variant", a.variant]
            if a.args:
                cmd += ["--args", a.args]
            if a.force:
                cmd += ["--force"]
            if a.keep_rows:
                cmd += ["--keep-rows"]
            name = "_".join(job[1::2][:4] + job[-1:] + ([a.variant] if a.variant else [])).replace("/", "-")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            t0 = time.time()
            with open(log_dir / f"{name}.log", "a") as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT)).returncode
            print(f"[gpu {gpu}] rc={rc} {time.time() - t0:6.0f}s  {' '.join(job)}", flush=True)
            if rc != 0:
                with lock:
                    failed.append(job)

    threads = [threading.Thread(target=worker, args=(g,)) for g in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failed


def cmd_run(a):
    maps, queries = job_lists(a)
    log_dir = Path(a.out) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    failed = run_jobs(maps, a, log_dir) + run_jobs(queries, a, log_dir)
    print(f"{len(maps)} map + {len(queries)} query jobs in {time.time() - t0:.0f} s; failed: {len(failed)}")
    for j in failed:
        print("  failed:", " ".join(j))
    # a failed run still writes a result.json (status failed), and query jobs of a failed map write none: list both
    cfg = dev_cfg(a.datasets)
    for system in a.systems:
        run = system + (f"@{a.variant}" if a.variant else "")
        for f in sorted(Path(a.out).glob(f"*/*/{run}/*/s*/t[123]/**/result.json")):
            r = json.loads(f.read_text())
            if r.get("status") != "ok":
                print(f"  failed result: {f.relative_to(a.out)}: {r.get('error') or 'rc ' + str(r.get('rc'))}")
        cells = load_run(Path(a.out), run, cfg, a.tier)
        expected = sum(1 + 2 * len(qs) for _, _, _, qs in scenes(cfg, a.tier))
        if len(cells) < expected:
            print(f"  {run}: {len(cells)} of {expected} result cells are ok (see the failed results / logs in {log_dir})")


# ---------------------------------------------------------------------------------------------------- comparison
def load_run(out: Path, run: str, cfg, tier):
    """{cell: result} of one run (<system>[@variant]); cell = (dataset, scene, track, sequence or query)."""
    cells = {}
    wanted = {(d, s) for d, s, _, _ in scenes(cfg, tier)}
    queries = {(d, s): set(qs) for d, s, _, qs in scenes(cfg, tier)}
    for f in out.glob(f"*/*/{run}/*/s*/t[123]/**/result.json"):
        r = json.loads(f.read_text())
        d, s = r["dataset"], r["scene"]
        if (d, s) not in wanted or r.get("status") != "ok":
            continue
        if r["track"] != "t1" and r.get("query") not in queries[(d, s)]:
            continue
        # time of the relative pose estimator: reloc_summary.json of a query run, map_meta.json of the map
        side = f.parent / "reloc_summary.json"
        if r["track"] == "t1":       # the map of a scene, or the native/ folder of another T1 sequence
            side = f.parents[2] / "maps" / f.parent.name / "map_meta.json"
            if not side.is_file():
                side = f.parent / "native" / "map_meta.json"
        if side.is_file():
            meta = json.loads(side.read_text())
            est = meta.get("timing", {}).get("estimate_pose", {})
            r["_est"] = (est.get("n", 0), est.get("total_s", 0.0))
            step = meta.get("timing", {}).get("step", {})       # System.step over all frames (loading excluded)
            if step.get("n"):
                r["_step"] = (int(step["n"]), float(step["total_s"]))
        cells[(d, s, r["track"], r.get("sequence") or r.get("query"))] = r
    return cells


_COV = {}


def covered_trial_starts(data: Path, cfg, dataset, scene, query, setup):
    """Trial starts whose last frame lies within the larger threshold of the map session's path and in which the robot
    moves in at least the rule's fraction of the frames (PROTOCOL.md T3)."""
    key = (dataset, scene, query, setup)
    if key not in _COV:
        from scipy.spatial import cKDTree
        d = cfg[dataset]
        root = data / d.get("data", dataset)
        sub = d["setups"][setup]

        from metrics import moving_fraction, moving_frames, moving_rule

        def poses(seq):
            return np.loadtxt(root / seq / sub / "poses_left.txt").reshape(-1, 4, 4)
        Gq = poses(query)
        gq, gm = Gq[:, :3, 3], poses(d["scenes"][scene]["map"])[:, :3, 3]
        ok = cKDTree(gm).query(gq)[0] < d["thresholds"][1]
        L = d["trial_len"]
        rule = moving_rule(d)
        mov = moving_frames(Gq, 10.0, **rule) if "t3" in rule["tracks"] else np.ones(len(Gq), bool)
        _COV[key] = {s for s in range(len(gq)) if s + L - 1 < len(gq) and ok[s + L - 1]
                     and moving_fraction(s, L, moving=mov) >= (rule["min_fraction"] if "t3" in rule["tracks"] else 0.0)}
    return _COV[key]


def t3_trials(r, cfg, data):
    cov = covered_trial_starts(data, cfg, r["dataset"], r["scene"], r["query"], r["setup"])
    t1, t2 = cfg[r["dataset"]]["thresholds"]
    out = {}
    for t in r["trials"]:
        if int(t.get("start") or 0) in cov:
            e = t.get("final_err")
            e = math.inf if e is None else e
            out[int(t.get("start") or 0)] = (e < t1, e < t2)
    return out


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def wall(cells):
    return sum((r.get("wall_s") or 0) for r in cells.values())


def cmd_compare(a):
    cfg = dev_cfg(a.datasets)
    out, data = Path(a.out), Path(a.data)
    runs = [load_run(out, r, cfg, a.tier) for r in a.runs]
    base = runs[0]
    keys = sorted(set().union(*[set(r) for r in runs]), key=lambda k: (k[2], k[0], k[1], str(k[3])))
    w = max(len(r) for r in a.runs) + 2
    head = "".join(f"{r:>{w}}" for r in a.runs)
    print(f"{'cell':<44}{'metric':<12}{head}")
    pooled = {r: {"t2": [0.0, 0.0, 0], "t3": [0, 0, 0], "flips": [0, 0, 0, 0]} for r in a.runs}
    for k in keys:
        d, s, track, seq = k
        thr = cfg[d]["thresholds"]
        label = f"{track} {d.replace('dev_', '')}/{seq}"
        if track == "t1":
            vals = [run.get(k, {}).get("ate_rmse") for run in runs]
            fps = [run.get(k, {}).get("fps") for run in runs]
            print(f"{label:<44}{'ATE m':<12}" + "".join(f"{fmt(v):>{w}}" for v in vals))
            print(f"{'':<44}{'FPS':<12}" + "".join(f"{fmt(v, 2):>{w}}" for v in fps))
        elif track == "t2":
            for x in thr:
                vals = [run.get(k, {}).get(f"lr@{x:g}") for run in runs]
                print(f"{label if x == thr[0] else '':<44}{f'LR@{x:g}':<12}" + "".join(f"{fmt(v):>{w}}" for v in vals))
            for name, run in zip(a.runs, runs):
                if k in run:
                    n = run[k].get("n_frames") or 0
                    p = pooled[name]["t2"]
                    p[0] += (run[k].get(f"lr@{thr[0]:g}") or 0) * n
                    p[1] += (run[k].get(f"lr@{thr[1]:g}") or 0) * n
                    p[2] += n
        else:
            tr = [t3_trials(run[k], cfg, data) if k in run else None for run in runs]
            for j, x in enumerate(thr):
                cells = []
                for i, t in enumerate(tr):
                    if t is None:
                        cells.append("-")
                        continue
                    n_ok = sum(v[j] for v in t.values())
                    cell = f"{n_ok}/{len(t)}"
                    if i > 0 and tr[0] is not None:          # paired flips against the first run on common trials
                        common = set(t) & set(tr[0])
                        won = sum(1 for st in common if t[st][j] and not tr[0][st][j])
                        lost = sum(1 for st in common if tr[0][st][j] and not t[st][j])
                        cell += f" +{won}-{lost}"
                        pooled[a.runs[i]]["flips"][2 * j] += won
                        pooled[a.runs[i]]["flips"][2 * j + 1] += lost
                    if j == 0:
                        pooled[a.runs[i]]["t3"][2] += len(t)
                    pooled[a.runs[i]]["t3"][j] += n_ok
                    cells.append(cell)
                print(f"{label if j == 0 else '':<44}{f'RS@{x:g}':<12}" + "".join(f"{c:>{w}}" for c in cells))
    print("-" * (56 + w * len(a.runs)))
    for i, x in enumerate(("lower", "upper")):
        vals = [pooled[r]["t2"] for r in a.runs]
        print(f"{'pooled T2 (frames)':<44}{f'LR {x} thr':<12}" + "".join(f"{fmt(v[i] / v[2] if v[2] else None):>{w}}" for v in vals))
    for i, x in enumerate(("lower", "upper")):
        cells = []
        for j, r in enumerate(a.runs):
            p = pooled[r]
            c = f"{p['t3'][i]}/{p['t3'][2]}"
            if j > 0:
                c += f" +{p['flips'][2 * i]}-{p['flips'][2 * i + 1]}"
            cells.append(c)
        print(f"{'pooled T3 (trials)':<44}{f'RS {x} thr':<12}" + "".join(f"{c:>{w}}" for c in cells))
    print(f"{'GPU time of the jobs':<44}{'min':<12}" + "".join(f"{wall(run) / 60:>{w}.1f}" for run in runs))
    est = [[sum(r.get("_est", (0, 0))[i] for r in run.values()) for i in (0, 1)] for run in runs]
    print(f"{'relative pose estimation':<44}{'calls':<12}" + "".join(f"{n:>{w}d}" for n, _ in est))
    print(f"{'':<44}{'ms / call':<12}" + "".join(f"{(t / n * 1e3 if n else 0):>{w}.0f}" for n, t in est))
    for track in ("t1", "t2", "t3"):        # processing rate: frames / time in System.step, pooled and worst cell
        cells = []
        for run in runs:
            st = [r["_step"] for k, r in run.items() if k[2] == track and "_step" in r]
            n, t = sum(x[0] for x in st), sum(x[1] for x in st)
            worst = min((x[0] / x[1] for x in st if x[1] > 0), default=0.0)
            cells.append(f"{(n / t if t else 0):.1f} ({worst:.1f})")
        print(f"{'FPS ' + track + ' pooled (worst cell)':<44}{'frames/s':<12}" + "".join(f"{c:>{w}}" for c in cells))
    gpus = [sorted({str(r.get('gpu')) for r in run.values()}) for run in runs]
    if len({tuple(g) for g in gpus}) > 1:
        print("warning: the runs used different GPUs:", dict(zip(a.runs, gpus)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--tier", choices=["quick", "full", "val", "occ"], default="quick")
        p.add_argument("--datasets", nargs="*", default=None, help="only these dev entries of datasets.yaml")
        p.add_argument("--data", default=os.environ.get("BENCH_DATA"))
        p.add_argument("--out", default=os.environ.get("BENCH_DEV_RESULTS"))
    r = sub.choices["run"]
    r.add_argument("--systems", nargs="+", required=True)
    r.add_argument("--gpus", nargs="+", default=["0"])
    r.add_argument("--jobs-per-gpu", type=int, default=1, help="more than 1 makes the recorded FPS meaningless")
    r.add_argument("--variant", default="")
    r.add_argument("--args", default="", help="extra CROSS arguments of this variant")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--force", action="store_true")
    r.add_argument("--keep-rows", action="store_true", help="keep the per-frame rows of the query runs (reloc_rows.json)")
    c = sub.choices["compare"]
    c.add_argument("runs", nargs="+", help="<system>[@<variant>]; the first one is the reference")
    a = ap.parse_args()
    if not a.data or not a.out:
        ap.error("--data / $BENCH_DATA and --out / $BENCH_DEV_RESULTS are required")
    cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    main()
