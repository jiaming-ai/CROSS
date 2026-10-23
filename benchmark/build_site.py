#!/usr/bin/env python3
"""Write the data of the results page benchmark/site/index.html from benchmark/results/results.json.

  python benchmark/build_site.py [--results benchmark/results/results.json]

The page is static: open benchmark/site/index.html from the file system or serve the folder (GitHub Pages).
Tables reuse the aggregation of make_tables.py; every table cell lists the runs behind it, whose trajectories,
error curves and trial outcomes the page plots.  Failure cases: the worst runs per system are selected here (see `failures`); the cause analysis and the case videos come
from benchmark/failure_analysis.py and benchmark/make_failure_assets.py (benchmark/results/failure_*.json).
The History tab shows the same tables per code version (the `commit` of the runs, from benchmark/results/history.json
written by collect.py), one row per commit that has results, so a change can be traced back to the commit that made it.

Files (all loaded with <script> tags, so the page also works from file://):
  site/data.js           what the page needs to open: tables, summary, systems, datasets (a few hundred KB)
  site/runs/<chunk>.js   the plotted runs of one track · dataset · scene (trajectories, error curves, trials), loaded
                         when a cell, plot or tab needs them; table cells reference runs as chunk * RUN_BASE + index
  site/failures.js       Failure cases tab;  site/history.js  History tab
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import make_tables as mt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "benchmark/site"


def write_default_changes():
    """Publish only changes adopted as defaults, separately from historical benchmark tables."""
    src = ROOT / "benchmark/results/default_changes.json"
    changes = json.loads(src.read_text()) if src.is_file() else []
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "default_changes.js").write_text(
        "window.CROSS_DEFAULT_CHANGES = " + json.dumps(changes, ensure_ascii=False) + ";\n")


def run_id(r):
    """Stable id of a result cell (also names its failure-case images in site/assets/failures/)."""
    parts = [r.get("track"), r.get("dataset"), r.get("scene"), r.get("system"), r.get("setup"), f"s{r.get('seed', 0)}",
             r.get("sequence") or f"{r.get('map')}__{r.get('query')}"]
    return "-".join(str(p) for p in parts).replace("/", "_").replace("+", "p")


def plane(traj_gt, traj_est):
    """The two world axes spanning the ground-truth path (a top-down view whatever the world convention)."""
    g = np.asarray(traj_gt, float)
    if len(g) < 3:
        return [0, 1]
    var = g.var(0)
    return sorted(np.argsort(var)[-2:].tolist())


# the fields of a run the page reads (plus lr@<threshold>); everything else stays in results.json
RUN_FIELDS = ("track", "dataset", "scene", "system", "setup", "sequence", "map", "query", "status", "rc", "ate_rmse",
              "completeness", "n_poses", "align", "scale", "scale_sim3", "failed", "ms_ate", "est_frac", "rs1", "rs2", "n_trials",
              "concat", "thresholds", "odom_fallback")
RUN_BASE = 100000          # a run reference = chunk index * RUN_BASE + index in the chunk


def path_cm(p):
    """A 2-D path as flat centimetre integers, the first point absolute and the others as steps (index.html
    decodePath): about a third of the text of [[x, y], ...] in metres."""
    a = np.round(np.asarray(p, float) * 100).astype(np.int64)
    return np.concatenate([a[:1], np.diff(a, axis=0)]).reshape(-1).tolist()


def slim_run(r):
    """The page's copy of a run: RUN_FIELDS, the top-down path of the ground truth (g) and of the aligned estimate (e)
    as path_cm, the error curve (c, centimetres, -1 = no estimate) and the T3 trials."""
    thr = r.get("thresholds", [1.0, 2.0])
    keep = {k: r[k] for k in RUN_FIELDS if r.get(k) is not None}
    keep.update({k: v for k, v in r.items() if k.startswith("lr@") and float(k[3:]) in thr})
    if r.get("traj_gt") is not None:
        ax = plane(r["traj_gt"], r.get("traj_est"))
        g = np.asarray(r["traj_gt"], float)[:, ax]
        keep["g"] = path_cm(g[:: max(1, len(g) // 300)])
        if r.get("traj_est") is not None:
            e = np.asarray(r["traj_est"], float)[:, ax]
            keep["e"] = path_cm(e[:: max(1, len(e) // 300)])
    if r.get("err_curve") is not None:
        c = np.asarray(r["err_curve"], float)[:: max(1, len(r["err_curve"]) // 400)]
        keep["c"] = [-1 if not np.isfinite(x) or x < 0 else int(round(x * 100)) for x in c]
    if r.get("trials") is not None:
        # [start, success at the larger threshold, final error, success at the smaller threshold, counted (1), or not:
        #  0 = the map does not cover it, 2 = the robot stands still (make_tables.rescore_t3)]
        keep["trials"] = [[t.get("start"), int(t.get("final_err") is not None and t["final_err"] < thr[1]),
                           None if t.get("final_err") is None else round(t["final_err"], 2),
                           int(t.get("final_err") is not None and t["final_err"] < thr[0]),
                           1 if t.get("counted", t.get("covered", True)) else (0 if not t.get("covered", True) else 2)]
                          for t in r["trials"]]
    return keep


def table_models(T: mt.Tables, runs_by_key):
    """{track: {dataset: {"cols": [...], "rows": [{"label", "cells": [{"text", "runs"}]}]}}} mirroring RESULTS.md."""
    out = {"t1": {}, "t2": {}, "t3": {}}
    ds, sy = T.ds, T.sy
    for dataset in ("kitti", "openloris", "rover"):
        scenes = list(ds[dataset]["scenes"])
        rows = []
        for system, setup in mt.rows_for(ds, sy, dataset):
            cells_all = T.t1_cells(dataset, system, setup)
            cells, allv, allf, allm, alln, allids = [], [], 0, 0, 0, []
            for sc in scenes:
                seqs = T.scene_seqs(dataset, sc)
                v, f, m = T.t1_agg(cells_all, seqs)
                allv += v; allf += f; allm += m; alln += len(seqs)
                text = T.t1_seq_str(cells_all.get(seqs[0])) if len(seqs) == 1 else T.t1_agg_str(v, f, m, len(seqs))
                ids = [runs_by_key[id(cells_all[q])] for q in seqs if q in cells_all]
                allids += ids
                cells.append({"text": text, "runs": ids})
            cells.append({"text": T.t1_agg_str(allv, allf, allm, alln), "runs": allids})
            seqs = [q for sc in scenes for q in T.scene_seqs(dataset, sc)]
            cells.append({"text": T.scale_str(T.t1_scale(cells_all, seqs)) if setup == "mono" and cells_all else mt.NA,
                          "runs": []})
            rows.append({"label": mt.row_label(sy, system, setup, dataset), "system": system, "setup": setup,
                         "pending": sy[system]["runner"] == "pending", "cells": cells})
        out["t1"][dataset] = {"cols": scenes + ["mean", "scale err. (mono)"], "rows": rows}
    for track in ("t2", "t3"):
        for dataset in ("openloris", "rover", "simchange"):
            cfg = ds[dataset]
            scenes = [s for s in cfg["scenes"] if cfg["scenes"][s].get("queries")] or \
                sorted({r["scene"] for k, v in T.idx.items() if k[0] == track and k[1] == dataset for r in v})
            if track == "t2" and dataset == "simchange":
                continue
            rows = []
            for system, setup in mt.rows_for(ds, sy, dataset):
                rs = T.idx[(track, dataset, system, setup)]
                cells = []
                for s in scenes + [None]:                    # None: all scenes of the dataset
                    sub = [r for r in rs if s is None or r["scene"] == s]
                    n = T.n_queries(dataset, s) or len(sub)
                    text = T.t2_str(sub, n, cfg["thresholds"]) if track == "t2" else T.t3_str(sub, n)
                    cells.append({"text": text, "runs": [runs_by_key[id(r)] for r in sub] if s is not None else []})
                rows.append({"label": mt.row_label(sy, system, setup, dataset), "system": system, "setup": setup,
                             "pending": sy[system]["runner"] == "pending" and not rs, "cells": cells})
            out[track][dataset] = {"cols": scenes + ["all scenes"], "rows": rows, "thresholds": cfg["thresholds"]}
    # T3 overall: every method's pooled success per dataset and the mean over the datasets
    rows = []
    for system, sc in sy.items():
        if sc.get("hidden"):
            continue
        for setup in sc["setups"]:
            cells = []
            for d in mt.T3_DATASETS:
                if setup not in ds[d]["setups"] or f"{d}/{setup}" in sc.get("skip", []):
                    cells.append({"text": mt.NA, "runs": []})
                    continue
                cells.append({"text": T.t3_str(T.idx[("t3", d, system, setup)], T.n_queries(d)), "runs": []})
            cells.append({"text": T.t3_overall_str(system, setup), "runs": []})
            rows.append({"label": mt.row_label(sy, system, setup), "system": system, "setup": setup,
                         "pending": sc["runner"] == "pending", "cells": cells})
    out["t3"] = {"all": {"cols": [mt.DS_NAMES.get(d, d) + (" (1 / 2 m)" if ds[d]["environment"] == "indoor" else " (3 / 5 m)")
                                  for d in mt.T3_DATASETS] + ["overall (mean of the datasets)"],
                         "rows": rows, "thresholds": None}, **out["t3"]}
    return out


def prepare(results, ds_cfg, sy, seed):
    """Runs of one seed, T3 re-scored at the dataset thresholds and failed queries counted (as the tables expect)."""
    results = mt.count_failed_queries(mt.rescore_t3([r for r in results if r.get("seed", 0) == seed], ds_cfg), ds_cfg, sy)
    for r in results:                      # the dataset's two thresholds travel with every run (page labels)
        r["thresholds"] = ds_cfg[r["dataset"]]["thresholds"]
    return results


def commit_id(r):
    """Code version of a run: the short hash when `commit` is one, else the label as recorded ('unrecorded' if none)."""
    c = (r.get("commit") or "").strip()
    if not c:
        return "unrecorded"
    return c[:7] if re.fullmatch(r"[0-9a-f]{7,40}", c) else c


def commit_info(ids, runs_by_commit):
    """{id: {date, subject, tags, runs, first, last}}: commit date and subject from git (hash ids), run times."""
    out = {}
    for c in ids:
        times = sorted(r.get("time") or "" for r in runs_by_commit[c].values())
        info = {"runs": len(runs_by_commit[c]), "first": times[0][:16] if times else "", "last": times[-1][:16] if times else ""}
        if re.fullmatch(r"[0-9a-f]{7}", c):
            g = subprocess.run(["git", "show", "-s", "--format=%ad%x00%s%x00%D", "--date=format:%Y-%m-%d %H:%M", c],
                               cwd=ROOT, capture_output=True, text=True)
            if g.returncode == 0 and g.stdout:
                date, subject, refs = (g.stdout.strip().split("\x00") + ["", ""])[:3]
                info.update(date=date, subject=subject,
                            tags=[x.strip()[5:] for x in refs.split(",") if x.strip().startswith("tag: ")])
        info.setdefault("date", info["first"])
        info.setdefault("subject", "" if re.fullmatch(r"[0-9a-f]{7}", c) else c)
        out[c] = info
    return out


class _NoIds(dict):
    def __missing__(self, k):
        return ""


def history_models(hist, current, ds_cfg, sy, seed):
    """The tables of table_models once per code version: {"commits": {...}, "tables": {track: {dataset: {"cols",
    "rows": [{system, setup, label, commit, current, cells: [text]}]}}}}; a row per commit that ran the system on the
    dataset; `current`: the version (one of the versions) behind the method's cells in the T1-T3 tabs."""
    shown = defaultdict(set)
    for r in current:
        shown[(r["track"], r["dataset"], r["system"], r["setup"])].add(commit_id(r))
    by_commit = defaultdict(dict)
    for r in hist:
        if r.get("seed", 0) != seed:
            continue
        c, k = commit_id(r), run_id(r)
        if k not in by_commit[c] or (r.get("time") or "") >= (by_commit[c][k].get("time") or ""):
            by_commit[c][k] = r
    tables = {}
    for c, cells in by_commit.items():
        T = mt.Tables(prepare([dict(r) for r in cells.values()], ds_cfg, sy, seed), seed)
        for track, per_ds in table_models(T, _NoIds()).items():
            for dataset, m in per_ds.items():
                dst = tables.setdefault(track, {}).setdefault(dataset, {"cols": m["cols"], "rows": []})
                for row in m["rows"]:
                    texts = [x["text"] for x in row["cells"]]
                    if row["pending"] or all(t in (mt.PENDING, mt.NA, "") for t in texts):
                        continue                   # not run with this code version
                    cur = any(c in shown[(t, d, row["system"], row["setup"])]
                              for t in ("t1", "t2", "t3") if t == track
                              for d in ([dataset] if dataset != "all" else ("openloris", "rover", "simchange")))
                    dst["rows"].append({"system": row["system"], "setup": row["setup"], "label": row["label"],
                                        "commit": c, "current": cur, "cells": texts})
    return {"commits": commit_info(list(by_commit), by_commit), "tables": tables}


def failures(results, runs_by_key, per_system=4):
    """Failure cases per system · setup: crashed / incomplete maps (T1), the worst-localized query sessions (T2) and
    failed relocalization trials (T3), worst first."""
    out = {}
    for r in results:
        k = f"{r['system']}|{r['setup']}"
        out.setdefault(k, {"t1": [], "t2": [], "t3": []})
        rid = runs_by_key[id(r)]
        if r["track"] == "t1" and (r.get("status") != "ok" or r.get("failed")):
            out[k]["t1"].append((1.0 - (r.get("completeness") or 0), rid))
        elif r["track"] == "t1" and r.get("ate_rmse") is not None:
            out[k]["t1"].append((-1.0 + min(r["ate_rmse"] / 10.0, 0.99), rid))   # worst successful maps after the failures
        elif r["track"] == "t2" and r.get("status") == "ok":
            lr = r.get(f"lr@{r.get('thresholds', [1, 2])[1]:g}") or 0.0
            out[k]["t2"].append((1.0 - lr, rid))
        elif r["track"] == "t3" and r.get("status") == "ok":
            nf = r.get("n_trials", 0) - r.get("n_s2", r.get("n_success", 0))
            if nf > 0:
                out[k]["t3"].append((nf / max(r["n_trials"], 1), rid))
    for k, v in out.items():
        for t in v:
            v[t] = [rid for _, rid in sorted(v[t], reverse=True)[:per_system]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "benchmark/results/results.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--history", default=str(ROOT / "benchmark/results/history.json"))
    ap.add_argument("--defaults-only", action="store_true", help="update adopted-default notes without rescoring tables")
    a = ap.parse_args()
    write_default_changes()
    if a.defaults_only:
        return
    results = json.loads(Path(a.results).read_text())["results"] if Path(a.results).is_file() else []
    ds_cfg, sy_cfg = mt.load()
    results, sy_cfg = mt.odom_rows(results, sy_cfg, ds_cfg)    # runs with VIO odometry etc. as rows of their own
    results = prepare(results, ds_cfg, sy_cfg, a.seed)
    # one file of runs per track · dataset · scene, the runs in run_id order (a file changes only when its runs do)
    groups = defaultdict(list)
    for r in results:
        groups[re.sub(r"[^A-Za-z0-9_.-]", "_", f"{r['track']}_{r['dataset']}_{r['scene']}")].append((run_id(r), r))
    chunks, runs_by_key = sorted(groups), {}
    for c, name in enumerate(chunks):
        groups[name].sort(key=lambda x: x[0])
        for i, (_, r) in enumerate(groups[name]):
            runs_by_key[id(r)] = c * RUN_BASE + i
    T = mt.Tables(results, a.seed)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:       # noqa: BLE001
        commit = None
    fa, fc = ROOT / "benchmark/results/failure_analysis.json", ROOT / "benchmark/results/failure_cases.json"
    fspec = ROOT / "benchmark/configs/failure_cases.yaml"
    failure_page = {"analysis": json.loads(fa.read_text()) if fa.is_file() else None,
                    "cases": json.loads(fc.read_text())["cases"] if fc.is_file() else [],
                    "text": {k: v for k, v in yaml.safe_load(fspec.read_text()).items() if k in ("intro", "findings", "groups", "notes")}
                    if fspec.is_file() else {}}
    legacy = ROOT / "benchmark/results/legacy.json"
    dstats = ROOT / "benchmark/results/datasets.json"
    samples = ROOT / "benchmark/results/dataset_samples.json"       # make_dataset_assets.py
    js = lambda x: json.dumps(x, separators=(",", ":"), ensure_ascii=False)       # noqa: E731
    parts = {"failures": failure_page,
             "history": history_models(mt.odom_rows(json.loads(Path(a.history).read_text())["runs"], sy_cfg, ds_cfg)[0],
                                       results, ds_cfg, sy_cfg, a.seed) if Path(a.history).is_file() else None}
    run_files = {name: "window.BENCH_RUNS(" + js(name) + "," + js([slim_run(r) for _, r in groups[name]]) + ");\n"
                 for name in chunks}
    parts_js = {k: f"window.BENCH_PART({js(k)},{js(v)});\n" for k, v in parts.items()}
    # the version the page appends to the files it loads later (?v=), so a browser never mixes two builds
    version = hashlib.sha1("".join(run_files[n] for n in chunks).encode() + "".join(parts_js.values()).encode()).hexdigest()[:10]
    order = [[k, su] for k, v in T.sy.items() if not v.get("hidden") for su in v["setups"]]
    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M"), "commit": commit, "seed": a.seed, "version": version,
        "systems": T.sy, "datasets": {k: {kk: vv for kk, vv in v.items() if kk != "scenes"} | {"scenes": list(v["scenes"])}
                                      for k, v in T.ds.items()},
        # summary tables: [key, title, note, markdown]; their rows are the methods in `order` (make_tables.summary_table)
        "summary": T.summary(), "tables": table_models(T, runs_by_key), "chunks": chunks, "run_base": RUN_BASE,
        "n_runs": len(results), "failures": failures(results, runs_by_key),
        "legacy": json.loads(legacy.read_text()) if legacy.is_file() else None,
        "datastats": json.loads(dstats.read_text()) if dstats.is_file() else {},
        "samples": json.loads(samples.read_text()) if samples.is_file() else {},
        "order": order,
        "splits": {d: [[sc, {"map": v["map"], "queries": v.get("queries", []), "thresholds": c["thresholds"]}]
                       for sc, v in c["scenes"].items()]
                   for d, c in T.ds.items()},
    }
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "data.js").write_text("window.BENCH = " + js(data) + ";\n")
    shutil.rmtree(SITE / "runs", ignore_errors=True)
    (SITE / "runs").mkdir()
    for name, text in run_files.items():
        (SITE / "runs" / f"{name}.js").write_text(text)
    for k, text in parts_js.items():
        (SITE / f"{k}.js").write_text(text)
    size = lambda p: p.stat().st_size / 1e6                                         # noqa: E731
    print(f"wrote {SITE / 'data.js'} ({size(SITE / 'data.js'):.2f} MB, {len(results)} runs), "
          f"{len(chunks)} run files in {SITE / 'runs'} ({sum(size(p) for p in (SITE / 'runs').glob('*.js')):.1f} MB), "
          + ", ".join(f"{k}.js ({size(SITE / f'{k}.js'):.2f} MB)" for k in parts))


if __name__ == "__main__":
    main()
