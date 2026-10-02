#!/usr/bin/env python3
"""Write benchmark/site/data.js (the data of the results page benchmark/site/index.html) from
benchmark/results/results.json.

  python benchmark/build_site.py [--results benchmark/results/results.json]

The page is static: open benchmark/site/index.html from the file system or serve the folder (GitHub Pages).
Tables reuse the aggregation of make_tables.py; every table cell lists the runs behind it, whose trajectories,
error curves and trial outcomes the page plots.  Failure cases are selected here (see `failures`).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import make_tables as mt

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "benchmark/site"


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


def slim_run(r, rid):
    thr = r.get("thresholds", [1.0, 2.0])
    keep = {k: v for k, v in r.items() if k not in ("traj_est", "traj_gt", "err_curve", "trials")
            and not (k.startswith("lr@") and float(k[3:]) not in thr)}
    keep["id"] = rid
    if r.get("traj_gt") is not None:
        ax = plane(r["traj_gt"], r.get("traj_est"))
        g = np.asarray(r["traj_gt"], float)[:, ax]
        keep["gt2"] = np.round(g[:: max(1, len(g) // 300)], 2).tolist()
        if r.get("traj_est") is not None:
            e = np.asarray(r["traj_est"], float)[:, ax]
            keep["est2"] = np.round(e[:: max(1, len(e) // 300)], 2).tolist()
    if r.get("err_curve") is not None:
        c = np.asarray(r["err_curve"], float)
        keep["err"] = np.round(c[:: max(1, len(c) // 400)], 2).tolist()
    if r.get("trials") is not None:
        # [start, success at the larger threshold, final error, success at the smaller threshold, covered by the map]
        keep["trials"] = [[t.get("start"), int(t.get("final_err") is not None and t["final_err"] < thr[1]),
                           None if t.get("final_err") is None else round(t["final_err"], 2),
                           int(t.get("final_err") is not None and t["final_err"] < thr[0]), int(t.get("covered", True))]
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
            rows.append({"label": mt.row_label(sy, system, setup, dataset), "system": system, "setup": setup,
                         "pending": sy[system]["runner"] == "pending", "cells": cells})
        out["t1"][dataset] = {"cols": scenes + ["mean"], "rows": rows}
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
    a = ap.parse_args()
    results = json.loads(Path(a.results).read_text())["results"] if Path(a.results).is_file() else []
    ds_cfg = mt.load()[0]
    results = mt.count_failed_queries(mt.rescore_t3([r for r in results if r.get("seed", 0) == a.seed], ds_cfg), ds_cfg, mt.load()[1])
    for r in results:                      # the dataset's two thresholds travel with every run (page labels)
        r["thresholds"] = ds_cfg[r["dataset"]]["thresholds"]
    runs, runs_by_key = {}, {}
    for r in results:
        rid = run_id(r)
        runs_by_key[id(r)] = rid
        runs[rid] = slim_run(r, rid)
    T = mt.Tables(results, a.seed)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:       # noqa: BLE001
        commit = None
    assets = sorted(str(p.relative_to(SITE)) for p in (SITE / "assets").rglob("*.jpg")) if (SITE / "assets").is_dir() else []
    legacy = ROOT / "benchmark/results/legacy.json"
    dstats = ROOT / "benchmark/results/datasets.json"
    samples = ROOT / "benchmark/results/dataset_samples.json"       # make_dataset_assets.py
    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M"), "commit": commit, "seed": a.seed,
        "systems": T.sy, "datasets": {k: {kk: vv for kk, vv in v.items() if kk != "scenes"} | {"scenes": list(v["scenes"])}
                                      for k, v in T.ds.items()},
        "summary": T.summary(), "tables": table_models(T, runs_by_key), "runs": runs,
        "failures": failures(results, runs_by_key), "assets": assets,
        "legacy": json.loads(legacy.read_text()) if legacy.is_file() else None,
        "datastats": json.loads(dstats.read_text()) if dstats.is_file() else {},
        "samples": json.loads(samples.read_text()) if samples.is_file() else {},
        "order": [[k, su] for k, v in T.sy.items() if not v.get("hidden") for su in v["setups"]],
        "splits": {d: [[sc, {"map": v["map"], "queries": v.get("queries", []), "thresholds": c["thresholds"]}]
                       for sc, v in c["scenes"].items()]
                   for d, c in T.ds.items()},
    }
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "data.js").write_text("window.BENCH = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {SITE / 'data.js'} ({(SITE / 'data.js').stat().st_size / 1e6:.2f} MB, {len(runs)} runs)")


if __name__ == "__main__":
    main()
