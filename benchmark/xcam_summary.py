#!/usr/bin/env python3
"""Tables of the cross-camera test (dataset xcam_simchange, benchmark/configs/datasets.yaml) from the run folders.

  python benchmark/xcam_summary.py $BENCH_RESULTS [--dataset xcam_simchange] [--per-query] [--json out.json]

Per scene, map camera, query camera and system (<system>[+odom][@variant]/<setup>/s<seed>), pooled over the query
sessions:
  T2  recall at the dataset's two thresholds and the median position error of the evaluated frames (covered by the map,
      robot moving; frames without a pose in the map count as failures / infinite error), in the map's alignment
      (errors.npz `err`); for the extension scenes also split by area: first session's area vs extension-only area
      (errors.npz `area`)
  T3  successes / counted trials at both thresholds (a trial counts when its last frame is covered and the robot moves
      in at least the moving rule's fraction of its frames), split by the area of the trial's last frame
  ext the extension session (ext/<map key>/result.json): ATE of its permanent keyframes in the loaded map's alignment
      (and re-aligned), the loaded keyframes' shift, the fraction of its frames with a pose in the map and their
      recall, completeness; with the T1 ATE of the first-session map
Cameras: a folder <variant>__camX is camera X, any other camA; an extension scene's map camera is A+X.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark" / "eval"))
from metrics import moving_rule  # noqa: E402

AREAS = {"all": None, "first": 1, "extension": 2}


def camera(seq: str) -> str:
    name = Path(seq).name
    return name.split("__cam", 1)[1] if "__cam" in name else "A"


def map_camera(sc: dict) -> str:
    return camera(sc["map"]) + (f"+{camera(sc['extend'])}" if sc.get("extend") else "")


def runs(root: Path, dataset: str):
    """(scene, system label, run folder) of every run of the dataset."""
    for d in sorted((root / dataset).glob("*/*/*/s*")):
        scene, system, setup, seed = d.parts[-4:]
        yield scene, f"{system}/{setup}/{seed}", d


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def fmt(x, nd=2):
    if x is None:
        return "–"
    if isinstance(x, float):
        return "inf" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="results root (BENCH_RESULTS)")
    ap.add_argument("--dataset", default="xcam_simchange")
    ap.add_argument("--per-query", action="store_true", help="also one row per query session")
    ap.add_argument("--json", default=None, help="write the tables as JSON too")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())[a.dataset]
    thr = cfg["thresholds"]
    min_frac = moving_rule(cfg)["min_fraction"] if "t3" in moving_rule(cfg)["tracks"] else 0.0
    root = Path(a.root)

    t2 = defaultdict(lambda: {"err": [], "n_q": 0, "failed": 0})      # key -> pooled evaluated errors
    t3 = defaultdict(lambda: {"n": 0, "k1": 0, "k2": 0, "n_q": 0, "failed": 0})
    ext_rows, missing = [], []
    for scene, system, d in runs(root, a.dataset):
        sc = cfg["scenes"].get(scene)
        if sc is None:
            continue
        mcam = map_camera(sc)
        areas = AREAS if sc.get("extend") else {"all": None}    # the area split only where there is an extension
        group = scene.rsplit("_", 1)[0]
        for res_file in sorted((d / "t2").glob("*/result.json")):
            r = load(res_file)
            if r is None:
                continue
            q = r["query"]
            keys = [(group, scene, mcam, camera(q), system, q if a.per_query else "*")]
            if a.per_query:
                keys.append((group, scene, mcam, camera(q), system, "*"))
            if r.get("status") != "ok":
                for k in keys:
                    t2[k + ("all",)]["failed"] += 1
                continue
            z = res_file.parent / "errors.npz"
            if not z.is_file():
                missing.append(str(z))
                continue
            e = np.load(z)
            keep = e["covered"] & e["moving"]
            err = np.where(np.isfinite(e["err"]), e["err"], np.inf)
            area = e["area"] if "area" in e.files else np.ones(len(err), np.int8)
            for k in keys:
                for name, av in areas.items():
                    sel = keep & ((area == av) if av is not None else True)
                    t2[k + (name,)]["err"].append(err[sel])
                    t2[k + (name,)]["n_q"] += 1
        for res_file in sorted((d / "t3").glob("*/result.json")):
            r = load(res_file)
            if r is None:
                continue
            q = r["query"]
            keys = [(group, scene, mcam, camera(q), system, q if a.per_query else "*")]
            if a.per_query:
                keys.append((group, scene, mcam, camera(q), system, "*"))
            if r.get("status") != "ok":
                for k in keys:
                    t3[k + ("all",)]["failed"] += 1
                continue
            for k in keys:
                for t in r.get("trials", []):
                    if not t.get("covered", True) or (t.get("moving") or 0) < min_frac:
                        continue
                    e = t.get("final_err")
                    e = np.inf if e is None else float(e)
                    names = ["all"] + ([{1: "first", 2: "extension"}[t["area"]]] if len(areas) > 1 and t.get("area") in (1, 2) else [])
                    for name in names:
                        c = t3[k + (name,)]
                        c["n"] += 1
                        c["k1"] += int(e < thr[0])
                        c["k2"] += int(e < thr[1])
                t3[k + ("all",)]["n_q"] += 1
        for res_file in sorted((d / "ext").glob("**/result.json")):
            r = load(res_file)
            if r is None:
                continue
            t1 = load(d / "t1" / r.get("map", "") / "result.json") or {}
            ext_rows.append({"scene": scene, "map_cam": mcam, "system": system, "status": r.get("status"),
                             "first_map_ate": t1.get("ate_rmse"), "first_map_kfs": t1.get("n_keyframes"),
                             **{k: r.get(k) for k in ("ate_rmse", "ate_median", "ate_rmse_realigned", "rot_err_median_deg",
                                                      "loaded_ate_before", "loaded_ate_after", "loaded_shift_median",
                                                      "n_new_permanent", "n_new_temporary", "localized_frac",
                                                      "first_localized_frame", "lr@1", "lr@2", "completeness",
                                                      "map_ate_rmse")}})

    out = {"t2": [], "t3": [], "ext": ext_rows}
    print(f"# Cross-camera summary ({a.dataset}, {root})\n")
    print(f"## T2: recall at {thr[0]:g} / {thr[1]:g} m and median error (m) of the evaluated frames, pooled over queries\n")
    print("| scene | map cam | query cam | system | query | area | queries | frames | "
          f"LR@{thr[0]:g} | LR@{thr[1]:g} | median |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(t2):
        v = t2[k]
        err = np.concatenate(v["err"]) if v["err"] else np.zeros(0)
        if not len(err) and not v["failed"]:
            continue
        group, scene, mcam, qcam, system, q, area = k
        row = {"scene": scene, "map_cam": mcam, "query_cam": qcam, "system": system, "query": q, "area": area,
               "n_queries": v["n_q"], "n_failed": v["failed"], "n_frames": int(len(err)),
               f"lr@{thr[0]:g}": float(np.mean(err < thr[0])) if len(err) else None,
               f"lr@{thr[1]:g}": float(np.mean(err < thr[1])) if len(err) else None,
               "median": float(np.median(err)) if len(err) else None}
        out["t2"].append(row)
        failed = f" ({v['failed']} failed)" if v["failed"] else ""
        print(f"| {scene} | {mcam} | {qcam} | {system} | {q} | {area} | {v['n_q']}{failed} | "
              f"{len(err)} | {fmt(row[f'lr@{thr[0]:g}'], 3)} | {fmt(row[f'lr@{thr[1]:g}'], 3)} | {fmt(row['median'])} |")
    print(f"\n## T3: successes / counted trials at {thr[0]:g} / {thr[1]:g} m (area of the trial's last frame)\n")
    print(f"| scene | map cam | query cam | system | query | area | queries | trials | RS@{thr[0]:g} | RS@{thr[1]:g} |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(t3):
        v = t3[k]
        if not v["n"] and not v["failed"]:
            continue
        group, scene, mcam, qcam, system, q, area = k
        out["t3"].append({"scene": scene, "map_cam": mcam, "query_cam": qcam, "system": system, "query": q, "area": area,
                          "n_trials": v["n"], "k1": v["k1"], "k2": v["k2"], "n_failed": v["failed"]})
        nq = f"{v['n_q']}" + (f" ({v['failed']} failed)" if v["failed"] else "") if area == "all" else ""
        print(f"| {scene} | {mcam} | {qcam} | {system} | {q} | {area} | {nq} | {v['n']} | {v['k1']}/{v['n']} | {v['k2']}/{v['n']} |")
    if ext_rows:
        print("\n## Extension sessions (new keyframes in the loaded map's alignment)\n")
        cols = ["first_map_ate", "ate_rmse", "ate_rmse_realigned", "rot_err_median_deg", "loaded_shift_median",
                "loaded_ate_after", "n_new_permanent", "localized_frac", "first_localized_frame", "lr@1", "lr@2",
                "completeness", "map_ate_rmse"]
        print("| scene | map cam | system | status | " + " | ".join(cols) + " |")
        print("|" + "---|" * (4 + len(cols)))
        for r in ext_rows:
            print(f"| {r['scene']} | {r['map_cam']} | {r['system']} | {r['status']} | " +
                  " | ".join(fmt(r.get(c), 3) for c in cols) + " |")
    if missing:
        print(f"\n{len(missing)} T2 results without errors.npz (not pooled), e.g. {missing[0]}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
