#!/usr/bin/env python3
"""Re-derive the relocalization metrics of stored CROSS runs from their per-frame rows.

`reloc_rows.json` keeps the estimated pose (map frame) and the ground-truth pose of every processed
frame, so every error and every summary can be recomputed offline.  This is needed once: the rotation
errors that the runs stored were computed with the trace formula on rotation blocks of poses whose
quaternions had drifted away from unit norm (float32 compositions in pypose; see
`cross.utils.lie_tensor.normalize_SE3`), which reports several degrees for a sub-degree error and
made the strict 1 m / 5 deg criterion rotation-limited.  Baseline runs (ORB-SLAM3, RTAB-Map,
MASt3R-SLAM) store proper rotations and need no recomputation.

    python scripts/recompute_reloc_metrics.py outputs/sim/hssd_house outputs/sim/hssd_restaurant ...

The original summary is kept as `reloc_summary.orig.json` (written once); `reloc_rows.json` is rewritten
with the corrected error columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reloc_metrics import _inv, _proper_rotation, _rot_deg, map_relative_errors, summarize_errors, summarize_trials  # noqa: E402


def recompute(d: Path) -> dict | None:
    rows_f, meta_f, summ_f = d / "reloc_rows.json", d / "map_meta.json", d / "reloc_summary.json"
    if not (rows_f.is_file() and meta_f.is_file() and summ_f.is_file()):
        return None
    rows = json.loads(rows_f.read_text())
    if not rows or "gt_pose" not in rows[0] or "c0_pose" not in rows[0]:
        return None                              # external baseline rows: nothing to recompute
    meta = json.loads(meta_f.read_text())
    summary = json.loads(summ_f.read_text())
    orig = d / "reloc_summary.orig.json"
    if not orig.is_file():
        orig.write_text(json.dumps(summary, indent=1))
    T_gt_from_map = np.asarray(meta["T_gt_from_map"])
    for r in rows:
        gt = np.asarray(r["gt_pose"]).reshape(4, 4)
        for key in ("c0", "best"):
            if r.get(f"{key}_pose") is None:
                continue
            T = T_gt_from_map @ np.asarray(r[f"{key}_pose"]).reshape(4, 4)
            err = _inv(gt) @ T
            r[f"{key}_t_err"] = float(np.linalg.norm(err[:3, 3]))
            r[f"{key}_r_err"] = _rot_deg(err[:3, :3])
    for r, e in zip(rows, map_relative_errors(rows, meta)):
        r.update(e)
    r_d = float(summary.get("trials", {}).get("r_d", 2.0))
    trial_summary = summarize_trials(rows, "c0_rel", r_d=r_d)
    trial_summary_best = summarize_trials(rows, "best_rel", r_d=r_d)
    c0 = np.array([[r["c0_t_err"], r["c0_r_err"]] for r in rows])
    best = np.array([[r["best_t_err"], r["best_r_err"]] for r in rows])

    def recall(e, t, rr):
        return float(np.mean((e[:, 0] < t) & (e[:, 1] < rr)))

    summary.update({
        "c0_recall_1m_5deg": recall(c0, 1.0, 5.0), "c0_recall_2m_10deg": recall(c0, 2.0, 10.0), "c0_recall_0.5m_5deg": recall(c0, 0.5, 5.0),
        "best_recall_1m_5deg": recall(best, 1.0, 5.0), "best_recall_2m_10deg": recall(best, 2.0, 10.0), "best_recall_0.5m_5deg": recall(best, 0.5, 5.0),
        "c0_t_err_median": float(np.median(c0[:, 0])), "c0_r_err_median": float(np.median(c0[:, 1])),
        "best_t_err_median": float(np.median(best[:, 0])), "best_r_err_median": float(np.median(best[:, 1])),
        "trials": trial_summary, "trials_best": trial_summary_best,
        "RS": trial_summary["RS"], "RS_1m_5deg": trial_summary["RS_1m_5deg"], "RS_0.5m_5deg": trial_summary["RS_0.5m_5deg"],
        "map_relative": summarize_errors(rows, "c0_rel"), "map_relative_best": summarize_errors(rows, "best_rel"),
        "metrics_recomputed": "rotation errors on SO(3)-projected rotations (scripts/recompute_reloc_metrics.py)",
    })
    rows_f.write_text(json.dumps(rows))
    summ_f.write_text(json.dumps(summary, indent=1))
    old = json.loads(orig.read_text())
    return {"dir": str(d), "RS": summary["RS"], "RS_1m_5deg_old": old.get("RS_1m_5deg"), "RS_1m_5deg": summary["RS_1m_5deg"],
            "r_med_old": old.get("map_relative", {}).get("r_err_median"), "r_med": summary["map_relative"]["r_err_median"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="scene output roots (outputs/sim/<scene>) or run directories")
    args = ap.parse_args()
    done = []
    for root in args.roots:
        root = Path(root)
        dirs = [root] if (root / "reloc_rows.json").is_file() else sorted(p.parent for p in root.glob("*/*/reloc_rows.json"))
        for d in dirs:
            res = recompute(d)
            if res:
                done.append(res)
                print(f"{res['dir']}: RS={res['RS']:.2f}  RS_1m_5deg {res['RS_1m_5deg_old']:.2f} -> {res['RS_1m_5deg']:.2f}  "
                      f"r_err_median {res['r_med_old']:.2f} -> {res['r_med']:.2f} deg")
    print(f"recomputed {len(done)} runs")


if __name__ == "__main__":
    main()
