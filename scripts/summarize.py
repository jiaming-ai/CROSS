#!/usr/bin/env python3
"""Aggregate outputs/relpose/*.json and outputs/reloc/*/reloc_summary.json into tables (markdown + csv + json)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("outputs")


def relpose_table():
    rows = []
    for f in sorted((OUT / "relpose").glob("*.json")):
        d = json.loads(f.read_text())
        a = d["args"]
        for gap, s in d["summary"].items():
            rows.append({
                "file": f.stem, "ref": Path(a["ref"]).name, "query": Path(a["query"] or a["ref"]).name,
                "estimator": a["estimator"] + ("-" + a["backend"] if a["estimator"] == "ff" else ""),
                "n_ref_anchors": a.get("n_ref_anchors"), "curr_anchor": not a.get("no_curr_anchor", False),
                "scale_method": a.get("scale_method"), "gap": int(gap), "gt_dist_median": s["gt_dist_median"],
                "valid_rate": s["valid_rate"], "t_err_median": s["t_err_median"], "r_err_median": s["r_err_median"],
                "succ_0.25m_2deg": s["success_0.25m_2deg"], "succ_0.5m_5deg": s["success_0.5m_5deg"],
                "succ_1m_10deg": s["success_1m_10deg"], "time_s": d["time_median_s"],
            })
    return rows


def reloc_table():
    rows = []
    for f in sorted((OUT / "reloc").glob("**/reloc_summary.json")):
        s = json.loads(f.read_text())
        a = json.loads((f.parent / "args.json").read_text())
        m = json.loads((f.parent / "map_meta.json").read_text())
        rows.append({
            "run": str(f.parent.relative_to(OUT / "reloc")), "map": Path(a["map"]).name, "query": Path(a["query"]).name,
            "estimator": a["estimator"] + ("-" + a["backend"] if a["estimator"] == "ff" else ""),
            "obs_interval": a.get("obs_max_interval"), "obs_min_t": a.get("obs_min_translation"),
            "map_ate": m["map_ate_rmse"], "map_kfs": m["n_permanent"],
            "n_frames": s["n_frames"], "n_obs": s["n_observations"],
            "rel_recall_0.5m": (s.get("map_relative") or {}).get("recall_0.5m_5deg"),
            "rel_recall_1m": (s.get("map_relative") or {}).get("recall_1m_5deg"),
            "rel_recall_2m": (s.get("map_relative") or {}).get("recall_2m_10deg"),
            "rel_first_1m": (s.get("map_relative") or {}).get("first_correct_step_1m"),
            "rel_t_med": (s.get("map_relative") or {}).get("t_err_median"),
            "rel_best_recall_1m": (s.get("map_relative_best") or {}).get("recall_1m_5deg"),
            "c0_recall_1m_5deg": s["c0_recall_1m_5deg"], "c0_recall_2m_10deg": s["c0_recall_2m_10deg"],
            "best_recall_1m_5deg": s["best_recall_1m_5deg"], "best_recall_2m_10deg": s["best_recall_2m_10deg"],
            "c0_first_2m": s["c0_first_correct_step_2m"], "best_first_2m": s["best_first_correct_step_2m"],
            "c0_first_1m": s["c0_first_correct_step_1m"],
            "c0_t_med": s["c0_t_err_median"], "c0_r_med": s["c0_r_err_median"],
            "conv_t_med": (s["c0_median_after_converge_2m"] or [None, None])[0],
            "fps": s["fps"], "step_mean_s": s["step_time_mean_s"],
        })
    return rows


def to_markdown(rows, cols):
    if not rows:
        return ""
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def main():
    rp = relpose_table()
    rl = reloc_table()
    (OUT / "summary_relpose.json").write_text(json.dumps(rp, indent=1))
    (OUT / "summary_reloc.json").write_text(json.dumps(rl, indent=1))
    md = ["# Relative pose (module level)\n", to_markdown(rp, ["file", "gap", "gt_dist_median", "valid_rate", "t_err_median", "r_err_median", "succ_0.5m_5deg", "succ_1m_10deg", "time_s"]),
          "\n\n# Relocalization (system level)\n", to_markdown(rl, ["run", "map_ate", "map_kfs", "n_frames", "n_obs", "rel_recall_0.5m", "rel_recall_1m", "rel_recall_2m", "rel_first_1m", "rel_t_med", "rel_best_recall_1m", "c0_recall_1m_5deg", "c0_recall_2m_10deg", "c0_t_med", "fps"])]
    (OUT / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
