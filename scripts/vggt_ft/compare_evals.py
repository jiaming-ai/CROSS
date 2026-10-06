#!/usr/bin/env python3
"""Compare held-out evaluations (vggt_ft.evaluate JSONs) against a reference (the released model).

    python scripts/vggt_ft/compare_evals.py released.json v1_002000.json [more.json ...] [--md]

Per test set: pose AUC@3 / AUC@30, depth AbsRel (per-frame aligned), covisibility AUROC (learned head if present, else
CROSS's geometric score from the predictions), and the metric-scale numbers of the scale head (AbsRel of metric depth
without alignment, |log scale error|, median predicted / true translation length).
"""
import argparse
import json
from pathlib import Path

COLS = [("auc3", "AUC@3"), ("auc30", "AUC@30"), ("d_abs", "AbsRel"), ("cov", "covis AUROC"), ("m_abs", "metric AbsRel"),
        ("lse", "|log s err|"), ("t_ratio_med", "t ratio")]


def cell(res, key):
    if key == "cov":
        v = res.get("cov_auroc", res.get("geo_auroc"))
        tag = "h" if "cov_auroc" in res else "g"
        return (v, tag)
    return (res.get(key), "")


def fmt(v):
    return "-" if v is None or v != v else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    runs = [(Path(f).stem, json.load(open(f))) for f in a.files]
    sets = [k for k, v in runs[0][1].items() if isinstance(v, dict)]
    names = [n for n, _ in runs]
    if a.md:
        print("| set | " + " | ".join(f"{c} ({'/'.join(names)})" for _, c in COLS) + " |")
        print("|---" * (len(COLS) + 1) + "|")
    for s in sets:
        row = []
        for key, _ in COLS:
            vals = []
            for _, r in runs:
                v, tag = cell(r.get(s, {}), key)
                vals.append(fmt(v) + tag)
            row.append(" / ".join(vals))
        if a.md:
            print(f"| {s} | " + " | ".join(row) + " |")
        else:
            print(f"{s:16s} " + "  ".join(f"{c}: {x}" for (_, c), x in zip(COLS, row)))


if __name__ == "__main__":
    main()
