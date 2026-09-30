#!/usr/bin/env python3
"""Local result page of the verified-loop-closure study: outputs/site/loop_closure.html (relative links only).

Links the report (PDF / HTML), the design note, the figures and the evaluation tables (eval_all.md), the
calibration files, and every recorded run directory.  Run scripts/viz/build_site.py afterwards so that the
index page picks up the new card (build_site.py links loop_closure.html when it exists).
"""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "outputs" / "site"
LC = ROOT / "outputs" / "lcstudy"

CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:0 auto;padding:16px 20px;line-height:1.5;color:#1d1d1b;background:#fcfcfb}
h1{font-size:1.5em}h2{font-size:1.15em;margin-top:1.6em;border-bottom:1px solid #e6e5e0;padding-bottom:4px}
table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #e6e5e0;padding:3px 6px;text-align:right}th{background:#f2f1ec}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}
figure{margin:12px 0}figcaption{font-size:12px;color:#52514e}img{max-width:100%}.tw{overflow-x:auto}
code{background:#f2f1ec;padding:1px 4px;border-radius:3px}a{color:#2a78d6}
"""


def md_table_to_html(md: str) -> str:
    rows = [r.strip().strip("|").split("|") for r in md.strip().splitlines() if r.strip().startswith("|")]
    rows = [r for r in rows if not all(set(c.strip()) <= set("-: ") for c in r)]
    if not rows:
        return ""
    h = ["<div class='tw'><table><tr>" + "".join(f"<th>{html.escape(c.strip())}</th>" for c in rows[0]) + "</tr>"]
    for r in rows[1:]:
        h.append("<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in r) + "</tr>")
    return "".join(h) + "</table></div>"


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    figdir = SITE / "figures"; figdir.mkdir(exist_ok=True)
    figs = []
    for name, cap in [("lc_calibration", "Visual-edge translation error vs distance with the fitted noise model and the std the previous system assigned."),
                      ("lc_backend_ate", "Map ATE of the same mapping graphs under different back ends (offline)."),
                      ("lc_inpass", "In-pass consistency: disagreement of reference pairs with the map, correct pairs vs pairs with a wrong reference."),
                      ("lc_ab_false_edges", "False loop-closure edges accepted per session, heuristic vs verified runs.")]:
        src = ROOT / "report" / "figures" / f"{name}.png"
        if src.exists():
            shutil.copy(src, figdir / f"{name}.png"); figs.append((name, cap))
    h = [f"<!doctype html><html><head><meta charset='utf-8'><title>Verified loop closure</title><style>{CSS}</style></head><body>",
         "<p><a href='index.html'>&larr; all results</a></p>",
         "<h1>Verified loop closure: calibrated consistency tests instead of tuned thresholds</h1>",
         "<p>Study of 2026-09-12. The intra-hypothesis PGO of 2026-09-08 missed loop closures because the back end was "
         "miscalibrated by 20-70x; it is replaced by three consistency tests (prior / in-pass / posterior) with one "
         "parameter (chi-square confidence), a noise model calibrated without ground truth from a minute of data, and an "
         "online noise scale for appearance change. Sources: <a href='../../design/loop_closure_verified.md'>design note</a>, "
         "report section <em>Verified Loop Closure</em> (<a href='report.html'>HTML</a>, <a href='../../report/main.pdf'>PDF</a>), "
         "code <code>cross/core/lc_verify.py</code>, tools <code>scripts/lc/</code>.</p>"]
    h.append("<h2>Figures</h2>")
    for name, cap in figs:
        h.append(f"<figure><img src='figures/{name}.png' alt='{html.escape(cap)}'><figcaption>{html.escape(cap)}</figcaption></figure>")
    for title, f in [("Recorded runs: heuristic vs verified (scripts/lc/eval_lc.py)", LC / "eval_all.md"), ("Baseline runs (current system)", LC / "eval_cur.md")]:
        if f.exists():
            h.append(f"<h2>{html.escape(title)}</h2>" + md_table_to_html(f.read_text()))
    h.append("<h2>Calibration files (ground-truth-free, one minute)</h2><ul>")
    for f in sorted((ROOT / "configs" / "noise").glob("*.yaml")):
        h.append(f"<li><a href='../../configs/noise/{f.name}'>{f.name}</a></li>")
    h.append("</ul><h2>Run directories</h2><ul>")
    for d in sorted(LC.glob("*/*/")):
        if (d / "trace.json").exists():
            h.append(f"<li><a href='../lcstudy/{d.parent.name}/{d.name}/'>{d.parent.name}/{d.name}</a></li>")
    h.append("</ul></body></html>")
    (SITE / "loop_closure.html").write_text("\n".join(h))
    print("wrote", SITE / "loop_closure.html")


if __name__ == "__main__":
    main()
