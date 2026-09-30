#!/usr/bin/env python3
"""Figures of the verified-loop-closure study (report/figures/lc_*.pdf|png).

Inputs: outputs/lcstudy/<scene>/cur/study_s0/{study.json,edges.json} (offline study of the baseline mapping graphs),
outputs/lcstudy/inpass_map.json (in-pass pair statistics), outputs/lcstudy/eval_all.json (A/B evaluation of the runs),
configs/noise/*_600.json (calibration vs ground truth).

usage: python scripts/lc/make_lc_figures.py [--out report/figures]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# categorical palette (fixed order; validated for CVD separation), text tokens, recessive grid
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SCENES = [("lonemonk", "Lone Monk"), ("hssd_house", "HSSD house"), ("hssd_restaurant", "HSSD restaurant")]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 200, "pdf.fonttype": 42})


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def fig_calibration(root: Path, out: Path):
    """Visual-edge translation error vs distance: per-scene tail-calibrated bins with the fitted model and the
    std the current system assigns (base std / (4 covis score) for typical covis 0.5, score 0.6)."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    for ax, (key, name), col in zip(axes, SCENES, C):
        f = root / key / "cur" / "study_s0" / "study.json"
        if not f.exists():
            ax.set_title(name + " (missing)"); continue
        d = json.loads(f.read_text())
        vf = d["calibration"]["visual_fit"]
        xs = np.array([b["d"] for b in vf["t_bins"]]); ys = np.array([b["sigma"] for b in vf["t_bins"]])
        ax.plot(xs, ys, "o", color=col, ms=4, label="measured (p90 / 2.5)")
        dd = np.linspace(0, max(xs.max() * 1.1, 1), 50)
        ax.plot(dd, vf["t_a"] + vf["t_b"] * dd, "-", color=col, lw=1.5, label=f"fit {vf['t_a']:.3f} + {vf['t_b']:.3f} d")
        sys_std = 0.2 / (4 * 0.5 * 0.6)
        ax.axhline(sys_std, color=INK2, lw=1, ls="--")
        ax.text(dd[-1], sys_std * 1.05, "std the system assigns", ha="right", va="bottom", fontsize=7, color=INK2)
        ax.set_title(name, fontsize=9, color=INK); ax.set_xlabel("edge distance |t| (m)")
        ax.set_yscale("log"); ax.legend(loc="lower right", fontsize=6.5)
    axes[0].set_ylabel("translation noise sigma (m)")
    save(fig, out, "lc_calibration")


def fig_backend(root: Path, out: Path):
    """Map ATE of the baseline mapping graphs under different back ends (offline)."""
    labels = ["odometry only", "current PGO\n(system std + Huber)", "online result\n(merge PGO)", "calibrated\nGaussian", "calibrated\nHuber", "calibrated\nGNC-TLS"]
    keys = ["ate_odom_only", "system+huber (current PGO)", "ate_online", "fitted gaussian", "fitted huber", "fitted GNC-TLS"]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    width = 0.26
    for si, (key, name) in enumerate(SCENES):
        f = root / key / "cur" / "study_s0" / "study.json"
        if not f.exists():
            continue
        p = json.loads(f.read_text())["pgo"]
        vals = []
        for k in keys:
            v = p.get(k)
            vals.append(v["ate"] if isinstance(v, dict) else v)
        x = np.arange(len(keys)) + (si - 1) * width
        bars = ax.bar(x, vals, width=width * 0.92, color=C[si], label=name, linewidth=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
    ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("map ATE (m)"); ax.set_yscale("log"); ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, out, "lc_backend_ate")


def fig_inpass(root: Path, out: Path):
    """In-pass consistency: disagreement of reference pairs with the map, correct pairs vs pairs with a wrong
    reference (ECDF per scene, log x)."""
    f = root / "inpass_map.json"
    if not f.exists():
        print("no inpass_map.json"); return
    data = json.loads(f.read_text())
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for si, (key, name) in enumerate(SCENES):
        rec = next((v for k, v in data.items() if key in k), None)
        if rec is None:
            continue
        for kind, ls in (("both_true", "-"), ("one_false", "--")):
            q = rec.get(kind, {}).get("d_gt")
            if not q:
                continue
            # reconstruct an approximate ECDF from the quantiles
            xs = [q["p50"], q["p90"], q["p99"]]; ys = [0.5, 0.9, 0.99]
            ax.plot(xs, ys, ls, color=C[si], lw=1.5, marker="o", ms=3,
                    label=f"{name}, {'correct pair' if kind == 'both_true' else 'one wrong reference'} (n={rec[kind]['n']})")
    ax.set_xscale("log"); ax.set_xlabel("pair disagreement with the map (m)"); ax.set_ylabel("fraction of pairs below")
    ax.legend(fontsize=5.5, loc="lower right")
    save(fig, out, "lc_inpass")


def fig_ab(root: Path, out: Path):
    """A/B: per session, false long-range edges accepted (count) and candidate recall, heuristic vs verified."""
    f = root / "eval_all.json"
    if not f.exists():
        print("no eval_all.json"); return
    runs = json.loads(f.read_text())
    rows = []
    import re
    for r in runs:
        tag = Path(r["run"]).name; scene = Path(r["run"]).parent.name
        if not (tag == "cur" or re.fullmatch(r"(heur|final)_s\d", tag) or re.fullmatch(r"kitti\d\ds?_(heur|final)", tag)):
            continue
        for s in r["sessions"]:
            e = s.get("edges", {}); kind = "loop" if s["id"] == 0 else "cross"
            ed = e.get(kind, {})
            rows.append((scene, tag, s["id"], s["variant"], ed.get("n_false"), ed.get("precision"), s.get("candidates", {}).get("recall"), s["error"]["mean"], s.get("map_ate")))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    labels = [f"{sc[:6]}/{tag}\ns{sid}" for sc, tag, sid, *_ in rows]
    x = np.arange(len(rows))
    ax.bar(x, [r[4] or 0 for r in rows], color=C[1], width=0.6, linewidth=0)
    ax.set_ylabel("false loop-closure edges accepted", color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax.grid(axis="x", visible=False)
    save(fig, out, "lc_ab_false_edges")


def fig_tolerance(root: Path, out: Path):
    """Acceptance of true and false long-range edges by the prior / posterior tests vs the confidence level
    (relocalization sessions, deployed noise model with the adaptive scale replayed offline)."""
    files = sorted(root.glob("*/*/tolerance_adaptive.json"))
    if not files:
        print("no tolerance files"); return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    levels = ["0.9", "0.99", "0.999", "0.9999", "0.999999"]
    x = np.arange(len(levels))
    k = 0
    for f in files:
        d = json.loads(f.read_text()); scene = f.parts[-3]
        for g, rec in d.items():
            sess = Path(g).stem.replace("graph_", "")
            if rec["n_false"] == 0:
                continue
            col = C[k % len(C)]; k += 1
            for ax, test in zip(axes, ("prior", "posterior")):
                tpr = [rec["levels"][l][test]["TPR"] if rec["levels"][l].get(test) else np.nan for l in levels]
                fpr = [rec["levels"][l][test]["FPR"] if rec["levels"][l].get(test) else np.nan for l in levels]
                ax.plot(x, tpr, "-o", color=col, ms=3, lw=1.5, label=f"{scene} {sess} true (n={rec['n_candidates'] - rec['n_false']})")
                ax.plot(x, fpr, "--s", color=col, ms=3, lw=1.5, label=f"{scene} {sess} false (n={rec['n_false']})")
    for ax, test in zip(axes, ("prior test", "posterior test")):
        ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=7); ax.set_xlabel("confidence level c"); ax.set_title(test, fontsize=9)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("accepted fraction"); axes[1].legend(fontsize=5, ncol=2, loc="lower right")
    save(fig, out, "lc_tolerance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/lcstudy")
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    fig_calibration(root, out)
    fig_backend(root, out)
    fig_inpass(root, out)
    fig_ab(root, out)
    fig_tolerance(root, out)


if __name__ == "__main__":
    main()
