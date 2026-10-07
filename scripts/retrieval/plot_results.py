"""Figures of the retrieval-index studies (PNG, light theme) from the JSON results in a page_data folder.

    python scripts/retrieval/plot_results.py <page_data>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def latency(pd):
    f = os.path.join(pd, "latency_scaling_5090.json")
    if not os.path.exists(f):
        return
    rows = json.load(open(f))["rows"]
    series = [("full_fp32_gpu", "full 16384-d fp32, GPU (original)"), ("code256_fp16_gpu", "256-d codes, GPU exact"),
              ("ivf256_gpu", "256-d codes, GPU IVF"), ("code256_fp32_cpu", "256-d codes, CPU exact (8 threads)"),
              ("ivf256_cpu", "256-d codes, CPU IVF (8 threads)")]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150, facecolor="white")
    for i, (k, lab) in enumerate(series):
        pts = [(r["n"], r[k]["p50_ms"]) for r in rows if k in r]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, "-o", color=SERIES[i], lw=2, ms=5, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    style(ax, "One retrieval (scores + top-10) vs database size, RTX 5090", "keyframes in the database",
          "latency p50 (ms)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(pd, "latency_scaling.png")); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150, facecolor="white")
    for i, (k, lab) in enumerate([("bytes_full_fp32", "full 16384-d fp32 (original)"),
                                   ("bytes_code512_fp16", "512-d fp16 codes"), ("bytes_code256_fp16", "256-d fp16 codes")]):
        x = [r["n"] for r in rows]; y = [r[k] / 1e9 for r in rows]
        ax.plot(x, y, "-o", color=SERIES[i], lw=2, ms=5, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    style(ax, "Descriptor memory vs database size", "keyframes", "GB")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(pd, "descriptor_memory.png")); plt.close(fig)


def projection(pd):
    f = os.path.join(pd, "projection_study_bench.json")
    if not os.path.exists(f):
        return
    rows = [r for r in json.load(open(f)) if r["dataset"] != "kitti"]
    sets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, facecolor="white")
    variants = [("self", "uc", "fitted on the map, uncentred (chosen)"), ("self", "pca", "fitted on the map, PCA + renorm"),
                ("others", "uc", "fitted on other datasets, uncentred"), ("others", "pca", "other datasets, PCA + renorm")]
    for j, ds in enumerate(sets):
        full = [r for r in rows if r["dataset"] == ds and r["source"] == "full"][0]["R@1"]
        for i, (src, mode, lab) in enumerate(variants):
            pts = sorted((r["dim"], r["R@1"] - full, r["mae_top50"]) for r in rows
                         if r["dataset"] == ds and r["source"] == src and r.get("mode") == mode)
            if pts:
                axes[0].plot([p[0] for p in pts], [p[1] for p in pts], ["-o", "--s", ":^"][j], color=SERIES[i], lw=2, ms=5,
                             label=f"{lab}" if j == 0 else None)
                axes[1].plot([p[0] for p in pts], [p[2] for p in pts], ["-o", "--s", ":^"][j], color=SERIES[i], lw=2, ms=5)
    for ax in axes:
        ax.set_xscale("log", base=2)
    style(axes[0], "Recall@1 change vs full descriptor", "code dimensions", "Delta R@1")
    style(axes[1], "Score error (50 best matches)", "code dimensions", "mean |score - full score|")
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK2)
    axes[1].text(0.98, 0.95, "marker: " + ", ".join(f"{m} {d}" for m, d in zip(["o", "s", "^"], sets)),
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=7, color=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(pd, "projection_study.png")); plt.close(fig)


if __name__ == "__main__":
    pd = sys.argv[1]
    latency(pd)
    projection(pd)
