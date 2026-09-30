#!/usr/bin/env python3
"""Generate report figures from outputs/relpose and outputs/reloc."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs")
FIG = Path("report/figures")
FIG.mkdir(parents=True, exist_ok=True)

C = {"ff-vggt_omega": "#2563eb", "ff-da3": "#7c3aed", "pnp": "#dc2626"}
LBL = {"ff-vggt_omega": "FF (VGGT-Ω) + stereo", "ff-da3": "FF (DA3-L) + stereo", "pnp": "XFeat+LightGlue+PnP"}
VK_CONDS = ["clone", "morning", "overcast", "sunset", "fog", "rain", "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right"]


def load_relpose(name):
    f = OUT / "relpose" / f"{name}.json"
    return json.loads(f.read_text()) if f.is_file() else None


def est_key(a):
    return a["estimator"] + ("-" + a["backend"] if a["estimator"] == "ff" else "")


def fig_vkitti_conditions():
    """Success rate (0.5 m, 5 deg) per query condition, FF vs PnP, one panel per gap."""
    gaps = [0, 5, 10, 20]
    fig, axes = plt.subplots(1, len(gaps), figsize=(16, 3.6), sharey=True)
    for ax, g in zip(axes, gaps):
        x = np.arange(len(VK_CONDS))
        w = 0.27
        for i, est in enumerate(["pnp", "ff-vggt_omega", "ff-da3"]):
            vals = []
            for c in VK_CONDS:
                d = load_relpose(f"vk01_{c}_{'ff_omega' if est == 'ff-vggt_omega' else 'ff_da3' if est == 'ff-da3' else 'pnp'}")
                vals.append(np.nan if d is None else d["summary"][str(g)]["success_0.5m_5deg"])
            ax.bar(x + (i - 1) * w, vals, w, color=C[est], label=LBL[est])
        ax.set_xticks(x)
        ax.set_xticklabels(VK_CONDS, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"reference gap {g} frames")
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("success rate (< 0.5 m, < 5°)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Virtual KITTI 2 Scene01: map = clone, query = other condition")
    fig.tight_layout()
    fig.savefig(FIG / "vkitti_conditions.pdf")
    fig.savefig(FIG / "vkitti_conditions.png", dpi=160)
    plt.close(fig)


def fig_gap_curves():
    """Median translation error and valid rate vs. gap for each dataset."""
    sets = [
        ("vKITTI2 clone→sunset", {"pnp": "vk01_sunset_pnp", "ff-vggt_omega": "vk01_sunset_ff_omega", "ff-da3": "vk01_sunset_ff_da3"}),
        ("TartanAir night P000", {"pnp": "ta_P000_pnp", "ff-vggt_omega": "ta_P000_ff_omega"}),
        ("KITTI raw 0009", {"pnp": "kitti09_pnp", "ff-vggt_omega": "kitti09_ff_omega", "ff-da3": "kitti09_ff_da3"}),
    ]
    fig, axes = plt.subplots(2, len(sets), figsize=(13, 6))
    for j, (title, runs) in enumerate(sets):
        for est, name in runs.items():
            d = load_relpose(name)
            if d is None:
                continue
            gaps = sorted(int(g) for g in d["summary"])
            dist = [d["summary"][str(g)]["gt_dist_median"] for g in gaps]
            succ = [d["summary"][str(g)]["success_0.5m_5deg"] for g in gaps]
            terr = [d["summary"][str(g)]["t_err_median"] or np.nan for g in gaps]
            axes[0, j].plot(dist, succ, "o-", color=C[est], label=LBL[est])
            axes[1, j].plot(dist, terr, "o-", color=C[est], label=LBL[est])
        axes[0, j].set_title(title)
        axes[0, j].set_ylabel("success (<0.5 m, <5°)")
        axes[1, j].set_ylabel("median t err of valid [m]")
        axes[1, j].set_xlabel("median GT distance ref→query [m]")
        for a in axes[:, j]:
            a.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "gap_curves.pdf")
    fig.savefig(FIG / "gap_curves.png", dpi=160)
    plt.close(fig)


def fig_ablation_anchors():
    names = [("curr pair only", "abl_vk01_sunset_anchors_curr_only"), ("curr + 2 ref", "vk01_sunset_ff_omega"),
             ("curr + 4 ref", "abl_vk01_sunset_anchors_4"), ("2 ref only", "abl_vk01_sunset_anchors_ref_only")]
    methods = [("adaptive", "vk01_sunset_ff_omega"), ("median", "abl_vk01_sunset_scale_median"), ("mean", "abl_vk01_sunset_scale_mean"),
               ("huber_log", "abl_vk01_sunset_scale_huber_log"), ("norm_ls", "abl_vk01_sunset_scale_norm_ls")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, items, title in ((axes[0], names, "stereo anchors"), (axes[1], methods, "scale estimator")):
        for lbl, n in items:
            d = load_relpose(n)
            if d is None:
                continue
            gaps = sorted(int(g) for g in d["summary"])
            ax.plot(gaps, [d["summary"][str(g)]["success_0.5m_5deg"] for g in gaps], "o-", label=lbl)
        ax.set_xlabel("reference gap [frames]")
        ax.set_ylabel("success (<0.5 m, <5°)")
        ax.set_title(f"Ablation: {title} (clone→sunset)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_anchors.pdf")
    fig.savefig(FIG / "ablation_anchors.png", dpi=160)
    plt.close(fig)


def fig_scale_hist():
    """Distribution of the per-query relative scale error (stereo vs. GT-implied) on clone->sunset."""
    d = load_relpose("vk01_sunset_ff_omega")
    if d is None:
        return
    rel = []
    for r in d["records"]:
        if r.get("valid") and r["gt_dist"] > 1.0 and "est_dist" in r:
            rel.append(r["est_dist"] / r["gt_dist"])
    if not rel:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(np.clip(rel, 0.5, 1.5), bins=30, color="#93c5fd", edgecolor="white")
    ax.axvline(1.0, color="k", lw=1)
    ax.set_xlabel("estimated / GT translation length")
    ax.set_ylabel("count")
    ax.set_title(f"Metric scale of relative poses (median {np.median(rel):.3f})")
    fig.tight_layout()
    fig.savefig(FIG / "scale_hist.pdf")
    fig.savefig(FIG / "scale_hist.png", dpi=160)
    plt.close(fig)


def fig_reloc_timelines():
    """Component-0 position error over time for a few query conditions, FF vs PnP."""
    conds = ["clone", "sunset", "fog", "30-deg-left"]
    fig, axes = plt.subplots(1, len(conds), figsize=(16, 3.4), sharey=True)
    for ax, c in zip(axes, conds):
        for est in ["pnp", "ff"]:
            f = OUT / "reloc" / f"vk01_{est}" / c / "reloc_rows.json"
            if not f.is_file():
                continue
            rows = json.loads(f.read_text())
            e = np.array([r["c0_t_err"] for r in rows])
            key = "ff-vggt_omega" if est == "ff" else "pnp"
            ax.plot(np.arange(len(e)), np.minimum(e, 30), color=C[key], lw=1.2, label=LBL[key])
        ax.set_title(f"query: {c}")
        ax.set_xlabel("frame")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("comp-0 position error [m]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Relocalization against the clone map (vKITTI2 Scene01)")
    fig.tight_layout()
    fig.savefig(FIG / "reloc_timelines.pdf")
    fig.savefig(FIG / "reloc_timelines.png", dpi=160)
    plt.close(fig)


def fig_reloc_recall():
    groups = [("vKITTI2 (map: clone)", "vk01", VK_CONDS), ("TartanAir night (map: P000)", "ta", ["P000", "P005", "P002", "P004"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [10, 4]})
    for ax, (title, prefix, conds) in zip(axes, groups):
        x = np.arange(len(conds))
        w = 0.38
        for i, est in enumerate(["pnp", "ff"]):
            vals = []
            for c in conds:
                f = OUT / "reloc" / f"{prefix}_{est}" / c / "reloc_summary.json"
                vals.append(json.loads(f.read_text())["c0_recall_1m_5deg"] if f.is_file() else np.nan)
            key = "ff-vggt_omega" if est == "ff" else "pnp"
            ax.bar(x + (i - 0.5) * w, vals, w, color=C[key], label=LBL[key])
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=60, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("frames localized (< 1 m, < 5°)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "reloc_recall.pdf")
    fig.savefig(FIG / "reloc_recall.png", dpi=160)
    plt.close(fig)


def fig_cadence():
    xs, rec, fps, first = [], [], [], []
    for iv in [1, 5, 10]:
        f = OUT / "reloc" / f"abl_cadence_{iv}" / "reloc_summary.json"
        if not f.is_file():
            continue
        s = json.loads(f.read_text())
        xs.append(iv)
        rec.append(s["c0_recall_1m_5deg"])
        fps.append(s["fps"])
        first.append(s["c0_first_correct_step_2m"] if s["c0_first_correct_step_2m"] is not None else np.nan)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(xs, rec, "o-", color="#2563eb", label="recall (<1 m, <5°)")
    ax.set_xlabel("observation interval [frames]")
    ax.set_ylabel("recall", color="#2563eb")
    ax2 = ax.twinx()
    ax2.plot(xs, fps, "s--", color="#059669", label="throughput [FPS]")
    ax2.set_ylabel("system throughput [frames/s]", color="#059669")
    ax.set_title("Observation cadence (clone→sunset)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "cadence.pdf")
    fig.savefig(FIG / "cadence.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_vkitti_conditions, fig_gap_curves, fig_ablation_anchors, fig_scale_hist, fig_reloc_timelines, fig_reloc_recall, fig_cadence):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as e:  # keep generating the others
            print("skip", fn.__name__, e)
