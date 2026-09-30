#!/usr/bin/env python3
"""Tables and figures for the SimChange benchmark.

Primary metric: relocalization success RS (CROSS-paper protocol: fraction of fixed-length trials whose final
estimate is within r_D = 2 m of the ground truth, map-relative), plus RS at 1 m / 5 deg.
Inputs: outputs/sim/<scene>/<system>/<variant>/reloc_summary.json (CROSS/MASt3R on GPU hosts, CPU baselines local;
the two trees are merged by copying outputs/sim_local into outputs/sim).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs/sim")
ABL = Path("outputs/sim_ablation")
FIG = Path("report/figures")
FIG.mkdir(parents=True, exist_ok=True)

SYSTEMS = [("ff_b0.30", "CROSS-stereo (ours, b=0.3 m)", "#2563eb"), ("pnpsgbm_b0.30", "CROSS-PnP (SGBM, b=0.3 m)", "#dc2626"),
           ("pnp", "CROSS-PnP (GT depth)", "#f472b6"),
           ("orbslam3_b0.10", "ORB-SLAM3 stereo (b=0.1 m, its best)", "#059669"), ("rtabmap", "RTAB-Map RGB-D (GT depth)", "#d97706"),
           ("rtabmapstereo_b0.30", "RTAB-Map stereo (b=0.3 m)", "#b45309"),
           ("mast3r", "MASt3R-SLAM mono", "#7c3aed")]
# SimChange-Long (Lone Monk): variants weighted toward the discriminating change types
FAMILIES_LONG = {
    "illumination": ["map", "light_morning", "light_evening", "light_overcast", "light_night"],
    "objects / background": ["map", "move_50", "background"],
    "viewpoint": ["map", "offset_1.0", "offset_2.0", "yaw_45"],
    "traversal / combined": ["map", "half", "reverse", "reverse+offset_1.0", "light_night+reverse", "light_evening+move_50+reverse+offset_1.0"],
}


def families(scene):
    return FAMILIES_LONG if scene.startswith("lonemonk") else FAMILIES


def systems(scene):
    """ORB-SLAM3 was rendered/run at 0.3 m only on the long scene."""
    if scene.startswith("lonemonk"):
        return [(k.replace("orbslam3_b0.10", "orbslam3_b0.30"), l.replace("b=0.1 m, its best", "b=0.3 m"), c) for k, l, c in SYSTEMS]
    return SYSTEMS
FAMILIES = {
    "illumination": ["map", "light_morning", "light_noon", "light_afternoon", "light_evening", "light_overcast", "light_night"],
    "objects / background": ["map", "move_25", "move_50", "move_100", "remove_50", "background"],
    "viewpoint": ["map", "offset_0.5", "offset_1.0", "height_0.4", "yaw_15", "yaw_30", "yaw_45"],
    "traversal": ["map", "half", "reverse", "combo"],
}
BASELINES = ["0.10", "0.30", "0.50"]
ABLATIONS = [("curr_only", "current pair only"), ("curr_ref2", "current + 2 keyframe pairs (default)"), ("curr_ref4", "current + 4 keyframe pairs"),
             ("ref2_only", "2 keyframe pairs only"), ("curr_ref2_odom", "current + 2 kf pairs + odometry anchor"), ("odom_only", "odometry anchor only (no stereo)"),
             ("scale_median", "median ratio"), ("scale_huber", "log-Huber"), ("scale_mean", "mean ratio")]


def load(scene, system, variant, key="RS", root=OUT):
    f = root / scene / system / variant / "reloc_summary.json"
    if not f.is_file():
        return np.nan
    d = json.loads(f.read_text())
    if "error" in d:
        return 0.0
    v = d.get(key)
    if v is None and key.startswith("rel_"):
        v = (d.get("map_relative") or {}).get(key[4:])
    return np.nan if v is None else float(v)


def all_variants(scene):
    seen, cols = set(), []
    for fam in families(scene).values():
        for v in fam:
            if v not in seen:
                seen.add(v)
                cols.append(v)
    return cols


def fmt(v):
    return "" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.2f}" if isinstance(v, float) else str(v))


def table(scene, systems_, key="RS", root=OUT):
    cols = all_variants(scene)
    rows = {}
    for sysk, label, *_ in systems_:
        vals = [load(scene, sysk, v, key, root) for v in cols]
        if not all(np.isnan(x) for x in vals):
            rows[label] = vals
    return cols, rows


def md_and_tex(name, cols, rows, caption):
    md = [f"\n### {caption}\n", "| system | " + " | ".join(cols) + " | mean |", "|---|" + "---|" * (len(cols) + 1)]
    def short(c):
        return c.replace("light_", "").replace("offset_", "off ").replace("height_", "h ").replace("yaw_", "yaw ").replace("move_", "mv ").replace("remove_", "rm ").replace("_", " ")
    tex = ["\\begin{tabular}{l" + "c" * len(cols) + "c}", "\\toprule", "System & " + " & ".join("\\rotatebox{60}{" + short(c) + "}" for c in cols) + " & \\rotatebox{60}{mean} \\\\", "\\midrule"]
    for label, vals in rows.items():
        arr = np.array(vals, dtype=float)
        mean = float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
        md.append(f"| {label} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(mean)} |")
        tex.append(label.replace("_", "\\_") + " & " + " & ".join("--" if np.isnan(v) else f"{v:.2f}" for v in arr) + f" & {fmt(mean)} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path("report/tables").mkdir(exist_ok=True)
    (Path("report/tables") / f"{name}.tex").write_text("\n".join(tex) + "\n")
    return md


def fig_tolerance(scene, key="RS"):
    fams = families(scene)
    fig, axes = plt.subplots(1, len(fams), figsize=(15, 3.9), sharey=True)
    for ax, (fam, variants) in zip(axes, fams.items()):
        x = np.arange(len(variants))
        for si, (sysk, label, color) in enumerate(reversed(systems(scene))):
            y = [load(scene, sysk, v, key) for v in variants]
            if all(np.isnan(v) for v in y):
                continue
            ours = sysk.startswith("ff_")
            ax.plot(x, y, "o-" if not ours else "s-", color=color, label=label, lw=2.8 if ours else 1.5, ms=6 if ours else 4,
                    zorder=10 if ours else 3, alpha=1.0 if ours else 0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("light_", "").replace("_", " ").replace("+", "\n+") for v in variants], rotation=45, ha="right", fontsize=9)
        ax.set_title(fam, fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("relocalization success" + (" (2 m)" if key == "RS" else " (1 m, 5°)"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle(f"SimChange / {scene}: relocalization success of independent trials against the map traversal", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / f"sim_rs_{scene}{'' if key == 'RS' else '_1m'}.pdf")
    fig.savefig(FIG / f"sim_rs_{scene}{'' if key == 'RS' else '_1m'}.png", dpi=160)
    plt.close(fig)


def fig_baselines(scene):
    """RS vs. stereo baseline for CROSS-stereo and ORB-SLAM3, averaged over each family."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (prefix, title, color) in zip(axes, (("ff_b", "CROSS-stereo", "#2563eb"), ("orbslam3_b", "ORB-SLAM3 stereo", "#059669"))):
        for fam, variants in families(scene).items():
            ys = []
            for b in BASELINES:
                vals = np.array([load(scene, f"{prefix}{b}", v) for v in variants if v != "map"], dtype=float)
                ys.append(np.nanmean(vals) if np.isfinite(vals).any() else np.nan)
            ax.plot([float(b) for b in BASELINES], ys, "o-", label=fam)
        ax.set_xlabel("stereo baseline [m]")
        ax.set_title(title)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean relocalization success (2 m)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"SimChange / {scene}: effect of the stereo baseline")
    fig.tight_layout()
    fig.savefig(FIG / f"sim_baseline_{scene}.pdf")
    fig.savefig(FIG / f"sim_baseline_{scene}.png", dpi=160)
    plt.close(fig)


def fig_rd_curves(scenes=("classroom", "archiviz")):
    """Relocalization success as a function of the success radius r_D (final-frame translation error of every
    trial over all query variants), plus the rotation-gated variant, for every system."""
    radii = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    fig, axes = plt.subplots(1, len(scenes), figsize=(5.2 * len(scenes), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, scene in zip(axes, scenes):
        for sysk, label, color in systems(scene):
            errs = []
            for f in (OUT / scene / sysk).glob("*/reloc_summary.json"):
                if f.parent.name == "map":
                    continue
                d = json.loads(f.read_text())
                if "error" in d:
                    continue
                errs += [t["final_t_err"] if t["final_t_err"] is not None else np.inf for t in d["trials"]["trials"]]
            if not errs:
                continue
            e = np.array(errs, dtype=float)
            y = [float(np.mean(e <= r)) for r in radii]
            ours = sysk.startswith("ff_")
            ax.plot(radii, y, "s-" if ours else "o-", color=color, label=label, lw=2.6 if ours else 1.4, ms=5 if ours else 3.5, zorder=10 if ours else 3)
            rows.append((scene, label, y))
        ax.set_xscale("log")
        ax.set_xticks(radii)
        ax.set_xticklabels([f"{r:g}" for r in radii], fontsize=8)
        ax.set_xlabel("success radius $r_D$ [m]")
        ax.set_title(scene)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("relocalization success")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(FIG / "sim_rd_curves.pdf")
    fig.savefig(FIG / "sim_rd_curves.png", dpi=160)
    plt.close(fig)
    Path("report/tables").mkdir(exist_ok=True)
    tex = ["\\begin{tabular}{ll" + "c" * len(radii) + "}", "\\toprule", "Scene & System & " + " & ".join(f"{r:g}" for r in radii) + " \\\\", "\\midrule"]
    for scene, label, y in rows:
        tex.append(f"{scene} & {label} & " + " & ".join(f"{v:.2f}" for v in y) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (Path("report/tables") / "sim_rd_curves.tex").write_text("\n".join(tex) + "\n")


def table_storage():
    """Default (A=2, stored right images) vs. storage-optimized (A=0, no right images) at b = 0.3 m."""
    rows = []
    for scene in ("classroom", "archiviz"):
        for sysk, label, store in (("ff_b0.30", "default ($A=2$)", "1.57 + 1.57"), ("ff_b0.30_curr", "no right images ($A=0$)", "1.57")):
            vals = np.array([load(scene, sysk, v) for v in all_variants(scene) if v != "map"], dtype=float)
            if np.isfinite(vals).any():
                vals1 = np.array([load(scene, sysk, v, "RS_1m_5deg") for v in all_variants(scene) if v != "map"], dtype=float)
                rows.append((scene, label, float(np.nanmean(vals)), float(np.nanmean(vals1)), int(np.isfinite(vals).sum()), store))
    Path("report/tables").mkdir(exist_ok=True)
    tex = ["\\begin{tabular}{llcccc}", "\\toprule", "Scene & Configuration & mean RS & mean RS$_{1\\mathrm{m}/5^\\circ}$ & variants & MB / keyframe \\\\", "\\midrule"]
    for scene, label, rs, rs1, n, store in rows:
        tex.append(f"{scene} & {label} & {rs:.2f} & {rs1:.2f} & {n} & {store} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (Path("report/tables") / "sim_storage.tex").write_text("\n".join(tex) + "\n")


def fig_ablation(scene="classroom"):
    variants = ["map", "light_evening", "light_night", "move_100", "background", "yaw_30", "reverse"]
    names = [a for a, _ in ABLATIONS if (ABL / scene / a).is_dir()]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(11, 3.8))
    M = np.array([[load(scene, a, v, "RS", ABL) for v in variants] for a in names])
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.replace("light_", "") for v in variants], rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([dict(ABLATIONS)[a] for a in names], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=ax, label="RS (2 m)")
    ax.set_title(f"Scale-inference ablation ({scene}, b = 0.3 m)")
    fig.tight_layout()
    fig.savefig(FIG / f"sim_ablation_{scene}.pdf")
    fig.savefig(FIG / f"sim_ablation_{scene}.png", dpi=160)
    plt.close(fig)


def fig_timelines(scene, variants=("light_night", "move_100", "yaw_30", "reverse")):
    fig, axes = plt.subplots(1, len(variants), figsize=(16, 3.3), sharey=True)
    for ax, v in zip(axes, variants):
        for sysk, label, color in systems(scene):
            f = OUT / scene / sysk / v / "reloc_rows.json"
            if not f.is_file():
                continue
            rows = json.loads(f.read_text())
            rows = [r for r in rows if r.get("trial", 0) == 0]
            e = np.array([r.get("c0_rel_t_err") if r.get("c0_rel_t_err") is not None else np.nan for r in rows], dtype=float)
            e[~np.isfinite(e)] = 50.0
            ax.plot(np.minimum(e, 50), color=color, lw=1.1, label=label)
        ax.set_title(f"query: {v} (trial 0)")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("frame")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("position error w.r.t. map [m]")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"sim_timelines_{scene}.pdf")
    fig.savefig(FIG / f"sim_timelines_{scene}.png", dpi=160)
    plt.close(fig)


def fig_examples(scene, variants=("map", "light_morning", "light_evening", "light_night", "light_overcast", "move_100", "background", "yaw_30")):
    import cv2
    ims = []
    for v in variants:
        f = Path("data/sim") / scene / v / "left" / "000030.png"
        if not f.is_file():
            continue
        im = cv2.resize(cv2.imread(str(f)), (320, 240))
        cv2.putText(im, v, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ims.append(im)
    if not ims:
        return
    while len(ims) % 4:
        ims.append(np.zeros_like(ims[0]))
    rows = [np.concatenate(ims[i:i + 4], 1) for i in range(0, len(ims), 4)]
    cv2.imwrite(str(FIG / f"sim_examples_{scene}.png"), np.concatenate(rows, 0))


def write_tables():
    md = []
    for scene in sorted(p.name for p in OUT.iterdir() if p.is_dir()):
        for key, name in (("RS", "RS (2 m)"), ("RS_1m_5deg", "RS (1 m, 5°)"), ("rel_t_err_median", "median position error [m]")):
            cols, rows = table(scene, systems(scene), key)
            if rows:
                md += md_and_tex(f"sim_{scene}_{key}", cols, rows, f"{scene}: {name}")
        bsys = [(f"ff_b{b}", f"CROSS-stereo b={b}") for b in BASELINES] + [(f"orbslam3_b{b}", f"ORB-SLAM3 b={b}") for b in BASELINES]
        cols, rows = table(scene, bsys, "RS")
        if rows:
            md += md_and_tex(f"sim_{scene}_baselines", cols, rows, f"{scene}: RS (2 m) vs. stereo baseline")
    if ABL.is_dir():
        for scene in sorted(p.name for p in ABL.iterdir() if p.is_dir()):
            cols, rows = table(scene, [(a, l) for a, l in ABLATIONS], "RS", ABL)
            if rows:
                md += md_and_tex(f"sim_ablation_{scene}", cols, rows, f"{scene}: scale-inference ablation, RS (2 m)")
    Path("outputs/summary_sim.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    if OUT.is_dir():
        for scene in sorted(p.name for p in OUT.iterdir() if p.is_dir()):
            for fn in (lambda s: fig_tolerance(s, "RS"), lambda s: fig_tolerance(s, "RS_1m_5deg"), fig_baselines, fig_timelines, fig_examples):
                try:
                    fn(scene)
                except Exception as e:
                    print("skip", scene, e)
        try:
            fig_ablation("classroom")
        except Exception as e:
            print("skip ablation", e)
        try:
            fig_rd_curves()
        except Exception as e:
            print("skip rd curves", e)
        try:
            table_storage()
        except Exception as e:
            print("skip storage table", e)
        write_tables()
