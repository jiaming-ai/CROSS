#!/usr/bin/env python3
"""Tables and figures for SimChange-Rearrange (HSSD house / restaurant).

Metric: relocalization success RS (CROSS protocol, 100-frame trials, r_D = 2 m) per variant and system; the rearrangement
axis is reported against the nominal level and against the measured changed-object pixel ratio (CPR) of each variant.
Inputs: outputs/sim/<scene>/<system>/<variant>/reloc_summary.json (GPU results synced from the NAS, CPU baselines local),
outputs/hssd_quant/<scene_id>/quantification.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs/sim")
FIG = Path("report/figures"); FIG.mkdir(parents=True, exist_ok=True)
TAB = Path("report/tables"); TAB.mkdir(parents=True, exist_ok=True)
SCENES = {"hssd_house": ("104348010", "house 104348010"), "hssd_restaurant": ("103997718", "restaurant 103997718")}
SYSTEMS = [("ff_b0.30", "CROSS-stereo (ours, b=0.3 m)", "#2563eb"), ("pnpsgbm_b0.30", "CROSS-PnP (SGBM, b=0.3 m)", "#dc2626"),
           ("pnp", "CROSS-PnP (GT depth)", "#f472b6"), ("orbslam3_b0.30", "ORB-SLAM3 stereo (b=0.3 m)", "#059669"),
           ("rtabmap", "RTAB-Map RGB-D (GT depth)", "#d97706"), ("rtabmapstereo_b0.30", "RTAB-Map stereo (b=0.3 m)", "#b45309"),
           ("mast3r", "MASt3R-SLAM mono", "#7c3aed")]
FAMILIES = {
    "rearrangement level": ["map", "rearr_10", "rearr_25", "rearr_50", "rearr_75", "rearr_100"],
    "rearrangement: seeds / operations": ["rearr_50", "rearr_50_s1", "rearr_50_s2", "rearr_50_remove", "rearr_50_relocate"],
    "illumination": ["map", "light_morning", "light_evening", "light_overcast", "light_night"],
    "viewpoint / traversal": ["map", "offset_1.0", "yaw_45", "half", "reverse"],
    "combined": ["rearr_50+light_night", "rearr_50+reverse", "rearr_50+offset_1.0+light_overcast", "rearr_100+light_evening+reverse"],
    "hard combined": ["rearr_100_relocate", "rearr_100+reverse", "rearr_100+light_night", "rearr_100+offset_1.0", "rearr_100_relocate+reverse",
                      "rearr_100+light_night+reverse", "rearr_100+offset_1.0+light_night+reverse"],
}
LEVEL = {"map": 0.0, "rearr_10": 0.10, "rearr_25": 0.25, "rearr_50": 0.5, "rearr_75": 0.75, "rearr_100": 1.0,
         "rearr_50_s1": 0.5, "rearr_50_s2": 0.5, "rearr_50_remove": 0.5, "rearr_50_relocate": 0.5, "rearr_100_relocate": 1.0}


def load(scene, system, variant, key="RS"):
    f = OUT / scene / system / variant / "reloc_summary.json"
    if not f.is_file():
        return np.nan
    d = json.loads(f.read_text())
    if "error" in d:
        return 0.0
    v = d.get(key)
    if v is None and key.startswith("rel_"):
        v = (d.get("map_relative") or {}).get(key[4:])
    return np.nan if v is None else float(v)


def cpr(scene, variant):
    """Mean changed-object pixel ratio of the variant's rearrangement (combined variants share the plan of their rearr part)."""
    sid = SCENES[scene][0]
    f = Path("outputs/hssd_quant") / sid / "quantification.json"
    if not f.is_file():
        return np.nan
    q = json.loads(f.read_text())
    key = next((p for p in variant.split("+") if p.startswith("rearr_")), None)
    if variant == "map":
        return 0.0
    if key is None:
        return np.nan
    return float(q[key]["cpr_obj_mean"]) if key in q else np.nan


def all_variants():
    seen, cols = set(), []
    for fam in FAMILIES.values():
        for v in fam:
            if v not in seen:
                seen.add(v); cols.append(v)
    return cols


def fmt(v):
    return "" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.2f}" if isinstance(v, float) else str(v))


def short(c):
    return (c.replace("light_", "").replace("offset_", "off ").replace("yaw_", "yaw ").replace("rearr_", "R").replace("_remove", " rm")
            .replace("_relocate", " rel").replace("_s1", " s1").replace("_s2", " s2"))


def md_and_tex(name, cols, rows, caption, extra_row=None):
    md = [f"\n### {caption}\n", "| system | " + " | ".join(cols) + " | mean |", "|---|" + "---|" * (len(cols) + 1)]
    tex = ["\\begin{tabular}{l" + "c" * len(cols) + "c}", "\\toprule",
           "System & " + " & ".join("\\rotatebox{60}{" + short(c).replace("_", "\\_") + "}" for c in cols) + " & \\rotatebox{60}{mean} \\\\", "\\midrule"]
    if extra_row is not None:
        lab, vals = extra_row
        md.append(f"| {lab} | " + " | ".join(fmt(v) for v in vals) + " | |")
        tex.append(lab + " & " + " & ".join("--" if np.isnan(v) else f"{v:.2f}" for v in vals) + " & \\\\ \\midrule")
    for label, vals in rows.items():
        arr = np.array(vals, dtype=float)
        mean = float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
        md.append(f"| {label} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(mean)} |")
        tex.append(label.replace("_", "\\_") + " & " + " & ".join("--" if np.isnan(v) else f"{v:.2f}" for v in arr) + f" & {fmt(mean)} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(tex) + "\n")
    return md


def table(scene, key="RS"):
    cols = all_variants()
    rows = {}
    for sysk, label, _ in SYSTEMS:
        vals = [load(scene, sysk, v, key) for v in cols]
        if not all(np.isnan(x) for x in vals):
            rows[label] = vals
    return cols, rows


def fig_tolerance(scene, key="RS"):
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(20, 4.2), sharey=True)
    for ax, (fam, variants) in zip(axes, FAMILIES.items()):
        x = np.arange(len(variants))
        for sysk, label, color in reversed(SYSTEMS):
            y = [load(scene, sysk, v, key) for v in variants]
            if all(np.isnan(v) for v in y):
                continue
            ours = sysk.startswith("ff_")
            ax.plot(x, y, "s-" if ours else "o-", color=color, label=label, lw=2.8 if ours else 1.5, ms=6 if ours else 4, zorder=10 if ours else 3)
        ax.set_xticks(x); ax.set_xticklabels([short(v).replace("+", "\n+") for v in variants], rotation=45, ha="right", fontsize=8)
        ax.set_title(fam, fontsize=11); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25)
    axes[0].set_ylabel("relocalization success" + (" (2 m)" if key == "RS" else " (1 m, 5°)"))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(f"SimChange-Rearrange / {SCENES[scene][1]}: relocalization success of 100-frame trials against the map traversal", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / f"hssd_rs_{scene}{'' if key == 'RS' else '_1m'}.pdf"); fig.savefig(FIG / f"hssd_rs_{scene}{'' if key == 'RS' else '_1m'}.png", dpi=160)
    plt.close(fig)


def fig_rearrangement(scenes=tuple(SCENES)):
    """RS versus the nominal level (left of each pair) and versus the measured changed-object pixel ratio (right)."""
    fig, axes = plt.subplots(1, 2 * len(scenes), figsize=(5.0 * 2 * len(scenes), 3.9), sharey=True)
    for si, scene in enumerate(scenes):
        levels = ["map", "rearr_10", "rearr_25", "rearr_50", "rearr_75", "rearr_100"]
        extra = ["rearr_50_s1", "rearr_50_s2", "rearr_50_remove", "rearr_50_relocate"]
        for ax_i, xkey in ((0, "level"), (1, "cpr")):
            ax = axes[2 * si + ax_i]
            for sysk, label, color in reversed(SYSTEMS):
                ys = [load(scene, sysk, v) for v in levels]
                if all(np.isnan(v) for v in ys):
                    continue
                xs = [LEVEL[v] if xkey == "level" else cpr(scene, v) for v in levels]
                ours = sysk.startswith("ff_")
                ax.plot(xs, ys, "s-" if ours else "o-", color=color, label=label, lw=2.8 if ours else 1.5, ms=6 if ours else 4, zorder=10 if ours else 3)
                xe = [LEVEL[v] if xkey == "level" else cpr(scene, v) for v in extra]; ye = [load(scene, sysk, v) for v in extra]
                ax.plot(xe, ye, "x", color=color, ms=6, mew=1.5, zorder=4)
            ax.set_xlabel("rearrangement level (fraction of movable objects changed)" if xkey == "level" else "measured change: mean changed-object pixel ratio")
            ax.set_title(f"{SCENES[scene][1]}", fontsize=10); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25)
            if xkey == "level":
                ax.set_xticks([0, 0.1, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_ylabel("relocalization success (2 m)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle("Tolerance to object rearrangement (crosses: other seeds and single-operation variants at level 50 %)", fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIG / "hssd_rearrangement.pdf"); fig.savefig(FIG / "hssd_rearrangement.png", dpi=160)
    plt.close(fig)


SPLIT_A = ["map", "rearr_10", "rearr_25", "rearr_50", "rearr_75", "rearr_100", "rearr_50_s1", "rearr_50_s2", "rearr_50_remove", "rearr_50_relocate",
           "light_morning", "light_evening", "light_overcast", "light_night"]

# Headline grouping of the 29 variants: everything that changes one factor, the mild combinations of
# section "combined", and the hard combined family that pairs a 100 % rearrangement with the other changes.
SINGLE_FACTOR = ["map", "rearr_10", "rearr_25", "rearr_50", "rearr_75", "rearr_100", "rearr_50_s1", "rearr_50_s2",
                 "rearr_50_remove", "rearr_50_relocate", "rearr_100_relocate", "light_morning", "light_evening",
                 "light_overcast", "light_night", "offset_1.0", "yaw_45", "half", "reverse"]
GROUPS = {"single factor": SINGLE_FACTOR, "combined": FAMILIES["combined"],
          "hard combined": [v for v in FAMILIES["hard combined"] if v != "rearr_100_relocate"], "all": None}
SHORT_GROUP = {"single factor": "single", "combined": "comb.", "hard combined": "hard", "all": "all"}


def write_group_table():
    """Mean RS per change group and scene -- the headline table of the section."""
    md = ["\n### Mean RS (2 m) per change group\n",
          "| system | " + " | ".join(f"{s.replace('hssd_', '')}: {g} ({len(vs) if vs else len(all_variants())})"
                                     for s in SCENES for g, vs in GROUPS.items()) + " |",
          "|---|" + "---|" * (len(SCENES) * len(GROUPS))]
    tex = ["\\begin{tabular}{l" + ("cccc" * len(SCENES)) + "}", "\\toprule",
           " & " + " & ".join("\\multicolumn{4}{c}{" + SCENES[s][1] + "}" for s in SCENES) + " \\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
           "System & " + " & ".join(f"{SHORT_GROUP[g]} ({len(vs) if vs else len(all_variants())})" for _ in SCENES for g, vs in GROUPS.items()) + " \\\\",
           "\\midrule"]
    for sysk, label, _ in SYSTEMS:
        vals = []
        for scene in SCENES:
            for g, vs in GROUPS.items():
                a = np.array([load(scene, sysk, v) for v in (vs or all_variants())], dtype=float)
                vals.append(float(np.nanmean(a)) if np.isfinite(a).any() else np.nan)
        md.append(f"| {label} | " + " | ".join(fmt(v) for v in vals) + " |")
        tex.append(label.replace("_", "\\_") + " & " + " & ".join("--" if np.isnan(v) else f"{v:.2f}" for v in vals) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "hssd_groups_RS.tex").write_text("\n".join(tex) + "\n")
    return md


def write_failure_table():
    """How the systems fail on the hard combined traversals: no pose returned vs. a confident pose in the wrong place."""
    hard = FAMILIES["hard combined"]
    md = ["\n### Failure mode on the hard combined traversals (pooled over the 7 variants x 2 scenes)\n",
          "| system | trials | failures | no pose returned | wrong place (>5 m) | near miss (2-5 m) | median failure error [m] |",
          "|---|---|---|---|---|---|---|"]
    tex = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
           "System & trials & fail. & no pose & $>$5\\,m & 2--5\\,m & med.\\ [m] \\\\", "\\midrule"]
    for sysk, label, _ in SYSTEMS:
        n = nf = nolock = far = near = 0
        errs = []
        for scene in SCENES:
            for v in hard:
                f = OUT / scene / sysk / v / "reloc_summary.json"
                if not f.is_file():
                    continue
                tr = (json.loads(f.read_text()).get("trials") or {}).get("trials") or []
                n += len(tr)
                for t in tr:
                    if t.get("success_rd"):
                        continue
                    nf += 1
                    e = t.get("final_t_err")
                    if e is None or not np.isfinite(e):
                        nolock += 1
                    else:
                        errs.append(e)
                        if e > 5.0:
                            far += 1
                        else:
                            near += 1
        if n == 0:
            continue
        med = f"{np.median(errs):.1f}" if errs else "--"
        cells = [str(n), str(nf), f"{nolock} ({nolock / max(nf, 1):.0%})", f"{far} ({far / max(nf, 1):.0%})",
                 f"{near} ({near / max(nf, 1):.0%})", med]
        md.append(f"| {label} | " + " | ".join(cells) + " |")
        tex.append(label.replace("_", "\\_") + " & " + " & ".join(c.replace("%", "\\%") for c in cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "hssd_failure_modes.tex").write_text("\n".join(tex) + "\n")
    return md



def write_tables():
    md = []
    for scene in SCENES:
        if not (OUT / scene).is_dir():
            continue
        cols = all_variants()
        cprs = [cpr(scene, v) for v in cols]
        for key, name in (("RS", "RS (2 m)"), ("RS_1m_5deg", "RS (1 m, 5°)"), ("rel_t_err_median", "median position error [m]")):
            cols, rows = table(scene, key)
            if rows:
                md += md_and_tex(f"hssd_{scene}_{key}", cols, rows, f"{scene}: {name}", extra_row=("measured change (CPR)", cprs) if key == "RS" else None)
                # page-width halves for the LaTeX report (the markdown keeps the full-width table)
                ia = [i for i, c in enumerate(cols) if c in SPLIT_A]; ib = [i for i, c in enumerate(cols) if c not in SPLIT_A]
                for suffix, idx in (("a", ia), ("b", ib)):
                    sub_rows = {lab: [vals[i] for i in idx] for lab, vals in rows.items()}
                    md_and_tex(f"hssd_{scene}_{key}_{suffix}", [cols[i] for i in idx], sub_rows, "",
                               extra_row=("measured change (CPR)", [cprs[i] for i in idx]) if key == "RS" else None)
    md += write_group_table()
    md += write_failure_table()
    Path("outputs/summary_hssd.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    write_tables()
    for scene in SCENES:
        if (OUT / scene).is_dir():
            for key in ("RS", "RS_1m_5deg"):
                try:
                    fig_tolerance(scene, key)
                except Exception as e:
                    print("skip", scene, key, e)
    try:
        fig_rearrangement([s for s in SCENES if (OUT / s).is_dir()])
    except Exception as e:
        print("skip rearrangement figure", e)
