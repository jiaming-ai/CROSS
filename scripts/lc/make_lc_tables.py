#!/usr/bin/env python3
"""LaTeX tables of the verified-loop-closure study (report/tables/lc_*.tex).

Sources: outputs/lcstudy/<scene>/cur/study_s0/study.json (offline study of the baseline mapping graphs),
configs/noise/<scene>_600.json (ground-truth-free calibration vs the ground-truth fit),
outputs/lcstudy/eval_all.json (A/B evaluation of the recorded runs, scripts/lc/eval_lc.py).
"""
from __future__ import annotations

import json
import re
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "tables"
SCENES = [("lonemonk", "Lone Monk"), ("hssd_house", "HSSD house"), ("hssd_restaurant", "HSSD restaurant")]


def f(v, nd=2, dash="--"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return dash
    return f"{v:.{nd}f}"


def table_offline():
    rows = []
    for key, name in SCENES:
        p = ROOT / "outputs/lcstudy" / key / "cur/study_s0/study.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text()); c = d["calibration"]; g = d["pgo"]; pt = d["prior_test"]["models"]
        loop = c["kinds"].get("loop", {})
        rows.append((name, loop.get("n"), f(loop.get("e_t_p50"), 3), f(loop.get("e_t_p99"), 2),
                     f(c["system_std_abs_z"]["median_over_0.674"], 3), f(c["odom"]["system_std_abs_z"]["p50_over_0.674"], 3),
                     f(c["odom"]["fit"]["k_t"], 3), f(c["odom"]["fit"]["k_r"], 3),
                     f(g["ate_odom_only"]), f(g["system+huber (current PGO)"]["ate"]), f(g["ate_online"]), f(g["fitted huber"]["ate"]), f(g["fitted GNC-TLS"]["ate"]),
                     f(100 * pt["odom=fitted,visual=fitted"]["p0.999"]["TPR"], 0), f(100 * g["fitted huber"].get("injected_post_accept_p0.999", 0), 0)))
    tex = [r"\begin{tabular}{l r r r r r r r r r r r r r r}", r"\toprule",
           r"scene & loop edges & $e_t$ p50 & $e_t$ p99 & $\hat\sigma_{\rm vis}/\sigma$ & $\hat\sigma_{\rm odo}/\sigma$ & $k_t$ & $k_r$ & ATE odo & ATE cur.\ PGO & ATE online & ATE cal.\ Huber & ATE cal.\ GNC & true loops acc.\ (\%) & inj.\ false acc.\ (\%)\\",
           r"\midrule"]
    for r in rows:
        tex.append(" & ".join(str(x) for x in r) + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "lc_offline.tex").write_text("\n".join(tex) + "\n")


def table_calibration():
    rows = []
    for key, name in SCENES:
        p = ROOT / "configs/noise" / f"{key}_600.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text()); gt = d.get("gt_fit", {}); v = d["visual"]; o = d["odom"]
        gv, go = gt.get("visual", {}), gt.get("odom", {})
        rows.append((name, f"{v['t_a']:.3f} + {v['t_b']:.3f}\\,d", f"{gv.get('t_a', float('nan')):.3f} + {gv.get('t_b', float('nan')):.3f}\\,d",
                     f(v.get("scale"), 3), f(gv.get("scale"), 3),
                     f(math.degrees(v["r_a"]), 2), f(math.degrees(gv.get("r_a", float("nan"))), 2) + " + " + f(math.degrees(gv.get("r_b", float("nan"))), 3) + "\\,d",
                     f(o["k_t"], 3) + (r"$^\dagger$" if "upper" in o.get("status_t", "") or "default" in o.get("status_t", "") else ""), f(go.get("k_t"), 3),
                     f(o["k_r"], 3), f(go.get("k_r"), 3), d.get("n_spans"), d.get("n_turn_spans")))
    tex = [r"\begin{tabular}{l l l r r r l r r r r r r}", r"\toprule",
           r"scene & $\sigma_t$ (m), 1 min, no GT & $\sigma_t$, GT fit (full run) & scale, 1 min & scale, GT & $\sigma_r$ (deg) & $\sigma_r$, GT & $k_t$ & $k_t$, GT & $k_r$ & $k_r$, GT & spans & turn spans\\",
           r"\midrule"]
    for r in rows:
        tex.append(" & ".join(str(x) for x in r) + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "lc_calibration.tex").write_text("\n".join(tex) + "\n")


def ab_run(tag: str) -> bool:
    """The A/B set: heuristic runs (cur = seed 0, heur_s1/2, kitti*_heur) and verified runs (final_s*, kitti*_final);
    development runs (ver*, vercal*, *_timing, *_v1 ...) stay out of the tables."""
    return tag == "cur" or bool(re.fullmatch(r"(heur|final)_s\d", tag)) or bool(re.fullmatch(r"kitti\d\ds?_(heur|final)", tag))


def table_ab():
    p = ROOT / "outputs/lcstudy/eval_all.json"
    if not p.exists():
        p = ROOT / "outputs/lcstudy/eval_cur.json"
    runs = json.loads(p.read_text()) if p.exists() else []
    tex = [r"\begin{tabular}{l l l r r r r r r r r r r}", r"\toprule",
           r"scene / run & session & map ATE (m) & err.\ mean & median & $>2$\,m & LC edges & false & precision & constr.\ false / all & rej.\ T/F & cand.\ recall & events F/C\\",
           r"\midrule"]
    for r in sorted(runs, key=lambda x: x["run"]):
        scene = Path(r["run"]).parent.name; tag = Path(r["run"]).name
        if not ab_run(tag):
            continue
        for s in r["sessions"]:
            kind = "loop" if s["id"] == 0 else "cross"
            e = s.get("edges", {}).get(kind, {}); c = s.get("candidates", {}); er = s["error"]; ev = s["events"]
            tex.append(f"{scene.replace('_', ' ')} / {tag.replace('_', ' ')} & s{s['id']} {s['variant'].replace('_', ' ')[:30]} & {f(s.get('map_ate'))} & {f(er['mean'])} & {f(er['median'])} & {f(er['frac_gt_2m'])} & "
                       f"{e.get('n', '--')} & {e.get('n_false', '--')} & {f(e.get('precision'), 3)} & {e.get('n_false_informative', '--')} / {e.get('n_informative', '--')} & "
                       f"{e.get('rejected_true', '--')} / {e.get('rejected_false', '--')} & {f(c.get('recall'), 3)} & {ev['false']}/{ev['fix']}\\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "lc_ab.tex").write_text("\n".join(tex) + "\n")


def table_tolerance():
    """Prior / posterior test acceptance of true and false long-range edges vs the confidence level, per session
    (deployed calibrated noise, per-type adaptive scale replayed offline), from outputs/lcstudy/<scene>/<tag>/tolerance*.json."""
    rows = []
    for key, name in SCENES:
        for tag in ("cur", "final_s0"):
            for fn in ("tolerance_adaptive.json", "tolerance.json"):
                p = ROOT / "outputs/lcstudy" / key / tag / fn
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                for g, rec in d.items():
                    sess = Path(g).stem.replace("graph_", "")
                    lv = rec["levels"]
                    cells = []
                    for c in ("0.99", "0.999", "0.9999"):
                        pr = lv[c]["prior"]; po = lv[c].get("posterior")
                        cells.append(f"{f(pr['TPR'], 2)} / {f(pr['FPR'], 2)}")
                        cells.append(f"{f(po['TPR'], 2)} / {f(po['FPR'], 2)}" if po else "--")
                    rows.append((name, tag.replace("_", " "), sess, "adaptive" if "adaptive" in fn else "static", rec["n_candidates"], rec["n_false"], *cells))
                break
    if not rows:
        return
    tex = [r"\begin{tabular}{l l l l r r c c c c c c}", r"\toprule",
           r"scene & run & session & scale & cand. & false & prior 0.99 & post.\ 0.99 & prior 0.999 & post.\ 0.999 & prior 0.9999 & post.\ 0.9999\\", r"\midrule"]
    for r in rows:
        tex.append(" & ".join(str(x) for x in r) + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "lc_tolerance.tex").write_text("\n".join(tex) + "\n")


def table_rs():
    """Relocalization success (CROSS protocol, 100-frame trials, stride 50, r_D = 2 m) per scene and variant,
    heuristic vs verified (final) mode, from outputs/lcstudy_rs/<scene>/<mode>/<variant>/reloc_summary.json."""
    import glob
    rows = {}
    for fp in glob.glob(str(ROOT / "outputs/lcstudy_rs/*/*/*/reloc_summary.json")):
        pth = Path(fp); scene, mode, var = pth.parts[-4], pth.parts[-3], pth.parts[-2]
        d = json.loads(pth.read_text())
        rows.setdefault((scene, var), {})[mode] = (d.get("RS"), d.get("RS_1m_5deg"), (d.get("map_relative") or {}).get("t_err_median"))
    if not rows:
        return
    tex = [r"\begin{tabular}{l l r r r r r r}", r"\toprule",
           r"scene & variant & RS heur. & RS verified & RS(1\,m,5$^\circ$) heur. & RS(1\,m,5$^\circ$) verified & med.\ err.\ heur. & med.\ err.\ verified\\", r"\midrule"]
    means = {}
    for (scene, var), m in sorted(rows.items()):
        h = m.get("heuristic", (None, None, None)); v = m.get("final", (None, None, None))
        tex.append(f"{scene.replace('_', ' ')} & {var.replace('_', ' ')} & {f(h[0], 2)} & {f(v[0], 2)} & {f(h[1], 2)} & {f(v[1], 2)} & {f(h[2], 2)} & {f(v[2], 2)}\\\\")
        for mode, t in (("heuristic", h), ("final", v)):
            if t[0] is not None:
                means.setdefault((scene, mode), []).append(t[0])
    tex.append(r"\midrule")
    for scene in sorted({sc for sc, _ in rows}):
        hm = means.get((scene, "heuristic")); vm = means.get((scene, "final"))
        tex.append(f"{scene.replace('_', ' ')} & mean over variants & {f(sum(hm) / len(hm) if hm else None, 3)} & {f(sum(vm) / len(vm) if vm else None, 3)} & & & & \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "lc_rs.tex").write_text("\n".join(tex) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table_offline(); table_calibration(); table_ab(); table_tolerance(); table_rs()
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
