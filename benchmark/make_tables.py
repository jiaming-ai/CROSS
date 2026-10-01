#!/usr/bin/env python3
"""Write benchmark/RESULTS.md from benchmark/results/results.json (generated file: do not edit RESULTS.md by hand).

  python benchmark/make_tables.py [--results benchmark/results/results.json] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SETUP_LABEL = {"rgbd": "RGB-D", "stereo": "stereo", "mono": "mono"}
PENDING, NA, FAIL = "·", "", "✗"


def load():
    ds = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())
    sy = yaml.safe_load((ROOT / "benchmark/configs/systems.yaml").read_text())
    return ds, sy


def rows_for(ds_cfg, sy, dataset):
    """(system, setup) rows available for a dataset, in systems.yaml order."""
    out = []
    for system, sc in sy.items():
        if sc.get("hidden"):
            continue
        for setup in sc["setups"]:
            if setup in ds_cfg[dataset].get("setups", {"rgbd": 1, "stereo": 1, "mono": 1}) and \
                    f"{dataset}/{setup}" not in sc.get("skip", []):
                out.append((system, setup))
    return out


def row_label(sy, system, setup, dataset=None):
    lab = sy[system]["label"]
    s = SETUP_LABEL[setup]
    if dataset == "kitti" and setup == "rgbd":
        s = "RGB-D*"
    odo = " ⁽ᵒ⁾" if sy[system].get("uses_odometry") else ""
    return f"{lab} · {s}{odo}"


def fmt(x, nd=3):
    return PENDING if x is None else f"{x:.{nd}f}"


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def t2_pool(rs, thr):
    """T2 over several query sessions, pooled over their frames: LR = localized frames / all frames; MS-ATE = RMSE over
    all frames that have an estimate (i.e. every session weighted by its length, as T3 pools trials)."""
    rs = [r for r in rs if r.get(f"lr@{thr[0]:g}") is not None and r.get(f"lr@{thr[1]:g}") is not None]
    n = [max(r.get("n_frames") or 0, 0) for r in rs]
    if sum(n) == 0:
        return None, None, None
    a = sum((r.get(f"lr@{thr[0]:g}") or 0) * k for r, k in zip(rs, n)) / sum(n)
    b = sum((r.get(f"lr@{thr[1]:g}") or 0) * k for r, k in zip(rs, n)) / sum(n)
    ne = [(r.get("est_frac") or 0) * k for r, k in zip(rs, n)]
    num = sum((r.get("ms_ate") or 0) ** 2 * e for r, e in zip(rs, ne) if r.get("ms_ate") is not None)
    ms = (num / sum(ne)) ** 0.5 if sum(ne) > 0 else None
    return a, b, ms


def dataset_section(ds_cfg):
    """Markdown overview of the prepared sequences (benchmark/results/datasets.json, written by dataset_stats.py)."""
    f = ROOT / "benchmark/results/datasets.json"
    if not f.is_file():
        return ""
    st = json.loads(f.read_text())
    names = {"kitti": "KITTI odometry", "openloris": "OpenLORIS-Scene", "rover": "ROVER campus_large", "simchange": "SimChange v2"}
    out = ["## Datasets", "",
           "A *session* is one recorded traversal. In each scene, the map session builds the map; every other session is a query "
           "session, run against that map once as a whole (T2) and as independent 10 s trials (T3). All sessions run at 10 Hz. "
           "Frame counts are those of the prepared sequences; the setups of one session differ by a few frames at most.", "",
           "| dataset | scene | sessions (map + queries) | map session | query sessions | T3 trials |",
           "|---|---|---|---|---|---|"]
    details = []
    for d in ("kitti", "openloris", "rover", "simchange"):
        for scene in ds_cfg.get(d, {}).get("scenes", {}):
            rows = st.get(d, {}).get(scene)
            sc = ds_cfg[d]["scenes"][scene]
            nq = len(sc.get("queries", []))
            if not rows:
                out.append(f"| {names[d]} | {scene} | 1 + {nq} | not prepared yet | | |")
                continue
            def fr(r):
                v = min(x["frames"] for x in r["setups"].values())
                p = max(x["path_m"] for x in r["setups"].values())
                return v, p
            m = rows.get(sc["map"])
            mtxt = (lambda v, p: f"{v} frames, {v / 600:.1f} min, {p:.0f} m")(*fr(m)) if m else "not prepared yet"
            qs = [rows[q] for q in sc.get("queries", []) if q in rows]
            if qs:
                fq = [fr(r)[0] for r in qs]
                qtxt = f"{len(qs)}/{nq} ready: {min(fq)}–{max(fq)} frames ({sum(fq) / 600:.1f} min in total)"
            else:
                qtxt = "–" if nq == 0 else f"0/{nq} ready"
            trials = sum(r.get("trials", 0) for r in qs)
            out.append(f"| {names[d]} | {scene} | 1 + {nq} | {mtxt} | {qtxt} | {trials if nq else '–'} |")
            if nq == 0:
                continue
            def cov(r):
                c = [x.get("covered") for x in r["setups"].values() if x.get("covered") is not None]
                return f"{100 * min(c):.0f} %" if c else "–"
            det = [f"| {n} | {r['role']} | {fr(r)[0]} | {fr(r)[0] / 10:.0f} s | {fr(r)[1]:.0f} m | {r.get('trials', '–')} | {cov(r)} |"
                   for n, r in rows.items()]
            details += ["", f"<details><summary>{names[d]} · {scene}: sessions</summary>", "",
                        "| session | role | frames | duration | path | T3 trials | covered by the map |", "|---|---|---|---|---|---|---|"] + det + ["", "</details>"]
    return "\n".join(out + details)


_COVERAGE = None


def uncovered(dataset, query, setup_dir):
    """Uncovered frame intervals of a query session (benchmark/results/datasets.json, from dataset_stats.py)."""
    global _COVERAGE
    if _COVERAGE is None:
        f = ROOT / "benchmark/results/datasets.json"
        _COVERAGE = json.loads(f.read_text()) if f.is_file() else {}
    for scene in _COVERAGE.get(dataset, {}).values():
        r = scene.get(query)
        if r and setup_dir in r.get("setups", {}):
            return r["setups"][setup_dir].get("uncovered")
    return None


def rescore_t3(results, ds_cfg):
    """T3 success at the dataset's two fixed position thresholds (1 m / 2 m indoors, 3 m / 5 m outdoors), from the stored
    final error of every trial: n_s1 / n_s2 successes, rs1 / rs2 rates."""
    out = []
    for r in results:
        if r.get("track") != "t3" or r.get("status") != "ok" or not r.get("trials") or "rs1" in r:
            out.append(r)          # (already re-scored: keep the same object)
            continue
        t1, t2 = ds_cfg[r["dataset"]]["thresholds"]
        gaps = uncovered(r["dataset"], r["query"], ds_cfg[r["dataset"]]["setups"].get(r["setup"], ""))
        trials = r["trials"]
        if gaps:          # only trials whose last frame the map session passed near (T3 in PROTOCOL.md)
            def last(t):
                return int(t.get("start") or 0) + int(ds_cfg[r["dataset"]]["trial_len"]) - 1
            trials = [t for t in trials if not any(a <= last(t) <= b for a, b in gaps)]
        e = [t.get("final_err") for t in trials]
        n = len(e)
        r = dict(r)
        r["n_trials_all"] = len(r["trials"])
        r["n_s1"] = sum(1 for x in e if x is not None and x < t1)
        r["n_s2"] = sum(1 for x in e if x is not None and x < t2)
        r["rs1"], r["rs2"] = (r["n_s1"] / n, r["n_s2"] / n) if n else (None, None)
        r["thresholds"] = [t1, t2]
        r["n_trials"] = n
        out.append(r)
    return out


class Tables:
    def __init__(self, results, seed):
        self.ds, self.sy = load()
        self.idx = defaultdict(list)
        for r in rescore_t3(results, self.ds):
            if r.get("seed", 0) != seed:
                continue
            self.idx[(r["track"], r["dataset"], r["system"], r["setup"])].append(r)

    def t1_cells(self, dataset, system, setup):
        return {r["sequence"]: r for r in self.idx[("t1", dataset, system, setup)]}

    def t1_seq_str(self, r):
        if r is None:
            return PENDING
        if r.get("status") != "ok" or r.get("ate_rmse") is None:
            return FAIL
        s = f"{r['ate_rmse']:.3f}"
        if r.get("failed"):
            return f"{FAIL} ({100 * r.get('completeness', 0):.0f}%)"
        if r.get("completeness", 1) < 0.995:
            s += f" ({100 * r['completeness']:.0f}%)"
        return s

    def t1_agg(self, cells, seqs):
        """Mean ATE over the sequences; None if any sequence is missing; counts failures."""
        vals, fails, missing = [], 0, 0
        for q in seqs:
            r = cells.get(q)
            if r is None:
                missing += 1
            elif r.get("status") != "ok" or r.get("ate_rmse") is None or r.get("failed"):
                fails += 1
            else:
                vals.append(r["ate_rmse"])
        return vals, fails, missing

    def t1_agg_str(self, vals, fails, missing, n):
        if missing == n:
            return PENDING
        s = f"{sum(vals) / len(vals):.3f}" if vals else FAIL
        extra = []
        if fails:
            extra.append(f"{fails}{FAIL}")
        if missing:
            extra.append(f"{missing} pending")
        return s + (f" ({', '.join(extra)})" if extra else "")

    def scene_seqs(self, dataset, scene):
        sc = self.ds[dataset]["scenes"][scene]
        return [sc["map"]] + list(sc.get("queries", []))

    # ---------------------------------------------------------------- T1
    def t1_table(self, dataset):
        scenes = list(self.ds[dataset]["scenes"])
        per_seq = dataset == "kitti"
        cols = scenes if per_seq else scenes
        head = ["system · setup"] + ([f"{c}" for c in cols]) + ["mean"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for system, setup in rows_for(self.ds, self.sy, dataset):
            cells = self.t1_cells(dataset, system, setup)
            row = [row_label(self.sy, system, setup, dataset)]
            all_vals, all_f, all_m, all_n = [], 0, 0, 0
            for sc in scenes:
                seqs = self.scene_seqs(dataset, sc)
                v, f, m = self.t1_agg(cells, seqs)
                all_vals += v
                all_f += f
                all_m += m
                all_n += len(seqs)
                row.append(self.t1_seq_str(cells.get(seqs[0])) if len(seqs) == 1 else self.t1_agg_str(v, f, m, len(seqs)))
            row.append(self.t1_agg_str(all_vals, all_f, all_m, all_n))
            if self.sy[system]["runner"] == "pending":
                row = [row[0]] + ["in development"] + [NA] * (len(row) - 2)
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    # ---------------------------------------------------------------- T2
    def t2_table(self, dataset):
        cfg = self.ds[dataset]
        thr = cfg["thresholds"]
        scenes = [s for s in cfg["scenes"] if cfg["scenes"][s].get("queries")]
        head = ["system · setup"] + [f"{s}" for s in scenes] + ["all queries"]
        lines = [f"Cells: LR@{thr[0]:g} m / LR@{thr[1]:g} m and MS-ATE (m), pooled over the covered frames of the scene's query sessions (frames within the larger threshold of the map session's path). "
                 "LR@x = fraction of query frames whose latest pose (at most 1 s old), expressed in the map frame, is within x m of "
                 "the ground truth; frames without such a pose count as failures. MS-ATE = RMSE over the frames that have a pose.", "",
                 "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for system, setup in rows_for(self.ds, self.sy, dataset):
            cells = {(r["map"], r["query"]): r for r in self.idx[("t2", dataset, system, setup)]}
            row = [row_label(self.sy, system, setup, dataset)]
            if self.sy[system]["runner"] == "pending":
                lines.append("| " + " | ".join([row[0], "in development"] + [NA] * len(scenes)) + " |")
                continue
            allr = []
            for s in scenes:
                sc = cfg["scenes"][s]
                rs = [cells.get((sc["map"], q)) for q in sc["queries"]]
                row.append(self.t2_str([r for r in rs if r is not None], len(rs), thr))
                allr += rs
            row.append(self.t2_str([r for r in allr if r is not None], len(allr), thr))
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def t2_str(self, rs, n, thr):
        if not rs:
            return PENDING
        ok = [r for r in rs if r.get("status") == "ok"]
        if not ok:
            return FAIL
        a, b, ms = t2_pool(ok, thr)
        if a is None:
            return PENDING
        s = f"{a:.2f} / {b:.2f}, {fmt(ms, 2)} m"
        miss = n - len([r for r in ok if r.get(f"lr@{thr[1]:g}") is not None])
        return s + (f" ({miss} pending/failed)" if miss else "")

    # ---------------------------------------------------------------- T3
    def t3_table(self, dataset):
        cfg = self.ds[dataset]
        scenes = [s for s in cfg["scenes"] if cfg["scenes"][s].get("queries")] or \
            sorted({r["scene"] for k, v in self.idx.items() if k[0] == "t3" and k[1] == dataset for r in v})
        t1, t2 = cfg["thresholds"]
        head = ["system · setup"] + scenes + [f"all queries [trials, 95 % CI of RS@{t2:g} m]"]
        cal = sorted({self.sy[k[2]]["label"] for k, v in self.idx.items() if k[0] == "t3" and k[1] == dataset
                      for r in v if r.get("calibrated")})
        note = (f" Rows marked *calibrated* ({', '.join(cal)}) use the noise model calibrated without ground truth on the "
                "first 600 frames of the map traversal." if cal else "")
        lines = [f"Cells: RS@{t1:g} m / RS@{t2:g} m, the fraction of trials whose final pose lies within {t1:g} m / {t2:g} m of the "
                 f"pose the map implies, pooled over the scene's trials whose last frame the map covers.{note}", "",
                 "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for system, setup in rows_for(self.ds, self.sy, dataset):
            rs_all = self.idx[("t3", dataset, system, setup)]
            row = [row_label(self.sy, system, setup, dataset) + (" *calibrated*" if any(r.get("calibrated") for r in rs_all) else "")]
            if self.sy[system]["runner"] == "pending" and not rs_all:
                lines.append("| " + " | ".join([row[0], "in development"] + [NA] * len(scenes)) + " |")
                continue
            for s in scenes:
                row.append(self.t3_str([r for r in rs_all if r["scene"] == s]))
            row.append(self.t3_str(rs_all, ci=True))
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def t3_str(self, rs, ci=False):
        ok = [r for r in rs if r.get("status") == "ok" and r.get("n_trials")]
        if not rs:
            return PENDING
        if not ok:
            return FAIL
        n = sum(r["n_trials"] for r in ok)
        k1 = sum(r["n_s1"] for r in ok)
        k2 = sum(r["n_s2"] for r in ok)
        s = f"{k1 / n:.2f} / {k2 / n:.2f}"
        if ci:
            lo, hi = wilson(k2, n)
            s += f" [{n}, {lo:.2f}–{hi:.2f}]"
        return s

    # ---------------------------------------------------------------- summary
    def summary(self):
        cols = [("KITTI ATE (m)", "t1", "kitti"), ("OpenLORIS ATE (m)", "t1", "openloris"), ("ROVER ATE (m)", "t1", "rover"),
                ("OpenLORIS LR@1/2 m", "t2", "openloris"), ("ROVER LR@3/5 m", "t2", "rover"), ("SimChange LR@1/2 m", "t2", "simchange"),
                ("OpenLORIS RS@1/2 m", "t3", "openloris"), ("ROVER RS@3/5 m", "t3", "rover"), ("SimChange RS@1/2 m", "t3", "simchange")]
        systems = []
        for system, sc in self.sy.items():
            if sc.get("hidden"):
                continue
            for setup in sc["setups"]:
                systems.append((system, setup))
        head = ["system · setup"] + [c[0] for c in cols]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for system, setup in systems:
            row = [row_label(self.sy, system, setup)]
            for _, track, dataset in cols:
                cfg = self.ds.get(dataset, {})
                if setup not in cfg.get("setups", {"rgbd": 1, "stereo": 1, "mono": 1}) or \
                        f"{dataset}/{setup}" in self.sy[system].get("skip", []):
                    row.append(NA)
                    continue
                rs = self.idx[(track, dataset, system, setup)]
                if track == "t1":
                    cells = {r["sequence"]: r for r in rs}
                    seqs = [q for s in cfg["scenes"] for q in self.scene_seqs(dataset, s)]
                    v, f, m = self.t1_agg(cells, seqs)
                    row.append(self.t1_agg_str(v, f, m, len(seqs)) if rs else PENDING)
                elif track == "t2":
                    ok = [r for r in rs if r.get("status") == "ok"]
                    a_, b_, _ = t2_pool(ok, cfg["thresholds"]) if ok else (None, None, None)
                    row.append(f"{a_:.2f} / {b_:.2f}" if a_ is not None else PENDING)
                else:
                    ok = [r for r in rs if r.get("status") == "ok" and r.get("n_trials")]
                    if ok:
                        n = sum(r["n_trials"] for r in ok)
                        row.append(f"{sum(r['n_s1'] for r in ok) / n:.2f} / {sum(r['n_s2'] for r in ok) / n:.2f}")
                    else:
                        row.append(PENDING)
            if self.sy[system]["runner"] == "pending" and all(c in (PENDING, NA) for c in row[1:]):
                row = [row[0]] + ["in development" if c == PENDING else c for c in row[1:]]
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "benchmark/results/results.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "benchmark/RESULTS.md"))
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text())["results"] if Path(a.results).is_file() else []
    T = Tables(res, a.seed)
    legacy = ROOT / "benchmark/results/legacy.md"
    parts = [
        "# CROSS benchmark results",
        "",
        f"_Generated by `benchmark/make_tables.py` from `benchmark/results/results.json` ({len(res)} runs, seed {a.seed}, "
        f"{time.strftime('%Y-%m-%d')}). Protocol: [PROTOCOL.md](PROTOCOL.md). Per-run details, trajectories and failure cases: "
        "[results page](site/index.html)._",
        "",
        f"Legend: `{PENDING}` not run yet, `{FAIL}` failed (crash, timeout or tracking completeness < 80 %), (xx%) completeness, "
        "⁽ᵒ⁾ the system uses the dataset's odometry (wheel odometry on OpenLORIS, OXTS dead reckoning on KITTI, "
        "simulated on ROVER and SimChange), RGB-D* = left image + stereo-matched depth (KITTI).",
        "",
        dataset_section(T.ds),
        "",
        "## Summary",
        "",
        T.summary(),
        "",
        "## T1 — mapping accuracy (ATE RMSE, m)",
        "",
        "Final trajectory after all loop closures, SE(3) alignment (Sim(3) for monocular systems without metric input). "
        "OpenLORIS and ROVER cells: mean over the scene's sequences.",
    ]
    for d, title in (("kitti", "KITTI odometry (outdoor)"), ("openloris", "OpenLORIS-Scene (indoor)"), ("rover", "ROVER campus_large (outdoor)")):
        parts += ["", f"### {title}", "", T.t1_table(d)]
    parts += ["", "## T2 — multi-session localization", "",
              "The query session runs once from its first frame against the stored map of the scene's map session."]
    for d, title in (("openloris", "OpenLORIS-Scene"), ("rover", "ROVER campus_large")):
        parts += ["", f"### {title}", "", T.t2_table(d)]
    parts += ["", "## T3 — relocalization success", "",
              "Independent 10 s trials (100 frames at 10 Hz, stride 50) that start without a pose; success when the final "
              "estimate is within 1 m / 2 m (indoors) or 3 m / 5 m (outdoors) of the pose the map implies."]
    for d, title in (("openloris", "OpenLORIS-Scene"), ("rover", "ROVER campus_large"), ("simchange", "SimChange v2")):
        parts += ["", f"### {title}", "", T.t3_table(d)]
    if legacy.is_file():
        parts += ["", legacy.read_text().strip()]
    Path(a.out).write_text("\n".join(parts) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
