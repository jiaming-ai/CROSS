#!/usr/bin/env python3
"""Write benchmark/RESULTS.md from benchmark/results/results.json (generated file: do not edit RESULTS.md by hand).

  python benchmark/make_tables.py [--results benchmark/results/results.json] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SETUP_LABEL = {"rgbd": "RGB-D", "stereo": "stereo", "mono": "mono"}
PENDING, NA, FAIL = "·", "", "✗"


def load():
    ds = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())
    ds = {k: v for k, v in ds.items() if not v.get("dev")}       # the development split is not part of the tables
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
        tl = int(ds_cfg[r["dataset"]]["trial_len"])
        annotated = [{**t, "covered": not any(a <= int(t.get("start") or 0) + tl - 1 <= b for a, b in (gaps or []))}
                     for t in r["trials"]]           # only trials whose last frame the map covers count (PROTOCOL.md, T3)
        trials = [t for t in annotated if t["covered"]]
        e = [t.get("final_err") for t in trials]
        n = len(e)
        r = dict(r)
        r["n_trials_all"] = len(r["trials"])
        r["trials"] = annotated
        r["n_s1"] = sum(1 for x in e if x is not None and x < t1)
        r["n_s2"] = sum(1 for x in e if x is not None and x < t2)
        r["rs1"], r["rs2"] = (r["n_s1"] / n, r["n_s2"] / n) if n else (None, None)
        r["thresholds"] = [t1, t2]
        r["n_trials"] = n
        out.append(r)
    return out


def query_stats(dataset, query, setup_dir):
    """(frames, covered fraction, uncovered intervals) of a prepared query session, from benchmark/results/datasets.json."""
    uncovered(dataset, query, setup_dir)            # loads _COVERAGE
    for scene in _COVERAGE.get(dataset, {}).values():
        r = scene.get(query)
        if r and setup_dir in r.get("setups", {}):
            st = r["setups"][setup_dir]
            return st["frames"], st.get("covered", 1.0), st.get("uncovered") or []
    return None


def count_failed_queries(results, ds_cfg, sy):
    """A query session whose run failed (crash or timeout after re-runs, or a failed map) counts as a failure of every
    covered frame (T2) and of every covered trial (T3): it adds its frames / trials to the pools with no success.  Systems
    without map persistence are scored on at most --concat-max-trials evenly spaced trials, so their failed queries add
    the same selection.  Failed runs of sessions without dataset statistics stay out of the pools."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    from reloc_metrics import build_trials
    out = []
    for r in results:
        if r.get("track") not in ("t2", "t3") or r.get("status") == "ok" or "query" not in r or r.get("counted_failure"):
            out.append(r)
            continue
        cfg = ds_cfg[r["dataset"]]
        st = query_stats(r["dataset"], r["query"], cfg["setups"].get(r["setup"], ""))
        if st is None:
            out.append(r)
            continue
        frames, cov, gaps = st
        r = dict(r, counted_failure=True)
        if r["track"] == "t2":
            r.update({f"lr@{x:g}": 0.0 for x in LR_GRID}, n_frames=int(round(frames * cov)), est_frac=0.0, ms_ate=None)
        else:
            tl = int(cfg["trial_len"])
            trials = build_trials(frames, tl, int(cfg["trial_stride"]))
            if sy[r["system"]]["runner"] == "concat" and len(trials) > CONCAT_MAX_TRIALS:
                trials = [trials[i] for i in np.linspace(0, len(trials) - 1, CONCAT_MAX_TRIALS).round().astype(int)]
            n = sum(1 for a, _ in trials if not any(g0 <= a + tl - 1 <= g1 for g0, g1 in gaps))
            r.update(n_trials=n, n_s1=0, n_s2=0, rs1=0.0, rs2=0.0, trials=[])
        out.append(r)
    return out


LR_GRID = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)      # as benchmark/run.py
CONCAT_MAX_TRIALS = 5                                 # benchmark/run.py --concat-max-trials


def scored(r):
    """Counts in the pooled T2 / T3 rates: a finished run, or a failed one counted as failure (count_failed_queries)."""
    return r.get("status") == "ok" or bool(r.get("counted_failure"))


class Tables:
    def __init__(self, results, seed):
        self.ds, self.sy = load()
        f = ROOT / "benchmark/results/datasets.json"
        self.dstats = json.loads(f.read_text()) if f.is_file() else {}
        self.idx = defaultdict(list)
        for r in count_failed_queries(rescore_t3(results, self.ds), self.ds, self.sy):
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

    def path_m(self, dataset, sequence, setup):
        """Ground-truth path length (m) of a sequence, from benchmark/results/datasets.json."""
        for sc in self.dstats.get(dataset, {}).values():
            st = sc.get(sequence, {}).get("setups", {})
            for k in (self.ds[dataset]["setups"].get(setup), setup, *st):
                if k in st and st[k].get("path_m"):
                    return st[k]["path_m"]
        return None

    def t1_agg_str(self, vals, fails, missing, n):
        if missing == n:
            return PENDING
        s = f"{sum(vals) / len(vals):.3f}" if vals else FAIL
        return cell(s, fails, n, missing)

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
                 "the ground truth; frames without such a pose count as failures, as do all covered frames of a failed query session (k/N ✗). MS-ATE = RMSE over the frames that have a pose.", "",
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

    def t2_str(self, rs, n, thr, ms_ate=True):
        if not rs:
            return PENDING
        ok = [r for r in rs if scored(r) and all(r.get(f"lr@{t:g}") is not None for t in thr)]
        a, b, ms = t2_pool(ok, thr) if ok else (None, None, None)
        s = (f"{a:.2f} / {b:.2f}" + (f", {fmt(ms, 2)} m" if ms_ate else "")) if a is not None else \
            FAIL if all(r.get("status") != "ok" for r in rs) else PENDING
        failed = len([r for r in rs if r.get("status") != "ok"])
        stale = len([r for r in rs if scored(r)]) - len(ok)        # results of an older format: re-score
        return cell(s, failed, n, n - len(rs) + stale)

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
                 f"pose the map implies, pooled over the scene's trials whose last frame the map covers. (k/N ✗): k of the N query "
                 f"sessions failed; their trials count as failures.{note}", "",
                 "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for system, setup in rows_for(self.ds, self.sy, dataset):
            rs_all = self.idx[("t3", dataset, system, setup)]
            row = [row_label(self.sy, system, setup, dataset) + (" *calibrated*" if any(r.get("calibrated") for r in rs_all) else "")]
            if self.sy[system]["runner"] == "pending" and not rs_all:
                lines.append("| " + " | ".join([row[0], "in development"] + [NA] * len(scenes)) + " |")
                continue
            for s in scenes:
                row.append(self.t3_str([r for r in rs_all if r["scene"] == s], self.n_queries(dataset, s)))
            row.append(self.t3_str(rs_all, self.n_queries(dataset), ci=True))
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def t3_str(self, rs, n_queries=None, ci=False):
        """RS@t1 / RS@t2 pooled over the trials of the query runs (a failed run's trials count as failures), then
        (k/N ✗) for k failed of N queries."""
        ok = [r for r in rs if scored(r) and r.get("n_trials")]
        if not rs:
            return PENDING
        n_q = n_queries or len(rs)
        failed = len([r for r in rs if r.get("status") != "ok"])
        note = fail_note(failed, n_q, max(n_q - len(rs), 0))
        if not ok:
            return cell(FAIL, failed, n_q, max(n_q - len(rs), 0))
        n = sum(r["n_trials"] for r in ok)
        k1 = sum(r["n_s1"] for r in ok)
        k2 = sum(r["n_s2"] for r in ok)
        s = f"{k1 / n:.2f} / {k2 / n:.2f}"
        if ci:
            lo, hi = wilson(k2, n)
            s += f" [{n}, {lo:.2f}–{hi:.2f}]"
        return s + note

    def t3_rates(self, rs):
        ok = [r for r in rs if scored(r) and r.get("n_trials")]
        n = sum(r["n_trials"] for r in ok)
        return (sum(r["n_s1"] for r in ok) / n, sum(r["n_s2"] for r in ok) / n) if n else None

    def n_queries(self, dataset, scene=None):
        sc = self.ds[dataset]["scenes"]
        return sum(len(v.get("queries", [])) for k, v in sc.items() if scene is None or k == scene)

    def has(self, dataset, system, setup):
        return setup in self.ds[dataset]["setups"] and f"{dataset}/{setup}" not in self.sy[system].get("skip", [])

    def t1_overall(self, system, setup):
        """Overall mapping error: relative ATE (ATE RMSE / ground-truth path length, %) averaged over each T1 dataset's
        finished sequences, then over the datasets (KITTI, OpenLORIS, ROVER), so that each dataset weighs the same
        whatever its scale; with the failed and pending sequences of all of them."""
        rates, failed, pending, n = [], 0, 0, 0
        for d in T1_DATASETS:
            if not self.has(d, system, setup):
                return None
            cells = self.t1_cells(d, system, setup)
            seqs = [q for sc in self.ds[d]["scenes"] for q in self.scene_seqs(d, sc)]
            rel = []
            for q in seqs:
                r = cells.get(q)
                if r is None:
                    pending += 1
                elif r.get("status") != "ok" or r.get("ate_rmse") is None or r.get("failed"):
                    failed += 1
                elif self.path_m(d, q, setup):
                    rel.append(100 * r["ate_rmse"] / self.path_m(d, q, setup))
            n += len(seqs)
            rates.append(mean(rel) if rel else None)
        return rates, failed, pending, n

    def t1_overall_str(self, system, setup):
        o = self.t1_overall(system, setup)
        if o is None:
            return NA
        rates, failed, pending, n = o
        if pending == n:
            return PENDING
        if any(x is None for x in rates):            # a dataset without a value: every sequence failed there, or pending
            return cell(FAIL if failed else PENDING, failed, n, pending)
        return f"{mean(rates):.2f} %" + fail_note(failed, n, pending)

    def t2_overall_str(self, system, setup):
        """Overall localization recall: the mean over the T2 datasets (OpenLORIS, ROVER, SimChange) of the pooled LR at
        each dataset's smaller / larger threshold, with the failed and pending queries of all of them."""
        rates, failed, pending, n, dead = [], 0, 0, 0, False
        for d in T3_DATASETS:
            if not self.has(d, system, setup):
                return NA
            thr = self.ds[d]["thresholds"]
            rs = self.idx[("t2", d, system, setup)]
            nq = self.n_queries(d)
            ok = [r for r in rs if scored(r) and all(r.get(f"lr@{t:g}") is not None for t in thr)]
            n += nq
            failed += len([r for r in rs if r.get("status") != "ok"])
            pending += max(nq - len(rs), 0) + len([r for r in rs if scored(r)]) - len(ok)
            a, b, _ = t2_pool(ok, thr) if ok else (None, None, None)
            rates.append(None if a is None else (a, b))
            dead |= bool(rs) and len(rs) == nq and all(r.get("status") != "ok" for r in rs)
        if pending == n:
            return PENDING
        if any(x is None for x in rates):
            return cell(FAIL if dead else PENDING, failed, n, pending)
        return f"{mean([x[0] for x in rates]):.2f} / {mean([x[1] for x in rates]):.2f}" + fail_note(failed, n, pending)

    def t3_overall(self, system, setup):
        """Overall relocalization success: the mean over the T3 datasets (OpenLORIS, ROVER, SimChange) of the pooled RS at
        each dataset's smaller / larger threshold, with the failed and pending queries of all of them."""
        rates, failed, pending, n, dead = [], 0, 0, 0, False
        for d in T3_DATASETS:
            if setup not in self.ds[d]["setups"] or f"{d}/{setup}" in self.sy[system].get("skip", []):
                return None
            rs = self.idx[("t3", d, system, setup)]
            nq = self.n_queries(d)
            n += nq
            failed += len([r for r in rs if r.get("status") != "ok"])
            pending += max(nq - len(rs), 0)
            rates.append(self.t3_rates(rs))
            dead |= bool(rs) and len(rs) == nq and all(r.get("status") != "ok" for r in rs)
        return rates, failed, pending, n, dead

    def t3_overall_table(self):
        head = ["system · setup"] + [f"{DS_NAMES[d]} RS@{self.ds[d]['thresholds'][0]:g}/{self.ds[d]['thresholds'][1]:g} m"
                                     for d in T3_DATASETS] + ["overall"]
        lines = ["Overall = mean over the three datasets of the pooled success at each dataset's smaller / larger threshold "
                 "(a method missing a dataset has no overall value). (k/N ✗): k of the N query sessions failed (crash or "
                 "timeout after re-runs, or a failed map); every covered trial of a failed session counts as a failure.", "",
                 "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for system, sc in self.sy.items():
            if sc.get("hidden"):
                continue
            for setup in sc["setups"]:
                row = [row_label(self.sy, system, setup)]
                for d in T3_DATASETS:
                    ok = setup in self.ds[d]["setups"] and f"{d}/{setup}" not in sc.get("skip", [])
                    row.append(self.t3_str(self.idx[("t3", d, system, setup)], self.n_queries(d)) if ok else NA)
                row.append(self.t3_overall_str(system, setup))
                lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def t3_overall_str(self, system, setup):
        o = self.t3_overall(system, setup)
        if o is None:
            return NA
        rates, failed, pending, n, dead = o
        if pending == n:
            return PENDING
        if any(x is None for x in rates):            # a dataset without a rate: every query failed there, or pending
            return cell(FAIL if dead else PENDING, failed, n, pending)
        return f"{mean([x[0] for x in rates]):.2f} / {mean([x[1] for x in rates]):.2f}" + fail_note(failed, n, pending)

    # ---------------------------------------------------------------- summary
    SUMMARY = (
        ("ate", "Mapping accuracy (T1): ATE RMSE (m), lower is better",
         [("KITTI (m)", "t1", "kitti"), ("OpenLORIS (m)", "t1", "openloris"), ("ROVER (m)", "t1", "rover"),
          ("overall (% of path)", "t1", "*")],
         "Cells: mean ATE over the dataset's sequences. Overall: ATE / ground-truth path length, averaged over each "
         "dataset's sequences, then over the three datasets (a method missing a dataset has no overall value)."),
        ("lr", "Multi-session localization (T2): localization recall, higher is better",
         [("OpenLORIS LR@1/2 m", "t2", "openloris"), ("ROVER LR@3/5 m", "t2", "rover"),
          ("SimChange LR@1/2 m", "t2", "simchange"), ("overall", "t2", "*")],
         "Overall: mean over the three datasets of the pooled recall at each dataset's smaller / larger threshold."),
        ("rs", "Relocalization (T3): relocalization success, higher is better",
         [("OpenLORIS RS@1/2 m", "t3", "openloris"), ("ROVER RS@3/5 m", "t3", "rover"),
          ("SimChange RS@1/2 m", "t3", "simchange"), ("overall", "t3", "*")],
         "Overall: mean over the three datasets of the pooled success at each dataset's smaller / larger threshold."),
    )

    def summary(self):
        """[(key, title, note, markdown table)] for the three summary tables (ATE, LR, RS)."""
        return [(key, title, note, self.summary_table(cols)) for key, title, cols, note in self.SUMMARY]

    def summary_table(self, cols):
        systems = [(system, setup) for system, sc in self.sy.items() if not sc.get("hidden") for setup in sc["setups"]]
        head = ["system · setup"] + [c[0] for c in cols]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        overall = {"t1": self.t1_overall_str, "t2": self.t2_overall_str, "t3": self.t3_overall_str}
        for system, setup in systems:
            row = [row_label(self.sy, system, setup)]
            for _, track, dataset in cols:
                if dataset == "*":
                    row.append(overall[track](system, setup))
                    continue
                cfg = self.ds[dataset]
                if not self.has(dataset, system, setup):
                    row.append(NA)
                    continue
                rs = self.idx[(track, dataset, system, setup)]
                if track == "t1":
                    cells = {r["sequence"]: r for r in rs}
                    seqs = [q for s in cfg["scenes"] for q in self.scene_seqs(dataset, s)]
                    v, f, m = self.t1_agg(cells, seqs)
                    row.append(self.t1_agg_str(v, f, m, len(seqs)) if rs else PENDING)
                elif track == "t2":
                    row.append(self.t2_str(rs, self.n_queries(dataset), cfg["thresholds"], ms_ate=False))
                else:
                    row.append(self.t3_str(rs, self.n_queries(dataset)))
            if self.sy[system]["runner"] == "pending" and all(c in (PENDING, NA) for c in row[1:]):
                row = [row[0]] + ["in development" if c == PENDING else c for c in row[1:]]
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


def cell(s, failed, n, pending=0):
    """A table cell with its failure count: '0.47 (2/8 ✗)', or '8/8 ✗' when every session or query failed."""
    if s == FAIL and failed:
        return f"{failed}/{n} {FAIL}" + (f" ({pending} pending)" if pending else "")
    return s + fail_note(failed, n, pending)


def fail_note(failed, n, pending=0):
    """' (k/N ✗)' for k failed of N sessions or queries, plus pending ones."""
    parts = ([f"{failed}/{n} {FAIL}"] if failed else []) + ([f"{pending} pending"] if pending else [])
    return f" ({', '.join(parts)})" if parts else ""


T1_DATASETS = ("kitti", "openloris", "rover")
T3_DATASETS = ("openloris", "rover", "simchange")
DS_NAMES = {"kitti": "KITTI", "openloris": "OpenLORIS", "rover": "ROVER", "simchange": "SimChange"}


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
        "\n\n".join(f"### {title}\n\n{note}\n\n{md}" for _, title, note, md in T.summary()),
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
    parts += ["", "### Overall", "", T.t3_overall_table()]
    for d, title in (("openloris", "OpenLORIS-Scene"), ("rover", "ROVER campus_large"), ("simchange", "SimChange v2")):
        parts += ["", f"### {title}", "", T.t3_table(d)]
    if legacy.is_file():
        parts += ["", legacy.read_text().strip()]
    Path(a.out).write_text("\n".join(parts) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
