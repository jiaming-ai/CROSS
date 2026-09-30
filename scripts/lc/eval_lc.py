#!/usr/bin/env python3
"""Evaluate loop-closure runs recorded with scripts/viz/record_trace.py (trace.json + graph_s*.json + system.log).

Per session:
  * map ATE (mapping session; Umeyama SE(3) alignment of the final keyframe poses to ground truth) and, for
    relocalization sessions, the map-relative position error of hypothesis 0 (mean / median / fraction > 2 m / final);
  * loop-closure edges accepted into hypothesis 0 (final graph): long-range edges (same session, >= --loop-gap
    steps apart) and session->map edges, labelled with ground truth (false = translation error > 1 m or rotation
    error > 10 deg): precision = 1 - false fraction;
  * candidate recall: retrieved references of the feed-forward pass (`ffp`, covisibility-valid) whose true relative
    pose was correct and that were long-range / session->map, versus the ones that ended up as hypothesis-0 edges;
  * loop-closure events (merges, PGOs) and their effect on the error (false: error grows by > 2 m; corrective:
    shrinks by > 2 m), verifier statistics and PGO count / time from the log.

usage: python scripts/lc/eval_lc.py outputs/lcstudy/<scene>/<tag> [...] [--out summary.json] [--md summary.md]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))


def mat(p):
    T = np.eye(4); q = np.asarray(p[3:7], float); q /= max(np.linalg.norm(q), 1e-12)
    T[:3, :3] = R.from_quat(q).as_matrix(); T[:3, 3] = p[:3]; return T


def inv(T):
    o = np.eye(4); o[:3, :3] = T[:3, :3].T; o[:3, 3] = -T[:3, :3].T @ T[:3, 3]; return o


def rot_deg(Rm):
    U, _, Vt = np.linalg.svd(Rm); d = np.sign(np.linalg.det(U @ Vt)); Rm = U @ np.diag([1, 1, d]) @ Vt
    return float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))


def umeyama_ate(src, dst):
    src, dst = np.asarray(src), np.asarray(dst)
    if len(src) < 3:
        return float("nan")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T; t = mu_d - Rm @ mu_s
    return float(np.sqrt(np.mean(np.sum(((Rm @ src.T).T + t - dst) ** 2, axis=1))))


def session_errors(d, s):
    """Mapping session: position error vs GT after first-keyframe alignment; reloc: map-relative error."""
    st = [e for e in d["steps"] if e["s"] == s]
    kg = d["kf_gt"]; fn = d["map_final_nodes"]
    ids = [k for k in fn if k in kg]
    G = np.array([mat(kg[k]) for k in ids]); M = np.array([mat(fn[k]) for k in ids])
    Tf = np.array(d["T_gt_from_map_first"])
    out = []
    for e in st:
        g = mat(e["gt"]); m = mat(e["mu"][0])
        if s == 0:
            out.append(np.linalg.norm((Tf @ m)[:3, 3] - g[:3, 3]))
        else:
            j = int(np.argmin(np.linalg.norm(G[:, :3, 3] - g[:3, 3], axis=1)))
            pred = M[j] @ inv(G[j]) @ g
            out.append(np.linalg.norm((inv(pred) @ m)[:3, 3]))
    return np.array(out), st


REL_TOL = 0.05   # tolerated translation error per metre of edge distance (metric scale of a long measurement)


def eval_graph(gpath, loop_gap=200, false_t=1.0, false_r=10.0):
    g = json.loads(Path(gpath).read_text())
    sess = g["meta"].get("session", 0)
    nodes = {n["id"]: n for n in g["nodes"]}
    gt = {n["id"]: np.asarray(n["gt"], float).reshape(4, 4) for n in g["nodes"] if n.get("gt") is not None}
    step = {n["id"]: n.get("step_created") for n in g["nodes"]}
    nsess = {n["id"]: n.get("session", 0) for n in g["nodes"]}
    # ATE of the session's keyframes (mapping: all; reloc: the session keyframes are not evaluated here)
    ids = [i for i in nodes if i in gt and nsess[i] == sess]
    ate = umeyama_ate([mat(nodes[i]["pose"])[:3, 3] for i in ids], [gt[i][:3, 3] for i in ids]) if sess == 0 else None
    res = {"session": sess, "n_nodes": len(nodes), "map_ate": ate, "edges": {}}
    for kind in ("loop", "cross"):
        n, n_false, errs = 0, 0, []
        rej_true, rej_false = 0, 0
        n_inf, n_false_inf = 0, 0            # edges that constrain the graph (information criterion)
        for e in g["visual"]:
            if not (e["fc"] == 0 and e["tc"] == 0 and e["hyp"] == 0):
                continue
            a, b = e["a"], e["b"]
            if a not in gt or b not in gt:
                continue
            same = nsess[a] == nsess[b]
            if kind == "loop":
                if not same or step.get(a) is None or step.get(b) is None or abs(step[b] - step[a]) < loop_gap:
                    continue
            else:
                if same:
                    continue
            T = mat(e["mean"]); Tg = inv(gt[a]) @ gt[b]
            E = inv(T) @ Tg
            et, er = float(np.linalg.norm(E[:3, 3])), rot_deg(E[:3, :3])
            # false: wrong place, not a metric-scale error of a long measurement (5 % of the edge distance is allowed)
            false = int(et > false_t + REL_TOL * float(np.linalg.norm(Tg[:3, 3])) or er > false_r)
            if e.get("rejected"):
                rej_false += false; rej_true += 1 - false
                continue
            n += 1; n_false += false; errs.append(et)
            if e.get("informative") is not False:
                n_inf += 1; n_false_inf += false
        res["edges"][kind] = {"n": n, "n_false": n_false, "precision": (1 - n_false / n) if n else None,
                              "n_informative": n_inf, "n_false_informative": n_false_inf,
                              "rejected_true": rej_true, "rejected_false": rej_false,
                              "e_t_p50": float(np.median(errs)) if errs else None, "e_t_p99": float(np.percentile(errs, 99)) if errs else None}
    res["accepted_keys"] = {(e["a"], e["b"]) for e in g["visual"] if e["fc"] == 0 and e["tc"] == 0 and e["hyp"] == 0 and not e.get("rejected")}
    return res, gt, step, nsess


def candidate_recall(trace, s, gres, gt, step, nsess, loop_gap=200, false_t=1.0):
    """True long-range candidates of the feed-forward passes of session s vs the accepted hypothesis-0 edges."""
    steps = [e for e in trace["steps"] if e["s"] == s and e.get("ffp")]
    gt_steps = {}
    for e in trace["steps"]:
        if e.get("nk") is not None:
            gt_steps[(e["s"], int(e["nk"]))] = mat(e["gt"])
    map_ids = {int(k) for k in trace["nodes"] if str(k).isdigit()}
    n_true, n_acc, n_false_cand = 0, 0, 0
    for e in steps:
        f = e["ffp"]; G_cur = mat(e["gt"]); c2w = [mat(p) for p in f["c2w"]]
        nk = e.get("nk")
        for k, kid in enumerate(f["ids"]):
            if not f["valid"][k]:
                continue
            key = (0, kid) if (s > 0 and kid in map_ids and (s, kid) not in gt_steps) else (s, kid)
            if key not in gt_steps:
                continue
            cross = key[0] != s
            if not cross:
                sk = trace["nodes"].get(str(kid) if s == 0 else f"{s}:{kid}", {}).get("step")
                if sk is None or e["i"] + (trace["sessions"][s]["start_step"]) - sk < loop_gap:
                    continue
            T_pred = inv(c2w[1 + k]) @ c2w[0]; T_gt = inv(gt_steps[key]) @ G_cur
            err = float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]))
            if err > false_t + REL_TOL * float(np.linalg.norm(T_gt[:3, 3])):
                n_false_cand += 1; continue
            n_true += 1
            if nk is not None and (kid, int(nk)) in gres["accepted_keys"]:
                n_acc += 1
    return {"n_true_candidates": n_true, "n_accepted_true": n_acc, "recall": (n_acc / n_true) if n_true else None, "n_false_candidates": n_false_cand}


def eval_run(run_dir, loop_gap=200):
    run_dir = Path(run_dir)
    trace = json.loads((run_dir / "trace.json").read_text())
    log = (run_dir / "system.log").read_text() if (run_dir / "system.log").exists() else ""
    out = {"run": str(run_dir), "map_ate_rmse": trace.get("map_ate_rmse"), "sessions": []}
    for sess in trace["sessions"]:
        s = sess["id"]
        gpath = run_dir / f"graph_s{s}.json"
        rec = {"id": s, "variant": sess["variant"], "kind": sess["kind"], "lc_stats": sess.get("lc_stats")}
        if gpath.exists():
            gres, gt, step, nsess = eval_graph(gpath, loop_gap=loop_gap)
            rec.update({k: v for k, v in gres.items() if k != "accepted_keys"})
            rec["candidates"] = candidate_recall(trace, s, gres, gt, step, nsess, loop_gap=loop_gap)
        err, st = session_errors(trace, s)
        ev = []
        tot = {"false": 0, "fix": 0, "neutral": 0}
        for e in st:
            if "lc" not in e:
                continue
            i = e["i"]; b = np.median(err[max(0, i - 3):i]) if i > 0 else err[0]; a = np.median(err[i + 3:i + 6]) if i + 6 <= len(err) else err[-1]
            kind = "false" if a - b > 2 else ("fix" if b - a > 2 else "neutral"); tot[kind] += 1
            ev.append({"step": i, "hyp": e["lc"], "before": float(b), "after": float(a), "kind": kind})
        rec["error"] = {"mean": float(err.mean()), "median": float(np.median(err)), "frac_gt_2m": float(np.mean(err > 2)), "final": float(err[-1]), "max": float(err.max())}
        rec["events"] = {"n": len(ev), **tot, "list": ev[:40]}
        dts = [e["dt"] for e in st]
        rec["fps"] = float(len(st) / max(sum(dts), 1e-9))
        out["sessions"].append(rec)
    out["log"] = {"pgo_verified": len(re.findall(r"Verified loop closure at step", log)),
                  "pgo_intra": len(re.findall(r"Intra-hypothesis loop closure at", log)),
                  "merges": len(re.findall(r"LC detected", log)),
                  "merge_rejected": len(re.findall(r"rejected:", log)),
                  "posterior_rejected_edges": sum(int(m) for m in re.findall(r"Verified loop closure at step \d+: (\d+) of \d+ new edges rejected", log))}
    return out


def md_table(results):
    lines = ["| run | session | map ATE | err mean / med / >2m / final | loop edges (n, false, prec) | cross edges (n, false, prec) | constraining false/all (loop, cross) | rejected true/false (loop, cross) | cand. recall | events false/fix/neutral | PGO | FPS |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for s in r["sessions"]:
            e = s["edges"] if "edges" in s else {}
            lo, cr = e.get("loop", {}), e.get("cross", {})
            c = s.get("candidates", {})
            er = s["error"]; ev = s["events"]
            f = lambda x, nd=2: "-" if x is None else f"{x:.{nd}f}"
            lines.append(f"| {Path(r['run']).parent.name}/{Path(r['run']).name} | s{s['id']} {s['variant'][:28]} | {f(s.get('map_ate'))} | {er['mean']:.2f} / {er['median']:.2f} / {er['frac_gt_2m']:.2f} / {er['final']:.2f} | "
                         f"{lo.get('n', '-')}, {lo.get('n_false', '-')}, {f(lo.get('precision'), 3)} | {cr.get('n', '-')}, {cr.get('n_false', '-')}, {f(cr.get('precision'), 3)} | "
                         f"{lo.get('n_false_informative', '-')}/{lo.get('n_informative', '-')}, {cr.get('n_false_informative', '-')}/{cr.get('n_informative', '-')} | "
                         f"{lo.get('rejected_true', '-')}/{lo.get('rejected_false', '-')}, {cr.get('rejected_true', '-')}/{cr.get('rejected_false', '-')} | {f(c.get('recall'), 3)} ({c.get('n_true_candidates', '-')}) | "
                         f"{ev['false']}/{ev['fix']}/{ev['neutral']} | {r['log']['pgo_verified'] + r['log']['pgo_intra']} | {s['fps']:.1f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--loop-gap", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    results = []
    for r in args.runs:
        try:
            results.append(eval_run(r, loop_gap=args.loop_gap))
        except Exception as ex:
            print(f"!! {r}: {ex}")
    table = md_table(results)
    print(table)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=float))
    if args.md:
        Path(args.md).write_text(table + "\n")


if __name__ == "__main__":
    main()
