#!/usr/bin/env python3
"""In-pass consistency of the retrieved references (feed-forward multi-view pass) as a loop-closure verifier.

Every observation of a trace recorded with scripts/viz/record_trace.py stores (`ffp`) the metric camera-to-world
poses predicted by the feed-forward model for the current view and for every retrieved reference in the same pass.
For a pair of references (i, j) the pass predicts a relative pose T_ij; the map knows T_ij as well (from the keyframe
poses, or from ground truth).  A reference that the model registered to the wrong place (perceptual aliasing, bad
registration) is inconsistent with the other references: |T_ij(pass) vs T_ij(map)| is large.  This script measures
that residual for pairs of correct references and for pairs containing a wrong reference (wrong = the edge current->
reference is a ground-truth outlier), with the map relative pose taken from ground truth and from the stored map.

usage: python scripts/lc/inpass_consistency.py trace.json [--out stats.json]
"""
import argparse, json, sys
import numpy as np
from scipy.spatial.transform import Rotation as R


def mat(p):
    T = np.eye(4); q = np.asarray(p[3:7], float); q /= max(np.linalg.norm(q), 1e-12)
    T[:3, :3] = R.from_quat(q).as_matrix(); T[:3, 3] = p[:3]; return T


def inv(T):
    o = np.eye(4); o[:3, :3] = T[:3, :3].T; o[:3, 3] = -T[:3, :3].T @ T[:3, 3]; return o


def rot_deg(Rm):
    return float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))


def load(f):
    t = open(f).read()
    if f.endswith(".js"):
        t = t[t.index("]=") + 2:].rstrip().rstrip(";")
    return json.loads(t)


def analyse(d, false_thr=1.0):
    steps = d["steps"]
    gt = {}
    for st in steps:
        if st.get("nk") is not None:
            gt[(st["s"], int(st["nk"]))] = mat(st["gt"])
    map_ids = {int(k) for k in d["nodes"] if str(k).isdigit()}
    map_pose = {int(k): mat(v["p"]) for k, v in d["nodes"].items() if str(k).isdigit()}   # map keyframe poses (final)
    pairs = []   # (both_true, d_gt, r_gt, d_map, r_map, session, covis_i, covis_j)
    refs_stats = []
    for st in steps:
        f = st.get("ffp")
        if not f:
            continue
        s = st["s"]
        G_cur = mat(st["gt"])
        c2w = [mat(p) for p in f["c2w"]]
        ids = f["ids"]
        info = []
        for k, kid in enumerate(ids):
            key = (0, kid) if (s > 0 and kid in map_ids and (s, kid) not in gt) else (s, kid)
            if key not in gt:
                info.append(None); continue
            T_pred = inv(c2w[1 + k]) @ c2w[0]               # current in ref (pass)
            T_gt = inv(gt[key]) @ G_cur
            err = float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]))
            info.append({"key": key, "err": err, "false": err > false_thr, "covis": f["covis"][k] if k < len(f["covis"]) else None,
                         "valid": f["valid"][k] if k < len(f["valid"]) else None, "map": kid in map_ids and key[0] == 0})
            refs_stats.append((err, info[-1]["covis"], info[-1]["valid"]))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if info[i] is None or info[j] is None:
                    continue
                T_ij_pass = inv(c2w[1 + i]) @ c2w[1 + j]
                T_ij_gt = inv(gt[info[i]["key"]]) @ gt[info[j]["key"]]
                E = inv(T_ij_gt) @ T_ij_pass
                d_gt, r_gt = float(np.linalg.norm(E[:3, 3])), rot_deg(E[:3, :3])
                d_map = r_map = None
                if info[i]["map"] and info[j]["map"] and ids[i] in map_pose and ids[j] in map_pose:
                    T_ij_map = inv(map_pose[ids[i]]) @ map_pose[ids[j]]
                    Em = inv(T_ij_map) @ T_ij_pass
                    d_map, r_map = float(np.linalg.norm(Em[:3, 3])), rot_deg(Em[:3, :3])
                both_true = (not info[i]["false"]) and (not info[j]["false"])
                one_false = info[i]["false"] != info[j]["false"]
                dist = float(np.linalg.norm(T_ij_gt[:3, 3]))
                pairs.append((both_true, one_false, d_gt, r_gt, d_map, r_map, s, dist))
    P = np.array([[p[0], p[1], p[2], p[3], p[4] if p[4] is not None else np.nan, p[5] if p[5] is not None else np.nan, p[6], p[7]] for p in pairs], dtype=float) if pairs else np.zeros((0, 8))
    out = {"n_obs": sum(1 for st in steps if st.get("ffp")), "n_pairs": len(P), "n_refs": len(refs_stats)}
    if len(refs_stats):
        e = np.array([r[0] for r in refs_stats]); out["ref_err_p50"] = float(np.median(e)); out["ref_false_frac"] = float(np.mean(e > false_thr))
    def q(x, ps=(50, 90, 99)):
        x = x[np.isfinite(x)]
        return {f"p{p}": float(np.percentile(x, p)) for p in ps} if len(x) else None
    for name, mask in (("both_true", P[:, 0] == 1), ("one_false", P[:, 1] == 1)):
        if len(P) and mask.any():
            sub = P[mask]
            rel = sub[:, 2] / np.maximum(sub[:, 7], 0.5)
            out[name] = {"n": int(mask.sum()), "d_gt": q(sub[:, 2]), "d_gt_rel": q(rel), "r_gt_deg": q(sub[:, 3]), "d_map": q(sub[:, 4]), "r_map_deg": q(sub[:, 5]),
                         "frac_d_gt_gt_1m": float(np.mean(sub[:, 2] > 1.0)), "frac_d_gt_gt_0.5m": float(np.mean(sub[:, 2] > 0.5)),
                         "frac_r_gt_gt_10deg": float(np.mean(sub[:, 3] > 10.0))}
    # tail-calibrated pair noise model sigma(d) = a + b d (translation, rotation) from the correct pairs, vs pair distance
    if len(P) and (P[:, 0] == 1).sum() > 50:
        from scipy.optimize import nnls
        from scipy.stats import chi2 as _chi2
        k = np.sqrt(_chi2.ppf(0.9, 3))
        sub = P[P[:, 0] == 1]
        fits = {}
        for name, col in (("t", 2), ("r", 3)):
            xs, ys, ws = [], [], []
            for lo, hi in zip([0, 0.5, 1, 2, 4, 8, 16, 32], [0.5, 1, 2, 4, 8, 16, 32, 64]):
                m = (sub[:, 7] >= lo) & (sub[:, 7] < hi)
                if m.sum() >= 30:
                    xs.append(float(np.median(sub[m, 7]))); ys.append(float(np.percentile(sub[m, col], 90) / k)); ws.append(np.sqrt(m.sum()))
            xs, ys, ws = np.asarray(xs), np.asarray(ys), np.asarray(ws)
            if len(xs) >= 2:
                A = np.stack([np.ones_like(xs), xs], 1) * ws[:, None]
                coef, _ = nnls(A, ys * ws)
                fits[name] = {"a": float(coef[0]), "b": float(coef[1]), "bins": [(float(x), float(y)) for x, y in zip(xs, ys)]}
        out["pair_fit"] = fits
    # separability: for thresholds on d_gt (m), TPR = fraction of both-true pairs below, FPR = fraction of one-false pairs below
    if len(P) and (P[:, 0] == 1).any() and (P[:, 1] == 1).any():
        out["thresholds"] = {}
        for thr in (0.25, 0.5, 1.0, 2.0):
            out["thresholds"][str(thr)] = {"true_pairs_consistent": float(np.mean(P[P[:, 0] == 1, 2] <= thr)),
                                           "false_pairs_consistent": float(np.mean(P[P[:, 1] == 1, 2] <= thr))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gate", action="store_true", help="replay the in-pass gate with the default / given pair model")
    ap.add_argument("--pair-t", type=float, nargs=2, default=(0.05, 0.025))
    ap.add_argument("--map-t", type=float, nargs=2, default=(0.05, 0.03))
    ap.add_argument("--confidence", type=float, default=0.999)
    args = ap.parse_args()
    res = {}
    for f in args.traces:
        dd = load(f)
        r = analyse(dd); res[f] = r
        if args.gate:
            r["gate"] = simulate_gate(dd, pair_t=tuple(args.pair_t), map_t=tuple(args.map_t), confidence=args.confidence)
            print(f"   gate replay (pair sigma_t {args.pair_t}, map {args.map_t}, c={args.confidence}): {r['gate']}")
        print(f"== {f}: {r['n_obs']} observations, {r['n_refs']} references (false {r.get('ref_false_frac')}), {r['n_pairs']} reference pairs")
        for k in ("both_true", "one_false"):
            if k in r:
                v = r[k]; print(f"   {k:9s} n={v['n']:6d} d_gt p50/p90/p99 {v['d_gt']}  rel {v['d_gt_rel']}  r_gt {v['r_gt_deg']}  d_map {v['d_map']}  frac>0.5m {v['frac_d_gt_gt_0.5m']:.3f} frac>1m {v['frac_d_gt_gt_1m']:.3f}")
        if "thresholds" in r:
            print("   thresholds:", r["thresholds"])
        if "pair_fit" in r:
            f = r["pair_fit"]; print(f"   pair fit: sigma_t = {f['t']['a']:.3f} + {f['t']['b']:.4f} d m (bins {[(round(x,1), round(y,3)) for x, y in f['t']['bins']]}); sigma_r = {f['r']['a']:.2f} + {f['r']['b']:.3f} d deg" if 't' in f and 'r' in f else f"   pair fit: {f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)




# ----------------------------------------------------------------------------- offline replay of the in-pass gate
def simulate_gate(d, pair_t=(0.05, 0.025), pair_r=(0.003, 0.0015), map_t=(0.05, 0.03), map_r=(0.02, 0.001), confidence=0.999, dof=3, false_thr=1.0):
    """Replay the in-pass consistency gate (largest mutually consistent reference set, chi-square test on the pair
    residual against the map relative pose) on every observation of a trace and score it against ground truth."""
    import itertools
    from scipy.stats import chi2 as _chi2
    thr = _chi2.ppf(confidence, dof)
    steps = d["steps"]
    gt = {}
    for st in steps:
        if st.get("nk") is not None:
            gt[(st["s"], int(st["nk"]))] = mat(st["gt"])
    map_ids = {int(k) for k in d["nodes"] if str(k).isdigit()}
    node_pose = {}
    for k, v in d["nodes"].items():
        node_pose[str(k)] = mat(v["p"])
    kept_true = kept_false = dropped_true = dropped_false = untested = 0
    for st in steps:
        f = st.get("ffp")
        if not f:
            continue
        s = st["s"]; G_cur = mat(st["gt"]); c2w = [mat(p) for p in f["c2w"]]
        ids = f["ids"]; valid = [bool(v) for v in f["valid"]]
        labels, poses = [], []
        for k, kid in enumerate(ids):
            key = (0, kid) if (s > 0 and kid in map_ids and (s, kid) not in gt) else (s, kid)
            nk = str(kid) if key[0] == 0 else f"{s}:{kid}"
            if key not in gt or not valid[k] or nk not in node_pose:
                labels.append(None); poses.append(None); continue
            T_pred = inv(c2w[1 + k]) @ c2w[0]; T_gt = inv(gt[key]) @ G_cur
            labels.append(float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])) > false_thr)
            poses.append(node_pose[nk])
        idx = [i for i in range(len(ids)) if labels[i] is not None]
        if len(idx) < 2:
            untested += sum(1 for i in idx)
            continue
        ok = {}
        for i, j in itertools.combinations(idx, 2):
            Pij = inv(c2w[1 + i]) @ c2w[1 + j]
            Mij = inv(poses[i]) @ poses[j]
            E = inv(Pij) @ Mij
            dd = float(np.linalg.norm(Pij[:3, 3])); dm = float(np.linalg.norm(Mij[:3, 3]))
            st_ = (pair_t[0] + pair_t[1] * dd) ** 2 + (map_t[0] + map_t[1] * dm) ** 2
            r_t = E[:3, 3]
            c2 = float(r_t @ r_t / st_)
            if dof == 6:
                sr = (pair_r[0] + pair_r[1] * dd) ** 2 + (map_r[0] + map_r[1] * dm) ** 2
                ang = np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))
                c2 += float(ang ** 2 / sr)
            ok[(i, j)] = c2 <= thr
        best = None
        for size in range(len(idx), 0, -1):
            for sub in itertools.combinations(idx, size):
                if all(ok[(i, j)] for i, j in itertools.combinations(sub, 2)):
                    best = sub; break
            if best is not None:
                break
        keep = set(best or ()) if (best is not None and 2 * len(best) > len(idx)) else set(idx)   # strict majority only
        for i in idx:
            if i in keep:
                kept_false += int(labels[i]); kept_true += int(not labels[i])
            else:
                dropped_false += int(labels[i]); dropped_true += int(not labels[i])
    n_false = kept_false + dropped_false; n_true = kept_true + dropped_true
    return {"n_true_refs": n_true, "n_false_refs": n_false, "untested_single": untested,
            "false_dropped_frac": (dropped_false / n_false) if n_false else None,
            "true_dropped_frac": (dropped_true / n_true) if n_true else None,
            "false_kept": kept_false, "true_dropped": dropped_true}


if __name__ == "__main__":
    main()
