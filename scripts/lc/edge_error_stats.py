#!/usr/bin/env python3
"""Calibration check of the visual-edge measurements against ground truth, from raw replay traces.

For every visual edge a->b recorded in a trace (scripts/viz/record_trace.py), the measured relative translation
(`rel`, T_ref_cam of keyframe b in the frame of keyframe a) is compared with the ground-truth relative translation
G_a^-1 G_b.  Errors are reported as a function of the covisibility confidence, the retrieval score, the distance and
the age of the edge (local tracking edge, loop edge inside a session, or session->map edge), and normalised by the
std the system assigns to the edge (base_std / (4 * covis * retrieval score)), which tells whether the covariance
model is calibrated (per-axis |z| should have median ~0.67 and 95th percentile ~1.96).

usage: python scripts/lc/edge_error_stats.py trace.json [trace.json ...] [--out stats.json]
"""
import argparse, json, sys
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_STD = np.array([0.2, 0.2, 0.3])


def mat(p):
    T = np.eye(4); q = np.asarray(p[3:7], float); q /= np.linalg.norm(q)
    T[:3, :3] = R.from_quat(q).as_matrix(); T[:3, 3] = p[:3]; return T


def load(f):
    t = open(f).read()
    if f.endswith(".js"):
        t = t[t.index("]=") + 2:].rstrip().rstrip(";")
    return json.loads(t)


def edge_table(d):
    steps = d["steps"]
    # ground-truth pose of every keyframe, keyed per session by the keyframe id (ids restart after a map load)
    gt = {}
    for st in steps:
        if st.get("nk") is not None:
            gt[(st["s"], int(st["nk"]))] = mat(st["gt"])
    map_ids = {int(k) for k in d["nodes"] if str(k).isdigit()}
    rows = []
    for e in d["edges"]:
        if e["t"] != "visual" or "rel" not in e:
            continue
        s, a, b = e["s"], int(e["a"]), int(e["b"])
        # the session-0 (map) keyframes keep their ids in later sessions
        ka = (0, a) if (s > 0 and a in map_ids and (s, a) not in gt) else (s, a)
        kb = (s, b)
        if ka not in gt or kb not in gt:
            continue
        st = steps[e["step"]] if e["step"] < len(steps) else None
        conf, rw = None, None
        if st is not None:
            for v in st.get("vk", []):
                if int(v[0]) == a:
                    conf, rw = float(v[1]), float(v[2])
        T_gt = np.linalg.inv(gt[ka]) @ gt[kb]
        t_gt = T_gt[:3, 3]
        t_meas = np.asarray(e["rel"], float)
        err = t_meas - t_gt
        step_a = next((x["step"] for k, x in d["nodes"].items() if (str(k) == str(a) if ka[0] == 0 else str(k) == f"{s}:{a}")), None)
        rows.append({
            "s": s, "a": a, "b": b, "cross_session": ka[0] != s, "conf": conf, "rw": rw,
            "dist": float(np.linalg.norm(t_gt)), "err": err.tolist(), "err_norm": float(np.linalg.norm(err)),
            "age_steps": (e["step"] - step_a) if (step_a is not None and ka[0] == s) else None,
            "fc": e.get("fc"), "tc": e.get("tc"),
        })
    return rows


def q(x, ps=(50, 90, 95, 99)):
    x = np.asarray(x, float)
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} if len(x) else {}


def summarize(rows, label):
    out = {"label": label, "n": len(rows)}
    if not rows:
        return out
    en = np.array([r["err_norm"] for r in rows])
    out["err_norm"] = q(en)
    out["frac_gt"] = {t: float(np.mean(en > t)) for t in (0.3, 0.5, 1.0, 2.0, 5.0)}
    have = [r for r in rows if r["conf"] is not None]
    if have:
        z = []
        for r in have:
            sd = BASE_STD / (4 * max(r["conf"], 1e-2) * max(r["rw"], 1e-3))
            z.append(np.abs(np.asarray(r["err"]) / sd))
        z = np.concatenate(z)
        out["abs_z_per_axis"] = q(z, (50, 90, 95, 99))
        out["z_median_over_0.674"] = float(np.median(z) / 0.674)
        # by covisibility bins
        bins = [(0.15, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
        out["by_conf"] = {}
        for lo, hi in bins:
            sel = [r["err_norm"] for r in have if lo <= r["conf"] < hi]
            if sel:
                out["by_conf"][f"{lo}-{hi}"] = {"n": len(sel), **q(sel, (50, 90, 99)), "frac>1m": float(np.mean(np.array(sel) > 1))}
    out["by_dist"] = {}
    for lo, hi in [(0, 1), (1, 3), (3, 8), (8, 20), (20, 100)]:
        sel = [r["err_norm"] for r in rows if lo <= r["dist"] < hi]
        if sel:
            out["by_dist"][f"{lo}-{hi}m"] = {"n": len(sel), **q(sel, (50, 90, 99)), "frac>1m": float(np.mean(np.array(sel) > 1))}
    kinds = {"local(<100 steps)": [r for r in rows if r["age_steps"] is not None and r["age_steps"] < 100],
             "loop(>=300 steps)": [r for r in rows if r["age_steps"] is not None and r["age_steps"] >= 300],
             "session->map": [r for r in rows if r["cross_session"]]}
    out["by_kind"] = {}
    for k, sel in kinds.items():
        if sel:
            en_k = np.array([r["err_norm"] for r in sel])
            out["by_kind"][k] = {"n": len(sel), **q(en_k, (50, 90, 99)), "frac>1m": float(np.mean(en_k > 1)), "frac>2m": float(np.mean(en_k > 2))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    allrows, res = [], []
    for f in args.traces:
        d = load(f)
        rows = edge_table(d)
        allrows += rows
        s = summarize(rows, f)
        res.append(s)
        print(f"== {f}: {len(rows)} visual edges with GT")
        if rows:
            print("   |err| quantiles:", {k: round(v, 3) for k, v in s['err_norm'].items()}, " frac>1m", round(s['frac_gt'][1.0], 4), " frac>2m", round(s['frac_gt'][2.0], 4))
            if "abs_z_per_axis" in s:
                print("   per-axis |z| (model std):", {k: round(v, 2) for k, v in s['abs_z_per_axis'].items()}, " median|z|/0.674 =", round(s['z_median_over_0.674'], 3))
                for k, v in s["by_conf"].items():
                    print(f"   covis {k}: n={v['n']} p50={v['p50']:.3f} p90={v['p90']:.3f} p99={v['p99']:.3f} frac>1m={v['frac>1m']:.3f}")
            for k, v in s["by_dist"].items():
                print(f"   dist {k}: n={v['n']} p50={v['p50']:.3f} p90={v['p90']:.3f} p99={v['p99']:.3f} frac>1m={v['frac>1m']:.3f}")
            for k, v in s["by_kind"].items():
                print(f"   {k}: n={v['n']} p50={v['p50']:.3f} p90={v['p90']:.3f} p99={v['p99']:.3f} frac>1m={v['frac>1m']:.3f} frac>2m={v['frac>2m']:.3f}")
    if len(args.traces) > 1:
        s = summarize(allrows, "ALL"); print("== ALL:", json.dumps({k: s[k] for k in ("n", "err_norm", "frac_gt", "abs_z_per_axis", "by_kind") if k in s}, indent=None, default=float)[:1500])
        res.append(s)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
