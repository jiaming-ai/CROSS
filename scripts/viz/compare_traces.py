#!/usr/bin/env python3
"""Compare replay traces (scripts/viz/compare_traces.py <trace.json|page trace .js> ...): loop-closure events (false / corrective / neutral) and per-session error statistics.
Errors: mapping session -> position error vs GT (first-keyframe alignment); reloc sessions -> map-relative position error
(computed here from mu / gt / map nodes, so it works on raw trace.json files without the page builder)."""
import sys, json, numpy as np
from scipy.spatial.transform import Rotation as R


def load(f):
    """trace.json, or a page trace (traces/<scene>__<method>.js)."""
    t = open(f).read()
    if f.endswith(".js"):
        t = t[t.index("]=") + 2:].rstrip().rstrip(";")
    return json.loads(t)
def mat(p):
    T = np.eye(4); q = np.array(p[3:7]); q = q / np.linalg.norm(q); T[:3, :3] = R.from_quat(q).as_matrix(); T[:3, 3] = p[:3]; return T
def inv(T):
    o = np.eye(4); o[:3, :3] = T[:3, :3].T; o[:3, 3] = -T[:3, :3].T @ T[:3, 3]; return o
def session_errors(d, s):
    st = [e for e in d["steps"] if e["s"] == s]
    if "err" in st[0] and st[0]["err"] and st[0]["err"][0] is not None:
        return np.array([e["err"][0] if s == 0 else e["err"][8] for e in st]), st
    # raw trace: map-relative error from kf_gt / map_final_nodes
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
def report(path, label=None):
    d = load(path); label = label or path
    print(f"== {label}  map ATE {d.get('map_ate_rmse', float('nan')):.2f}")
    tot = {"FALSE": 0, "fix": 0, "neutral": 0}
    for sess in d["sessions"]:
        s = sess["id"]; err, st = session_errors(d, s)
        ev = []
        for e in st:
            if "lc" not in e: continue
            i = e["i"]; b = np.median(err[max(0, i - 3):i]) if i > 0 else err[0]; a = np.median(err[i + 3:i + 6]) if i + 6 <= len(err) else err[-1]
            kind = "FALSE" if a - b > 2 else ("fix" if b - a > 2 else "neutral"); tot[kind] += 1
            ev.append(f"{i}:{'h%s' % e['lc']}:{b:.1f}->{a:.1f}:{kind}")
        print(f"   s{s} {sess['variant']:42s} err mean {err.mean():6.2f} med {np.median(err):5.2f} frac>2m {np.mean(err > 2):.2f} | LC {' '.join(ev)}")
    print("   totals:", tot)
if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
