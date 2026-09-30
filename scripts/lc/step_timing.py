#!/usr/bin/env python3
"""Per-step runtime of recorded runs (trace.json): observation steps vs other steps, forward-pass time, PGO time.
usage: python scripts/lc/step_timing.py outputs/lcstudy/<scene>/<run> [...]"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np


def summarize(run: Path) -> dict:
    t = json.loads((run / "trace.json").read_text())
    out = {"run": str(run), "sessions": []}
    for s_idx, sess in enumerate(t["sessions"]):
        steps = [st for st in t["steps"] if st.get("s") == s_idx]
        if not steps:
            continue
        dt = np.array([st["dt"] for st in steps]); obs = np.array([bool(st.get("obs")) for st in steps])
        ffp = [st["ffp"].get("t") or st["ffp"].get("time") or st["ffp"].get("dt") for st in steps if isinstance(st.get("ffp"), dict)]
        ffp = [x for x in ffp if x is not None]
        lc = sess.get("lc_stats") or {}
        out["sessions"].append({
            "id": s_idx, "variant": sess.get("variant"), "n_steps": len(steps), "obs_frac": float(obs.mean()),
            "dt_obs_mean": float(dt[obs].mean()) if obs.any() else None, "dt_obs_median": float(np.median(dt[obs])) if obs.any() else None,
            "dt_other_mean": float(dt[~obs].mean()) if (~obs).any() else None,
            "ffp_mean": float(np.mean(ffp)) if ffp else None, "n_views_mean": float(np.mean([len(st.get("vk", [])) + 1 for st in steps if st.get("obs")])) if obs.any() else None,
            "fps": float(len(steps) / dt.sum()), "pgo": lc.get("pgo"), "pgo_time": lc.get("pgo_time"),
            "wall_s": float(dt.sum()),
        })
    return out


def main():
    rows = []
    for r in sys.argv[1:]:
        d = summarize(Path(r))
        for s in d["sessions"]:
            rows.append((Path(r).parent.name + "/" + Path(r).name, s))
    print("| run | session | steps | obs frac | dt obs mean/med (s) | dt other (s) | ff pass (s) | views | FPS | PGO n / time (s) | wall (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"
    for name, s in rows:
        print(f"| {name} | s{s['id']} {str(s['variant'])[:18]} | {s['n_steps']} | {s['obs_frac']:.2f} | {f(s['dt_obs_mean'])} / {f(s['dt_obs_median'])} | {f(s['dt_other_mean'])} | {f(s['ffp_mean'])} | {f(s['n_views_mean'], 1)} | {s['fps']:.1f} | {s['pgo']} / {f(s['pgo_time'], 1)} | {s['wall_s']:.0f} |")


if __name__ == "__main__":
    main()
