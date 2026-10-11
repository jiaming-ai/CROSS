"""CPU-only planner comparison on cached nav_viz_export --save-arrays output.

Tune weights on validation windows, then freeze the choice before test export.
Bootstrap paired success differences by window, not by correlated goals.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cross_nav.plan import evaluate_planning, risk_cost, summarize


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("arrays", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=[1., 4., 16.])
    ap.add_argument("--goals", type=int, default=10)
    a = ap.parse_args()
    rows = []
    for path in sorted(a.arrays.glob("*.npz")):
        with np.load(path) as z:
            gt, res = z["gt"], float(z["res"])
            for cond, pk, uk in (("oracle", "po", "uo"), ("mono", "pm", "um")):
                p = z[pk]
                maps = {"head": p > .5}
                costs = {}
                for weight in a.weights:
                    name = f"risk_{weight:g}"
                    maps[name] = np.zeros_like(gt, bool)
                    costs[name] = risk_cost(p, res, weight=weight)
                    if uk in z:
                        name += "_uncertainty"
                        maps[name] = np.zeros_like(gt, bool)
                        costs[name] = risk_cost(p, res, weight=weight, uncertainty=z[uk], uncertainty_weight=4.)
                recs = evaluate_planning(gt, maps, res, (gt.shape[0]//2, gt.shape[1]//2),
                                         n_goals=a.goals, seed=int(z["seed"]), costs=costs)
                rows.append({"window": path.stem, "condition": cond,
                             "methods": {m: summarize(recs, m) for m in maps}})
        print(path.stem, flush=True)
    totals = {}
    for cond in ("oracle", "mono"):
        rr = [r for r in rows if r["condition"] == cond and r["methods"]["head"]["n"]]
        if not rr:
            continue
        rng = np.random.default_rng(123)
        draws = rng.integers(len(rr), size=(2000, len(rr)))
        totals[cond] = {}
        for m in rr[0]["methods"]:
            n = np.array([r["methods"][m]["n"] for r in rr])
            score = {k: float(np.average([r["methods"][m][k] for r in rr], weights=n))
                     for k in ("success", "collision", "no_path", "detour", "spl", "known_success", "unknown_path_rate")}
            delta = np.array([r["methods"][m]["success"] - r["methods"]["head"]["success"] for r in rr])
            boot = (delta[draws] * n[draws]).sum(1) / n[draws].sum(1)
            score.update(n=int(n.sum()), windows=len(rr),
                         paired_success_delta_ci95=np.quantile(boot, [.025, .975]).tolist())
            totals[cond][m] = score
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"totals": totals, "windows": rows}, indent=2))
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
