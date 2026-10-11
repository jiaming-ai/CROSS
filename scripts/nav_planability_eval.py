"""CPU evaluation of explicit planability on cached BEV predictions.

Candidates are sampled before examining either GT or predictions. Use an
independent held-out cache for final reporting after tuning on validation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from cross_nav.plan import risk_cost, evaluate_planning
from cross_nav.planability import evaluate_goals, summarize_goals


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('arrays', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--candidates', type=int, default=100)
    ap.add_argument('--goal-source', choices=['unfiltered', 'historical'], default='unfiltered')
    ap.add_argument('--radius', type=float, default=.2)
    ap.add_argument('--risk-weight', type=float, default=4.)
    args = ap.parse_args()
    rows = []
    for file in sorted(args.arrays.glob('*.npz')):
        with np.load(file) as z:
            gt, res, p = z['gt'], float(z['res']), z['pm']
            start = (gt.shape[0]//2, gt.shape[1]//2)
            if args.goal_source == 'historical':
                old = evaluate_planning(gt, {}, res, start, n_goals=10, seed=int(z['seed']))
                goals = [r['goal'] for r in old]
            else:
                rr, cc = np.indices(gt.shape)
                distance = np.hypot(rr-start[0], cc-start[1])*res
                candidates = np.argwhere((distance >= 2) & (distance <= 8))
                rng = np.random.default_rng(int(z['seed']))
                goals = candidates[rng.choice(len(candidates), min(args.candidates, len(candidates)), replace=False)].tolist()
            maps = {'head': p > .5, 'soft_risk': np.zeros_like(gt, bool),
                    'gt_known_diagnostic': gt != 1}
            costs = {'soft_risk': risk_cost(p, res, robot_radius=args.radius, weight=args.risk_weight)}
            if 'um' in z:
                maps['soft_uncertainty'] = np.zeros_like(gt, bool)
                costs['soft_uncertainty'] = risk_cost(p, res, robot_radius=args.radius, weight=args.risk_weight,
                                                      uncertainty=z['um'], uncertainty_weight=4)
            if 'gm' in z:
                fused = np.maximum(p, np.where(z['gm'] == 2, .9, 0))
                maps['soft_geometry_fusion'] = np.zeros_like(gt, bool)
                costs['soft_geometry_fusion'] = risk_cost(fused, res, robot_radius=args.radius, weight=args.risk_weight)
            recs = evaluate_goals(gt, maps, start, goals, res, robot_radius=args.radius, costs=costs)
            summaries = {m: summarize_goals(recs, m) for m in maps}
            if summaries['gt_known_diagnostic']['planable']:
                assert summaries['gt_known_diagnostic']['success_rate'] == 1, file
            rows.append({'window': file.stem, 'methods': summaries, 'goals': recs})
            print(file.stem, summaries['soft_risk'], flush=True)
    totals = {}
    if rows:
        for m in rows[0]['methods']:
            rr = [r['methods'][m] for r in rows]
            total = {k: sum(r[k] for r in rr) for k in ['candidates', 'planable', 'unplanable', 'unknown']}
            total['outcomes'] = {k: sum(r['outcomes'][k] for r in rr) for k in rr[0]['outcomes']}
            total['success_rate'] = total['outcomes']['success']/total['planable'] if total['planable'] else None
            totals[m] = total
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'protocol': 'exact-start-known-footprint-v1', 'goal_source': args.goal_source,
                                    'radius': args.radius, 'totals': totals, 'windows': rows}))
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
