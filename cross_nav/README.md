# CROSS Nav uncertainty and soft planning

The BEV head predicts occupancy on a gravity-aligned metric grid. Set
`bev_head.uncertainty: true` to add a learned **occupancy decision error** output.
It estimates whether the decision `P(occupied) > 0.5` is wrong; it is not an
epistemic variance or a visibility label. Unlabelled GT cells have no loss.
Unseen cells with GT labels supervise completion errors normally.

The auxiliary head reads detached decoder features, so its loss cannot change
the occupancy features. `loss.w_error` defaults to 1. Set `train.error_only: true`
to freeze occupancy and gravity as well. Warm-starting from an old head accepts
only the missing auxiliary parameters; other checkpoint mismatches still fail.
Old configurations and checkpoints keep their tensor-returning interface.
For mixture ablations, set `data.require_all_datasets: true` to fail if a requested
dataset cannot load instead of silently changing the training mixture.

`NavModel.heads(..., return_aux=True)` returns `logits` and `error_logits` (the
latter is `None` for old heads). `BEVPredictor` adds `uncertainty` to its output:
a sigmoid error map, or `None`. Check calibration, occupied-cell errors and
selective error on held-out scenes before using it for decisions. An unobserved
cell can be confidently predicted free and still contain an obstacle.

## Planning

```python
from cross_nav.plan import evaluate_planning, risk_cost
import numpy as np

maps = {"threshold": prob > 0.5, "soft": np.zeros_like(prob, dtype=bool)}
costs = {"soft": risk_cost(prob, res=0.05, robot_radius=0.2, weight=4.0)}
records = evaluate_planning(gt, maps, 0.05, start=(200, 200), costs=costs)
```

The soft cost is `1 + weight * -log(1 - max_footprint(P(occupied)))`.
The maximum covers the robot footprint, without assuming neighbouring cells
are independent. This is a risk score, not a calibrated path collision probability.
Optional `uncertainty` and `uncertainty_weight` add an error penalty. Supply hard
obstacles in `maps` when needed; they are inflated as in the threshold baseline.
All methods share sampled goals and the historical 8-connected grid semantics.

The historical score checks 0.1 m clearance from labelled obstacles and a maximum
3x GT detour. It does not establish safety in unknown GT cells. `summarize` also
reports `known_success` (successful paths entirely on labelled cells) and
`unknown_path_rate`. Threshold and soft methods must use the same scoring rules.

## Evaluation and visualization

```bash
python scripts/nav_viz_export.py --heads HEADS --backbone BACKBONE \
  --out VIZ/win --per-dataset 0 --goals 10 --save-arrays WINDOWS.json
python scripts/nav_plan_sweep.py VIZ/win/arrays --out SWEEP.json
python scripts/nav_viz_pack.py VIZ PAGE 'checkpoint label'
python -m http.server --directory PAGE 8765
```

The exporter includes uncertainty heatmaps, a selective BEV that leaves cells
unknown above 0.2 predicted error, and soft-risk/uncertainty planner overlays.
The 0.2 cutoff is illustrative, not a validated operating point. Old checkpoints
still export without uncertainty panels. `--save-arrays` preserves floating-point
predictions for paired CPU planner comparisons. Tune costs on validation data and
freeze them for test evaluation. Optional `VIZ/note.txt` appears on the page.
