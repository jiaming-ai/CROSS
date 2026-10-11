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

Optional `data.context_lags: [0, 4, 8, 16, 32, 64, 128, 256]` with
`data.frames: [8, 8]` trains on a fixed current frame and seven past frames.
This is a temporal proxy for keyframes, not live CROSS retrieval. Short histories
fill the largest temporal gaps without duplicate or future frames; anchors with
fewer than eight available frames or invalid ground height are rejected.
This recipe replaces cross-session and negative references with same-sequence
past context. World-level labels remain unchanged and can include observations
outside the input time range. Omitting `context_lags` preserves existing sampling.

`NavModel.heads(..., return_aux=True)` returns `logits` and `error_logits` (the
latter is `None` for old heads). `BEVPredictor` adds `uncertainty` to its output:
a sigmoid error map, or `None`. Check calibration, occupied-cell errors and
selective error on held-out scenes before using it for decisions. An unobserved
cell can be confidently predicted free and still contain an obstacle.

## Planning

For receding-horizon evaluation, `cross_nav.observed_plan.ObservedAudit` scores
only the proposed execution prefix inside observed space. It stops before an
unobserved full robot footprint, including diagonal side cells. Visible collisions,
goal completion, safe progress to re-observe, and insufficient progress are separate
outcomes. Defaults require 1 m travel and 0.5 m reduction in goal distance for a
successful re-observation segment. Cases without a GT-supported observed route
offering that progress are reported separately, not counted as model failures.
The present robot footprint is assumed observed only where GT is free.

Observation masks must come from the input views with occlusion accounted for;
the historical `vis` depth-hit mask is not a complete visibility mask. GT masks
are used only in evaluation. Local segment success is not final-goal completion
or a closed-loop navigation result. Keep the full-map audit as a separate diagnostic.

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

## Joint learned attenuation

Set `bev_head.uncertainty_mode: attenuated` together with `uncertainty: true`.
The auxiliary output predicts `s = softplus(raw) >= 0`; the labelled-cell BCE
term becomes `exp(-s) * BCE + beta * s`, where `loss.attenuation_beta` defaults
to 0.3 and must be positive. The original unweighted occupied Dice and gravity
loss remain. Both occupancy and uncertainty receive gradients through the
shared decoder. This is a regularized robust classification objective, not an
exact Bernoulli likelihood or a calibrated uncertainty posterior.

Here `uncertainty = 1 - exp(-s) = sigmoid(raw)` is an attenuation score, not
P(error). Checkpoint metadata and exported manifests preserve that distinction;
error-probability calibration metrics are omitted for this mode. The regularizer
prevents all cells becoming uncertain, and `s >= 0` keeps the objective bounded
below even at zero BCE. No target exists for unlabelled cells. This objective
alone does not guarantee high uncertainty throughout unseen space.

## Explicit planability audit

`cross_nav.planability.evaluate_goals` evaluates a supplied, fixed candidate list
without snapping the start. A goal is **planable** when a known-free route exists
for the full robot footprint, **unplanable** when no route exists even treating
unknown labels as free, and **unknown** otherwise. Eight-connected paths cannot
cut blocked corners. The same radius is used for feasibility and route checking;
outside-grid space is blocked. Missing labels along a predicted path produce an
`unknown_path` outcome rather than verified success. All planable goals remain
in the denominator, including planner abstentions. A zero denominator yields
`None`, not 100% success. Historical scores remain unchanged.

```bash
python scripts/nav_planability_eval.py VIZ/win/arrays --out VIZ/planability.json
python scripts/nav_viz_pack.py VIZ PAGE 'checkpoint with planability audit'
```

For soft planning, `--diagonal-risk` charges the maximum risk of the two
endpoints and both side cells on each diagonal edge. This accounts for the
side-cell footprints checked by the collision audit, even when the soft map
leaves every cell traversable. It changes the planner, not goal feasibility or
success criteria. The flag is opt-in to preserve earlier comparisons; the
output records whether it was enabled.

An experimental training loss, `loss.w_route` (default zero), adds positive BCE
on labelled obstacles crossed by predicted routes. It samples up to
`loss.route_goals` (default 32) reachable training goals, uses the diagonal-risk
planner with `loss.route_radius` (default 0.2 m), and keeps the base occupancy
loss. Mining uses detached predictions on the CPU; GT supplies training targets
and feasible training goals only. Unknown cells are not obstacle targets. The
logs report `loss_route`, `route_pixels` and `route_goals`; held-out evaluation is
needed before adopting this loss.

By default candidates are sampled uniformly in the 2–8 m annulus before reading
GT or predictions. `--goal-source historical` instead audits the old sampled
goals from the exact start, helping diagnose differences between protocols.
The viewer shows feasibility counts and verified success separately from the
historical score. The GT-known diagnostic is a label-based ceiling, not a model
result. Sparse labels can leave few verifiably planable goals; use sufficiently
complete held-out ground truth for conclusions, and report the denominator.

For queued GPU jobs, export cached predictions with `--save-arrays --no-images
--skip-planning`, then run `nav_planability_eval.py` in a separate CPU process.
This releases the GPU for the next training run while path searches continue.
The export manifest records `planning_skipped: true`; its empty planning fields
are not evaluation results.

Use `nav_planability_eval.py --condition oracle` only to diagnose the effect of
scale/gravity errors on the same candidates. The default mono condition remains
the deployment score; oracle results are labelled separately.
