# CROSS Nav: navigation from one camera and odometry

Optional package of CROSS (`bash install.sh --nav`; `--nav-sim` adds the Habitat simulator in its own Python 3.9
environment). Everything runs on the frozen VGGT-Omega backbone of the stereo mode.

| Part | Where | What |
|---|---|---|
| BEV head | `cross_nav/{model,head,train,train_visibility,infer}.py`, `cross_nav/configs/train/` | occupancy (and observability / error) on a gravity-aligned metric grid around the camera; `BEVPredictor`; also usable inside a CROSS session (`pose_est.ff.nav_bev_head`) |
| Planners | `cross_nav/{plan,local_plan,observed_plan}.py`, `cross_nav/policy/bev_policy.py`, `cross_nav/configs/policy/` | local planning on the BEV, closed-loop rules and episode map (`bev_planner_best.json`) |
| Trajectory policy | `cross_nav/policy/{head,train_sim,server,recovery}.py` | direct 16-waypoint path from four views, the goal and an odometry trail (flow head) |
| Robot agent | `cross_nav/policy/robot.py` | `NavigationAgent` with `TrajectoryPlanner`, `BEVPlanner` or `HybridPlanner` (`hybrid_best.json`) |
| Simulation | `cross_nav/policy/sim/` | Habitat closed-loop rollouts (HSSD, AI2-THOR), expert / DAgger data, side-by-side viewer |
| Data | `cross_nav/sim/`, `cross_nav/policy/{data,real_cache,prepare_gnm}.py`, `cross_nav/policy/datasets/` | BEV labels from rendered scenes; real-robot demonstrations (GNM mixture, SCAND, FrodoBots-2K, CAST, ...; `pip install -e ".[nav-data]"`) |

Usage and deployment: [`cross_nav/policy/README.md`](policy/README.md). Paths to data and checkpoints come from arguments or the
`CROSS_NAV_ROOT` environment variable (training configs and the shipped planner configs use `${CROSS_NAV_ROOT}`).

## BEV uncertainty and soft planning


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

`LocalWaypointPlanner` uses predicted observability to restrict candidate motion
to visible footprints, then selects a low-risk waypoint making useful progress
toward a goal outside that region. It can explicitly request a stop to re-observe.
Pass that action to `ObservedAudit.score(..., stop_for_reobserve=True)` and label
the evaluation `observed-local-stop-v2`. A safe terminal waypoint counts only if
it meets the same travel and goal-progress thresholds; collisions, unknown-space
boundaries, and goal eligibility retain their existing rules. The default
`observed-prefix-v1` behavior still treats an unfinished path as incomplete.
The optional `visibility_weight` adds `weight * -log(min_footprint(P(observed)))`
to occupancy risk. It favors routes with stronger observation support; it is a
heuristic cost, not a calibrated probability of successful execution. Its default
is zero. Occupancy uncertainty and input observability are distinct predictions.

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

## Observability and local re-observation planning

An optional full BEV observability head predicts which cells the input views
observe. This differs from occupancy uncertainty: a likely occupied cell may
still be unobserved, and an observed cell may have an uncertain occupancy label.
Wider temporal windows can supervise occupancy during training; inference uses
only the supplied current and past views.

`BEVPredictor` can run both heads on the same cached backbone features:

```python
predictor = BEVPredictor(
    occupancy_checkpoint, "cuda",
    visibility_path=observability_checkpoint,
    visibility_gravity_path=observability_gravity_checkpoint,
)
output = predictor(**cross_pass_tensors)
# output: prob, uncertainty, uncertainty_kind, observability, res, ...
```

The observability checkpoint must have `task="observability"`, its BEV weights,
training configuration, and the frozen gravity-head hash. Supply the checkpoint
containing that exact gravity head; loading rejects a mismatch. Occupancy and
observability retain their respective gravity predictions. Existing callers
without observability weights receive `observability=None`; their occupancy and
uncertainty computations are unchanged. Reproducing the evaluated policy uses
predicted metric scale and depth for every supplied view, without GT geometry or
an IMU override. Actual CROSS keyframe retrieval remains a separate integration
choice from the evaluated temporal input sampling.

```python
from cross_nav.local_plan import AdaptiveVisibilityPlanner

p = output["prob"].cpu().numpy()
v = output["observability"].cpu().numpy()
planner = AdaptiveVisibilityPlanner(
    p, v, start=(p.shape[0] // 2, p.shape[1] // 2), res=output["res"],
    thresholds=(.75, .625, .5, .375, .25),
    radius=.2, risk_weight=128., visibility_weight=16.,
)
segment = planner.plan(goal_cell)
```

The planner chooses the highest predicted visibility threshold that supports a
route to the goal or a local waypoint with at least 1 m displacement and 0.5 m
goal progress. `stop_for_reobserve` explicitly marks a local waypoint; the robot
must observe and plan again there. `path=None` means no such predicted route was
found. Confidence costs are heuristics, not calibrated collision probabilities.

`ObservedAudit` scores the executed prefix up to the first unobserved full robot
footprint, including side cells on diagonal steps. GT visibility is used only by
this audit. In `observed-local-stop-v2`, a local stop counts after at least 1 m
travel and 0.5 m goal progress; missed obstacles in unobserved space are excluded
from this primary score. Report visible collisions, stalls, and no-path outcomes,
and separately report globally unplanable, unknown-feasibility, and insufficient
observed-horizon goals. This is local-segment evaluation, not closed-loop final
goal completion. Its denominator and protocol differ from full-map planability
evaluation and historical scores.
