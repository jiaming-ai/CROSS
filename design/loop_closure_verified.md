# Verified loop closure (2026-09-12)

Question from the user: the intra-hypothesis PGO of 2026-09-08 misses loop closures, and its nine thresholds
(`intra_*`) are scene tuning.  Is there a principled, generic way to detect more true loop closures with very few
false positives — e.g. deciding from the residual after the PGO, or using VGGT itself — that transfers to real robot
data (KITTI) without per-dataset tuning?

## 1. What the stored data say (offline study, `scripts/lc/`)

Every run recorded with `scripts/viz/record_trace.py` now dumps the full pose graph with ground truth
(`graph_s<k>.json`, `scripts/lc/graph_io.py`) and, per observation, the metric camera poses of *all* views of the
feed-forward pass (`ffp`).  `scripts/lc/offline_graph_study.py` labels every visual edge with ground truth, fits noise
models, and replays the decision tests and the pose-graph optimisation offline; `scripts/lc/inpass_consistency.py`
analyses the pass-internal consistency of the retrieved references; `scripts/lc/edge_error_stats.py` checks the
calibration of the system's own edge std against the stored traces.

Mapping graphs of the current system (seed 0; Lone Monk 525 keyframes / 1885 visual edges, HSSD house 1045 / 4986,
HSSD restaurant 1010 / 4358):

| finding | Lone Monk | HSSD house | HSSD restaurant |
|---|---|---|---|
| visual edge translation error, loop edges, p50 / p99 | 0.13 / 0.85 m | 0.012 / 0.14 m | 0.011 / 0.28 m |
| std the system assigns to these edges vs. their actual error (median normalised error / 0.674) | 0.016 | 0.05 | 0.05 |
| std the system assigns to odometry edges vs. actual (same ratio) | 0.03 | 0.05 | 0.05 |
| odometry noise constant fitted from GT, per unit motion (simulator injects 1/(10 sqrt 3) = 0.058) | 0.058 | 0.061 | 0.061 |
| ATE of the graph re-optimised with the system's stds + Huber (odometry init) | 2.21 m | 2.01 m | 2.01 m |
| ATE odometry only | 2.13 m | 1.79 m | 1.73 m |
| ATE of the online system (merge PGO at the loop) | 0.75 m | 0.12 m | 0.34 m |
| ATE with calibrated stds + Huber | **0.21 m** | **0.066 m** | **0.116 m** |
| true loop edges accepted by the prior chi-square test at 0.999 (fitted odometry model) | 100 % | 100 % | 100 % |
| injected aliasing loops (3-15 m off) accepted after the optimisation, calibrated model | 0 % | 0 % | 0 % |
| wrong-place references among the retrieved references (all sessions) | 20.7 % (12 % mapping) | 1.1 % | 0.15 % |
| in-pass: reference pairs that disagree with the map by > 0.5 m, both references correct | 8.5 % (1 m: 0.2 %) | 0.3 % | 0.5 % |
| in-pass: same, one reference wrong (median disagreement) | 98.8 % (8.8 m) | 99.6 % (14 m) | 100 % (5 m) |

Three conclusions.  (i) The back end was miscalibrated by 20-30x on both edge types; that, not the detector, is why
the intra-hypothesis PGO "missed" loops (a 0.5 m drift is never significant against a 1 m std) and why the memo found
the plain graph "dragged back to the odometry chain".  (ii) With calibrated covariances the standard machinery works:
the odometry-chain prior accepts every true loop, the posterior residual rejects every injected false one, and the
optimised map is 3-6x better than the online result.  (iii) After a 140 m loop the odometry prior is +-3.5 m / +-14 deg
(analytic compounding verified against Monte Carlo), so no prior test can reject aliasing at the arcade spacing of
Lone Monk; the strong signal there is the *in-pass* consistency of the references, which separates wrong references
from right ones by an order of magnitude.

## 2. The method

**Why a calibrated noise model rather than a sensitivity setting.**  A gate on loop-closure candidates has to
answer one question: is the disagreement between the measurement and what the graph already knows explainable by
the noise of the two?  The chi-square test answers it at one fixed confidence level; how "sensitive" it is follows
from the covariances, not from the level.  With covariances that are too wide (the system's stored stds were
20-70x wider than the actual errors, section 1) every measurement passes, false ones included, and no measurement
is ever significant enough to move the graph — both failure modes of the intra-hypothesis PGO at once.  With
covariances that are too narrow, true loops fail the test.  Calibrating the covariances from the robot's own data
(section 3) is therefore what gives the fixed level its meaning and lets the same rule transfer between datasets
without per-scene tuning; the tolerance sweep (`scripts/lc/tolerance.py`, report table lc_tolerance) shows what the level itself controls (recall by
10-20 %, precision little).

A loop closure is a visual measurement that is more informative than the graph's own estimate of the relative pose.
Every candidate passes three consistency tests that share one decision parameter, the chi-square confidence level
`c` (default 0.999); the pose-graph optimisation uses the same calibrated noise model.  The test statistic is the
Mahalanobis distance of the *translation* residual (3 dof) whose covariance includes the rotation-induced position
uncertainty transported along the chain: a wrong-place reference is a translation phenomenon, and the rotation noise
of the estimator is the least well calibrated quantity (`test_dof: full` switches to 6 dof).

1. **Prior consistency** (`LoopClosureVerifier.prior_gate`).  Each measurement (reference -> current view) is compared
   with the dead-reckoned relative pose along the odometry chain between the reference keyframe and the current pose
   (or, in a relocalization session, from the last keyframe that has a verified map edge, through that edge and the
   stored map's local consistency), with the chain covariance compounded through the adjoints and inflated 2x for the
   uncertainty of the odometry model itself.  Consistent measurements are hypothesis-0 measurements — also when they
   are further from the belief than the GMM alignment radius (drift): no new hypothesis is born for them, their edges
   go to hypothesis 0 and the optimisation moves the belief.  Inconsistent ones may only feed or spawn other
   hypotheses (the SPRT / merge machinery of CROSS is unchanged and remains the tool for the lost-belief case).
   In a relocalization session the prediction for a stored-map reference runs through the *session anchor*, the
   last map edge of hypothesis 0 that was itself verified: an edge anchors the session only when it passed the
   prior test or, before any prior exists, when another map reference of the same pass corroborates it (the pair
   test against the stored map is tight), and the anchor is void whenever hypothesis 0 is replaced by a merge or
   an adoption.  A lone untested edge never anchors: in the hardest restaurant session a wrong first anchor made the
   prior test reject 80 of the next 106 true measurements and the session stayed 2-30 m off for 1500 steps until
   the hypothesis machinery recovered.  The prior also checks itself with the statistic of the online noise scale:
   when the median normalised innovation of the map measurements exceeds what the scale may absorb (its bound of
   5), the prior is the inconsistent party and the anchor is dropped (`anchor_dropped` in the statistics); the
   next verified or corroborated map edge re-anchors the session.
2. **In-pass consistency** (`inpass_gate`).  The forward pass registers the current view and up to six references
   together; for every pair of references the pass predicts a relative pose and the graph knows it too (odometry
   chain for same-session keyframes, stored map otherwise).  The best mutually consistent subset of references (most
   members with a positive prior verdict, then size, then covisibility) is kept; a reference outside it is dropped
   when its own prior verdict is negative or when the subset is a strict majority of the tested references, otherwise
   (one against one without prior evidence) nothing is decided.  Dropped references never become proposals or edges.
3. **Posterior consistency** (`System._maybe_verified_loop_closure`, `posterior_outliers`).  When a new
   loop-closure candidate (the graph's own prediction of the relative pose is less certain than the measurement)
   disagrees with the poses the graph holds (residual beyond the calibrated noise, plus the stored map's consistency
   for session->map edges), the graph is optimised (calibrated Gaussian odometry factors, calibrated Huber visual
   factors).  New edges that remain outliers afterwards — the graph's own verdict, at the same confidence level, that
   the measurement cannot be reconciled with everything else — are quarantined and the graph is re-optimised
   without them (`posterior_action: remove`; one truncated-least-squares step).  A robust kernel alone is not
   enough: on Lone Monk seed 0 four aliasing measurements 17 m off passed the prior test (after 140 m the odometry
   prior is +-17 m at the far end of the loop) and, although they stayed outliers after the Huber optimisation
   (chi2 270-640), their constant Huber pull rotated the whole loop — the odometry chain is soft in heading (a
   0.05 rad bend spread over 340 edges costs almost nothing against the 0.001 rad floor) — and the map ended at
   9.5 m ATE; the true loop closure 60 steps later then failed the test against the warped map.  (An earlier
   run in which removal seemed to lock the seed-2 map into a wrong first solution was produced with the verdicts
   not reaching the graph, section 5, and is not evidence.)  The same test runs over every informative measurement
   at the final optimisation of the map (at most three rounds).  The posterior test also rejects merges of other
   hypotheses whose edges do not fit the optimised graph.  No windows, counts or cooldowns: after an optimisation
   the revisit is consistent and nothing fires until new drift accumulates.

**Which measurements constrain the graph.**  A visual measurement enters the pose-graph optimisation only if it
is *informative*: the odometry chain between its endpoints (un-inflated, at the time of the observation) is less
certain than the measurement (translation covariance trace).  Measurements to the keyframes just behind the robot
are then explained by the odometry and stay out of the graph (they still feed the belief); revisits and session->map
edges are in.  This is the same criterion that triggers an optimisation, so it adds no parameter, and it matters on
real data: on KITTI-07 the local VGGT measurements are noisy (0.35 m median at 1 m/frame) and biased (metric scale
from a 0.54 m baseline at 20-40 m depth), and re-optimising the recorded graph with all edges gives 8.7 m ATE against
2.35 m for odometry alone, whereas the 58 informative edges of 2749 give 0.71 m; on the simulator scenes the
criterion keeps 25-85 % of the edges and changes the ATE by a few centimetres.

**Online noise scale.**  Under appearance change (night, rearrangement, reversed traversal) the estimator's correct
measurements against the map are up to 10x noisier than on the clean calibration data (median error 0.15-0.30 m
instead of 0.01 m in the HSSD night sessions); a static calibration then rejects ~40 % of the true edges.  The
verifier therefore keeps the normalised translation residual of its own prior test (prediction + calibrated
measurement covariance, before scaling) for the last 150 measurements and inflates the visual and pair noise by
median / 1.538 (the median of a 3-dof Gaussian norm), bounded to [1, 5].  The median is robust to the wrong-place
references the tests reject; session-internal spans alone would not see the change, because the degradation is
between the query and the map.  The scale is stamped on every stored edge so that the optimisation weights each
measurement by the noise level under which it was taken.

**Online metric scale.**  The feed-forward estimator's metric scale comes from the stereo anchors and is biased
where the anchors are far: on KITTI-06 the measured translations are 0.82 of the odometry's, on KITTI-07 0.89,
on Lone Monk 1.0-1.2 depending on the place, on the HSSD scenes 0.99 (ratios against ground truth and against the
odometry agree within 1 %).  The error of a long measurement is therefore mostly along its bearing (KITTI-06 at 5 m:
0.64 m along, 0.11 m across), and with an isotropic noise model the optimisation either trusts biased lengths or
discards good directions — on KITTI-06 the informative edges gave a map worse than the odometry alone (4.25 m
against 3.48 m).  The odometry is metric, so the ratio is observable without ground truth: the verifier keeps the
ratio of the raw measured translation to the odometry-chain translation over short chains (<= 5 edges, >= 1 m) of
the last 150 measurements and the estimator divides its translations by the median (a running calibration of the
estimator's scale against the odometry, like the noise scale above; a constant from the first minute would
over-correct Lone Monk, whose ratio drifts from 1.18 in the first minute to 1.03 overall).  Re-optimising the
recorded KITTI-06 graph with the corrected lengths gives 2.46 m.  The calibration file carries the first-minute
value as the initial ratio (`visual_scale`).

Removed: `intra_min_edges`, `intra_window_steps`, `intra_min_loop_steps`, `intra_min_conf`, `intra_min_residual_t`,
`intra_min_residual_r_deg`, `intra_cooldown_steps`, `intra_min_residual_sigma`, `intra_after_merge_cooldown_steps`,
`verify_outlier_sigma` (still present for the `heuristic` mode, which is kept for the A/B comparison:
`mapping.loop_closure.mode`).

## 3. Calibration without ground truth (`scripts/lc/calibrate_noise.py`)

The noise model must not be fitted per dataset against ground truth.  About a minute of the robot's own data
(any environment, ordinary motion with a few turns) recorded with the trace recorder gives:

* visual edge noise sigma_t(d) = a + b d, sigma_r: the innovation of visual edges against the odometry chain over
  spans without a turn (<= 10 keyframes, <= 10 m), tail-calibrated per distance bin with the chain's own translation
  variance (default odometry constants) subtracted so that the slope is the estimator's and not the odometry's; the
  intercept is bounded below by the unsubtracted innovation of the shortest straight spans, because a linear fit
  through the bins can extrapolate to a zero intercept, which no estimator has — a zero intercept gave a
  measurement at zero distance a factor of infinite weight in the first online runs (section 5).  The noise model
  itself floors every sigma at 1 mm / 0.1 mrad for the same reason;
* pair noise (in-pass test): the raw residual of reference pairs of one pass against their short odometry chain;
* odometry rotation constant k_r: innovation-based estimation on spans that contain a turn (the odometry's turn noise
  dominates the rotation innovation there); the translation constant k_t is only upper-bounded on short spans (the
  odometry is better than the visual measurement there) and kept at its default when unidentifiable — the safe side
  for a gate;
* the metric-scale ratio of the estimator (median measured / odometry-chain translation over short spans of at
  least 1 m, no ground truth; the online estimate takes over from it after 20 measurements);
* the map-consistency model (relative pose of two stored map keyframes) defaults to the visual model; a saved map
  carries its own model, fitted at save time from the posterior residuals of its hypothesis-0 edges against the final
  keyframe poses (no ground truth), and later sessions use it for the session->map tests.

Validation against the ground-truth fits (first 600 frames of each simulator scene, no GT used): see the run log
of the calibration in `configs/noise/*_600.json`.

## 4. Runtime

Observation step (VGGT-Omega, A100): 24 ms per view (3 views 77 ms, 10 views 247 ms); non-observation steps ~7 ms;
end-to-end 3-6 FPS with the motion-gated cadence.  Measured on the same idle A100 (Lone Monk seed 1, 1413 steps,
37 % observation steps, 4.6 views per pass): heuristic mode 0.308 s per observation step and 8.3 FPS overall,
verified mode 0.313 s and 8.2 FPS — the three tests cost about 5 ms per observation step.  The odometry-chain
predictor behind the prior and in-pass tests uses prefix products (relative pose and compounded covariance between
any two keyframes in O(1), a table append per new keyframe, a vectorised rebuild of ~1 ms per 500 edges when a
temporary keyframe is removed): the six-reference prior test takes 1.2 ms on a 524-keyframe chain, against 27 ms
with edge-by-edge compounding, and the cost no longer grows with the map.  A pose-graph optimisation of 500-1000
keyframes takes 0.1-0.8 s (Huber factors for the informative visual measurements, Gaussian odometry factors) and
runs at loop events (once in the Lone Monk mapping sessions) and once more when the map is saved: the revisit
edges recorded after the online loop closure are consistent with the graph and would otherwise never enter it
(Lone Monk seed 2: 0.40 m after the single online optimisation with 80 informative edges, 0.15 m with all 322
offline).

## 5. Implementation defects found in the first online runs (2026-09-12)

The offline study (section 1) was right and the first online runs were not, for three reasons that had nothing to do
with the tests themselves; they are recorded here because the earlier online numbers (Lone Monk seed 1: 0.92 m,
seed 2: 0.93 m against 0.32-0.44 m for the same graphs re-optimised offline; KITTI-07: 7.3 m) were produced with
them.

1. The observation routine of the system builds its result dictionary at the end and did not include the verifier's
   verdicts (`h0_ok`, `h0_chi2`, `h0_loop`).  Downstream, every measurement was therefore stored as *informative*
   (all 1872 edges of the Lone Monk map entered the optimisation instead of ~330), the optimisation fired on any
   inconsistent edge instead of loop candidates, and the proposal alignment never saw the prior verdicts (no
   inconsistent proposal was ever dropped).  `scripts/lc/replay_pgo.py` reproduces an online optimisation from the
   saved map and was what exposed it (stored flags all True, offline criterion 322 True).
2. The odometry-chain predictor rebuilt its successor table only when the number of odometry edges changed; the
   removal of a temporary keyframe (two edges replaced by one bridging edge) followed by a new keyframe leaves the
   count unchanged, so the table went stale: the references' chains could not be found, the prior test was skipped
   for them and they were flagged informative.  The table now follows a mutation counter of the edge set (and the
   newest edge key), with a regression test.
3. The ground-truth-free calibration of Lone Monk fitted a zero intercept (sigma_t = 0.087 d).  One measurement
   at ~zero distance then received a sigma of 1e-9 m and a weight of 1e18: the optimisation at step 1408 satisfied
   that one factor at a cost of 1e8 for everything else and folded the map around keyframes 253-270 (odometry
   residuals of 30 sigma); the robust visual factors then held the fold (the online state re-optimised from itself
   stays at 2.4 m ATE, from the odometry initialisation the same graph gives 0.17 m).  Calibration intercept floor
   and model floors as in section 3.
4. `apply_pgo_result` halved the belief std of every optimised keyframe at every optimisation.  The heuristic mode
   optimised at most once per session; the verified mode optimises at every loop event (HSSD house: 100-400 per
   mapping session), so the stds underflowed to exactly zero (1037 of 1045 keyframes), the belief fusion produced
   garbage poses (a keyframe 30 m from its neighbours, odometry residuals of 4000 sigma), the optimiser could not
   take a single step from such a state (cost 4e7, 0 iterations) and the house maps came out at 0.25-1.8 m where the
   same graphs optimise to 0.06 m offline.  The std is now left unchanged (no marginals are computed).
5. The pypose <-> gtsam pose conversions of the optimiser did not renormalise quaternions.  gtsam builds the
   rotation matrix assuming a unit quaternion; a non-unit one gives a scaled, non-orthonormal matrix, and the round
   trip of every optimisation compounded the error (a keyframe of the restaurant map reached |q| = 1.67 after ~100
   optimisations; the optimiser then reports a cost of 14k for values that cost 167k in a fresh graph, takes no
   further step, and the save-time rounds rejected true edges).  The graph dumps hid it because they normalise.
   Both conversions renormalise now, with a round-trip test.
