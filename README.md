# CROSS-Mono

Experimental monocular extension of [CROSS](https://arxiv.org/abs/2605.02227), built upon its global retrieval observation message, continuous multi-modal pose hypotheses and delayed commitment. Those existing formulations remain the foundation of this system. This separate research repository adds monocular inputs and uncertain learned metric priors.

The monocular API accepts **RGB, timestamps and camera calibration**. It does not consume sensor depth, input odometry, IMU readings or ground-truth poses. Learned metric depth supplies an uncertain prior; its absolute scale may remain biased out of distribution.

## Monocular architecture

- Motion: lightweight XFeat features with robust PnP against periodic metric-depth anchors. Optional person exclusion reduces foreground tracking. DA3 compact-memory and DPVO scalar-scale frontends are retained as experimental alternatives.
- Metric priors: DA3-Metric-Large supplies uncertain depth shape and scale. The DPVO alternative uses robust log-scale observations, persistent-alternative recovery and a systematic uncertainty floor.
- Mapping: the existing CROSS hypothesis lifecycle. Learned retrieval poses are verified with image correspondences; scale uncertainty enters the original diagonal SE(3) filter conservatively.
- Evaluation: separate ground-truth reader, rigid-aligned metric ATE, similarity-aligned ATE and fitted scale, metric RPE, tracking coverage and runtime including periodic inference.

The single-session baseline runs at camera rate in the measured consumer-GPU tests. Multi-session mapping remains experimental. Optional two-view verification recovers difficult viewpoint and lighting queries, and an accumulated three-session map also recovers the people query. A longer seven-session chain exposes a late false map merge; the experimental shared-geometry option rejects this reproduced failure in the controls below. Broad changed-session robustness is not established. The default diagonal metric bridge is not a full joint Sim(3) posterior; a conditional pose/source option is described below.

The first frozen OpenLORIS home/café controls expose further failures. At 20 Hz,
five completed runs emit all 9,799 selected frames without a 50 ms deadline miss,
but reference ATE is 1.15 m (home) and 0.66 m (café). After the first received
map commitment, query position RMSE is 1.52 m for home and 1.36–1.40 m for café,
using only the original reference's rigid alignment. Home's shared-geometry
query crashes after 554 of 2,000 outputs because another hypothesis survives
at graph refresh. It is a failed run. Only 26.45% of the home query outputs
have GT associations under the declared 0.1 s interpolation-gap cutoff.
These results do not establish robust cross-environment operation; the frontend
also accumulates substantial orientation error around tracking losses.

The monocular bridge defaults to CROSS's original `full` policy: the global observation updates the active pose as well as competing hypotheses and delayed-commitment evidence. `--filter-mode skip_active` reproduces the earlier monocular bridge, where the active pose follows local motion; `adaptive` exposes the inherited adaptive gate. These select existing policies; they do not change the observation message or commitment thresholds. Report the selected policy in comparisons, including when reproducing historical runs that used `skip_active`.

## Monocular installation and usage

Tested environments use Linux and Python 3.11: PyTorch 2.5.1/CUDA 12.4 on A6000, and PyTorch 2.8.0/CUDA 12.8 with torchvision 0.23.0 on RTX 5090. First install CROSS's dependencies using the instructions below, then add the monocular dependencies and pinned DA3 implementation:

```bash
uv pip install -e '.[mono,dev]'
uv pip install gtsam==4.2

git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git /path/to/Depth-Anything-3
git -C /path/to/Depth-Anything-3 checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
uv pip install --no-deps /path/to/Depth-Anything-3
```

For DPVO, follow its [official build instructions](https://github.com/princeton-vl/DPVO) at commit `0ac95b656d1fda91c271d2a106460d19ad966fc7`, with a CUDA toolkit matching PyTorch, matching `torch-scatter`, and `numba`. Add the built checkout to `PYTHONPATH` and obtain the official `dpvo.pth`. Models/binaries are not vendored. `HF_HOME` chooses the model cache and `CROSS_TORCH_HUB` can select an existing XFeat/BoQ hub cache.

On the tested PyTorch 2.8/CUDA 12.8 setup, apply
[this compatibility patch](patches/dpvo-torch28-dispatch.patch) before building:

```bash
git -C /path/to/DPVO apply --check /path/to/CROSS/patches/dpvo-torch28-dispatch.patch
git -C /path/to/DPVO apply /path/to/CROSS/patches/dpvo-torch28-dispatch.patch
```

It replaces 42 deprecated tensor dispatch calls with `scalar_type()` in the
correlation and Lie-group extensions; optimizer equations and model weights are
unchanged. The patch contains MIT-licensed DPVO context; its notice is retained
in [DPVO-LICENSE](patches/DPVO-LICENSE). Our 5090 build used CUDA 12.8,
`TORCH_CUDA_ARCH_LIST=12.0`, Eigen 3.4.0, `yacs==0.1.8`, `ninja==1.11.1.4`,
`numba==0.61.2` and the matching `torch_scatter==2.1.2+pt28cu128`
[PyG wheel](https://data.pyg.org/whl/torch-2.8.0%2Bcu128.html).
Choose the CUDA architecture for the target GPU and rebuild there; the tested
5090 binary does not establish 4090 compatibility or performance.

```bash
# Fetch RGB and metadata; sensor-depth images are omitted during extraction.
python scripts/download_mono_benchmarks.py --root /path/to/benchmarks \
  --sequences freiburg1_desk freiburg1_xyz bonn_balloon2

python -m cross.mono.run /path/to/benchmarks/rgbd_dataset_freiburg1_desk \
  --frontend metric_pnp --mask-people --mask-interval 3 \
  --metric-interval 30 --mapping-interval 15 --retrieval-pose da3 \
  --output outputs/desk_mono --save-map

# DA3 baseline / ablation
python -m cross.mono.run /path/to/benchmarks/rgbd_dataset_freiburg1_desk \
  --frontend da3 --pose-refinement xfeat --refinement-anchor-only --metric-shape \
  --output outputs/desk_da3

# Ground truth is read only after inference.
python -m cross.mono.evaluate outputs/desk_mono/trajectory.txt \
  /path/to/benchmarks/rgbd_dataset_freiburg1_desk/groundtruth.txt \
  --output outputs/desk_mono/metrics.json

python scripts/summarize_mono_runs.py outputs --output outputs/summary.json
python -m pytest tests -q
```

The recommended profile above does not require DPVO or its CUDA extensions. The motion features and optional person detector are small; the metric teacher and retrieved-pair geometry run at lower rates. These calls are synchronous, so average throughput does not guarantee a fixed frame deadline.

The experimental streaming path moves metric inference and the existing CROSS mapping updates to bounded workers:

```bash
python -m cross.mono.run /path/to/benchmarks/rgbd_dataset_freiburg1_desk \
  --frontend streaming_pnp --mask-people --mask-interval 3 \
  --metric-interval 30 --mapping-interval 15 \
  --retrieval-pose metric_pnp --mapping-process --input-buffer 4 \
  --input-fps 20 --warmup-models --paced-input-worker \
  --output outputs/desk_streaming
```

The operating target is 20–30 FPS at 640×480 on one consumer GPU such as an RTX 4090. Shared RTX 5090 measurements are reported below; RTX 4090 performance remains unmeasured. Measure capture-to-pose latency and deadline misses as well as throughput; include initialization, mapping lag and combined GPU memory. `--warmup-models` uses only the first image and records its time separately. The paced input worker starts preprocessing after simulated capture time and fails on queue overflow (two frames by default, configurable with `--input-buffer`); it never silently drops evaluation frames. A bounded input queue does not itself bound capture-time lag when preprocessing falls behind. Check the actual input storage medium and GPU occupancy throughout timing runs: local rotational disks can still stall, and a GPU can become occupied after launch.

`--freeze-gc` performs a collection before capture starts and freezes startup objects, while leaving collection of new objects enabled. On the shared 5090, two counterbalanced repeats each of TUM room and walking reduced post-bootstrap maximum latency from 131–145 ms to 20.7–24.1 ms, with zero post-bootstrap 33 ms deadline misses and no input drops. Every run still missed its first-frame deadline. These short controls test the combined collection/freezing policy, not each operation separately or long-session guarantees. Scheduling can affect geometry: walking frontend error improved while mapped ATE changed from 3.87 cm to 4.06 cm.

The command above paces every selected image uniformly at 20 Hz. For playback at the original motion speed with approximately 20 images per second, add `--sample-fps 20 --replay-timestamps`. Sampling keeps the first available RGB image in each time bin; gaps remain gaps, timestamps and source indices remain unchanged, and the selected frame count is recorded. Arrival intervals can vary with the original capture cadence. `--input-fps 20` then specifies the 50 ms pose deadline while the recorded timestamps determine arrival times. Report this sampling protocol separately from all-frame throughput replay.

`--input-process` optionally moves decoding and undistortion into a separate CPU process. Its startup finishes before the capture clock starts and is recorded with initialization. Every selected image still becomes readable only after its capture time; the bounded queue fails on overflow. Capture-to-pose latency includes transfer to the inference process. Per-frame input diagnostics separate decoding, undistortion and color conversion, with wall time and reader-thread CPU time for each stage. Queue peaks are sampled in process mode; the queue itself enforces the configured bound. Process isolation must be benchmarked and does not by itself establish a frame deadline.

Teacher outputs retain their source image and timestamp. Mapping receives accumulated motion even if a pending low-rate image is replaced. Previously emitted poses are not rewritten. The mapping worker calls the inherited global observation-mixture, hypothesis filtering and delayed commitment code. `--mapping-process` optionally isolates that worker in a separate process on the same GPU; both processes' allocated-memory peaks are reported. This alone does not guarantee deadlines. The class-specific person detector batches its class postprocessing while retaining the pretrained scores and NMS.

`--trace-metric-sources` records content identities for the pinned metric teacher's RGB, calibration, weights and preprocessing. It carries signed source-scale sensitivities through delayed anchors and replaced mapping images, records the relative-motion Jacobians, and persists each keyframe's depth-source identity separately from coordinate charts. Missing provenance in older maps remains explicit. This diagnostic leaves inference unchanged; these input derivatives are not filtered pose/bias posteriors. The trace is a prerequisite for testing correlated metric priors, not an implemented scale-correction method.

`--conditional-sources --chart-aware --session-recovery` enables an experimental shared-source extension with `streaming_pnp` and `metric_pnp` retrieval. Each live CROSS hypothesis retains a joint pose/source belief. Saved nodes contain conditional pose messages; the committed bias belief is persisted once. Shared source responses are combined before marginalization, so a reused teacher prediction is not treated as a fresh independent metric prior. The declared per-prediction log-depth standard deviation is 0.12 (`--source-log-std`); it has not been empirically calibrated. This mode requires rebuilding reference maps, and uses synchronous PGO inside the mapping worker. Its dense source covariance has quadratic memory; long-map scalability is unvalidated.

`--schmidt-map-geometry` additionally retains shared map geometry after a committed
graph solve. A standard Schmidt update holds mature map means and their covariance
block fixed while updating the camera, metric biases and cross-covariances. The
original CROSS global message and delayed commitment tests are unchanged. This
experimental option requires conditional sources, a complete connected graph,
one deterministic gauge and one surviving mode at the graph refresh. Unsupported
refreshes raise an error before publishing the new state. Newly inserted node
residuals remain independent until the next refresh; the dense representation
does not bound long-map memory or mapping latency. Saved maps containing shared
geometry require this option when loaded. An existing conditional map can opt in
at its next commitment. These are established Gaussian and Schmidt operations,
not a novelty claim.

Saved node poses are conditional nominal poses. To evaluate a map, evaluate each
`ConditionalPose` at the map's persisted `SourceState` mean; the serialized pose
alone need not be its posterior mean. Keep the original reference alignment and
include final mapper commitments when assessing map correctness.

The conditional graph solve preserves source sensitivities through chart joins without a second bias update. It uses the final robust Gauss–Newton linearization and retains conditional node covariance rather than shrinking it on graph-edge reuse. General correlations among saved geometry/pixels are still approximated as independent given the declared sources. Global retrieval clustering, competing pose hypotheses and the delayed commitment tests remain the CROSS foundation. Improved robustness, calibrated uncertainty and publication novelty require further experiments; algebra and controlled graph tests alone do not establish them.

`--delayed-recovery` enables experimental reverse-PnP correction when delayed depth arrives. It is disabled by default because it produced large pose jumps in development. Asynchronous scheduling can change anchor timing and trajectories; report repeated timed runs.

`--teacher-lag-frames L` optionally holds a ready metric result until its source is at least `L` frames old. This tests sensitivity to delivery timing without blocking the pose stream. Late results are applied when available and their lateness is recorded; it is not a guarantee of deterministic tracking. The default is zero. Final shutdown drains remaining results for mapping, while already emitted poses stay unchanged.

`--adaptive-anchor` is an experimental streaming option, disabled by default. When a valid PnP estimate has fewer than 80 inliers (four times the acceptance minimum), it requests metric depth before tracking fails. Requests remain at least five input frames apart. A ready proactive result renews the anchor at its verified source pose, retaining that image's timestamp and features; it never rewrites emitted poses. This can increase teacher and mapping work compared with fixed cadence. Logs record proactive requests and actual worker counts. The option retains CROSS's existing observation mixture and commitment tests; it is not a joint scale/bias inference method.

`--stable-teacher-cadence` optionally fixes regular teacher requests to input-index intervals. An emergency can defer the next regular request until the five-frame cooldown ends, but cannot permanently shift later intervals. Configured intervals shorter than five frames retain their rate. There is no catch-up backlog; the existing worker still keeps one running and one replaceable pending image. This is a reference-construction sensitivity experiment, disabled by default. Compare actual teacher/map counts as well as accuracy: the rate bound is shared with the original policy, but individual request counts can differ.

Two OpenLORIS reference rebuilds per policy retained 16 views and equal request counts, but shared only four saved images between policies. Fixed-grid references reduced reference ATE from about 3.85 cm to 3.03 cm while reducing verified historical pairs for viewpoint (6 to 3) and illumination (2 to 0); neither query recovered. Object-change recovery remained successful, with post-commit error about 4.5 cm versus 5.2 cm. This mixed result does not support enabling the option by default. All 16 reference/query runs met their 50 ms deadline with GC freezing enabled.

The native-rate consumer profile processes every RGB frame with original timestamps:

```bash
python -m cross.mono.run /path/to/sequence \
  --frontend streaming_pnp --filter-mode full \
  --mask-people --mask-interval 3 --metric-interval 20 --mapping-interval 10 \
  --retrieval-pose metric_pnp --image-size 640 480 \
  --mapping-process --input-buffer 4 --input-fps 30 \
  --warmup-models --paced-input-worker --replay-timestamps \
  --output outputs/native_rate
```

Measured on a **shared RTX 5090**, source `f20bebc`, one native-rate attempt per sequence:

| Sequence | Frames emitted | Pose FPS | Metric ATE (cm) | Capture-to-pose p95 (ms) |
|---|---:|---:|---:|---:|
| TUM fr1/desk | 613 / 613 | 30.02 | 9.54 | 16.51 |
| TUM fr1/room | 1362 / 1362 | 30.01 | 28.18 | 16.35 |
| TUM fr3/walking_xyz | 859 / 859 | 29.72 | 4.70 | 14.99 |
| Bonn crowd2 | 895 / 895 | 29.83 | 6.05 | 13.91 |

All-frame statistics include held invalid poses. ATE uses rigid alignment, with raw Bonn GT translations. Initialization and final draining are recorded separately. Between 0.073% and 0.326% of frames miss the 33.33 ms deadline. These development runs use a shared device and do not establish RTX 4090 performance or hard real-time guarantees. All four apply zero loop closures, so these results assess monocular tracking and inherited observation fusion, not new topological robustness.

At original-time 20 Hz, two same-seed scheduling replays with `--adaptive-anchor` give ATE ranges of 9.15–9.16, 18.82–18.83, 5.388–5.389 and 7.646–7.649 cm on those sequences, with p95 13.71–18.08 ms and no 50 ms misses. The option increases teacher and mapping work. Crowd2's frontend error worsens despite better mapped error; this is not a uniform frontend improvement or a held-out evaluation.

To reproduce the earlier synchronous multi-seed suite on a remote server:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_mono.py \
  --data-root /path/to/benchmarks --output outputs/mono_suite \
  --sequences rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg1_xyz \
    rgbd_dataset_freiburg1_desk2 rgbd_dataset_freiburg1_room \
    rgbd_bonn_balloon2 rgbd_bonn_crowd2 \
    rgbd_dataset_freiburg2_xyz rgbd_bonn_balloon rgbd_bonn_crowd \
  --seeds 0 1 2
```

DPVO/scalar-only comparison: select `--frontend dpvo --dpvo-checkpoint /path/to/dpvo.pth`. `--dpvo-metric-bootstrap` is an experimental initialization option, not part of the recommended profile. `rotation_metric` tests DPVO rotation with metric-anchor translation. `--retrieval-pose metric_pnp` is an experimental bidirectional learned-depth verification adapter; it improves proposal availability and supports the same-sequence restart experiment; changed-session robustness remains unvalidated.

DPVO uses parallel floating-point accumulation. Repeating a real native BA
solve with byte-identical inputs on the 5090 produced different poses and
depths; equal seeds therefore do not ensure identical motion observations.
Report repeated runs and actual observation parity when comparing downstream
scale or mapping policies. The native constructor also sets PyTorch's CPU
thread count to two, overriding `--threads` for these frontends.

`--frontend streaming_pnp --rotation-tracker dpvo --dpvo-checkpoint /path/to/dpvo.pth`
enables an optional causal streaming comparison. DPVO supplies rotation; the
periodic metric teacher and calibrated image matches supply translation.
The feature extractor and DPVO share the current padded person mask. Delayed
reverse recovery uses the rotation recorded at the depth's source frame and
does not rewrite emitted poses. Existing metric-source identities and CROSS's
global message and delayed commitment are retained. This option is disabled by
default; its source-bias uncertainty remains an approximation, and synchronous
DPVO quality results do not establish this streaming path's performance.

`--frontend learned_rotation_pnp --frontend-only` tests DA3-Small rotation between the current image and the metric anchor, with translation fitted from calibrated matches and metric depth. It ignores the small model's translation and depth. This synchronous experimental frontend uses at most two images per prediction; it has no shared-source mapping integration or validated robustness advantage. Invalid or implausible learned rotations fall back to geometric PnP.

`--frontend-only` isolates motion/scale and does not require GTSAM. `--scale-mode` selects scalar-policy ablations for DA3/DPVO; metric-anchor PnP uses depth priors directly. `--stride` defaults to 1 (every RGB frame). Known TUM/Bonn sequence names select calibration and distortion; custom sequences require `--intrinsics fx fy cx cy` plus a TUM-format `rgb.txt`. DPVO currently requires image dimensions divisible by 16; select its physical GPU using `CUDA_VISIBLE_DEVICES`.

Each run records configuration, source fingerprint, model provenance, causal system/frontend trajectories, per-frame diagnostics and timing. Existing result directories are not overwritten. Failed PnP verification holds the pose and marks the frame invalid; rotation-prior frontends can still update orientation while holding translation. DPVO validity records initialization only; it is not an independent accuracy or tracking-success check. Evaluate all emitted poses and report the validity fraction separately. Similarity alignment can conceal metric-scale error, so report both ATE alignments and the fitted scale. For Bonn rotational RPE, also request `--groundtruth-frame bonn-camera`: its raw mocap orientation has a different body-frame convention. Keep raw and calibrated results separate. `cross.mono.replay_scale` enables causal scalar-policy ablations using exactly shared DPVO geometry and teacher observations, without assigning replay outputs inference timing.

OpenLORIS packages can be read directly from `color.txt` and the D400 color section of `sensors.yaml`. The monocular reader needs no depth, IMU, odometry, extrinsics or ground truth. It reads the package's `fx, cx, fy, cy` ordering, checked against the [authors' camera matrix](https://github.com/lifelong-robotic-vision/openloris-scene-tools/blob/ce6a4839f618bf036d3f3dbae14561bfc7413641/dataprocess/segway_transforms.py). Add `--image-size 640 480` to resize after undistortion; intrinsics are adjusted using OpenCV's pixel-centre convention. The original camera dimensions and both intrinsic matrices are recorded.

For saved-map tests, use `--save-map` on the reference run, then `--load-map /path/to/reference/map.pkl` in a fresh query run. Convert OpenLORIS ground truth separately to camera poses, retaining the shared world frame:

```bash
python -m cross.mono.openloris_groundtruth /path/to/office1-1 --output /path/to/evaluation/office1-1-camera.txt
python -m cross.mono.openloris_groundtruth /path/to/office1-2 --output /path/to/evaluation/office1-2-camera.txt
python -m cross.mono.evaluate_restart \
  /path/to/reference/trajectory.txt /path/to/evaluation/office1-1-camera.txt \
  /path/to/query/trajectory.txt /path/to/evaluation/office1-2-camera.txt \
  --diagnostics /path/to/query/diagnostics.jsonl \
  --mapping-events /path/to/query/mapping_events.json \
  --output /path/to/evaluation/restart.json
```

The restart evaluator fits one rigid transform using only the reference trajectory and applies it unchanged to the query. Held outputs remain included. It records the emitted pose receiving each applied commitment; pose correctness at that instant is a diagnostic, not a label proving a correct topological merge. Independently aligned query ATE cannot establish relocalization success.

`--mapping-events` additionally accounts for recorded commitments that never
reached a camera output, including results drained at shutdown. Their accuracy
remains unknown until the saved map is evaluated. Check the run summary's
`mapping_audit_complete` flag: accounting can only cover the supplied event log.
Missing, duplicated or inconsistent commitment records are rejected.

Two opt-in session experiments are available. `--session-recovery` replaces the distance guard for an unanchored historical-map candidate with direct historical support across the inherited evidence window and at least two reference keyframes. The existing overlap/confidence tests and delayed hypothesis realization still apply. Support from new query nodes, duplicate deliveries, and a reused hypothesis slot cannot substitute for this history. After a successful reference merge, the original distance guard resumes. This addresses the arbitrary chart displacement in that guard. Without the separate chart-aware option below, clustering and proposal alignment still compare numerical coordinates across sessions. `--historical-retrieval-slots 1` reserves one of the existing three candidates for saved-map keyframes without lowering retrieval thresholds or increasing the geometry budget. Both options default off and need broader aliasing evaluation. Retrieval diagnostics include source IDs, geometric rejection reasons and proposal assignments.

`--chart-aware --session-recovery` additionally retains a coordinate-chart ID for each node pose component and tracking hypothesis. Proposal clustering and prior matching only compare poses within the same chart; coincidence of unrelated chart origins cannot merge modes or create proximity edges. Legacy map charts are inferred from committed graph connectivity, and saved chart-aware maps require this option when loaded. An accepted graph join transports the affected charts and rejects stale asynchronous results. The global observation scores, mixture filter and delayed temporal tests are retained. This opt-in extension handles SE(3) chart provenance. Shared source inference is a separate experimental option described above; general gauge invariance remains unvalidated.

`--retrieval-matcher lighterglue` optionally uses the released XFeat LighterGlue model for `metric_pnp` retrieved pairs. Frame-rate motion keeps its original matcher. Both retrieval directions still pass the same PnP and cycle checks before entering CROSS's observation message. The adapter verifies the checkpoint hash and every learned matcher parameter. The checkpoint must be present in the XFeat torch-hub cache.

The experimental `--retrieval-pose metric_two_view --retrieval-matcher superpoint_lightglue`
keeps a verified original XFeat/LighterGlue metric-PnP proposal when available and
otherwise uses SuperPoint/LightGlue to estimate calibrated two-view rotation and
translation direction. Reference predicted depth sets translation
magnitude; current predicted depth screens compatibility. Each retrieved node still
contributes at most one proposal to CROSS's existing global observation mixture.
Competing hypotheses and delayed commitment remain in the existing mapper. The
conditional-source mode carries the same identified reference-depth bias through
this proposal and deduplicates exact geometric factors.

Install its optional upstream implementation with:

```bash
pip install 'git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8'
```

Released SuperPoint/LightGlue checkpoints are downloaded to the Torch hub cache
and verified by full SHA256 before loading. Upstream code and weights retain their
respective licenses. The frame-rate tracker keeps XFeat. This is an experimental
geometry baseline: its compatibility thresholds and inherited geometric covariance
floors are not calibrated, and a successful pair fit is not a place commitment.
It is disabled by default; pair-level results alone do not establish robustness,
real-time performance, or a novel contribution.

In a development test of this fallback on an RTX 5090, two repeated 640×480,
20 Hz runs per OpenLORIS change condition each committed to the saved office1-1
map correctly. Both original metric-PnP controls failed viewpoint and illumination
recovery. All 3,800 fallback outputs met the 50 ms deadline: capture-to-output
p95 was 17.50–18.57 ms, maximum 36.97 ms, with no dropped input. Model loading
and background shutdown are excluded. Summed process peak allocated GPU memory
was 3.24–3.28 GB, excluding contexts and reservations.

| Changed session | Received commitment delay | Post-commit position RMSE |
|---|---:|---:|
| office1-2, viewpoint | 6.50 s | 37.63–37.64 cm |
| office1-4, illumination | 11.38 s | 27.39–27.41 cm |
| office1-6, objects | 5.47 s | 5.26–5.27 cm |

These use one rigid alignment fitted to the reference only; no query alignment
or scale fit. Whole-query RMSE, including the unanchored prefix, is respectively
2.48, 3.38 and 0.90 m. The difficult sessions still drift after a correct
commitment. This small development cohort does not establish held-out robustness,
calibration, a benefit from source-bias inference, or RTX 4090 performance.

A subsequent frozen test used three previously unused sequences in the same
office, with two scheduling repeats per policy and the same saved reference.
Both policies recovered office1-3 (turn-around) and office1-5 (lighting); neither
recovered office1-7 (moving people). The fallback worsened post-commit position
RMSE on office1-3 from 8.65 cm to 22.69–22.76 cm, and changed office1-5 from
6.16 cm to 6.39–6.40 cm. These errors use reference-only alignment. A wrong
two-view proposal pulled an established branch despite successful initial
recovery. All 8,240 outputs met the 50 ms deadline with no dropped input;
capture-to-output p95 was 14.96–18.87 ms and maximum 35.75 ms on the RTX 5090.
The fallback remains optional: broader pair acceptance did not improve recovery
in this cohort. The sequences become development data for subsequent changes.

`--two-view-rotation-check` adds an experimental agreement screen to this
fallback using the configured pose model (DA3-Small by default). The learned
and two-view rotations must agree within the existing 0.1 rad cycle threshold.
Accepted poses, confidences, metric-source identities and delayed-commitment
tests remain the same. The screen runs only after primary PnP fails and a
two-view pose passes verification. It can reject useful matches as well as
errors; its effect on recovery must be measured. It is disabled by default.

In two development repeats on all six office queries, this screen preserved
five recoveries and the failure on office1-7. On identical post-recovery time
intervals, viewpoint RMSE fell from 37.67–37.68 to 23.73 cm and turn-around
from 22.76 to 7.54–7.61 cm. Harder lighting worsened from 29.37–29.39 to
40.44 cm, with recovery delayed from 11.38 to 14.35–14.42 s. The other lighting
and object cases remained close to their controls. All 7,920 outputs met the
50 ms deadline on the RTX 5090, with p95 16.09–19.03 ms and maximum 36.26 ms.
This tradeoff does not support a uniform robustness claim or default adoption.

Loading the accumulated office1-1→1-2 map lets office1-7 recover in both
repeats at 7.67 s, with 16.24–16.25 cm post-recovery RMSE. Evaluation keeps
the original office1-1-only rigid alignment; it never fits the added session
or query. Nearby original views face the opposite direction, so spatial
coverage alone does not establish usable visual overlap. Map augmentation
changes geometry and image coverage together.

Two longer chains through office1-3/4/5/6/7 recover all five queries initially,
but both commit a false association late in office1-7. That correction returns
after the last camera output: the live post-recovery RMSE is 21.34 cm, while
the saved map contains 65 incorrect poses among 184 ground-truth-associated
nodes under the 1 m/30° diagnostic (one node is unassociated). Maximum saved
node error is 2.45 m. Evaluate final maps and drained updates as well as live
trajectories; the long-chain maps are unsuitable as robust reference maps.

All 6,720 camera outputs in those ten queries meet the 50 ms deadline without
input loss: p95 17.28–19.24 ms, maximum 35.94 ms, on the shared RTX 5090.
However, maximum mapper turnaround grows from about 0.64 to 3.97 s. The
latest-only mapper replaces 34 pending low-rate snapshots, retaining their
accumulated motion; final draining/shutdown takes up to 6.25 s. Peak allocated
GPU memory summed across processes is 3.21–3.72 GB, excluding contexts and
reserved memory. These results establish a fast local pose stream, not a
20 Hz global mapper or a long-map latency bound.

Repeated graph joins also exposed quaternion roundoff and partial publication
on a failed conversion. Near-unit rotations are now normalized in double
precision before graph conversion, and all converted updates are validated
before publication. Both regression tests fail before this fix; all 129 CPU
tests pass afterward. No geometric or commitment threshold was relaxed.
A private control using independent graph marginals reduces the late alias's
evidence advantage but still makes the false merge in both repeats. Retaining
joint map geometry rejects that alias, but initially worsens short-map trajectory
RMSE from 16.25 cm to 42.50 cm. The standard Schmidt control brings this back to
16.55 cm while retaining alias rejection. These contrasting controls motivated
the optional shared-geometry implementation described above.

Public source `f775418`, with `--schmidt-map-geometry`, was tested twice on each
of three fixed office1-7 input maps. Each run recovers correctly and avoids the
late false merge, including final mapper updates. Saved-map means use the
persisted source posterior and the original reference-only rigid alignment.

| Input map | Received recovery | Post-recovery position RMSE | Maximum saved-node error |
|---|---:|---:|---:|
| Short, sessions 1→2 | 7.67 s | 16.54–16.55 cm | 28.83 cm |
| Long, sessions 1→2→3→4→5→6 | 9.77 s | 22.74–22.80 cm | 53.25 cm |
| Persisted Schmidt chain | 10.71–10.81 s | 19.48–19.59 cm | 51.83 cm |

All 4,560 selected outputs meet the 50 ms deadline without input loss, with
capture-to-pose p95 18.05–18.98 ms on the shared RTX 5090. The inherited
observation mixture and delayed thresholds are unchanged. Avoiding repeated
validation of unchanged covariance reduces committed graph service on the
persisted map from 10.20 to 3.61–3.71 s; the mapper still runs below camera
rate. Its changed schedule affects which low-rate images are processed.
The 158 CPU tests pass, including independent Gaussian checks, persistence,
chart joins and mutation isolation. These are development controls with dense
state, uncalibrated priors and no RTX 4090 measurement. Generic shared covariance,
Schmidt conditioning and these implementation fixes are established techniques;
a publication contribution requires further formulation and validation.

`--historical-min-score 0` with reserved historical slots permits a bounded number of weaker saved-map candidates, while new query nodes retain the original score thresholds. Original retrieval scores remain available to the inherited uncertainty calculation and keyframe policy; no geometry or temporal-commitment test is relaxed. This experimental search option can spend the entire retrieval budget on historical views if all three slots are reserved. Both search and learned matching need false-association and online runtime evaluation; accepted pairs alone do not establish map recovery.

After a new keyframe realizes a temporally supported branch, commitment is checked once more with the same evidence and completed graph edges. This corrects an ordering issue that otherwise requires a later successful retrieval just to examine the newly realized branch; evidence is not counted twice and thresholds are unchanged. With the older saved reference, shared 5090 tests reconnect the reversed-viewpoint query at 12.21 s with 23.1 cm post-commit fixed-reference RMSE, and the object-change query at 11.17 s with 7.9 cm afterward. Rebuilt references at source `b7ea914` recover object change in both modes (baseline 5.20 s / 5.50 cm; conditional 5.47 s / 5.25 cm), but neither recovers viewpoint or illumination. This exposes reference-construction sensitivity; the small error difference does not establish a conditional-inference benefit. Each query uses its own reference's rigid alignment, with no query fit. Errors before recovery and all failures remain part of evaluation; systematic false-association testing remains outstanding.

Monocular code lives in `cross/mono/`; inherited mapping is in `cross/core/`. Private research notes and runs belong in ignored `docs/` and `outputs/`. External models retain their own licenses. This implementation is informed by [AMB3R-SLAM](https://arxiv.org/abs/2609.19518); it is not a reproduction of that paper.

The remaining instructions document the inherited CROSS RGB-D interface.

## Installation

### Quick Start (recommended)

```bash
# From this checkout:
bash install.sh
```

The install script handles everything: creates a virtual environment via [uv](https://docs.astral.sh/uv/), installs PyTorch with CUDA, installs CROSS and its dependencies, and builds GTSAM.

<details>
<summary><strong>Install options</strong></summary>

```bash
# CPU-only (no CUDA)
bash install.sh --cpu

# Specific CUDA version, it should match the cuda version on your PC, i.e. nvcc --version
CUDA_VERSION=cu121 bash install.sh
```
</details>

### Manual Installation

```bash
# 1. Create and activate environment
uv venv --python 3.11
source .venv/bin/activate

# 2. Install PyTorch (match your CUDA version — check with nvcc --version)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install CROSS
uv pip install -e ".[all]"

# 4. Install GTSAM (required for pose graph optimization)
git clone --depth 1 https://github.com/borglab/gtsam.git thirdparty/gtsam
cd thirdparty/gtsam
uv pip install -r python/dev_requirements.txt
mkdir -p build && cd build
cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.11 -GNinja
ninja python-install
cd ../../..
```

### Optional Dependencies

```bash
# Object detection (semantic memory)
uv pip install -e ".[detection]"

# R3D recording support
uv pip install -e ".[recording]"
```

## Usage

### Run mapping on a dataset

```bash
uv run python run.py data/r3d/lab2.r3d
```

Options:

| Flag | Description |
|------|-------------|
| `--no-viz` | Disable Rerun visualization |
| `--frames N` | Process only the first N frames |
| `--start N` | Start from frame N |
| `--loader {r3d,rosbag,loris,tum}` | Force dataset loader (default: auto-detect) |
| `--snr FLOAT` | Signal-to-noise ratio for R3D datasets |
| `--async` | Enable async step pipeline |

### Interactive demo

```bash
python examples/demo.py
```

Provides a REPL with commands: `l` (load sequence), `s` (set start), `g` (go/run), `p` (plan path), `v` (visualize graph), `q` (quit).

### Multi-session mapping

```bash
uv run python examples/multi_session.py
```

### Save, load, and plan

```bash
uv run python examples/planner.py --map-scene data/r3d/lab_obj.r3d --reloc-scene data/rosbag/lab_office_dog
```

## Datasets

### OpenLORIS

Download the [TUM version](https://lifelong-robotic-vision.github.io/dataset/scene.html) and place it in `data/loris/`.

### R3D

Place `.r3d` recordings in `data/r3d/`.

### ROS Bags

Place processed ROS bag directories in `data/rosbag/`.

## Project Structure

| Module | Description |
|--------|-------------|
| `cross.core.system` | Main pipeline: motion prior, observation, GMM filtering, keyframe insertion |
| `cross.core.hypothesis` | Multi-hypothesis GMM belief, evidence tracking, loop-closure detection |
| `cross.core.pgo` | Pose graph construction and GTSAM optimization |
| `cross.core.lc_engine` | Asynchronous loop-closure engine (background thread) |
| `cross.core.simple_topo` | Topological planning graph with proximity edges |
| `cross.core.planner` | A\*/Dijkstra path planning over sparse graph |
| `cross.core.mem` | Text-conditioned semantic memory search |
| `cross.db.db` | Keyframe database with embedding-based VPR |
| `cross.cv.pose_est_pnp` | PnP-based relative pose estimation |
| `cross.visualization.viz_rr` | Rerun-based 3D visualization |

## Citation

If you use CROSS in your research, please cite:

```bibtex
@inproceedings{wang2026cross,
  title     = {Change-Robust Online Topological Memory for Long-Term
               Relocalization and Semantic Navigation},
  author    = {Wang, Jiaming and Chen, Jizhuo and Liu, Diwen and
               Ghotavadekar, Atharva and Da, Jiaxuan and K{\"a}stner, Linh
               and Soh, Harold},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  eprint    = {2605.02227},
  archivePrefix = {arXiv}
}
```

## License

[MIT](LICENSE)
