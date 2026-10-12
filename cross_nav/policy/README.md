# Direct trajectory expert

This experimental head cross-attends to frozen fine-tuned VGGT tokens and predicts a local geometric route. The current experiment uses **16 waypoints at 0.5 m spacing (8 m path horizon)**. Waypoints are metres in `(forward, left)` coordinates relative to the current robot heading; distance is travelled path length, not a promise of 8 m straight-line displacement. The head does not predict execution times or motor commands.

Inputs are a metric local goal and four current/past RGB views, selected at approximately 0/0.5/1/2 m past travel. Use estimated CROSS poses to select history on a robot; offline experiments use recorded past poses. No future image is an input. Images are aspect-preserving resized and letterboxed, including upsampling small GNM images. Prediction quality at other resolutions or history policies requires separate verification.

The policy supports last-layer or four-layer visual memory, pooled or original last-layer patch tokens, an optional learned compression module, direct regression and rectified flow with 10 Euler steps. Metric labels alone do not guarantee calibrated scale.

Scale modes:

- `none`: learns metric output from visual tokens and the metric goal.
- `scalar`: also conditions visual memory on the frozen dense scale-head estimate.
- `metric`: backprojects frozen predicted depth into camera0, multiplies those coordinates by the scale estimate, and embeds these metric positions into visual memory before cross-attention. It still predicts a path directly and does not construct a BEV.

`log_scale` is **log(metres per VGGT geometry unit)**. If CROSS already has a scale from stereo, odometry or metric map anchors, pass that value in place of the monocular estimate. Do not multiply the final path by scale: it is already in metres. `odometry_log_scale` estimates the multiplier from corresponding past camera-centre baselines; account for sensor extrinsics and reject degenerate motion. A scale head is an estimate, not a guarantee. The experiments audit its agreement with past metric motion and the route's response to scale perturbations.

```python
import torch
from cross_nav.policy.infer import TrajectoryPredictor

policy = TrajectoryPredictor(head_checkpoint, backbone_checkpoint)
# images: B,4,3,H,W at checkpoint resolution (main experiment 512px); goal: B,2, metres forward/left
waypoints = policy(images, goal)
# Optional trusted CROSS scale, one log multiplier per window:
waypoints = policy(images, goal, log_scale=cross_log_scale)

# Reuse an existing pass instead of rerunning the frozen backbone:
policy = TrajectoryPredictor(head_checkpoint)
waypoints = policy.from_backbone(backbone_outputs, goal, log_scale=cross_log_scale)
```

Train/cache paths are supplied as arguments; datasets and checkpoints do not belong in git:

```bash
python -m cross_nav.policy.data --cache "$SCENE_CACHE" --bev-root "$BEV_LABELS" \
  --dataset nclt --horizon 16 --out "$RUN/nclt.json"
python -m cross_nav.policy.cache --manifest "$RUN/nclt.json" \
  --checkpoint "$VGGT_CHECKPOINT" --size 512 --metric --out "$RUN/cache/nclt"
python -m cross_nav.policy.train --cache "$RUN/cache/nclt" \
  --horizon 16 --geometry metric --source last --steps 6000 --out "$RUN/model"
```

For GNM/Open-X data, `prepare_gnm` consumes public TFRecords whose checksums match a downloaded storage inventory; `data --gnm-root ...` creates the policy manifest. Recordings/days are assigned wholly to one split. **All RECON is test-only**, enforced by both the manifest builder and training cache loader. Checkpoints are selected on development macro ADE; final test must not guide tuning. Cache metadata binds backbone and manifest hashes. The earlier four-metre pilot has a separate manifest and should not be compared directly to eight-metre scores.

Offline evaluation reports ADE/FDE, 5 m waypoint error, source breakdowns, visual shuffling, goal-only controls, scale perturbations and odometry-aided scale. Flow uses a fixed single random sample as primary output, with additional seeds reported separately. `analyze` supplies paired recording-cluster intervals and curved-route slices. These imitation metrics do not establish collision avoidance or closed-loop goal completion; a robot still needs its controller, stopping logic and safety checks.
