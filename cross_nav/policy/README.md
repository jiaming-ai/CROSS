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

## Closed-loop navigation and robot deployment

Two navigation systems share one robot interface (`cross_nav/policy/robot.py`) and are evaluated in the same closed loop
in Habitat (`cross_nav/policy/sim/`, HSSD scenes, point goals 1.5-20 m, success within 0.2 m):

- **Trajectory policy** (`TrajectoryPlanner`): this direct head, trained on expert, DAgger and real-robot data, with an
  odometry trail (past positions) as memory and stuck recovery (`recovery.py`, on by default).
- **BEV + planner** (`BEVPlanner`): the BEV occupancy and observability heads, an episode map in the odometry frame
  (fusion + A*) and the closed-loop rules in `configs/policy/bev_planner_best.json` (`bev_planner.json` is the
  frozen planner without them).

```python
from cross_nav.policy.robot import NavigationAgent, TrajectoryPlanner, BEVPlanner

planner = TrajectoryPlanner(head_checkpoint, backbone_checkpoint)          # or BEVPlanner("configs/policy/bev_planner_best.json")
agent = NavigationAgent(planner)                # recovery='auto': stuck recovery for the trajectory planner only
agent.set_goal(goal_xy)                         # odometry frame, metres
for rgb, pose, t in camera_and_odometry():      # pose (x, y, yaw), REP-103; t in seconds
    out = agent.step(rgb, pose, t)
    if out['arrived']:
        break
    publish(out['cmd'])                         # (v m/s, omega rad/s); out['path_odom'] for display
```

Requirements and behaviour on a robot:

- **Camera**: RGB, 512x384, about 90 degrees horizontal FoV, level, about 0.8 m high: the test camera. Other mounts
  (height 0.35-1.2 m, FoV 70-110 degrees, pitch -15..5 degrees) cost the trajectory policy about 5 points of success in
  simulation; validate a new mount before relying on it. The BEV config assumes a level camera (`gravity: imu`).
- **Odometry**: wheel or visual odometry in a fixed frame. It selects the history views by travelled distance, gives the
  goal in the robot frame, the trajectory policy's trail and the BEV map. Frames are stored every 0.1 m or 10 degrees.
- **Time stamps**: pass them to `step`. The stuck recovery then fires after 10 s with less than 0.3 m of displacement
  (in-place 90 degree turn, alternating sides), and the BEV bump rule compares commanded with measured motion
  (without it the BEV planner loses about 20 points in simulation).
- **Rate**: validated at 10 Hz with v_max 0.5 m/s (`rollout --control robot`). For a planner slower than the camera,
  use `replan_every`; between plans the agent follows the last path in the odometry frame.
- **Safety**: the agent is a local planner. It does not check collisions itself and touches obstacles in simulation
  (see contacts per episode in the report). Keep the robot's own safety layer: bumpers or a depth/lidar stop, an
  emergency stop and speed limits.

Simulation tools:

```bash
# Episodes, expert or policy rollouts against a planner server (one server per GPU, many rollout workers)
python -m cross_nav.policy.server --socket "$SOCK" --backbone "$VGGT_CHECKPOINT" --head "$HEAD" --scale odometry
python -m cross_nav.policy.server --socket "$SOCK" --backbone "$VGGT_CHECKPOINT" --bev configs/policy/bev_planner_best.json
python -m cross_nav.policy.sim.rollout --hssd "$SCENES" --episodes "$EPISODES" --socket "$SOCK" --mode policy \
  --out "$RUN" --camera nominal --max-steps 500 --viz-dir "$RUN/viz" --viz-every 5   # --control robot --stamps: the agent loop
# Side-by-side viewer of saved runs
python -m cross_nav.policy.sim.viewer --runs "trajectory policy=$RUN_A" "BEV + planner=$RUN_B" --out "$VIEW"
```

`--hssd` takes an HSSD (`hssd-hab`) or AI2-THOR (`ai2thor-hab`) scene root; the dataset is detected from the root.
Training data for the trajectory policy comes from `--mode expert` and `--mode dagger` rollouts with `--record`, and from
real-robot recordings (`real_cache.py`); `train_sim.py` trains on the mixture.
