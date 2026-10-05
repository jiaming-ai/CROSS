# Using CROSS

A quick guide: pick a mode, run a sequence, change the configuration, map and relocalize across sessions, and run
the benchmark. Installation is in the [README](../README.md#installation).

## 1. Modes

A run is two independent choices:

| `--mode` | what the system sees | relative pose of a retrieved keyframe | needs |
|---|---|---|---|
| `rgbd` (default) | colour + depth | XFeat + LightGlue + PnP-RANSAC on keyframe depth | `install.sh` |
| `stereo` | left + right image | VGGT-Omega (one forward pass), metric scale from the stereo baseline | `install.sh --stereo` |
| `mono` | colour only | metric two-view matching, Depth Anything 3 as fallback | `install.sh --mono` |
| `mono --mono-estimator ff` | colour only | VGGT-Omega as in the stereo mode, metric scale from the motion and the map | `install.sh --mono --stereo` |

| `--odometry` | motion between frames |
|---|---|
| `external` (default) | the dataset's odometry (wheel, OXTS, or simulated from ground truth with `--snr`) |
| `visual` | DPVO from the images; metric scale from the mode's depth (sensor, stereo, or learned for mono) |
| `vio` (mono mode) | DPVO with its metric scale from the IMU (visual-inertial; learned depth as a prior) |
| `vgio` (mono mode with `--mono-estimator ff`, stereo mode) | no DPVO: VGGT-Omega relative poses and the IMU in a local pose graph; metric scale from learned depth (mono) or the stereo pair (stereo) |

All modes run with `external` and `visual`; `vio` is for the mono mode, `vgio` for the mono (`ff`) and stereo modes. Visual odometry needs DPVO (`install.sh
--mono`) and its weights in `models/dpvo.pth` (or `--dpvo-checkpoint`, env `CROSS_DPVO_CHECKPOINT`).

**External odometry from a stereo VIO.** A prepared folder can hold several odometry files; `--odom-file NAME`
(`scripts/map_and_reloc*.py`) reads `NAME` instead of `odom_left.txt` when it exists (else `odom_left.txt`, else the
simulated odometry). `benchmark/datasets/prepare_vio.py` writes `odom_vio.txt`: Basalt stereo-inertial odometry run
causally on the folder's stereo pair and IMU (build with `scripts/vio/install_basalt.sh`, then `export BASALT_VIO=...`):

```bash
python benchmark/datasets/prepare_vio.py kitti $BENCH_DATA/kitti --raw /path/kitti_raw --extract /path/kitti_extract
python benchmark/datasets/prepare_vio.py rover $BENCH_DATA/rover --raw /path/rover
python benchmark/datasets/prepare_vio.py simchange $BENCH_DATA/simchange --baseline 0.3
python benchmark/datasets/eval_odometry.py $BENCH_DATA/kitti/*/stereo      # accuracy of each odometry source
```

In the benchmark, `--odom-source vio` (`benchmark/run.py`, `benchmark/jobs.py`) runs any system that takes odometry
(CROSS RGB-D / stereo / mono with external odometry, RTAB-Map) on `odom_vio.txt` instead of the dataset's odometry
(wheel odometry where there is no `odom_vio.txt`); results go to `<system>+vio/` and appear as rows of their own:

```bash
python benchmark/jobs.py --dataset kitti --systems cross_stereo cross_rgbd cross_mono_ff_odom rtabmap --odom-source vio > q.txt
```

**Mono back ends.** `--mono-estimator da3` (default) is the monocular system of `cross/mono/` (learned metric depth,
two-view matching, DA3 fallback). `--mono-estimator ff` uses the stereo mode's VGGT-Omega multi-view estimator on the
single image; the metric scale of each forward pass comes from the previous observation and the (metric) odometry
between the two, and from pairs of retrieved keyframes whose relative pose the map knows.

**Visual-inertial odometry (`--odometry vio`).** The IMU must be rigidly attached to the camera. A sequence folder
carries it as `imu.txt` (`t wx wy wz ax ay az`, IMU frame, rad/s and m/s², specific force including gravity) and
`imu.json` (`T_cam_imu`: the IMU's pose in the camera frame; noise densities; optional `frame_times`: the file of the
image timestamps on the IMU clock, default `times.txt`), see `cross/dataloader/imu.py`; `benchmark/datasets/prepare_imu.py`
writes them for the benchmark datasets. DPVO tracks the images; a sliding-window estimate (`cross/imu/scale_filter.py`)
finds the metric scale of DPVO's trajectory from the preintegrated IMU (velocity, gravity and accelerometer bias as
unknowns, in a gyro-propagated frame so that DPVO's rotation drift does not tilt gravity), with learned depth (Depth
Anything 3) as a prior; the gyro bias and the camera-IMU time offset are calibrated online against DPVO's rotations.
The back end receives the metric DPVO motion as odometry, as with external odometry. Settings: `--mono-args "--imu-config
key=value"` (fields of `ImuConfig`).

**VGGT-inertial odometry (`--odometry vgio`).** No DPVO: the IMU carries the pose between frames, and every third
frame a VGGT-Omega forward pass (the back end's own pass when it observes then) measures the relative poses of the
current frame, the last measured frame and a keyframe ~2 s older. A sliding-window pose graph (`cross/imu/vgi_graph.py`)
optimizes these relative poses, the preintegrated IMU, gauge links between passes, tracked-corner rotations and the
online calibration (gyro and accelerometer biases, gravity, the passes' rotation scale, the camera-IMU time offset).
Each pass has a scale of its own: in the mono mode learned metric depth (DA3) observes it, with a bias state; in the
stereo mode (`--mode stereo --odometry vgio`) the stereo pair does, through classical stereo depth (SGBM) of the current
pair against the pass's depth map (no learned depth; `--mono-args "--imu-config vgio_stereo_source=baseline"` uses the
right image as one more view of the pass instead). The stereo mode needs the IMU next to the stereo folder
(`prepare_imu.py <dataset> ... --setup stereo` for OpenLORIS and ROVER, whose stereo pair is the T265 with its own IMU).

## 2. Run one sequence

```bash
python run.py data/r3d/lab2.r3d                                          # RGB-D, external odometry
python run.py data/kitti_raw/2011_09_30/2011_09_30_drive_0027_sync --mode stereo --config configs/outdoor.yaml
python run.py data/sim/lonemonk/map --mode stereo --baseline 0.3         # SimChange: which rendered baseline
python run.py data/posed/home1-1 --loader posed --odometry visual        # RGB-D, no odometry input
python run.py data/posed/home1-1 --loader posed --mode mono --odometry visual
python run.py $BENCH_DATA/openloris/home1-1/rgbd --loader posed --mode mono --mono-estimator ff --odometry vio
python run.py $BENCH_DATA/kitti/07/stereo --mode stereo --odometry vgio --config configs/outdoor.yaml   # stereo + IMU
```

Useful flags: `--no-viz` (no Rerun window; use it on a server), `--frames N`, `--start N`, `--loader` (default:
auto-detect), `--snr` (simulated odometry noise). `python run.py -h` lists all.

**Data formats.** R3D, ROS bag, OpenLORIS, TUM, *posed folders* (`rgb/`, `depth/`, `poses_left.txt`, optional
`odom_left.txt`, `calib.json`), and stereo sequences (KITTI raw, TartanAir V2, Virtual KITTI 2, SimChange). Converters
are in `scripts/datasets/` and `benchmark/datasets/`.

## 3. Configuration

Defaults are in `configs/default.yaml`; every other config is a layer on top of it.

```bash
python run.py <seq> --config configs/stereo.yaml configs/outdoor.yaml configs/noise/my_robot.yaml   # merged left to right
python scripts/map_and_reloc_rgbd.py ... --set mapping.loop_closure.noise_file=configs/noise/my_robot.yaml  pose_est.obs_max_interval_steps=2
```

| file | use |
|---|---|
| `configs/stereo.yaml` | stereo mode preset (loaded by `--mode stereo`): VGGT-Omega, observation gating |
| `configs/outdoor.yaml` | outdoor scale (clustering and hypothesis-alignment radius); `outdoor_noown.yaml` is its variant for the same scale |
| `configs/noise/*.yaml` | calibrated noise model for one robot (see below) |
| `configs/mono_*.json` | mono profiles: the arguments of `python -m cross.mono.run`. `mono_benchmark_10hz.json` for offline runs at 10 Hz, `mono_streaming_dpvo_v2_20hz.json` for real time; select with `--mono-profile`, extra arguments with `--mono-args` |

Override one value with `--set section.key=value` (the `scripts/` runners; `run.py` has no `--set`) or put it in a small
YAML file and pass it with `--config`. Some options (charts, session recovery, historical retrieval slots, ...) are off by default and are
enabled by the mono profiles.

**Camera mounting and the vertical.** Relocalization proposals are clustered, and matched to hypotheses, in place
coordinates: position on the horizontal plane plus heading (`mapping.projection`). The default assumes a
forward-looking camera on a ground robot, so the vertical is the camera's y axis in the first frame and height is
ignored. For other robots:
- tilted camera or unknown mounting: `mapping.projection.estimate_vertical=true` estimates the vertical from the axis
  the keyframes turn about.
- down-looking camera (e.g. an AUV survey camera): `mapping.projection.vertical=z`.
- places stacked vertically (multi-floor buildings, 3D terrain): `vertical_weight` (0–1) adds the weighted height to
  the clustering coordinates, and `vertical_gate` (metres) keeps proposals apart when their heights differ by more than
  the gate.

**Noise calibration for a new robot** (optional, no ground truth needed; the defaults suit wheeled indoor robots):

```bash
python scripts/map_and_reloc_rgbd.py --map <seq> --query <seq> --out outputs/calib --map-end 600 --skip-reloc --dump-graph
python scripts/lc/calibrate_noise.py --graph outputs/calib/graph_s0.json --out configs/noise/my_robot.yaml
```

## 4. Multi-session: map one session, relocalize in another

Mapping and relocalization are two steps: build and save a map, then load it and localize a later session against it.
The two harnesses do both and score the result (map ATE; relocalization success over independent 100-frame trials that
start without a pose; a trial succeeds when the final estimate is within `--r-d` of the pose the map implies):

```bash
# RGB-D / mono on posed folders
python scripts/map_and_reloc_rgbd.py --map data/posed/home1-1 --query data/posed/home1-2 --out outputs/home
python scripts/map_and_reloc_rgbd.py --map ... --query ... --out outputs/home_mono --mode mono --odometry visual
# stereo mode (VGGT-Omega); --estimator pnp --pnp-depth gt|sgbm gives the PnP baseline on the same data
python scripts/map_and_reloc.py --map data/sim/hssd_house/map --query data/sim/hssd_house/light_night \
    --out outputs/house --estimator ff --baseline 0.3 --snr 10 --trial-len 100 --trial-stride 50
```

The map is written to `--out` (`map.pkl`, `map_meta.json`); `--skip-map` reuses it for another query and `--skip-reloc`
only maps. Another query in the same scene is another call with the same `--out`/`--skip-map`. In code, use
`System.save_map(path)` / `System.load_map(path)`; `examples/multi_session.py` runs several sequences through one
`System`, and `examples/planner.py` loads a map and plans paths on it.

Stereo-mode quick-run options: `--obs-min-translation/--obs-min-rotation/--obs-max-interval` (observation gating),
`--max-refs`, `--n-ref-anchors`. Without a right camera pass `--set pose_est.ff.use_odom_anchor=true ...` (see the README
OpenLORIS example).

## 5. Benchmark

[`benchmark/`](../benchmark/README.md) runs the fixed protocol ([PROTOCOL.md](../benchmark/PROTOCOL.md)):
**T1** mapping ATE, **T2** multi-session localization recall, **T3** relocalization success, on KITTI,
OpenLORIS-Scene, ROVER and SimChange, in RGB-D, stereo and mono setups. Results so far:
[RESULTS.md](../benchmark/RESULTS.md) and the interactive page `benchmark/site/index.html` (work in progress).

### Choosing systems (including baselines)

A **system** is an entry of `benchmark/configs/systems.yaml`: its runner, setups, and extra arguments. Pass the names to
`jobs.py --systems`:

| system | what it is |
|---|---|
| `cross_rgbd`, `cross_rgbd_vo` | CROSS RGB-D mode, external / visual odometry |
| `cross_stereo`, `cross_stereo_vo` | CROSS stereo mode, external / visual odometry |
| `cross_mono_odom`, `cross_mono` | CROSS mono mode, external / visual odometry |
| `cross_mono_vio` | CROSS mono mode, visual-inertial odometry (IMU) |
| `cross_mono_ff_odom`, `cross_mono_ff`, `cross_mono_ff_vio` | CROSS mono mode with VGGT-Omega (`--mono-estimator ff`), external / visual / visual-inertial odometry |
| `orbslam3`, `rtabmap` | baselines (drivers in `scripts/baselines/`, built separately) |
| `mast3r_slam`, `vggt_slam` | baselines without map persistence (map + query run as one stream) |

To add one, append an entry to `systems.yaml` (runner, `setups`, `uses_odometry`, `map_reuse`, `args`) and, for a new
runner, write a function in `benchmark/run.py` that produces the same `result.json` ([schema](../benchmark/eval/schema.md)).

### Run

```bash
export BENCH_DATA=/path/to/benchmark_data BENCH_RESULTS=/path/to/runs
bash benchmark/datasets/download_openloris.sh /path/to/openloris                          # once per dataset
python benchmark/datasets/prepare_openloris.py /path/to/openloris $BENCH_DATA/openloris
python benchmark/jobs.py --dataset openloris --systems cross_rgbd cross_stereo orbslam3 > queue.txt
#   restrict with --setups rgbd stereo mono, --scenes office home, --tracks ...
while read -r job; do python benchmark/run.py $job; done < queue.txt                      # or split the lines over workers / GPUs
python benchmark/collect.py $BENCH_RESULTS && python benchmark/make_tables.py && python benchmark/build_site.py
```

- Each line of `queue.txt` is one job (`--task map | t1 | query`). Maps come first; a query job builds a missing map itself,
  with a lock so parallel workers share it. Exit code 3 means the job is not ready (inputs missing, or another worker is
  building its map): retry it later. Finished jobs are skipped; `--force` redoes one.
- Try a variant of one system without editing its entry: `run.py ... --variant myvariant --args "--max-refs 4"`; results go
  to `<system>@myvariant`. `--args` are extra arguments of the CROSS command line.
- Outdoor datasets automatically add `configs/outdoor.yaml`. Do not tune per dataset: the protocol forbids it.

### Quick check of a change: the development split

A small subset with the same protocol: about 7 min (quick tier) or 14 min (full tier) per mode on one GPU
([DEV.md](../benchmark/DEV.md)).

```bash
export BENCH_DEV_RESULTS=/path/to/dev_runs
python benchmark/datasets/make_dev.py                                                    # once
python benchmark/dev.py run --systems cross_stereo --tier quick --gpus 0                 # reference
python benchmark/dev.py run --systems cross_stereo --tier quick --gpus 0 --variant views7 --args "--max-refs 4 --n-ref-anchors 1"
python benchmark/dev.py compare cross_stereo cross_stereo@views7                         # side by side, paired T3 flips
```

Compare runs from the same GPU only, and read single-query T2 swings and single T3 flips as noise.

## 6. Programmatic use

```python
from cross.pipeline import build_session, mono_config_from_profile
session = build_session("mono", "visual", camera, SystemConfig(),
                        mono_config=mono_config_from_profile("configs/mono_benchmark_10hz.json", "models/dpvo.pth"))
for frame in dataset.replay_data():            # dict: rgb (uint8), timestamp[, depth, rgb_right, delta_pose]
    session.process(frame)
    T_c0, T_best, weights = session.belief(to_matrix)
```

For the back end alone (any mode, external odometry), use `System(camera=camera, config=cfg)` and call
`system.step(obs={"rgb": ..., "depth": ..., "delta_pose": ..., "timestamp": t})`; stereo adds `rgb_right` and
`T_right_in_left=` on the constructor. Other tools: `scripts/viz/record_trace.py` and `build_trace_page.py` (replay of
how a map is built), `scripts/eval_relpose.py` (relative-pose accuracy of the estimators).
