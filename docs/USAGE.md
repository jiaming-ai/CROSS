# Using CROSS

A quick guide: pick a mode, run a sequence, change the configuration, map and relocalize across sessions, and run
the benchmark. Installation is in the [README](../README.md#installation).

## 1. Modes

Pick a mode by name with `--mode <name>`; each name is a shipped configuration file (`configs/modes/<name>.yaml`,
section 3) and sets the sensors, the odometry and the back end in one word:

| `--mode` | sensors | motion between frames (odometry) | needs |
|---|---|---|---|
| `stereo_wheel` (default of the stereo runners) | left + right image + wheel / external odometry | the dataset's odometry (wheel, OXTS, or simulated with `--snr`) | `install.sh --stereo` |
| `stereo_basalt` | stereo + IMU | Basalt stereo-inertial VIO file (`odom_vio.txt`, see below) | `install.sh --stereo`, Basalt |
| `stereo_imu` | stereo + IMU | VGGT-Omega + IMU local pose graph, scale from the stereo pair | `install.sh --stereo` |
| `stereo_dpvo` | stereo | DPVO, scale from stereo depth | `install.sh --stereo --mono` |
| `rgbd_wheel` (default of `run.py`) | colour + depth + external odometry | the dataset's odometry | `install.sh` |
| `rgbd_basalt` | colour + depth, stereo + IMU for the VIO | Basalt VIO file | `install.sh`, Basalt |
| `rgbd_dpvo` | colour + depth | DPVO, scale from the depth | `install.sh --mono` |
| `rgbd_ff_wheel` | colour + depth + external odometry; VGGT-Omega relative poses scaled by the depth | the dataset's odometry | `install.sh --stereo` |
| `rgbd_ff_basalt` | as `rgbd_ff_wheel`, stereo + IMU for the VIO | Basalt VIO file | `install.sh --stereo`, Basalt |
| `mono_wheel` | one colour camera + external odometry | the dataset's odometry | `install.sh --mono --stereo` |
| `mono_basalt` | one colour camera, stereo + IMU for the VIO | Basalt VIO file | `install.sh --mono --stereo`, Basalt |
| `mono_imu` | one colour camera + IMU | VGGT-Omega + IMU local pose graph, scale from learned depth | `install.sh --mono --stereo` |
| `mono_vio` | one colour camera + IMU | DPVO scaled by the IMU (a reference system) | `install.sh --mono --stereo` |
| `mono_dpvo` | one colour camera | DPVO, scale from learned depth (Depth Anything 3 back end) | `install.sh --mono` |

How a retrieved keyframe is registered: stereo and mono modes run VGGT-Omega (one forward pass over the current image
and the retrieved keyframes; metric scale from the stereo baseline, or in mono from the odometry and the map); rgbd
modes match XFeat + LightGlue features and solve PnP-RANSAC on the keyframe depth, except `rgbd_ff_*`, which run the
stereo mode's VGGT-Omega passes on the colour image and take each pass's metric scale from the measured depth (median
ratio of measured to predicted depth over the current frame's confident pixels; `pose_est.ff.depth_scale`, dense or
sparse depth such as a phone's LiDAR); `mono_dpvo` uses the older
monocular back end of `cross/mono/` (metric two-view matching, Depth Anything 3 as fallback).

**Sensor mode + options (other combinations).** `--mode rgbd | stereo | mono` with `--odometry external | basalt |
visual | vio | vgio`, in the mono mode `--mono-estimator ff | da3` and in the rgbd mode `--rgbd-estimator pnp | ff`
(default pnp; ff with external / basalt odometry only) select the same fourteen files and a few
reference combinations without a file of their own (e.g. the DA3 mono back end with DPVO + IMU, `--mode mono --odometry
vio`). `ff` (feed-forward) = VGGT-Omega, `da3` = the Depth Anything 3 back end (the default with a sensor mode, except
that `--odometry vgio` implies `ff`). A mode name and a conflicting `--odometry` / `--mono-estimator` is an error.
`vio` is for the mono mode, `vgio` for the mono and stereo modes. Visual odometry needs DPVO (`install.sh --mono`) and
its weights in `models/dpvo.pth` (or `--dpvo-checkpoint`, env `CROSS_DPVO_CHECKPOINT`).

**External odometry from a stereo VIO.** A prepared folder can hold several odometry files; `--odom-file NAME`
(`run.py`, `scripts/map_and_reloc*.py`) reads `NAME` instead of `odom_left.txt` when it exists (else `odom_left.txt`, else
the simulated odometry). `--odometry basalt` is external odometry from `odom_vio.txt` (the same fallback) with the
`*_basalt.yaml` mode file. `benchmark/datasets/prepare_vio.py` writes `odom_vio.txt`: Basalt stereo-inertial odometry run
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
Each pass has a scale of its own: in the mono mode learned metric depth observes it, with a bias state: DA3, or the
metric-scale head of a `vggt_ft` checkpoint (`--set pose_est.ff.checkpoint=<ckpt> --mono-args "--imu-config
depth_prior_source=head"`; no second network). In the
stereo mode (`--mode stereo --odometry vgio`) the stereo pair adds two constraints: classical stereo depth (SGBM) of the
current pair against the pass's depth map observes the pass's scale (no learned depth, no bias state), and the tracked
corners lifted to 3-D with that depth give the metric motion between measured frames (PnP, with its own covariance).
Every visual translation is tested against the IMU's prediction (the IMU arbitrates when the visual cues disagree, e.g.
when vehicles alongside fill the view). The stereo mode needs the IMU next to the stereo folder (`prepare_imu.py
<dataset> ... --setup stereo` for OpenLORIS and ROVER, whose stereo pair is the T265 with its own IMU). Options:
`--mono-args "--imu-config key=value"`, e.g. `vgio_stereo_pnp=false`, `vgio_stereo_source=baseline` (the right image as
one more view of the pass instead of SGBM depth; biased on KITTI).

## 2. Run one sequence

```bash
python run.py data/r3d/lab2.r3d                                          # RGB-D, external odometry
python run.py data/kitti_raw/2011_09_30/2011_09_30_drive_0027_sync --mode stereo --config configs/overlays/outdoor.yaml
python run.py data/sim/lonemonk/map --mode stereo --baseline 0.3         # SimChange: which rendered baseline
python run.py data/posed/home1-1 --loader posed --mode rgbd_dpvo           # RGB-D, no odometry input
python run.py data/posed/home1-1 --loader posed --mode mono_dpvo           # RGB only
python run.py $BENCH_DATA/openloris/home1-1/rgbd --loader posed --mode mono_vio
python run.py $BENCH_DATA/kitti/07/stereo --mode stereo_imu --config configs/overlays/outdoor.yaml   # stereo + IMU
```

Useful flags: `--no-viz` (no Rerun window; use it on a server), `--frames N`, `--start N`, `--loader` (default:
auto-detect), `--snr` (simulated odometry noise). `python run.py -h` lists all.

**Data formats.** R3D, ROS bag, OpenLORIS, TUM, *posed folders* (`rgb/`, `depth/`, `poses_left.txt`, optional
`odom_left.txt`, `calib.json`), and stereo sequences (KITTI raw, TartanAir V2, Virtual KITTI 2, SimChange). Converters
are in `scripts/datasets/` and `benchmark/datasets/`.

## 3. Configuration

```
configs/
  modes/<mode>.yaml             one file per mode (the 12 names of section 1)
  base/                         shared pieces the mode files include:
    stereo.yaml, mono_ff.yaml     the stereo / mono VGGT-Omega back end
    mono_frontend.yaml            the monocular frontends' MonoConfig (DPVO, people masking, learned-depth cadence)
    vggt_finetuned.yaml           settings that apply only with the fine-tuned VGGT-Omega checkpoint
    common.yaml                   every mode: background pose-graph optimisation
    odometry_vio.yaml             odometry noise of visual-inertial odometry (included by the VIO modes)
  overlays/
    fast/<mode>.yaml              --fast: constrained compute
    outdoor.yaml                  outdoor scale (the benchmark adds it on outdoor datasets)
    stereo_lowband.yaml           opt-in: left image only, half the rate
    odometry_vio.yaml             opt-in: external odometry from a VIO (ARKit / ARCore, a robot's VIO)
```

Every run loads one shipped file, `configs/modes/<mode>.yaml`, chosen by `--mode <name>` (or the sensor mode with
`--odometry` / `--mono-estimator`, section 1), with `--fast` the mode's fast overlay, then the user's `--config` files
(merged left to right) and `--set` overrides. The same files serve a local run and a remote one (section 6): where the
session runs is chosen by the runner's `--remote-server` / the `cross-edge` client, not by a configuration file.

`--fast` adds `configs/overlays/fast/<mode>.yaml`: constrained compute (RTX 4000 Ada), only where relocalization stays within 5 % (T3 and T2 at 1 m indoors / 3 m outdoors): `stereo_basalt` (6 images per pass at 416 px: query 10.6 -> 20.7 FPS) and `stereo_dpvo` (416 px: 6.5 -> 8.9 FPS). The other modes' overlays are empty: 416 / 448 px cost more than 5 % there (each file says what was tested); their defaults on an RTX 4000 Ada: stereo + wheel 10.6, stereo + IMU 6.8, mono + wheel 18.5, mono + IMU 9.7, RGB-D 40, mono DPVO 15 query FPS. A map built with `--fast` must be relocalized in with `--fast` (its stored images are smaller; checked at load).

| mode file (`--mode`) | = sensor mode / `--odometry` (/ `--mono-estimator`) |
|---|---|
| `stereo_wheel`, `stereo_basalt`, `stereo_imu`, `stereo_dpvo` | `stereo` / `external`, `basalt`, `vgio`, `visual` |
| `rgbd_wheel`, `rgbd_basalt`, `rgbd_dpvo` | `rgbd` / `external`, `basalt`, `visual` |
| `rgbd_ff_wheel`, `rgbd_ff_basalt` | `rgbd` / `external`, `basalt` (`--rgbd-estimator ff`) |
| `mono_wheel`, `mono_basalt`, `mono_imu` | `mono` / `external`, `basalt`, `vgio` (`--mono-estimator ff`; implied by `vgio`) |
| `mono_dpvo` | `mono` / `visual` (the DA3 back end) |
| `mono_vio` | `mono` / `vio` (`--mono-estimator ff`: VGGT-Omega + DPVO + IMU, a reference system; merges in mapping sessions and hypothesis 0 updated by every observation: ROVER T3 +360 trials; no adoption in mapping sessions (it broke the KITTI 10 map)) |

Reference combinations without a file of their own use their back end's file: mono DA3 with any odometry `mono_dpvo`,
mono `ff` with DPVO `mono_wheel`. A mode file includes shared pieces of `configs/base/` (key `include:`, paths
relative to the file; `common.yaml` is in every mode); keys below the include override them. Besides the SystemConfig
sections a mode file holds the monocular frontends' settings: `mono:` (fields of `MonoConfig`, `cross/mono/config.py`:
DPVO, learned-depth cadence, people masking; for the DA3 back end also its retrieval, charts and `cross_overrides`, the
SystemConfig lines it applies after its presets of `cross/mono/system.py`) and `imu:` (fields of `ImuConfig` of the
inertial frontends). `--mono-args` overrides them from the command line: `"--imu-config key=value"` and, for the DA3
back end, `"--cross-config section.key=value"`; any other `mono` setting goes in a `--config` file's `mono:` section.

A configuration file of your own uses the same sections as a mode file; only the keys you set change:

```yaml
# my_robot.yaml: python run.py <seq> --mode mono_imu --config my_robot.yaml
pose_est:
  obs_max_interval_steps: 2      # SystemConfig (cross/core/config.py)
mono:
  mask_people: false             # MonoConfig of the monocular frontend (cross/mono/config.py)
imu:
  depth_prior: false             # ImuConfig of the inertial frontend (cross/imu/scale_filter.py)
```

```bash
python run.py <seq> --mode stereo_wheel --fast --config configs/overlays/outdoor.yaml my_overrides.yaml
python scripts/map_and_reloc_rgbd.py ... --set mapping.loop_closure.confidence=0.9995  pose_est.obs_max_interval_steps=2
```

| file | use |
|---|---|
| `configs/base/vggt_finetuned.yaml` | the fine-tuned VGGT-Omega's settings (included by the bases of every mode that runs VGGT-Omega; applies only with a fine-tuned checkpoint, see below): the learned covisibility head |
| `configs/overlays/outdoor.yaml` | outdoor scale (clustering and hypothesis-alignment radius) and, with the fine-tuned VGGT-Omega, a looser covisibility gate (0.10); the benchmark adds it on outdoor datasets |
| `configs/overlays/stereo_lowband.yaml` | opt-in for the stereo mode with trusted external odometry, e.g. remote sessions over a thin link: the back end observes on the left image (`pose_est.ff.right_image: left`; scale from the odometry) at half the rate: a third of the uplink; blinds the odometry scale guard (not with a VIO that can run away); not tested with `--fast` |
| `configs/base/odometry_vio.yaml` | how far the odometry may drift, for the loop-closure consistency test and the pose graph (`mapping.loop_closure.noise.odom_*`), set for visual-inertial odometry: included by the VIO modes (`stereo_basalt`, `stereo_imu`, `mono_imu`, `mono_vio`, `mono_basalt`, `rgbd_basalt`, `rgbd_ff_basalt`). The generic code defaults suit wheel odometry (`*_wheel`); the vision-only `*_dpvo` modes set their own, looser values. Too loose for a VIO, the generic values let look-alike loop closures through and the pose graph bends the map |
| `configs/overlays/odometry_vio.yaml` | opt-in: the same values for external odometry that comes from a VIO (a phone's ARKit / ARCore, a VIO on the robot) in a `*_wheel` mode |
| `configs/base/mono_frontend.yaml` | the monocular frontends' shared `mono:` settings (DPVO frontend, people masking, a learned-depth scale observation every 10 frames), included by the mono modes and `stereo_imu`. The DA3 back end's own lines are in `modes/mono_dpvo.yaml`; with external odometry or `vio` it drops their odometry-noise lines (`mapping.loop_closure.noise.odom_*`, `tracking.*`) |

Override one value with `--set section.key=value` (`run.py` and the `scripts/` runners) or put it in a small
YAML file and pass it with `--config`. Some options (charts, session recovery, historical retrieval slots, ...) are off by default and are
enabled by the DA3 back end's mode file (`configs/modes/mono_dpvo.yaml`).

**Configuration caveats.** The left-image overlay (`pose_est.ff.right_image: left`) has two documented costs, checked
on the fully resolved configuration when a session starts (`cross.pipeline.config_risks`; `run.py`, the
`scripts/map_and_reloc*.py` runners, and the remote server, whose warning the `cross-edge` CLI shows): with odometry
that can run away (`--odometry basalt`, `visual`, `vio`, or an odometry file named `*vio*`) the odometry scale guard
cannot see a runaway (ROVER night with Basalt: map ATE 38.2 m vs 11.2 m with the pair), and with 4 references per
pass relocalization drops (development split T3 28 -> 24 of 33; the stereo base has 6); `--fast` (416 px) with the left
image is untested. The run prints a warning naming the caveat and
the safer alternative and asks `[y/N]`; without a terminal it aborts unless `--accept-config-risk` is given (the
benchmark harness passes nothing, so a risky experiment must opt in with `--args "--accept-config-risk"`). The left
image with trusted external odometry, the overlay's purpose, and with `--odometry vgio` (the frontend keeps the pair for
its stereo depth) runs without a question.

**Reproducible runs.** The shipped files of every folder optimise long loop-closure pose graphs in a background worker
(`mapping.loop_closure.async_pgo`, below), whose results land at machine-dependent times. For experiments pass
`--reproducible` (synchronous optimisation whatever the configuration says); the benchmark harness (`benchmark/run.py`,
`benchmark/dev.py`) always does.

**VGGT-Omega checkpoint.** `pose_est.ff.checkpoint` is `auto` by default: the fine-tuned checkpoint
(`models/VGGT-Omega-CROSS/vggt_omega_cross_1b_512.pt`, installed by `install.sh --stereo`) when it is there, else the
released one (`models/VGGT-Omega/vggt_omega_1b_512.pt`, `install.sh --vggt=released`). `finetuned`, `released` or a path
select one (`--set pose_est.ff.checkpoint=released`; `scripts/map_and_reloc.py --checkpoint released`). A config file's
top-level `vggt_finetuned:` section holds settings for the fine-tuned model: it applies after the rest of its file when
the checkpoint in use is fine-tuned, i.e. carries a `covis_head_map` (install.sh's default, or any `vggt_ft` checkpoint
calibrated with `calibrate_covis.py --map_geo`). The released checkpoint runs with the settings outside these sections.
`configs/base/vggt_finetuned.yaml` turns on the learned covisibility head (`covis_source: auto`; the IMU bases
`base/stereo_imu.yaml` / `base/mono_imu.yaml` keep the geometric score for the VGGT-inertial odometry);
`configs/overlays/outdoor.yaml` lowers the covisibility gate to 0.10.

**Keyframe quality filter** (`mapping.keyframe_quality`, `cross/core/kf_quality.py`). Views that are mostly blocked or
empty are not stored as permanent keyframes: a person or object right in front of the camera, a wall or shelf at arm's
length, darkness, glare, a covered lens. They waste storage, and views of the same occluder at different places look
alike, so they get retrieved at the wrong place. Such a frame becomes a temporary node: the odometry chain is kept,
with no image and no descriptor. The test is the share of the view that shows textured, well-exposed scene content
at a normal distance. The near field comes from sensor depth, or from stereo matching in the stereo mode; people are
found by a small detector. Thresholds are relative to the session's running medians.
A view that is only textureless (a white wall) is kept unless it shows no structure above the sensor noise. On by
default; `--set mapping.keyframe_quality.enabled=false` turns it off. The test set with injected junk views is
`benchmark/datasets/inject_junk.py` (`benchmark/dev.py ... --tier occ`).

**Background pose-graph optimisation** (`mapping.loop_closure.async_pgo`, `cross/core/async_pgo.py`; on in every shipped mode file, off in the dataclass defaults and with `--reproducible`).
On a large map one loop-closure optimisation takes seconds, and the front end processes no frame meanwhile. With
`--set mapping.loop_closure.async_pgo=true` the optimisation of a verified loop closure (and the GNSS-triggered one)
runs in a forked worker process: it works on a copy-on-write snapshot of the system, and the front end keeps processing
frames. The result is applied when it arrives, to the state of that moment: keyframes added meanwhile and the tracked
pose move with the correction of the latest keyframe at the fork. A thread would not help: GTSAM holds the GIL for the
whole optimisation. Linux only; the state must be on the CPU (the default `state_device`); coordinate-chart (mono
DA3 back end) graphs stay synchronous. Details:
- In free-running mode (`async_pgo_lag_steps=-1`, the default) the result is applied as soon as the worker is done, so
  *when* it lands depends on the machine's speed and a run is not bit-reproducible. An optimisation expected to take less
  than `async_pgo_min_s` (0.1 s; and twice the cost of a fork, ~60 ms in a CUDA process) runs in the front end as before, so
  small maps are unchanged (the expectation is the number of keyframes it covers times the cost per keyframe measured so far).
- `async_pgo_lag_steps=k` (k >= 0) applies the result at the start of step s+k (the front end waits when the worker is
  not done), so the run is reproducible. `0` is the synchronous behaviour and `1` (the start of the next step, nothing
  processed in between) gives the same map; both are bit-identical to `async_pgo=false` (only the pose reported for the
  frame that closes the loop is one correction behind with `1`). Larger lags and the free-running mode process frames on
  the uncorrected map meanwhile and change the results: equal within noise on the dev split and ROVER and (with the
  tail smoothing and the provisional optimisation of the latest `async_pgo_provisional_kf` = 1000 keyframes before the fork)
  on the 6.4 km NCLT drive (3.72 m vs 3.68 m synchronous; `outputs/2026-10-08_async_pgo/REPORT.md`), with no optimisation-related
  stall over 1 s.
  Use `1` to test; benchmarks run synchronously (`--reproducible`).
- One job at a time; loop closures detected meanwhile are re-checked against the corrected poses when it has been applied.
  A merge of hypotheses, the final optimisation of a save and every other whole-graph operation wait for the job first.

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

## 4. Multi-session: map one session, relocalize in another

Mapping and relocalization are two steps: build and save a map, then load it and localize a later session against it.
The two harnesses do both and score the result (map ATE; relocalization success over independent 100-frame trials that
start without a pose; a trial succeeds when the final estimate is within `--r-d` of the pose the map implies):

```bash
# RGB-D / mono on posed folders
python scripts/map_and_reloc_rgbd.py --map data/posed/home1-1 --query data/posed/home1-2 --out outputs/home
python scripts/map_and_reloc_rgbd.py --map ... --query ... --out outputs/home_mono --mode mono_dpvo
# stereo sequences: stereo mode (VGGT-Omega) by default; --mode rgbd_wheel --pnp-depth gt|sgbm gives the PnP baseline on the same data,
# --mode rgbd_ff_wheel --pnp-depth gt|sgbm VGGT-Omega scaled by that depth (posed RGB-D folders: --pnp-depth gt reads depth/)
python scripts/map_and_reloc.py --map data/sim/hssd_house/map --query data/sim/hssd_house/light_night \
    --out outputs/house --baseline 0.3 --snr 10 --trial-len 100 --trial-stride 50
```

The map is written to `--out` (`map.pkl`, `map_meta.json`); `--skip-map` reuses it for another query and `--skip-reloc`
only maps. Another query in the same scene is another call with the same `--out`/`--skip-map`. `--extend-map DIR` makes
the map run a second session in the map of an earlier run (`DIR/map.pkl` + `map_meta.json`): it relocalizes, adds
keyframes where the map has none, and saves the extended map with a `map_meta.json` covering every keyframe (session in
`kf_session`), so queries in it are scored like in any map; `map_meta.json` "extension" scores the new keyframes in the
loaded map's ground-truth alignment (ATE, how far the loaded keyframes moved, fraction of frames localized). With
`--map-start/--map-end` the first session can map part of a route. In code, use
`System.save_map(path)` / `System.load_map(path)`; `examples/multi_session.py` runs several sequences through one
`System`, and `examples/planner.py` loads a map and plans paths on it.

**A mono session in a stereo map.** A map built in the stereo mode can be used later by a robot with one camera (the
map's left camera): run the query in a VGGT-Omega mono mode (`--mode mono_wheel`, `mono_basalt` or `mono_imu`), with the normal mono configuration files; nothing else is needed. Pairs of
map references with their relative pose in the map anchor the metric scale of each pass. Tested 2026-10-10 on
OpenLORIS, ROVER and KITTI 07: relocalization equal to a stereo query in the same map or better, and with IMU the
query inherits the better stereo map (scale within 0.8 % on ROVER vs 2 % in a mono map). Also using the right images
stored with a stereo map's keyframes as extra anchors changed nothing measurable, so it is not done (tag
`opt-mono-right-anchors`). With the benchmark harness: `benchmark/run.py --system cross_mono_ff_odom --setup stereo --task query --map-from
cross_stereo ...` (the query relocalizes in `cross_stereo`'s map of the same run).

**Another camera.** A map made with one camera can be relocalized in and extended with another (another robot,
another lens, another stereo rig), in the feed-forward modes (stereo, mono with `--mono-estimator ff`). The map keeps
a camera table (`save_map` "cameras": the intrinsics of the stored images and the stereo rig of each camera) and every
keyframe the id of the camera that took it. Loading a map: the session's images are made at the size of the stored
ones (the largest centred crop with their aspect ratio, resized; a wider camera loses the sides), its camera gets an
id (the same camera within 1 % of the intrinsics and the rig is the same id), and in every forward pass each
reference gets the principal-point correction of its own camera and, as a scale anchor, its stored right image with
its own camera's rig. The model predicts each view's field of view, so different focal lengths need nothing else.
Not supported: PnP (the RGB-D mode) and the DA3 mono back end (they refuse another camera), and with `--odometry vgio`
a camera whose images need another size than the map's. Maps saved before the table load as one camera. In simulated
tests (four scenes; cameras with 90°, 75° and 60° field of view, 0.30 / 0.12 / 0.20 m baselines, a shifted principal
point and another height) a stereo session with the other camera relocalized as often as one with the map's camera,
and after an extension with the other camera, queries in the extended area localized almost as well as in the
original one (recall at 1 m 0.93-0.99 against 0.95-0.99). Ignoring the camera ids placed the new keyframes worse (one
scene: recall at 1 m in the extended area 0.81 instead of 0.95). Untested: the learned scale of the mono modes,
whose scale head takes the session camera's focal length for the whole pass.

**Map files.** A map is `map.pkl` (the graph: keyframe poses, edges, metadata) plus the directory `map.pkl.store/`
next to it (keyframe images and retrieval descriptors); copy or move both together. Keyframe images of a loaded map
are read from disk when they are used, so a large map does not have to fit in memory. The `storage` config section
chooses the encoding: `storage.image_codec` `webp_lossless` (default; exact) or `png`, or the lossy `jpeg` / `webp` at
`storage.image_quality`; `storage.depth_drop_bits` (lossy depth, RGB-D); `storage.descriptor_dtype` (`float16` by default, `float32`);
keyframe images live in host RAM and are copied to the GPU only for the pass that uses them;
`storage.max_ram_images` (2000 by default; 0 = all) keeps only the newest keyframes' images in memory during a long session. Maps written by
older versions (one `map.pkl` with everything inside) still load; `storage.format pickle` writes that format, and
`python scripts/convert_map.py old/map.pkl new/map.pkl [--image-codec jpeg --image-quality 95] --verify` converts a map.

**GNSS and compass: maps on Earth.** With a GNSS receiver (and optionally a magnetometer), `--gnss` (both
runners; config `geo.enabled`) anchors the map to latitude / longitude: every keyframe of the saved map gets its
position on Earth (`python scripts/geo/export_map_geo.py out/map.pkl --out out/map_geo` writes GeoJSON / KML). The
fixes are quality-controlled before they are used, so weak GNSS (indoors, near buildings, right after reacquisition)
gets no weight instead of pulling the map: each fix must agree with the robot's own track over the last 20 s (a
receiver that holds a stale position while the robot moves fails this), a new epoch after a gap waits 10 s, and a fix
must agree with the map's prediction at the verified loop closure's chi-square level. Used fixes become pose-graph
factors with a robust kernel, spaced one correlation time of the receiver error apart (estimated online; consumer
receivers repeat the same error for tens of seconds), and the graph is optimised when they say the map drifted. In a
relocalization session on a geo-anchored map, the current fix drops references that contradict it (a calibrated compass
also drops references with a contradicting heading), and the stored keyframes within ~3.7 sigma + 5 m of a trusted fix
take the map retrieval budget except two global slots (`geo.retrieval_focus`, on; NCLT cross-season relocalization 60 vs
56 of 75 trials within 5 m, degraded GPS no worse than none).
Input: a prepared folder (or its parent) with `gnss.txt` (`t lat lon alt [mode num_sats [hdop [sigma_h]]]`, times on
the clock of `times.txt`), optionally `mag.txt` / `ms25.txt` (magnetometer + accelerometer, forward-right-down) or
`heading.txt` (see `cross/dataloader/geo.py`); in code, `obs["gnss"] = {"t", "lat", "lon", "alt", ...}` and
`obs["compass"]` per step. `--gnss-degrade sigma=8,bias=15,drop=0.3,outage=60:120` degrades the query's fixes
(testing). Main options (`geo.*`): `gate`, `decimate`, `robust` (the method; switch off for ablations),
`sigma_v_factor` (4: consumer altitude is poor, but horizontal-only factors let the optimisation tilt the map),
`retrieval_gate` (the fix as a location prior), `proposal_gate`, `compass_gate`. On by default (`--no-gnss` /
`geo.enabled=false` turns it off): `--gnss` only feeds the folder's GNSS data; without GNSS input the system's results
are unchanged, and a geo-anchored map keeps its anchor and keyframe latitude / longitude when it is saved again.

Stereo-mode quick-run options: `--obs-min-translation/--obs-min-rotation/--obs-max-interval` (observation gating;
default: the mode's configuration file), `--max-refs`, `--n-ref-anchors`. Without a right camera pass
`--set pose_est.ff.right_image=left` (scale from the odometry; see the README OpenLORIS example).

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
| `orbslam3`, `rtabmap`, `rtabmap_vo` | baselines (drivers in `scripts/baselines/`, built separately); RTAB-Map with the dataset odometry / its own visual odometry |
| `orbslam3_imu`, `rtabmap_vi` | the same baselines with the setup's IMU: ORB-SLAM3's inertial modes, RTAB-Map's visual odometry with the IMU orientation (stereo setups need `prepare_vio.py <dataset> ... --write-imu imu_vio` first) |
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
- Re-score stored runs with the current metrics without running the system: `run.py ... --reeval` (map / t1 tasks: the
  T1 result; query tasks of the baselines: T2 / T3 from their pose files).
- Outdoor datasets automatically add `configs/overlays/outdoor.yaml`. Do not tune per dataset: the protocol forbids it.

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

## 6. Remote sessions: the GPU work on a server, the odometry on the robot

A session can be split (`cross/remote/`): the **server** runs the back end (retrieval, the VGGT-Omega passes,
hypotheses, loop closure, the map) and the GPU side of the VGGT-inertial odometry (its own passes, learned / stereo
depth scale, depth ratios); the **edge** (the robot's computer, no GPU) runs the IMU propagation, the tracked corners
and the local pose graph, or forwards the external odometry.  Every frame the edge sends its motion, and images only
on the frames the back end will observe or the odometry measures; the replies come back late.  A late measurement
joins the local graph at its frame's time and the IMU carries the state to the current frame again; the published map
pose is the back end's pose of the last replied frame carried by the odometry since.  Supported: external odometry
(rgbd / stereo / mono with `--mono-estimator ff`) and `--odometry vgio` (mono ff, stereo).

Simulated network, on the dataset's clock (deterministic; any runner or benchmark variant):

```bash
python scripts/map_and_reloc.py ... --odometry vgio --remote --remote-rtt 0.1           # 100 ms round trip
#   --remote-compute model|measured|zero   server time (model: cross/remote/link.py, --remote-costs to override)
#   --remote-jitter 0.05  --remote-outage every:30:3  --remote-jpeg 90  --remote-uplink-mbps 5
#   --remote-max-backlog 0.3 (default; 0: off)   a server behind the edge skips stale observations instead of queueing
#   --remote-obs-cap 0.1 (default; 0: off)   the edge sends an observation only if the server could start it within
#                          0.1 s (its model of the server's queue); else the back end observes a later frame
#   --set pose_est.obs_confident_max_interval_steps=10   fewer observations while localized; the edge follows (no images)
#   vgio measures on the back end's passes in the stereo mode (imu.vgio_align, on there; off for mono, where it costs
#   scale accuracy): 20 % fewer uploads;  --mono-args '--imu-config vgio_align=false' turns it off
#   --config configs/overlays/stereo_lowband.yaml   stereo + trusted external odometry: left image only, half the rate (a third
#                                          of the uplink; the odometry scale guard is blind then)
#   --remote --remote-compute zero --remote-upload all   reproduces the local session exactly
python benchmark/dev.py run --systems cross_stereo_vgio --variant rtt100 --args "--remote --remote-rtt 0.1 --online-poses"
```

Real network (gRPC; `pip install grpcio` on both machines):

```bash
python scripts/remote/serve.py --port 50051                                              # GPU machine
python scripts/map_and_reloc.py ... --remote-server gpu-host:50051 --remote-realtime      # edge: frames at their timestamps
#   --remote-extra-delay 0.1   emulate a longer round trip;  --remote-jpeg 90   lossy uploads
```

Standalone edge (the robot's computer, CPU only, no CROSS / torch): the `cross-edge` package in `edge/` (its own
`pyproject.toml`; `edge/README.md`) runs Basalt's stereo-inertial VIO live (`edge/native/install_basalt_live.sh` builds
`basalt_live`), copies the back end's observation cadence and publishes the map-frame pose of every frame:

```bash
pip install ./edge && edge/native/install_basalt_live.sh ~/basalt_build      # on the robot
python scripts/remote/serve.py --port 50051                                  # GPU machine (full CROSS)
cross-edge run --server gpu-host:50051 --data <stereo folder> --realtime --save-map /maps/a.pkl --out run_map
cross-edge run --server gpu-host:50051 --data <stereo folder> --realtime --load-map /maps/a.pkl --out run_query
```

The server builds the session from its own shipped file for the mode (`configs/modes/stereo_basalt.yaml`;
`stereo_wheel.yaml` for another pose file than `odom_vio.txt`), plus `--config-file` paths in its
`configs/` folder (e.g. `overlays/outdoor.yaml`) and `--set` overrides, and sends the cadence settings back; apart from
the background pose-graph optimisation the effective configuration equals the benchmark's `cross_stereo` (external
odometry).  Uploads are lossless by default (`--jpeg 0`; JPEG 90 cost the stereo mode relocalization, office1-7 T2
0.86 -> 0.56).  `cross.remote.edge.RemotePipeline`
(the CROSS runners' edge, also with the VGGT-inertial frontend) extends the package's `EdgeSession`.

Map files are written on the server at the paths the edge names.  `--online-poses` (map runs) scores the pose the
session published at every frame (`map_meta.json` `online`), the number latency changes; `remote` in `map_meta.json` /
`reloc_summary.json` holds the link's statistics (latency, uploads, server time per kind of message, timeline).

## 7. 3D reconstruction of a map

`cross_world/` trains a Gaussian-splatting scene from a saved map (novel views, web viewer), see
[WORLD.md](WORLD.md). `--set mapping.world_capture.enabled=true` on a mapping run also keeps the keyframes' input
frames at full resolution, uncropped, next to the map (`map.pkl.capture/`; the map itself is unchanged) for a later
reconstruction without the dataset folder.

```bash
python -m cross_world.cli build --map runs/k07/map.pkl --source $SEQ --out worlds/k07 --novel 150
python -m cross_world.cli export --world worlds/k07/world.pt --map runs/k07/map.pkl --source $SEQ --out web/k07
```

## 8. Navigation: BEV traversability maps

Optional (`bash install.sh --nav`). Closed-loop navigation with the BEV planner, the trajectory policy or their hybrid
(`NavigationAgent`) is described in `cross_nav/policy/README.md`.

`cross_nav/` adds a head to the VGGT-Omega pass that predicts a bird's-eye-view grid around the current view
(400 x 400 cells of 5 cm, camera at the centre, gravity-aligned, heading = optical axis): P(occupied) per cell, where
flat walkable ground is free and anything 0.1-1.8 m above it is occupied whatever its height. It reads the backbone's
multi-layer tokens and lifts them with the pass's own depth and cameras, the pass's metric scale (stereo, odometry or
the scale head) and gravity (IMU or the head's own estimate). It also predicts cells no view sees, from scene priors.
A heads file is trained for one backbone checkpoint (`python -m cross_nav.train --config cross_nav/configs/train/<run>.yaml`).

```bash
# online, every pass of a session: PoseEstFeedForward.last_bev (prob, up_cam, R_bev_from_cam, log_scale)
python run.py ... --set pose_est.ff.nav_bev_head=models/nav/heads.pt [--set pose_est.ff.nav_bev_every=2]
# offline, a fused navigation map of a saved map (one BEV per keyframe, anchored to the keyframes)
python -m cross_nav.navmap --map runs/office/map.pkl --heads models/nav/heads.pt --backbone $CKPT --out nav/office
```

Gravity from an IMU: `pose_est.set_gravity(up_in_camera)` before the step whose pass should use it. The head adds
~12 ms per pass (RTX 5090, 6-10 views) and makes the depth head cover the current view and the references.
Labels for training come from the scene cache of `vggt_ft` (`python -m cross_nav.labels`: all frames of a place
accumulated, free-space carving) and from simulated HSSD scenes with mesh labels (`cross_nav/sim/`).

## Calibrating a two-iPhone stereo rig

Two phones mounted side by side, each recording images, ARKit poses and (optionally) LiDAR depth with the capture
app, can be calibrated from their own recordings, without a calibration target:

```bash
python scripts/rig/calibrate_iphone_rig.py \
    --pair walk1/recording_1_A walk1/recording_1_B \
    --pair walk2/recording_2_A walk2/recording_2_B \
    --out rig.yaml [--compare stereo_extrinsics.yaml]
```

A recording can also be read from inside the capture zip: `capture.zip::capture_1/<...>/data/recording_<...>_B`.
Phone B is the left camera, A the right one.  Per recording (`cross/rig/selfcalib.py`):

1. **Clock offset** between the phones: both turn at the same angular speed at every instant, so the offset is the
   shift that best aligns the two ARKit tracks' angular speed (a few ms).
2. **Extrinsic** (rotation and metric translation between the cameras) by hand-eye calibration of the two ARKit
   tracks over thousands of motion pairs; the walk has to turn.
3. **Image refinement**: SIFT matches between a B image and the A image nearest in time; the relative pose of the two
   exposures includes the camera motion between them (from ARKit), so imperfect synchronisation does not bias it.
   Rotation and translation direction minimise the epipolar (Sampson) error.  `--no-images` skips it.
4. **Baseline length** from LiDAR: matches triangulated with the rig against phone B's LiDAR depth.
   `--no-lidar` keeps the hand-eye length.

Prepare the stereo folders with `scripts/rig/prepare_iphone_stereo.py` (pairs by true time, rectifies each pair
with its own relative pose; writes ARKit as the odometry). ARKit is a visual-inertial odometry: map with
`--mode stereo_wheel --config configs/overlays/odometry_vio.yaml` (section 3).

The recordings are combined into one calibration (rotations averaged, median baseline); the YAML keeps every
recording's clock offset and its rotation difference from the combined one, which shows whether the rig moved
between recordings.  The output uses the layout of `stereo_extrinsics.yaml` (`R`, `T`: x_B = R x_A + T in OpenCV
camera axes of the raw images; camera matrices from ARKit).

## 9. Programmatic use

```python
from cross.pipeline import build_session, mono_config_from_files
session = build_session("mono", "visual", camera, SystemConfig(),
                        mono_config=mono_config_from_files(["configs/modes/mono_dpvo.yaml"], "models/dpvo.pth"))
for frame in dataset.replay_data():            # dict: rgb (uint8), timestamp[, depth, rgb_right, delta_pose]
    session.process(frame)
    T_c0, T_best, weights = session.belief(to_matrix)
```

For the back end alone (any mode, external odometry), use `System(camera=camera, config=cfg)` and call
`system.step(obs={"rgb": ..., "depth": ..., "delta_pose": ..., "timestamp": t})`; stereo adds `rgb_right` and
`T_right_in_left=` on the constructor. Other tools: `scripts/viz/record_trace.py` and `build_trace_page.py` (replay of
how a map is built), `scripts/eval_relpose.py` (relative-pose accuracy of the estimators).

## 10. Fine-tuning VGGT-Omega (`vggt_ft/`)

The fine-tuned checkpoint (`Jamie-nus/VGGT-Omega-CROSS`) adds a metric-scale head and a pairwise covisibility head to
the released VGGT-Omega and is trained with multi-session windows. You do not need this section to run CROSS. It
describes how to reproduce or retrain the checkpoint. `vggt_ft/configs/` holds that checkpoint's recipe only:

| config | step |
|---|---|
| `v5a.yaml` | fine-tune of the released model (8 blocks frozen, scale-aligned losses, distillation to the released model, EMA) |
| `v10c.yaml` | continues v5a@10000 EMA with close-range / people data (HOI4D, PointOdyssey, Dynamic Replica) |
| `vfin2_s0.yaml` | the scale and covisibility heads only, on the 0.7 weight blend of v10c@3000 EMA with the released model |
| `vdl2_s1.yaml` | a dense local scale head on vfin2's frozen model (`pose_est.ff.scale_source=head`; still evaluated against DA3) |
| `eval_suite.yaml`, `val_suite.yaml` | held-out test suite and validation suite (`python -m vggt_ft.evaluate`) |
| `smoke.yaml` | a 40-step trainer check |

Paths in these files are `${VGGT_FT_ROOT}/...`, filled in from the environment when a config is loaded. An unset
variable is an error. Lay out the root as `cache/` (scene cache), `ckpt/` (the released `vggt_omega_1b_512.pt`; the
DA3 labelers for `vdl2_s1`), `runs/` (outputs) and `test/` (evaluation scenes and windows).

```bash
export VGGT_FT_ROOT=/path/to/vggt_ft_root
# scene cache: one converter per dataset, then index + check
python -m vggt_ft.prep.<dataset> ...            # writes $VGGT_FT_ROOT/cache/<dataset>
python -m vggt_ft.prep.common index  $VGGT_FT_ROOT/cache/<dataset>
python -m vggt_ft.prep.common verify $VGGT_FT_ROOT/cache/<dataset>
# train (8 GPUs; key.sub=value overrides; resumes from runs/<exp>/ckpt_last.pt)
torchrun --nproc_per_node=8 -m vggt_ft.train --config vggt_ft/configs/v5a.yaml
torchrun --nproc_per_node=8 -m vggt_ft.train --config vggt_ft/configs/v10c.yaml
python scripts/vggt_ft/interpolate.py --released $VGGT_FT_ROOT/ckpt/vggt_omega_1b_512.pt \
    --ft $VGGT_FT_ROOT/runs/v10c/ckpt_003000_ema_bf16.pt --alpha 0.7 --tag s3000 --out_dir $VGGT_FT_ROOT/runs/wise_v10c
torchrun --nproc_per_node=8 -m vggt_ft.train --config vggt_ft/configs/vfin2_s0.yaml
# covisibility map that CROSS needs to use the head (and to apply the `vggt_finetuned:` settings)
python scripts/vggt_ft/calibrate_covis.py --config vggt_ft/configs/vfin2_s0.yaml \
    --ckpt $VGGT_FT_ROOT/runs/vfin2_s0/ckpt_003000_ema_bf16.pt --map_geo --out vfin2_headmap.pt
# evaluate any checkpoint (the released one too)
python -m vggt_ft.evaluate --ckpt vfin2_headmap.pt --suite vggt_ft/configs/eval_suite.yaml --out res.json
python scripts/vggt_ft/compare_evals.py released.json res.json --md
```

Use the result with `--set pose_est.ff.checkpoint=vfin2_headmap.pt`. Some datasets and labelers in the recipe are
non-commercial (Stereo4D, FoundationStereo depth, Oxford Spires, HOI4D, Dynamic Replica); check their licences before
use. `scripts/vggt_ft/launch_h20.sh`, `evalwatch.sh` and `sync_h20.sh` are the authors' cluster helpers.
