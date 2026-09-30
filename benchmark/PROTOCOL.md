# CROSS evaluation protocol

This protocol is the fixed yardstick for CROSS development: every change to the system is scored with it, and
every result in [`RESULTS.md`](RESULTS.md) and on the [results page](site/index.html) follows it. It has three
tracks. Tracks 1 and 2 place CROSS next to metric SLAM systems on familiar terms. Track 3 is the headline metric of the CROSS
paper.

| track | question | metric | datasets |
|---|---|---|---|
| **T1 Mapping accuracy** | How accurate is the map built in one session? | ATE RMSE of the final trajectory (+ completeness) | KITTI (outdoor), OpenLORIS-Scene (indoor), ROVER (outdoor) |
| **T2 Multi-session localization** | Given a map from an earlier session, how accurately is a new session localized in it? | localization recall and ATE of the new session, in the map frame | OpenLORIS-Scene, ROVER |
| **T3 Relocalization success** | Starting with no pose, how often does the system relocalize in a map from another session under change? | relocalization success RS (CROSS paper, 10 s trials) | SimChange (synthetic), OpenLORIS-Scene, ROVER |

Monocular CROSS is still in development; its columns are in the tables but stay empty until it is released.

## 1. Datasets and splits

All sequences run at **10 Hz** (OpenLORIS and ROVER are subsampled from 30 Hz with the nearest-timestamp frame; KITTI and
SimChange are native 10 Hz). Ground truth is interpolated to the image timestamps. The sessions of one scene share
one ground-truth frame, which T2 and T3 need.

| dataset | environment | sensors used | odometry given to systems that take odometry | ground truth | T1 sequences | T2 / T3 map → queries |
|---|---|---|---|---|---|---|
| [KITTI odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (raw drives) | outdoor, car, 0.4–5 km | stereo colour (cam 2/3, 0.54 m), mono (cam 2) | dead reckoning of the OXTS velocities and angular rates (no GPS) | KITTI odometry poses (RTK/INS) | 00, 01, 02, 04–10 (03 has no raw drive) | none (single session per route) |
| [OpenLORIS-Scene](https://lifelong-robotic-vision.github.io/dataset/scene.html) | indoor, wheeled robot; office, corridor, home, cafe, market | RGB-D (D435i colour + aligned depth), stereo (T265 fisheye pair, rectified), mono (D435i colour) | robot wheel odometry | motion capture (office), 2D LiDAR SLAM (others) | all 22 sequences | office1-1 → 1-2…1-7; corridor1-1 → 1-2…1-5; home1-1 → 1-2…1-5; cafe1-1 → 1-2; market1-1 → 1-2, 1-3 (17 queries) |
| [ROVER](https://iis-esslingen.github.io/rover/) campus_large | outdoor, ground robot, multi-season | RGB-D (D435i), stereo (T265, rectified), mono (D435i colour) | see §3 | total station (mm) | all 8 recordings | day (2024-09-25) → summer, autumn, winter, spring, dusk, night, night-light (7 queries) |
| [SimChange](https://github.com/jiaming-ai/SimChange) v2 | indoor (HSSD house, restaurant, classroom, Lone Monk) | rendered RGB-D, stereo (0.1/0.3/0.5 m), mono | ground truth + SNR 10 noise (seeded) | exact | – | map → all change variants of the scene (lighting, rearrangement, viewpoint, reverse, combinations) |

Notes.
- **RGB-D outdoors.** No real outdoor dataset has usable active depth in sunlight. ROVER's D435i depth is used as
  recorded. On KITTI the RGB-D setup (written **RGB-D\***) is the left image plus depth from stereo matching of the rectified pair
  (OpenCV SGBM, `cross.dataloader.stereo_loader.SGBMDepth`). Every system that takes depth gets the same depth maps.
- **Rectified stereo.** The T265 fisheye pairs of OpenLORIS and ROVER are rectified once to a pinhole stereo pair
  (`benchmark/datasets/prepare_*.py`, 640×480, 90° horizontal field of view). All systems receive the same rectified
  images, so no system gets an advantage from native fisheye support. ROVER's D435i colour images carry lens distortion.
  They are undistorted to a pinhole camera, and the registered depth maps get the same undistortion.
- **ROVER ground truth** is the 3-D position of a prism on the robot, measured by a total station at ~5 Hz. It has no
  orientation. The benchmark derives camera poses from it:
  - the heading is the direction of travel of the smoothed prism track, since the robot drives forward at a constant 0.5 m/s;
  - roll and pitch are zero;
  - the camera sits at the calibrated offset from the prism (about 0.5 m).

  A 2° heading error moves the camera by about 2 cm. On ROVER, therefore, only position errors are reported: there is no
  strict 1 m / 5° RS.
- **Why these datasets.** KITTI is the most widely reported outdoor SLAM benchmark. OpenLORIS-Scene (indoor) and ROVER
  (outdoor) are the two real datasets with RGB-D, stereo, mono, odometry or IMU, and repeated sessions of one place under
  changing conditions. Both were used in the CROSS paper (OpenLORIS Corridor and ROVER Campus). SimChange changes one controlled
  factor at a time.

## 2. Setups

| setup | input | CROSS mode |
|---|---|---|
| **RGB-D** | colour + depth (+ odometry) | RGB-D mode: XFeat + LightGlue + PnP-RANSAC on keyframe depth (`--mode rgbd`, shipped `configs/default.yaml`) |
| **stereo** | rectified stereo pair (+ odometry) | stereo mode: VGGT-Omega multi-view relative pose, stereo-baseline scale (`--mode stereo`, `configs/stereo.yaml`) |
| **mono** | colour only (+ odometry) | monocular mode (in development) |

Outdoor sequences (KITTI, ROVER) add `configs/outdoor.yaml`. It is an environment preset that coarsens clustering and the
hypothesis-alignment radius to the outdoor scale; it is not tuned per dataset. **No per-dataset or per-sequence tuning against ground truth is
allowed.** A system may use a noise model calibrated without ground truth on the first minute (600 frames) of the map
sequence (`scripts/lc/calibrate_noise.py`). Results using such a model are marked *calibrated*. The default rows use the
shipped defaults.

## 3. Odometry

CROSS fuses an odometry stream with its visual observations. Each dataset gives it the most realistic odometry it has:

- **OpenLORIS**: the robot's wheel odometry (`odom.txt`), untouched.
- **KITTI**: dead reckoning of the OXTS forward/left/up velocities and body angular rates at 10 Hz. It drifts about 1–2 % of
  the distance. The GPS positions are *not* used, because the ground truth is derived from them.
- **ROVER**: the platform records no wheel odometry, only IMUs. The benchmark uses simulated odometry: ground-truth increments
  perturbed with SNR 10 noise, seeded, as in the CROSS paper's noise study and in SimChange. It is marked *sim-odom* in the tables.
- **SimChange**: simulated, SNR 10, seed 0 for the map session and seed 1 for queries.

Baselines that accept external odometry (RTAB-Map) get the same odometry stream. Visual(-inertial) systems (ORB-SLAM3,
MASt3R-SLAM, VGGT-SLAM, DROID-SLAM) run on images only. The tables mark every system that uses odometry or IMU, so the
comparison stays interpretable.

## 4. Metrics

### T1 — mapping accuracy (single session)
- **ATE RMSE** (m) of the *final* trajectory. For CROSS this is the permanent map keyframes after all loop closures and pose-graph
  optimization. For other systems it is their final (keyframe) trajectory. Estimates are associated with ground truth by frame index, then aligned with **SE(3)**
  Umeyama for metric setups (RGB-D, stereo, or any system with odometry/IMU), and with **Sim(3)** only for monocular systems
  without metric input (the recovered scale is reported).
- **Completeness**: the fraction of the sequence's frames that fall within the evaluated trajectory. A frame counts when the final
  trajectory has a pose at that frame or at a keyframe no more than 1 s away in the same map. When a system splits the run into several maps (ORB-SLAM3
  atlas), only the largest map is evaluated. **A run with completeness below 80 % counts as a tracking failure**: its ATE is shown in
  grey with its completeness, and it is excluded from means.
- Secondary: **online ATE**, the causal per-frame estimate (CROSS: the mean of hypothesis 0 at each step), with the same alignment.
  Also mapping FPS, peak GPU memory, and map size on disk.

### T2 — multi-session localization
The map session is mapped as in T1 and the map is saved. The query session is then run **once, continuously**, from its first
frame, against the loaded map, with no initial pose. Estimates are expressed in the ground-truth frame through the *map
session's* alignment (T1). The query session is **not** re-aligned: a system that localizes in the wrong place or in a disjoint map
is penalized.
- **Localization recall LR@x**: the fraction of *all* query frames whose position error is below x. Frames without an estimate
  count as failures. Thresholds: x = 0.5 m and 1 m indoors, 1 m and 5 m outdoors. This is the headline T2 number because it
  compares systems that report no pose until they relocalize with systems that always report one.
- **MS-ATE** (m): the RMSE over the frames that have an estimate, reported with their fraction.
- **Time to localize**: the first frame after which the error stays below 1 m (indoor) or 5 m (outdoor) for 5 consecutive frames.

### T3 — relocalization success (CROSS paper, §5.1)
The query session is split into independent trials. Each trial loads the stored map, starts without knowing its pose, and runs
for a fixed number of frames. **A trial succeeds when its final pose estimate is within r_D of the ground truth**: r_D = 2 m
indoors, 5 m outdoors, position only. RS is the fraction of successful trials. The final estimate is the system's
latest pose in the trial. It must be at most 1 s older than the trial's last frame, because some systems (MASt3R-SLAM, VGGT-SLAM) report poses only at
keyframes. Monocular systems are aligned with Sim(3) on the map session, so their errors are in metres too.
- Trial length: **100 frames at 10 Hz (10 s), a new trial every 50 frames**, on every dataset. The CROSS paper used
  200-frame trials at the native 30 Hz, about 7 s. At the benchmark's 10 Hz, 200 frames would be 20 s, which makes
  relocalization easier. The same trials are used for the SimChange v2 runs. Trials overlap by half, so the Wilson intervals
  are approximate.
- The error is measured against the pose the map implies for the query frame (`scripts/reloc_metrics.py`). The pose of the
  nearest map keyframe in the map is composed with its ground-truth offset to the query frame, so map drift does not count as a
  relocalization error. The absolute variant (map-session alignment, as in T2) is stored as well.
- Also reported: strict RS at 1 m / 5°, the median final error of the failed trials, and 95 % Wilson intervals.
- Systems without map persistence (MASt3R-SLAM, VGGT-SLAM, DROID-SLAM) run the map session followed by the trial in one
  stream. Only the trial frames are scored. Every trial re-runs the whole map session, so these systems are scored on at most
  20 evenly spaced trials per query session. This is marked in the tables because the system keeps the map session's live
  state, which a persistence-based system does not have.

## 5. Systems

| system | setups | odometry / IMU | map reuse (T2, T3) | source |
|---|---|---|---|---|
| CROSS (RGB-D, PnP) | RGB-D, RGB-D\* | odometry | save / load map | this repository |
| CROSS (stereo, VGGT-Omega) | stereo | odometry | save / load map | this repository |
| CROSS (mono) | mono | odometry | save / load map | in development |
| ORB-SLAM3 | mono, stereo, RGB-D | none (visual) | atlas save / load, multi-map merge | [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) + `scripts/baselines/orbslam3_reloc.cc` |
| RTAB-Map | RGB-D, stereo | same odometry as CROSS | database, localization mode | [introlab/rtabmap](https://github.com/introlab/rtabmap) + `scripts/baselines/rtabmap_reloc.cc`; every frame processed, `Mem/STMSize 30` (RTAB-Map's KITTI setting) |
| MASt3R-SLAM | mono | none | concatenated stream | [rmurai0610/MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM) |
| VGGT-SLAM | mono | none | concatenated stream | [MIT-SPARK/VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM) |
| DROID-SLAM | mono, stereo, RGB-D | none | concatenated stream | [princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) (planned) |

Baselines run with their published default parameters for the sensor type. The same parameters are used on every dataset, apart
from the camera calibration and frame rate. A baseline that crashes or times out (default 2 h per session) is reported as a
failure of that run, not dropped.

## 6. Run rules

- **Seeds.** The quick protocol uses seed 0. Claims in papers use the full protocol, seeds 0, 1 and 2, reported as mean ± std.
  The seed controls simulated odometry noise and RANSAC and sampling. With real odometry, runs still differ slightly through GPU
  non-determinism.
- **Pairing.** Two configurations are compared only on the cells both have completed, on the same GPU model. Every result file records the
  commit, GPU, command and wall-clock time.
- **Frames.** Every frame at 10 Hz is given to every system in real-time order. There is no frame dropping and no lookahead. Systems
  that gate their own observations (e.g. the CROSS stereo preset) still see every frame.
- **Timing.** FPS is measured end to end (loading excluded) on the recorded GPU. It is indicative only on shared servers.

## 7. Output format and reproduction

Each run writes one `result.json` (schema in [`eval/schema.md`](eval/schema.md)) under
`<results_root>/<track>/<dataset>/<scene>/<system>/<setup>/<map>__<query>/`. The collector merges them into
[`results/results.json`](results/results.json), from which [`RESULTS.md`](RESULTS.md) and the web page are generated:

```bash
bash benchmark/datasets/download_openloris.sh  $DATA/openloris          # and download_kitti.sh, download_rover.sh
python benchmark/datasets/prepare_openloris.py $DATA/openloris $DATA/bench/openloris
python benchmark/run.py --track t1 t2 t3 --dataset openloris --system cross_rgbd --out $RESULTS
python benchmark/collect.py $RESULTS && python benchmark/make_tables.py && python benchmark/build_site.py
```

A new system is added by writing a runner that produces the same `result.json` (`benchmark/eval/`), plus a row in
`benchmark/configs/systems.yaml`.

## 8. Limitations

- T1 compares a topological map (CROSS) with metric SLAM systems at their keyframes. CROSS keyframes are sparse and placed where the
  appearance changes, so its ATE is measured on fewer poses.
- KITTI RGB-D\* depth comes from stereo matching, not from a depth sensor.
- ROVER odometry is simulated (§3). ROVER has no rotation ground truth: the camera orientation is derived from the direction of
  travel (§1), so ROVER results report position errors only.
- SimChange is synthetic. Its variants isolate change factors but its appearance is not real.
