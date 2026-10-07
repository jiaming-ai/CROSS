# CROSS evaluation protocol

This protocol is the fixed yardstick for CROSS development: every change to the system is scored with it, and
every result in [`RESULTS.md`](RESULTS.md) and on the [results page](site/index.html) follows it. It has three
tracks. Tracks 1 and 2 place CROSS next to metric SLAM systems on familiar terms. Track 3 is the headline metric of the CROSS
paper.

| track | question | metric | datasets |
|---|---|---|---|
| **T1 Mapping accuracy** | How accurate is the map built in one session? | ATE RMSE of the final trajectory (+ completeness) | KITTI (outdoor), OpenLORIS-Scene (indoor), ROVER (outdoor) |
| **T2 Multi-session localization** | Given a map from an earlier session, how accurately is a new session localized in it? | localization recall and ATE of the new session, in the map frame | OpenLORIS-Scene, ROVER |
| **T3 Relocalization success** | Starting with no pose, how often does the system relocalize in a map from another session under change? | relocalization success RS@x (CROSS paper, 10 s trials; x = 1 / 2 m indoors, 3 / 5 m outdoors) | SimChange (synthetic), OpenLORIS-Scene, ROVER |

CROSS runs in all six combinations of observation mode (RGB-D with PnP, stereo with VGGT-Omega, monocular with Depth
Anything 3) and motion source (the dataset's external odometry, or DPVO visual odometry from the images).

## 1. Datasets and splits

All sequences run at **10 Hz** (OpenLORIS and ROVER are subsampled from 30 Hz with the nearest-timestamp frame; KITTI and
SimChange are native 10 Hz). Ground truth is interpolated to the image timestamps. The sessions of one scene share
one ground-truth frame, which T2 and T3 need.

| dataset | environment | sensors used | odometry given to systems that take odometry | ground truth | T1 sequences | T2 / T3 map → queries |
|---|---|---|---|---|---|---|
| [KITTI odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (raw drives) | outdoor, car, 0.4–5 km | stereo colour (cam 2/3, 0.54 m), mono (cam 2) | dead reckoning of the OXTS level-frame velocities and yaw rate, INS roll / pitch (no GPS position) | KITTI odometry poses (RTK/INS) | 00, 01, 02, 04–10 (03 has no raw drive) | none (single session per route) |
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
  - the total-station coordinates are a left-handed frame (the track turns opposite to the robot's gyroscope and to
    visual odometry of both cameras), so y is negated; `prepare_rover.py` checks the turning direction against the
    VN-100 gyroscope of every recording;
  - the heading is the direction of travel of the smoothed prism track, since the robot drives forward at a constant 0.5 m/s;
  - roll and pitch are zero;
  - the camera sits at the calibrated offset from the prism (about 0.5 m).

  A 2° heading error moves the camera by about 2 cm. All localization thresholds are on position only. The total station was set up anew for every recording, so the recordings do not share a frame.
  Each recording's prism track is registered to the map recording's track by 2-D ICP (x, y, heading) plus a height offset.
  This works because the robot drives the same lawn-edge route. The residuals are stored in each sequence's `calib.json`.
  On autumn → summer, for example, the registered tracks lie 0.13 m apart at the median and 0.42 m at the 90th percentile.
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
- **KITTI**: dead reckoning of the OXTS unit at 10 Hz. Its forward / left / up velocities are given in the level frame
  (parallel to the earth surface), so the positions integrate them turned by the heading, which integrates the yaw rate
  about the up axis; the orientation is that heading with the INS roll and pitch (gravity-referenced). The GPS positions are
  *not* used, because the ground truth is derived from them. The odometry ATE over 00–10 is 0.3–29 m (4.9 m mean).
  Before 2026-10-03 the velocities were integrated as body-frame velocities together with the body angular rates, so the
  direction of travel disagreed with the orientation by the vehicle's pitch / roll relative to the level (0.5–2.3°;
  odometry ATE 9.3 m mean); every KITTI result with external odometry from before that date used it.
- **ROVER**: the platform records no wheel odometry, only IMUs. The benchmark uses simulated odometry: ground-truth increments
  perturbed with SNR 10 noise, seeded, as in the CROSS paper's noise study and in SimChange. It is marked *sim-odom* in the tables.
- **SimChange**: simulated, SNR 10, seed 0 for the map session and seed 1 for queries.

**Odometry source (`run.py --odom-source external|vio`).** Every system that takes odometry (CROSS RGB-D, stereo,
mono with external odometry; RTAB-Map) runs with either source, and results are reported for both: `external` is the
dataset odometry above, `vio` a stereo VIO (rows "… stereo VIO odometry", results under `<system>+vio/`). KITTI's INS velocities and the simulated odometry of ROVER and
SimChange are derived from (or close to) the ground truth. With `--odom-source vio` the systems instead get the odometry of a stereo-inertial
VIO, Basalt (Usenko et al., RA-L 2020), run causally on the platform's own stereo pair and IMU
(`benchmark/datasets/prepare_vio.py`, written as `odom_vio.txt`; no loop closure, the pose of each frame is the one
estimated when that frame arrived). Datasets with wheel odometry keep it (OpenLORIS). One Basalt configuration (its
EuRoC configuration) and the IMU noise of each sensor's datasheet / calibration are used everywhere; nothing is tuned on
the ground truth. Inputs:

- **KITTI**: the rectified colour pair (10 Hz) and the OXTS RT3003 accelerations / angular rates at 100 Hz from the raw
  `*_extract` drives (`fetch_kitti_oxts.py` downloads only the `oxts/` folders); the camera's own timestamps.
- **ROVER**: the rectified T265 pair (10 Hz) and the T265's IMU (~200 Hz, Kalibr calibration). The RGB-D folder (D435i)
  gets the same trajectory interpolated to its frame times and moved to the D435i camera through the prism calibration.
- **SimChange**: the rendered pair (0.3 m) and the simulated IMU described below.

Accuracy over the 100-frame trial windows (`eval_odometry.py`, mean; details in the run report): KITTI 1.0–2.5 % of
the distance (OXTS dead reckoning 0.8–2.0 %), except 01 (highway, 7 %); ROVER ~0.3 m per window, about half of which is
the ground truth's own heading error (heading from the direction of travel), except the night session (no light, 3.4 m);
SimChange 0.09–0.27 m (simulated SNR 10: 0.27–0.46 m).

A real VIO can fail: on ROVER's night session Basalt's velocity runs away to ~30× the true one over the last 500 frames.
CROSS's odometry scale guard (`mapping.loop_closure.odom_guard_factor`, on by default) applies to the RGB-D and stereo
modes, whose visual translations are metric without the odometry. It compares the measured and the odometry
translations of short spans, one sample per frame, and fires when their ratio leaves [½, 2]× its long-run value in two
consecutive windows of 30 frames.
- **Attribution.** A departure counts only when the odometry's own speed changed in the matching direction. A visual
  estimator that loses the motion at car speed (PnP on depth and VGGT-Omega stereo on KITTI 01 / 04 with VIO) leaves
  the odometry's pace unchanged, so the guard does not fire.
- **After a firing.** The odometry is rescaled. Its translations become as uncertain as the measured error, over the
  stretch that departed and from then on, so the visual measurements and loop closures correct the map. The guard then
  follows the fault window by window.
- **Relocalization sessions** (`odom_guard_map`): the map measurements' fixes feed the guard too.

**IMU.** The monocular visual-inertial setups (CROSS mono with `--odometry vio`) get the IMU rigidly attached to the
monocular camera, at its native rate, with its calibrated extrinsics and noise (`benchmark/datasets/prepare_imu.py`,
written as `imu.txt` / `imu.json` next to the images):

- **OpenLORIS**: the D435i IMU (gyroscope 400 Hz, accelerometer 250 Hz, the factory intrinsics of `sensors.yaml`
  applied), extrinsics to the D435i colour camera from `trans_matrix.yaml`.
- **KITTI**: the OXTS RT3003 accelerations and angular rates in the vehicle frame (`ax ay az`, `wx wy wz`), 10 Hz in the
  synced drives; no velocities and no GPS. The OXTS timestamps jitter by ±5 ms, so the frames' times are the camera's
  (`image_02/timestamps.txt`).
- **ROVER**: the D435i IMU (about 270 Hz; Kalibr extrinsics and noise of `calib_d435i.yaml`). The dusk recording has no
  D435i IMU and uses the VN-100 (66 Hz) with its calibrated extrinsics.
- **SimChange**: simulated from the ground truth (C2 spline, 200 Hz) with the noise and bias of the D435i's IMU
  (BMI055), seeded by the sequence name.

The stereo visual-inertial setup (CROSS stereo with `--odometry vgio`) gets the IMU rigidly attached to the stereo pair
(`prepare_imu.py <dataset> ... --setup stereo`): on OpenLORIS and ROVER the T265's own IMU (BMI055, gyroscope 200 Hz,
accelerometer interpolated to it; OpenLORIS factory intrinsics, ROVER Kalibr calibration `calib_t265.yaml`), with its
pose in the rectified left camera; KITTI and SimChange use the same IMU as the monocular setup (their stereo folder is
the monocular setup's).

The system calibrates the camera-IMU time offset and the gyroscope bias online (against its visual odometry); nothing is
taken from the ground truth. IMU setups are marked ⁽ⁱ⁾ in the tables.

The baselines with an IMU (ORB-SLAM3 visual-inertial, RTAB-Map visual-inertial odometry; rows next to their
vision-only rows) get the IMU rigidly attached to the camera of their setup: the IMU above for the monocular and
RGB-D setups; for the stereo setups, the stereo pair's IMU (`prepare_vio.py --write-imu imu_vio`): KITTI's OXTS at
100 Hz from the raw extract drives (also for KITTI mono, whose camera is the stereo pair's left one; the 10 Hz of the
synced drives is too slow for inertial ORB-SLAM3), the T265's own BMI055 on OpenLORIS and ROVER (factory intrinsics
and extrinsics on OpenLORIS, Kalibr on ROVER), the simulated IMU on SimChange.  Every IMU extrinsic was checked
against the ground truth: the gyroscope's rate about the vertical axis, moved into the camera frame, correlates
with the ground truth's at 1.00 (OpenLORIS, KITTI, SimChange) and 0.65 (ROVER, whose heading is the direction of
travel).

Baselines that accept external odometry (RTAB-Map) get the same odometry stream; RTAB-Map also runs with its own visual
odometry (`rtabmap_vo`). Visual(-inertial) systems (ORB-SLAM3,
MASt3R-SLAM, VGGT-SLAM 2.0, DROID-SLAM) run on images only. The tables mark every system that uses odometry or IMU, so the
comparison stays interpretable.

## 4. Metrics

### T1 — mapping accuracy (single session)
- **ATE RMSE** (m) of the *final* trajectory. For CROSS this is the permanent map keyframes after all loop closures and pose-graph
  optimization. For other systems it is their final (keyframe) trajectory. Estimates are associated with ground truth by frame index, then aligned with **SE(3)**
  Umeyama for metric setups (RGB-D, stereo, or any system with odometry/IMU), and with **Sim(3)** only for monocular systems
  without metric input (the recovered scale is reported).
- **Completeness**: the fraction of the sequence's frames covered by the system's poses in one map. A frame counts when a pose
  of that map lies **within 1 s of it, or within 1 m (indoors) / 2 m (outdoors) of it along the travelled ground-truth path**.
  The time window catches lost tracking; the path distance accepts the sparse keyframes of a slow or standing robot. It is
  path distance, not straight-line distance: on a revisit the poses of the first pass are near in space but a whole loop away
  along the path, so a session that lost its second pass is still incomplete. When a system splits the run into several maps
  (ORB-SLAM3 atlas), only the largest map is evaluated. The same rule applies to every system. It is applied to every pose a
  system holds, even when its ATE uses fewer: CROSS's temporary keyframes of revisits count for completeness, and its ATE
  stays on the permanent map keyframes. **A run with completeness below 80 % counts as a tracking failure**: its ATE is
  shown in grey with its completeness, and it is excluded from means. (Before 2026-10-06 the rule was the 1 s window
  alone for the baselines, and CROSS was given 100 %; `completeness_rule` in a result says which rule scored it.)
- **Overall ATE** of a system (%): its ATE divided by the sequence's ground-truth path length, averaged over each dataset's
  finished sequences, then over KITTI, OpenLORIS and ROVER. Dividing by the path length puts the kilometre-scale KITTI
  drives and the room-scale OpenLORIS sessions on one scale, so each dataset weighs the same.
- Secondary: **online ATE**, the causal per-frame estimate (CROSS: the mean of hypothesis 0 at each step), with the same alignment.
  Also mapping FPS, peak GPU memory, and map size on disk.

### T2 — multi-session localization
The map session is mapped as in T1 and the map is saved. The query session is then run **once, continuously**, from its first
frame, against the loaded map, with no initial pose. Estimates are expressed in the ground-truth frame through the *map
session's* alignment (T1). The query session is **not** re-aligned: a system that localizes in the wrong place or in a disjoint map
is penalized.
- **Localization recall LR@x**: the fraction of *all* query frames whose position error is below x. Each frame is scored with
  the system's latest pose, if that pose is at most 1 s old; this is the pose a robot would get by asking the system at that frame,
  and it matters for systems that report poses only at keyframes. Frames with no pose that recent count as failures. The pose
  must be the one the system reported at that time, not a trajectory it writes after the session: ORB-SLAM3 counts as
  localized only from the frame at which its session has been merged into the stored map (before 2026-10-06 its trajectory
  after shutdown was used, which localizes frames before a late merge in hindsight). The systems without map persistence
  (MASt3R-SLAM, VGGT-SLAM 2.0) still log their keyframe poses only after their final optimization, which favours them. Thresholds: x = 1 m and 2 m indoors, 3 m and 5 m outdoors, the same as T3. This is the headline T2 number because it
  compares systems that report no pose until they relocalize with systems that always report one.
- **MS-ATE** (m): the RMSE over the frames that have an estimate, reported with their fraction.
- **Coverage**: only query frames that the map covers are evaluated. A frame is covered when its ground-truth position lies
  within the larger threshold (2 m indoors, 5 m outdoors) of the map session's ground-truth path. Where the map session never
  went, no system can relocalize. Coverage is close to 100 % on OpenLORIS and SimChange. On ROVER, the 2023 and spring
  recordings drive a longer route than the September 2024 map session. Coverage fractions are listed per session.
- **Aggregation**: LR and MS-ATE are computed per query session. Scene and dataset cells pool the frames of all query
  sessions (localized frames / all frames), so each session counts in proportion to its length, as T3 pools trials.
- **Overall LR** of a system: the mean over OpenLORIS, ROVER and SimChange of its pooled LR at each dataset's smaller and
  larger threshold, as for the overall RS (T3).
- **Time to localize**: the first frame after which the error stays below 1 m (indoor) or 5 m (outdoor) for 5 consecutive frames.

### T3 — relocalization success (CROSS paper, §5.1)
The query session is split into independent trials. Each trial loads the stored map, starts without knowing its pose, and runs
for a fixed number of frames. **RS@x is the fraction of trials whose final pose estimate lies within x of the ground truth**, position only. There are two
fixed thresholds per environment: **x = 1 m and 2 m indoors, 3 m and 5 m outdoors**. The larger ones are the CROSS paper's
radii. RS is the fraction of successful trials. The final estimate is the system's
latest pose in the trial, as reported at that time (see T2). It must be at most 1 s older than the trial's last frame, because some systems (MASt3R-SLAM, VGGT-SLAM 2.0) report poses only at
keyframes. Monocular systems are aligned with Sim(3) on the map session, so their errors are in metres too.
- Only trials whose last frame is covered by the map (see T2, Coverage) are counted.
- Trial length: **100 frames at 10 Hz (10 s), a new trial every 50 frames**, on every dataset. The CROSS paper used
  200-frame trials at the native 30 Hz, about 7 s. At the benchmark's 10 Hz, 200 frames would be 20 s, which makes
  relocalization easier. The same trials are used for the SimChange v2 runs. Trials overlap by half, so the Wilson intervals
  are approximate.
- The error is measured against the pose the map implies for the query frame (`scripts/reloc_metrics.py`). The pose of the
  nearest map keyframe in the map is composed with its ground-truth offset to the query frame, so map drift does not count as a
  relocalization error. The absolute variant (map-session alignment, as in T2) is stored as well.
- Also reported: the median final error of the failed trials, and 95 % Wilson intervals.
- **Overall RS** of a system: the mean over OpenLORIS, ROVER and SimChange of its pooled RS at each dataset's smaller and
  larger threshold (each dataset weighs the same, whatever its number of trials).
- **Failed query sessions** (a crash or timeout that persists after re-runs, or a failed map) are shown as *k/N* ✗ (k of
  the N query sessions failed). Every covered trial of a failed session counts as a failed trial (systems without map
  persistence: the 5 evenly spaced trials they would have run), and in T2 every covered frame counts as not localized, so a
  system that crashes cannot score higher than one that runs and fails. T1 means leave failed sequences out and give their
  count with the same notation.
- Systems without map persistence (MASt3R-SLAM, VGGT-SLAM 2.0, DROID-SLAM) run the map session followed by the trial in one
  stream. Only the trial frames are scored. Every trial re-runs the whole map session, so these systems are scored on at most
  5 evenly spaced trials per query session. A run that crashes (out of memory) or times out is reported as failed and re-run;
  it is not scored as a relocalization failure. This is marked in the tables because the system keeps the map session's live
  state, which a persistence-based system does not have.

## 5. Systems

| system | setups | odometry / IMU | map reuse (T2, T3) | source |
|---|---|---|---|---|
| CROSS (PnP) | RGB-D, RGB-D\* | external odometry | save / load map | this repository (`--mode rgbd`) |
| CROSS (PnP) | RGB-D, RGB-D\* | visual odometry (DPVO, metric scale from the depth) | save / load map | `--mode rgbd --odometry visual` |
| CROSS (VGGT-Omega) | stereo | external odometry | save / load map | `--mode stereo` |
| CROSS (VGGT-Omega) | stereo | visual odometry (DPVO, scale from stereo depth) | save / load map | `--mode stereo --odometry visual` |
| CROSS (Depth Anything 3) | mono | external odometry | save / load map | `--mode mono --odometry external` |
| CROSS (Depth Anything 3) | mono | visual odometry (DPVO + learned metric scale) | save / load map | `--mode mono --odometry visual` |
| CROSS (VGGT-Omega mono) | mono | external odometry | save / load map | `--mode mono --mono-estimator ff` |
| CROSS (VGGT-Omega mono) | mono | visual-inertial odometry (DPVO + IMU scale) | save / load map | `--mode mono --mono-estimator ff --odometry vio` |
| ORB-SLAM3 | mono, stereo, RGB-D | none (visual) | atlas save / load, multi-map merge | [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) + `scripts/baselines/orbslam3_reloc.cc` |
| RTAB-Map | RGB-D, stereo | same odometry as CROSS | database, localization mode | [introlab/rtabmap](https://github.com/introlab/rtabmap) + `scripts/baselines/rtabmap_reloc.cc`; every frame processed, `Mem/STMSize 30` (RTAB-Map's KITTI setting) |
| RTAB-Map (visual odometry) | RGB-D, stereo | none: RTAB-Map's own visual odometry (frame-to-map, reset to the latest pose after a lost frame) | database, localization mode | as above, `rtabmap_reloc --vo` |
| ORB-SLAM3 (visual-inertial) | mono, stereo, RGB-D | IMU (§3) | atlas save / load, multi-map merge | as ORB-SLAM3, `--sensor imu_mono / imu_stereo / imu_rgbd` (its IMU_* modes, default IMU initialization) |
| RTAB-Map (visual-inertial odometry) | RGB-D, stereo | IMU (§3): its visual odometry with the IMU orientation (RTAB-Map's complementary filter), gravity constraints in its odometry and map graph | database, localization mode | as above, `rtabmap_reloc --vo --imu` |
| MASt3R-SLAM | mono | none | concatenated stream | [rmurai0610/MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM) |
| VGGT-SLAM 2.0 | mono | none | concatenated stream | [MIT-SPARK/VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM) |
| DROID-SLAM | mono, stereo, RGB-D | none | concatenated stream | [princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) (planned) |

Baselines run with their published default parameters for the sensor type. The same parameters are used on every dataset, apart
from the camera calibration and frame rate. A baseline that crashes or times out (default 2 h per session) is reported as a
failure of that run, not dropped. ORB-SLAM3 carries bug fixes only (`scripts/baselines/orbslam3.patch`): saving and
loading atlases with culled keyframes; two crashes while saving the atlas at shutdown (`Map::PreSave` erased map points
from the set it was iterating over, in about one run in six, and the atlas was saved while the mapping and loop-closing
threads still changed the map); and the trajectory export of the largest map with each frame's map id (monocular too).
Two declared deviations from its defaults, for the visual-inertial rows only: its inertial modes start a map only after
an acceleration change of 0.5 m/s² between frames and run place recognition (the only way back into a stored map) only
after their second inertial BA, ~15 s after the IMU initialisation — gates built for long sessions, which a 10 s
relocalization trial never passes. The visual-inertial runs set `IMU.fastInit: 1` (ORB-SLAM3's own option, which its
v1.0 settings loader no longer read) and `LoopClosing.mergeAfterImuInit: 1` (place recognition once the IMU is
initialised; its merge already completes the inertial initialisation of a map that merges before the second BA, and
merges are still verified on three consecutive keyframes).

## 6. Run rules

- **Seeds.** The quick protocol uses seed 0. Claims in papers use the full protocol, seeds 0, 1 and 2, reported as mean ± std.
  The seed controls simulated odometry noise and RANSAC and sampling. With real odometry, runs still differ slightly through GPU
  non-determinism.
- **Pairing.** Two configurations are compared only on the cells both have completed, on the same GPU model. Every result file records the
  commit, GPU, command and wall-clock time.
- **Frames.** Every frame at 10 Hz is given to every system in real-time order. There is no frame dropping and no lookahead. Systems
  that gate their own observations (e.g. the CROSS stereo preset) still see every frame. ORB-SLAM3, whose mapping, loop
  closing and map merging run in threads beside the tracking, is fed at the real rate (one frame every 100 ms, as in its
  own examples), so those threads get the time they would get on the robot.
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

Every run records the code version it ran (`commit`: the `COMMIT` file of a copied code snapshot, which should hold
the commit hash, else `git rev-parse --short HEAD`, with `+dirty` for uncommitted changes). `collect.py` keeps every
run in `results/history.json` (`--backfill-git` rebuilds it from the committed versions of `results.json`), and the
page's History tab shows the tables once per code version, so a regression can be traced to the commit that made it.

A new system is added by writing a runner that produces the same `result.json` (`benchmark/eval/`), plus a row in
`benchmark/configs/systems.yaml`.

## 8. Limitations

- T1 compares a topological map (CROSS) with metric SLAM systems at their keyframes. CROSS keyframes are sparse and placed where the
  appearance changes, so its ATE is measured on fewer poses.
- KITTI RGB-D\* depth comes from stereo matching, not from a depth sensor.
- ROVER odometry is simulated (§3). ROVER has no rotation ground truth: the camera orientation is derived from the direction of
  travel (§1), so ROVER results report position errors only.
- SimChange is synthetic. Its variants isolate change factors but its appearance is not real.
