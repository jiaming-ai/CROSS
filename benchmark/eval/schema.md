# result.json schema

Every benchmark job (`benchmark/run.py`) writes one `result.json` per track and cell. `benchmark/collect.py` merges them
into `benchmark/results/results.json` (`{"results": [...]}`). Fields that do not apply are absent. Non-finite numbers
are written as `null`.

## Common fields

| field | meaning |
|---|---|
| `track` | `t1`, `t2`, `t3`, or `ext` (the extension session of a map, cross-camera datasets) |
| `dataset`, `scene` | keys of `benchmark/configs/datasets.yaml` |
| `system`, `setup`, `label` | key, setup (`rgbd`, `stereo`, `mono`) and display name from `benchmark/configs/systems.yaml` |
| `seed` | run seed |
| `variant`, `odom` | the variant label (`run.py --variant`) and the odometry source (`vio`), when not the defaults |
| `uses_odometry` | the system consumed the dataset's odometry |
| `status` | `ok` or `failed` (crash, timeout, missing output, or the map it needed failed) |
| `rc`, `error` | return code and error message of a failed run |
| `wall_s` | wall-clock seconds of the run |
| `host`, `gpu`, `time`, `commit` | where, on what (`cpu` when `CUDA_VISIBLE_DEVICES` is not set), when and with which code the run was made (`commit`: short hash, `+dirty` with uncommitted changes) |

## T1: single-session mapping (`sequence`)

| field | meaning |
|---|---|
| `ate_rmse`, `ate_median`, `ate_max` | position error (m) of the final trajectory after alignment |
| `align`, `scale` | `se3` or `sim3`, and the Sim(3) scale (1 for SE(3)) |
| `completeness`, `completeness_rule` | fraction of the sequence's frames covered by a pose of the map (within 1 s, or within 1 m indoors / 2 m outdoors along the ground-truth path; PROTOCOL.md §4), and the rule that scored it |
| `failed` | completeness < 0.8 (tracking failure) |
| `n_poses`, `n_frames`, `n_keyframes_all` | evaluated poses, frames of the sequence, all keyframes (temporary ones too) |
| `fps`, `n_keyframes`, `map_bytes` | mapping rate, keyframes, map size on disk (when the system reports them) |
| `traj_est`, `traj_gt` | downsampled aligned estimate and ground-truth positions (for plots) |

## T2: multi-session localization (`map`, `query`)

| field | meaning |
|---|---|
| `lr@<x>` | fraction of the evaluated query frames (covered and moving) whose position error in the map frame is below x m (no estimate = failure) |
| `ms_ate`, `ms_median` | RMSE and median position error (m) over the evaluated frames with an estimate |
| `est_frac` | fraction of evaluated frames with an estimate |
| `time_to_localize` | first frame after which the error stays below the smaller threshold (1 m indoors, 3 m outdoors) for 5 frames |
| `covered`, `moving`, `evaluated`, `n_frames_all` | fractions of the query's frames covered by the map, at which the robot moves, and both; all frames |
| `all_frames` | the recalls and RMSE over all query frames |
| `map_relative` | the same recalls / RMSE against the pose the map implies (map drift removed) |
| `fps` | processing rate of the query session |
| `area_split`, `map_key` | cross-camera datasets: recalls in the first session's area and in the extension's, and the map's key |
| `err_curve` | downsampled per-frame error, `-1` where the system gave no estimate |
| `concat` | the system ran the map and query sessions as one stream (no map persistence) |

## T3: relocalization success (`map`, `query`)

| field | meaning |
|---|---|
| `n_trials`, `n_success` | trials and successful trials of this query session |
| `rs` | success rate at r_D |
| `rs_ci95` | 95 % Wilson interval of `rs` |
| `fail_err_median` | median final error (m) of the failed trials |
| `trials` | per trial: `start` frame, `success` (within r_D), `success_strict` (within 1 m and 5°), `final_err`, `moving` (fraction of moving frames); cross-camera datasets also `area`, `covered` |
| `r_d`, `trial_len` | success radius (m) and trial length (frames) |

## ext: extension session (`map`, `extend`)

The second session of a map on the cross-camera datasets (`scripts/map_and_reloc.py --extend-map`): the scores of
`map_meta.json` "extension" (the new keyframes in the loaded map's ground-truth alignment, how far the loaded
keyframes moved, the fraction of frames localized), `completeness` over the keyframes the session created on its own
sequence, and the extended map's `map_ate_rmse`.
