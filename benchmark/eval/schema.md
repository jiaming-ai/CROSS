# result.json schema

Every benchmark job (`benchmark/run.py`) writes one `result.json` per track and cell. `benchmark/collect.py` merges them
into `benchmark/results/results.json` (`{"results": [...]}`). Fields that do not apply are absent. Non-finite numbers
are written as `null`.

## Common fields

| field | meaning |
|---|---|
| `track` | `t1`, `t2` or `t3` |
| `dataset`, `scene` | keys of `benchmark/configs/datasets.yaml` |
| `system`, `setup`, `label` | key, setup (`rgbd`, `stereo`, `mono`) and display name from `benchmark/configs/systems.yaml` |
| `seed` | run seed |
| `uses_odometry` | the system consumed the dataset's odometry |
| `status` | `ok` or `failed` (crash, timeout, missing output, or the map it needed failed) |
| `rc`, `error` | return code and error message of a failed run |
| `wall_s` | wall-clock seconds of the run |
| `host`, `gpu`, `time`, `commit` | where, on what (`cpu` for CPU-only runs), when and with which code the run was made |

## T1: single-session mapping (`sequence`)

| field | meaning |
|---|---|
| `ate_rmse`, `ate_median`, `ate_max` | position error (m) of the final trajectory after alignment |
| `align`, `scale` | `se3` or `sim3`, and the Sim(3) scale (1 for SE(3)) |
| `completeness` | fraction of the sequence's frames within 1 s of an evaluated pose |
| `failed` | completeness < 0.8 (tracking failure) |
| `n_poses`, `n_frames` | evaluated poses, frames of the sequence |
| `fps`, `n_keyframes`, `map_bytes` | mapping rate, keyframes, map size on disk (when the system reports them) |
| `traj_est`, `traj_gt` | downsampled aligned estimate and ground-truth positions (for plots) |

## T2: multi-session localization (`map`, `query`)

| field | meaning |
|---|---|
| `lr@<x>` | fraction of all query frames whose position error in the map frame is below x m (no estimate = failure) |
| `ms_ate`, `ms_median` | RMSE and median position error (m) over the frames with an estimate |
| `est_frac` | fraction of query frames with an estimate |
| `time_to_localize` | first frame after which the error stays below 1 m (indoor) or r_D (outdoor) for 5 frames |
| `map_relative` | the same recalls / RMSE against the pose the map implies (map drift removed) |
| `err_curve` | downsampled per-frame error, `-1` where the system gave no estimate |
| `concat` | the system ran the map and query sessions as one stream (no map persistence) |

## T3: relocalization success (`map`, `query`)

| field | meaning |
|---|---|
| `n_trials`, `n_success` | trials and successful trials of this query session |
| `rs`, `rs_strict` | success rate at r_D, and at 1 m / 5° |
| `rs_ci95` | 95 % Wilson interval of `rs` |
| `fail_err_median` | median final error (m) of the failed trials |
| `trials` | per trial: `start` frame, `success`, `success_strict`, `final_err` |
| `r_d`, `trial_len` | success radius (m) and trial length (frames) |
