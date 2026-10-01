# CROSS benchmark

A fixed evaluation protocol for CROSS and for SLAM systems compared with it. It has three tracks:

- **T1 mapping accuracy.** ATE of a single session.
- **T2 multi-session localization.** A later session is localized in a stored map.
- **T3 relocalization success.** The CROSS paper's metric.

The datasets are KITTI, OpenLORIS-Scene, ROVER and SimChange, each with RGB-D, stereo and monocular setups.

- [PROTOCOL.md](PROTOCOL.md): datasets, splits, setups, odometry, metrics and run rules.
- [RESULTS.md](RESULTS.md): result tables, generated from `results/results.json`.
- [DEV.md](DEV.md): a development split with the same protocol, to test a change in minutes (`dev.py`).
- [site/index.html](site/index.html): interactive results page with trajectories, localization error curves, trial outcomes
  and failure cases per system. Open it from the file system, or serve the `site/` folder.

## Layout

| path | content |
|---|---|
| `configs/datasets.yaml` | datasets, scenes, map/query splits, success radius, trial length |
| `configs/systems.yaml` | systems, their setups and how they are run |
| `datasets/` | download scripts (`download_*.sh`) and converters to the benchmark folder layout (`prepare_*.py`) |
| `run.py` | runs one job (`--task map | t1 | query`) and writes `result.json` ([schema](eval/schema.md)) |
| `jobs.py` | prints the job list of a dataset / set of systems (one `run.py` argument line per job) |
| `dev.py` | runs the development split and compares two runs ([DEV.md](DEV.md)); `datasets/make_dev.py` writes its clipped sequences |
| `eval/metrics.py` | ATE, completeness, localization recall, Wilson intervals |
| `collect.py`, `make_tables.py`, `build_site.py` | merge results, write RESULTS.md, write the page data (`site/data.js`) |
| `results/` | merged results (`results.json`) and earlier results obtained with other protocols (`legacy.*`) |

## Running

```bash
export BENCH_DATA=/path/to/benchmark_data BENCH_RESULTS=/path/to/runs
# 1. data (OpenLORIS-Scene shown; KITTI and ROVER likewise)
bash benchmark/datasets/download_openloris.sh /path/to/openloris
python benchmark/datasets/prepare_openloris.py /path/to/openloris $BENCH_DATA/openloris
python benchmark/datasets/prepare_kitti.py /path/to/kitti_raw $BENCH_DATA/kitti
NO_UNZIP=1 bash benchmark/datasets/download_rover.sh /path/to/rover campus_large     # keeps the zips (~320 GB)
python benchmark/datasets/prepare_rover.py /path/to/rover $BENCH_DATA/rover          # reads the frames from the zips
# 2. jobs: maps first, then single-session sequences, then query sessions (T2 + T3)
python benchmark/jobs.py --dataset openloris --systems cross_rgbd cross_stereo orbslam3 rtabmap > queue.txt
while read -r job; do python benchmark/run.py $job; done < queue.txt     # or distribute the lines over workers
# 3. tables and page
python benchmark/collect.py $BENCH_RESULTS && python benchmark/make_tables.py && python benchmark/build_site.py
```

Query jobs build the scene's map themselves when it is missing (with a lock, so parallel workers share one map).
A job whose input folders are not prepared yet exits with code 3 without writing a result.

Baselines: ORB-SLAM3 and RTAB-Map run through the drivers in `scripts/baselines/` (`orbslam3_reloc.cc`,
`rtabmap_reloc.cc`, built with `scripts/baselines/CMakeLists.txt` against ORB-SLAM3 and RTAB-Map). MASt3R-SLAM and
VGGT-SLAM run through `scripts/baselines/run_mast3r_slam.py` and `run_vggt_slam.py`; VGGT-SLAM installs with
`scripts/baselines/install_vggt_slam.sh`.

## Adding a system

Write a runner that produces the same `result.json` fields (see `run.py`: the CROSS, baseline and
concatenated-stream runners), and add an entry to `configs/systems.yaml` with its setups, whether it uses odometry, and how it
reuses a map.
