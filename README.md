<div align="center">

# CROSS: Change-Robust Online Topological Memory for Long-Term Relocalization and Semantic Navigation

### NeurIPS 2026

Jiaming Wang, Jizhuo Chen, Diwen Liu, Atharva Ghotavadekar, Jiaxuan Da, Linh Kästner, Harold Soh

National University of Singapore

[![Project Page](https://img.shields.io/badge/Project-Page-e8622c)](https://jiaming.im/CROSS/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.02227-b31b1b.svg)](https://arxiv.org/abs/2605.02227)
![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<a href="https://jiaming.im/CROSS/"><img src="assets/teaser.jpg" alt="A quadruped relocalizes in a crowded canteen with a map built the evening before, then navigates to a language goal" width="100%"></a>

</div>

**Pose-aware topological mapping for RGB-D and monocular inputs.**

CROSS builds probabilistic topological maps from RGB-D or monocular camera streams. It maintains a Gaussian mixture belief over SE(3) poses, tracks multiple hypotheses, detects loop closures, and optimizes pose graphs — enabling robust long-term navigation in indoor environments.

## Key Features

- **Multi-hypothesis tracking** — Gaussian mixture model (GMM) over SE(3) with evidence-driven lifecycle (birth, realization, removal).
- **Loop closure** — Overlap-based detection with asynchronous pose graph optimization (GTSAM) and hypothesis merging.
- **Topological planning** — Lightweight graph over keyframes with odometry and proximity edges; supports A\* and Dijkstra path planning.
- **Visual place recognition** — Keyframe database with embedding-based retrieval for relocalization.
- **Semantic memory** — Text-conditioned object search across the map using open-vocabulary detectors.
- **Multiple dataset formats** — R3D, ROS bags, OpenLORIS, TUM RGB-D.

## Architecture

```
cross/
├── core/           # System pipeline, hypothesis management, PGO, planning
├── cv/             # Pose estimation (PnP, VGGT), feature extraction, detection
├── db/             # Keyframe database and visual place recognition
├── dataloader/     # Dataset loaders (R3D, ROS bag, OpenLORIS, TUM)
├── utils/          # Math (Lie algebra, rotations), profiling, camera models
└── visualization/  # Rerun-based 3D visualization, graph plotting
```

## Installation

### Quick Start (recommended)

```bash
git clone https://github.com/jiaming-ai/CROSS.git
cd CROSS
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

### Benchmark: mapping accuracy and relocalization success

`scripts/eval/map_and_reloc.py` builds a map on one posed RGB-D sequence and measures relocalization success on another
(independent 100-frame trials that start without knowing the pose; a trial succeeds when its final estimate is within
`--r-d` of the pose the map implies, 2 m indoors), plus the keyframe ATE of the map.

```bash
# OpenLORIS-Scene (package format) -> posed RGB-D folders; the robot's wheel odometry is kept and used as odometry
python scripts/eval/convert_openloris.py data/openloris/home1-1 data/posed/home1-1
python scripts/eval/convert_openloris.py data/openloris/home1-2 data/posed/home1-2
python scripts/eval/map_and_reloc.py --map data/posed/home1-1 --query data/posed/home1-2 --out outputs/home
# TUM RGB-D: scripts/eval/convert_tum.py (simulated noisy odometry: --snr 10 --seed 0)
```

### Noise calibration for a new robot (optional)

The verified loop closure uses a noise model of the relative-pose estimator and the odometry.  The defaults work for
wheeled indoor robots; for another platform, record about a minute of data and calibrate without ground truth:

```bash
python scripts/eval/map_and_reloc.py --map <seq> --query <seq> --out outputs/calib --map-end 600 --skip-reloc --dump-graph
python scripts/eval/calibrate_noise.py --graph outputs/calib/graph_s0.json --out configs/noise/my_robot.yaml
# then: mapping.loop_closure.noise_file: configs/noise/my_robot.yaml
```

Main defaults: verified loop closure (consistency tests at one chi-square level), hypothesis 0 updated only by
measurements that are more informative than the odometry chain, keyframe images stored as uint8.  For speed, observation
gating (`pose_est.obs_min_translation: 0.3`, `obs_min_rotation: 0.15`, `obs_max_interval_steps: 3`) runs 1.4-1.8x faster;
it is off by default because it cost relocalization success on one real-robot scene (OpenLORIS home).

## Datasets

### OpenLORIS

Download the package format from [Hugging Face](https://huggingface.co/datasets/shixuesong/openloris-scene) (see the
[dataset page](https://lifelong-robotic-vision.github.io/dataset/scene.html)) and convert sequences with
`scripts/eval/convert_openloris.py` (above); the legacy `data/loris/` loader (`--loader loris`) still works.

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
