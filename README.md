# CROSS-Mono

Experimental monocular extension of [CROSS](https://arxiv.org/abs/2605.02227), built upon its global retrieval observation message, continuous multi-modal pose hypotheses and delayed commitment. Those existing formulations remain the foundation of this system. This separate research repository adds monocular inputs and uncertain learned metric priors.

The monocular API accepts **RGB, timestamps and camera calibration**. It does not consume sensor depth, input odometry, IMU readings or ground-truth poses. Learned metric depth supplies an uncertain prior; its absolute scale may remain biased out of distribution.

## Monocular architecture

- Motion: lightweight XFeat features with robust PnP against periodic metric-depth anchors. Optional person exclusion reduces foreground tracking. DA3 compact-memory and DPVO scalar-scale frontends are retained as experimental alternatives.
- Metric priors: DA3-Metric-Large supplies uncertain depth shape and scale. The DPVO alternative uses robust log-scale observations, persistent-alternative recovery and a systematic uncertainty floor.
- Mapping: the existing CROSS hypothesis lifecycle. Learned retrieval poses are verified with image correspondences; scale uncertainty enters the original diagonal SE(3) filter conservatively.
- Evaluation: separate ground-truth reader, rigid-aligned metric ATE, similarity-aligned ATE and fitted scale, metric RPE, tracking coverage and runtime including periodic inference.

The current scope is a single-session research baseline. The diagonal metric bridge is not a full joint Sim(3) posterior. Save/load starts a new session. A same-sequence restart can commit after accumulated evidence, but actual changed-session relocalization is not validated.

## Monocular installation and usage

The tested environment uses Linux, Python 3.11, PyTorch 2.5.1/CUDA 12.4 and an NVIDIA GPU. First install CROSS's dependencies using the instructions below, then add the monocular dependencies and pinned DA3 implementation:

```bash
uv pip install -e '.[mono,dev]'
uv pip install gtsam==4.2

git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git /path/to/Depth-Anything-3
git -C /path/to/Depth-Anything-3 checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
uv pip install --no-deps /path/to/Depth-Anything-3
```

For DPVO, follow its [official build instructions](https://github.com/princeton-vl/DPVO) at commit `0ac95b656d1fda91c271d2a106460d19ad966fc7`, with a CUDA toolkit matching PyTorch, matching `torch-scatter`, and `numba`. Add the built checkout to `PYTHONPATH` and obtain the official `dpvo.pth`. Models/binaries are not vendored. `HF_HOME` chooses the model cache and `CROSS_TORCH_HUB` can select an existing XFeat/BoQ hub cache.

```bash
# Fetch RGB and metadata; sensor-depth images are omitted during extraction.
python scripts/download_mono_benchmarks.py --root /path/to/benchmarks \
  --sequences freiburg1_desk freiburg1_xyz bonn_balloon2

python -m cross.mono.run /path/to/benchmarks/rgbd_dataset_freiburg1_desk \
  --frontend metric_pnp --mask-people --mask-interval 3 \
  --metric-interval 30 --mapping-interval 15 --retrieval-pose da3 \
  --output outputs/desk_mono --save-map

# DA3 baseline / ablation
python -m cross.mono.run /path/to/benchmarks/rgbd_dataset_freiburg1_desk \
  --frontend da3 --pose-refinement xfeat --refinement-anchor-only --metric-shape \
  --output outputs/desk_da3

# Ground truth is read only after inference.
python -m cross.mono.evaluate outputs/desk_mono/trajectory.txt \
  /path/to/benchmarks/rgbd_dataset_freiburg1_desk/groundtruth.txt \
  --output outputs/desk_mono/metrics.json

python scripts/summarize_mono_runs.py outputs --output outputs/summary.json
python -m pytest tests -q
```

The recommended profile above does not require DPVO or its CUDA extensions. The motion features and optional person detector are small; the metric teacher and retrieved-pair geometry run at lower rates. These calls are synchronous, so average throughput does not guarantee a fixed frame deadline.

The experimental streaming path moves metric inference and the existing CROSS mapping updates to bounded workers:

```bash
python -m cross.mono.run /path/to/benchmarks/rgbd_dataset_freiburg1_desk \
  --frontend streaming_pnp --mask-people --mask-interval 3 \
  --metric-interval 30 --mapping-interval 15 \
  --retrieval-pose metric_pnp --mapping-process --input-buffer 4 \
  --input-fps 20 --warmup-models --paced-input-worker \
  --output outputs/desk_streaming
```

The operating target is 20–30 FPS at 640×480 on one consumer GPU such as an RTX 4090. This is a target, not a verified hardware claim. Measure capture-to-pose latency and deadline misses as well as throughput; include initialization, mapping lag and combined GPU memory. `--warmup-models` uses only the first image and records its time separately. The paced input worker starts preprocessing after simulated capture time and fails on queue overflow (two frames by default, configurable with `--input-buffer`); it never silently drops evaluation frames. A bounded input queue does not itself bound capture-time lag when preprocessing falls behind. Local server storage avoids including NAS congestion in GPU comparisons.

The command above paces every selected image uniformly at 20 Hz. For playback at the original motion speed with approximately 20 images per second, add `--sample-fps 20 --replay-timestamps`. Sampling keeps the first available RGB image in each time bin; gaps remain gaps, timestamps and source indices remain unchanged, and the selected frame count is recorded. Arrival intervals can vary with the original capture cadence. `--input-fps 20` then specifies the 50 ms pose deadline while the recorded timestamps determine arrival times. Report this sampling protocol separately from all-frame throughput replay.

Teacher outputs retain their source image and timestamp. Mapping receives accumulated motion even if a pending low-rate image is replaced. Previously emitted poses are not rewritten. The mapping worker calls the inherited global observation-mixture, hypothesis filtering and delayed commitment code. `--mapping-process` optionally isolates that worker in a separate process on the same GPU; both processes' allocated-memory peaks are reported. This alone does not guarantee deadlines. The class-specific person detector batches its class postprocessing while retaining the pretrained scores and NMS.

`--delayed-recovery` enables experimental reverse-PnP correction when delayed depth arrives. It is disabled by default because it produced large pose jumps in development. Asynchronous scheduling can change anchor timing and trajectories; report repeated timed runs. The joint place/scale/shared-bias message extension remains research work, and has not yet been implemented or validated as a new contribution.

`--teacher-lag-frames L` optionally holds a ready metric result until its source is at least `L` frames old. This tests sensitivity to delivery timing without blocking the pose stream. Late results are applied when available and their lateness is recorded; it is not a guarantee of deterministic tracking. The default is zero. Final shutdown drains remaining results for mapping, while already emitted poses stay unchanged.

To reproduce a complete multi-seed suite on a remote server:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_mono.py \
  --data-root /path/to/benchmarks --output outputs/mono_suite \
  --sequences rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg1_xyz \
    rgbd_dataset_freiburg1_desk2 rgbd_dataset_freiburg1_room \
    rgbd_bonn_balloon2 rgbd_bonn_crowd2 \
    rgbd_dataset_freiburg2_xyz rgbd_bonn_balloon rgbd_bonn_crowd \
  --seeds 0 1 2
```

DPVO/scalar-only comparison: select `--frontend dpvo --dpvo-checkpoint /path/to/dpvo.pth`. `--dpvo-metric-bootstrap` is an experimental initialization option, not part of the recommended profile. `rotation_metric` tests DPVO rotation with metric-anchor translation. `--retrieval-pose metric_pnp` is an experimental bidirectional learned-depth verification adapter; it improves proposal availability and supports the same-sequence restart experiment; changed-session robustness remains unvalidated.

`--frontend-only` isolates motion/scale and does not require GTSAM. `--scale-mode` selects scalar-policy ablations for DA3/DPVO; metric-anchor PnP uses depth priors directly. `--stride` defaults to 1 (every RGB frame). Known TUM/Bonn sequence names select calibration and distortion; custom sequences require `--intrinsics fx fy cx cy` plus a TUM-format `rgb.txt`. DPVO currently requires image dimensions divisible by 16; select its physical GPU using `CUDA_VISIBLE_DEVICES`.

Each run records configuration, source fingerprint, model provenance, causal system/frontend trajectories, per-frame diagnostics and timing. Existing result directories are not overwritten. Failed PnP verification holds the pose and marks the frame invalid. DPVO validity records initialization only; it is not an independent accuracy or tracking-success check. Evaluate all emitted poses and report the validity fraction separately. Similarity alignment can conceal metric-scale error, so report both ATE alignments and the fitted scale. For Bonn rotational RPE, also request `--groundtruth-frame bonn-camera`: its raw mocap orientation has a different body-frame convention. Keep raw and calibrated results separate. `cross.mono.replay_scale` enables causal scalar-policy ablations using exactly shared DPVO geometry and teacher observations, without assigning replay outputs inference timing.

Monocular code lives in `cross/mono/`; inherited mapping is in `cross/core/`. Private research notes and runs belong in ignored `docs/` and `outputs/`. External models retain their own licenses. This implementation is informed by [AMB3R-SLAM](https://arxiv.org/abs/2609.19518); it is not a reproduction of that paper.

The remaining instructions document the inherited CROSS RGB-D interface.

## Installation

### Quick Start (recommended)

```bash
# From this checkout:
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

## Datasets

### OpenLORIS

Download the [TUM version](https://lifelong-robotic-vision.github.io/dataset/scene.html) and place it in `data/loris/`.

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
