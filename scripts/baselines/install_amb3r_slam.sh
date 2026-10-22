#!/usr/bin/env bash
# Install AMB3R-SLAM (https://github.com/HengyiWang/amb3r-slam, pinned commit) for the baseline driver
# scripts/baselines/run_amb3r_slam.py:
#
#   AMB3R_ROOT=/path/to/dir bash scripts/baselines/install_amb3r_slam.sh
#
# Creates under $AMB3R_ROOT (default third_party/amb3r_slam):
#   amb3r-slam/   the repository at $AMB3R_COMMIT, its native pieces built (thirdparty/build.sh: DBoW2 + dpretrieval,
#                 ALIKED's get_patches op) and its checkpoints (checkpoints/download.sh: DA3NESTED-GIANT-LARGE-1.1,
#                 DA3-SMALL, ORBvoc.txt, ALIKED aliked-n16)
#   env/          Python 3.9 (micromamba, conda-forge: OpenCV C++ for DBoW2, cmake, compilers), torch 2.5.1 cu118 (the
#                 upstream tested setup: Python 3.9, PyTorch 2.5, CUDA 11.8), requirements.txt
# The runner finds them through AMB3R_DIR (= $AMB3R_ROOT/amb3r-slam) and AMB3R_PY (= $AMB3R_ROOT/env/bin/python).
# The ALIKED CUDA op is compiled with nvcc of CUDA 11.x (CUDA_HOME, default /usr/local/cuda) and the environment's own
# gcc / g++ 11 (the newest host compiler CUDA 11.x accepts).  Package caches go under $AMB3R_ROOT (small home disks).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${AMB3R_ROOT:-$HERE/third_party/amb3r_slam}"
COMMIT="${AMB3R_COMMIT:-54184655c259078bd60bc5ddebc838d8f5acda55}"
MAMBA="${MAMBA:-$(command -v micromamba || echo "$HOME/.local/bin/micromamba")}"
export MAMBA_ROOT_PREFIX="$ROOT/mamba" PIP_CACHE_DIR="$ROOT/cache/pip" HF_HOME="$ROOT/cache/hf" TMPDIR="$ROOT/tmp"
mkdir -p "$ROOT" "$TMPDIR"
cd "$ROOT"

# 1. repository
if [ ! -d amb3r-slam ]; then
    git clone https://github.com/HengyiWang/amb3r-slam.git
fi
git -C amb3r-slam checkout -q "$COMMIT"

# 2. environment
if [ ! -x env/bin/python ]; then
    "$MAMBA" create -y -p "$ROOT/env" -c conda-forge python=3.9 libopencv cmake make pkg-config "gxx_linux-64=11" \
        "gcc_linux-64=11" eigen
fi
PY="$ROOT/env/bin/python"
run() { "$MAMBA" run -p "$ROOT/env" "$@"; }
run pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
# the LiDAR modality's packages (kiss-icp, open3d) are imported only by its own modules: not needed for the camera setups
grep -v -E '^(kiss-icp|open3d)' amb3r-slam/requirements.txt > requirements_camera.txt
run pip install -r requirements_camera.txt

# 3. native pieces (thirdparty/build.sh, the compilers chosen per part)
TP="$ROOT/amb3r-slam/thirdparty"
run bash -c "cd '$TP' && cmake -S DBoW2 -B DBoW2/build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX='$TP/DBoW2/install' -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && cmake --build DBoW2/build -j \$(nproc) && cmake --build DBoW2/build --target install"
run bash -c "cd '$TP' && CMAKE_ARGS='-DDBoW2_DIR=$TP/DBoW2/install/lib/cmake/DBoW2 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
-DCMAKE_CXX_STANDARD=14 -DCMAKE_BUILD_RPATH=$TP/DBoW2/install/lib -DCMAKE_INSTALL_RPATH=$TP/DBoW2/install/lib' \
    pip install --no-build-isolation ./DPRetrieval"
run bash -c "cd '$TP/ALIKED/custom_ops' && CUDA_HOME='${CUDA_HOME:-/usr/local/cuda}' \
    CC='$ROOT/env/bin/x86_64-conda-linux-gnu-gcc' CXX='$ROOT/env/bin/x86_64-conda-linux-gnu-g++' \
    python setup.py build_ext --inplace"

# 4. checkpoints
run bash -c "cd '$ROOT/amb3r-slam' && bash checkpoints/download.sh"

"$PY" -c "import torch, dpretrieval, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cv2', cv2.__version__)"
echo "AMB3R_DIR=$ROOT/amb3r-slam AMB3R_PY=$PY"
