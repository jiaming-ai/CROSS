#!/usr/bin/env bash
set -euo pipefail

# CROSS — One-script installation
#
# Usage:
#   bash install.sh              # Full install (CUDA 12.4)
#   bash install.sh --cpu        # CPU-only (no CUDA)
#   CUDA_VERSION=cu121 bash install.sh  # Specify CUDA version

CUDA_VERSION="${CUDA_VERSION:-cu124}"
CPU_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --cpu) CPU_ONLY=true ;;
    esac
done

echo "==> CROSS installer"

# ── Check for uv ──────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> uv $(uv --version)"

# ── Create venv ───────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment (.venv, Python 3.11)..."
    uv venv --python 3.11
fi

# ── Install PyTorch ───────────────────────────────────────────
echo "==> Installing PyTorch..."
if [ "$CPU_ONLY" = true ]; then
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    uv pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
fi

# ── Install CROSS ─────────────────────────────────────────────
echo "==> Installing CROSS..."
uv pip install -e ".[all]"

# ── Install GTSAM ─────────────────────────────────────────────
echo "==> Installing GTSAM (pose graph optimization)..."

GTSAM_DIR="thirdparty/gtsam"
if [ ! -d "$GTSAM_DIR" ]; then
    echo "    Cloning GTSAM..."
    mkdir -p thirdparty
    git clone --depth 1 https://github.com/borglab/gtsam.git "$GTSAM_DIR"
fi

VENV_DIR="$(pwd)/.venv"
PYTHON_EXE="$VENV_DIR/bin/python"
PYTHON_VERSION=$("$PYTHON_EXE" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "    Building GTSAM Python bindings (Python ${PYTHON_VERSION})..."
cd "$GTSAM_DIR"
uv pip install -r python/dev_requirements.txt

# Disable -Werror (fails with GCC 13+ due to Eigen false positives)
sed -i 's/-Werror //' cmake/GtsamBuildTypes.cmake
sed -i 's/-Werror=format-security//' cmake/GtsamBuildTypes.cmake

mkdir -p build && cd build
PATH="$VENV_DIR/bin:$PATH" cmake .. \
    -DGTSAM_BUILD_PYTHON=1 \
    -DGTSAM_PYTHON_VERSION="$PYTHON_VERSION" \
    -DPython_ROOT_DIR="$VENV_DIR" \
    -DPython_FIND_VIRTUALENV=ONLY \
    -GNinja 2>&1 | tail -5
PATH="$VENV_DIR/bin:$PATH" ninja 2>&1 | tail -5

# Install the built bindings into the venv (ninja python-install requires pip)
cd python
uv pip install .
cd ../../../..

echo ""
echo "==> Installation complete!"
echo ""
echo "    Run CROSS (no activation needed):"
echo "      uv run python run.py data/r3d/lab2.r3d"
echo ""
echo "    Or activate the environment first:"
echo "      source .venv/bin/activate"
echo "      python run.py data/r3d/lab2.r3d"
echo ""
echo "    Examples:"
echo "      uv run python examples/demo.py"
echo "      uv run python examples/multi_session.py scene1.r3d scene2.r3d"
echo "      uv run python examples/planner.py --map-scene data/r3d/lab.r3d --reloc-scene data/rosbag/lab"
echo ""
