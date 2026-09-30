#!/usr/bin/env bash
set -euo pipefail

# CROSS — One-script installation
#
# Usage:
#   bash install.sh              # Full install (CUDA 12.4), RGB-D mode
#   bash install.sh --stereo     # + stereo mode (VGGT-Omega feed-forward estimator; weights need a Hugging Face login
#                                #   with access to facebook/VGGT-Omega)
#   bash install.sh --stereo --da3   # + the optional Depth Anything 3 backend of the stereo mode
#   bash install.sh --cpu        # CPU-only (no CUDA)
#   CUDA_VERSION=cu121 bash install.sh  # Specify CUDA version

CUDA_VERSION="${CUDA_VERSION:-cu124}"
CPU_ONLY=false
STEREO=false
DA3=false

for arg in "$@"; do
    case "$arg" in
        --cpu) CPU_ONLY=true ;;
        --stereo) STEREO=true ;;
        --da3) DA3=true; STEREO=true ;;
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

# ── Stereo mode (optional) ────────────────────────────────────
if [ "$STEREO" = true ]; then
    echo "==> Installing the stereo mode (feed-forward estimator, vendored under third_party/)..."
    uv pip install -e ".[stereo]"
    uv pip install --no-deps -e third_party/vggt-omega
    mkdir -p models/VGGT-Omega
    if [ ! -f models/VGGT-Omega/vggt_omega_1b_512.pt ]; then
        echo "    Downloading VGGT-Omega-1B-512 (gated: request access at https://huggingface.co/facebook/VGGT-Omega)..."
        .venv/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/VGGT-Omega', 'vggt_omega_1b_512.pt', local_dir='models/VGGT-Omega')" \
            || echo "    !! download failed: log in (huggingface-cli login) after access is granted, or place the file at models/VGGT-Omega/vggt_omega_1b_512.pt"
    fi
    if [ "$DA3" = true ]; then
        echo "==> Installing the Depth Anything 3 backend..."
        uv pip install -e ".[da3]"
        uv pip install --no-deps -e third_party/depth-anything-3
        [ -d models/DA3-LARGE-1.1 ] || .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/DA3-LARGE-1.1', local_dir='models/DA3-LARGE-1.1')" \
            || echo "    !! DA3 download failed: place the checkpoint under models/DA3-LARGE-1.1/"
    fi
fi

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
echo "    Stereo mode (after install.sh --stereo):"
echo "      uv run python run.py <stereo sequence> --mode stereo"
echo ""
echo "    Examples:"
echo "      uv run python examples/demo.py"
echo "      uv run python examples/multi_session.py scene1.r3d scene2.r3d"
echo "      uv run python examples/planner.py --map-scene data/r3d/lab.r3d --reloc-scene data/rosbag/lab"
echo ""
