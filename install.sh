#!/usr/bin/env bash
set -euo pipefail

# CROSS — One-script installation
#
# Usage:
#   bash install.sh              # Full install (CUDA 12.4), RGB-D mode
#   bash install.sh --stereo     # + stereo mode (VGGT-Omega feed-forward estimator; weights need a Hugging Face login
#                                #   with access to facebook/VGGT-Omega)
#   bash install.sh --stereo --da3   # + the optional Depth Anything 3 backend of the stereo mode
#   bash install.sh --mono       # + mono mode and visual odometry (DPVO built from source with the CUDA toolkit of
#                                #   PyTorch, Depth Anything 3, LightGlue); implies --stereo --da3
#   bash install.sh --cpu        # CPU-only (no CUDA)
#   CUDA_VERSION=cu121 bash install.sh  # Specify CUDA version

CUDA_VERSION="${CUDA_VERSION:-cu124}"
CPU_ONLY=false
STEREO=false
DA3=false
MONO=false

for arg in "$@"; do
    case "$arg" in
        --cpu) CPU_ONLY=true ;;
        --stereo) STEREO=true ;;
        --da3) DA3=true; STEREO=true ;;
        --mono) MONO=true; DA3=true; STEREO=true ;;
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

# ── Mono mode and visual odometry (optional) ──────────────────
if [ "$MONO" = true ]; then
    echo "==> Installing the mono mode / visual odometry (DPVO, LightGlue)..."
    uv pip install -e ".[mono]"
    uv pip install 'git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8'
    TORCH_TAG=$(.venv/bin/python -c "import torch; v=torch.__version__.split('+'); print('pt'+''.join(v[0].split('.')[:2])+'0+'+(torch.version.cuda and 'cu'+torch.version.cuda.replace('.','') or 'cpu'))")
    uv pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_TAG#pt}.html" \
        || echo "    !! torch-scatter wheel not found for ${TORCH_TAG}; build it from source"
    DPVO_DIR="thirdparty/DPVO"
    if [ ! -d "$DPVO_DIR" ]; then
        git clone https://github.com/princeton-vl/DPVO.git "$DPVO_DIR"
        git -C "$DPVO_DIR" checkout 0ac95b656d1fda91c271d2a106460d19ad966fc7
        # PyTorch >= 2.8 removed the deprecated tensor dispatch used by DPVO's extensions (patches/)
        git -C "$DPVO_DIR" apply "$(pwd)/patches/dpvo-torch28-dispatch.patch" || echo "    (DPVO patch not applied)"
        [ -d "$DPVO_DIR/thirdparty/eigen-3.4.0" ] || (cd "$DPVO_DIR/thirdparty" && \
            curl -LO https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && unzip -q eigen-3.4.0.zip)
    fi
    (cd "$DPVO_DIR" && "$PYTHON_EXE" setup.py build_ext --inplace) || echo "    !! DPVO build failed (needs nvcc matching PyTorch's CUDA)"
    mkdir -p models
    [ -f models/dpvo.pth ] || (curl -L -o /tmp/dpvo_models.zip https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip && \
        unzip -o -q /tmp/dpvo_models.zip -d /tmp/dpvo_models && cp /tmp/dpvo_models/dpvo.pth models/dpvo.pth) \
        || echo "    !! place the released DPVO weights at models/dpvo.pth"
    echo "    Add DPVO to the path when running: export PYTHONPATH=$(pwd)/$DPVO_DIR:\$PYTHONPATH"
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
echo "    Mono mode, visual odometry (after install.sh --mono):"
echo "      uv run python run.py <sequence> --mode mono --odometry visual"
echo ""
echo "    Examples:"
echo "      uv run python examples/demo.py"
echo "      uv run python examples/multi_session.py scene1.r3d scene2.r3d"
echo "      uv run python examples/planner.py --map-scene data/r3d/lab.r3d --reloc-scene data/rosbag/lab"
echo ""
