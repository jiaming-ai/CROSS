#!/usr/bin/env bash
set -euo pipefail

# CROSS — One-script installation
#
# Usage:
#   bash install.sh              # RGB-D modes with PnP (rgbd_wheel, rgbd_basalt)
#   bash install.sh --stereo     # + the VGGT-Omega modes (stereo_wheel / _basalt / _imu, rgbd_ff_*, mono_wheel,
#                                #   mono_basalt) with the fine-tuned VGGT-Omega checkpoint (public:
#                                #   huggingface.co/Jamie-nus/VGGT-Omega-CROSS)
#   bash install.sh --stereo --vggt=released  # the released VGGT-Omega checkpoint instead (gated: needs a Hugging Face
#                                #   login with access to facebook/VGGT-Omega); --vggt=both downloads both
#   bash install.sh --stereo --da3   # + the optional Depth Anything 3 backend of the stereo mode
#   bash install.sh --mono       # + every other mode: DPVO visual odometry (built from source with the CUDA toolkit of
#                                #   PyTorch), Depth Anything 3, LightGlue; implies --stereo --da3
#   bash install.sh --nav        # + navigation (cross_nav: BEV head, trajectory policy, planners, robot agent); implies
#                                #   --stereo
#   bash install.sh --nav-sim    # + the Habitat simulator for closed-loop navigation tests (its own Python 3.9
#                                #   environment in third_party/install/habitat; needs micromamba, mamba or conda)
#   bash install.sh --cpu        # CPU-only PyTorch (tests only: the mapping modes need a CUDA GPU)
#   CUDA_VERSION=cu121 bash install.sh  # PyTorch's CUDA build (default: cu128 on Blackwell GPUs, else cu124)

CUDA_VERSION="${CUDA_VERSION:-}"
CPU_ONLY=false
STEREO=false
DA3=false
MONO=false
NAV=false
NAV_SIM=false
VGGT=finetuned      # VGGT-Omega checkpoint: finetuned | released | both

for arg in "$@"; do
    case "$arg" in
        --cpu) CPU_ONLY=true ;;
        --stereo) STEREO=true ;;
        --da3) DA3=true; STEREO=true ;;
        --mono) MONO=true; DA3=true; STEREO=true ;;
        --nav) NAV=true; STEREO=true ;;
        --nav-sim) NAV_SIM=true; NAV=true; STEREO=true ;;
        --vggt=*) VGGT="${arg#--vggt=}" ;;
        -h|--help) sed -n '4,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "!! unknown option '$arg' (bash install.sh --help)"; exit 1 ;;
    esac
done
case "$VGGT" in
    finetuned|released|both) ;;
    *) echo "!! --vggt=finetuned | released | both, not '$VGGT'"; exit 1 ;;
esac

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
VENV_DIR="$(pwd)/.venv"
PYTHON_EXE="$VENV_DIR/bin/python"

# ── Install PyTorch ───────────────────────────────────────────
echo "==> Installing PyTorch..."
if [ "$CPU_ONLY" = true ]; then
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    if [ -z "$CUDA_VERSION" ]; then
        # Blackwell GPUs (compute capability 10.x / 12.x) need CUDA 12.8 builds; cu124 runs on older drivers too
        CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
        if [ -n "$CAP" ] && [ "${CAP%%.*}" -ge 10 ]; then CUDA_VERSION=cu128; else CUDA_VERSION=cu124; fi
    fi
    echo "    CUDA build: $CUDA_VERSION (set CUDA_VERSION to change it)"
    uv pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
fi

# ── Install CROSS ─────────────────────────────────────────────
echo "==> Installing CROSS..."
uv pip install -e ".[all]"

# ── Install GTSAM ─────────────────────────────────────────────
echo "==> Installing GTSAM (pose graph optimization)..."
if ! uv pip install "gtsam>=4.3"; then
    # no prebuilt wheel for this platform: build the Python bindings from source (needs cmake, ninja, a C++ compiler)
    echo "    No GTSAM wheel; building it from source..."
    GTSAM_DIR="thirdparty/gtsam"
    if [ ! -d "$GTSAM_DIR" ]; then
        mkdir -p thirdparty
        git clone --depth 1 https://github.com/borglab/gtsam.git "$GTSAM_DIR"
    fi
    PYTHON_VERSION=$("$PYTHON_EXE" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    (
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
        cd python && uv pip install .
    )
fi

# ── Stereo mode (optional) ────────────────────────────────────
if [ "$STEREO" = true ]; then
    echo "==> Installing the stereo mode (feed-forward estimator, vendored under third_party/)..."
    uv pip install -e ".[stereo]"
    uv pip install --no-deps -e third_party/vggt-omega
    # CROSS uses the fine-tuned checkpoint when it is installed (with its settings, configs/base/vggt_finetuned.yaml), else
    # the released one with the default settings; --set pose_est.ff.checkpoint=released picks the released one
    if [ "$VGGT" != released ] && [ ! -f models/VGGT-Omega-CROSS/vggt_omega_cross_1b_512.pt ]; then
        echo "    Downloading the fine-tuned VGGT-Omega-1B-512 (https://huggingface.co/Jamie-nus/VGGT-Omega-CROSS, 2.3 GB)..."
        "$PYTHON_EXE" -c "from huggingface_hub import hf_hub_download; hf_hub_download('Jamie-nus/VGGT-Omega-CROSS', 'vggt_omega_cross_1b_512.pt', local_dir='models/VGGT-Omega-CROSS')" \
            || echo "    !! download failed: place the file at models/VGGT-Omega-CROSS/vggt_omega_cross_1b_512.pt"
    fi
    if [ "$VGGT" != finetuned ] && [ ! -f models/VGGT-Omega/vggt_omega_1b_512.pt ]; then
        echo "    Downloading the released VGGT-Omega-1B-512 (gated: request access at https://huggingface.co/facebook/VGGT-Omega)..."
        "$PYTHON_EXE" -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/VGGT-Omega', 'vggt_omega_1b_512.pt', local_dir='models/VGGT-Omega')" \
            || echo "    !! download failed: log in (huggingface-cli login) after access is granted, or place the file at models/VGGT-Omega/vggt_omega_1b_512.pt"
    fi
    if [ "$DA3" = true ]; then
        echo "==> Installing the Depth Anything 3 backend..."
        uv pip install -e ".[da3]"
        uv pip install --no-deps -e third_party/depth-anything-3
        [ -d models/DA3-LARGE-1.1 ] || "$PYTHON_EXE" -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/DA3-LARGE-1.1', local_dir='models/DA3-LARGE-1.1')" \
            || echo "    !! DA3 download failed: place the checkpoint under models/DA3-LARGE-1.1/"
    fi
fi

# ── Mono mode and visual odometry (optional) ──────────────────
if [ "$MONO" = true ]; then
    echo "==> Installing the mono mode / visual odometry (DPVO, LightGlue)..."
    uv pip install -e ".[mono]"
    uv pip install 'git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8'
    # torch-scatter (DPVO): a prebuilt wheel of PyG for this PyTorch build when there is one, else built from source
    TORCH_TAG=$("$PYTHON_EXE" -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'{v[0]}.{v[1]}.0+' + ('cu' + torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu'))")
    uv pip install --only-binary torch-scatter torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_TAG}.html" \
        || uv pip install --no-build-isolation torch-scatter \
        || echo "    !! torch-scatter could not be installed for torch ${TORCH_TAG} (DPVO needs it)"
    DPVO_DIR="thirdparty/DPVO"
    if [ ! -d "$DPVO_DIR" ]; then
        git clone https://github.com/princeton-vl/DPVO.git "$DPVO_DIR"
        git -C "$DPVO_DIR" checkout 0ac95b656d1fda91c271d2a106460d19ad966fc7
        # PyTorch >= 2.8 removed the deprecated tensor dispatch used by DPVO's extensions (patches/)
        git -C "$DPVO_DIR" apply "$(pwd)/patches/dpvo-torch28-dispatch.patch" || echo "    (DPVO patch not applied)"
        mkdir -p "$DPVO_DIR/thirdparty"
        [ -d "$DPVO_DIR/thirdparty/eigen-3.4.0" ] || (cd "$DPVO_DIR/thirdparty" && \
            curl -LO https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && unzip -q eigen-3.4.0.zip) \
            || echo "    !! Eigen 3.4.0 download failed: unzip it into $DPVO_DIR/thirdparty/eigen-3.4.0"
    fi
    (cd "$DPVO_DIR" && "$PYTHON_EXE" setup.py build_ext --inplace) || echo "    !! DPVO build failed (needs nvcc matching PyTorch's CUDA)"
    # make DPVO importable from the environment (no PYTHONPATH needed)
    SITE_DIR=$("$PYTHON_EXE" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    echo "$(pwd)/$DPVO_DIR" > "$SITE_DIR/cross_dpvo.pth"
    mkdir -p models
    # the released DPVO weights (the link of DPVO's download_models_and_data.sh)
    DPVO_MODELS="https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AK0DY9WLZ25elq-7N7sJcB0/models.zip?rlkey=4j336vcc4h324j4p4ji3ppl0t&dl=1"
    [ -f models/dpvo.pth ] || (TMP=$(mktemp -d) && curl -sL -o "$TMP/dpvo_models.zip" "$DPVO_MODELS" && \
        unzip -o -q "$TMP/dpvo_models.zip" -d "$TMP" && cp "$TMP/dpvo.pth" models/dpvo.pth && rm -rf "$TMP") \
        || echo "    !! place the released DPVO weights at models/dpvo.pth"
fi

# ── Navigation (optional) ─────────────────────────────────────
if [ "$NAV" = true ]; then
    echo "==> Installing navigation (cross_nav)..."
    uv pip install -e ".[nav]"
fi
if [ "$NAV_SIM" = true ]; then
    # the simulator runs in its own environment (habitat-sim's conda build) and talks to the policy server of this
    # environment over a socket (cross_nav/policy/README.md)
    MM=$(command -v micromamba || command -v mamba || command -v conda || true)
    if [ -z "$MM" ]; then
        echo "    !! --nav-sim needs micromamba, mamba or conda on the PATH"
    else
        echo "==> Creating the Habitat simulator environment (third_party/install/habitat)..."
        "$MM" create -y -p "$(pwd)/third_party/install/habitat" -c conda-forge -c aihabitat \
            habitat-sim=0.3.3 headless python=3.9 numpy pillow \
            || echo "    !! habitat-sim install failed"
    fi
fi

echo ""
echo "==> Installation complete!"
echo ""
echo "    Activate the environment, then run from this folder:"
echo "      source .venv/bin/activate"
echo "      python -m pytest -q                                          # check the installation"
echo "      python run.py <sequence> --mode rgbd_wheel                   # RGB-D + odometry"
if [ "$STEREO" = true ]; then
    echo "      python run.py <sequence> --mode stereo_wheel                 # stereo + odometry"
fi
if [ "$MONO" = true ]; then
    echo "      python run.py <sequence> --mode mono_dpvo                    # one camera, no odometry"
fi
echo "      python run.py <sequence> --mode <mode> --save-map maps/a.pkl # build and save a map"
echo "      python run.py <sequence> --mode <mode> --load-map maps/a.pkl # relocalize in it"
if [ "$NAV" = true ]; then
    echo "    Navigation: cross_nav/README.md"
fi
echo "    Guide: docs/GUIDE.md"
echo ""
