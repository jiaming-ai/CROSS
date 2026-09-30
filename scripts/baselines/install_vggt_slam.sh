#!/usr/bin/env bash
# Install the VGGT-SLAM 2.0 baseline (https://github.com/MIT-SPARK/VGGT-SLAM, main = v2.0) into a target dir.
#
#   bash scripts/baselines/install_vggt_slam.sh [TARGET_DIR]      # default /home/storage/jiaming/vggt_slam
#
# Layout after install:
#   TARGET/VGGT-SLAM                 repo (+ third_party/{salad,vggt}); pinned commits below
#   TARGET/.venv                     Python 3.11 venv (uv); use TARGET/.venv/bin/python
#   TARGET/torch_home/hub/...        VGGT-1B weights (model.pt), SALAD checkpoint, DINOv2 hub repo + weights
# Runtime needs TORCH_HOME=TARGET/torch_home (run_vggt_slam.py sets it from the Python path).
# Only tracking/mapping + loop closure are installed: SAM 3 / Perception Encoder (optional --run_os
# open-set detection) are skipped.  Needs network, no GPU.  Venv lives on local disk (NAS is noexec).
set -euo pipefail

TARGET=$(realpath -m "${1:-/home/storage/jiaming/vggt_slam}")
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
VGGT_SLAM_COMMIT=35327ac28b7d193df9ccc39ba6346052bb6f1207   # VGGT-SLAM main, 2026-06-29
SALAD_COMMIT=33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d        # Dominic101/salad
VGGT_COMMIT=6e6e16107b88e8e76c751826af10d4295d87ecd2         # MIT-SPARK/VGGT_SPARK

mkdir -p "$TARGET"
cd "$TARGET"
export UV_CACHE_DIR="$TARGET/.uv_cache"          # keep the cache off quota-limited home dirs
export TORCH_HOME="$TARGET/torch_home"

clone() {  # url dir commit
    [ -d "$2/.git" ] || git clone -q "$1" "$2"
    git -C "$2" fetch -q origin && git -C "$2" checkout -q "$3"
}
clone https://github.com/MIT-SPARK/VGGT-SLAM.git VGGT-SLAM "$VGGT_SLAM_COMMIT"
mkdir -p VGGT-SLAM/third_party
clone https://github.com/Dominic101/salad.git VGGT-SLAM/third_party/salad "$SALAD_COMMIT"
clone https://github.com/MIT-SPARK/VGGT_SPARK.git VGGT-SLAM/third_party/vggt "$VGGT_COMMIT"
rm -f VGGT-SLAM/office_loop.zip                  # 17 MB demo data, not needed

[ -x .venv/bin/python ] || "$UV" venv -q -p 3.11 .venv
PY="$TARGET/.venv/bin/python"
# setup.sh steps 1-3 and 6 (requirements, salad, VGGT fork, the repo itself); torch 2.3.1 wheels are cu121
"$UV" pip install -p "$PY" -r VGGT-SLAM/requirements.txt
"$UV" pip install -p "$PY" -e VGGT-SLAM/third_party/salad -e VGGT-SLAM/third_party/vggt -e VGGT-SLAM
"$UV" pip install -p "$PY" "numpy<2"             # VGGT fork pins numpy<2 (torch 2.3.1 is built against numpy 1.x)

# Weights (the code downloads them lazily into TORCH_HOME; fetch now so runs need no network)
mkdir -p "$TORCH_HOME/hub/checkpoints"
[ -s "$TORCH_HOME/hub/checkpoints/model.pt" ] || \
    wget -q -O "$TORCH_HOME/hub/checkpoints/model.pt" https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt
[ -s "$TORCH_HOME/hub/checkpoints/dino_salad.ckpt" ] || \
    wget -q -O "$TORCH_HOME/hub/checkpoints/dino_salad.ckpt" https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
# DINOv2 backbone used by SALAD: torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
"$PY" -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')" >/dev/null

"$UV" cache clean -q || true
rm -rf "$UV_CACHE_DIR"
"$PY" -c "import gtsam, salad.eval, vggt.models.vggt, vggt_slam.solver; print('VGGT-SLAM install OK')"
du -sh "$TARGET"
