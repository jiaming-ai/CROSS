#!/usr/bin/env bash
# Install LingBot-Map (robbyant/lingbot-map) for the CROSS benchmark baseline (scripts/baselines/run_lingbot_map.py):
# the repository at a pinned commit, a Python 3.10 venv with torch 2.8.0 + cu128 (README), FlashInfer, and the
# lingbot-map.pt checkpoint (the README's paper / benchmark checkpoint) at a pinned Hugging Face revision.
#
#   bash scripts/baselines/install_lingbot_map.sh [DEST]      # default: third_party/lingbot_map
#
# Then point the runner at it (defaults match DEST=third_party/lingbot_map):
#   LINGBOT_MAP_DIR=$DEST/lingbot-map LINGBOT_MAP_PY=$DEST/venv/bin/python LINGBOT_MAP_CKPT=$DEST/ckpt/lingbot-map.pt
# Without FlashInfer, set LINGBOT_MAP_SDPA=1 (the upstream --use_sdpa backend).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:-$ROOT/third_party/lingbot_map}"
COMMIT=b9dd1b1de535e6d36e777fb42b8433c781bfd783          # 2026-10-14 "docs: correct camera pose convention comments"
HF_REVISION=204754b72bb24f561f8d7e7e1e4e4cd9e809adf9     # robbyant/lingbot-map
UV="${UV:-uv}"
mkdir -p "$DEST" && cd "$DEST"
if [ ! -d lingbot-map ]; then git clone https://github.com/robbyant/lingbot-map.git; fi
git -C lingbot-map fetch -q origin && git -C lingbot-map checkout -q "$COMMIT"
"$UV" venv --python 3.10 venv
"$UV" pip install --python venv/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python venv/bin/python -e ./lingbot-map
"$UV" pip install --python venv/bin/python --index-url https://pypi.org/simple flashinfer-python \
  || echo "FlashInfer did not install: run with LINGBOT_MAP_SDPA=1"
mkdir -p ckpt
venv/bin/python - "$HF_REVISION" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download("robbyant/lingbot-map", "lingbot-map.pt", revision=sys.argv[1], local_dir="ckpt"))
EOF
echo "LingBot-Map installed in $DEST"
