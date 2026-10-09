#!/bin/bash
# Copy the CROSS Nav code (cross_nav, vggt_ft, the vendored VGGT-Omega, configs) to the shared JuiceFS folder of the H20
# servers.  Never deletes on the remote side.
set -e
cd "$(dirname "$0")/.."
DEST=${DEST:-/mnt/datasets-livsyn/jz/code/CROSS-nav}
ssh h20-2 "mkdir -p $DEST"
rsync -az --exclude __pycache__ --exclude '*.pyc' -R cross cross_world cross_nav vggt_ft tests configs scripts benchmark/eval configs/cross_nav scripts/nav_* third_party/vggt-omega pyproject.toml \
  h20-2:$DEST/ 2>/dev/null
echo "synced to h20-2:$DEST"
