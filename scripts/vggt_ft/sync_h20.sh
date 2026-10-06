#!/bin/bash
# Copy the fine-tuning code (vggt_ft, configs, scripts, the vendored VGGT-Omega) to the shared JuiceFS folder of the
# H20 servers (seen by h20-1..4).  Never deletes on the remote side.
set -e
cd "$(dirname "$0")/../.."
DEST=${DEST:-/mnt/datasets-livsyn/jz/code/CROSS-ft}
ssh h20-2 "mkdir -p $DEST"
rsync -az --exclude __pycache__ --exclude '*.pyc' -R vggt_ft configs/vggt_ft scripts/vggt_ft third_party/vggt-omega pyproject.toml \
  h20-2:$DEST/ 2>/dev/null || rsync -az --exclude __pycache__ -R vggt_ft scripts/vggt_ft third_party/vggt-omega h20-2:$DEST/
echo "synced to h20-2:$DEST"
