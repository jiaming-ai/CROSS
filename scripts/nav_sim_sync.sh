#!/bin/bash
# Copy the CROSS Nav simulation code to the H20s' shared JuiceFS folder: DEST = a copy of the coordinator's CROSS-nav code
# (cross_nav, vggt_ft, ...) with cross_nav/sim and scripts/nav_sim_*.sh from this tree on top.  Never deletes remotely.
set -e
cd "$(dirname "$0")/.."
HOST=${HOST:-h20-4}
SRC=${SRC:-/mnt/datasets-livsyn/jz/code/CROSS-nav}
DEST=${DEST:-/mnt/datasets-livsyn/jz/code/CROSS-navsim}
ssh $HOST "mkdir -p $DEST && rsync -a --exclude __pycache__ --exclude cross_nav/sim $SRC/ $DEST/" 2>/dev/null
rsync -az --exclude __pycache__ --exclude '*.pyc' -R cross_nav/sim scripts/nav_sim_*.sh $HOST:$DEST/ 2>/dev/null
echo "synced to $HOST:$DEST"
