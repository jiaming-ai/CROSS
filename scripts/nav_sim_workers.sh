#!/bin/bash
# Start CROSS Nav simulation workers on this H20 in tmux sessions jz_navsim_<stage><i> (survive disconnects).
#   bash nav_sim_workers.sh prep <n_workers> [scenes.txt]
#   bash nav_sim_workers.sh render "<gpu ids>" [scenes.txt]      # one worker per listed GPU (shared GPUs: the worker
#                                                                 # waits while the GPU has < 16 GB free)
# Blender uses every core it sees: each worker is pinned (taskset) to PREP_CORES / RENDER_CORES cores of its own
# (prep from core 96 up, render from core 148 up) so that the shared machine keeps most of its CPUs.
set -e
STAGE=$1; WHAT=$2; SCENES=${3:-/mnt/datasets-livsyn/jz/hssd/scenes_train.txt}
CODE=${CODE:-/mnt/datasets-livsyn/jz/code/CROSS-navsim}
export BPY_PYTHON=/mnt/datasets-livsyn/jz/envs/bpy/bin/python
export SIMCHANGE_ROOT=/mnt/datasets-livsyn/jz/code/SimChange
export NAVSIM_TMP=/data0/jz/navsim/tmp
export CUDA_CACHE_PATH=/data0/jz/cuda_cache CUDA_CACHE_MAXSIZE=4294967296
export NUMBA_CACHE_DIR=/data0/jz/numba_cache OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/mnt/datasets-livsyn/jz/opt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}    # libSM / libICE for bpy (missing on some H20s)
export PYTHONPATH=$CODE:$CODE/third_party/vggt-omega
mkdir -p /data0/jz/logs /data0/jz/navsim/tmp /data0/jz/cuda_cache
ENVS="BPY_PYTHON=$BPY_PYTHON SIMCHANGE_ROOT=$SIMCHANGE_ROOT NAVSIM_TMP=$NAVSIM_TMP CUDA_CACHE_PATH=$CUDA_CACHE_PATH CUDA_CACHE_MAXSIZE=$CUDA_CACHE_MAXSIZE NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR OMP_NUM_THREADS=$OMP_NUM_THREADS PYTHONPATH=$PYTHONPATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
H=$(hostname)
PC=${PREP_CORES:-4}; RC=${RENDER_CORES:-4}
if [ "$STAGE" = prep ]; then
  for i in $(seq 1 $WHAT); do
    c0=$((96 + (i - 1) * PC)); c1=$((c0 + PC - 1))
    tmux new -d -s jz_navsim_prep$i "cd /tmp && env $ENVS taskset -c $c0-$c1 $BPY_PYTHON -m cross_nav.sim.pipeline worker --stage prep --scenes $SCENES >> /data0/jz/logs/navsim_prep$i.log 2>&1"
  done
else
  for g in $WHAT; do
    c0=$((148 + g * RC)); c1=$((c0 + RC - 1))
    tmux new -d -s jz_navsim_render$g "cd /tmp && env $ENVS CUDA_VISIBLE_DEVICES=$g taskset -c $c0-$c1 $BPY_PYTHON -m cross_nav.sim.pipeline worker --stage render --gpu $g --scenes $SCENES >> /data0/jz/logs/navsim_render$g.log 2>&1"
  done
fi
tmux ls | grep navsim
