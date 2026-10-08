#!/bin/bash
# Launch vggt_ft.train on one H20 node (run once per node; NODE_RANK 0 is the master).
#   GPUS=0,1,2,3 NNODES=2 NODE_RANK=0 MASTER=192.168.26.62 scripts/vggt_ft/launch_h20.sh configs/vggt_ft/<run>.yaml [overrides]
# (NODE_RANK only names the log; with NNODES > 1 the c10d rendezvous on MASTER assigns ranks, GPU counts may differ)
# Inter-node traffic goes over RoCE (mlx5_1..4, GID 7; settings from the platform's NCCL test).  Logs to
# /data0/jz/logs/<exp>_node<rank>.log.
set -e
CFG=$1; shift
cd ${CODE_DIR:-/mnt/datasets-livsyn/jz/code/CROSS-ft}     # CODE_DIR: another deployed copy
. /mnt/datasets-livsyn/jz/envs/ft/bin/activate
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4 NCCL_IB_GID_INDEX=7 NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_PCI_RELAXED_ORDERING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# EXTRA_PYTHONPATH: e.g. the labelers' packages for loss.da3_teacher
[ -n "$EXTRA_PYTHONPATH" ] && export PYTHONPATH=$EXTRA_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}
EXP=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['exp_name'])")
for o in "$@"; do case $o in exp_name=*) EXP=${o#exp_name=};; esac; done
mkdir -p /data0/jz/logs
if [ "${NNODES:-1}" -gt 1 ]; then      # c10d rendezvous: nodes may run different numbers of GPUs
  RDZV="--rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT:-29511} --rdzv_id=${EXP}"
else
  RDZV="--master_addr=127.0.0.1 --master_port=${PORT:-29511}"
fi
exec torchrun --nnodes=${NNODES:-1} --nproc_per_node=$NPROC $RDZV \
  -m vggt_ft.train --config $CFG "$@" >> /data0/jz/logs/${EXP}_node${NODE_RANK:-0}.log 2>&1
