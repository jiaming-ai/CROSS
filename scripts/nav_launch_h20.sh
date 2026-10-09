#!/bin/bash
# Launch a CROSS Nav training run in tmux on the current H20:  bash scripts/nav_launch_h20.sh <config> <gpus> [overrides...]
#   e.g. bash scripts/nav_launch_h20.sh configs/cross_nav/nav_v1.yaml 0,1,2,3
# Log: /mnt/datasets-livsyn/jz/nav/logs/<exp>.log ; tmux session jz_<exp>.  resume: auto continues from ckpt_last.pt.
set -e
CFG=$1; GPUS=$2; shift 2
EXP=$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))['exp_name'])" 2>/dev/null || basename $CFG .yaml)
for ov in "$@"; do case $ov in exp_name=*) EXP=${ov#exp_name=};; esac; done
N=$(echo $GPUS | tr ',' '\n' | wc -l)
PORT=$((29500 + RANDOM % 1000))
LOG=/mnt/datasets-livsyn/jz/nav/logs/$EXP.log
tmux new -d -s jz_$EXP "source /mnt/datasets-livsyn/jz/code/nav_env.sh; cd \$NAV; export CUDA_VISIBLE_DEVICES=$GPUS; \
  torchrun --nproc_per_node=$N --master_port=$PORT -m cross_nav.train --config $CFG $* 2>&1 | grep --line-buffered -v 'skip window' >> $LOG"
echo "started jz_$EXP on GPUs $GPUS, log $LOG"
