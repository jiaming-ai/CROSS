#!/bin/bash
# evalwatch.sh <gpu> <run> [<run> ...]: evaluate every new checkpoint (ckpt_<step>_bf16.pt, ckpt_<step>_ema_bf16.pt) of
# the runs under $RUNS on the held-out suite -> $EVAL/<run>_<step>[_ema].json (+ .done); loops until killed.
# CKPT_GLOB limits the checkpoints (e.g. 'ckpt_*_ema_bf16.pt': EMA weights only).
GPU=$1; shift
RUNS=${RUNS:-/mnt/datasets-livsyn/jz/runs}; EVAL=${EVAL:-/mnt/datasets-livsyn/jz/eval}
cd "$(dirname "$0")/../.." && . /mnt/datasets-livsyn/jz/envs/ft/bin/activate
while true; do
  for run in "$@"; do
    for ck in $(ls $RUNS/$run/${CKPT_GLOB:-ckpt_*_bf16.pt} 2>/dev/null); do
      tag=$(basename $ck .pt | sed 's/^ckpt_//; s/_bf16$//'); out=$EVAL/${run}_${tag}.json
      # <out>.lock (atomic mkdir): several watchers can share runs without evaluating the same checkpoint twice
      if [ ! -e $out.done ] && mkdir $out.lock 2>/dev/null; then
        echo "$(date +%F_%T) eval $run $tag"
        CUDA_VISIBLE_DEVICES=$GPU python -m vggt_ft.evaluate --ckpt $ck --suite configs/vggt_ft/eval_suite.yaml --out $out \
          > /data0/jz/logs/eval_${run}_${tag}.log 2>&1 && touch $out.done
        rmdir $out.lock
      fi
      # validation split of the training datasets (checkpoint selection) for runs trained with data.val_mod
      vo=${out%.json}_val.json
      if [ -n "$VAL_RUNS" ] && echo " $VAL_RUNS " | grep -q " $run " && [ ! -e $vo.done ] && mkdir $vo.lock 2>/dev/null; then
        CUDA_VISIBLE_DEVICES=$GPU python -m vggt_ft.evaluate --ckpt $ck --suite configs/vggt_ft/val_suite.yaml \
          --out $vo > /data0/jz/logs/eval_${run}_${tag}_val.log 2>&1 && touch $vo.done
        rmdir $vo.lock
      fi
    done
  done
  sleep 300
done
