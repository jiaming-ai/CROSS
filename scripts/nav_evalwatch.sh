#!/bin/bash
# Evaluate every new EMA checkpoint of the listed runs on the fixed window sets (one GPU, loops forever).
#   bash scripts/nav_evalwatch.sh <gpu> "<run> [run...]" "<windows.json> [windows.json...]"
# Results: /mnt/datasets-livsyn/jz/nav/eval/<run>/<windows name>/<ckpt>.json (+ figs/)
GPU=$1; RUNS=$2; WINS=$3
source /mnt/datasets-livsyn/jz/code/nav_env.sh
cd $NAV
export CUDA_VISIBLE_DEVICES=$GPU
while true; do
  for r in $RUNS; do
    for ck in $(ls /mnt/datasets-livsyn/jz/nav/runs/$r/heads_*_ema.pt 2>/dev/null | sort); do
      s=$(basename $ck .pt)
      for w in $WINS; do
        wn=$(basename $w .json)
        out=/mnt/datasets-livsyn/jz/nav/eval/$r/$wn/$s.json
        [ -f $out ] && continue
        mkdir -p $(dirname $out)
        echo "$(date +%H:%M) eval $r $s $wn"
        # the baselines do not depend on the checkpoint: computed with the first one evaluated on a window set
        nb=""; ls $(dirname $out)/heads_*.json >/dev/null 2>&1 && nb="--no-baselines"
        python -m cross_nav.evaluate run --windows $w --heads $ck --out $out --figs $(dirname $out)/figs_$s --n-figs 4 $nb \
          > $(dirname $out)/$s.log 2>&1 || echo "failed $r $s $wn"
      done
    done
  done
  sleep 120
done
