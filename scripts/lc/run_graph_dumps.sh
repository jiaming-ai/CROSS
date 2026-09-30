#!/usr/bin/env bash
# Record a mapping session (+ optional relocalization sessions) with the trace recorder and dump the pose graphs
# with ground truth (outputs/lcstudy/<scene>/<tag>/graph_s*.json) for the offline loop-closure study.
#   bash scripts/lc/run_graph_dumps.sh <gpu> <scene> <map_variant> [query variants...]
# env: TAG (default cur), SEED (0), EXTRA (extra record_trace.py arguments, e.g. --no-intra-lc)
set -uo pipefail
cd "$(dirname "$0")/../.."
GPU=$1; SCENE=$2; MAP=$3; shift 3; VARS="$*"
export CUDA_VISIBLE_DEVICES=$GPU MPLBACKEND=Agg
PY=.venv/bin/python
FF="--obs-min-translation 0.3 --obs-min-rotation 0.15 --obs-max-interval 3"
TAG=${TAG:-cur}; O=outputs/lcstudy/$SCENE/$TAG; mkdir -p $O logs
echo "[$(date +%F_%T)] START $SCENE/$TAG gpu $GPU seed ${SEED:-0} vars: $VARS" | tee -a logs/lcstudy.log
$PY scripts/viz/record_trace.py --scene $SCENE --map $MAP --variants $VARS --out $O --baseline 0.3 --snr 10 $FF --no-frames \
    --seed ${SEED:-0} ${EXTRA:-} > logs/lcstudy_${SCENE}_$TAG.log 2>&1
echo "[$(date +%F_%T)] DONE $SCENE/$TAG (exit $?) $(grep -o 'map ATE.*' $O/system.log | tail -1); merges $(grep -c 'LC detected' $O/system.log); intra PGOs $(grep -c 'Intra-hypothesis loop closure at' $O/system.log); rejected $(grep -c 'rejected' $O/system.log)" | tee -a logs/lcstudy.log
