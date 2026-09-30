#!/usr/bin/env bash
# Seeded A/B experiments for the loop-closure verification, the retrieval split and the intra-hypothesis PGO.
#   bash scripts/run_lc_experiments.sh <gpu> <job>
# jobs: lm_reloc_new | lm_reloc_legacy   relocalization sessions of Lone Monk on the stored map of the replay trace
#                                        (outputs/viz/lonemonk/trace_cross_stereo/map.pkl), new vs legacy LC test, seed 0
#       lm_map_ab                         Lone Monk mapping, seeds 0 1 2, intra-hypothesis PGO on/off
#       lm_map_split                      Lone Monk mapping, seed 0, retrieval split enabled (150) with the new score gate
#       hs_reloc_new | hs_reloc_legacy    HSSD house hardest variant on the fixed-code map, new vs legacy LC test
#       hs_map_ab                         HSSD house mapping, seed 0, intra on/off
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=$1; JOB=$2
export CUDA_VISIBLE_DEVICES=$GPU MPLBACKEND=Agg
PY=.venv/bin/python
FF="--obs-min-translation 0.3 --obs-min-rotation 0.15 --obs-max-interval 3"
OUT=outputs/lc; mkdir -p $OUT logs
log() { echo "[$(date +%F_%T)] $*" | tee -a logs/lc_experiments.log; }
LMV="reverse offset_2.0 light_evening+move_50+reverse+offset_1.0 light_night"
case $JOB in
  lm_reloc_new|lm_reloc_new2|lm_reloc_legacy)
    tag=${JOB#lm_reloc_}; O=$OUT/lonemonk/reloc_$tag; mkdir -p $O
    cp -n outputs/viz/lonemonk/trace_cross_stereo/map.pkl outputs/viz/lonemonk/trace_cross_stereo/trace_map.json $O/
    EXTRA=""; [ $tag = legacy ] && EXTRA="--legacy-lc"
    log "START $JOB"
    $PY scripts/viz/record_trace.py --scene lonemonk --map map_loop --variants $LMV --out $O --baseline 0.3 --snr 10 $FF --no-frames --skip-map --seed 0 $EXTRA > logs/lc_$JOB.log 2>&1
    log "DONE $JOB (exit $?)" ;;
  lm_map_ab)
    for seed in 0 1 2; do for tag in intra nointra; do
      O=$OUT/lonemonk/map_s${seed}_$tag; [ -f $O/trace.json ] && continue
      EXTRA=""; [ $tag = nointra ] && EXTRA="--no-intra-lc"
      log "START lm_map seed $seed $tag"
      $PY scripts/viz/record_trace.py --scene lonemonk --map map_loop --variants --out $O --baseline 0.3 --snr 10 $FF --no-frames --seed $seed $EXTRA > logs/lc_lm_map_s${seed}_$tag.log 2>&1
      log "DONE lm_map seed $seed $tag ($(grep -o 'map ATE.*' $O/system.log | tail -1); intra PGOs $(grep -c 'Intra-hypothesis loop closure at' $O/system.log); merges $(grep -c 'LC detected' $O/system.log); rejected $(grep -c 'rejected' $O/system.log))"
    done; done ;;
  lm_map_intra2)
    # intra-hypothesis PGO with the post-merge cooldown and the significance gate, same seeds as lm_map_ab
    for seed in 0 1 2; do
      O=$OUT/lonemonk/map_s${seed}_intra2; [ -f $O/trace.json ] && continue
      log "START lm_map seed $seed intra2"
      $PY scripts/viz/record_trace.py --scene lonemonk --map map_loop --variants --out $O --baseline 0.3 --snr 10 $FF --no-frames --seed $seed > logs/lc_lm_map_s${seed}_intra2.log 2>&1
      log "DONE lm_map seed $seed intra2 ($(grep -o 'map ATE.*' $O/system.log | tail -1); intra PGOs $(grep -c 'Intra-hypothesis loop closure at' $O/system.log); merges $(grep -c 'LC detected' $O/system.log); rejected $(grep -c 'rejected' $O/system.log))"
    done ;;
  hs_map_intra2)
    O=$OUT/hssd_house/map_s0_intra2
    log "START hs_map seed 0 intra2"
    $PY scripts/viz/record_trace.py --scene hssd_house --map map --variants --out $O --baseline 0.3 --snr 10 $FF --no-frames --seed 0 > logs/lc_hs_map_s0_intra2.log 2>&1
    log "DONE hs_map seed 0 intra2 ($(grep -o 'map ATE.*' $O/system.log | tail -1); intra PGOs $(grep -c 'Intra-hypothesis loop closure at' $O/system.log); merges $(grep -c 'LC detected' $O/system.log))" ;;
  lm_map_split)
    O=$OUT/lonemonk/map_s0_split150; mkdir -p $O
    log "START lm_map_split"
    $PY scripts/viz/record_trace.py --scene lonemonk --map map_loop --variants --out $O --baseline 0.3 --snr 10 $FF --no-frames --seed 0 --recent-window 150 > logs/lc_lm_map_split.log 2>&1
    log "DONE lm_map_split ($(grep -o 'map ATE.*' $O/system.log | tail -1); intra PGOs $(grep -c 'Intra-hypothesis loop closure at' $O/system.log); merges $(grep -c 'LC detected' $O/system.log); rejected $(grep -c 'rejected' $O/system.log))" ;;
  hs_reloc_new|hs_reloc_new2|hs_reloc_legacy)
    tag=${JOB#hs_reloc_}; O=$OUT/hssd_house/reloc_$tag; mkdir -p $O
    cp -n outputs/fix/hssd_house/trace_cross_stereo/map.pkl outputs/fix/hssd_house/trace_cross_stereo/trace_map.json $O/
    EXTRA=""; [ $tag = legacy ] && EXTRA="--legacy-lc"
    log "START $JOB"
    $PY scripts/viz/record_trace.py --scene hssd_house --map map --variants rearr_100+offset_1.0+light_night+reverse offset_1.0 --out $O --baseline 0.3 --snr 10 $FF --no-frames --skip-map --seed 0 $EXTRA > logs/lc_$JOB.log 2>&1
    log "DONE $JOB (exit $?)" ;;
  hs_map_ab)
    for tag in intra nointra; do
      O=$OUT/hssd_house/map_s0_$tag; [ -f $O/trace.json ] && continue
      EXTRA=""; [ $tag = nointra ] && EXTRA="--no-intra-lc"
      log "START hs_map seed 0 $tag"
      $PY scripts/viz/record_trace.py --scene hssd_house --map map --variants --out $O --baseline 0.3 --snr 10 $FF --no-frames --seed 0 $EXTRA > logs/lc_hs_map_s0_$tag.log 2>&1
      log "DONE hs_map seed 0 $tag ($(grep -o 'map ATE.*' $O/system.log | tail -1); intra PGOs $(grep -c 'Intra-hypothesis loop closure at' $O/system.log); merges $(grep -c 'LC detected' $O/system.log))"
    done ;;
esac
log "job $JOB finished"
