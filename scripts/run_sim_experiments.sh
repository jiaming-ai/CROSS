#!/usr/bin/env bash
# SimChange benchmark: CROSS-stereo (ff), CROSS-PnP (GT depth), ORB-SLAM3 stereo, RTAB-Map RGB-D, MASt3R-SLAM.
# Usage: bash scripts/run_sim_experiments.sh <scene> [systems...]
set -uo pipefail
cd "$(dirname "$0")/.."
SCENE="${1:-classroom}"; shift || true
SYSTEMS="${*:-ff pnp orbslam3 rtabmap mast3r}"
PY=.venv/bin/python
D=data/sim/$SCENE
M=$D/${MAP_NAME:-map}          # map sequence (MAP_NAME=map_loop: closed-loop map with a replayed start)
OUT=${OUT_ROOT:-outputs/sim}/$SCENE      # OUT_ROOT: alternative output tree (e.g. a local disk for the CPU baselines)
SNR=10
TRIAL="--trial-len ${TRIAL_LEN:-40} --trial-stride ${TRIAL_STRIDE:-20} --r-d ${R_D:-2.0}"
FF="--estimator ff --obs-min-translation 0.3 --obs-min-rotation 0.15 --obs-max-interval 3 ${FF_EXTRA:-}"
VARIANTS=${VARIANTS_OVERRIDE:-$(ls $D | grep -v "^map$" | grep -v "^map_loop$" | grep -v assets)}
FF="--estimator ff --obs-min-translation 0.3 --obs-min-rotation 0.15 --obs-max-interval 3 ${FF_EXTRA:-}"
mkdir -p $OUT logs
run() { echo "[$(date +%H:%M:%S)] $*"; "$@" > /dev/null 2> logs/last_stderr.txt || { echo "FAILED: $*"; echo "=== $*" >> logs/sim_failures.log; tail -30 logs/last_stderr.txt >> logs/sim_failures.log; }; }
for spec in $SYSTEMS; do
  # a system token may carry a stereo baseline, e.g. ff:0.3 or orbslam3:0.5 (SimChange multi-baseline renders)
  sys=${spec%%:*}; B=""; TAG=$sys
  if [[ "$spec" == *:* ]]; then B="--baseline ${spec#*:}"; TAG=${sys}_b$(printf "%.2f" ${spec#*:}); fi
  TAG="${TAG}${TAG_SUFFIX:-}"
  case $sys in
    ff)
      [ -f $OUT/$TAG/map/map.pkl ] && [ -f $OUT/$TAG/map/reloc_summary.json ] || run $PY scripts/map_and_reloc.py --map $M --query $D/map --out $OUT/$TAG/map $FF $B --snr $SNR $TRIAL
      for v in $VARIANTS; do
        [ -f $OUT/$TAG/$v/reloc_summary.json ] && continue
        mkdir -p $OUT/$TAG/$v; ln -sf ../map/map.pkl $OUT/$TAG/$v/map.pkl; cp $OUT/$TAG/map/map_meta.json $OUT/$TAG/$v/
        run $PY scripts/map_and_reloc.py --map $M --query $D/$v --out $OUT/$TAG/$v $FF $B --snr $SNR --skip-map $TRIAL
      done ;;
    pnp)
      [ -f $OUT/pnp/map/map.pkl ] && [ -f $OUT/pnp/map/reloc_summary.json ] || run $PY scripts/map_and_reloc.py --map $M --query $D/map --out $OUT/pnp/map --estimator pnp --pnp-depth gt --snr $SNR $TRIAL
      for v in $VARIANTS; do
        [ -f $OUT/pnp/$v/reloc_summary.json ] && continue
        mkdir -p $OUT/pnp/$v; ln -sf ../map/map.pkl $OUT/pnp/$v/map.pkl; cp $OUT/pnp/map/map_meta.json $OUT/pnp/$v/
        run $PY scripts/map_and_reloc.py --map $M --query $D/$v --out $OUT/pnp/$v --estimator pnp --pnp-depth gt --snr $SNR --skip-map $TRIAL
      done ;;
    pnpsgbm)
      # original CROSS on a stereo robot: PnP on SGBM depth from the stereo pair (baseline token selects the right camera)
      [ -f $OUT/$TAG/map/map.pkl ] && [ -f $OUT/$TAG/map/reloc_summary.json ] || run $PY scripts/map_and_reloc.py --map $M --query $D/map --out $OUT/$TAG/map --estimator pnp --pnp-depth sgbm $B --snr $SNR $TRIAL
      for v in $VARIANTS; do
        [ -f $OUT/$TAG/$v/reloc_summary.json ] && continue
        mkdir -p $OUT/$TAG/$v; ln -sf ../map/map.pkl $OUT/$TAG/$v/map.pkl; cp $OUT/$TAG/map/map_meta.json $OUT/$TAG/$v/
        run $PY scripts/map_and_reloc.py --map $M --query $D/$v --out $OUT/$TAG/$v --estimator pnp --pnp-depth sgbm $B --snr $SNR --skip-map $TRIAL
      done ;;
    orbslam3|rtabmap|rtabmapstereo)
      [ $sys = rtabmapstereo ] && sys=rtabmap_stereo
      [ -f $OUT/$TAG/map/reloc_summary.json ] && { [ -f $OUT/$TAG/map/atlas.osa ] || [ -f $OUT/$TAG/map/map.db ]; } || run $PY scripts/baselines/run_baselines.py --map $M --query $D/map --system $sys --out $OUT/$TAG/map --snr $SNR $B $TRIAL
      for v in $VARIANTS; do
        [ -f $OUT/$TAG/$v/reloc_summary.json ] && continue
        mkdir -p $OUT/$TAG/$v
        for f in atlas.osa map.db map_poses.txt map_time.json; do [ -f $OUT/$TAG/map/$f ] && cp $OUT/$TAG/map/$f $OUT/$TAG/$v/; done
        run $PY scripts/baselines/run_baselines.py --map $M --query $D/$v --system $sys --out $OUT/$TAG/$v --snr $SNR $B $TRIAL
      done ;;
    mast3r)
      for v in map $VARIANTS; do
        [ -f $OUT/mast3r/$v/reloc_summary.json ] && continue
        run $PY scripts/baselines/run_mast3r_slam.py --map $M --query $D/$v --out $OUT/mast3r/$v --config ${M3_CONFIG:-config/reloc_permissive.yaml} $TRIAL
      done ;;
  esac
done
echo "[$(date +%H:%M:%S)] sim experiments done for $SCENE ($SYSTEMS)"
