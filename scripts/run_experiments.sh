#!/usr/bin/env bash
# Batch experiments: module-level relative pose evaluation + system-level map/relocalize.
# Usage: bash scripts/run_experiments.sh [relpose|reloc|all]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODE="${1:-all}"
VK=data/vkitti2/Scene01
TA=data/tartanair_v2/ArchVizTinyHouseNight/Data_easy
KITTI=data/kitti_raw/2011_09_26/2011_09_26_drive_0009_sync
mkdir -p outputs/relpose outputs/reloc logs

run() { echo "[$(date +%H:%M:%S)] $*"; "$@" > /dev/null 2>&1 || echo "FAILED: $*"; }

if [[ "$MODE" == "relpose" || "$MODE" == "all" ]]; then
  # --- vKITTI2: map condition = clone, query conditions = lighting/weather/viewpoint variants
  for cond in clone sunset fog rain morning overcast 15-deg-left 15-deg-right 30-deg-left 30-deg-right; do
    run $PY scripts/eval_relpose.py --ref $VK/clone --query $VK/$cond --estimator ff --backend vggt_omega --gaps 0,5,10,20 --n-queries 40 --out outputs/relpose/vk01_${cond}_ff_omega.json
    run $PY scripts/eval_relpose.py --ref $VK/clone --query $VK/$cond --estimator pnp --gaps 0,5,10,20 --n-queries 40 --out outputs/relpose/vk01_${cond}_pnp.json
  done
  # DA3 backend on a subset of conditions
  for cond in clone sunset fog 15-deg-left; do
    run $PY scripts/eval_relpose.py --ref $VK/clone --query $VK/$cond --estimator ff --backend da3 --gaps 0,5,10,20 --n-queries 40 --out outputs/relpose/vk01_${cond}_ff_da3.json
  done
  # anchor ablation (clone->sunset)
  run $PY scripts/eval_relpose.py --ref $VK/clone --query $VK/sunset --estimator ff --n-ref-anchors 0 --gaps 0,5,10,20 --n-queries 40 --out outputs/relpose/abl_vk01_sunset_anchors_curr_only.json
  run $PY scripts/eval_relpose.py --ref $VK/clone --query $VK/sunset --estimator ff --n-ref-anchors 4 --gaps 0,5,10,20 --n-queries 40 --out outputs/relpose/abl_vk01_sunset_anchors_4.json
  run $PY scripts/eval_relpose.py --ref $VK/clone --query $VK/sunset --estimator ff --n-ref-anchors 2 --no-curr-anchor --gaps 0,5,10,20 --n-queries 40 --out outputs/relpose/abl_vk01_sunset_anchors_ref_only.json
  for m in median mean huber_log norm_ls; do
    run $PY scripts/eval_relpose.py --ref $VK/clone --query $VK/sunset --estimator ff --scale-method $m --gaps 0,5,10,20 --n-queries 40 --out outputs/relpose/abl_vk01_sunset_scale_${m}.json
  done
  # --- TartanAir (indoor, low light): self pairs with frame gaps
  for p in P000 P005; do
    run $PY scripts/eval_relpose.py --ref $TA/$p --estimator ff --gaps 0,3,6,12,24 --n-queries 40 --out outputs/relpose/ta_${p}_ff_omega.json
    run $PY scripts/eval_relpose.py --ref $TA/$p --estimator pnp --gaps 0,3,6,12,24 --n-queries 40 --out outputs/relpose/ta_${p}_pnp.json
  done
  # --- KITTI raw (real images): frame gaps
  run $PY scripts/eval_relpose.py --ref $KITTI --estimator ff --gaps 0,2,5,10,20 --n-queries 40 --out outputs/relpose/kitti09_ff_omega.json
  run $PY scripts/eval_relpose.py --ref $KITTI --estimator ff --backend da3 --gaps 0,2,5,10,20 --n-queries 40 --out outputs/relpose/kitti09_ff_da3.json
  run $PY scripts/eval_relpose.py --ref $KITTI --estimator pnp --gaps 0,2,5,10,20 --n-queries 40 --out outputs/relpose/kitti09_pnp.json
fi

if [[ "$MODE" == "reloc" || "$MODE" == "all" ]]; then
  SNR=10
  FF="--estimator ff --obs-min-translation 0.5 --obs-min-rotation 0.15 --obs-max-interval 3"
  # --- vKITTI2: map clone once per estimator, relocalize under every other condition
  run $PY scripts/map_and_reloc.py --map $VK/clone --query $VK/clone --out outputs/reloc/vk01_ff/clone $FF --snr $SNR
  run $PY scripts/map_and_reloc.py --map $VK/clone --query $VK/clone --out outputs/reloc/vk01_pnp/clone --estimator pnp --snr $SNR
  for cond in sunset fog rain morning overcast 15-deg-left 15-deg-right 30-deg-left 30-deg-right; do
    for est in ff pnp; do
      mkdir -p outputs/reloc/vk01_${est}/$cond
      cp outputs/reloc/vk01_${est}/clone/map.pkl outputs/reloc/vk01_${est}/clone/map_meta.json outputs/reloc/vk01_${est}/$cond/
      if [[ $est == ff ]]; then A="$FF"; else A="--estimator pnp"; fi
      run $PY scripts/map_and_reloc.py --map $VK/clone --query $VK/$cond --out outputs/reloc/vk01_${est}/$cond $A --snr $SNR --skip-map
    done
  done
  # --- TartanAir: map P000, relocalize on overlapping trajectories
  run $PY scripts/map_and_reloc.py --map $TA/P000 --query $TA/P000 --out outputs/reloc/ta_ff/P000 $FF --snr $SNR
  run $PY scripts/map_and_reloc.py --map $TA/P000 --query $TA/P000 --out outputs/reloc/ta_pnp/P000 --estimator pnp --snr $SNR
  for p in P005 P002 P004; do
    for est in ff pnp; do
      mkdir -p outputs/reloc/ta_${est}/$p
      cp outputs/reloc/ta_${est}/P000/map.pkl outputs/reloc/ta_${est}/P000/map_meta.json outputs/reloc/ta_${est}/$p/
      if [[ $est == ff ]]; then A="$FF"; else A="--estimator pnp"; fi
      run $PY scripts/map_and_reloc.py --map $TA/P000 --query $TA/$p --out outputs/reloc/ta_${est}/$p $A --snr $SNR --skip-map
    done
  done
  # --- observation cadence ablation (vKITTI clone -> sunset, FF)
  for iv in 1 5 10; do
    mkdir -p outputs/reloc/abl_cadence_$iv
    cp outputs/reloc/vk01_ff/clone/map.pkl outputs/reloc/vk01_ff/clone/map_meta.json outputs/reloc/abl_cadence_$iv/
    run $PY scripts/map_and_reloc.py --map $VK/clone --query $VK/sunset --out outputs/reloc/abl_cadence_$iv --estimator ff --obs-min-translation 0 --obs-max-interval $iv --snr $SNR --skip-map
  done
fi
echo "[$(date +%H:%M:%S)] done"
