#!/usr/bin/env bash
# Relocalization-success trial protocol (CROSS paper: 100-frame trials, stride 50, r_D = 2 m) for one scene and one
# loop-closure mode; the map is built by the same mode (variant "map" first).
#   bash scripts/lc/run_rs_trials.sh <gpu> <scene> <mode: heuristic|final> [variants...]
# outputs: outputs/lcstudy_rs/<scene>/<mode>/<variant>/reloc_summary.json
set -uo pipefail
cd "$(dirname "$0")/../.."
GPU=$1; SCENE=$2; MODE=$3; shift 3
export CUDA_VISIBLE_DEVICES=$GPU MPLBACKEND=Agg
PY=.venv/bin/python
FF="--obs-min-translation 0.3 --obs-min-rotation 0.15 --obs-max-interval 3"
D=data/sim/$SCENE; R=outputs/lcstudy_rs/$SCENE/$MODE; mkdir -p $R logs
if [ $SCENE = lonemonk ]; then M=$D/map_loop; NOISE=configs/noise/lonemonk_full.yaml; else M=$D/map; NOISE=configs/noise/${SCENE}_600.yaml; fi
if [ $MODE = heuristic ]; then LC="--lc-mode heuristic"; else LC="--lc-mode verified --noise-config $NOISE"; fi
VARS="$*"; [ -z "$VARS" ] && VARS="map"
log() { echo "[$(date +%F_%T)] $*" | tee -a logs/lcstudy_rs.log; }
# the map first (built once per mode), then the variants against it
if [ ! -f $R/map/reloc_summary.json ]; then
  mkdir -p $R/map; mkdir $R/map/.lock 2>/dev/null && {
    log "START $SCENE/$MODE map (gpu $GPU)"
    $PY scripts/map_and_reloc.py --map $M --query $M --out $R/map --estimator ff --baseline 0.3 --snr 10 $FF $LC \
        --trial-len 100 --trial-stride 50 --r-d 2.0 --seed 0 > logs/rs_${SCENE}_${MODE}_map.log 2>&1
    rmdir $R/map/.lock
    log "DONE $SCENE/$MODE map ($(python3 -c "import json;d=json.load(open('$R/map/reloc_summary.json'));print('RS',round(d['RS'],3))" 2>/dev/null); map ATE $(python3 -c "import json;print(round(json.load(open('$R/map/map_meta.json'))['map_ate_rmse'],3))" 2>/dev/null))"
  }
fi
until [ -f $R/map/map.pkl ] && [ -f $R/map/map_meta.json ] && [ -f $R/map/reloc_summary.json ]; do sleep 30; done
for v in $VARS; do
  [ $v = map ] && continue
  [ -f $R/$v/reloc_summary.json ] && continue
  mkdir -p $R/$v; mkdir $R/$v/.lock 2>/dev/null || continue
  ln -sf ../map/map.pkl $R/$v/map.pkl; cp $R/map/map_meta.json $R/$v/
  log "START $SCENE/$MODE $v (gpu $GPU)"
  $PY scripts/map_and_reloc.py --map $M --query $D/$v --out $R/$v --estimator ff --baseline 0.3 --snr 10 $FF $LC --skip-map \
      --trial-len 100 --trial-stride 50 --r-d 2.0 --seed 0 > logs/rs_${SCENE}_${MODE}_$v.log 2>&1
  rmdir $R/$v/.lock 2>/dev/null
  log "DONE $SCENE/$MODE $v ($(python3 -c "import json;d=json.load(open('$R/$v/reloc_summary.json'));print('RS',round(d['RS'],3),'RS_1m_5deg',round(d['RS_1m_5deg'],3))" 2>/dev/null))"
done
log "rs $SCENE/$MODE finished on gpu $GPU"
