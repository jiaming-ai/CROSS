#!/usr/bin/env bash
# ROVER (Schmidt et al., T-RO 2025; MIT license): outdoor multi-season recordings of a ground robot with
# D435i (RGB-D + IMU), T265 (stereo fisheye + IMU), Pi camera, VN100 IMU; ground truth from a total station.
# Hugging Face: iis-esslingen/ROVER (chunked zips).  Usage: download_rover.sh <dest> [location] [scenario ...]
# Default: campus_large, all eight scenarios (~320 GB), as used by the CROSS paper's outdoor benchmark.
# NO_UNZIP=1 keeps the merged zips (extracting ~150k files per recording onto NFS takes hours).
set -euo pipefail
DEST=${1:?dest dir}; LOC=${2:-campus_large}; shift $(( $# >= 2 ? 2 : $# )) || true
SCEN=${*:-"summer autumn winter spring day dusk night night-light"}
URL=https://huggingface.co/datasets/iis-esslingen/ROVER/resolve/main
mkdir -p "$DEST"; cd "$DEST"
[ -f calibration.zip ] || { curl -sfL -o calibration.zip "$URL/calibration.zip" && unzip -q -o calibration.zip; }
LIST=$(curl -sf "https://huggingface.co/api/datasets/iis-esslingen/ROVER/tree/main" | python3 -c "import json,sys; print('\n'.join(x['path'] for x in json.load(sys.stdin)))")
for s in $SCEN; do
  for name in $(echo "$LIST" | grep -E "^${LOC}_${s}_[0-9-]+(_[0-9])?\.zip\.part-" | sed 's/\.part-.*//' | sort -u); do
    [ -f "$name.done" ] && continue
    echo "[$(date +%T)] $name"
    # parts in parallel (Hugging Face throttles single connections to ~5 MB/s); resumable
    echo "$LIST" | grep -F "$name.part-" | xargs -P "${JOBS:-6}" -I{} sh -c \
      'for i in 1 2 3; do curl -sfL -C - -o "{}" "'"$URL"'/{}" && exit 0; sleep 5; done; exit 1'
    n_exp=$(echo "$LIST" | grep -cF "$name.part-"); n_got=$(ls "$name".part-* 2>/dev/null | wc -l)
    [ "$n_got" -eq "$n_exp" ] || { echo "missing parts of $name"; continue; }
    cat $(echo "$LIST" | grep -F "$name.part-" | sort) > "$name" && rm -f "$name".part-*
    if [ "${NO_UNZIP:-0}" = 1 ]; then       # keep the zip: prepare_rover.py reads the frames it needs from it
      touch "$name.done" && echo "[$(date +%T)] done $name (zip kept)"
    else
      unzip -q -o "$name" && rm -f "$name" && touch "$name.done" && echo "[$(date +%T)] done $name"
    fi
  done
done
echo ROVER_DL_DONE
