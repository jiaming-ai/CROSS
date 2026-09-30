#!/usr/bin/env bash
# Render one replay MP4 per (scene, method) trace found under outputs/viz/<scene>/trace_<method>/.
# The page must have been built first (frames under outputs/viz/page/frames/).
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
W=${WORKERS:-24}
for scene in "$@"; do
  for tr in outputs/viz/$scene/trace_*/; do
    m=$(basename "$tr"); m=${m#trace_}
    [ -f "$tr/trace.json" ] || continue
    out=outputs/viz/videos/${scene}__${m}.mp4
    [ -f "$out" ] && [ "$out" -nt "$tr/trace.json" ] && continue
    mkdir -p outputs/viz/videos
    echo "[$(date +%H:%M:%S)] $scene $m"
    $PY scripts/viz/render_trace_video.py --trace "$tr" --frames-root outputs/viz/page --out "$out" --stride 2 --fps 20 --workers $W --hold 20 2>&1 | grep -v "findfont\|Ignoring fixed" | tail -1
  done
done
