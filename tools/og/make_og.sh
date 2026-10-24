#!/usr/bin/env bash
# Render the social preview cards (assets/img/og/<page>.jpg, 1200x630) from tools/og/card.html with headless Chrome.
# Needs google-chrome and ImageMagick; serves the repo on a local port while it runs.
#   bash tools/og/make_og.sh [port=8931]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PORT=${1:-8931}
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$ROOT" >/dev/null 2>&1 &
SERVER=$!
PROFILE=$(mktemp -d)
trap 'kill $SERVER; rm -rf "$PROFILE"' EXIT
sleep 1
for page in home leaderboard replays docs; do
  google-chrome --headless=new --disable-gpu --hide-scrollbars --no-first-run --user-data-dir="$PROFILE" \
    --window-size=1200,630 --virtual-time-budget=8000 --screenshot="$PROFILE/$page.png" \
    "http://127.0.0.1:$PORT/tools/og/card.html?page=$page" >/dev/null 2>&1
  convert "$PROFILE/$page.png" -quality 86 "$ROOT/assets/img/og/$page.jpg"
  echo "assets/img/og/$page.jpg"
done
