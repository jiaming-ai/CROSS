#!/bin/bash
# publish_page.sh [--push]: build the results page (per-frame data from the Hub, as pinned by results/frames.json) and
# copy benchmark/site into the gh-pages branch (benchmark/), committed; --push also pushes gh-pages.
# The page links PROTOCOL.md / RESULTS.md on master: push master first.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
cd "$ROOT"
[ -z "$(git status --porcelain -- benchmark)" ] || { echo "benchmark/ has uncommitted changes: commit them first"; exit 1; }
"$PY" benchmark/build_site.py
WT=$(mktemp -d -p "$ROOT/.." gh-pages-XXXX)
trap 'git worktree remove --force "$WT"' EXIT
git fetch -q origin gh-pages
git worktree add -q -B gh-pages "$WT" origin/gh-pages
rsync -a --delete benchmark/site/ "$WT/benchmark/"
git -C "$WT" add -A benchmark
if git -C "$WT" diff --cached --quiet; then
  echo "gh-pages already up to date"
else
  git -C "$WT" commit -qm "Benchmark page: copy of benchmark/site from master $(git rev-parse --short HEAD)"
  echo "gh-pages: $(git -C "$WT" log --oneline -1)"
fi
if [ "${1:-}" = --push ]; then git -C "$WT" push origin gh-pages; fi
