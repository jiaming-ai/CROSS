#!/usr/bin/env bash
# OpenLORIS-Scene (Shi et al., ICRA 2020) package format from Hugging Face (shixuesong/openloris-scene).
# Usage: download_openloris.sh <dest> [package ...]   (default: all five scenes)
set -euo pipefail
DEST=${1:?dest dir}; shift || true
PKGS=${*:-"office1-1_7 corridor1-1 corridor1-2_5 home1-1_5 cafe1-1_2 market1-1_3"}
PY7Z=${PY7Z:-python3}   # a python with py7zr (the packages hold one .7z per sequence)
URL=https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/main
mkdir -p "$DEST"; cd "$DEST"
[ -f groundtruth.zip ] || curl -sfL -o groundtruth.zip "$URL/package/groundtruth.zip" || true
for p in $PKGS; do
  case $p in *_*) f=$p-package.tar ;; *) f=$p.7z ;; esac
  [ -f $f.done ] && continue
  echo "[$(date +%T)] $f"
  curl -sfL -C - -o $f "$URL/package/$f"
  case $f in *.tar) tar xf $f && rm -f $f ;; esac
  touch $f.done && echo "[$(date +%T)] done $f"
done
for z in *.7z; do            # one archive per sequence
  [ -e "$z" ] || continue; d=$(basename "$z" .7z)
  [ -d "$d" ] || "$PY7Z" -c "import py7zr,sys; py7zr.SevenZipFile(sys.argv[1]).extractall('.')" "$z"
  [ -d "$d" ] && rm -f "$z" && echo "extracted $d"
done
echo OPENLORIS_DL_DONE
