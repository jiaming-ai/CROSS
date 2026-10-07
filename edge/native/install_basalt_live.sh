#!/usr/bin/env bash
# Build basalt_live (edge/native/basalt_live.cpp): Basalt's VIO fed from a stream, for cross_edge.basalt.
#   edge/native/install_basalt_live.sh <dir>   -> <dir>/basalt/build/release/basalt_live  (export BASALT_LIVE=...)
# The Basalt commit and fixes of scripts/vio/install_basalt.sh (the build that produced the benchmark's odom_vio.txt),
# plus the basalt_live target; only that target is built.  Needs cmake >= 3.24 (< 4), ninja, a C++17 compiler; vcpkg
# builds the other dependencies (its Pangolin port needs the OpenGL / X11 headers).  Tested on x86-64 Ubuntu 22.04;
# aarch64 (the edge computer) is untested: vcpkg's triplet becomes arm64-linux.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DIR=$(realpath "${1:?usage: install_basalt_live.sh <dir>}")
mkdir -p "$DIR" && cd "$DIR"
[ -d basalt ] || git clone --recursive https://gitlab.com/VladyslavUsenko/basalt.git
cd basalt
git checkout -q "${BASALT_COMMIT:-0f3b2b52c807f70ff4e2973ce253c73329eea7bc}"          # 2026-03-22
git submodule update --init --recursive
python3 - <<'PY'
import json
d = json.load(open("vcpkg.json"))
d["dependencies"] = [x for x in d["dependencies"] if not (isinstance(x, dict) and x.get("name") == "realsense2")]
d.pop("overrides", None)
json.dump(d, open("vcpkg.json", "w"), indent=2)
PY
sed -i "s/ -Werror//g" CMakeLists.txt
cp "$HERE/basalt_live.cpp" src/basalt_live.cpp
grep -q "basalt_live" CMakeLists.txt || cat >> CMakeLists.txt <<'CMAKE'

# cross-edge: the VIO fed from a stream (edge/native/basalt_live.cpp of the CROSS repository)
add_executable(basalt_live src/basalt_live.cpp)
target_link_libraries(basalt_live basalt)
CMAKE
mkdir -p "$DIR/linkfix"
for l in wayland-egl wayland-cursor wayland-client xkbcommon; do
  f=$(ls /usr/lib/*-linux-gnu/lib$l.so.* 2>/dev/null | head -1 || true)
  [ -n "$f" ] && [ ! -e "$(dirname "$f")/lib$l.so" ] && ln -sf "$f" "$DIR/linkfix/lib$l.so"
done
export LIBRARY_PATH="$DIR/linkfix${LIBRARY_PATH:+:$LIBRARY_PATH}"
[ -x thirdparty/vcpkg/vcpkg ] || thirdparty/vcpkg/bootstrap-vcpkg.sh -disableMetrics
cmake --preset release -DBUILD_TESTS=OFF
cmake --build --preset release --target basalt_live -j "${JOBS:-$(nproc)}"
echo "BASALT_LIVE=$DIR/basalt/build/release/basalt_live"
