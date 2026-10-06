#!/usr/bin/env bash
# Build Basalt VIO (https://gitlab.com/VladyslavUsenko/basalt) for benchmark/datasets/prepare_vio.py --vio basalt.
#   scripts/vio/install_basalt.sh <dir>        -> <dir>/basalt/build/release/basalt_vio  (export BASALT_VIO=...)
# Needs cmake >= 3.24, ninja, a C++17 compiler and the OpenGL / X11 headers (Pangolin); vcpkg builds every other
# dependency.  No sudo: missing tools can come from conda-forge (micromamba create -p <env> cmake ninja).
# Pinned to the Basalt commit that produced the benchmark's odom_vio.txt (BASALT_COMMIT=<sha> overrides it).
# Fixes applied here: RealSense support dropped (not needed, its vcpkg port needs libusb / udev); -Werror removed
# (unused-variable errors with newer GCC); Wayland / xkbcommon linked through their runtime .so files when the -dev
# symlinks are missing.
set -euo pipefail
DIR=$(realpath "${1:?usage: install_basalt.sh <dir>}")
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
mkdir -p "$DIR/linkfix"
for l in wayland-egl wayland-cursor wayland-client xkbcommon; do
  f=$(ls /usr/lib/x86_64-linux-gnu/lib$l.so.* 2>/dev/null | head -1 || true)
  [ -n "$f" ] && [ ! -e /usr/lib/x86_64-linux-gnu/lib$l.so ] && ln -sf "$f" "$DIR/linkfix/lib$l.so"
done
export LIBRARY_PATH="$DIR/linkfix${LIBRARY_PATH:+:$LIBRARY_PATH}"
[ -x thirdparty/vcpkg/vcpkg ] || thirdparty/vcpkg/bootstrap-vcpkg.sh -disableMetrics
cmake --preset release -DBUILD_TESTS=OFF
cmake --build --preset release -j "$(nproc)"
echo "BASALT_VIO=$DIR/basalt/build/release/basalt_vio"
