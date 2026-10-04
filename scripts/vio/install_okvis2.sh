#!/usr/bin/env bash
# Build OKVIS2-X (https://github.com/ethz-mrl/OKVIS2-X), vision-inertial only, for prepare_vio.py --vio okvis2.
#   scripts/vio/install_okvis2.sh <dir> <conda env prefix>   -> <dir>/OKVIS2-X/build/okvis_app_synchronous
# The env provides the dependencies without sudo:
#   micromamba create -p <env> -c conda-forge cmake ninja gxx_linux-64=12 gcc_linux-64=12 eigen=3.4 glog gflags \
#       boost-cpp opencv suitesparse liblapack libblas geographiclib-cpp
# Fixes applied here: submodules over https; PCL (used only to write PLY point clouds of the dense maps, which VIO
# does not produce) replaced by a header-only stand-in; GLOG_USE_GLOG_EXPORT for glog >= 0.7; no CUDA in Ceres.
set -euo pipefail
DIR=$(realpath "${1:?usage: install_okvis2.sh <dir> <env>}"); ENV=$(realpath "${2:?conda env prefix}")
HERE=$(dirname "$(realpath "$0")")
mkdir -p "$DIR" && cd "$DIR"
if [ ! -d OKVIS2-X ]; then
  git clone https://github.com/ethz-mrl/OKVIS2-X.git
  (cd OKVIS2-X && git -c url."https://github.com/".insteadOf=git@github.com: submodule update --init --recursive)
fi
mkdir -p pcl_stub && cp -r "$HERE/pcl_stub/." pcl_stub/
export PATH=$ENV/bin:$PATH CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++
export CMAKE_PREFIX_PATH="$ENV:$DIR/pcl_stub" CMAKE_POLICY_VERSION_MINIMUM=3.5
mkdir -p OKVIS2-X/build && cd OKVIS2-X/build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_NN=OFF -DUSE_CUDA=OFF -DCMAKE_CXX_FLAGS=-DGLOG_USE_GLOG_EXPORT \
  -DHAVE_LIBREALSENSE=OFF -DBUILD_ROS2=OFF "-DCMAKE_PREFIX_PATH=$ENV;$DIR/pcl_stub" -DPCL_DIR="$DIR/pcl_stub/share/pcl" \
  -DCMAKE_INSTALL_RPATH="$ENV/lib" -DCMAKE_BUILD_RPATH="$ENV/lib" ..
ninja -j "$(nproc)"
echo "OKVIS2_APP=$DIR/OKVIS2-X/build/okvis_app_synchronous"
