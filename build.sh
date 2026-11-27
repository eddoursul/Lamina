#!/bin/sh
# Builds Lamina from this clone with a preset of CMakePresets.json: native (the default: this PC's GPUs and CPU, into
# build/) or portable (the releases' build for other PCs, into build-portable/). lamina and lamina-engine
# run from there.
#   ./build.sh [native|portable] [CMake options, e.g. -DLAMINA_BUILD_TESTS=ON]
# It needs CMake 3.22 or newer, Ninja, GCC 11 or newer and the CUDA toolkit 13.0 or newer (the nvcc on the PATH or in
# /usr/local/cuda; CUDACXX names another; CXX=g++-14 a compiler, for nvcc too).
cd "$(dirname "$0")" || exit 1
preset=native
case "$1" in native|portable) preset=$1; shift ;; esac
out=build
[ "$preset" = portable ] && out=build-portable
cmake --preset "$preset" "$@" || exit 1
# CUDA's ptxas now and then fails to read a file it has just written (upstream #45): the same command gets past it
cmake --build --preset "$preset" ||
  { echo "(the build stopped: trying it once more)"; cmake --build --preset "$preset"; } || exit 1
echo "Done: the programs are in $out/ ($out/lamina -m MODEL.gguf runs a model)"
