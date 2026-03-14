#!/bin/bash
# Build the C++ GWC TensorRT plugin (libGwcVolumePlugin.so).
# The plugin is required for building a single TensorRT engine from foundation_stereo.onnx.
#
# Usage:
#   ./scripts/build_gwc_plugin.sh
#   # Output: plugins/gwc_plugin_cpp/build/libGwcVolumePlugin.so
#
# Then build engine with:
#   python scripts/build_engine.py --onnx output/480x864/foundation_stereo.onnx \
#       --engine output/480x864/foundation_stereo.engine \
#       --plugin plugins/gwc_plugin_cpp/build/libGwcVolumePlugin.so

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/../plugins/gwc_plugin_cpp" && pwd)"
BUILD_DIR="$PLUGIN_DIR/build"

# Ensure CUDA compiler is found
if [ -z "${CUDACXX}" ] && [ -x "/usr/local/cuda/bin/nvcc" ]; then
    export CUDACXX=/usr/local/cuda/bin/nvcc
fi

cd "$PLUGIN_DIR"
mkdir -p build
cd build
CMAKE_OPTS=()
if [ -x "/usr/local/cuda/bin/nvcc" ]; then
    CMAKE_OPTS+=(-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc)
fi
cmake "${CMAKE_OPTS[@]}" ..
make -j$(nproc)

echo ""
echo "Built: $BUILD_DIR/libGwcVolumePlugin.so"
echo ""
echo "Build engine with:"
echo "  python scripts/build_engine.py --onnx <onnx> --engine <engine> --plugin $BUILD_DIR/libGwcVolumePlugin.so"
