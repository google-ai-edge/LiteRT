# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e

echo "Building LiteRT with CMake..."

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/litert_cmake_build"
TFLITE_BUILD_DIR="${SCRIPT_DIR}/../tflite_build"

# Check if TFLite has been built
if [ ! -f "${TFLITE_BUILD_DIR}/libtensorflow-lite.a" ]; then
    echo "TFLite not found at ${TFLITE_BUILD_DIR}. Building TFLite first..."
    cd "${SCRIPT_DIR}/.."
    ./build_tflite.sh
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
echo "Configuring LiteRT..."
cmake .. \
    -DTFLITE_BUILD_DIR="${TFLITE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building LiteRT..."
# The following command optimizes the build process by parallelizing it.
# It uses the number of available processors to parallelize the build process.
# - `nproc 2>/dev/null`: tries to get the number of processors using `nproc`.
# - `sysctl -n hw.ncpu 2>/dev/null`: if `nproc` fails (e.g., on macOS), this tries to get the number of processors using `sysctl`.
# - `echo 4`: if both `nproc` and `sysctl` fail, it defaults to 4 processors.
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Build complete!"

# dummy change to trigger presubmit
