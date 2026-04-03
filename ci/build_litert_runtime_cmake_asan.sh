#!/usr/bin/env bash
# Copyright 2026 The AI Edge LiteRT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LITERT_DIR="${REPO_ROOT}/litert"

PRESET="${PRESET:-default-debug}"
BUILD_DIR="${BUILD_DIR:-${LITERT_DIR}/cmake_build_debug}"
TARGET="${TARGET:-litert_runtime_c_api_shared_lib}"
if [[ -n "${C_COMPILER:-}" ]]; then
  C_COMPILER="${C_COMPILER}"
elif command -v clang >/dev/null 2>&1; then
  C_COMPILER="clang"
else
  C_COMPILER="gcc"
fi

if [[ -n "${CXX_COMPILER:-}" ]]; then
  CXX_COMPILER="${CXX_COMPILER}"
elif command -v clang++ >/dev/null 2>&1; then
  CXX_COMPILER="clang++"
else
  CXX_COMPILER="g++"
fi
OPT_LEVEL="${OPT_LEVEL:-0}"
SANITIZER="${SANITIZER:-address}"
JOBS="${JOBS:-$(nproc)}"
CLEAN="${CLEAN:-0}"
LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-${REPO_ROOT}/.tmp}"
GENERATOR="${GENERATOR:-}"

if [[ "${CLEAN}" == "1" ]]; then
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"

cd "${LITERT_DIR}"

cmake_args=(
  --preset "${PRESET}"
  -DCMAKE_C_COMPILER="${C_COMPILER}"
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}"
  -DCMAKE_C_FLAGS_DEBUG="-O${OPT_LEVEL} -g -fno-omit-frame-pointer -fsanitize=${SANITIZER}"
  -DCMAKE_CXX_FLAGS_DEBUG="-O${OPT_LEVEL} -g -fno-omit-frame-pointer -fsanitize=${SANITIZER}"
  -DCMAKE_SHARED_LINKER_FLAGS_DEBUG="-fsanitize=${SANITIZER}"
  -DCMAKE_EXE_LINKER_FLAGS_DEBUG="-fsanitize=${SANITIZER}"
  -DLITERT_DEPENDENCY_BUILD_PARALLEL_LEVEL="${JOBS}"
)
if [[ -n "${GENERATOR}" ]]; then
  cmake_args+=(-G "${GENERATOR}")
fi
cmake "${cmake_args[@]}"

cmake --build "${BUILD_DIR}" --target "${TARGET}" --parallel "${JOBS}"

echo "Build completed."
echo "Searching for runtime shared library:"
find "${BUILD_DIR}" -type f \( -name "libLiteRtRuntimeCApi.so" -o -name "libLiteRtRuntimeCApi.dylib" \) -print
echo "Recommended runtime env:"
echo "  export ASAN_OPTIONS='symbolize=1:halt_on_error=1:abort_on_error=1'"
