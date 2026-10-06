#!/bin/bash
# Copyright 2026 Google LLC.
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

# Verifies that a cpu_only LiteRT runtime binary contains no non-CPU (GPU,
# OpenCL, WebGPU, Dawn, OpenGL) symbols or dynamic library dependencies.
# Usage: minimal_compiled_model_symbol_test.sh <binary>

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <binary>"
  exit 1
fi

BINARY="$1"

if [ ! -f "${BINARY}" ]; then
  echo "ERROR: Binary not found: ${BINARY}"
  exit 1
fi

SUCCESS=1

echo "Checking dynamic library dependencies in $(basename "${BINARY}")..."
GPU_LIBS=$(readelf -d "${BINARY}" 2>/dev/null | grep -iE "\(NEEDED\).*\[lib(OpenCL|EGL|GLES)" || true)
if [ -n "${GPU_LIBS}" ]; then
  echo "ERROR: Found forbidden GPU shared library dependencies:"
  echo "${GPU_LIBS}"
  SUCCESS=0
fi

echo "Checking symbol table in $(basename "${BINARY}")..."
BANNED_SYMBOL_PATTERN="(clCreate|clGetPlatform|opencl_wrapper|qcom_wrapper|wgpu[A-Z]|wgpuBuffer|UploadWeightsOnWeb|CreateFromWebGpuBuffer|CreateFromOpenClMemory|CreateFromGlBuffer|CreateFromGlTexture)"

LEAKED_SYMBOLS=$(nm -C "${BINARY}" 2>/dev/null | grep -E "${BANNED_SYMBOL_PATTERN}" || true)
if [ -n "${LEAKED_SYMBOLS}" ]; then
  echo "ERROR: Found forbidden GPU symbols in $(basename "${BINARY}"):"
  echo "${LEAKED_SYMBOLS}"
  SUCCESS=0
fi

if [ "${SUCCESS}" -eq 1 ]; then
  echo "PASS: No GPU symbols found in $(basename "${BINARY}")"
else
  echo "FAIL: Non-CPU symbols leaked into $(basename "${BINARY}")"
  exit 1
fi
