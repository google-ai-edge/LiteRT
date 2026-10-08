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

# Verifies that a minimal (cpu_only + selective) LiteRT runtime binary contains
# no non-CPU (GPU, OpenCL, WebGPU, Dawn, OpenGL, NPU dispatch), unselected
# builtin-op, external weight loader, absl::Status/Cord, or C++ stream symbols
# (iostream/stringstream/ifstream) or dynamic library dependencies.
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
# GPU symbols: Ensures GPU backends (OpenCL, WebGPU/Dawn, OpenGL/GLES) and their
# buffer/texture wrappers are not pulled into CPU-only builds.
BANNED_GPU_SYMBOLS="clCreate|clGetPlatform|opencl_wrapper|qcom_wrapper|wgpu[A-Z]|wgpuBuffer|UploadWeightsOnWeb|CreateFromWebGpuBuffer|CreateFromOpenClMemory|CreateFromGlBuffer|CreateFromGlTexture"
# NPU symbols: Ensures NPU dispatch accelerator, registration, and dispatch
# delegate kernels are excluded in minimal CPU runtime builds.
BANNED_NPU_SYMBOLS="DispatchAccelerator|LiteRtRegisterNpuAccelerator|LiteRtCreateDispatchDelegate|DispatchDelegateKernel|LiteRtDispatchOpOptions"

# Builtin-op symbols: Ensures unselected TFLite builtin ops and the default CPU
# accelerator table are not linked when selective op registration is enabled.
BANNED_BUILTIN_OP_SYMBOLS="LiteRtRegisterCpuAccelerator|BuiltinOpResolverWithoutDefaultDelegates|Register_CONV_2D|Register_LSTM|Register_SVDF|ReplaceMagicNumbersIfAny|ClassicLocale|BuiltinOptionsUnion::UnPack"

# Weight loader symbols: Ensures external weight loader flatbuffers, schema, and
# internal CPU weight restoration logic are omitted in selective LiteRT builds
# (gated by LITERT_DISABLE_EXTERNAL_WEIGHTS).
BANNED_WEIGHT_LOADER_SYMBOLS="weight_loader::|CreateLiteRtWeightLoader|RestoreExternalWeightsForCpu"

# Stream symbols: C++ std::iostream / std::stringstream / std::ifstream pull in
# heavy virtual tables, locales, and formatting logic. Ensures LiteRT and TFLite
# dependencies use lighter alternatives.
BANNED_STREAM_SYMBOLS="litert::.*basic_.*stream|default_delete<.*basic_.*stream|unique_ptr<.*basic_.*stream|basic_ifstream|basic_stringstream|basic_filebuf<.*>::|basic_istream<.*>::(read|seekg|tellg)"

# Status symbols: Heavy absl::Status implementations pull in Cord representation
# and heap-allocated error payloads. Minimal builds rely on lighter status paths.
BANNED_STATUS_SYMBOLS="StatusRep|CordRep|ErrorStatusBuilder::ToAbslStatus|ErrorConversion<absl::Status|FormatPack|StrFormat"
BANNED_SYMBOL_PATTERN="(${BANNED_GPU_SYMBOLS}|${BANNED_NPU_SYMBOLS}|${BANNED_BUILTIN_OP_SYMBOLS}|${BANNED_WEIGHT_LOADER_SYMBOLS}|${BANNED_STREAM_SYMBOLS}|${BANNED_STATUS_SYMBOLS})"

SYMBOLS=$(nm -C "${BINARY}")
if [ -z "${SYMBOLS}" ]; then
  echo "ERROR: Failed to read symbol table from $(basename "${BINARY}")"
  exit 1
fi

LEAKED_SYMBOLS=$(grep -E "${BANNED_SYMBOL_PATTERN}" <<< "${SYMBOLS}" || true)
if [ -n "${LEAKED_SYMBOLS}" ]; then
  echo "ERROR: Found forbidden symbols in $(basename "${BINARY}"):"
  echo "${LEAKED_SYMBOLS}"
  SUCCESS=0
fi

if [ "${SUCCESS}" -eq 1 ]; then
  echo "PASS: No forbidden symbols found in $(basename "${BINARY}")"
else
  echo "FAIL: Forbidden symbols leaked into $(basename "${BINARY}")"
  exit 1
fi
