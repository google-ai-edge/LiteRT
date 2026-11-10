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

# Verifies that the LiteRT JNI shared library (liblitert_jni.so) contains no
# forbidden GPU/NPU/builtin-op runtime symbols, C++ stream symbols
# (iostream/stringstream/ifstream), absl::LogMessage, absl::StrFormat, or
# absl::Status/Cord symbols.
# Usage: litert_jni_symbol_test.sh <so_file>

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <so_file>"
  exit 1
fi

BINARY="$1"

if [ ! -f "${BINARY}" ]; then
  echo "ERROR: Shared library not found: ${BINARY}"
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
SYMBOLS=$(nm -C "${BINARY}")
if [ -z "${SYMBOLS}" ]; then
  echo "ERROR: Failed to read symbol table from $(basename "${BINARY}")"
  exit 1
fi

if ! grep -q "Java_com_google_ai_edge_litert_" <<< "${SYMBOLS}"; then
  echo "ERROR: Expected JNI entrypoints (Java_com_google_ai_edge_litert_*) not found in $(basename "${BINARY}")"
  exit 1
fi

# Stream symbols: C++ std::iostream / std::stringstream / std::ifstream pull in
# heavy virtual tables, locales, formatting logic, and global constructors.
BANNED_STREAM_SYMBOLS="basic_.*stream|basic_streambuf|basic_filebuf|ios_base::Init"

# Status, formatting, and logging symbols: Heavy absl::Status/Cord,
# absl::StrFormat, and absl::log_internal::LogMessage implementations pull in
# large formatting tables, synchronization, and ostream dependencies.
BANNED_STATUS_AND_LOG_SYMBOLS="StatusRep|CordRep|ErrorStatusBuilder::ToAbslStatus|ErrorConversion<absl::Status|FormatPack|StrFormat|absl::.*LogMessage"

BANNED_SYMBOL_PATTERN="(${BANNED_STREAM_SYMBOLS}|${BANNED_STATUS_AND_LOG_SYMBOLS})"

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
