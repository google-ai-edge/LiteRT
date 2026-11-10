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

# Verifies that the full LiteRT C API and runtime binary (including all LiteRt*
# C API entrypoints, Lrt* option entrypoints, compilation cache, external weight
# loader, and runtime internals) contains no forbidden C++ stream symbols
# (iostream, stringstream, istringstream, ifstream, ofstream, fstream, filebuf).
# Usage: litert_runtime_symbol_test.sh <binary>

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

echo "Checking symbol table in $(basename "${BINARY}")..."
SYMBOLS=$(nm -C "${BINARY}")
if [ -z "${SYMBOLS}" ]; then
  echo "ERROR: Failed to read symbol table from $(basename "${BINARY}")"
  exit 1
fi

if ! grep -q "LiteRtCreateCompiledModel" <<< "${SYMBOLS}"; then
  echo "ERROR: Expected C API entrypoint (LiteRtCreateCompiledModel) not found in $(basename "${BINARY}")"
  exit 1
fi

if ! grep -q "LrtIntelOpenVinoOptionsCreate" <<< "${SYMBOLS}"; then
  echo "ERROR: Expected Lrt* options entrypoint (LrtIntelOpenVinoOptionsCreate) not found in $(basename "${BINARY}")"
  exit 1
fi

# Stream symbols: C++ std::iostream / std::stringstream / std::ifstream /
# std::ofstream pull in heavy virtual tables, locales, and formatting logic.
# Ensures LiteRT runtime, options, cache, and weight loader use lighter
# alternatives.
BANNED_STREAM_SYMBOLS="litert::.*basic_.*stream|default_delete<.*basic_.*stream|unique_ptr<.*basic_.*stream|basic_ifstream|basic_ofstream|basic_fstream|basic_stringstream|basic_istringstream|basic_iostream|basic_filebuf<.*>::|basic_istream<.*>::(read|seekg|tellg)"

LEAKED_SYMBOLS=$(grep -E "${BANNED_STREAM_SYMBOLS}" <<< "${SYMBOLS}" || true)
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
