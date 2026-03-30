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

SHARED_LIB="$1"
LITERT_LIB="$2"

if [[ -z "${SHARED_LIB}" || -z "${LITERT_LIB}" ]]; then
  echo "Usage: $0 <path_to_dynamic_runtime_lib> <path_to_libLiteRt.so>"
  exit 1
fi

LITERT_SYMS=$(mktemp)
RUNTIME_SYMS=$(mktemp)
trap 'rm -f "${LITERT_SYMS}" "${RUNTIME_SYMS}"' EXIT

# Extract globally defined dynamic symbols
nm -D -g --defined-only "${LITERT_LIB}" | awk '{print $NF}' | sort > "${LITERT_SYMS}"
nm -D -g --defined-only "${SHARED_LIB}" | awk '{print $NF}' | sort > "${RUNTIME_SYMS}"

if [[ ! -s "${LITERT_SYMS}" ]]; then
  echo "ERROR: Failed to extract symbols from ${LITERT_LIB} using nm."
  exit 1
fi

if [[ ! -s "${RUNTIME_SYMS}" ]]; then
  echo "ERROR: Failed to extract symbols from ${SHARED_LIB} using nm."
  exit 1
fi

# Ignore standard C/C++ linker and allocator boilerplate
BOILERPLATE_REGEX="^_init$|^_fini$|^_edata$|^_end$|^__bss_start$|^__start_.*|^__stop_.*"

# TEMPORARY ALLOWLIST: Read from the BUILD file environment variable.
# If not set, default to ^$ (matches empty string) so we don't accidentally filter everything.
ALLOWLIST="${ALLOWLIST_REGEX:-^$}"

# Find duplicates and pipe through both filters
DUPLICATES=$(comm -12 "${LITERT_SYMS}" "${RUNTIME_SYMS}" | \
             grep -v -E "${BOILERPLATE_REGEX}" | \
             grep -v -E "${ALLOWLIST}")

if [[ -n "${DUPLICATES}" ]]; then
  echo "ERROR: Duplicate symbol linkage detected!"
  echo "The library ${SHARED_LIB} statically links the following symbols that are owned by ${LITERT_LIB}:"
  echo ""
  echo "${DUPLICATES}"
  echo ""
  echo "Targets in dynamic_runtime should NOT statically link these functions to avoid duplicate linkage with libLiteRt.so at runtime."
  exit 1
fi

echo "SUCCESS: No unexpected overlapping symbols found between dynamic_runtime and libLiteRt.so."
exit 0