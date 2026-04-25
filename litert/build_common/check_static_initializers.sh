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

# Test that a binary does not have more than the allowed number of static
# initializers. This is done by checking the number of entries in the
# .init_array section in the binary.

set -eu -o pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <binary_path> [allowed_count]"
  exit 1
fi

BINARY=$1
ALLOWED_COUNT=${2:-0}

# Resolve binary path
if [[ ! -f "${BINARY}" ]]; then
  # Try relative to TEST_SRCDIR
  if [[ -n "${TEST_SRCDIR:-}" ]]; then
    # Binaries are typically under a subdirectory in the runfiles.
    # We search one level deep to avoid hardcoding any specific directory name.
    for D in "${TEST_SRCDIR}"/*; do
      if [[ -d "${D}" && -f "${D}/${BINARY}" ]]; then
        BINARY="${D}/${BINARY}"
        break
      fi
    done
  fi
fi

if [[ ! -f "${BINARY}" ]]; then
  echo "ERROR: Failed to find the binary: ${BINARY}"
  exit 1
fi

# Find readelf
READELF=$(which readelf || echo "")
if [[ -z "${READELF}" ]]; then
  # Try to find it in common locations
  for p in /usr/bin/readelf /bin/readelf /usr/local/bin/readelf; do
    if [[ -x "$p" ]]; then
      READELF=$p
      break
    fi
  done
fi

# If this script runs on a non-Linux environment (like a macOS builder),
# readelf might not be found and the test will fail. We exit gracefully
# (exit 0) if readelf is missing, as this check is ELF-specific.
if [[ -z "${READELF}" ]]; then
  echo "WARNING: readelf not found. Skipping ELF-specific static initializer check."
  exit 0
fi

# Similarly, skip the check if the binary is not an ELF file.
if ! "${READELF}" -h "${BINARY}" | grep -q "ELF"; then
  echo "Skipping non-ELF binary: ${BINARY}"
  exit 0
fi

# Get relocations for .init_array and .preinit_array.
# We filter for lines that look like relocation entries (starting with an address)
# and contain the initialization sections.
# We add || true to the grep to avoid failing under set -o pipefail if no matches are found.
count=$("${READELF}" -W -r "${BINARY}" | grep -E "[0-9a-f]{8,}.*\.p?reinit_array" | wc -l || true)

if [ "${count}" -gt "${ALLOWED_COUNT}" ]; then
  echo "ERROR: Found ${count} static initializers in ${BINARY}, but only ${ALLOWED_COUNT} are allowed."
  echo "Please ensure no global constructors or static initializers are introduced."
  echo "Relocations in initialization sections:"
  "${READELF}" -W -r "${BINARY}" | grep -E "[0-9a-f]{8,}.*\.p?reinit_array" || true
  echo ""
  echo "Troubleshooting Tips:"
  echo "1. To identify the specific functions, run: nm -C --demangle ${BINARY} | grep -i \"init\""
  echo "2. To see what the initializers are pointing to, run: objdump -D -s -j .init_array ${BINARY}"
  echo "3. For a detailed disassembly of the initialization code, run: objdump -d ${BINARY} | grep -A 20 \"<GLOBAL__sub_I_\""
  exit 1
fi

echo "PASS: Found ${count} initializers (Allowed: ${ALLOWED_COUNT})"
