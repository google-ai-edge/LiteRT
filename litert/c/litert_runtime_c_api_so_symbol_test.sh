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

# Function to exit with an error message
die() {
  echo "ERROR: $@"
  exit 1
}

# Check if TEST_SRCDIR is set
if [[ -z "${TEST_SRCDIR}" ]]; then
  die "TEST_SRCDIR is not set"
fi

SO_FILE="${TEST_SRCDIR}/litert/c/libLiteRt.so"
EXPECTED_SYMBOLS="${TEST_SRCDIR}/litert/c/litert_runtime_c_api_so_symbols.txt"

if [[ ! -f "${SO_FILE}" ]]; then
  die "Failed to find the shared library: ${SO_FILE}"
fi

if [[ ! -f "${EXPECTED_SYMBOLS}" ]]; then
  die "Failed to find the expected symbols file: ${EXPECTED_SYMBOLS}"
fi

# Get all dynamic symbols
symbols=$(nm -D ${SO_FILE})

# Filter for LiteRt symbols (type T means text/code symbol, exported)
# Remove @@VERS_1.0 to match the file content
litert_symbols=$(echo "$symbols" | grep " T LiteRt" | awk '{print $3}' | sed 's/@@VERS_1.0//' | LC_ALL=C sort)

# Filter out comments and empty lines from expected symbols
expected_filtered=$(grep -v "^#" "${EXPECTED_SYMBOLS}" | grep -v "^$")

# Compare with expected symbols
diff_output=$(diff -u <(echo "$expected_filtered") <(echo "$litert_symbols"))

if [ $? -ne 0 ]; then
  echo "ERROR: Exported symbols do not match expected symbols."
  echo "Diff:"
  echo "$diff_output"
  exit 1
fi

echo "PASS"
