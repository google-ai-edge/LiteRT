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

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <expected_symbol> <so_file1> [<so_file2> ...]"
  exit 1
fi

EXPECTED_SYMBOL="$1"
shift

SUCCESS=1

for SO_FILE in "$@"; do
  if [ ! -f "${SO_FILE}" ]; then
    echo "ERROR: Failed to find SO file: ${SO_FILE}"
    SUCCESS=0
    continue
  fi

  echo "Checking $(basename "${SO_FILE}")..."

  if readelf -d "${SO_FILE}" | grep -q "\(NEEDED\).*\[libLiteRt.so\]"; then
    echo "ERROR: $(basename "${SO_FILE}") depends on libLiteRt.so"
    SUCCESS=0
  fi

  symbols=$(nm -D "${SO_FILE}" | grep "LiteRt")

  num_lines=$(echo "${symbols}" | wc -l)
  echo "Found ${num_lines} LiteRt symbols:"
  echo "${symbols}"

  if [ "${num_lines}" -ne 1 ]; then
    echo "ERROR: Expected one 'LiteRt' symbol in $(basename "${SO_FILE}"), but found ${num_lines}."
    SUCCESS=0
    continue
  fi

  if ! grep -q "${EXPECTED_SYMBOL}" <<< "${symbols}"; then
    echo "ERROR: ${EXPECTED_SYMBOL} should be exported in $(basename "${SO_FILE}")."
    SUCCESS=0
  fi
done

if [ $SUCCESS -eq 1 ]; then
  echo "PASS"
else
  echo "FAIL"
  exit 1
fi
