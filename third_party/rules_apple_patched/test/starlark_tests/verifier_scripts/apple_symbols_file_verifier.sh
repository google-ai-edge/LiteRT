#!/bin/bash

# Copyright 2020 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Usage: assert_unpacked_archive_contains_symbols_directory <archive> <paths...>
#
# Asserts that the unpacked application (.ipa/.zip) archive at `archive`
# contains a Symbols directory with a `.symbols` file for each platform of each
# binary inside the unpacked archive matching the UUID of that binary.
#
# This exists separately from `archive_contents_test.sh` since the names of the
# `*.symbols` file(s) depend on the UUIDs of the binaries in the unpacked
# archive.
function assert_unpacked_archive_contains_symbols_directory() {
  local archive="$1" ; shift

  # Get the list of UUIDs and platforms present in all the `$UUID.symbols`
  # files in the archive under `Symbols/*.symbols`.
  local symbols_uuid_output="${TEST_TMPDIR}/symbols_uuid_output"
  xcrun symbols -noDaemon -uuid "${archive}/Symbols" > "${symbols_uuid_output}"

  for binary in "$@"; do
    # Each binary can contain one or more architectures, each with its own UUID.
    dwarfdump -u "${archive}/${binary}" | while read line ; do
      local -a uuid_and_platform=(
        $(echo "${line}" | sed -e 's/UUID: \([^ ]*\) (\([^)]*\)).*/\1 \2/') )
      local uuid="${uuid_and_platform[0]}"
      local platform="${uuid_and_platform[1]}"
      assert_contains "${uuid}[[:space:]]\{1,\}${platform}" \
        "${symbols_uuid_output}"
    done
  done

  rm -f "${symbols_uuid_output}"
}

assert_unpacked_archive_contains_symbols_directory "${ARCHIVE_ROOT}" \
  "${BINARY_PATHS[@]}"
