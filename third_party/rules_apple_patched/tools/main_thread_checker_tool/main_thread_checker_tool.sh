# Copyright 2023 The Bazel Authors. All rights reserved.
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

# A tool that gets the libMainThreadChecker.dylib from the `DEVELOPER_DIR` and
# copies it to the requested output path for bundling.

set -euo pipefail

output_path=$1
developer_dir="${DEVELOPER_DIR:-}"

if [[ -z "${developer_dir}" ]]; then
  echo "DEVELOPER_DIR is not set, unable to find libMainThreadChecker.dylib"
  exit 1
fi

# Xcode 15+ example: /Applications/Xcode.15.0.0.15A240d.app/Contents/Developer/usr/lib/libMainThreadChecker.dylib
lib_path="${developer_dir}/usr/lib/libMainThreadChecker.dylib"

if [[ ! -f "${lib_path}" ]]; then
  echo "libMainThreadChecker.dylib not found at: ${lib_path}"
  exit 1
fi

cp "${lib_path}" "${output_path}"
