#!/bin/bash

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

set -exuo pipefail

# Collect templated variables
additional_symbol_graph_dirs=({symbol_graph_dirs})
docc_bundle_path="{docc_bundle}"
fallback_bundle_identifier="{fallback_bundle_identifier}"
fallback_bundle_version="{fallback_bundle_version}"
fallback_display_name="{fallback_display_name}"
platform="{platform}"
sdk_version="{sdk_version}"
xcode_version="{xcode_version}"

arguments=()
output_dir="$(mktemp -d)"

# Add all symbol graph directories to the arguments
for additional_symbol_graph_dir in "${additional_symbol_graph_dirs[@]}"; do
  arguments+=("--additional-symbol-graph-dir" "$additional_symbol_graph_dir")
done

# Add the docc bundle path to the arguments
if [ -d "$docc_bundle_path" ]; then
  arguments+=("$docc_bundle_path")
fi

# Preview the docc archive
cd "$BUILD_WORKSPACE_DIRECTORY"
env -i \
  APPLE_SDK_PLATFORM="$platform" \
  APPLE_SDK_VERSION_OVERRIDE="$sdk_version" \
  XCODE_VERSION_OVERRIDE="$xcode_version" \
  /usr/bin/xcrun docc preview \
  --fallback-display-name "$fallback_display_name" \
  --fallback-bundle-identifier "$fallback_bundle_identifier" \
  --fallback-bundle-version "$fallback_bundle_version" \
  --output-dir "$output_dir" \
  "${arguments[@]:-}"
