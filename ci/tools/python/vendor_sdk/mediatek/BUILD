# Copyright 2025 The AI Edge LiteRT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("//ci/tools/python/vendor_sdk:sdist.bzl", "sdist_rule")
load("//litert:version.bzl", "LITERT_SDK_MEDIATEK_FULL_VERSION")

package(
    default_visibility = ["//visibility:private"],
)

PROJECT_NAME = "ai_edge_litert_sdk_mediatek"

sdist_rule(
    name = "ai_edge_litert_sdk_mediatek_sdist",
    package_name = PROJECT_NAME,
    manifest_in = "MANIFEST.in",
    nightly_suffix = "//ci/tools/python/wheel:nightly_iso_date",
    package_srcs = glob(["ai_edge_litert_sdk_mediatek/**"]),
    setup_py = "setup.py",
    version = LITERT_SDK_MEDIATEK_FULL_VERSION,
)
