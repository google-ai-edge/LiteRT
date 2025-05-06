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

package(
    default_visibility = ["//visibility:private"],
)

# TODO(weiyiw): Figure out some way to utilize the version from BUILD.
# NOTE: When changing this version, please also update setup.py as well to match this version.
VERSION = "0.0.0"

sdist_rule(
    name = "ai_edge_litert_sdk_qualcomm_sdist",
    package_name = "ai_edge_litert_sdk_qualcomm",
    manifest_in = "MANIFEST.in",
    package_srcs = glob(["ai_edge_litert_sdk_qualcomm/**"]),
    sdist_archive_name = "ai_edge_litert_sdk_qualcomm-{VERSION}.tar.gz".format(VERSION = VERSION),
    setup_py = "setup.py",
)
