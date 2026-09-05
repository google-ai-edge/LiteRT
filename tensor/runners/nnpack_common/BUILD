# Copyright 2026 Google LLC.
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

load("@rules_cc//cc:defs.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//tensor:__subpackages__"],
)

cc_library(
    name = "runner",
    hdrs = ["runner.h"],
    deps = [
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/nnpack_common:conversion",
        "//tensor/backends/nnpack_common:utils",
        "//tensor/internal:graph",
        "//tensor/utils:macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "runner_test_suite",
    testonly = True,
    hdrs = ["runner_test_suite.h"],
    deps = [
        ":runner",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:datatypes",
        "//tensor/utils:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)
