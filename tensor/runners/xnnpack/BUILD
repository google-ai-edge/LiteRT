# Copyright 2025 Google LLC.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "runner",
    srcs = ["runner.cc"],
    hdrs = ["runner.h"],
    deps = [
        "//tensor",
        "//tensor:buffer",
        "//tensor/backends/common_nnpack:graph",
        "//tensor/backends/xnnpack:conversion",
        "//tensor/backends/xnnpack:graph",
        "//tensor/backends/xnnpack:utils",
        "//tensor/runners/common_nnpack:runner",
        "//tensor/utils:macros",
        "@XNNPACK",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@pthreadpool",
    ],
)

cc_test(
    name = "runner_test",
    srcs = ["runner_test.cc"],
    deps = [
        ":runner",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:datatypes",
        "//tensor/backends/xnnpack:arithmetic",
        "//tensor/runners/common_nnpack:runner_test_suite",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "runner_profiling_test",
    srcs = ["runner_profiling_test.cc"],
    env = {"LRT_XNN_PROFILE": "1"},
    deps = [
        ":runner",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/xnnpack:arithmetic",
        "//tensor/utils:matchers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_googletest//:gtest_main",
    ],
)
