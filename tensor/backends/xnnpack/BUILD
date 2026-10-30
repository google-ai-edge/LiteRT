# Copyright 2025 Google LLC.
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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "graph",
    hdrs = ["graph.h"],
    deps = [
        ":utils",
        "//tensor/backends/common_nnpack:graph",
        "//tensor/utils:macros",
        "@XNNPACK",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = [
        "//tensor/utils:macros",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:source_location",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "//tensor/utils:macros",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:source_location",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "arithmetic",
    srcs = ["arithmetic.cc"],
    hdrs = ["arithmetic.h"],
    deps = [
        ":conversion",
        ":utils",
        "//tensor:arithmetic_graph",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/common_nnpack:utils",
        "//tensor/internal:graph",
        "//tensor/internal:mixin",
        "//tensor/internal:type_id",
        "//tensor/utils:macros",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "conversion",
    srcs = ["conversion.cc"],
    hdrs = ["conversion.h"],
    deps = [
        ":graph",
        ":utils",
        "//tensor",
        "//tensor:datatypes",
        "//tensor/backends/common_nnpack:conversion",
        "//tensor/backends/common_nnpack:graph",
        "//tensor/internal:graph",
        "//tensor/internal:graph_traversal",
        "//tensor/internal:type_id",
        "//tensor/utils:macros",
        "@XNNPACK",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "conversion_test",
    srcs = ["conversion_test.cc"],
    deps = [
        ":arithmetic",
        ":conversion",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:datatypes",
        "//tensor/backends/common_nnpack:graph",
        "//tensor/utils:matchers",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "xnnpack_conversion_numerical_test",
    srcs = ["xnnpack_conversion_numerical_test.cc"],
    deps = [
        ":arithmetic",
        ":conversion",
        "//tensor",
        "//tensor/backends/testing:numerical_test_suite",
        "//tensor/runners/xnnpack:runner",
        "//tensor/utils:macros",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "arithmetic_test",
    srcs = ["arithmetic_test.cc"],
    deps = [
        ":arithmetic",
        ":conversion",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/common_nnpack:graph",
        "//tensor/runners/xnnpack:runner",
        "//tensor/utils:matchers",
        "@XNNPACK",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
