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
    name = "tensor",
    srcs = [
        "tensor.cc",
    ],
    hdrs = [
        "tensor.h",
    ],
    includes = ["utils"],  # copybara:comment
    deps = [
        ":buffer",
        ":datatypes",
        "//tensor/internal:graph",
        "//tensor/utils:macros",
        "//tensor/utils:source_location",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "buffer",
    srcs = ["buffer.cc"],
    hdrs = ["buffer.h"],
    includes = ["utils"],  # copybara:comment
    deps = [
        "//tensor:datatypes",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "buffer_test",
    srcs = ["buffer_test.cc"],
    deps = [
        ":buffer",
        "//tensor:datatypes",
        "//tensor/internal:matchers",
        "//tensor/utils:matchers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "datatypes",
    hdrs = ["datatypes.h"],
    deps = [
        "//tensor/internal:fp16",
    ],
)

cc_test(
    name = "datatypes_test",
    srcs = ["datatypes_test.cc"],
    deps = [
        ":datatypes",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "arithmetic_graph",
    hdrs = ["arithmetic_graph.h"],
    deps = [
        "//tensor:datatypes",
        "//tensor/internal:graph",
        "//tensor/internal:mixin",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "arithmetic",
    hdrs = ["arithmetic.h"],
    deps = [
        ":arithmetic_graph",
        ":buffer",
        ":datatypes",
        ":tensor",
        "//tensor/internal:arithmetic_helpers",
        "//tensor/internal:graph",
        "//tensor/utils:macros",
        "//tensor/utils:source_location",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":arithmetic",
        ":buffer",
        ":datatypes",
        ":tensor",
        "//tensor/internal:graph",
        "//tensor/utils:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "arithmetic_test",
    srcs = ["arithmetic_test.cc"],
    deps = [
        ":arithmetic",
        ":arithmetic_graph",
        ":tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/internal:graph",
        "//tensor/utils:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
    ],
)
