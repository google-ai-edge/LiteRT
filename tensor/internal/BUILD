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
    name = "mixin",
    hdrs = ["mixin.h"],
)

cc_library(
    name = "type_id",
    hdrs = ["type_id.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "type_id_test",
    srcs = ["type_id_test.cc"],
    deps = [
        ":type_id",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "arithmetic_helpers",
    hdrs = ["arithmetic_helpers.h"],
    deps = [
        ":graph",
        ":mixin",
        ":shape",
        ":utils",
        "//tensor",
        "//tensor:datatypes",
        "//tensor/utils:macros",
        "//tensor/utils:source_location",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "shape",
    srcs = ["shape.cc"],
    hdrs = ["shape.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "compile",
    hdrs = ["compile.h"],
    deps = [
        "//tensor",
        "//tensor:buffer",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "compile_test",
    srcs = ["compile_test.cc"],
    deps = [
        ":compile",
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "graph",
    srcs = ["graph.cc"],
    hdrs = ["graph.h"],
    defines = select({
        ":op_debugger_enabled": ["LITERT_OP_DEBUGGER_ENABLED=1"],
        "//conditions:default": [],
    }),
    includes = ["../utils"],  # copybara:comment
    deps = [
        ":type_id",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/utils:macros",
        "//tensor/utils:source_location",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "graph_test",
    srcs = ["graph_test.cc"],
    deps = [
        ":graph",
        ":type_id",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/utils:matchers",
        "//tensor/utils:source_location",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        "@com_google_googletest//:gtest",
    ],
)

# This config_setting checks if the Bazel flag "OP_DEBUGGER" is set to "true"
config_setting(
    name = "op_debugger_enabled",
    define_values = {
        "OP_DEBUGGER": "true",  # The key is the flag name, the value is the expected value
    },
    visibility = ["//visibility:private"],
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = [
        ":graph",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "graph_probe",
    srcs = ["graph_probe.cc"],
    hdrs = ["graph_probe.h"],
    deps = [
        ":graph",
        "//tensor",
        "//tensor/internal:graph_traversal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "graph_traversal",
    srcs = ["graph_traversal.cc"],
    hdrs = ["graph_traversal.h"],
    deps = [
        ":graph",
        "//tensor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "graph_traversal_test",
    srcs = ["graph_traversal_test.cc"],
    deps = [
        ":arithmetic_helpers",
        ":graph",
        ":graph_traversal",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor/utils:source_location",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "op_debugger_test",
    srcs = [
        "op_debugger_test.cc",
    ],
    copts = [
        "-DLITERT_OP_DEBUGGER_ENABLED=1",
    ],
    deps = [
        ":graph",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:datatypes",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fp16",
    hdrs = ["fp16.h"],
)
