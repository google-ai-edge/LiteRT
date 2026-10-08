# Copyright 2026 Google LLC.
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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "bundle_loader",
    hdrs = [
        "bundle_loader.h",
        "kv_cache.h",
    ],
    deps = [
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/examples/gemma4:gemma4_config",
        "//tensor/examples/gemma4/helpers:quantized_embedding",
        "//tensor/examples/utils:minijson",
        "//tensor/utils:macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "graph",
    hdrs = [
        "active_graph_context.h",
        "attention.h",
        "bundle_matched_ops.h",
        "feed_forward_network.h",
        "gemma4_graph.h",
        "mobile_fully_connected.h",
        "transformer.h",
    ],
    deps = [
        ":bundle_loader",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/tflite:arithmetic_tflite",
        "//tensor/examples/gemma4:gemma4_config",
        "//tensor/examples/ops/transformer:transformer_ops",
        "//tensor/internal:graph",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "export_model",
    srcs = ["export_model.cc"],
    copts = ["-ffp-contract=off"],
    # Abseil strips flag names by default on Android; these are CLI tools.
    local_defines = ["GOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    deps = [
        ":bundle_loader",
        ":graph",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/tflite:arithmetic_tflite",
        "//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "//tensor/examples/gemma4:gemma4_config",
        "//tensor/utils:macros",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "runner_lib",
    hdrs = [
        "host_buffer.h",
        "kv_bank.h",
        "runner.h",
    ],
    deps = [
        ":bundle_loader",
        "//litert/cc:litert_common",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//tensor",
        "//tensor:buffer",
        "//tensor/examples/gemma4:gemma4_config",
        "//tensor/runners/litert:litert_dynamic_runner",
        "//tensor/utils:macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "runner",
    srcs = ["main.cc"],
    copts = ["-ffp-contract=off"],
    local_defines = ["GOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    deps = [
        ":runner_lib",
        "//litert/cc:litert_common",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_cpu_options",
        "//tensor:buffer",
        "//tensor/utils:macros",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "kv_bank_test",
    srcs = ["kv_bank_test.cc"],
    deps = [
        ":runner_lib",
        "@com_google_googletest//:gtest_main",
    ],
)
