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
# ==============================================================================

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "gemma4_config",
    hdrs = ["gemma4_config.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "gemma4_config_test",
    srcs = ["gemma4_config_test.cc"],
    deps = [
        ":gemma4_config",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gemma4_graph",
    hdrs = ["gemma4_graph.h"],
    deps = [
        ":gemma4_config",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/examples/gemma4/helpers:transformer",
        "//tensor/examples/ops/transformer:transformer_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "gemma4_graph_test",
    srcs = ["gemma4_graph_test.cc"],
    deps = [
        ":gemma4_config",
        ":gemma4_graph",
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/xnnpack:arithmetic",
        "//tensor/examples/ops/transformer:transformer_ops_xnnpack",
        "//tensor/runners/xnnpack:runner",
        "//tensor/utils:matchers",
        "//tflite/delegates/xnnpack:weight_cache",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gemma4_weights",
    srcs = ["gemma4_weights.cc"],
    hdrs = ["gemma4_weights.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "gemma4_xnnpack_main",
    srcs = ["xnnpack_main.cc"],
    deps = [
        ":gemma4_config",
        ":gemma4_graph",
        ":gemma4_weights",
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/xnnpack:arithmetic",
        "//tensor/examples/gemma3:tokenizer",
        "//tensor/examples/gemma3:util",
        "//tensor/examples/gemma4/helpers:quantized_embedding",
        "//tensor/examples/gemma4/helpers:rope",
        "//tensor/examples/ops/transformer:transformer_ops_xnnpack",
        "//tensor/examples/utils:initialization",
        "//tensor/examples/utils:perfetto_session",
        "//tensor/examples/utils:safetensor_loader",
        "//tensor/runners/xnnpack:runner",
        "//tensor/utils:macros",
        "//tflite/delegates/xnnpack:weight_cache",
        "@XNNPACK",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@perfetto//:libperfetto_client_experimental",
    ],
)
