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

package(default_visibility = ["//visibility:public"])

# This POSIX native experiment uses XNNPACK's public target for both its API and
# transitive private headers. Private XNNPACK Bazel targets are not externally
# visible. Keep the dependency pinned to the validated source revision; see README.
cc_library(
    name = "native_support",
    srcs = ["model/helpers/int8_kv_cache.cc"],
    hdrs = [
        "active_kv_bank.h",
        "active_runtime_audit.h",
        "driver_support.h",
        "matched_bundle_loader.h",
        "memory_snapshot.h",
        "stage_runner.h",
        "model/gemma4_graph.h",
        "model/helpers/active_graph_context.h",
        "model/helpers/attention.h",
        "model/helpers/bundle_matched_ops.h",
        "model/helpers/feed_forward_network.h",
        "model/helpers/int8_kv_cache.h",
        "model/helpers/mobile_fully_connected.h",
        "model/helpers/static_int2_fully_connected.h",
        "model/helpers/transformer.h",
    ],
    textual_hdrs = ["model/helpers/published_kv_specs.inc"],
    copts = ["-ffp-contract=off"],
    deps = [
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/xnnpack:arithmetic",
        "//tensor/backends/xnnpack:conversion",
        "//tensor/backends/xnnpack:graph",
        "//tensor/backends/xnnpack:utils",
        "//tensor/examples/gemma4:gemma4_config",
        "//tensor/examples/gemma4/helpers:float_activation_fully_connected",
        "//tensor/examples/gemma4/helpers:quantized_embedding",
        "//tensor/examples/gemma4/helpers:rope",
        "//tensor/examples/ops/transformer:transformer_ops",
        "//tensor/examples/ops/transformer:transformer_ops_xnnpack",
        "//tensor/examples/utils:minijson",
        "//tensor/runners/common_nnpack:runner",
        "//tensor/runners/xnnpack:runner",
        "//tensor/utils:macros",
        "@XNNPACK",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@pthreadpool",
    ],
)

cc_binary(
    name = "gemma4_native_runner",
    srcs = ["driver.cc"],
    copts = ["-ffp-contract=off"],
    deps = [":native_support"],
)

# These tests provide their own main and need no model files or gtest_main.

cc_test(
    name = "stage_runner_smoke_test",
    srcs = ["tests/stage_runner_smoke_test.cc"],
    copts = ["-ffp-contract=off"],
    deps = [":native_support"],
)

cc_test(
    name = "stage_workspace_test",
    srcs = ["tests/stage_workspace_test.cc"],
    copts = ["-ffp-contract=off"],
    deps = [":native_support"],
)

cc_test(
    name = "active_kv_bank_test",
    srcs = ["tests/active_kv_bank_test.cc"],
    copts = ["-ffp-contract=off"],
    deps = [":native_support"],
)

cc_test(
    name = "active_attention_test",
    srcs = ["tests/active_attention_test.cc"],
    copts = ["-ffp-contract=off"],
    deps = [":native_support"],
)

cc_test(
    name = "static_int2_test",
    srcs = ["tests/static_int2_test.cc"],
    copts = ["-ffp-contract=off"],
    deps = [":native_support"],
)

# The rounding test textually reuses the attention test's helpers with main
# renamed. Declare the included source as a textual header to avoid a second TU.
cc_library(
    name = "active_attention_test_source",
    testonly = True,
    textual_hdrs = ["tests/active_attention_test.cc"],
    deps = [":native_support"],
)

cc_test(
    name = "active_extent_rounding_test",
    srcs = ["tests/active_extent_rounding_test.cc"],
    copts = ["-ffp-contract=off"],
    deps = [":active_attention_test_source"],
)

# This integration executable takes a verified matched-bundle directory argv.
# Keep it out of wildcard test runs; the multi-GB fixture is not repository data.
cc_binary(
    name = "static_int2_loader_test",
    testonly = True,
    srcs = ["tests/static_int2_loader_test.cc"],
    copts = ["-ffp-contract=off"],
    tags = ["manual"],
    deps = [":native_support"],
)
