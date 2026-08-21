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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_test(
    name = "rmsnorm_test",
    srcs = ["rmsnorm_test.cc"],
    deps = [
        "//tensor",
        "//tensor:datatypes",
        "//tensor/backends/xnnpack:arithmetic",
        "//tensor/examples/ops/transformer:transformer_ops",
        "//tensor/examples/ops/transformer:transformer_ops_graph",
        "//tensor/examples/ops/transformer:transformer_ops_xnnpack",
        "//tensor/internal:arithmetic_helpers",
        "//tensor/internal:graph",
        "//tensor/runners/xnnpack:runner",
        "//tensor/utils:matchers",
        "//tensor/utils:source_location",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rope",
    hdrs = ["rope.h"],
    deps = [
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "rope_test",
    srcs = ["rope_test.cc"],
    deps = [
        ":rope",
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/xnnpack:arithmetic",
        "//tensor/runners/xnnpack:runner",
        "//tensor/utils:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "feed_forward_network",
    hdrs = ["feed_forward_network.h"],
    deps = [
        "//tensor",
        "//tensor:arithmetic",
    ],
)

cc_test(
    name = "feed_forward_network_test",
    srcs = ["feed_forward_network_test.cc"],
    deps = [
        ":feed_forward_network",
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/xnnpack:arithmetic",
        "//tensor/runners/xnnpack:runner",
        "//tensor/utils:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "attention",
    hdrs = ["attention.h"],
    deps = [
        ":rope",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/examples/gemma4:gemma4_config",
        "//tensor/examples/ops/transformer:transformer_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "attention_test",
    srcs = ["attention_test.cc"],
    deps = [
        ":attention",
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/xnnpack:arithmetic",
        "//tensor/examples/gemma4:gemma4_config",
        "//tensor/examples/ops/transformer:transformer_ops_xnnpack",
        "//tensor/runners/xnnpack:runner",
        "//tensor/utils:matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "transformer",
    hdrs = ["transformer.h"],
    deps = [
        ":attention",
        ":feed_forward_network",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:datatypes",
        "//tensor/examples/gemma4:gemma4_config",
        "//tensor/examples/ops/transformer:transformer_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "transformer_test",
    srcs = ["transformer_test.cc"],
    deps = [
        ":transformer",
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/xnnpack:arithmetic",
        "//tensor/examples/gemma4:gemma4_config",
        "//tensor/examples/ops/transformer:transformer_ops_xnnpack",
        "//tensor/runners/xnnpack:runner",
        "//tensor/utils:matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
    ],
)
