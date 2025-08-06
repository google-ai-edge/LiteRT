load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
load("//tflite/experimental/acceleration/mini_benchmark:special_rules.bzl", "libjpeg_handle_deps", "register_selected_ops_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//tflite/experimental/acceleration/mini_benchmark:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "grafter",
    srcs = ["grafter.cc"],
    hdrs = ["grafter.h"],
    deps = [
        "//tflite:framework",
        "//tflite/schema:schema_fbs_with_reflection",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
    ],
)

cc_library(
    name = "embedder",
    srcs = ["embedder.cc"],
    hdrs = ["embedder.h"],
    deps = [
        ":validation_graph_builder",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
        "//tflite:framework",
        "//tflite:minimal_logging",
        "//tflite/core:framework",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/experimental/acceleration/mini_benchmark:decode_jpeg_status",
        "//tflite/experimental/acceleration/mini_benchmark:jpeg_common",
        "//tflite/experimental/acceleration/mini_benchmark:jpeg_header_parser",
        # TODO(bekzhan): Remove duplicate dependency when only one of the two schemas is used.
        "//tflite/schema:schema_fbs",
        "//tflite/schema:schema_fbs_with_reflection",
    ],
)

cc_library(
    name = "validation_graph_builder",
    srcs = ["validation_graph_builder.cc"],
    hdrs = ["validation_graph_builder.h"],
    deps = [
        ":grafter",
        "//tflite:framework",
        "//tflite:string_util",
        "//tflite/core/c:common",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
    ],
)

cc_binary(
    name = "embedder_cmdline",
    srcs = ["embedder_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":embedder",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core:model_builder",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/experimental/acceleration/mini_benchmark:call",
        "//tflite/experimental/acceleration/mini_benchmark:decode_jpeg",
        "//tflite/schema:schema_fbs",
        "//tflite/schema:schema_fbs_with_reflection",
        "//tflite/tools:command_line_flags",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ] + libjpeg_handle_deps() + register_selected_ops_deps(),
)

cc_library(
    name = "custom_validation_embedder",
    srcs = ["custom_validation_embedder.cc"],
    hdrs = ["custom_validation_embedder.h"],
    deps = [
        "//tflite:stderr_reporter",
        "//tflite/core/api:error_reporter",
        "//tflite/core/tools:verifier",
        "//tflite/experimental/acceleration/mini_benchmark:constants",
        "//tflite/experimental/acceleration/mini_benchmark:status_codes",
        "//tflite/schema:schema_fbs",
        "@flatbuffers",
    ],
)

cc_test(
    name = "custom_validation_embedder_test",
    srcs = ["custom_validation_embedder_test.cc"],
    deps = [
        ":custom_validation_embedder",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/api:error_reporter",
        "//tflite/core/c:c_api_types",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/experimental/acceleration/mini_benchmark:call",
        "//tflite/experimental/acceleration/mini_benchmark:embedded_mobilenet_model",
        "//tflite/experimental/acceleration/mini_benchmark:mini_benchmark_test_helper",
        "//tflite/experimental/acceleration/mini_benchmark:status_codes",
        "//tflite/kernels/internal:tensor",
        "//tflite/schema:schema_fbs",
        "//tflite/tools:model_loader",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ] + libjpeg_handle_deps() + register_selected_ops_deps(),
)
