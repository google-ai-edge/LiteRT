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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//litert/build_common:litert_build_defs.bzl", "commandline_flag_copts")
load(
    "//litert/build_common:special_rule.bzl",
    "litert_android_linkopts",
    "litert_gpu_accelerator_deps",
    "litert_gpu_accelerator_prebuilts",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "safetensors",
    hdrs = ["safetensors.h"],
)

cc_library(
    name = "tokenizer",
    srcs = ["tokenizer.cc"],
    hdrs = ["tokenizer.h"],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@sentencepiece//:sentencepiece_processor",
    ],
)

cc_library(
    name = "safetensor_loader",
    srcs = ["safetensor_loader.cc"],
    hdrs = ["safetensor_loader.h"],
    deps = [
        ":safetensors",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        # copybara:uncomment "//tensor/backends/xnnpack:arithmetic",
        "//tensor/utils:macros",
    ],
)

# copybara:uncomment_begin(google_only)
# cc_library(
#     name = "gemma3_graph",
#     srcs = ["gemma3_graph.cc"],
#     hdrs = ["gemma3_graph.h"],
#     deps = [
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/log:absl_check",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/strings",
#         "//tensor",
#         "//tensor:arithmetic",
#         "//tensor:buffer",
#         "//tensor:datatypes",
#         "//tensor/backends/ml_drift:arithmetic_ml_drift",
#         "//tensor/backends/xnnpack:arithmetic",
#         "//tensor/examples/ops/transformer:transformer_ops",
#         "//tensor/examples/ops/transformer:transformer_ops_ml_drift",
#         "//tensor/examples/ops/transformer:transformer_ops_xnnpack",
#         "//tensor/internal:graph",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        # copybara:uncomment "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "graph_helpers",
    srcs = ["graph_helpers.cc"],
    hdrs = ["graph_helpers.h"],
    deps = [
        ":config",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/tflite:arithmetic_tflite",
        "//tensor/examples/ops/transformer:transformer_ops",
        "//tensor/utils:macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "config",
    hdrs = ["config.h"],
)

cc_library(
    name = "tflite_loader",
    srcs = ["tflite_loader.cc"],
    hdrs = ["tflite_loader.h"],
    deps = [
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/utils:macros",
        "//tflite:model_builder",
        "//tflite/core:model_builder",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "gemma3",
    hdrs = ["gemma3.h"],
    deps = [
        ":config",
        ":graph_helpers",
        "//tensor",
        "//tensor:arithmetic_graph",
        "//tensor:buffer",
        "//tensor:datatypes",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

# copybara:uncomment_begin(google_only)
# cc_library(
#     name = "benchmark_utils",
#     hdrs = ["benchmark_utils.h"],
#     deps = [":tokenizer"],
# )
#
# cc_library(
#     name = "model_config",
#     srcs = ["model_config.cc"],
#     hdrs = ["model_config.h"],
#     deps = [
#         ":gemma3_graph",
#         ":safetensor_loader",
#         ":tflite_loader",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/status:statusor",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/strings:string_view",
#     ],
# )
# copybara:uncomment_end

cc_binary(
    name = "litert_main",
    srcs = ["litert_main.cc"],
    copts = select({
        "//litert:android": [],
        "//conditions:default": ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    }) + commandline_flag_copts(),
    data = litert_gpu_accelerator_prebuilts(),
    linkopts = litert_android_linkopts(),
    deps = [
        ":config",
        ":gemma3",
        ":graph_helpers",
        ":safetensor_loader",
        ":tokenizer",
        ":util",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/cc:litert_common",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_gpu_options",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:arithmetic_graph",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/tflite:arithmetic_tflite",
        "//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "//tensor/runners/litert:litert_dynamic_runner",
        "//tensor/utils:macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ] + litert_gpu_accelerator_deps(),
)

# copybara:uncomment_begin(google_only)
# cc_binary(
#     name = "tflite_fb_main",
#     srcs = ["tflite_fb_main.cc"],
#     copts = select({
#         "//litert:android": [],
#         "//conditions:default": ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
#     }) + commandline_flag_copts(),
#     data = litert_gpu_accelerator_prebuilts(),
#     linkopts = litert_android_linkopts(),
#     deps = [
#         ":config",
#         ":gemma3",
#         ":graph_helpers",
#         ":tflite_loader",
#         ":tokenizer",
#         ":util",
#         "//base",
#         "@com_google_absl//absl/cleanup",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/flags:flag",
#         "@com_google_absl//absl/flags:parse",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/status:statusor",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/time",
#         "@com_google_absl//absl/types:span",
#         "//third_party/gloop/base:init_google",
#         "//litert/cc:litert_api_with_dynamic_runtime",
#         "//litert/cc:litert_common",
#         "//litert/cc:litert_environment",
#         "//litert/cc:litert_options",
#         "//litert/cc/options:litert_gpu_options",
#         "//tensor",
#         "//tensor:arithmetic",
#         "//tensor:arithmetic_graph",
#         "//tensor:buffer",
#         "//tensor:datatypes",
#         "//tensor/backends/tflite:arithmetic_tflite",
#         "//tensor/backends/tflite:tflite_flatbuffer_conversion",
#         "//tensor/runners/litert:litert_dynamic_runner",
#         "//tensor/utils:macros",
#     ] + litert_gpu_accelerator_deps(),
# )
#
# cc_binary(
#     name = "xnnpack_main",
#     srcs = ["xnnpack_main.cc"],
#     copts = [
#         "-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
#     ],
#     deps = [
#         ":config",
#         ":gemma3",
#         ":gemma3_graph",
#         ":graph_helpers",
#         ":model_config",
#         ":safetensor_loader",
#         ":tokenizer",
#         ":util",
#         "//base",
#         "@com_google_absl//absl/algorithm:container",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/flags:flag",
#         "@com_google_absl//absl/flags:parse",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/time",
#         "@com_google_absl//absl/types:span",
#         "//tensor",
#         "//tensor:buffer",
#         "//tensor:datatypes",
#         "//tensor/backends/xnnpack:arithmetic",
#         "//tensor/runners/xnnpack:runner",
#         "//tensor/utils:macros",
#     ],
# )
#
# cc_binary(
#     name = "webgpu_main",
#     srcs = ["webgpu_main.cc"],
#     deps = [
#         ":benchmark_utils",
#         ":gemma3_graph",
#         ":model_config",
#         ":safetensor_loader",
#         ":tokenizer",
#         "//base",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/flags:flag",
#         "@com_google_absl//absl/flags:parse",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/status:statusor",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/time",
#         "@com_google_absl//absl/types:span",
#         "@ml_drift//ml_drift/common:precision",
#         "//tensor",
#         "//tensor:buffer",
#         "//tensor:datatypes",
#         "//tensor/backends/ml_drift:arithmetic_ml_drift",
#         "//tensor/backends/xnnpack:arithmetic",
#         "//tensor/backends/xnnpack:conversion",
#         "//tensor/internal:graph",
#         "//tensor/runners/ml_drift:ml_drift_webgpu_model_runner",
#         "//tensor/utils:macros",
#     ],
# )
#
# cc_binary(
#     name = "opencl_main",
#     srcs = ["opencl_main.cc"],
#     copts = [
#         "-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
#     ],
#     deps = [
#         ":benchmark_utils",
#         ":gemma3_graph",
#         ":model_config",
#         ":safetensor_loader",
#         ":tokenizer",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/flags:flag",
#         "@com_google_absl//absl/flags:parse",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/status:statusor",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/strings:string_view",
#         "@com_google_absl//absl/time",
#         "@com_google_absl//absl/types:span",
#         "@ml_drift//ml_drift/common:precision",
#         "//tensor",
#         "//tensor:buffer",
#         "//tensor:datatypes",
#         "//tensor/backends/ml_drift:arithmetic_ml_drift",
#         "//tensor/internal:graph",
#         "//tensor/runners/ml_drift:ml_drift_cl_model_runner",
#         "//tensor/utils:macros",
#     ],
# )
#
# cc_binary(
#     name = "mldrift_main",
#     srcs = ["mldrift_main.cc"],
#     copts = [
#         "-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
#     ],
#     deps = [
#         ":benchmark_utils",
#         ":gemma3_graph",
#         ":model_config",
#         ":tflite_loader",
#         ":tokenizer",
#         "@com_google_absl//absl/algorithm:container",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/flags:flag",
#         "@com_google_absl//absl/flags:parse",
#         "@com_google_absl//absl/log:absl_check",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/status:statusor",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/strings:string_view",
#         "@com_google_absl//absl/time",
#         "@com_google_absl//absl/types:span",
#         "@ml_drift//ml_drift/cl:environment",
#         "@ml_drift//ml_drift/cl:opencl_wrapper",
#         "@ml_drift//ml_drift/common:gpu_model",
#         "@ml_drift//ml_drift/common:precision",
#         "//tensor",
#         "//tensor:buffer",
#         "//tensor:datatypes",
#         "//tensor/backends/ml_drift:arithmetic_ml_drift",
#         "//tensor/backends/ml_drift:ml_drift_buffer",
#         "//tensor/backends/ml_drift:ml_drift_helpers",
#         "//tensor/internal:graph",
#         "//tensor/runners/ml_drift:ml_drift_cl_model_runner",
#         "//tensor/utils:macros",
#     ],
# )
# copybara:uncomment_end
