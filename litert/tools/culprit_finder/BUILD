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
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "culprit_finder_utils",
    hdrs = ["culprit_finder_utils.h"],
    copts = select({
        "//litert:without_builtin_ops": ["-DLITERT_NO_BUILTIN_OPS"],
        "//conditions:default": [],
    }),
    deps = [
        "//litert/c/internal:litert_logging",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:cc_api_stable",
    ] + select({
        "//litert:without_builtin_ops": [],
        "//conditions:default": ["//tflite/kernels:builtin_ops"],
    }) + [
        "//tflite/tools:command_line_flags",
        "//tflite/tools:tool_params",
    ],
)

cc_library(
    name = "model_metadata_lib",
    srcs = ["model_metadata_lib.cc"],
    hdrs = ["model_metadata_lib.h"],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],  # Incompatible with -fexceptions.
    visibility = ["//visibility:public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//tflite:framework_stable",
        "//tflite/profiling:model_runtime_info",
        "//tflite/tools:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "model_metadata_lib_test",
    srcs = ["model_metadata_lib_test.cc"],
    data = [
        "//litert/test:testdata/mobilenet_v2_1.0_224.tflite",
    ],
    deps = [
        ":interpreter_handler",
        ":model_metadata_lib",
        "//litert/cc:litert_expected",
        "//tflite:framework",
        "//tflite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        "//tflite/tools/delegates:delegate_provider_hdr",
        "//tflite/tools/delegates:delegate_provider_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tflite_input_manager",
    srcs = ["tflite_input_manager.cc"],
    hdrs = ["tflite_input_manager.h"],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],  # Incompatible with -fexceptions.
    visibility = ["//visibility:public"],
    deps = [
        "//litert/c/internal:litert_logging",
        "//tflite:framework_stable",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/tools:utils",
    ],
)

cc_library(
    name = "interpreter_handler",
    srcs = ["interpreter_handler.cc"],
    hdrs = ["interpreter_handler.h"],
    copts = ["-fexceptions"] + select({
        "//litert:without_builtin_ops": ["-DLITERT_NO_BUILTIN_OPS"],
        "//conditions:default": [],
    }),
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],  # Incompatible with -fexceptions.
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_input_manager",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/core:model_builder",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ] + select({
        "//litert:without_builtin_ops": [],
        "//conditions:default": ["//tflite/kernels:builtin_ops"],
    }) + [
        "//tflite/tools:model_loader",
        "//tflite/tools/delegates:delegate_provider_hdr",
    ],
)

cc_library(
    name = "culprit_finder_lib",
    srcs = ["culprit_finder_lib.cc"],
    hdrs = ["culprit_finder_lib.h"],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],  # Incompatible with -fexceptions.
    visibility = ["//visibility:public"],
    deps = [
        ":culprit_finder_utils",
        ":interpreter_handler",
        ":model_metadata_lib",
        ":tflite_input_manager",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment_begin(google-only)
        # "//third_party/odml/infra/ml_drift_delegate:ml_drift_delegate_provider",
        # copybara:uncomment_end
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_tflite_error_status_builder",
        "//tflite/profiling/proto:model_runtime_info_cc",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:cc_api_stable",
    ] + select({
        "//litert:without_builtin_ops": [],
        "//conditions:default": ["//tflite/kernels:builtin_ops"],
    }) + [
        "//tflite/profiling:memory_usage_monitor",
        "//tflite/profiling:model_runtime_info",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:logging",
        "//tflite/tools:model_loader",
        "//tflite/tools:tool_params",
        "//tflite/tools:utils",
        "//tflite/tools/delegates:default_execution_provider",
        "//tflite/tools/delegates:delegate_provider_hdr",
    ] + select({
        # Metal GPU delegate for iOS has its own setups for kernel tests, so
        # skipping linking w/ the gpu_delegate_provider.
        "@org_tensorflow//tensorflow:ios": [],
        "//conditions:default": [
            "//tflite/tools/delegates:gpu_delegate_provider",
        ],
    }),
)

cc_binary(
    name = "culprit_finder_main",
    srcs = ["culprit_finder_main.cc"],
    deps = [
        ":culprit_finder_lib",
        "//litert/c/internal:litert_logging",
    ],
)
