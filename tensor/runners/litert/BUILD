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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "compiled_model_runner",
    hdrs = ["compiled_model_runner.h"],
    deps = [
        ":litert_buffer",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/tflite:arithmetic_tflite",
        "//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "//tensor/internal:graph",
        "//tensor/internal:graph_probe",
        "//tensor/internal:graph_traversal",
        "//tflite:framework",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_dynamic_runner",
    hdrs = ["litert_dynamic_runner.h"],
    deps = [
        ":litert_buffer",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/utils:macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "lambda_model_runner",
    hdrs = ["lambda_model_runner.h"],
    deps = [
        ":compiled_model_runner",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//tensor",
        "//tensor/backends/tflite:arithmetic_tflite",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_buffer",
    hdrs = ["litert_buffer.h"],
    deps = [
        "//litert/cc:litert_tensor_buffer",
        "//tensor:buffer",
        "//tensor/internal:type_id",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_test(
    name = "litert_buffer_test",
    srcs = ["litert_buffer_test.cc"],
    deps = [
        ":litert_buffer",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc:litert_tensor_buffer_types",
        "//tensor:buffer",
        "//tensor/internal:type_id",
        "//tensor/utils:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "lambda_model_runner_test",
    srcs = ["lambda_model_runner_test.cc"],
    deps = [
        ":lambda_model_runner",
        ":litert_dynamic_runner",
        "//litert/cc:litert_common",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:datatypes",
        "//tensor/backends/tflite:arithmetic_tflite",
        "//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
