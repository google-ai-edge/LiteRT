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
    name = "ml_drift_cl_model_runner",
    hdrs = ["ml_drift_cl_model_runner.h"],
    deps = [
        ":ml_drift_model_runner_base",
        "//tensor:buffer",
        "//tensor/backends/ml_drift:ml_drift_buffer",
        "//tensor/runners/litert:litert_buffer",
        "//tensor/utils:macros",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/cl:environment",
        "@ml_drift//ml_drift/cl:inference_context",
        "@ml_drift//ml_drift/cl:opencl_wrapper",
        "@ml_drift//ml_drift/cl:tensor",
        "@ml_drift//ml_drift/cl:util",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/task:profiling_info",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "ml_drift_dummy_weight_creator",
    srcs = ["ml_drift_dummy_weight_creator.cc"],
    hdrs = ["ml_drift_dummy_weight_creator.h"],
    deps = [
        "//tensor",
        "//tensor:datatypes",
        "//tensor/backends/ml_drift:arithmetic_ml_drift",
        "//tensor/backends/ml_drift:ml_drift_helpers",
        "//tensor/internal:graph",
        "//tensor/utils:macros",
        "@XNNPACK",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_conversion",
        "@ml_drift//ml_drift/common/task:weights_layout",
    ],
)

cc_test(
    name = "ml_drift_cl_model_runner_test",
    srcs = ["ml_drift_cl_model_runner_test.cc"],
    tags = ["requires-gpu-sm90"],  # OpenCL/MIG-compatible
    deps = [
        ":ml_drift_cl_model_runner",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:datatypes",
        "//tensor/backends/ml_drift:arithmetic_ml_drift",
        "//tensor/utils:matchers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:precision",
    ],
)

cc_library(
    name = "ml_drift_model_runner_base",
    hdrs = ["ml_drift_model_runner_base.h"],
    deps = [
        ":ml_drift_dummy_weight_creator",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:datatypes",
        "//tensor/backends/ml_drift:arithmetic_ml_drift",
        "//tensor/backends/ml_drift:ml_drift_conversion",
        "//tensor/backends/ml_drift:ml_drift_helpers",
        "//tensor/internal:graph",
        "//tensor/internal:graph_probe",
        "//tensor/internal:graph_traversal",
        "//tensor/utils:macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "ml_drift_webgpu_model_runner",
    hdrs = ["ml_drift_webgpu_model_runner.h"],
    deps = [
        ":ml_drift_model_runner_base",
        "//tensor:buffer",
        "//tensor/runners/litert:litert_buffer",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/webgpu:environment",
        "@ml_drift//ml_drift/webgpu:execution_environment",
        "@ml_drift//ml_drift/webgpu:inference_context",
        "@ml_drift//ml_drift/webgpu:spatial_tensor",
        "@ml_drift//ml_drift/webgpu:webgpu_api_util",
        "@ml_drift//ml_drift/webgpu:webgpu_headers",
    ],
)

cc_library(
    name = "compiled_ml_drift_model_runner",
    hdrs = ["compiled_ml_drift_model_runner.h"],
    deps = [
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/ml_drift:ml_drift_conversion",
        "//tensor/backends/ml_drift:ml_drift_helpers",
        "//tensor/internal:graph",
        "//tensor/internal:graph_traversal",
        "//tensor/utils:macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:model_hints",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common:util",
        "@ml_drift//ml_drift/common/task:profiling_info",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_conversion",
        "@ml_drift//ml_drift/common/task:weights_layout",
    ],
)

cc_test(
    name = "compiled_ml_drift_model_runner_test",
    srcs = ["compiled_ml_drift_model_runner_test.cc"],
    tags = ["requires-gpu-sm90"],  # OpenCL/MIG-compatible
    deps = [
        ":compiled_ml_drift_model_runner",
        ":ml_drift_cl_model_runner",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:datatypes",
        "//tensor/backends/ml_drift:arithmetic_ml_drift",
        "//tensor/internal:compile",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ml_drift_model_stage",
    hdrs = ["ml_drift_model_stage.h"],
    deps = [
        ":compiled_ml_drift_model_runner",
        ":ml_drift_cl_model_runner",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_tensor_buffer_types",
        "//tensor",
        "//tensor:datatypes",
        "//tensor/runners:model_chain",
        "//tensor/runners/litert:litert_buffer",
        "//tensor/utils:macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/cl:environment",
        "@ml_drift//ml_drift/common:precision",
    ],
)

cc_test(
    name = "ml_drift_model_stage_test",
    srcs = ["ml_drift_model_stage_test.cc"],
    data = [
        "//litert/test:tflite_test_data",
    ],
    tags = ["requires-gpu-sm90"],  # OpenCL/MIG-compatible
    deps = [
        ":compiled_ml_drift_model_runner",
        ":ml_drift_cl_model_runner",
        ":ml_drift_model_stage",
        "//litert/cc:litert_common",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/test:common",
        "//litert/test:simple_model",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:datatypes",
        "//tensor/backends/ml_drift:arithmetic_ml_drift",
        "//tensor/runners:model_chain",
        "//tensor/runners/litert:litert_buffer",
        "//tensor/utils:matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:precision",
    ],
)
