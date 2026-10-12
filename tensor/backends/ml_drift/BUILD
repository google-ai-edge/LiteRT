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
    name = "ml_drift_buffer",
    srcs = ["ml_drift_buffer.cc"],
    hdrs = ["ml_drift_buffer.h"],
    deps = [
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/internal:type_id",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/cl:cl_command_queue",
        "@ml_drift//ml_drift/cl:cl_context",
        "@ml_drift//ml_drift/cl:cl_memory",
        "@ml_drift//ml_drift/cl:inference_context",
        "@ml_drift//ml_drift/cl:opencl_wrapper",
        "@ml_drift//ml_drift/cl:tensor",
        "@ml_drift//ml_drift/cl:util",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@opencl_headers",
    ],
)

cc_library(
    name = "arithmetic_ml_drift",
    srcs = ["arithmetic_ml_drift.cc"],
    hdrs = ["arithmetic_ml_drift.h"],
    deps = [
        ":ml_drift_helpers",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:arithmetic_graph",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/internal:graph",
        "//tensor/internal:mixin",
        "//tensor/internal:type_id",
        "//tensor/utils:macros",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/kernels:elementwise",
        "@ml_drift//ml_drift/common/kernels:embedding_lookup",
        "@ml_drift//ml_drift/common/kernels:fully_connected",
        "@ml_drift//ml_drift/common/kernels:one_hot",
        "@ml_drift//ml_drift/common/kernels:prelu",
        "@ml_drift//ml_drift/common/kernels:reduce",
        "@ml_drift//ml_drift/common/selectors:simple_selectors",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_layout",
    ],
)

cc_library(
    name = "trace_ml_drift_runner",
    hdrs = ["trace_ml_drift_runner.h"],
    deps = [
        ":arithmetic_ml_drift",
        ":ml_drift_buffer",
        ":ml_drift_conversion",
        "//tensor",
        "//tensor:buffer",
        "//tensor/internal:graph",
        "//tensor/runners/ml_drift:ml_drift_cl_model_runner",
        "//tensor/runners/ml_drift:ml_drift_model_runner_base",
        "//tensor/utils:macros",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "ml_drift_conversion",
    srcs = ["ml_drift_conversion.cc"],
    hdrs = ["ml_drift_conversion.h"],
    deps = [
        ":arithmetic_ml_drift",
        ":ml_drift_helpers",
        "//tensor",
        "//tensor/internal:graph",
        "//tensor/internal:graph_traversal",
        "//tensor/utils:macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:gpu_model_builder",
    ],
)

cc_test(
    name = "ml_drift_conversion_test",
    srcs = ["ml_drift_conversion_test.cc"],
    tags = [
        "noasan",
        "requires-gpu-sm90",  # OpenCL/MIG-compatible
    ],
    deps = [
        ":arithmetic_ml_drift",
        ":ml_drift_buffer",
        ":ml_drift_conversion",
        ":ml_drift_helpers",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/internal:graph",
        "//tensor/runners/ml_drift:ml_drift_dummy_weight_creator",
        "//tensor/utils:macros",
        "//tensor/utils:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/cl:cl_test",
        "@ml_drift//ml_drift/cl:inference_context",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:model_hints",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_conversion",
    ],
)

cc_test(
    name = "ml_drift_conversion_numerical_test",
    srcs = ["ml_drift_conversion_numerical_test.cc"],
    tags = [
        "noasan",
        "requires-gpu-sm90",  # OpenCL/MIG-compatible
    ],
    deps = [
        ":arithmetic_ml_drift",
        ":ml_drift_buffer",
        ":ml_drift_conversion",
        ":ml_drift_helpers",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/testing:numerical_test_suite",
        "//tensor/internal:graph",
        "//tensor/internal:graph_traversal",
        "//tensor/internal:type_id",
        "//tensor/runners/ml_drift:ml_drift_dummy_weight_creator",
        "//tensor/utils:macros",
        "//tensor/utils:matchers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/cl:environment",
        "@ml_drift//ml_drift/cl:inference_context",
        "@ml_drift//ml_drift/cl:opencl_wrapper",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_test(
    name = "trace_ml_drift_runner_test",
    srcs = ["trace_ml_drift_runner_test.cc"],
    tags = [
        "noasan",
        "requires-gpu-sm90",  # OpenCL/MIG-compatible
    ],
    deps = [
        ":arithmetic_ml_drift",
        ":trace_ml_drift_runner",
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/internal:compile",
        "//tensor/utils:matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:precision",
    ],
)

cc_library(
    name = "ml_drift_helpers",
    srcs = ["ml_drift_helpers.cc"],
    hdrs = ["ml_drift_helpers.h"],
    deps = [
        ":ml_drift_buffer",
        "//tensor:datatypes",
        "//tensor/internal:graph",
        "//tensor/utils:macros",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)
