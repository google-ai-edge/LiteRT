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
# copybara:uncomment load("@rules_cc//cc:cc_test.bzl", "cc_test")

# Basic build rules for the transformer ops library.
package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "transformer_ops",
    hdrs = ["transformer_ops.h"],
    deps = [
        ":transformer_ops_graph",
        "//tensor",
        "//tensor:datatypes",
        "//tensor/internal:arithmetic_helpers",
        "//tensor/internal:graph",
        "//tensor/utils:macros",
        "//tensor/utils:source_location",
    ],
)

cc_library(
    name = "transformer_ops_graph",
    hdrs = ["transformer_ops_graph.h"],
    deps = [
        "//tensor/internal:graph",
        "//tensor/internal:mixin",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# copybara:uncomment_begin(google-only)
#
# cc_library(
#     name = "transformer_ops_ml_drift",
#     srcs = ["transformer_ops_ml_drift.cc"],
#     hdrs = ["transformer_ops_ml_drift.h"],
#     deps = [
#         ":transformer_ops_graph",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/strings:string_view",
#         "@ml_drift//ml_drift/common:data_type",
#         "@ml_drift//ml_drift/common:gpu_model_builder",
#         "@ml_drift//ml_drift/common:operations",
#         "@ml_drift//ml_drift/common:shape",
#         "@ml_drift//ml_drift/common:tensor",
#         "@ml_drift//ml_drift/common:types",
#         "@ml_drift//ml_drift/common/task:buffer_desc",
#         "@ml_drift//ml_drift/common/task:gpu_object_desc",
#         "@ml_drift//ml_drift/common/task:gpu_operation",
#         "@ml_drift//ml_drift/common/task:gpu_tensor",
#         "@ml_drift//ml_drift/common/task:tensor_desc",
#         "//tensor:buffer",
#         "//tensor:datatypes",
#         "//tensor/backends/ml_drift:arithmetic_ml_drift",
#         "//tensor/backends/ml_drift:ml_drift_helpers",
#         "//tensor/internal:graph",
#         "//tensor/internal:mixin",
#         "//tensor/utils:macros",
#     ],
# )
#
# cc_library(
#     name = "transformer_ops_xnnpack",
#     srcs = ["transformer_ops_xnnpack.cc"],
#     hdrs = ["transformer_ops_xnnpack.h"],
#     deps = [
#         ":transformer_ops_graph",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/strings:str_format",
#         "@com_google_absl//absl/strings:string_view",
#         "//tensor:buffer",
#         "//tensor:datatypes",
#         "//tensor/backends/xnnpack:arithmetic",
#         "//tensor/internal:graph",
#         "//tensor/internal:mixin",
#         "//tensor/utils:macros",
#     ],
# )
#
# cc_test(
#     name = "transformer_ops_numerical_test",
#     srcs = ["transformer_ops_numerical_test.cc"],
#     tags = [
#         "requires-gpu-nvidia",
#     ],
#     deps = [
#         ":transformer_ops",
#         ":transformer_ops_ml_drift",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/types:any",
#         "@ml_drift//ml_drift/cl:environment",
#         "@ml_drift//ml_drift/cl:inference_context",
#         "@ml_drift//ml_drift/cl:opencl_wrapper",
#         "@ml_drift//ml_drift/common:data_type",
#         "@ml_drift//ml_drift/common:gpu_model",
#         "@ml_drift//ml_drift/common:gpu_model_builder",
#         "@ml_drift//ml_drift/common:model",
#         "@ml_drift//ml_drift/common:precision",
#         "@ml_drift//ml_drift/common:shape",
#         "@ml_drift//ml_drift/common:tensor",
#         "@ml_drift//ml_drift/common/task:tensor_desc",
#         "//tensor",
#         "//tensor:datatypes",
#         "//tensor/backends/ml_drift:arithmetic_ml_drift",
#         "//tensor/backends/ml_drift:ml_drift_conversion",
#         "//tensor/backends/ml_drift:ml_drift_helpers",
#         "//tensor/runners/ml_drift:ml_drift_dummy_weight_creator",
#     ],
# )
#
# copybara:uncomment_end
