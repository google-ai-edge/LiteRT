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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        "//ml_drift_delegate/delegate:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
)

cc_library(
    name = "add_values_to_cache_parser",
    srcs = ["add_values_to_cache_parser.cc"],
    hdrs = ["add_values_to_cache_parser.h"],
    deps = [
        "//ml_drift_delegate/delegate/composite:add_values_to_cache_parser",
        "//ml_drift_delegate/tflite:custom_ir_operation_parser",
        "//ml_drift_delegate/tflite:ir_model_builder_helper",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_library(
    name = "moe_experts_parser",
    srcs = ["moe_experts_parser.cc"],
    hdrs = ["moe_experts_parser.h"],
    deps = [
        "//ml_drift_delegate/tflite:custom_ir_operation_parser",
        "//ml_drift_delegate/tflite:ir_model_builder_helper",
        "//ml_drift_delegate/tflite:model_builder_helper",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "runtime_batched_matmul_parser",
    srcs = ["runtime_batched_matmul_parser.cc"],
    hdrs = ["runtime_batched_matmul_parser.h"],
    deps = [
        "//ml_drift_delegate/delegate/composite:add_values_to_cache_parser",
        "//ml_drift_delegate/delegate/composite:runtime_batched_matmul_parser",
        "//ml_drift_delegate/tflite:custom_ir_operation_parser",
        "//ml_drift_delegate/tflite:ir_model_builder_helper",
        "//ml_drift_delegate/tflite:model_builder_helper",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/kernels:fully_connected",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_layout",
    ],
)

cc_library(
    name = "sdpa_transposed_parser",
    srcs = ["sdpa_transposed_parser.cc"],
    hdrs = ["sdpa_transposed_parser.h"],
    deps = [
        "//ml_drift_delegate/delegate/composite:sdpa_transposed_parser",
        "//ml_drift_delegate/tflite:custom_ir_operation_parser",
        "//ml_drift_delegate/tflite:ir_model_builder_helper",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/kernels:fully_connected",
        "@ml_drift//ml_drift/common/task:weights_layout",
    ],
)

cc_library(
    name = "swiglu_parser",
    srcs = ["swiglu_parser.cc"],
    hdrs = ["swiglu_parser.h"],
    deps = [
        "//ml_drift_delegate/delegate/composite:swiglu_parser",
        "//ml_drift_delegate/tflite:custom_ir_operation_parser",
        "//ml_drift_delegate/tflite:ir_model_builder_helper",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_library(
    name = "qkv_norm_rope_parser",
    srcs = ["qkv_norm_rope_parser.cc"],
    hdrs = ["qkv_norm_rope_parser.h"],
    deps = [
        "//ml_drift_delegate/delegate/composite:qkv_norm_rope_parser",
        "//ml_drift_delegate/tflite:custom_ir_operation_parser",
        "//ml_drift_delegate/tflite:ir_model_builder_helper",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_library(
    name = "custom_parsers",
    srcs = ["custom_parsers.cc"],
    hdrs = ["custom_parsers.h"],
    deps = [
        ":add_values_to_cache_parser",
        ":moe_experts_parser",
        ":qkv_norm_rope_parser",
        ":runtime_batched_matmul_parser",
        ":sdpa_transposed_parser",
        ":swiglu_parser",
        "//ml_drift_delegate/tflite:custom_ir_operation_parser",
        "//tflite/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "litert_op_selector",
    srcs = ["litert_op_selector.cc"],
    hdrs = ["litert_op_selector.h"],
    deps = [
        "//ml_drift_delegate/delegate/composite:add_values_to_cache_kernel",
        "//ml_drift_delegate/delegate/composite:moe_experts_kernel",
        "//ml_drift_delegate/delegate/composite:qkv_norm_rope_kernel",
        "//ml_drift_delegate/delegate/composite:runtime_batched_matmul_kernel",
        "//ml_drift_delegate/delegate/composite:sdpa_transposed_kernel",
        "//ml_drift_delegate/delegate/composite:swiglu_kernel",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:ir_model_util",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common/selectors:operation_selector",
        "@ml_drift//ml_drift/common/selectors:special_selector",
        "@ml_drift//ml_drift/common/task:gpu_operation",
    ],
)

cc_test(
    name = "add_values_to_cache_parser_test",
    srcs = ["add_values_to_cache_parser_test.cc"],
    deps = [
        ":add_values_to_cache_parser",
        "//ml_drift_delegate/delegate/composite:add_values_to_cache_parser",
        "//ml_drift_delegate/tflite:custom_ir_operation_parser",
        "//ml_drift_delegate/tflite/convert:convert_testing_utils",
        "//ml_drift_delegate/tflite/convert:stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_test(
    name = "moe_experts_parser_test",
    srcs = ["moe_experts_parser_test.cc"],
    deps = [
        ":moe_experts_parser",
        "//ml_drift_delegate/tflite:custom_ir_operation_parser",
        "//ml_drift_delegate/tflite/convert:convert_testing_utils",
        "//ml_drift_delegate/tflite/convert:stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_test(
    name = "runtime_batched_matmul_parser_test",
    srcs = ["runtime_batched_matmul_parser_test.cc"],
    deps = [
        ":runtime_batched_matmul_parser",
        "//ml_drift_delegate/delegate/composite:runtime_batched_matmul_parser",
        "//ml_drift_delegate/tflite:custom_ir_operation_parser",
        "//ml_drift_delegate/tflite/convert:convert_testing_utils",
        "//ml_drift_delegate/tflite/convert:stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_test(
    name = "sdpa_transposed_parser_test",
    srcs = ["sdpa_transposed_parser_test.cc"],
    deps = [
        ":sdpa_transposed_parser",
        "//ml_drift_delegate/delegate/composite:sdpa_transposed_parser",
        "//ml_drift_delegate/tflite:custom_ir_operation_parser",
        "//ml_drift_delegate/tflite/convert:convert_testing_utils",
        "//ml_drift_delegate/tflite/convert:stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_test(
    name = "swiglu_parser_test",
    srcs = ["swiglu_parser_test.cc"],
    deps = [
        ":swiglu_parser",
        "//ml_drift_delegate/delegate/composite:swiglu_parser",
        "//ml_drift_delegate/tflite:custom_ir_operation_parser",
        "//ml_drift_delegate/tflite/convert:convert_testing_utils",
        "//ml_drift_delegate/tflite/convert:stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_test(
    name = "qkv_norm_rope_parser_test",
    srcs = ["qkv_norm_rope_parser_test.cc"],
    deps = [
        ":qkv_norm_rope_parser",
        "//ml_drift_delegate/delegate/composite:qkv_norm_rope_parser",
        "//ml_drift_delegate/tflite:custom_ir_operation_parser",
        "//ml_drift_delegate/tflite/convert:convert_testing_utils",
        "//ml_drift_delegate/tflite/convert:stub_delegate",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)
