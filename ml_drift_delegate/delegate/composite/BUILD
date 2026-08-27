# Copyright 2026 The ML Drift Authors.
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

load("@ml_drift//ml_drift/common/kernels/tests:mld_kernel_test.bzl", "mld_kernel_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc:objc_library.bzl", "objc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        "//ml_drift_delegate/delegate:__subpackages__",
        "//ml_drift_delegate/tflite:__subpackages__",
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
)

cc_library(
    name = "add_values_to_cache_parser",
    srcs = ["add_values_to_cache_parser.cc"],
    hdrs = ["add_values_to_cache_parser.h"],
    deps = [
        "//ml_drift_delegate/tflite:model_builder_helper",
        "//ml_drift_delegate/tflite:object_reader",
        "//ml_drift_delegate/tflite:operation_parser",
        "//tflite/c:common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "add_values_to_cache_kernel",
    srcs = ["add_values_to_cache_kernel.cc"],
    hdrs = ["add_values_to_cache_kernel.h"],
    deps = [
        ":add_values_to_cache_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:buffer_desc",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "runtime_batched_matmul_parser",
    srcs = ["runtime_batched_matmul_parser.cc"],
    hdrs = ["runtime_batched_matmul_parser.h"],
    deps = [
        ":add_values_to_cache_parser",
        "//ml_drift_delegate/tflite:model_builder_helper",
        "//ml_drift_delegate/tflite:object_reader",
        "//ml_drift_delegate/tflite:operation_parser",
        "//tflite/c:common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/kernels:fully_connected",
        "@ml_drift//ml_drift/common/task:weights_layout",
    ],
)

cc_library(
    name = "runtime_batched_matmul_kernel",
    srcs = ["runtime_batched_matmul_kernel.cc"],
    hdrs = ["runtime_batched_matmul_kernel.h"],
    deps = [
        ":runtime_batched_matmul_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_layout",
    ],
)

cc_library(
    name = "moe_experts_parser",
    srcs = ["moe_experts_parser.cc"],
    hdrs = ["moe_experts_parser.h"],
    deps = [
        "//ml_drift_delegate/tflite:model_builder_helper",
        "//ml_drift_delegate/tflite:object_reader",
        "//ml_drift_delegate/tflite:operation_parser",
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common:tensor",
    ],
)

cc_library(
    name = "experts_remap_builder",
    srcs = ["experts_remap_builder.cc"],
    hdrs = ["experts_remap_builder.h"],
    deps = [
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:util",
        "@ml_drift//ml_drift/common/kernels:conv_apple_mpp",
        "@ml_drift//ml_drift/common/kernels:conv_generic",
        "@ml_drift//ml_drift/common/kernels:conv_wave_matrix",
        "@ml_drift//ml_drift/common/kernels:conv_wave_memory",
        "@ml_drift//ml_drift/common/kernels:fully_connected",
        "@ml_drift//ml_drift/common/kernels/google/custom:experts_remap",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "moe_experts_kernel",
    srcs = ["moe_experts_kernel.cc"],
    hdrs = ["moe_experts_kernel.h"],
    deps = [
        ":experts_remap_builder",
        ":moe_experts_parser",
        "//ml_drift_delegate/delegate/composite/ir:moe_experts_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/kernels:fully_connected",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_layout",
    ],
)

cc_library(
    name = "rope_parser",
    srcs = ["rope_parser.cc"],
    hdrs = ["rope_parser.h"],
    deps = [
        "//ml_drift_delegate/tflite:model_builder_helper",
        "//ml_drift_delegate/tflite:object_reader",
        "//ml_drift_delegate/tflite:operation_parser",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "sdpa_transposed_parser",
    srcs = ["sdpa_transposed_parser.cc"],
    hdrs = ["sdpa_transposed_parser.h"],
    deps = [
        ":runtime_batched_matmul_parser",
        "//ml_drift_delegate/tflite:model_builder_helper",
        "//ml_drift_delegate/tflite:object_reader",
        "//ml_drift_delegate/tflite:operation_parser",
        "//tflite/c:common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/kernels:fully_connected",
        "@ml_drift//ml_drift/common/task:weights_layout",
    ],
)

cc_library(
    name = "sdpa_transposed_kernel",
    srcs = ["sdpa_transposed_kernel.cc"],
    hdrs = ["sdpa_transposed_kernel.h"],
    deps = [
        ":sdpa_transposed_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:kernel_info",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:tuning_type",
        "@ml_drift//ml_drift/common/task:weights_layout",
    ],
)

mld_kernel_test(
    name = "sdpa_transposed_kernel_test",
    srcs = ["sdpa_transposed_kernel_test.cc"],
    platforms = [
        "opencl",
        "webgpu",
    ],
    deps = [
        ":sdpa_transposed_kernel",
        ":sdpa_transposed_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/kernels:fully_connected",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:testing_util",
        "@ml_drift//ml_drift/common/task:weights_layout",
    ],
)

# The Mac test targets are defined separately from mld_kernel_test to tag the
# objc_library and cc_test with "manual", "notap", and "nobuilder". This prevents
# non-Apple presubmits (TAP on Linux) from failing during target analysis due to
# objc_library not being supported on non-Apple platforms, while still allowing
# explicit test execution with:
# bazel test -c opt --config=darwin_arm64 //third_party/odml/litert/ml_drift/delegate/composite:sdpa_transposed_kernel_test_mac
objc_library(
    name = "sdpa_transposed_kernel_test_objc_lib",
    testonly = True,
    srcs = [
        "sdpa_transposed_kernel_test.cc",
        "@ml_drift//ml_drift/common/kernels/tests:kernel_test.h",
        "@ml_drift//ml_drift/common/kernels/tests:metal_kernel_test.mm",
    ],
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":sdpa_transposed_kernel",
        ":sdpa_transposed_parser",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        # Dependency placeholder: apple_frameworks Metal,
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/kernels:fully_connected",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:testing_util",
        "@ml_drift//ml_drift/common/task:weights_layout",
        "@ml_drift//ml_drift/metal/testing:test_util",
    ],
)

cc_test(
    name = "sdpa_transposed_kernel_test_mac",
    tags = [
        "mac",
        "manual",
        "nobuilder",
        "notap",
        "nozapfhahn",
    ],
    deps = [
        ":sdpa_transposed_kernel_test_objc_lib",
    ],
)

cc_library(
    name = "swiglu_parser",
    srcs = ["swiglu_parser.cc"],
    hdrs = ["swiglu_parser.h"],
    deps = [
        "//ml_drift_delegate/tflite:model_builder_helper",
        "//ml_drift_delegate/tflite:object_reader",
        "//ml_drift_delegate/tflite:operation_parser",
        "//tflite/c:common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:status",
    ],
)

cc_library(
    name = "swiglu_kernel",
    srcs = ["swiglu_kernel.cc"],
    hdrs = ["swiglu_kernel.h"],
    deps = [
        ":swiglu_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "qkv_norm_rope_parser",
    srcs = ["qkv_norm_rope_parser.cc"],
    hdrs = ["qkv_norm_rope_parser.h"],
    deps = [
        "//ml_drift_delegate/tflite:object_reader",
        "//ml_drift_delegate/tflite:operation_parser",
        "//tflite/c:common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:model",
    ],
)

alias(
    name = "qwen3_qkv_norm_rope_parser",
    actual = ":qkv_norm_rope_parser",
)

cc_library(
    name = "qkv_norm_rope_kernel",
    srcs = ["qkv_norm_rope_kernel.cc"],
    hdrs = ["qkv_norm_rope_kernel.h"],
    deps = [
        ":qkv_norm_rope_parser",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:kernel_info",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:tuning_type",
    ],
)

alias(
    name = "qwen3_qkv_norm_rope_kernel",
    actual = ":qkv_norm_rope_kernel",
)

cc_library(
    name = "custom_parsers",
    srcs = ["custom_parsers.cc"],
    hdrs = [
        "custom_parsers.h",
        "//ml_drift_delegate/tflite:custom_parsers.h",
    ],
    deps = [
        ":add_values_to_cache_parser",
        ":moe_experts_parser",
        ":qkv_norm_rope_parser",
        ":rope_parser",
        ":runtime_batched_matmul_parser",
        ":sdpa_transposed_parser",
        ":swiglu_parser",
        "//ml_drift_delegate/tflite:model_builder",
        "//ml_drift_delegate/tflite:operation_parser",
        "//ml_drift_delegate/tflite:unimplemented_operation_parser",
        "@com_google_absl//absl/strings",
    ],
)

mld_kernel_test(
    name = "add_values_to_cache_kernel_test",
    srcs = ["add_values_to_cache_kernel_test.cc"],
    platforms = [
        "opencl",
        "webgpu",
    ],
    deps = [
        ":add_values_to_cache_kernel",
        ":add_values_to_cache_parser",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:testing_util",
    ],
)

cc_library(
    name = "litert_op_selector",
    srcs = ["litert_op_selector.cc"],
    hdrs = ["litert_op_selector.h"],
    deps = [
        ":add_values_to_cache_kernel",
        ":add_values_to_cache_parser",
        ":moe_experts_kernel",
        ":moe_experts_parser",
        ":qkv_norm_rope_kernel",
        ":qkv_norm_rope_parser",
        ":runtime_batched_matmul_kernel",
        ":runtime_batched_matmul_parser",
        ":sdpa_transposed_kernel",
        ":sdpa_transposed_parser",
        ":swiglu_kernel",
        ":swiglu_parser",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:gpu_model_util",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common/selectors:operation_selector",
        "@ml_drift//ml_drift/common/selectors:special_selector",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)
