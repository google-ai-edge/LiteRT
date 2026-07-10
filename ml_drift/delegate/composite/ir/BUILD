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
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
        "//third_party/odml/litert/ml_drift/delegate:__subpackages__",
    ],
)

cc_library(
    name = "add_values_to_cache_parser",
    srcs = ["add_values_to_cache_parser.cc"],
    hdrs = ["add_values_to_cache_parser.h"],
    deps = [
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/delegate/composite:add_values_to_cache_parser",
        "//third_party/odml/litert/ml_drift/tflite:custom_ir_operation_parser",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)

cc_library(
    name = "moe_experts_parser",
    srcs = ["moe_experts_parser.cc"],
    hdrs = ["moe_experts_parser.h"],
    deps = [
        "//tflite:builtin_ops",
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/tflite:custom_ir_operation_parser",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "//third_party/odml/litert/ml_drift/tflite:model_builder_helper",
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
        "//tflite/c:common",
        "//tflite/kernels:kernel_util",
        "//third_party/odml/litert/ml_drift/delegate/composite:add_values_to_cache_parser",
        "//third_party/odml/litert/ml_drift/delegate/composite:runtime_batched_matmul_parser",
        "//third_party/odml/litert/ml_drift/tflite:custom_ir_operation_parser",
        "//third_party/odml/litert/ml_drift/tflite:ir_model_builder_helper",
        "//third_party/odml/litert/ml_drift/tflite:model_builder_helper",
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
    name = "custom_parsers",
    srcs = ["custom_parsers.cc"],
    hdrs = ["custom_parsers.h"],
    deps = [
        ":add_values_to_cache_parser",
        ":moe_experts_parser",
        ":runtime_batched_matmul_parser",
        "//tflite/c:common",
        "//third_party/odml/litert/ml_drift/tflite:custom_ir_operation_parser",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "litert_op_selector",
    srcs = ["litert_op_selector.cc"],
    hdrs = ["litert_op_selector.h"],
    deps = [
        "//third_party/odml/litert/ml_drift/delegate/composite:add_values_to_cache_kernel",
        "//third_party/odml/litert/ml_drift/delegate/composite:moe_experts_kernel",
        "//third_party/odml/litert/ml_drift/delegate/composite:runtime_batched_matmul_kernel",
        "@com_google_absl//absl/status",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:ir_model",
        "@ml_drift//ml_drift/common:ir_model_util",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common/selectors:operation_selector",
        "@ml_drift//ml_drift/common/task:gpu_operation",
    ],
)

cc_test(
    name = "add_values_to_cache_parser_test",
    srcs = ["add_values_to_cache_parser_test.cc"],
    deps = [
        ":add_values_to_cache_parser",
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/delegate/composite:add_values_to_cache_parser",
        "//third_party/odml/litert/ml_drift/tflite:custom_ir_operation_parser",
        "//third_party/odml/litert/ml_drift/tflite/convert:convert_testing_utils",
        "//third_party/odml/litert/ml_drift/tflite/convert:stub_delegate",
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
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/tflite:custom_ir_operation_parser",
        "//third_party/odml/litert/ml_drift/tflite/convert:convert_testing_utils",
        "//third_party/odml/litert/ml_drift/tflite/convert:stub_delegate",
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
        "//tflite:builtin_ops",
        "//tflite:framework_stable",
        "//tflite/c:common",
        "//tflite/core/c:private_common",
        "//third_party/odml/litert/ml_drift/delegate/composite:runtime_batched_matmul_parser",
        "//third_party/odml/litert/ml_drift/tflite:custom_ir_operation_parser",
        "//third_party/odml/litert/ml_drift/tflite/convert:convert_testing_utils",
        "//third_party/odml/litert/ml_drift/tflite/convert:stub_delegate",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:ir_model",
    ],
)
