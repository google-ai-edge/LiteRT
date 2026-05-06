# Copyright (C) 2026 Samsung Electronics Co. LTD.
# SPDX-License-Identifier: Apache-2.0
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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/samsung/compiler:__subpackages__"],
)

cc_library(
    name = "op_wrapper",
    hdrs = ["op_wrapper.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "@exynos_ai_litecore//:ai_litecore_headers",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "batch_matmul_op_builder",
    srcs = ["batch_matmul_op_builder.cc"],
    hdrs = ["batch_matmul_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "cast_op_builder",
    srcs = ["cast_op_builder.cc"],
    hdrs = ["cast_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "concat_op_builder",
    srcs = ["concat_op_builder.cc"],
    hdrs = ["concat_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "conv2d_op_builder",
    srcs = ["conv2d_op_builder.cc"],
    hdrs = ["conv2d_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
        "//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "elementwise_op_builder",
    srcs = ["elementwise_op_builder.cc"],
    hdrs = ["elementwise_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "fully_connected_op_builder",
    srcs = ["fully_connected_op_builder.cc"],
    hdrs = ["fully_connected_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "logistic_op_builder",
    srcs = ["logistic_op_builder.cc"],
    hdrs = ["logistic_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
        "//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "pad_op_builder",
    srcs = ["pad_op_builder.cc"],
    hdrs = ["pad_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
        "//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "pool2d_op_builder",
    srcs = ["pool2d_op_builder.cc"],
    hdrs = ["pool2d_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
        "//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "reduce_op_builder",
    srcs = ["reduce_op_builder.cc"],
    hdrs = ["reduce_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "relu_op_builder",
    srcs = ["relu_op_builder.cc"],
    hdrs = ["relu_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "relu1_op_builder",
    srcs = ["relu1_op_builder.cc"],
    hdrs = ["relu1_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "relu6_op_builder",
    srcs = ["relu6_op_builder.cc"],
    hdrs = ["relu6_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "reshape_op_builder",
    srcs = ["reshape_op_builder.cc"],
    hdrs = ["reshape_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "rms_norm_op_builder",
    srcs = ["rms_norm_op_builder.cc"],
    hdrs = ["rms_norm_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_op_options",
    ],
)

cc_library(
    name = "softmax_op_builder",
    srcs = ["softmax_op_builder.cc"],
    hdrs = ["softmax_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "transpose_conv_op_builder",
    srcs = ["transpose_conv_op_builder.cc"],
    hdrs = ["transpose_conv_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
        "//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "transpose_op_builder",
    srcs = ["transpose_op_builder.cc"],
    hdrs = ["transpose_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "gathernd_op_builder",
    srcs = ["gathernd_op_builder.cc"],
    hdrs = ["gathernd_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "slice_op_builder",
    srcs = ["slice_op_builder.cc"],
    hdrs = ["slice_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "spacetodepth_op_builder",
    srcs = ["spacetodepth_op_builder.cc"],
    hdrs = ["spacetodepth_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "hardswish_op_builder",
    srcs = ["hardswish_op_builder.cc"],
    hdrs = ["hardswish_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "strided_slice_op_builder",
    srcs = ["strided_slice_op_builder.cc"],
    hdrs = ["strided_slice_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "split_op_builder",
    srcs = ["split_op_builder.cc"],
    hdrs = ["split_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "gather_op_builder",
    srcs = ["gather_op_builder.cc"],
    hdrs = ["gather_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "leakyrelu_op_builder",
    srcs = ["leakyrelu_op_builder.cc"],
    hdrs = ["leakyrelu_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "gelu_op_builder",
    srcs = ["gelu_op_builder.cc"],
    hdrs = ["gelu_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "tanh_op_builder",
    srcs = ["tanh_op_builder.cc"],
    hdrs = ["tanh_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "resizebilinear_op_builder",
    srcs = ["resizebilinear_op_builder.cc"],
    hdrs = ["resizebilinear_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "cumsum_op_builder",
    srcs = ["cumsum_op_builder.cc"],
    hdrs = ["cumsum_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "select_op_builder",
    srcs = ["select_op_builder.cc"],
    hdrs = ["select_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "selectv2_op_builder",
    srcs = ["selectv2_op_builder.cc"],
    hdrs = ["selectv2_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "argmax_op_builder",
    srcs = ["argmax_op_builder.cc"],
    hdrs = ["argmax_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "resizenearestneighbor_op_builder",
    srcs = ["resizenearestneighbor_op_builder.cc"],
    hdrs = ["resizenearestneighbor_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model",
    ],
)

cc_library(
    name = "l2normalization_op_builder",
    srcs = ["l2normalization_op_builder.cc"],
    hdrs = ["l2normalization_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "conv3d_op_builder",
    srcs = ["conv3d_op_builder.cc"],
    hdrs = ["conv3d_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "relun1to1_op_builder",
    srcs = ["relun1to1_op_builder.cc"],
    hdrs = ["relun1to1_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "broadcastto_op_builder",
    srcs = ["broadcastto_op_builder.cc"],
    hdrs = ["broadcastto_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "quantize_op_builder",
    srcs = ["quantize_op_builder.cc"],
    hdrs = ["quantize_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "dequantize_op_builder",
    srcs = ["dequantize_op_builder.cc"],
    hdrs = ["dequantize_op_builder.h"],
    deps = [
        ":op_wrapper",
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)

cc_library(
    name = "reversev2_op_builder",
    srcs = ["reversev2_op_builder.cc"],
    hdrs = ["reversev2_op_builder.h"],
    deps = [
        ":op_wrapper",
        ":utils",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
    ],
)
