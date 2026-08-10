# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "convolution_selector",
    srcs = ["convolution_selector.cc"],
    deps = [
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:model_hints",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/task:weights_layout",
        "//tflite/delegates/gpu/common/tasks:conv_constants",
        "//tflite/delegates/gpu/common/tasks:conv_generic",
        "//tflite/delegates/gpu/common/tasks:conv_metal_simd",
        "//tflite/delegates/gpu/common/tasks:conv_weights_converter",
    ],
)

cc_library(
    name = "convolution_transposed_selector",
    srcs = ["convolution_transposed_selector.cc"],
    deps = [
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/task:weights_layout",
        "//tflite/delegates/gpu/common/tasks:convolution_transposed",
        "//tflite/delegates/gpu/common/tasks:convolution_transposed_3x3",
        "//tflite/delegates/gpu/common/tasks:convolution_transposed_3x3_thin",
        "//tflite/delegates/gpu/common/tasks:convolution_transposed_4x4",
        "//tflite/delegates/gpu/common/tasks:convolution_transposed_thin",
    ],
)

cc_library(
    name = "default_selector",
    srcs = ["default_selector.cc"],
    deps = [
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_hints",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common/selectors:subgraph",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dw_convolution_selector",
    srcs = ["dw_convolution_selector.cc"],
    deps = [
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/task:tensor_desc",
        "//tflite/delegates/gpu/common/tasks:depthwise_conv",
        "//tflite/delegates/gpu/common/tasks:depthwise_conv_3x3",
        "//tflite/delegates/gpu/common/tasks:depthwise_conv_3x3_stride_h2",
    ],
)

cc_library(
    name = "fully_connected_selector",
    srcs = ["fully_connected_selector.cc"],
    deps = [
        "//tflite/delegates/gpu/common:gpu_info",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/tasks:conv_generic",
        "//tflite/delegates/gpu/common/tasks:fully_connected",
    ],
)
