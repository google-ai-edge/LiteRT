# Copyright 2024 Google LLC.
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
    default_visibility = ["//litert/vendors/mediatek/compiler:__subpackages__"],
)

cc_library(
    name = "operand_map",
    srcs = ["operand_map.cc"],
    hdrs = [
        "extra_data_mgr.h",
        "operand_map.h",
    ],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":neuron_utils",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
        "@com_google_absl//absl/container:flat_hash_map",
        "@neuro_pilot//:v8_latest_host_headers",
    ],
)

cc_library(
    name = "neuron_utils",
    srcs = ["neuron_utils.cc"],
    hdrs = [
        "neuron_utils.h",
    ],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "legalize_helper",
    hdrs = [
        "legalize_helper.h",
    ],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "@neuro_pilot//:v8_latest_host_headers",
    ],
)

cc_library(
    name = "add_op_legalization",
    srcs = ["add_op_legalization.cc"],
    hdrs = ["add_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "mul_op_legalization",
    srcs = ["mul_op_legalization.cc"],
    hdrs = ["mul_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "batch_matmul_op_legalization",
    srcs = ["batch_matmul_op_legalization.cc"],
    hdrs = ["batch_matmul_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "fully_connected_op_legalization",
    srcs = ["fully_connected_op_legalization.cc"],
    hdrs = ["fully_connected_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":legalize_helper",
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "reshape_op_legalization",
    srcs = ["reshape_op_legalization.cc"],
    hdrs = ["reshape_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "transpose_op_legalization",
    srcs = ["transpose_op_legalization.cc"],
    hdrs = ["transpose_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "rsqrt_op_legalization",
    srcs = ["rsqrt_op_legalization.cc"],
    hdrs = ["rsqrt_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "concat_op_legalization",
    srcs = ["concat_op_legalization.cc"],
    hdrs = ["concat_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "quantize_op_legalization",
    srcs = ["quantize_op_legalization.cc"],
    hdrs = ["quantize_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "common_op_legalization",
    srcs = ["common_op_legalization.cc"],
    hdrs = ["common_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":neuron_utils",
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "sub_op_legalization",
    srcs = ["sub_op_legalization.cc"],
    hdrs = ["sub_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "softmax_op_legalization",
    srcs = ["softmax_op_legalization.cc"],
    hdrs = ["softmax_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "mean_op_legalization",
    srcs = ["mean_op_legalization.cc"],
    hdrs = ["mean_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "gelu_op_legalization",
    srcs = ["gelu_op_legalization.cc"],
    hdrs = ["gelu_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "squared_difference_op_legalization",
    srcs = ["squared_difference_op_legalization.cc"],
    hdrs = ["squared_difference_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "resize_bilinear_op_legalization",
    srcs = ["resize_bilinear_op_legalization.cc"],
    hdrs = ["resize_bilinear_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "resize_nearest_neighbor_op_legalization",
    srcs = ["resize_nearest_neighbor_op_legalization.cc"],
    hdrs = ["resize_nearest_neighbor_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "transpose_conv_op_legalization",
    srcs = ["transpose_conv_op_legalization.cc"],
    hdrs = ["transpose_conv_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":legalize_helper",
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "rms_norm_op_legalization",
    srcs = ["rms_norm_op_legalization.cc"],
    hdrs = ["rms_norm_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "strided_slice_op_legalization",
    srcs = ["strided_slice_op_legalization.cc"],
    hdrs = ["strided_slice_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "split_op_legalization",
    srcs = ["split_op_legalization.cc"],
    hdrs = ["split_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        ":operand_map",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/vendors/mediatek:neuron_adapter_api",
    ],
)
