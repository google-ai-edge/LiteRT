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
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert/vendors/samsung/compiler:__subpackages__"],
)

litert_dynamic_lib(
    name = "compiler_plugin",
    srcs = ["compiler_plugin.cc"],
    hdrs = ["//litert/vendors/c:litert_compiler_plugin.h"],
    copts = [
        "-Os",
        "-fno-exceptions",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    export_litert_only = True,
    linkopts = [
        # Wheel package places the shared library in the root directory.
        "-Wl,-rpath,$$ORIGIN/../../..",
    ],
    linkstatic = 1,
    shared_lib_name = "compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_Samsung.so",
    ungrte = True,
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
    deps = [
        ":compile_model",
        ":create_model",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_logging_helper",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_context_wrapper",
        "//litert/cc/internal:litert_opaque_options_wrapper",
        "//litert/cc/internal:litert_options_wrapper",
        "//litert/cc/options:litert_samsung_options",
        "//litert/vendors/samsung:ai_litecore_api",
        "//litert/vendors/samsung:soc_model",
    ],
)

cc_library(
    name = "create_model",
    srcs = ["create_model.cc"],
    hdrs = ["create_model.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_common",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_op_options",
        "//litert/cc/internal:litert_consts",
        "//litert/vendors/samsung:ai_litecore_api",
        "//litert/vendors/samsung/compiler/builders:argmax_op_builder",
        "//litert/vendors/samsung/compiler/builders:batch_matmul_op_builder",
        "//litert/vendors/samsung/compiler/builders:broadcastto_op_builder",
        "//litert/vendors/samsung/compiler/builders:cast_op_builder",
        "//litert/vendors/samsung/compiler/builders:concat_op_builder",
        "//litert/vendors/samsung/compiler/builders:conv2d_op_builder",
        "//litert/vendors/samsung/compiler/builders:conv3d_op_builder",
        "//litert/vendors/samsung/compiler/builders:cumsum_op_builder",
        "//litert/vendors/samsung/compiler/builders:dequantize_op_builder",
        "//litert/vendors/samsung/compiler/builders:elementwise_op_builder",
        "//litert/vendors/samsung/compiler/builders:fully_connected_op_builder",
        "//litert/vendors/samsung/compiler/builders:gather_op_builder",
        "//litert/vendors/samsung/compiler/builders:gathernd_op_builder",
        "//litert/vendors/samsung/compiler/builders:gelu_op_builder",
        "//litert/vendors/samsung/compiler/builders:hardswish_op_builder",
        "//litert/vendors/samsung/compiler/builders:l2normalization_op_builder",
        "//litert/vendors/samsung/compiler/builders:leakyrelu_op_builder",
        "//litert/vendors/samsung/compiler/builders:logistic_op_builder",
        "//litert/vendors/samsung/compiler/builders:op_wrapper",
        "//litert/vendors/samsung/compiler/builders:pad_op_builder",
        "//litert/vendors/samsung/compiler/builders:pool2d_op_builder",
        "//litert/vendors/samsung/compiler/builders:quantize_op_builder",
        "//litert/vendors/samsung/compiler/builders:reduce_op_builder",
        "//litert/vendors/samsung/compiler/builders:relu1_op_builder",
        "//litert/vendors/samsung/compiler/builders:relu6_op_builder",
        "//litert/vendors/samsung/compiler/builders:relu_op_builder",
        "//litert/vendors/samsung/compiler/builders:relun1to1_op_builder",
        "//litert/vendors/samsung/compiler/builders:reshape_op_builder",
        "//litert/vendors/samsung/compiler/builders:resizebilinear_op_builder",
        "//litert/vendors/samsung/compiler/builders:resizenearestneighbor_op_builder",
        "//litert/vendors/samsung/compiler/builders:reversev2_op_builder",
        "//litert/vendors/samsung/compiler/builders:rms_norm_op_builder",
        "//litert/vendors/samsung/compiler/builders:select_op_builder",
        "//litert/vendors/samsung/compiler/builders:selectv2_op_builder",
        "//litert/vendors/samsung/compiler/builders:slice_op_builder",
        "//litert/vendors/samsung/compiler/builders:softmax_op_builder",
        "//litert/vendors/samsung/compiler/builders:spacetodepth_op_builder",
        "//litert/vendors/samsung/compiler/builders:split_op_builder",
        "//litert/vendors/samsung/compiler/builders:strided_slice_op_builder",
        "//litert/vendors/samsung/compiler/builders:tanh_op_builder",
        "//litert/vendors/samsung/compiler/builders:transpose_conv_op_builder",
        "//litert/vendors/samsung/compiler/builders:transpose_op_builder",
        "//litert/vendors/samsung/compiler/builders:utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:str_format",
        "@exynos_ai_litecore//:ai_litecore_headers",
    ],
)

cc_library(
    name = "compile_model",
    srcs = ["compile_model.cc"],
    hdrs = ["compile_model.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_common",
        "//litert/cc:litert_expected",
        "//litert/vendors/samsung:ai_litecore_api",
        "@exynos_ai_litecore//:ai_litecore_headers",
    ],
)
