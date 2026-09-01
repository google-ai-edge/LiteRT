# Copyright 2026 Google LLC.
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
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//litert:__subpackages__"],
)

cc_library(
    name = "subbyte_gemv_plugin",
    srcs = ["subbyte_gemv_plugin.cc"],
    hdrs = [
        "subbyte_gemv_plugin.h",
        "tensorrt_rtx_plugin_compat.h",
    ],
    tags = ["manual"],
    deps = [
        "//litert/vendors/nvidia/trtllm:gemma4_head_kernels",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
    ],
)

cc_test(
    name = "subbyte_gemv_plugin_test",
    srcs = ["subbyte_gemv_plugin_test.cc"],
    tags = [
        "exclusive",
        "manual",
        "requires-gpu",
    ],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":subbyte_gemv_plugin",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
        "@local_tensorrt_rtx//:tensorrt_rtx_stub",
    ],
)

cc_library(
    name = "tensorrt_graph_builder",
    srcs = ["tensorrt_graph_builder.cc"],
    hdrs = ["tensorrt_graph_builder.h"],
    tags = ["manual"],
    deps = [
        ":subbyte_gemv_plugin",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/c:litert_op_options",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc/internal:litert_op_options",
        "//litert/compiler/cc:litert_model",
        "//litert/vendors/nvidia:bytecode",
        "//litert/vendors/nvidia:tensorrt_logger",
        "@com_google_absl//absl/types:span",
        "@local_cuda//:cuda_headers",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
    ],
)

litert_dynamic_lib(
    name = "compiler_plugin",
    srcs = ["compiler_plugin.cc"],
    hdrs = ["//litert/vendors/c:litert_compiler_plugin.h"],
    export_litert_only = True,
    linkopts = [
        "-Wl,-rpath,$$ORIGIN",
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
    ],
    shared_lib_name = "compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_Nvidia.so",
    tags = ["manual"],
    target_compatible_with = select({
        "//litert:linux_x86_64": [],
        "//litert:linux_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = [
        "//litert:__subpackages__",
        "//litert:litert_friends",
    ],
    deps = [
        ":tensorrt_graph_builder",
        "//litert/c:litert_builder",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_logging_helper_with_compiler_context",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_c_types_printing",
        "//litert/compiler/cc:litert_model",
        "//litert/compiler/plugin:algo",
        "//litert/core/model",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/nvidia:bytecode",
        "@local_cuda//:cuda_runtime",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
        "@local_tensorrt_rtx//:tensorrt_rtx_stub",
    ],
)
