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

load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")
load("//third_party/nvidia_cuda:build_defs.bzl", "cuda_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//litert:__subpackages__"],
)

cuda_library(
    name = "cache_update_kernel_cuda",
    srcs = ["cache_update_kernel.cu"],
    hdrs = ["cache_update_kernel.h"],
    # copybara:uncomment linkstatic = 1,
    tags = ["manual"],
    deps = ["@local_cuda//:cuda_runtime"],
)

cc_import(
    name = "cache_update_kernel_archive",
    static_library = ":cache_update_kernel_cuda",
    tags = ["manual"],
)

cc_library(
    name = "cache_update_kernel",
    hdrs = ["cache_update_kernel.h"],
    tags = ["manual"],
    deps = [
        ":cache_update_kernel_archive",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_library(
    name = "cache_update_plugin",
    srcs = ["cache_update_plugin.cc"],
    hdrs = ["cache_update_plugin.h"],
    tags = ["manual"],
    deps = [
        ":cache_update_kernel",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
    ],
)

cc_test(
    name = "cache_update_plugin_test",
    srcs = ["cache_update_plugin_test.cc"],
    tags = ["manual"],
    deps = [
        ":cache_update_plugin",
        "@com_google_googletest//:gtest_main",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
        "@local_tensorrt_rtx//:tensorrt_rtx_stub",
    ],
)

cc_test(
    name = "cache_update_plugin_gpu_test",
    srcs = ["cache_update_plugin_gpu_test.cc"],
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
        ":cache_update_kernel",
        ":cache_update_plugin",
        ":subbyte_gemv_plugin",  # TensorRT-RTX addPluginV3 compatibility header.
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
        "@local_tensorrt_rtx//:tensorrt_rtx_stub",
    ],
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

cc_library(
    name = "decode_attention_plugin",
    srcs = ["decode_attention_plugin.cc"],
    hdrs = ["decode_attention_plugin.h"],
    tags = ["manual"],
    deps = [
        "//litert/vendors/nvidia/trtllm:gemma4_head_kernels",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
    ],
)

cc_test(
    name = "decode_attention_plugin_test",
    srcs = ["decode_attention_plugin_test.cc"],
    tags = ["manual"],
    deps = [
        ":decode_attention_plugin",
        "@com_google_googletest//:gtest_main",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
        "@local_tensorrt_rtx//:tensorrt_rtx_stub",
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
        ":cache_update_plugin",
        ":decode_attention_plugin",
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
        "//litert/vendors/nvidia:cache_layout",
        "//litert/vendors/nvidia:memory_profile",
        "//litert/vendors/nvidia:tensorrt_logger",
        "@com_google_absl//absl/types:span",
        "@local_cuda//:cuda_headers",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
    ],
)

cc_test(
    name = "tensorrt_graph_builder_test",
    srcs = ["tensorrt_graph_builder_test.cc"],
    env = {
        "LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS": "bf16",
        "LITERT_NVIDIA_TENSORRT_SHARED_WEIGHTS": "0",
    },
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
        ":tensorrt_graph_builder",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/c/internal:litert_compiler_context",
        "//litert/cc:litert_buffer_ref",
        "//litert/compiler/cc:litert_model",
        "//litert/core/model",
        "//litert/vendors/nvidia:cache_layout",
        "//litert/vendors/nvidia:tensorrt_logger",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
        "@local_tensorrt_rtx//:tensorrt_rtx_stub",
    ],
)

cc_test(
    name = "compiler_plugin_test",
    srcs = ["compiler_plugin_test.cc"],
    env = {
        "LITERT_NVIDIA_TENSORRT_SHARED_WEIGHTS": "1",
    },
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
        ":compiler_plugin",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_options",
        "//litert/c/internal:litert_compiler_context",
        "//litert/cc:litert_buffer_ref",
        "//litert/core/model",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/nvidia:bytecode",
        "//litert/vendors/nvidia:cache_layout",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
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
        "//litert/core:filesystem",
        "//litert/core/model",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/nvidia:bytecode",
        "//litert/vendors/nvidia:cache_layout",
        "//litert/vendors/nvidia:memory_profile",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:string_view",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
        "@local_tensorrt_rtx//:tensorrt_rtx_stub",
    ],
)
