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
load("//litert/build_common:litert_build_defs.bzl", "litert_dispatch_api")
load("//third_party/nvidia_cuda:build_defs.bzl", "cuda_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//litert:__subpackages__"],
)

cc_library(
    name = "greedy_sampler_c_api",
    hdrs = ["greedy_sampler_c_api.h"],
    visibility = ["//visibility:public"],
    deps = ["//litert/c:litert_common"],
)

cuda_library(
    name = "greedy_sampler_kernel_cuda",
    srcs = ["greedy_sampler_kernel.cu"],
    hdrs = ["greedy_sampler_kernel.h"],
    linkstatic = 1,
    tags = ["manual"],
    deps = ["@local_cuda//:cuda_runtime"],
)

cc_import(
    name = "greedy_sampler_kernel_archive",
    static_library = ":greedy_sampler_kernel_cuda",
    tags = ["manual"],
    alwayslink = 1,
)

cc_library(
    name = "greedy_sampler_kernel",
    hdrs = ["greedy_sampler_kernel.h"],
    tags = ["manual"],
    deps = [
        ":greedy_sampler_kernel_archive",
        "@local_cuda//:cuda_headers",
    ],
)

litert_dispatch_api(
    name = "dispatch_api",
    srcs = ["dispatch_api.cc"],
    hdrs = [
        "tensor_buffer_view.h",
        "//litert/vendors/c:litert_dispatch.h",
    ],
    export_litert_only = True,
    linkopts = [
        "-Wl,-soname=libLiteRtDispatch_Nvidia.so",
        "-Wl,-rpath,$$ORIGIN",
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
    ],
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_Nvidia.so",
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
        ":greedy_sampler_c_api",
        ":greedy_sampler_kernel",
        "//litert/c:litert_common",
        "//litert/c:litert_custom_tensor_buffer",
        "//litert/c:litert_environment_header",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_event_header",
        "//litert/c:litert_metrics",
        "//litert/c:litert_model_types",
        "//litert/c:litert_options_header",
        "//litert/c:litert_tensor_buffer_header",
        "//litert/c:litert_tensor_buffer_requirements",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_logging_helper_with_runtime_context",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/c/internal:litert_scheduling_info",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/nvidia:bytecode",
        "//litert/vendors/nvidia:tensorrt_logger",
        "//litert/vendors/nvidia/trtllm:gemma4_head_kernels",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
        "@local_tensorrt_rtx//:tensorrt_rtx_headers",
        "@local_tensorrt_rtx//:tensorrt_rtx_stub",
    ],
)

cc_test(
    name = "tensor_buffer_view_test",
    srcs = [
        "tensor_buffer_view.h",
        "tensor_buffer_view_test.cc",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "greedy_sampler_kernel_test",
    srcs = ["greedy_sampler_kernel_test.cc"],
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
        ":greedy_sampler_kernel",
        "@com_google_googletest//:gtest_main",
        "@local_cuda//:cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)
