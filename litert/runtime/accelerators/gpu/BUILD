# Copyright 2025 Google LLC.
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

# copybara:uncomment_begin
# load("//devtools/deps/check:deps_check.bzl", "check_dependencies")
# load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
# load("@rules_cc//cc:cc_library.bzl", "cc_library")
# load("@rules_cc//cc:cc_test.bzl", "cc_test")
# load("@rules_shell//shell:sh_test.bzl", "sh_test")
# copybara:uncomment_end

load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework")
load("@build_bazel_rules_apple//apple:macos.bzl", "macos_dylib")
load("//litert/build_common:litert_build_defs.bzl", "copy_file", "litert_accelerator_library")

# Experimental libraries & runtimes -- DO NOT use or depend on these.
package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

# GPU Accelerator module for LiteRT. This Accelerator supports both OpenCL and WebGPU and OpenGL.
# NOTE: This is experimental and should be only used for testing.
litert_accelerator_library(
    name = "ml_drift_allgpu_accelerator",
    srcs = ["ml_drift_allgpu_accelerator.cc"],
    shared_lib_name = "libLiteRtAllGpuAccelerator.so",
    deps = [
        ":ml_drift_delegate_create",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_custom_tensor_buffer",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/c/options:litert_gpu_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer",
        "//litert/core:environment",
        "//litert/core:options",
        "//litert/runtime/accelerators:gpu_static_registry",
        "//ml_drift_delegate/delegate:buffer_handler_opencl",
        "//ml_drift_delegate/delegate:buffer_handler_webgpu",
        "//ml_drift_delegate/delegate:delegate_opencl",
        "//ml_drift_delegate/delegate:delegate_opengl",
        "//ml_drift_delegate/delegate:delegate_types",
        "//ml_drift_delegate/delegate:delegate_webgpu",
        "//tflite/core/c:c_api_types",
        "@ml_drift//ml_drift/cl:opencl_wrapper",
    ],
)

# GPU Accelerator module for LiteRT. This Accelerator supports both OpenCL and WebGPU.
litert_accelerator_library(
    name = "ml_drift_gpu_accelerator",
    srcs = ["ml_drift_gpu_accelerator.cc"],
    shared_lib_name = "libLiteRtGpuAccelerator.so",
    deps = [
        ":ml_drift_delegate_create",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_custom_tensor_buffer",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/c/options:litert_gpu_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer",
        "//litert/core:environment",
        "//litert/core:options",
        "//litert/runtime/accelerators:gpu_static_registry",
        "//ml_drift_delegate/delegate:buffer_handler_opencl",
        "//ml_drift_delegate/delegate:buffer_handler_webgpu",
        "//ml_drift_delegate/delegate:delegate_opencl",
        "//ml_drift_delegate/delegate:delegate_types",
        "//ml_drift_delegate/delegate:delegate_webgpu",
        "//tflite/core/c:c_api_types",
        "@ml_drift//ml_drift/cl:opencl_wrapper",
    ],
)

cc_library(
    name = "ml_drift_delegate_create",
    srcs = ["ml_drift_delegate_create.cc"],
    hdrs = ["ml_drift_delegate_create.h"],
    visibility = ["//litert/tools:__pkg__"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_accelerator",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_logging_helper_with_runtime_context",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/c/options:litert_gpu_options",
        "//ml_drift_delegate/delegate:delegate_options",
        "//ml_drift_delegate/delegate:delegate_types",
        "//ml_drift_delegate/delegate:precision",
    ],
)

cc_test(
    name = "ml_drift_delegate_create_test",
    srcs = ["ml_drift_delegate_create_test.cc"],
    deps = [
        ":ml_drift_delegate_create",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_options",
        "//litert/c/internal:litert_accelerator_registration",
        "//litert/c/internal:litert_runtime_context_testonly",
        "//litert/c/options:litert_gpu_options",
        "//ml_drift_delegate/delegate:delegate_options",
        "//ml_drift_delegate/delegate:delegate_types",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

# OpenCL Accelerator module for LiteRT. The Accelerator is not bundled with the LiteRT runtime.
# Adding this module to the LiteRT runtime will enable OpenCL support via Accelerator API.
# Static version of the OpenCL Accelerator module for LiteRT.
litert_accelerator_library(
    name = "ml_drift_cl_accelerator",
    srcs = ["ml_drift_cl_accelerator.cc"],
    shared_lib_name = "libLiteRtOpenClAccelerator.so",
    deps = [
        ":ml_drift_delegate_create",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/cc:litert_macros",
        "//litert/core:options",
        "//litert/runtime/accelerators:gpu_static_registry",
        "//ml_drift_delegate/delegate:buffer_handler_opencl",
        "//ml_drift_delegate/delegate:delegate_opencl",
        "//ml_drift_delegate/delegate:delegate_types",
        "//tflite/core/c:c_api_types",
    ],
)

# WebGPU Accelerator module for LiteRT. The Accelerator is not bundled with the LiteRT runtime.
# Adding this module to the LiteRT runtime will enable WebGPU support via Accelerator API.
# Static version of the WebGPU Accelerator module for LiteRT.
litert_accelerator_library(
    name = "ml_drift_webgpu_accelerator",
    srcs = ["ml_drift_webgpu_accelerator.cc"],
    shared_lib_name = "libLiteRtWebGpuAccelerator.so",
    deps = [
        ":ml_drift_delegate_create",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/c/internal:litert_tensor_buffer_registry_header",
        "//litert/cc:litert_macros",
        "//litert/core:options",
        "//litert/runtime/accelerators:gpu_static_registry",
        "//ml_drift_delegate/delegate:buffer_handler_webgpu",
        "//ml_drift_delegate/delegate:delegate_types",
        "//ml_drift_delegate/delegate:delegate_webgpu",
        "//tflite/core/c:c_api_types",
    ],
)

# copybara:uncomment_begin
# sh_test(
#     name = "ml_drift_gpu_accelerators_so_symbol_test",
#     srcs = ["//litert/test:litert_c_api_dependency_test.sh"],
#     args = [
#         "LiteRtAcceleratorImpl",
#         "$(location :ml_drift_allgpu_accelerator_so)",
#         "$(location :ml_drift_cl_accelerator_so)",
#         "$(location :ml_drift_cl_gl_accelerator_so)",
#         "$(location :ml_drift_gpu_accelerator_so)",
#         "$(location :ml_drift_vulkan_accelerator_so)",
#         "$(location :ml_drift_webgpu_accelerator_so)",
#     ],
#     data = [
#         ":ml_drift_allgpu_accelerator_so",
#         ":ml_drift_cl_accelerator_so",
#         ":ml_drift_cl_gl_accelerator_so",
#         ":ml_drift_gpu_accelerator_so",
#         ":ml_drift_vulkan_accelerator_so",
#         ":ml_drift_webgpu_accelerator_so",
#     ],
# )
# copybara:uncomment_end

macos_dylib(
    name = "build_ml_drift_webgpu_accelerator_dylib",
    exported_symbols_lists = [
        "//litert/build_common:export_litert_gpu_accelerator_darwin.lds",
    ],
    linkopts = [
        "-undefined dynamic_lookup",
        "-install_name @rpath/libLiteRtWebGpuAccelerator.dylib",
    ],
    minimum_os_version = "14.0",
    visibility = ["//visibility:private"],
    deps = [":ml_drift_webgpu_accelerator_runtimecapi"],
)

copy_file(
    name = "ml_drift_webgpu_accelerator_dylib",
    src = ":build_ml_drift_webgpu_accelerator_dylib",
    target = "libLiteRtWebGpuAccelerator.dylib",
    visibility = ["//litert:litert_friends"],
)

# Target for windows libLiteRtWebGpuAccelerator.dll.
cc_binary(
    name = "libLiteRtWebGpuAccelerator",
    linkshared = 1,
    tags = [
        "nobuilder",
        "notap",
        "windows",
    ],
    target_compatible_with = ["@platforms//os:windows"],
    visibility = ["//visibility:private"],
    win_def_file = ":windows_exported_symbols.def",
    deps = [":ml_drift_webgpu_accelerator_runtimecapi"],
)

alias(
    name = "ml_drift_webgpu_accelerator_dll",
    actual = ":libLiteRtWebGpuAccelerator",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "ml_drift_opengl_accelerator",
    srcs = ["ml_drift_opengl_accelerator.cc"],
    defines = ["LITERT_USE_STATIC_LINKED_GPU_ACCELERATOR"],
    visibility = ["//research/drishti:__subpackages__"],
    deps = [
        ":ml_drift_delegate_create",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options_header",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/cc:litert_macros",
        "//litert/runtime/accelerators:gpu_static_registry",
        "//ml_drift_delegate/delegate:delegate_opengl",
        "//ml_drift_delegate/delegate:delegate_types",
        "//tflite/core/c:c_api_types",
    ],
    alwayslink = 1,
)

litert_accelerator_library(
    name = "ml_drift_cl_gl_accelerator",
    srcs = ["ml_drift_cl_gl_accelerator.cc"],
    shared_lib_name = "libLiteRtClGlAccelerator.so",
    deps = [
        ":ml_drift_delegate_create",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_logging",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/c/options:litert_gpu_options",
        "//litert/cc:litert_macros",
        "//litert/runtime/accelerators:gpu_static_registry",
        "//ml_drift_delegate/delegate:buffer_handler_opencl",
        "//ml_drift_delegate/delegate:delegate_opencl",
        "//ml_drift_delegate/delegate:delegate_opengl",
        "//ml_drift_delegate/delegate:delegate_types",
        "//tflite/core/c:c_api_types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@ml_drift//ml_drift/cl:opencl_wrapper",
    ],
)

litert_accelerator_library(
    name = "ml_drift_vulkan_accelerator",
    srcs = ["ml_drift_vulkan_accelerator.cc"],
    shared_lib_name = "libLiteRtVulkanAccelerator.so",
    visibility = [
        "//litert:__subpackages__",
        "//third_party/odml/infra:__subpackages__",
        "//third_party/odml/litert:__subpackages__",
    ],
    deps = [
        ":ml_drift_delegate_create",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/cc:litert_macros",
        "//litert/runtime/accelerators:gpu_static_registry",
        "//ml_drift_delegate/delegate:buffer_handler_vulkan",
        "//ml_drift_delegate/delegate:delegate_types",
        "//ml_drift_delegate/delegate:delegate_vulkan",
        "//tflite/core/c:c_api_types",
    ],
)

litert_accelerator_library(
    name = "ml_drift_metal_accelerator",
    srcs = ["ml_drift_metal_accelerator.cc"],
    macos_dylib = True,
    shared_lib_name = "libLiteRtMetalAccelerator.dylib",
    tags = [
        "apple_build",
        "manual",
    ],
    deps = [
        ":ml_drift_delegate_create",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options_header",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_accelerator_def",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/cc:litert_macros",
        "//litert/core:options",
        "//litert/runtime/accelerators:gpu_static_registry",
        "//ml_drift_delegate/delegate:buffer_handler_metal",
        "//ml_drift_delegate/delegate:buffer_handler_metal_headers",
        "//ml_drift_delegate/delegate:delegate_metal",
        "//ml_drift_delegate/delegate:delegate_metal_headers",
        "//ml_drift_delegate/delegate:delegate_types",
        "//tflite/core/c:c_api_types",
    ],
)

macos_dylib(
    name = "build_ml_drift_metal_accelerator_dylib",
    exported_symbols_lists = [
        "//litert/build_common:export_litert_gpu_accelerator_darwin.lds",
    ],
    linkopts = [
        "-undefined dynamic_lookup",
        "-install_name @rpath/libLiteRtMetalAccelerator.dylib",
    ],
    minimum_os_version = "14.0",
    deps = [":ml_drift_metal_accelerator_runtimecapi"],
)

copy_file(
    name = "ml_drift_metal_accelerator_dylib",
    src = ":build_ml_drift_metal_accelerator_dylib",
    target = "libLiteRtMetalAccelerator.dylib",
    visibility = ["//litert:litert_friends"],
)

ios_framework(
    name = "build_ml_drift_metal_accelerator_framework",
    bundle_id = "com.google.litert.metal_accelerator",
    bundle_name = "libLiteRtMetalAccelerator.dylib",
    exported_symbols_lists = [
        "//litert/build_common:export_litert_gpu_accelerator_darwin.lds",
    ],
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = ["AcceleratorInfo.plist"],
    linkopts = [
        "-undefined dynamic_lookup",
        "-install_name @rpath/libLiteRtMetalAccelerator.dylib",
    ],
    minimum_os_version = "15.0",
    deps = [":ml_drift_metal_accelerator_runtimecapi"],
)

# Target for OpenGL Accelerator build test
# This is needed since there is no other target that uses this accelerator in LiteRT.
cc_binary(
    name = "benchmark_model_gl",
    tags = [
        "manual",
        "no_oss",
        "nofastbuild",
        "noios",
        "nolinux",
        "nomacos",
        "nowindows",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":ml_drift_opengl_accelerator",  # buildcleaner: keep
        "//litert/tools:benchmark_model_main",
    ],
)

# copybara:uncomment_begin
# check_dependencies(
#     name = "gpu_accelerator_deps",
#     of = [
#         ":ml_drift_cl_accelerator",
#         ":ml_drift_webgpu_accelerator",
#         ":ml_drift_opengl_accelerator",
#         ":ml_drift_cl_gl_accelerator",
#         ":ml_drift_vulkan_accelerator",
#         ":ml_drift_metal_accelerator",
#         ":libLiteRtWebGpuAccelerator",
#     ],
#     dont_match_regexp = "|".join([
#         "//litert/c:.*_impl",
#     ]),
# )
# copybara:uncomment_end
