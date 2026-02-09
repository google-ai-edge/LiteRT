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
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# copybara:uncomment load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("//litert/build_common:litert_build_defs.bzl", "cc_library_with_testonly_vis")
load("//litert/build_common:special_rule.bzl", "gles_headers", "gles_linkopts", "litert_android_linkopts")
load("//litert/integration_test:litert_device.bzl", "litert_device_exec", "litert_device_test")
# copybara:uncomment load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
# copybara:uncomment load("@build_bazel_rules_apple//apple:macos.bzl", "macos_unit_test")

# copybara:uncomment LITERT_MIN_IOS_VERSION = "14.0"

# copybara:uncomment LITERT_MIN_MACOS_VERSION = "14.0"

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    features = ["allow_starlark_transitions"],
)

# -----------------------------------------------------------------------------
# LiteRT C++ APIs which don't use C API calls to framework.
#
# Visibility: litert_public
# -----------------------------------------------------------------------------

cc_library(
    name = "litert_any",
    hdrs = ["litert_any.h"],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_any_test",
    srcs = ["litert_any_test.cc"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-llog"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_any",
        "//litert/c:litert_any",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_buffer_ref",
    hdrs = ["litert_buffer_ref.h"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_buffer_ref_test",
    srcs = ["litert_buffer_ref_test.cc"],
    deps = [
        ":litert_buffer_ref",
        "//litert/core/util:flatbuffer_tools",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_common",
    hdrs = ["litert_common.h"],
    deps = [
        "//litert/c:litert_common",
    ],
)

cc_library(
    name = "litert_macros",
    srcs = ["litert_macros.cc"],
    hdrs = ["litert_macros.h"],
    deps = [
        ":litert_expected",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc/internal:litert_source_location",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "litert_macros_test",
    srcs = ["litert_macros_test.cc"],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//litert/c:litert_common",
        "//litert/cc/internal:litert_logging",
        "//litert/test:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_expected",
    hdrs = ["litert_expected.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_common",
        "//litert/cc/internal:litert_detail",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "litert_expected_test",
    srcs = ["litert_expected_test.cc"],
    deps = [
        ":litert_buffer_ref",
        ":litert_expected",
        "//litert/c:litert_common",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_element_type",
    hdrs = ["litert_element_type.h"],
    deps = [
        "//litert/c:litert_model_types",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_numerics",
    ],
)

cc_test(
    name = "litert_element_type_test",
    srcs = ["litert_element_type_test.cc"],
    deps = [
        ":litert_element_type",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_layout",
    hdrs = ["litert_layout.h"],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//litert/c:litert_layout",
        "//litert/cc/internal:litert_consts",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_layout_test",
    srcs = ["litert_layout_test.cc"],
    deps = [
        ":litert_layout",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_opaque_options",
    srcs = ["litert_opaque_options.cc"],
    hdrs = ["litert_opaque_options.h"],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_opaque_options_test",
    srcs = ["litert_opaque_options_test.cc"],
    deps = [
        ":litert_expected",
        ":litert_macros",
        ":litert_opaque_options",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/cc/internal:litert_handle",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_environment_options",
    hdrs = ["litert_environment_options.h"],
    visibility = ["//litert:litert_public"],
    deps = [
        ":litert_any",
        ":litert_common",
        ":litert_expected",
        ":litert_macros",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options_header",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_environment_options_test",
    srcs = ["litert_environment_options_test.cc"],
    deps = [
        ":litert_any",
        ":litert_environment",
        ":litert_environment_options",
        ":litert_expected",
        "//litert/c:litert_environment_options",
        "@com_google_googletest//:gtest_main",
    ],
)

# -----------------------------------------------------------------------------
# LiteRT C++ APIs which needs to use (static) C Runtime Framework.
#
# Visibility: litert_cc_users_static_link
# Note: Modules are ordered alphabeticaly.
# -----------------------------------------------------------------------------

cc_library(
    name = "litert_compiled_model",
    srcs = [":litert_compiled_model.cc"],
    hdrs = [":litert_compiled_model.h"],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_static_link",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_buffer_ref",
        ":litert_common",
        ":litert_environment",
        ":litert_expected",
        ":litert_layout",
        ":litert_macros",
        ":litert_model",
        ":litert_options",
        ":litert_profiler",
        ":litert_ranked_tensor_type",
        ":litert_tensor_buffer",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_metrics",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

# copybara:uncomment_begin(google-only)
# litert_device_test(
#     name = "litert_compiled_model_google_tensor_test",
#     srcs = ["litert_compiled_model_google_tensor_test.cc"],
#     backend_id = "google_tensor",
#     data = [
#         "//litert/test:testdata/simple_model_npu_google_tensor_precompiled.tflite",
#     ],
#     linkopts = gles_linkopts(),
#     deps = [
#         ":litert_api_with_dynamic_runtime",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/strings:string_view",
#         "@com_google_absl//absl/types:span",
#         "//litert/test:common",
#         "//litert/test:matchers",
#         "//litert/test:simple_model",
#     ] + select({
#         "@org_tensorflow//tensorflow:android": [
#             "//tflite/delegates/gpu/gl:egl_environment",
#         ],
#         "//conditions:default": [],
#     }),
# )
# copybara:uncomment_end(google-only)

litert_device_test(
    name = "litert_compiled_model_qualcomm_test",
    srcs = ["litert_compiled_model_qualcomm_test.cc"],
    backend_id = "qualcomm",
    data = [
        "//litert/test:testdata/simple_model_qualcomm_sm8650_precompiled.tflite",
    ],
    deps = [
        ":litert_api_with_dynamic_runtime",
        "//litert/c/internal:litert_logging",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

# copybara:uncomment_begin(google-only)
#
# cc_test(
#     name = "litert_compiled_model_gpu_test",
#     srcs = [":litert_compiled_model_gpu_test.cc"],
#     data = [
#         "//litert/test:mlir_test_data",
#         "//litert/test:tflite_test_data",
#     ],
#     tags = [
#         "no_oss",
#         "requires-gpu-nvidia",
#     ],
#     deps = [
#         ":litert_common",
#         ":litert_compiled_model",
#         ":litert_element_type",
#         ":litert_environment",
#         ":litert_event",
#         ":litert_expected",
#         ":litert_layout",
#         ":litert_macros",
#         ":litert_model",
#         ":litert_options",
#         ":litert_profiler",
#         ":litert_ranked_tensor_type",
#         ":litert_tensor_buffer",
#         ":litert_tensor_buffer_types",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/debugging:leak_check",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/strings:string_view",
#         "@com_google_absl//absl/types:span",
#         "//litert/c:litert_common",
#         "//litert/c:litert_environment_options",
#         "//litert/c:litert_event",
#         "//litert/c:litert_event_type",
#         "//litert/c:litert_profiler_event",
#         "//litert/c:litert_tensor_buffer",
#         "//litert/c:litert_tensor_buffer_types",
#         "//litert/cc/internal:litert_platform_support",
#         "//litert/cc/options:litert_gpu_options",
#         "//litert/cc/options:litert_runtime_options",
#         "//litert/test:common",
#         "//litert/test:matchers",
#         "//litert/test:simple_model",
#         "@opencl_headers",
#     ] + select({
#         "@org_tensorflow//tensorflow:android": [
#             "//litert/runtime/accelerators/gpu:ml_drift_cl_gl_accelerator",  # buildcleaner: keep
#             "//tflite/delegates/gpu/cl:cl_device",
#             "//tflite/delegates/gpu/cl:gl_interop",
#             "//tflite/delegates/gpu/cl:opencl_wrapper",
#             "//tflite/delegates/gpu/gl:egl_environment",
#         ],
#         "//conditions:default": [
#             "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",  # buildcleaner: keep
#         ],
#     }),
# )
#
# litert_device_exec(
#     name = "litert_compiled_model_gpu_device_test",
#     backend_id = "gpu",
#     data = [
#         "//litert/test:testdata/constant_output_tensor.tflite",
#         "//litert/test:testdata/simple_add3d_cst_int32.tflite",
#         "//litert/test:testdata/simple_cast_and_add_op.tflite",
#         "//litert/test:testdata/simple_model.tflite",
#     ],
#     target = "//litert/cc:litert_compiled_model_gpu_test",
# )
#
# cc_test(
#     name = "litert_compiled_model_webgpu_test",
#     srcs = [":litert_compiled_model_webgpu_test.cc"],
#     data = ["//litert/test:testdata/simple_model.tflite"],
#     tags = [
#         "no_oss",
#         "requires-gpu-nvidia",
#     ],
#     deps = [
#         ":litert_common",
#         ":litert_compiled_model",
#         ":litert_environment",
#         ":litert_expected",
#         ":litert_macros",
#         ":litert_options",
#         ":litert_tensor_buffer",
#         ":litert_tensor_buffer_types",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/debugging:leak_check",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/types:span",
#         "@dawn//dawn:dawn_headers",
#         "@ml_drift//ml_drift/webgpu:execution_environment",
#         "@ml_drift//ml_drift/webgpu:webgpu_headers",
#         "//litert/c:litert_environment_options",
#         "//litert/cc/options:litert_gpu_options",
#         "//litert/runtime/accelerators/gpu:ml_drift_webgpu_accelerator",  # buildcleaner: keep
#         "//litert/test:common",
#         "//litert/test:matchers",
#         "//litert/test:simple_model",
#         "//third_party/vulkan_loader",
#     ],
# )
#
# litert_device_exec(
#     name = "litert_compiled_model_webgpu_device_test",
#     backend_id = "gpu",
#     data = ["//litert/test:testdata/simple_model.tflite"],
#     target = ":litert_compiled_model_webgpu_test",
# )
#
# cc_test(
#     name = "litert_compiled_model_vulkan_test",
#     srcs = [":litert_compiled_model_vulkan_test.cc"],
#     data = ["//litert/test:testdata/simple_model.tflite"],
#     tags = [
#         "no_oss",
#         "requires-gpu-nvidia",
#     ],
#     deps = [
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/debugging:leak_check",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/types:span",
#         "//litert/c:litert_common",
#         "//litert/c:litert_environment_options",
#         "//litert/c:litert_event",
#         "//litert/c:litert_event_type",
#         "//litert/c:litert_profiler_event",
#         "//litert/c:litert_tensor_buffer_types",
#         "//litert/cc:litert_common",
#         "//litert/cc:litert_compiled_model",
#         "//litert/cc:litert_environment",
#         "//litert/cc:litert_event",
#         "//litert/cc:litert_expected",
#         "//litert/cc:litert_macros",
#         "//litert/cc:litert_model",
#         "//litert/cc:litert_options",
#         "//litert/cc:litert_tensor_buffer",
#         "//litert/cc/internal:litert_platform_support",
#         "//litert/cc/options:litert_gpu_options",
#         "//litert/cc/options:litert_runtime_options",
#         "//litert/runtime/accelerators/gpu:ml_drift_vulkan_accelerator",  # buildcleaner: keep
#         "//litert/test:common",
#         "//litert/test:matchers",
#         "//litert/test:simple_model",
#         "//third_party/vulkan_loader",
#     ],
# )
#
# litert_device_exec(
#     name = "litert_compiled_model_vulkan_device_test",
#     backend_id = "gpu",
#     data = ["//litert/test:testdata/simple_model.tflite"],
#     target = ":litert_compiled_model_vulkan_test",
# )
#
# objc_library(
#     name = "litert_compiled_model_metal_test",
#     testonly = True,
#     srcs = [":litert_compiled_model_metal_test.mm"],
#     data = [
#         "//litert/test:testdata/constant_output_tensor.tflite",
#         "//litert/test:testdata/simple_add3d_cst_int32.tflite",
#         "//litert/test:testdata/simple_cast_and_add_op.tflite",
#         "//litert/test:testdata/simple_model.tflite",
#     ],
#     deps = [
#         ":litert_common",
#         ":litert_compiled_model",
#         ":litert_environment",
#         ":litert_expected",
#         ":litert_options",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/types:span",
#         "//third_party/apple_frameworks:XCTest",
#         "//litert/c:litert_common",
#         "//litert/c:litert_environment_options",
#         "//litert/cc/options:litert_gpu_options",
#         "//litert/runtime/accelerators/gpu:ml_drift_metal_accelerator",
#         "//litert/test:common",
#         "//litert/test:matchers",
#         "//litert/test:metal_test_helper",
#         "//litert/test:simple_model",
#     ],
# )
#
# # Should use `--copt=-DCL_DELEGATE_NO_GL` to build for Desktop.
# cc_test(
#     name = "litert_compiled_model_jetgpu_test",
#     srcs = [":litert_compiled_model_jetgpu_test.cc"],
#     data = [
#         "//litert/test:mlir_test_data",
#         "//litert/test:tflite_test_data",
#     ],
#     tags = [
#         "manual",
#         "no_oss",
#         "nofastbuild",
#         "noios",
#         "nolinux",
#         "nomacos",
#         "nowindows",
#         "requires-gpu-nvidia",
#     ],
#     deps = [
#         ":litert_common",
#         ":litert_compiled_model",
#         ":litert_element_type",
#         ":litert_environment",
#         ":litert_event",
#         ":litert_expected",
#         ":litert_layout",
#         ":litert_macros",
#         ":litert_model",
#         ":litert_options",
#         ":litert_profiler",
#         ":litert_tensor_buffer",
#         ":litert_tensor_buffer_types",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/debugging:leak_check",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/strings:string_view",
#         "@com_google_absl//absl/types:span",
#         "//litert/c:litert_common",
#         "//litert/c:litert_environment_options",
#         "//litert/c:litert_event",
#         "//litert/c:litert_event_type",
#         "//litert/c:litert_profiler_event",
#         "//litert/c:litert_tensor_buffer",
#         "//litert/c:litert_tensor_buffer_types",
#         "//litert/cc/internal:litert_platform_support",
#         "//litert/cc/options:litert_gpu_options",
#         "//litert/cc/options:litert_runtime_options",
#         "//litert/test:common",
#         "//litert/test:matchers",
#         "//litert/test:simple_model",
#         "@opencl_headers",
#     ] + select({
#         "@org_tensorflow//tensorflow:android": [
#             "//litert/runtime/accelerators/gpu/google:jet_gpu_accelerator",  # buildcleaner: keep
#             "//tflite/delegates/gpu/cl:cl_device",
#             "//tflite/delegates/gpu/cl:gl_interop",
#             "//tflite/delegates/gpu/cl:opencl_wrapper",
#             "//tflite/delegates/gpu/gl:egl_environment",
#         ],
#         "//conditions:default": [],
#     }),
# )
#
# copybara:uncomment_end

cc_library(
    name = "litert_custom_op_kernel",
    srcs = [":litert_custom_op_kernel.cc"],
    hdrs = [":litert_custom_op_kernel.h"],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_static_link",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_expected",
        ":litert_layout",
        "//litert/c:litert_common",
        "//litert/c:litert_custom_op_kernel",
        "//litert/c:litert_layout",
        "//litert/c:litert_tensor_buffer",
        "//litert/c/internal:litert_logging",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:litert_tensor_buffer_without_registry",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_custom_op_test",
    srcs = ["litert_custom_op_test.cc"],
    data = ["//litert/test:testdata/simple_model_custom_op.tflite"],
    deps = [
        ":litert_common",
        ":litert_compiled_model",
        ":litert_custom_op_kernel",
        ":litert_environment",
        ":litert_expected",
        ":litert_layout",
        ":litert_macros",
        ":litert_model",
        ":litert_options",
        ":litert_tensor_buffer",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model_custom_op",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "litert_compiled_model_test",
    srcs = ["litert_compiled_model_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":litert_common",
        ":litert_compiled_model",
        ":litert_element_type",
        ":litert_environment",
        ":litert_expected",
        ":litert_layout",
        ":litert_macros",
        ":litert_model",
        ":litert_options",
        ":litert_profiler",
        ":litert_ranked_tensor_type",
        ":litert_tensor_buffer",
        ":litert_tensor_buffer_types",
        "//litert/c:litert_common",
        "//litert/c:litert_compiled_model",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/options:litert_runtime_options",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_testonly_vis(
    name = "litert_environment",
    hdrs = ["litert_environment.h"],
    vis = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_static_link",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_any",
        ":litert_environment_options",
        ":litert_expected",
        ":litert_macros",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:litert_runtime_builtin",
        "//litert/cc/internal:litert_runtime_proxy",
        "@com_google_absl//absl/types:span",
    ],
)

# Flavor aliases for CC environment
cc_test(
    name = "litert_environment_test",
    srcs = ["litert_environment_test.cc"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-llog"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_common",
        ":litert_compiled_model",
        ":litert_environment",
        ":litert_expected",
        ":litert_model",
        "//litert/c:litert_common",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_event",
    hdrs = [":litert_event.h"],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_static_link",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//litert/c:litert_common",
        "//litert/c:litert_event_type",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_opencl_types",
        "//litert/c:litert_profiler_event",
        "//litert/cc:litert_environment",
        "//litert/cc/internal:litert_handle",
    ],
)

cc_test(
    name = "litert_event_test",
    srcs = ["litert_event_test.cc"],
    linkopts = litert_android_linkopts() + gles_linkopts(),
    deps = [
        ":litert_event",
        "//litert/c:litert_event_type",
        "//litert/cc:litert_environment",
        "//litert/cc/internal:litert_platform_support",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

litert_device_exec(
    name = "litert_event_device_test",
    backend_id = "cpu",
    data = [],
    target = ":litert_event_test",
)

cc_library(
    name = "litert_model",
    srcs = [":litert_model.cc"],
    hdrs = [":litert_model.h"],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_static_link",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_buffer_ref",
        ":litert_element_type",
        ":litert_expected",
        ":litert_macros",
        ":litert_ranked_tensor_type",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_model_types",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_ranked_tensor_type",
    hdrs = ["litert_ranked_tensor_type.h"],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_buffer_ref",
        ":litert_element_type",
        ":litert_expected",
        ":litert_layout",
        ":litert_macros",
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_consts",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_ranked_tensor_type_test",
    srcs = ["litert_ranked_tensor_type_test.cc"],
    deps = [
        ":litert_element_type",
        ":litert_layout",
        ":litert_ranked_tensor_type",
        "//litert/c:litert_model_types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "litert_model_test",
    srcs = ["litert_model_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":litert_element_type",
        ":litert_layout",
        ":litert_model",
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/core/model",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_tensor_buffer_requirements",
    hdrs = [":litert_tensor_buffer_requirements.h"],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_static_link",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_expected",
        ":litert_macros",
        ":litert_tensor_buffer_types",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_tensor_buffer_requirements_test",
    srcs = ["litert_tensor_buffer_requirements_test.cc"],
    deps = [
        ":litert_tensor_buffer",
        ":litert_tensor_buffer_types",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_tensor_buffer",
    hdrs = [
        ":litert_tensor_buffer.h",
        ":litert_tensor_buffer_requirements.h",
    ],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_static_link",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_environment",
        ":litert_event",
        ":litert_expected",
        ":litert_macros",
        ":litert_ranked_tensor_type",
        ":litert_tensor_buffer_types",
        "//litert/c:litert_common",
        "//litert/c:litert_custom_tensor_buffer",
        "//litert/c:litert_event",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_model_types",
        "//litert/c:litert_opencl_types",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c:litert_webgpu_types",
        "//litert/c/internal:litert_tensor_buffer_registry",  # buildcleaner: keep
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:litert_tensor_buffer_without_registry",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_options_test",
    srcs = ["litert_options_test.cc"],
    deps = [
        ":litert_common",
        ":litert_options",
        "//litert/cc/internal:scoped_file",
        "//litert/cc/internal:scoped_weight_source",
        "//litert/core:options",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

# copybara:uncomment_begin(google-only)
# cc_test(
#     name = "litert_tensor_buffer_test",
#     srcs = ["litert_tensor_buffer_test.cc"],
#     linkopts = litert_android_linkopts(),
#     deps = [
#         ":litert_element_type",
#         ":litert_environment",
#         ":litert_event",
#         ":litert_expected",
#         ":litert_layout",
#         ":litert_macros",
#         ":litert_model",
#         ":litert_ranked_tensor_type",
#         ":litert_tensor_buffer",
#         ":litert_tensor_buffer_types",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/debugging:leak_check",
#         "@com_google_absl//absl/log:absl_check",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/types:span",
#         "//litert/c:litert_common",
#         "//litert/c:litert_environment",
#         "//litert/c:litert_model_types",
#         "//litert/c:litert_tensor_buffer",
#         "//litert/c:litert_tensor_buffer_types",
#         "//litert/cc/internal:litert_handle",
#         "//litert/cc/internal:litert_platform_support",
#         "//litert/runtime:tensor_buffer",
#         "//litert/test:matchers",
#         "@opencl_headers",
#         "//tflite/delegates/gpu/cl:buffer",
#         "//tflite/delegates/gpu/cl:cl_command_queue",
#         "//tflite/delegates/gpu/cl:cl_context",
#         "//tflite/delegates/gpu/cl:cl_device",
#         "//tflite/delegates/gpu/cl:opencl_wrapper",
#     ] + select({
#         "@org_tensorflow//tensorflow:android": [
#             "//tflite/delegates/gpu/cl:gl_interop",
#             "//tflite/delegates/gpu/gl:egl_environment",
#         ],
#         "//conditions:default": [],
#     }),
# )
#
# litert_device_exec(
#     name = "litert_tensor_buffer_device_test",
#     backend_id = "cpu",
#     data = [],
#     target = ":litert_tensor_buffer_test",
# )
# copybara:uncomment_end(google-only)

cc_library_with_testonly_vis(
    name = "litert_options",
    srcs = ["litert_options.cc"],
    hdrs = ["litert_options.h"],
    vis = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_static_link",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_common",
        ":litert_custom_op_kernel",
        ":litert_expected",
        ":litert_macros",
        ":litert_opaque_options",
        "//litert/c:litert_common",
        "//litert/c:litert_custom_op_kernel",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_options",
        "//litert/c/internal:litert_logging",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:scoped_file",
        "//litert/cc/internal:scoped_weight_source",
        "//litert/cc/options:litert_compiler_options",
        "//litert/cc/options:litert_cpu_options",
        "//litert/cc/options:litert_google_tensor_options",
        "//litert/cc/options:litert_gpu_options",
        "//litert/cc/options:litert_intel_openvino_options",
        "//litert/cc/options:litert_mediatek_options",
        "//litert/cc/options:litert_qualcomm_options",
        "//litert/cc/options:litert_runtime_options",
        "//litert/core:options",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "litert_profiler_test",
    srcs = ["litert_profiler_test.cc"],
    data = [
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":litert_common",
        ":litert_compiled_model",
        ":litert_environment",
        ":litert_options",
        "//litert/test:common",
        "//litert/test:simple_model",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_profiler",
    hdrs = [":litert_profiler.h"],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_static_link",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//litert/c:litert_common",
        "//litert/c:litert_profiler_event",
        "//litert/cc:litert_environment",
        "//litert/cc/internal:litert_handle",
    ],
)

# copybara:uncomment_begin(google-only)
# objc_library(
#     name = "LitertTensorBufferTestLibrary",
#     testonly = True,
#     srcs = ["litert_tensor_buffer_test.mm"],
#     data = [
#         "//litert/test:testdata/simple_model.tflite",
#     ],
#     deps = [
#         ":litert_compiled_model",
#         ":litert_environment",
#         ":litert_event",
#         ":litert_expected",
#         ":litert_layout",
#         ":litert_macros",
#         ":litert_tensor_buffer",
#         "//third_party/apple_frameworks:XCTest",
#         "//litert/cc/internal:litert_handle",
#         "//litert/cc/internal:litert_platform_support",
#         "//litert/runtime/accelerators/gpu:ml_drift_metal_accelerator",
#         "//litert/test:common",
#         "//litert/test:matchers",
#         "//litert/test:metal_test_helper",
#         "//litert/test:simple_model",
#         "//tflite/delegates/gpu/metal:metal_device",
#     ],
# )
#
# macos_unit_test(
#     name = "litert_tensor_buffer_test_cc_macos",
#     minimum_os_version = LITERT_MIN_MACOS_VERSION,
#     deps = [":LitertTensorBufferTestLibrary"],
# )
#
# ios_unit_test(
#     name = "litert_tensor_buffer_test_cc_ios",
#     minimum_os_version = LITERT_MIN_IOS_VERSION,
#     runner = "@build_bazel_rules_apple//apple/testing/default_runner:ios_default_runner",
#     tags = [
#         "nobuilder",
#         "notap",
#     ],
#     deps = [":LitertTensorBufferTestLibrary"],
# )
#
# macos_unit_test(
#     name = "litert_compiled_model_metal_test_cc_macos",
#     minimum_os_version = LITERT_MIN_MACOS_VERSION,
#     deps = [":litert_compiled_model_metal_test"],
# )
#
# ios_unit_test(
#     name = "litert_compiled_model_metal_test_cc_ios",
#     minimum_os_version = LITERT_MIN_IOS_VERSION,
#     # copybara:uncomment runner = "@build_bazel_rules_apple//apple/testing/default_runner:ios_default_runner",
#     tags = ["notap"],
#     deps = [":litert_compiled_model_metal_test"],
# )
# copybara:uncomment_end(google-only)

# -----------------------------------------------------------------------------
# LiteRT C++ APIs which needs to use (dynamic) C Runtime Framework.
#
# Visibility: litert_public
# -----------------------------------------------------------------------------

# User facing C/C++ API target that depends on the dynamic runtime `libLiteRt.so`.
cc_library(
    name = "litert_api_with_dynamic_runtime",
    hdrs = [
        # C++ API headers.
        "litert_any.h",
        "litert_common.h",
        "litert_buffer_ref.h",
        "litert_tensor_buffer_types.h",
        "litert_compiled_model.h",
        "litert_custom_op_kernel.h",
        "litert_element_type.h",
        "litert_environment.h",
        "litert_environment_options.h",
        "litert_event.h",
        "litert_expected.h",
        "litert_layout.h",
        "litert_macros.h",
        "litert_model.h",
        "litert_options.h",
        "litert_opaque_options.h",
        "litert_profiler.h",
        "litert_ranked_tensor_type.h",
        "litert_tensor_buffer.h",
        "litert_tensor_buffer_requirements.h",
        "//litert/cc/options:litert_cpu_options.h",
        "//litert/cc/options:litert_google_tensor_options.h",
        "//litert/cc/options:litert_gpu_options.h",
        "//litert/cc/options:litert_mediatek_options.h",
        "//litert/cc/options:litert_qualcomm_options.h",
        "//litert/cc/options:litert_runtime_options.h",
        "//litert/cc/options:litert_intel_openvino_options.h",
        # Other headers needed by the C++ API headers.
        "//litert/cc/internal:litert_c_types_printing.h",
        "//litert/cc/internal:litert_consts.h",
        "//litert/cc/internal:litert_detail.h",
        "//litert/cc/internal:litert_handle.h",
        "//litert/cc/internal:litert_logging.h",
        "//litert/cc/internal:litert_model_predicates.h",
        "//litert/cc/internal:litert_extended_model.h",
        "//litert/cc/internal:litert_numerics.h",
        "//litert/cc/internal:litert_platform_support.h",
        "//litert/cc/internal:litert_source_location.h",
        "//litert/c:c_api_headers",
        "//litert/c/internal:internal_headers",
        "//litert/c/options:options_headers",
        "//tflite:builtin_ops.h",  # Needed for litert/c/litert_op_code.h
        "//tflite/c:tensorflowlite_c_api_hdrs_filegroup",  # Needed for litert/c/litert_model.h
        "//tflite/core/c:headers_filegroup",
    ],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        # C++ API targets.
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/functional:any_invocable",
        "//litert/cc/dynamic_runtime:litert_compiled_model",
        "//litert/cc/dynamic_runtime:litert_custom_op_kernel",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/cc/dynamic_runtime:litert_event",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/dynamic_runtime:litert_extended_model",
        "//litert/cc/dynamic_runtime:litert_model_predicates",
        "//litert/cc/dynamic_runtime:litert_profiler",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer_requirements",
        "//litert/cc/internal:litert_runtime_builtin_runtimecapi",
        # C++ Common Options API targets.
        "//litert/cc/dynamic_runtime:litert_opaque_options",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/cc/dynamic_runtime/options:litert_compiler_options",
        "//litert/cc/dynamic_runtime/options:litert_cpu_options",
        "//litert/cc/dynamic_runtime/options:litert_gpu_options",
        "//litert/cc/dynamic_runtime/options:litert_intel_openvino_options",
        "//litert/cc/dynamic_runtime/options:litert_qualcomm_options",
        "//litert/cc/dynamic_runtime/options:litert_runtime_options",
        # Other (static) C++ API targets.
        ":litert_any",
        ":litert_tensor_buffer_types",
        ":litert_expected",
        ":litert_common",
        ":litert_macros",
        ":litert_buffer_ref",
        ":litert_element_type",
        ":litert_layout",
        ":litert_environment_options",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:litert_c_types_printing",
        "//litert/cc/internal:litert_runtime_proxy",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "//litert/build_common:build_config",
        "@com_google_absl//absl/types:span",
        "//litert/c/internal:litert_logging",
        "//litert/cc/internal:litert_logging",
        "//litert/core:options",
        "//litert/cc/internal:scoped_file",
        "//litert/cc/internal:scoped_weight_source",
        "@opencl_headers",
        "//tflite/core/api",
        "//tflite/profiling:memory_info",
        "//tflite/profiling:profile_buffer",
    ] + select({
        # Windows excludes some options to avoid unnecessary dependencies.
        "@platforms//os:windows": [],
        "//conditions:default": [
            "//litert/cc/dynamic_runtime:litert_op_options",
            # C++ NPU Options API targets.
            "//litert/cc/dynamic_runtime/options:litert_google_tensor_options",
            "//litert/cc/dynamic_runtime/options:litert_mediatek_options",
        ],
    }) + gles_headers(),
)

cc_library(
    name = "litert_tensor_buffer_types",
    srcs = ["litert_tensor_buffer_types.cc"],
    hdrs = ["litert_tensor_buffer_types.h"],
    deps = [
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc/internal:litert_tensor_buffer_utils",
    ],
)

exports_files(glob([
    "*.h",
    "*.cc",
]))
