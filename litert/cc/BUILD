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
load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:special_rule.bzl", "gles_headers", "gles_linkopts")
load("//litert/integration_test:litert_device.bzl", "litert_device_test")
# copybara:uncomment load("//tools/build_defs/apple:ios.bzl", "ios_unit_test")
# copybara:uncomment load("//tools/build_defs/apple:macos.bzl", "macos_unit_test")

# copybara:uncomment LITERT_MIN_IOS_VERSION = "14.0"

# copybara:uncomment LITERT_MIN_MACOS_VERSION = "14.0"

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # By default, visibility is restricted to litert_cc_users_static_link. Individual targets can be
        # made public by setting their visibility to litert_public.
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_static_link",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

# -----------------------------------------------------------------------------
# LiteRT C++ APIs which don't use C API calls to framework.
# -----------------------------------------------------------------------------

cc_library(
    name = "litert_c_types_printing",
    hdrs = ["litert_c_types_printing.h"],
    visibility = ["//litert:litert_public"],
    deps = [
        ":litert_logging",
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_c_types_printing_test",
    srcs = ["litert_c_types_printing_test.cc"],
    deps = [
        ":litert_c_types_printing",
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_any",
    hdrs = ["litert_any.h"],
    visibility = ["//litert:litert_public"],
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
    name = "litert_detail_test",
    srcs = ["litert_detail_test.cc"],
    deps = [
        ":litert_detail",
        "@com_google_googletest//:gtest_main",
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
    name = "litert_consts",
    hdrs = ["litert_consts.h"],
    visibility = ["//litert:litert_public"],
)

cc_library(
    name = "litert_handle",
    hdrs = ["litert_handle.h"],
    visibility = [
        "//litert:litert_public",
    ],
)

cc_test(
    name = "litert_handle_test",
    srcs = ["litert_handle_test.cc"],
    deps = [
        ":litert_handle",
        "//litert/c:litert_common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_buffer_ref",
    hdrs = ["litert_buffer_ref.h"],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
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
    name = "litert_source_location",
    hdrs = ["litert_source_location.h"],
)

cc_test(
    name = "litert_source_location_test",
    srcs = ["litert_source_location_test.cc"],
    deps = [
        ":litert_source_location",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_macros",
    srcs = ["litert_macros.cc"],
    hdrs = ["litert_macros.h"],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_expected",
        ":litert_source_location",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "litert_macros_test",
    srcs = ["litert_macros_test.cc"],
    deps = [
        ":litert_expected",
        ":litert_logging",
        ":litert_macros",
        "//litert/c:litert_common",
        "//litert/test:matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_tflite_error_status_builder",
    hdrs = ["litert_tflite_error_status_builder.h"],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_macros",
        "//litert/c:litert_common",
        "//tflite/c:c_api_types",
    ],
)

cc_library(
    name = "litert_expected",
    hdrs = ["litert_expected.h"],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_detail",
        "//litert/c:litert_common",
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
    name = "litert_detail",
    hdrs = ["litert_detail.h"],
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "litert_element_type",
    hdrs = ["litert_element_type.h"],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_detail",
        "//litert/c:litert_model_types",
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
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_consts",
        ":litert_expected",
        ":litert_macros",
        "//litert/c:litert_layout",
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
    name = "litert_tensor_buffer_utils",
    srcs = ["litert_tensor_buffer_utils.cc"],
    hdrs = ["litert_tensor_buffer_utils.h"],
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_logging",
        "//litert/c:litert_tensor_buffer_types",
    ],
)

cc_library(
    name = "litert_shared_library",
    srcs = ["litert_shared_library.cc"],
    hdrs = ["litert_shared_library.h"],
    visibility = [
        # copybara:uncomment_begin(oss)
        # "//litert:litert_public",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "test_litert_shared_library_impl",
    srcs = ["test_shared_library.cc"],
)

cc_shared_library(
    name = "test_litert_shared_library",
    shared_lib_name = "test_shared_library.so",
    deps = [":test_litert_shared_library_impl"],
)

cc_test(
    name = "litert_shared_library_test",
    srcs = ["litert_shared_library_test.cc"],
    data = [":test_litert_shared_library"],
    defines = ["LITERT_DEFINE_GTEST_STATUS_PRINTER"],
    deps = [
        ":litert_shared_library",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_opaque_options",
    srcs = ["litert_opaque_options.cc"],
    hdrs = ["litert_opaque_options.h"],
    visibility = ["//litert:litert_public"],
    deps = [
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_opaque_options_test",
    srcs = ["litert_opaque_options_test.cc"],
    deps = [
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
        ":litert_opaque_options",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_platform_support",
    hdrs = ["litert_platform_support.h"],
    deps = ["//litert/c:litert_platform_support"],
)

cc_library(
    name = "litert_logging",
    hdrs = ["litert_logging.h"],
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
    ],
)

cc_test(
    name = "litert_logging_test",
    srcs = ["litert_logging_test.cc"],
    deps = [
        ":litert_logging",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_environment_options",
    hdrs = ["litert_environment_options.h"],
    visibility = ["//litert:litert_public"],
    deps = [
        ":litert_any",
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
    ],
)

cc_test(
    name = "litert_environment_options_test",
    srcs = ["litert_environment_options_test.cc"],
    deps = [
        ":litert_any",
        ":litert_environment_options",
        ":litert_expected",
        "//litert/c:litert_environment_options",
        "//litert/cc/internal:litert_environment",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_rng",
    srcs = ["litert_rng.cc"],
    hdrs = ["litert_rng.h"],
    visibility = ["//litert:litert_public"],
    deps = [
        ":litert_detail",
        ":litert_expected",
        ":litert_macros",
        ":litert_numerics",
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_rng_test",
    srcs = ["litert_rng_test.cc"],
    deps = [
        ":litert_numerics",
        ":litert_rng",
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/test:matchers",
        "//litert/test:rng_fixture",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_numerics",
    hdrs = ["litert_numerics.h"],
    deps = [":litert_detail"],
)

# -----------------------------------------------------------------------------
# LiteRT C++ APIs which needs to use (static) C Runtime Framework.
# -----------------------------------------------------------------------------

litert_device_test(
    name = "litert_compiled_model_google_tensor_test",
    srcs = ["litert_compiled_model_google_tensor_test.cc"],
    backend_id = "google_tensor",
    data = [
        "//litert/test:testdata/simple_model_npu_google_tensor_precompiled.tflite",
    ],
    linkopts = gles_linkopts(),
    deps = [
        ":litert_api_with_dynamic_runtime",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "//tflite/delegates/gpu/gl:egl_environment",
        ],
        "//conditions:default": [],
    }),
)

litert_device_test(
    name = "litert_compiled_model_qualcomm_test",
    srcs = ["litert_compiled_model_qualcomm_test.cc"],
    backend_id = "qualcomm",
    data = [
        "//litert/test:testdata/simple_model_qualcomm_sm8650_precompiled.tflite",
    ],
    deps = [
        ":litert_api_with_dynamic_runtime",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

# copybara:uncomment_begin(google-only)
# cc_test(
#     name = "litert_compiled_model_gpu_test",
#     srcs = ["litert_compiled_model_gpu_test.cc"],
#     data = [
#         "//litert/test:mlir_test_data",
#         "//litert/test:tflite_test_data",
#     ],
#     tags = ["requires-gpu-nvidia"],
#     deps = [
#         ":litert_api_with_dynamic_runtime",
#         "@com_google_googletest//:gtest_main_no_heapcheck",  # To workaround the memory leak in Nvidia's driver
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/strings:string_view",
#         "@com_google_absl//absl/types:span",
#         "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator_shared_lib",  # buildcleaner: keep
#         "//litert/test:common",
#         "//litert/test:matchers",
#         "//litert/test:simple_model",
#         "@opencl_headers",
#     ] + select({
#         "@org_tensorflow//tensorflow:android": [
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

# copybara:uncomment_begin(google-only)
# objc_library(
#     name = "LitertTensorBufferTestLibrary",
#     testonly = True,
#     srcs = ["litert_tensor_buffer_test.mm"],
#     deps = [
#         ":litert_expected",
#         ":litert_handle",
#         ":litert_layout",
#         ":litert_macros",
#         ":litert_platform_support",
#         "//third_party/apple_frameworks:XCTest",
#         "//litert/c:litert_common",
#         "//litert/c:litert_environment",
#         "//litert/c:litert_environment_options",
#         "//litert/c:litert_model_types",
#         "//litert/c:litert_tensor_buffer_types",
#         "//litert/cc/internal:litert_environment",
#         "//litert/cc/internal:litert_event",
#         "//litert/cc/internal:litert_model",
#         "//litert/cc/internal:litert_tensor_buffer",
#         "//litert/runtime:tensor_buffer",
#         "//litert/runtime/accelerators/gpu:ml_drift_metal_accelerator",
#         "//tflite/delegates/gpu/metal:metal_device",
#     ],
# )
#
# objc_library(
#     name = "litert_compiled_model_metal_test",
#     testonly = True,
#     srcs = [":litert_compiled_model_metal_test.mm"],
#     data = ["//litert/test:testdata/simple_model.tflite"],
#     tags = ["notap"],
#     deps = [
#         ":litert_expected",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/types:span",
#         "//third_party/apple_frameworks:XCTest",
#         "//litert/c:litert_common",
#         "//litert/c:litert_environment_options",
#         "//litert/cc/internal:litert_compiled_model",
#         "//litert/cc/internal:litert_environment",
#         "//litert/cc/internal:litert_gpu_options",
#         "//litert/cc/internal:litert_options",
#         "//litert/runtime/accelerators/gpu:ml_drift_metal_accelerator",
#         "//litert/test:common",
#         "//litert/test:matchers",
#         "//litert/test:simple_model",
#     ],
# )
#
# macos_unit_test(
#     name = "litert_tensor_buffer_test_cc_macos",
#     minimum_os_version = LITERT_MIN_MACOS_VERSION,
#     tags = ["notap"],
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
#     tags = ["notap"],
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
# -----------------------------------------------------------------------------

# User facing C/C++ API target that depends on the dynamic runtime `libLiteRtRuntimeCApi.so`.
cc_library(
    name = "litert_api_with_dynamic_runtime",
    hdrs = [
        # C++ API headers.
        "litert_any.h",
        "litert_buffer_ref.h",
        "litert_c_types_printing.h",
        "litert_compiled_model.h",
        "litert_consts.h",
        "litert_custom_op_kernel.h",
        "litert_detail.h",
        "litert_platform_support.h",
        "litert_element_type.h",
        "litert_environment.h",
        "litert_environment_options.h",
        "litert_event.h",
        "litert_expected.h",
        "litert_handle.h",
        "litert_layout.h",
        "litert_logging.h",
        "litert_macros.h",
        "litert_model.h",
        "litert_model_predicates.h",
        "litert_numerics.h",
        "litert_opaque_options.h",
        "litert_options.h",
        "litert_profiler.h",
        "litert_source_location.h",
        "litert_tensor_buffer.h",
        "litert_tensor_buffer_requirements.h",
        "//litert/cc/options:litert_gpu_options.h",
        "//litert/cc/options:litert_cpu_options.h",
        "//litert/cc/options:litert_runtime_options.h",
    ] + [
        # Other headers needed by the C++ API headers.
        "//litert/c:c_api_headers",
        "//litert/c/internal:internal_headers",
        "//litert/c/options:options_headers",
        "//tflite:builtin_ops.h",  # Needed for litert/c/litert_op_code.h
        "//tflite/c:tensorflowlite_c_api_hdrs_filegroup",  # Needed for litert/c/litert_model.h
        "//tflite/core/c:headers_filegroup",
    ],
    visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_dynamic_link",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        # C++ API targets.
        "//litert/cc/dynamic_runtime:litert_compiled_model",
        "//litert/cc/dynamic_runtime:litert_custom_op_kernel",
        "//litert/cc/dynamic_runtime:litert_environment",
        "//litert/cc/dynamic_runtime:litert_event",
        "//litert/cc/dynamic_runtime:litert_model",
        "//litert/cc/dynamic_runtime:litert_model_predicates",
        "//litert/cc/dynamic_runtime:litert_op_options",
        "//litert/cc/dynamic_runtime:litert_options",
        "//litert/cc/dynamic_runtime:litert_profiler",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer",
        "//litert/cc/dynamic_runtime:litert_tensor_buffer_requirements",
    ] + [
        # C++ Options API targets.
        "//litert/cc/dynamic_runtime/options:litert_cpu_options",
        "//litert/cc/dynamic_runtime/options:litert_gpu_options",
        "//litert/cc/dynamic_runtime/options:litert_runtime_options",
        "//litert/cc/options:litert_runtime_options",
    ] + [
        # Other (static) C++ API targets.
        ":litert_any",
        ":litert_expected",
        ":litert_macros",
        ":litert_handle",
        ":litert_c_types_printing",
        ":litert_buffer_ref",
        ":litert_element_type",
        ":litert_layout",
        ":litert_opaque_options",
        ":litert_environment_options",
    ] + [
        "//tflite/core/api",
        "//tflite/profiling:memory_info",
        "//tflite/profiling:profile_buffer",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ] + gles_headers(),
)

exports_files(glob([
    "*.h",
    "*.cc",
]))
