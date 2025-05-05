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

load("//litert/build_common:special_rule.bzl", "gles_linkopts")
load("//litert/integration_test:litert_device.bzl", "litert_device_exec", "litert_device_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",  # copybara:comment_replace "//litert:litert_stable_abi_users",
    ],
)

# LiteRT C++ APIs which don't use C API calls to framework.

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
    srcs = [
        "litert_any_test.cc",
    ],
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
    hdrs = [
        "litert_consts.h",
    ],
)

cc_library(
    name = "litert_model",
    srcs = ["litert_model.cc"],
    hdrs = [
        "litert_model.h",
    ],
    deps = [
        ":litert_buffer_ref",
        ":litert_consts",
        ":litert_detail",
        ":litert_element_type",
        ":litert_expected",
        ":litert_handle",
        ":litert_layout",
        ":litert_macros",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_model_test",
    srcs = [
        "litert_model_test.cc",
    ],
    data = [
        "//litert/test:mlir_test_data",
    ],
    deps = [
        ":litert_element_type",
        ":litert_layout",
        ":litert_model",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/core/model",
        "//litert/test:common",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_handle",
    hdrs = ["litert_handle.h"],
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
    hdrs = [
        "litert_buffer_ref.h",
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
        ":litert_macros",
        "//litert/c:litert_common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_expected",
    hdrs = ["litert_expected.h"],
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
    deps = [
        "//litert/c:litert_common",
        "@com_google_absl//absl/log:absl_check",
    ],
)

# Dispatch Delegate of LiteRt.
# Warning: This API is not ABI stable and is subject to change.
cc_library(
    name = "litert_dispatch_delegate",
    hdrs = [
        "litert_dispatch_delegate.h",
    ],
    deps = [
        "//litert/c:litert_environment_options",
        "//litert/runtime/dispatch:dispatch_delegate",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/delegates/utils:simple_opaque_delegate",
    ],
)

cc_library(
    name = "litert_element_type",
    hdrs = ["litert_element_type.h"],
    deps = ["//litert/c:litert_model"],
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
    name = "litert_model_predicates",
    srcs = ["litert_model_predicates.cc"],
    hdrs = ["litert_model_predicates.h"],
    deps = [
        ":litert_detail",
        ":litert_model",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_layout",
    hdrs = ["litert_layout.h"],
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

cc_test(
    name = "litert_model_predicates_test",
    srcs = ["litert_model_predicates_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
    ],
    deps = [
        ":litert_element_type",
        ":litert_model",
        ":litert_model_predicates",
        "//litert/c:litert_op_code",
        "//litert/test:common",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_tensor_buffer_utils",
    srcs = ["litert_tensor_buffer_utils.cc"],
    hdrs = ["litert_tensor_buffer_utils.h"],
    deps = [
        "//litert/c:litert_logging",
        "//litert/c:litert_tensor_buffer_types",
    ],
)

cc_library(
    name = "litert_op_options",
    srcs = ["litert_op_options.cc"],
    hdrs = ["litert_op_options.h"],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/c:litert_op_options",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_op_options_test",
    srcs = ["litert_op_options_test.cc"],
    deps = [
        ":litert_op_options",
        "//litert/c:litert_op_code",
        "//litert/core/model",
        "//litert/core/util:flatbuffer_tools",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_shared_library",
    srcs = ["litert_shared_library.cc"],
    hdrs = ["litert_shared_library.h"],
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
    hdrs = [
        "litert_opaque_options.h",
    ],
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
    srcs = [],
    hdrs = ["litert_platform_support.h"],
    deps = [
        "//litert/c:litert_platform_support",
    ],
)

cc_library(
    name = "litert_logging",
    hdrs = ["litert_logging.h"],
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
        ":litert_environment",
        ":litert_environment_options",
        ":litert_expected",
        "//litert/c:litert_environment_options",
        "@com_google_googletest//:gtest_main",
    ],
)

# LiteRT C++ APIs which will use libLiteRtRuntimeCApi.so.
# - litert::Environment
# - litert::Event
# - litert::TensorBuffer
# - litert::TensorBufferRequirements
# - litert::CompiledModel
# - litert::Options
# - litert::CustomOpKernel
# - litert::DispatchDelegate

LITERT_C_API_DEPS = [
    "//litert/c:litert_any",
    "//litert/c:litert_common",
    "//litert/c:litert_compiled_model",
    "//litert/c:litert_custom_op_kernel",
    "//litert/c:litert_environment",
    "//litert/c:litert_environment_options",
    "//litert/c:litert_event",
    "//litert/c:litert_event_type",
    "//litert/c:litert_gl_types",
    "//litert/c:litert_layout",
    "//litert/c:litert_logging",
    "//litert/c:litert_metrics",
    "//litert/c:litert_model",
    "//litert/c:litert_opaque_options",
    "//litert/c:litert_options",
    "//litert/c:litert_tensor_buffer",
    "//litert/c:litert_tensor_buffer_types",
    "//tflite:framework",
    "//tflite/c:c_api_opaque",
    "//tflite/c:c_api_types",
    "//tflite/c:common",
    "//tflite/core:private_cc_api_stable",
    "//tflite/kernels:builtin_ops",
]

cc_library(
    name = "litert_environment",
    hdrs = ["litert_environment.h"],
    deps = [
        ":litert_any",
        ":litert_environment_options",
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_environment_test",
    srcs = [
        "litert_environment_test.cc",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-llog"],
        "//conditions:default": [],
    }),
    deps = [
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
    hdrs = ["litert_event.h"],
    deps = [
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_event_test",
    srcs = ["litert_event_test.cc"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": [
            "-landroid",
        ],
        "//conditions:default": [],
    }) + gles_linkopts(),
    deps = [
        ":litert_event",
        ":litert_platform_support",
        "//litert/c:litert_common",
        "//litert/c:litert_event_type",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

litert_device_exec(
    name = "litert_event_device_test",
    backend_id = "cpu",
    data = [
    ],
    target = ":litert_event_test",
)

cc_library(
    name = "litert_tensor_buffer",
    srcs = [
        "litert_tensor_buffer.cc",
    ],
    hdrs = [
        "litert_tensor_buffer.h",
        "litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_detail",
        ":litert_event",
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
        ":litert_model",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_tensor_buffer_test",
    srcs = [
        "litert_tensor_buffer_test.cc",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_element_type",
        ":litert_event",
        ":litert_handle",
        ":litert_layout",
        ":litert_macros",
        ":litert_model",
        ":litert_platform_support",
        ":litert_tensor_buffer",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_environment",
        "//litert/runtime:gpu_environment",
        "//litert/runtime:tensor_buffer",
        "//litert/test:matchers",
        "//tflite/delegates/gpu/cl:buffer",
        "//tflite/delegates/gpu/cl:cl_command_queue",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@opencl_headers",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "//tflite/delegates/gpu/gl:egl_environment",
        ],
        "//conditions:default": [],
    }),
)

litert_device_exec(
    name = "litert_tensor_buffer_device_test",
    backend_id = "cpu",
    data = [],
    target = ":litert_tensor_buffer_test",
)

cc_library(
    name = "litert_tensor_buffer_requirements",
    hdrs = [
        "litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_detail",
        ":litert_handle",
        ":litert_macros",
        "//litert/cc:litert_expected",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_tensor_buffer_requirements_test",
    srcs = [
        "litert_tensor_buffer_requirements_test.cc",
    ],
    deps = [
        ":litert_handle",
        ":litert_tensor_buffer",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_compiled_model",
    srcs = ["litert_compiled_model.cc"],
    hdrs = ["litert_compiled_model.h"],
    deps = [
        ":litert_custom_op_kernel",
        ":litert_environment",
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
        ":litert_model",
        ":litert_options",
        ":litert_tensor_buffer",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_compiled_model_test",
    srcs = ["litert_compiled_model_test.cc"],
    data = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_compiled_model",
        ":litert_environment",
        ":litert_model",
        ":litert_tensor_buffer",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
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

litert_device_test(
    name = "litert_compiled_model_npu_test",
    srcs = ["litert_compiled_model_npu_test.cc"],
    backend_id = "google_tensor",
    data = [
        "//litert/test:testdata/simple_model_google_tensor.bin",
        "//litert/test:testdata/simple_model_npu.tflite",
    ],
    deps = [
        ":litert_buffer_ref",
        ":litert_compiled_model",
        ":litert_environment",
        ":litert_event",
        ":litert_model",
        ":litert_platform_support",
        ":litert_tensor_buffer",
        "//litert/c:litert_common",
        "//litert/c:litert_dispatch_headers",
        "//litert/core/model:model_buffer",
        "//litert/runtime:tensor_buffer",
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

# copybara:uncomment_begin(google-only)
# cc_test(
#     name = "litert_compiled_model_gpu_test",
#     srcs = ["litert_compiled_model_gpu_test.cc"],
#     data = [
#         "//litert/test:testdata/simple_add3d_cst_int32.tflite",
#         "//litert/test:testdata/simple_cast_and_add_op.tflite",
#         "//litert/test:testdata/simple_model.tflite",
#     ],
#     tags = [
#         "requires-gpu-nvidia",
#     ],
#     deps = [
#         ":litert_compiled_model",
#         ":litert_environment",
#         ":litert_event",
#         ":litert_expected",
#         ":litert_macros",
#         ":litert_model",
#         ":litert_options",
#         ":litert_platform_support",
#         ":litert_tensor_buffer",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/debugging:leak_check",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/strings:string_view",
#         "@com_google_absl//absl/types:span",
#         "//litert/c:litert_common",
#         "//litert/c:litert_event",
#         "//litert/c:litert_event_type",
#         "//litert/c:litert_tensor_buffer_types",
#         "//litert/runtime/accelerators/gpu:accelerator_options",
#         "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",  # buildcleaner: keep
#         "//litert/test:common",
#         "//litert/test:matchers",
#         "//litert/test:simple_model",
#         "//tflite:framework",
#         "//tflite/c:c_api_opaque",
#         "//tflite/c:common",
#         "//tflite/kernels:builtin_ops",
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
# litert_device_exec(
#     name = "litert_compiled_model_gpu_device_test",
#     backend_id = "gpu",
#     data = [
#         "//litert/test:testdata/simple_add3d_cst_int32.tflite",
#         "//litert/test:testdata/simple_cast_and_add_op.tflite",
#         "//litert/test:testdata/simple_model.tflite",
#     ],
#     target = ":litert_compiled_model_gpu_test",
# )
# copybara:uncomment_end

cc_library(
    name = "litert_custom_op_kernel",
    srcs = ["litert_custom_op_kernel.cc"],
    hdrs = ["litert_custom_op_kernel.h"],
    deps = [
        ":litert_expected",
        ":litert_handle",
        ":litert_layout",
        ":litert_tensor_buffer",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_custom_op_test",
    srcs = ["litert_custom_op_test.cc"],
    data = [
        "//litert/test:testdata/simple_model_custom_op.tflite",
    ],
    deps = [
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

cc_library(
    name = "litert_options",
    hdrs = ["litert_options.h"],
    deps = [
        ":litert_custom_op_kernel",
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
        ":litert_opaque_options",
    ] + select({
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

exports_files(glob([
    "*.h",
    "*.cc",
]))
