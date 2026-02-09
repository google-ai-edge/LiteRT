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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:special_rule.bzl", "gles_linkopts", "litert_android_linkopts")
load("//litert/integration_test:litert_device.bzl", "litert_device_exec")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # By default, visibility is restricted to litert_cc_users_dynamic_link. Individual targets can be
        # made public by setting their visibility to litert_public.
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_cc_users_dynamic_link",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

LITERT_C_API_DEPS = [
    "//litert/c:litert_any",
    "//litert/c:litert_common",
    "//litert/c:litert_compiled_model",
    "//litert/c:litert_custom_op_kernel",
    "//litert/c:litert_custom_tensor_buffer",
    "//litert/c:litert_environment",
    "//litert/c:litert_environment_options",
    "//litert/c:litert_event",
    "//litert/c:litert_event_type",
    "//litert/c:litert_gl_types",
    "//litert/c:litert_layout",
    "//litert/c/internal:litert_logging",
    "//litert/c:litert_metrics",
    "//litert/c:litert_model",
    "//litert/c:litert_builder",
    "//litert/c:litert_model_types",
    "//litert/c:litert_op_code",
    "//litert/c:litert_op_options",
    "//litert/c:litert_opaque_options",
    "//litert/c:litert_options",
    "//litert/c:litert_profiler",
    "//litert/c:litert_profiler_event",
    "//litert/c:litert_tensor_buffer",
    "//litert/c:litert_tensor_buffer_types",
    "//tflite:framework",
    "//tflite/c:c_api_opaque",
    "//tflite/c:c_api_types",
    "//tflite/c:common",
    "//tflite/core:private_cc_api_stable",
    "//tflite/kernels:builtin_ops",
]

### LiteRT C++ APIs which will use libLiteRt.so.
#
# Note: If you uses any one of the following classes, you should use
# //litert/c:litert_runtime_c_api_shared_lib for C APIs.
#
# If you're application developer, you'd better use
# //litert/cc:litert_api_with_shared_runtime instead.

cc_library(
    name = "litert_custom_op_kernel",
    srcs = ["//litert/cc:litert_custom_op_kernel.cc"],
    hdrs = ["//litert/cc:litert_custom_op_kernel.h"],
    deps = [
        ":litert_environment",
        ":litert_tensor_buffer",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
            "//litert/c/internal:litert_logging",
        ],
    }),
)

cc_test(
    name = "litert_custom_op_test",
    srcs = ["//litert/cc:litert_custom_op_test.cc"],
    data = ["//litert/test:testdata/simple_model_custom_op.tflite"],
    deps = [
        ":litert_compiled_model",
        ":litert_custom_op_kernel",
        ":litert_environment",
        ":litert_model",
        ":litert_options",
        ":litert_tensor_buffer",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
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
    name = "litert_event",
    hdrs = ["//litert/cc:litert_event.h"],
    visibility = [
        "//litert:litert_public",
    ],
    deps = [
        ":litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_handle",
    ] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_event_test",
    srcs = ["//litert/cc:litert_event_test.cc"],
    linkopts = litert_android_linkopts() + gles_linkopts(),
    deps = [
        ":litert_environment",
        ":litert_event",
        "//litert/c:litert_event_type",
        "//litert/cc/internal:litert_platform_support",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

litert_device_exec(
    name = "litert_event_device_test",
    backend_id = "cpu",
    target = ":litert_event_test",
)

cc_library(
    name = "litert_model",
    srcs = ["//litert/cc:litert_model.cc"],
    hdrs = ["//litert/cc:litert_model.h"],
    visibility = [
        "//litert:litert_public",
    ],
    deps = [
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc/internal:litert_consts",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_library(
    name = "litert_tensor_buffer_types",
    hdrs = ["//litert/cc:litert_tensor_buffer_types.h"],
    visibility = ["//litert:litert_public"],
    deps = [] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_library(
    name = "litert_extended_model",
    srcs = ["//litert/cc/internal:litert_extended_model.cc"],
    hdrs = ["//litert/cc/internal:litert_extended_model.h"],
    visibility = [
        "//litert:litert_public",
    ],
    deps = [
        ":litert_model",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_consts",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_library(
    name = "litert_builder",
    srcs = ["//litert/cc/internal:litert_builder.cc"],
    hdrs = ["//litert/cc/internal:litert_builder.h"],
    visibility = [
        "//litert:litert_public",
    ],
    deps = [
        ":litert_extended_model",
        ":litert_op_options",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc/internal:litert_detail",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_library(
    name = "litert_matchers",
    hdrs = ["//litert/cc/internal:litert_matchers.h"],
    visibility = [
        "//litert:litert_public",
    ],
    deps = [
        ":litert_extended_model",
        ":litert_op_options",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/c/internal:litert_logging",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_model_test",
    srcs = ["//litert/cc:litert_model_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":litert_model",
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_layout",
        "//litert/core/model",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_model_predicates",
    srcs = ["//litert/cc/internal:litert_model_predicates.cc"],
    hdrs = ["//litert/cc/internal:litert_model_predicates.h"],
    visibility = ["//litert:litert_public"],
    deps = [
        ":litert_extended_model",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc/internal:litert_detail",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_model_predicates_test",
    srcs = ["//litert/cc/internal:litert_model_predicates_test.cc"],
    data = ["//litert/test:mlir_test_data"],
    deps = [
        ":litert_extended_model",
        ":litert_model_predicates",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_element_type",
        "//litert/test:common",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_opaque_options",
    srcs = ["//litert/cc:litert_opaque_options.cc"],
    hdrs = ["//litert/cc:litert_opaque_options.h"],
    deps = [
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_opaque_options_test",
    srcs = ["//litert/cc:litert_opaque_options_test.cc"],
    deps = [
        ":litert_opaque_options",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_handle",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_op_options",
    srcs = ["//litert/cc/internal:litert_op_options.cc"],
    hdrs = ["//litert/cc/internal:litert_op_options.h"],
    visibility = ["//litert:litert_public"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers//:runtime_cc",
        # copybara:uncomment "@flatbuffers//src:util",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
    ] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_op_options_test",
    srcs = ["//litert/cc/internal:litert_op_options_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    # Things allocated accros ABI boundaries are causing mismatched-size-delete.
    # use system_malloc, and gunit_main_no_heapcheck to avoid this. Those workaound will be removed
    # once in OSS.
    # copybara:uncomment malloc = "//base:system_malloc",
    tags = [
        "nosan",
    ],
    deps = [
        ":litert_op_options",
        # copybara:uncomment_begin(no OSS for gunit_main_no_heapcheck)
        # "@com_google_googletest//:gtest_main_no_heapcheck",
        # copybara:uncomment_end_and_comment_begin
        "@com_google_googletest//:gtest_main",
        # copybara:comment_end
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/core/model",
        "//litert/core/model:buffer_manager",
        "//litert/core/util:flatbuffer_tools",
        "//litert/test:common",
        "//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "litert_tensor_buffer",
    srcs = ["//litert/cc:litert_tensor_buffer.cc"],
    hdrs = [
        "//litert/cc:litert_tensor_buffer.h",
        "//litert/cc:litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_environment",
        ":litert_event",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_tensor_buffer_test",
    srcs = ["//litert/cc:litert_tensor_buffer_test.cc"],
    linkopts = litert_android_linkopts(),
    deps = [
        ":litert_environment",
        ":litert_event",
        ":litert_model",
        ":litert_tensor_buffer",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:litert_platform_support",
        "//litert/runtime:tensor_buffer",
        "//litert/test:matchers",
        "//tflite/delegates/gpu/cl:buffer",
        "//tflite/delegates/gpu/cl:cl_command_queue",
        "//tflite/delegates/gpu/cl:cl_context",
        "//tflite/delegates/gpu/cl:cl_device",
        "//tflite/delegates/gpu/cl:opencl_wrapper",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@opencl_headers",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "//tflite/delegates/gpu/cl:gl_interop",
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
    hdrs = ["//litert/cc:litert_tensor_buffer_requirements.h"],
    deps = [
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_tensor_buffer_requirements_test",
    srcs = ["//litert/cc:litert_tensor_buffer_requirements_test.cc"],
    deps = [
        ":litert_tensor_buffer",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_environment",
    hdrs = ["//litert/cc:litert_environment.h"],
    deps = [
        "//litert/cc:litert_any",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:litert_runtime_builtin_runtimecapi",
        "//litert/cc/internal:litert_runtime_proxy_runtimecapi",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_environment_test",
    srcs = ["//litert/cc:litert_environment_test.cc"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-llog"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_compiled_model",
        ":litert_environment",
        ":litert_model",
        "//litert/c:litert_common",
        "//litert/cc:litert_common",
        "//litert/cc:litert_expected",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_compiled_model",
    srcs = ["//litert/cc:litert_compiled_model.cc"],
    hdrs = ["//litert/cc:litert_compiled_model.h"],
    deps = [
        ":litert_custom_op_kernel",
        ":litert_environment",
        ":litert_model",
        ":litert_options",
        ":litert_profiler",
        ":litert_tensor_buffer",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/cc/internal:litert_handle",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_test(
    name = "litert_compiled_model_test",
    srcs = ["//litert/cc:litert_compiled_model_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":litert_compiled_model",
        ":litert_environment",
        ":litert_model",
        ":litert_options",
        ":litert_profiler",
        ":litert_tensor_buffer",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_common",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/cc/dynamic_runtime/options:litert_runtime_options",
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

# copybara:uncomment_begin(google-only)
# cc_test(
#     name = "litert_compiled_model_gpu_test",
#     srcs = ["//litert/cc:litert_compiled_model_gpu_test.cc"],
#     data = [
#         "//litert/test:mlir_test_data",
#         "//litert/test:tflite_test_data",
#     ],
#     tags = ["requires-gpu-nvidia"],
#     deps = [
#         "@com_google_googletest//:gtest_main_no_heapcheck",  # To workaround the memory leak in Nvidia's driver
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/strings:string_view",
#         "@com_google_absl//absl/types:span",
#         "//litert/cc:litert_api_with_dynamic_runtime",
#         "//litert/cc:litert_ranked_tensor_type",
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

cc_library(
    name = "litert_options",
    srcs = ["//litert/cc:litert_options.cc"],
    hdrs = ["//litert/cc:litert_options.h"],
    deps = [
        ":litert_custom_op_kernel",
        ":litert_opaque_options",
        "//litert/cc:litert_common",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/dynamic_runtime/options:litert_compiler_options",
        "//litert/cc/dynamic_runtime/options:litert_cpu_options",
        "//litert/cc/dynamic_runtime/options:litert_google_tensor_options",
        "//litert/cc/dynamic_runtime/options:litert_gpu_options",
        "//litert/cc/dynamic_runtime/options:litert_intel_openvino_options",
        "//litert/cc/dynamic_runtime/options:litert_mediatek_options",
        "//litert/cc/dynamic_runtime/options:litert_qualcomm_options",
        "//litert/cc/dynamic_runtime/options:litert_runtime_options",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:scoped_file",
        "//litert/cc/internal:scoped_weight_source",
        "//litert/core:options",
        "@com_google_absl//absl/container:flat_hash_map",
    ] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)

cc_library(
    name = "litert_profiler",
    hdrs = ["//litert/cc:litert_profiler.h"],
    deps = [
        ":litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_handle",
    ] + select({
        "@org_tensorflow//tensorflow:emscripten": LITERT_C_API_DEPS,
        "@org_tensorflow//tensorflow:ios": LITERT_C_API_DEPS,
        "//conditions:default": [
            "//litert/c:litert_runtime_c_api_shared_lib",
        ],
    }),
)
