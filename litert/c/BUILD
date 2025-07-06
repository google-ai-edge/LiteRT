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

# copybara:uncomment load("//devtools/deps/check:deps_check.bzl", "check_dependencies")
load("//litert/build_common:litert_build_defs.bzl", "export_lrt_runtime_only_linkopt", "export_lrt_runtime_only_script")
load("//litert/build_common:special_rule.bzl", "gles_deps", "gles_headers", "gles_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_public",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

cc_library(
    name = "litert_common",
    srcs = ["litert_common.cc"],
    hdrs = ["litert_common.h"],
    deps = [
        "//tflite/core/api",
        "//tflite/profiling:memory_info",
    ],
)

cc_test(
    name = "litert_common_test",
    srcs = ["litert_common_test.cc"],
    deps = [
        ":litert_common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_custom_op_kernel",
    srcs = [],
    hdrs = ["litert_custom_op_kernel.h"],
    deps = [
        ":litert_common",
        ":litert_layout",
    ],
)

cc_test(
    name = "litert_custom_op_test",
    srcs = [
        "litert_custom_op_test.cc",
    ],
    data = [
        "//litert/test:testdata/simple_model_custom_op.tflite",
    ],
    deps = [
        ":litert_common",
        ":litert_compiled_model",
        ":litert_custom_op_kernel",
        ":litert_environment",
        ":litert_environment_options",
        ":litert_layout",
        ":litert_model",
        ":litert_options",
        ":litert_tensor_buffer",
        ":litert_tensor_buffer_types",
        "//litert/cc:litert_macros",
        "//litert/test:common",
        "//litert/test:simple_model_custom_op",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_any",
    hdrs = ["litert_any.h"],
)

cc_library(
    name = "litert_metrics",
    srcs = ["litert_metrics.cc"],
    hdrs = ["litert_metrics.h"],
    deps = [
        ":litert_any",
        ":litert_common",
        "//litert/runtime:metrics",
    ],
)

cc_library(
    name = "litert_environment",
    srcs = ["litert_environment.cc"],
    hdrs = ["litert_environment.h"],
    deps = [
        ":litert_common",
        ":litert_environment_options",
        "//litert/cc:litert_macros",
        "//litert/core:environment",
        "//litert/runtime:gpu_environment",
        "//litert/runtime/accelerators:auto_registration",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_environment_options",
    srcs = ["litert_environment_options.cc"],
    hdrs = ["litert_environment_options.h"],
    deps = [
        ":litert_any",
        ":litert_common",
        "//litert/cc:litert_macros",
        "//litert/core:environment_options",
    ],
)

cc_library(
    name = "litert_environment_options_header",
    hdrs = ["litert_environment_options.h"],
    tags = ["avoid_dep"],
    deps = [
        ":litert_any",
        ":litert_common",
    ],
)

cc_test(
    name = "litert_environment_options_test",
    srcs = ["litert_environment_options_test.cc"],
    deps = [
        ":litert_any",
        ":litert_common",
        ":litert_environment_options",
        "//litert/core:environment_options",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_logging",
    srcs = [
        "litert_logging.cc",
    ],
    hdrs = [
        "litert_logging.h",
    ],
    deps = [
        ":litert_common",
        "//tflite:minimal_logging",
        "//tflite/core/c:private_common",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_logging_test",
    srcs = [
        "litert_logging_test.cc",
    ],
    deps = [
        ":litert_common",
        ":litert_logging",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_layout",
    srcs = ["litert_layout.cc"],
    hdrs = ["litert_layout.h"],
    deps = [":litert_common"],
)

cc_library(
    name = "litert_model",
    srcs = ["litert_model.cc"],
    hdrs = ["litert_model.h"],
    deps = [
        ":litert_common",
        ":litert_layout",
        ":litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_macros",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//litert/core/model:model_serialize",
        "//tflite/core/c:private_c_api_types",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# Headers only lib for users who require only types and want to avoid
# potential circular dependencies.
cc_library(
    name = "litert_model_types",
    hdrs = [
        "litert_model.h",
    ],
    deps = [
        ":litert_common",
        ":litert_layout",
        ":litert_op_code",
        "//tflite/core/c:private_c_api_types",
    ],
)

cc_test(
    name = "litert_model_test",
    srcs = ["litert_model_test.cc"],
    deps = [
        ":litert_common",
        ":litert_model",
        ":litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/core/model",
        "//litert/core/util:flatbuffer_tools",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_op_code",
    hdrs = ["litert_op_code.h"],
    deps = ["//tflite:builtin_ops"],
)

cc_library(
    name = "litert_op_options",
    srcs = ["litert_op_options.cc"],
    hdrs = [
        "litert_op_options.h",
    ],
    deps = [
        ":litert_common",
        ":litert_op_code",
        "//litert/core/model",
        "//litert/core/util:flatbuffer_tools",
        "//tflite/c:c_api_types",
        "//tflite/schema:schema_fbs",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/core:model_builder_base",
    ],
)

cc_test(
    name = "litert_op_options_test",
    srcs = ["litert_op_options_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
    ],
    tags = ["no_oss"],
    deps = [
        ":litert_common",
        ":litert_op_options",
        "//litert/cc:litert_model",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "litert_event_type",
    hdrs = ["litert_event_type.h"],
)

cc_library(
    name = "litert_event",
    srcs = ["litert_event.cc"],
    hdrs = ["litert_event.h"],
    deps = [
        ":litert_common",
        ":litert_event_type",
        ":litert_logging",
        "//litert/cc:litert_macros",
        "//litert/runtime:event",
    ],
)

cc_library(
    name = "litert_tensor_buffer_types",
    srcs = [],
    hdrs = ["litert_tensor_buffer_types.h"],
)

cc_library(
    name = "litert_gl_types",
    srcs = [],
    hdrs = ["litert_gl_types.h"],
)

cc_library(
    name = "litert_tensor_buffer",
    srcs = [
        "litert_tensor_buffer.cc",
        "litert_tensor_buffer_requirements.cc",
    ],
    hdrs = [
        "litert_tensor_buffer.h",
        "litert_tensor_buffer_requirements.h",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": [
            "-landroid",
        ],
        "//conditions:default": [],
    }) + gles_linkopts(),
    deps = [
        ":litert_common",
        ":litert_event",
        ":litert_gl_types",
        ":litert_logging",
        ":litert_model",
        ":litert_tensor_buffer_types",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/runtime:tensor_buffer",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ] + gles_deps(),
)

cc_test(
    name = "litert_tensor_buffer_test",
    srcs = [
        "litert_tensor_buffer_test.cc",
    ],
    # require GPU to run OpenCL tests.
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        ":litert_any",
        ":litert_common",
        ":litert_environment",
        ":litert_environment_options",
        ":litert_model",
        ":litert_tensor_buffer",
        ":litert_tensor_buffer_types",
        "//litert/cc:litert_any",
        "//litert/cc:litert_layout",
        "//litert/runtime:tensor_buffer",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "litert_tensor_buffer_requirements_test",
    srcs = [
        "litert_tensor_buffer_requirements_test.cc",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_common",
        ":litert_tensor_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_dispatch_delegate",
    hdrs = [
        "litert_dispatch_delegate.h",
    ],
    deps = [
        ":litert_common",
        ":litert_environment_options",
        ":litert_metrics",
        "//litert/runtime/dispatch:dispatch_delegate",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/delegates/utils:simple_opaque_delegate",
    ],
)

cc_library(
    name = "litert_options",
    srcs = ["litert_options.cc"],
    hdrs = [
        "litert_options.h",
    ],
    deps = [
        ":litert_common",
        ":litert_logging",
        ":litert_opaque_options",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_opaque_options",
        "//litert/core:options",
    ],
)

cc_test(
    name = "litert_options_test",
    srcs = ["litert_options_test.cc"],
    deps = [
        ":litert_common",
        ":litert_opaque_options",
        ":litert_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_compiled_model",
    srcs = ["litert_compiled_model.cc"],
    hdrs = [
        "litert_compiled_model.h",
    ],
    deps = [
        ":litert_common",
        ":litert_environment",
        ":litert_logging",
        ":litert_metrics",
        ":litert_model",
        ":litert_options",
        ":litert_tensor_buffer",
        "//litert/cc:litert_macros",
        "//litert/runtime:compiled_model",
        "//tflite/c:c_api_types",
    ],
)

cc_test(
    name = "litert_compiled_model_test",
    srcs = [
        "litert_compiled_model_test.cc",
    ],
    data = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_common",
        ":litert_compiled_model",
        ":litert_environment",
        ":litert_model",
        ":litert_options",
        ":litert_tensor_buffer",
        "//litert/test:common",
        "//litert/test:simple_model",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

# The same test as `litert_compiled_model_test` but using the shared library `libLiteRtRuntimeCApi.so`.
cc_test(
    name = "litert_compiled_model_shared_lib_test",
    srcs = [
        "litert_compiled_model_test.cc",
    ],
    data = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_runtime_c_api_shared_lib",
        "//litert/test:common",
        "//litert/test:simple_model",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_accelerator",
    srcs = ["litert_accelerator.cc"],
    hdrs = ["litert_accelerator.h"],
    deps = [
        ":litert_common",
        "//litert/cc:litert_expected",
        "//litert/core:environment",
        "//litert/runtime:accelerator",
    ],
)

cc_test(
    name = "litert_accelerator_test",
    srcs = ["litert_accelerator_test.cc"],
    deps = [
        ":litert_accelerator",
        ":litert_accelerator_registration",
        ":litert_common",
        ":litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/runtime:accelerator",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_opaque_options",
    srcs = ["litert_opaque_options.cc"],
    hdrs = ["litert_opaque_options.h"],
    deps = [":litert_common"],
)

cc_test(
    name = "litert_opaque_options_test",
    srcs = ["litert_opaque_options_test.cc"],
    deps = [
        ":litert_common",
        ":litert_opaque_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:version",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_accelerator_registration",
    srcs = ["litert_accelerator_registration.cc"],
    hdrs = ["litert_accelerator_registration.h"],
    deps = [
        ":litert_accelerator",
        ":litert_common",
        ":litert_metrics",
        ":litert_opaque_options",
        "//litert/cc:litert_expected",
        "//litert/core:environment",
        "//litert/runtime:accelerator",
    ],
)

cc_library(
    name = "litert_platform_support",
    hdrs = ["litert_platform_support.h"],
    deps = [":litert_common"],
)

cc_test(
    name = "litert_accelerator_registration_test",
    srcs = ["litert_accelerator_registration_test.cc"],
    deps = [
        ":litert_accelerator",
        ":litert_accelerator_registration",
        ":litert_common",
        ":litert_environment",
        ":litert_opaque_options",
        "//litert/runtime:accelerator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_profiler",
    srcs = ["litert_profiler.cc"],
    hdrs = [
        "litert_profiler.h",
    ],
    deps = [
        ":litert_common",
        ":litert_profiler_event",
        "//litert/cc:litert_macros",
        "//litert/runtime:profiler",
    ],
)

cc_library(
    name = "litert_profiler_event",
    hdrs = [
        "litert_profiler_event.h",
    ],
    deps = [
        "//tflite/core/api",
        "//tflite/profiling:memory_info",
    ],
)

cc_test(
    name = "litert_profiler_test",
    srcs = ["litert_profiler_test.cc"],
    deps = [
        ":litert_common",
        ":litert_profiler",
        ":litert_profiler_event",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "litert_model_srcs",
    srcs = ["litert_model.cc"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "litert_model_hdrs",
    srcs = ["litert_model.h"],
    visibility = ["//litert/core/model:__pkg__"],
)

# Collection of all C API targets.
LITERT_C_API_COMMON_DEPS = [
    ":litert_accelerator",
    ":litert_accelerator_registration",
    ":litert_any",
    ":litert_common",
    ":litert_compiled_model",
    ":litert_options",
    ":litert_dispatch_delegate",
    ":litert_event",
    ":litert_environment",
    ":litert_layout",
    ":litert_logging",
    ":litert_metrics",
    ":litert_model",
    ":litert_op_code",
    ":litert_op_options",
    ":litert_profiler",
    ":litert_tensor_buffer",
    "//litert/c/options:litert_cpu_options",
    "//litert/c/options:litert_qualcomm_options",
    "//litert/c/options:litert_google_tensor_options",
    "//litert/c/options:litert_mediatek_options",
    "//litert/c/options:litert_gpu_options",
    "//litert/c/options:litert_runtime_options",
    "//litert/c/options:litert_google_tensor_options_type",
]

# This test verifies that the C API header files can build via C compiler.
cc_test(
    name = "litert_c_api_common_test",
    srcs = ["litert_c_api_common_test.c"],
    copts = ["--std=c11"],
    linkopts = ["-ldl"],
    deps = LITERT_C_API_COMMON_DEPS + [":litert_profiler_event"],
)

# Build `litert/c:litert_runtime_c_api_so` for `libLiteRtRuntimeCApi.so`.
cc_shared_library(
    name = "litert_runtime_c_api_so",
    additional_linker_inputs = export_lrt_runtime_only_script(),
    shared_lib_name = "libLiteRtRuntimeCApi.so",
    user_link_flags = export_lrt_runtime_only_linkopt() + select({
        "@org_tensorflow//tensorflow:android": [
            "-landroid",
            "-Wl,-soname=libLiteRtRuntimeCApi.so",
        ],
        "@org_tensorflow//tensorflow:ios": [],
        "@org_tensorflow//tensorflow:macos": [],
        "//conditions:default": ["-Wl,-soname=libLiteRtRuntimeCApi.so"],
    }),
    deps = LITERT_C_API_COMMON_DEPS,
)

# Allows referring to the above as a target with the name :libLiteRtRuntimeCApi.so.
filegroup(
    name = "libLiteRtRuntimeCApi.so",
    srcs = [":litert_runtime_c_api_so"],
)

cc_library(
    name = "litert_dispatch_headers",
    hdrs = [
        ":litert_accelerator.h",
        ":litert_any.h",
        ":litert_common.h",
        ":litert_compiled_model.h",
        ":litert_custom_op_kernel.h",
        ":litert_environment.h",
        ":litert_environment_options.h",
        ":litert_event.h",
        ":litert_event_type.h",
        ":litert_gl_types.h",
        ":litert_layout.h",
        ":litert_logging.h",
        ":litert_metrics.h",
        ":litert_model.h",
        ":litert_op_code.h",
        ":litert_op_options.h",
        ":litert_opaque_options.h",
        ":litert_options.h",
        ":litert_platform_support.h",
        ":litert_tensor_buffer.h",
        ":litert_tensor_buffer_requirements.h",
        ":litert_tensor_buffer_types.h",
        # Needed for litert/c/litert_op_code.h
        "//tflite:builtin_ops.h",
        # Neeeded for litert/c/litert_model.h
        "//tflite/c:tensorflowlite_c_api_hdrs_filegroup",
        "//tflite/c:tensorflowlite_c_impl_hdrs_filegroup",
    ],  # Export all header files (.h) in this directory
    deps = [
        "//tflite/core/api",
        "//tflite/profiling:memory_info",
        "@opencl_headers",
    ],
)

# Config for LiteRT client executables to export LiteRt* symbols and let accelerator shared libs
# resolve the symbols, not from libLiteRtRuntimeCApi.so.
config_setting(
    name = "resolve_symbols_in_exec",
    define_values = {"resolve_symbols_in_exec": "true"},
)

# This is cc_library target based on `libLiteRtRuntimeCApi.so`.
cc_library(
    name = "litert_runtime_c_api_shared_lib",
    srcs = select({
        ":resolve_symbols_in_exec": [],
        "//conditions:default": [":litert_runtime_c_api_so"],
    }),
    hdrs = glob(["litert_*.h"]) + [
        "//litert/c/options:options_headers",
        "//tflite:builtin_ops.h",  # Needed for litert/c/litert_op_code.h
        "//tflite/c:tensorflowlite_c_api_hdrs_filegroup",  # Needed for litert/c/litert_model.h
        "//tflite/core/c:headers_filegroup",
    ],
    linkstatic = 1,
    deps = [
        # only depend on headers
        "@opencl_headers",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//tflite/core/api",
        "//tflite/profiling:memory_info",
        "//tflite/profiling:profile_buffer",
    ] + gles_headers(),
)

# copybara:uncomment_begin(google-only)
# # Check that litert runtime doesn't depend on MLIR.
# check_dependencies(
#     of = [":litert_runtime_c_api_shared_lib"],
#     dont_match_regexp = "^//third_party/llvm/llvm-project/mlir",
# )
# copybara:uncomment_end

exports_files(srcs = glob(["litert_*.h"]))
