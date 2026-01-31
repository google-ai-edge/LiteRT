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
load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment_begin(oss)
        # "//litert:litert_internal_users",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

internal_static_library_visibility = [
    # copybara:uncomment_begin(oss)
    # "//litert:litert_cc_users_static_link",
    # "//litert/vendors/examples:__subpackages__",
    # copybara:uncomment_end_and_comment_begin
    "//visibility:public",
    # copybara:comment_end
]

next_library_visibility = [
    # copybara:uncomment_begin(oss)
    # "//litert:litert_friends",
    # "//litert/runtime/dispatch:__subpackages__",
    # "//litert/ats:__subpackages__",
    # "//litert/integration_test:__subpackages__",
    # "//litert/js/packages:__subpackages__",  # TODO: b/459587509 - Remove this with CompiledModel migration.
    # copybara:uncomment_end_and_comment_begin
    "//visibility:public",
    # copybara:comment_end
]

# -----------------------------------------------------------------------------
# Runtime only internal C++ libraries.
# -----------------------------------------------------------------------------

cc_library(
    name = "litert_dispatch_delegate",
    hdrs = ["litert_dispatch_delegate.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/runtime/dispatch:dispatch_delegate",
        "//tflite/c:c_api",
        "//tflite/c:c_api_opaque",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/delegates/utils:simple_opaque_delegate",
    ],
)

# Same as :litert_tensor_buffer, but without the registry dependency.
# This is for accelerator shared libs not to depend on the registry, but depend on CAPI.so instead.
cc_library(
    name = "litert_tensor_buffer_without_registry",
    srcs = ["//litert/cc:litert_tensor_buffer.cc"],
    hdrs = [
        "//litert/cc:litert_tensor_buffer.h",
        "//litert/cc:litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_handle",
        "//litert/c:litert_common",
        "//litert/c:litert_custom_tensor_buffer",
        "//litert/c:litert_event",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_model_types",
        "//litert/c:litert_opencl_types",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c:litert_webgpu_types",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_event",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/cc:litert_tensor_buffer_types",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ],
)

cc_library(
    name = "litert_external_litert_buffer_context",
    hdrs = ["litert_external_litert_buffer_context.h"],
    deps = [
        ":litert_handle",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_external_litert_buffer_context",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_tensor_buffer",
        "//tflite/c:common",
    ],
)

# -----------------------------------------------------------------------------
# Common internal C++ libraries.
# -----------------------------------------------------------------------------

cc_library(
    name = "litert_c_types_printing",
    hdrs = ["litert_c_types_printing.h"],
    deps = [
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/cc/internal:litert_logging",
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
    name = "litert_consts",
    hdrs = ["litert_consts.h"],
)

cc_library(
    name = "litert_detail",
    hdrs = ["litert_detail.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "@com_google_absl//absl/log:absl_check",
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
    name = "litert_numerics",
    hdrs = ["litert_numerics.h"],
    deps = [
        ":litert_detail",
    ],
)

cc_library(
    name = "litert_platform_support",
    hdrs = ["litert_platform_support.h"],
    deps = ["//litert/c:litert_platform_support"],
)

cc_library(
    name = "litert_rng",
    srcs = ["litert_rng.cc"],
    hdrs = ["litert_rng.h"],
    deps = [
        ":litert_detail",
        ":litert_numerics",
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "litert_tensor_buffer_utils",
    srcs = ["litert_tensor_buffer_utils.cc"],
    hdrs = ["litert_tensor_buffer_utils.h"],
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_logging",
    ],
)

cc_library(
    name = "litert_tflite_error_status_builder",
    hdrs = ["litert_tflite_error_status_builder.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_macros",
        "//tflite/c:c_api_types",
    ],
)

cc_test(
    name = "litert_rng_test",
    srcs = ["litert_rng_test.cc"],
    deps = [
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/cc/internal:litert_numerics",
        "//litert/cc/internal:litert_rng",
        "//litert/test:matchers",
        "//litert/test:rng_fixture",
        "@com_google_absl//absl/strings:str_format",
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
    name = "litert_logging",
    hdrs = ["litert_logging.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
    ],
)

cc_test(
    name = "litert_logging_test",
    srcs = ["litert_logging_test.cc"],
    deps = [
        ":litert_logging",
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_shared_library",
    srcs = ["litert_shared_library.cc"],
    hdrs = ["litert_shared_library.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
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

# -----------------------------------------------------------------------------
# Internal static C++ libraries.
#
# There are shared version of these libraries available in cc/dynamic_runtime/.
# -----------------------------------------------------------------------------

cc_library(
    name = "litert_model_predicates",
    srcs = ["litert_model_predicates.cc"],
    hdrs = ["litert_model_predicates.h"],
    visibility = internal_static_library_visibility,
    deps = [
        ":litert_detail",
        ":litert_extended_model",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_ranked_tensor_type",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_model_predicates_test",
    srcs = ["litert_model_predicates_test.cc"],
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
    name = "litert_op_options",
    srcs = ["litert_op_options.cc"],
    hdrs = ["litert_op_options.h"],
    visibility = internal_static_library_visibility,
    deps = [
        "//litert/c:litert_builder",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/c:litert_op_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "litert_op_options_test",
    srcs = ["litert_op_options_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":litert_op_options",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/core/model",
        "//litert/core/model:buffer_manager",
        "//litert/core/util:flatbuffer_tools",
        "//litert/test:common",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_builder",
    srcs = ["litert_builder.cc"],
    hdrs = ["litert_builder.h"],
    visibility = internal_static_library_visibility,
    deps = [
        ":litert_detail",
        ":litert_extended_model",
        ":litert_handle",
        ":litert_op_options",
        "//litert/c:litert_builder",
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_ranked_tensor_type",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_builder_test",
    srcs = ["litert_builder_test.cc"],
    deps = [
        ":litert_builder",
        ":litert_extended_model",
        ":litert_op_options",
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_ranked_tensor_type",
        "//litert/core/model",
        "//litert/core/model:buffer_manager",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_extended_model",
    srcs = ["litert_extended_model.cc"],
    hdrs = ["litert_extended_model.h"],
    visibility = internal_static_library_visibility,
    deps = [
        ":litert_consts",
        ":litert_detail",
        ":litert_handle",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c:litert_model_types",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_ranked_tensor_type",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_extended_model_test",
    srcs = ["litert_extended_model_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":litert_extended_model",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_expected",
        "//litert/core/model",
        "//litert/core/util:flatbuffer_tools_testonly",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

exports_files(glob([
    "*.h",
    "*.cc",
]))

# -----------------------------------------------------------------------------
# New / experimental features.
# -----------------------------------------------------------------------------

# Advanced CompiledModel with new / experimental features.
cc_library(
    name = "litert_compiled_model_next",
    srcs = ["litert_compiled_model_next.cc"],
    hdrs = ["litert_compiled_model_next.h"],
    visibility = next_library_visibility,
    deps = [
        ":litert_handle",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_metrics",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_options",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_compiled_model_next_test",
    srcs = ["litert_compiled_model_next_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":litert_compiled_model_next",
        "//litert/c:litert_compiled_model",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_common",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_model",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc:litert_tensor_buffer_types",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "//tflite:framework",
        "//tflite/c:c_api_opaque",
        "//tflite/c:common",
        "//tflite/kernels:builtin_ops",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "scoped_file",
    srcs = ["scoped_file.cc"] + select({
        "@platforms//os:windows": ["scoped_file_win.cc"],
        "//conditions:default": ["scoped_file_posix.cc"],
    }),
    hdrs = ["scoped_file.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "scoped_weight_source",
    hdrs = ["scoped_weight_source.h"],
    deps = [
        ":scoped_file",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "scoped_file_test",
    srcs = ["scoped_file_test.cc"],
    deps = [
        ":scoped_file",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

# Do not use this target directly. It should only be used by Environment.
cc_library(
    name = "litert_runtime_builtin",
    srcs = ["litert_runtime_builtin.cc"],
    hdrs = ["litert_runtime_builtin.h"],
    visibility = [
        "//litert/cc:__pkg__",
    ],
    deps = [
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_compiled_model",
        "//litert/c:litert_custom_op_kernel",
        "//litert/c:litert_custom_tensor_buffer",
        "//litert/c:litert_environment_header",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_event",
        "//litert/c:litert_event_type",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_layout",
        "//litert/c:litert_metrics",
        "//litert/c:litert_model",
        "//litert/c:litert_model_types",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_options",
        "//litert/c:litert_profiler",
        "//litert/c:litert_tensor_buffer",
        "//litert/c/internal:litert_runtime_c_api",
    ],
)

# Do not use this target directly. It should only be used by Environment.
cc_library(
    name = "litert_runtime_builtin_runtimecapi",
    srcs = ["litert_runtime_builtin.cc"],
    hdrs = ["litert_runtime_builtin.h"],
    visibility = [
        "//litert/cc:__subpackages__",
    ],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_runtime_c_api_runtimecapi",
    ],
)

cc_library(
    name = "litert_matchers",
    hdrs = ["litert_matchers.h"],
    visibility = ["//litert:litert_public"],
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
    name = "litert_matchers_test",
    srcs = ["litert_matchers_test.cc"],
    deps = [
        ":litert_extended_model",
        ":litert_matchers",
        ":litert_op_options",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/core/model",
        "//litert/core/util:flatbuffer_tools_testonly",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_runtime_proxy",
    hdrs = ["litert_runtime_proxy.h"],
    deps = [
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_custom_op_kernel",
        "//litert/c:litert_custom_tensor_buffer",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_event_type",
        "//litert/c:litert_gl_types",
        "//litert/c:litert_layout",
        "//litert/c:litert_metrics",
        "//litert/c:litert_model_types",
        "//litert/c:litert_opaque_options",
        "//litert/c:litert_opencl_types",
        "//litert/c:litert_profiler_event",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c:litert_webgpu_types",
        "//litert/c/internal:litert_runtime_c_api",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

cc_library(
    name = "litert_runtime_proxy_runtimecapi",
    hdrs = ["litert_runtime_proxy.h"],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_runtime_c_api_runtimecapi",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:die_if_null",
    ],
)
