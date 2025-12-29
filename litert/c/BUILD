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

# copybara:uncomment_begin(google-only)
# load("@build_bazel_rules_apple//apple:macos.bzl", "macos_dylib", "macos_unit_test")
# copybara:uncomment_end_and_comment_begin
load("@build_bazel_rules_apple//apple:macos.bzl", "macos_dylib")
# copybara:comment_end

# copybara:uncomment load("//devtools/deps/check:deps_check.bzl", "check_dependencies")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# copybara:uncomment load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//litert/build_common:litert_build_defs.bzl",
    "copy_file",
    "export_lrt_runtime_only_linkopt",
    "export_lrt_runtime_only_script",
    "export_lrt_tflite_runtime_linkopt",
    "export_lrt_tflite_runtime_script",
)
load(
    "//litert/build_common:special_rule.bzl",
    "gles_deps",
    "gles_headers",
    "gles_linkopts",
    "litert_android_linkopts",
    "litert_metal_linkopts",
    "litert_metal_opts",
)
# copybara:uncomment load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")

LITERT_MIN_IOS_VERSION = "14.0"

LITERT_MIN_MACOS_VERSION = "14.0"

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment_begin(oss litert_lm)
        # "//litert:litert_c_users",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
)

cc_library(
    name = "litert_common",
    srcs = ["litert_common.cc"],
    hdrs = ["litert_common.h"],
    deps = ["//litert/build_common:build_config"],
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
    hdrs = ["litert_custom_op_kernel.h"],
    deps = [
        ":litert_common",
        ":litert_layout",
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
    # Keep symbols in libLiteRt.dylib.
    alwayslink = 1,
)

cc_library(
    name = "litert_environment_header",
    hdrs = ["litert_environment.h"],
    deps = [
        ":litert_common",
        ":litert_environment_options",
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
        "//litert/runtime/accelerators:auto_registration",
        "@com_google_absl//absl/types:span",
    ] + select({
        "//litert/build_common:build_include_gpu_enabled": [
            "//litert/runtime:gpu_environment",
        ],
        "//conditions:default": [],
    }),
    # Keep symbols in libLiteRt.dylib.
    alwayslink = 1,
)

cc_library(
    name = "litert_environment_options",
    srcs = ["litert_environment_options.cc"],
    hdrs = ["litert_environment_options.h"],
    deps = [
        ":litert_any",
        ":litert_common",
        "//litert/c:litert_custom_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
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
        "//litert/c:litert_custom_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
    ],
)

cc_test(
    name = "litert_environment_options_test",
    srcs = ["litert_environment_options_test.cc"],
    deps = [
        ":litert_any",
        ":litert_common",
        ":litert_environment",
        ":litert_environment_header",
        ":litert_environment_options",
        "//litert/core:environment_options",
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
    name = "litert_model_types",
    hdrs = ["litert_model_types.h"],
    deps = [
        ":litert_common",
        ":litert_layout",
        ":litert_op_code",
    ],
)

cc_library(
    name = "litert_model",
    srcs = ["litert_model.cc"],
    hdrs = ["litert_model.h"],
    deps = [
        ":litert_common",
        ":litert_layout",
        ":litert_model_types",
        ":litert_op_code",
        "//litert/build_common:build_config",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_macros",
        "//litert/core/model",
        "//litert/core/model:model_load",
        "//tflite/core/c:private_c_api_types",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ] + select({
        "//litert/build_common:build_include_npu_enabled": [
            "//litert/core/model:model_serialize",
        ],
        "//conditions:default": [],
    }),
)

# Flavor aliases for model (NPU serialization gated by flavor)
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
)

cc_library(
    name = "litert_op_options",
    srcs = ["litert_op_options.cc"],
    hdrs = ["litert_op_options.h"],
    deps = [
        ":litert_common",
        ":litert_layout",
        ":litert_op_code",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_macros",
        "//litert/core/model",
        "//tflite/c:c_api_types",
        "//tflite/converter/core:model_builder_base",
        "//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "litert_op_options_test",
    srcs = ["litert_op_options_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    tags = ["no_oss"],
    deps = [
        ":litert_common",
        ":litert_op_code",
        ":litert_op_options",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_model",
        "//litert/core/model",
        "//litert/core/model:buffer_manager",
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
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_macros",
        "//litert/runtime:event",
        "//tflite/delegates/gpu/cl:opencl_wrapper",
        "@opencl_headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "litert_tensor_buffer_types",
    hdrs = ["litert_tensor_buffer_types.h"],
)

cc_library(
    name = "litert_gl_types",
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
    copts = litert_metal_opts(),
    linkopts = litert_android_linkopts() + gles_linkopts() + litert_metal_linkopts(),
    deps = [
        ":litert_common",
        ":litert_custom_tensor_buffer",
        ":litert_event",
        ":litert_gl_types",
        ":litert_model_types",
        ":litert_tensor_buffer_types",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/runtime:tensor_buffer",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ] + gles_deps(),
    alwayslink = 1,
)

cc_test(
    name = "litert_tensor_buffer_test",
    srcs = ["litert_tensor_buffer_test.cc"],
    # require GPU to run OpenCL tests.
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":litert_any",
        ":litert_common",
        ":litert_environment",
        ":litert_environment_options",
        ":litert_platform_support",
        ":litert_tensor_buffer",
        ":litert_tensor_buffer_types",
        "//litert/cc:litert_any",
        "//litert/cc:litert_layout",
        "//litert/runtime:event",
        "//litert/test:matchers",
        "//tflite/delegates/gpu/cl:opencl_wrapper",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "litert_tensor_buffer_requirements_test",
    srcs = ["litert_tensor_buffer_requirements_test.cc"],
    linkopts = litert_android_linkopts(),
    deps = [
        ":litert_common",
        ":litert_tensor_buffer",
        ":litert_tensor_buffer_types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_options",
    srcs = ["litert_options.cc"],
    hdrs = ["litert_options.h"],
    deps = [
        ":litert_common",
        ":litert_opaque_options",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_handle",
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
    hdrs = ["litert_compiled_model.h"],
    deps = [
        ":litert_common",
        ":litert_environment",
        ":litert_layout",
        ":litert_metrics",
        ":litert_model",
        ":litert_options",
        ":litert_tensor_buffer",
        "//litert/c/internal:litert_logging",
        "//litert/cc:litert_macros",
        "//litert/runtime:compiled_model",
        "//tflite/c:c_api_types",
        "@com_google_absl//absl/types:span",
    ],
    # Keep symbols in libLiteRt.dylib.
    alwayslink = 1,
)

cc_test(
    name = "litert_compiled_model_test",
    srcs = ["litert_compiled_model_test.cc"],
    data = [
        "//litert/test:testdata/dynamic_add_model.tflite",
        "//litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_common",
        ":litert_compiled_model",
        ":litert_environment",
        ":litert_environment_options",
        ":litert_layout",
        ":litert_model",
        ":litert_options",
        ":litert_tensor_buffer",
        "//litert/c/internal:litert_logging",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

# The same test as `litert_compiled_model_test` but using the shared library `libLiteRt.so`.
cc_test(
    name = "litert_compiled_model_shared_lib_test",
    srcs = ["litert_compiled_model_test.cc"],
    data = [
        "//litert/test:testdata/dynamic_add_model.tflite",
        "//litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_environment_options",
        ":litert_runtime_c_api_shared_lib",
        "//litert/c/internal:litert_logging",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_model",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_opaque_options",
    srcs = ["litert_opaque_options.cc"],
    hdrs = ["litert_opaque_options.h"],
    deps = [":litert_common"],
    alwayslink = 1,
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
    name = "litert_custom_tensor_buffer",
    hdrs = ["litert_custom_tensor_buffer.h"],
    deps = [
        ":litert_common",
        ":litert_model_types",
        ":litert_tensor_buffer_types",
    ],
)

cc_library(
    name = "litert_platform_support",
    hdrs = ["litert_platform_support.h"],
    deps = [":litert_common"],
)

cc_library(
    name = "litert_profiler",
    srcs = ["litert_profiler.cc"],
    hdrs = ["litert_profiler.h"],
    deps = [
        ":litert_common",
        ":litert_profiler_event",
        "//litert/cc:litert_macros",
        "//litert/runtime:profiler",
    ],
)

cc_library(
    name = "litert_profiler_event",
    hdrs = ["litert_profiler_event.h"],
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

cc_library(
    name = "litert_builder",
    srcs = ["litert_builder.cc"],
    hdrs = ["litert_builder.h"],
    deps = [
        ":litert_common",
        ":litert_model",
        ":litert_op_code",
        "//litert/core/model",
        "//litert/core/model:buffer_manager",
        "//litert/core/util:flatbuffer_tools",
        "//tflite/converter/schema:schema_fbs",
    ],
)

cc_test(
    name = "litert_builder_test",
    srcs = ["litert_builder_test.cc"],
    deps = [
        ":litert_builder",
        ":litert_common",
        ":litert_model",
        ":litert_op_code",
        ":litert_op_options",
        "//litert/cc:litert_buffer_ref",
        "//litert/core/model",
        "//litert/core/model:buffer_manager",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "litert_model_hdrs",
    srcs = [
        "litert_model.h",
        "litert_model_types.h",
    ],
    visibility = ["//litert/core/model:__pkg__"],
)

# Collection of all C API targets.
LITERT_C_API_COMMON_DEPS = [
    ":litert_any",
    ":litert_common",
    ":litert_compiled_model",
    ":litert_custom_tensor_buffer",
    ":litert_options",
    ":litert_event",
    ":litert_environment",
    ":litert_layout",
    ":litert_metrics",
    ":litert_model",
    ":litert_model_types",
    ":litert_op_code",
    ":litert_op_options",
    ":litert_opaque_options",
    ":litert_profiler",
    ":litert_builder",
    ":litert_tensor_buffer",
    "//litert/c/internal:litert_accelerator_registration",
    "//litert/c/internal:litert_accelerator",
    "//litert/c/internal:litert_dispatch_delegate",
    "//litert/c/internal:litert_external_litert_buffer_context",
    "//litert/c/internal:litert_logging",
    "//litert/c/internal:litert_tensor_buffer_registry",
    "//litert/c/options:litert_cpu_options",
    "//litert/c/options:litert_darwinn_runtime_options",
    "//litert/c/options:litert_google_tensor_options_type",
    "//litert/c/options:litert_google_tensor_options",
    "//litert/c/options:litert_gpu_options",
    "//litert/c/options:litert_mediatek_options",
    "//litert/c/options:litert_qualcomm_options",
    "//litert/c/options:litert_intel_openvino_options",
    "//litert/c/options:litert_runtime_options",
]

# This test verifies that the C API header files can build via C compiler.
cc_test(
    name = "litert_c_api_common_test",
    srcs = ["litert_c_api_common_test.c"],
    copts = ["--std=c11"],
    linkopts = ["-ldl"],
    deps = LITERT_C_API_COMMON_DEPS + [":litert_profiler_event"],
)

# TODO: b/437381008
cc_library(
    name = "litert_runtime_c_api_so_shim",
    linkopts = select({
        "@platforms//os:macos": [],
        "@platforms//os:ios": [],
        "//conditions:default": ["-Wl,--undefined-glob=LiteRt*"],
    }),
    deps = LITERT_C_API_COMMON_DEPS,
)

# Build `litert/c:litert_runtime_c_api_so` for `libLiteRt.so`.
cc_shared_library(
    name = "litert_runtime_c_api_so",
    additional_linker_inputs = export_lrt_runtime_only_script(),
    shared_lib_name = "libLiteRt.so",
    user_link_flags = export_lrt_runtime_only_linkopt() + select({
        "@platforms//os:macos": ["-Wl,-install_name,@rpath/libLiteRt.so"],
        "//conditions:default": ["-Wl,-soname=libLiteRt.so"],
    }) + litert_android_linkopts(),
    deps = [":litert_runtime_c_api_so_shim"],
)

macos_dylib(
    name = "build_litert_runtime_c_api_dylib",
    exported_symbols_lists = export_lrt_runtime_only_script(),
    linkopts = ["-install_name @rpath/libLiteRt.dylib"],
    minimum_os_version = LITERT_MIN_MACOS_VERSION,
    deps = LITERT_C_API_COMMON_DEPS,
)

copy_file(
    name = "litert_runtime_c_api_dylib",
    src = ":build_litert_runtime_c_api_dylib",
    target = "libLiteRt.dylib",
)

# Target for windows libLiteRt.dll.
cc_binary(
    name = "libLiteRt",
    linkshared = 1,
    tags = [
        "nobuilder",
        "notap",
        "windows",
    ],
    target_compatible_with = ["@platforms//os:windows"],
    visibility = ["//visibility:private"],
    win_def_file = ":windows_exported_symbols.def",
    deps = LITERT_C_API_COMMON_DEPS,
)

alias(
    name = "litert_runtime_c_api_dll",
    actual = ":libLiteRt",
)

filegroup(
    name = "litert_runtime_c_api_dll_interface_lib",
    srcs = [":litert_runtime_c_api_dll"],
    output_group = "interface_library",
)

# Import library to link libLiteRt.dll.
cc_import(
    name = "litert_runtime_c_api_dll_import",
    interface_library = ":litert_runtime_c_api_dll_interface_lib",
    shared_library = ":litert_runtime_c_api_dll",
    target_compatible_with = ["@platforms//os:windows"],
)

genrule(
    name = "litert_runtime_c_api_exe_interface_lib",
    srcs = [":windows_exported_symbols.def"],
    outs = ["litert_runtime_c_api_exe_import.lib"],
    # Prepend LIBRARY name before building the import library.
    cmd = "echo 'LIBRARY litert_lm_main.exe' > $(RULEDIR)/def.tmp; " +
          "cat $(SRCS) >> $(RULEDIR)/def.tmp; " +
          "$(AR) /def:$(RULEDIR)/def.tmp /out:$(OUTS) /machine:x64; " +
          "rm $(RULEDIR)/def.tmp",
    tags = ["manual"],
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
)

# Import library to resolve symbols in executables.
cc_import(
    name = "litert_runtime_c_api_exe_import",
    interface_library = ":litert_runtime_c_api_exe_interface_lib",
    system_provided = True,
    target_compatible_with = ["@platforms//os:windows"],
)

alias(
    name = "litert_runtime_c_api_import",
    actual = select({
        ":resolve_symbols_in_exec": ":litert_runtime_c_api_exe_import",
        "//conditions:default": ":litert_runtime_c_api_dll_import",
    }),
)

exports_files(srcs = ["windows_exported_symbols.def"])

# TODO: b/437381008
cc_library(
    name = "litert_tflite_runtime_c_api_so_shim",
    linkopts = select({
        "@platforms//os:macos": [],
        "@platforms//os:ios": [],
        "//conditions:default": ["-Wl,--undefined-glob=LiteRt*"],
    }),
    deps = LITERT_C_API_COMMON_DEPS + [
        "//tflite/c:c_api",
        "//tflite/c:c_api_experimental",
    ],
)

# Build `litert/c:litert_tflite_runtime_c_api_so` for `libLiteRt.so`.
# Warning:Don't build it together with `litert_runtime_c_api_so` in one command.
# When bazel analyzed the build graph, it detected that two different actions would
# produce the same output file path. This violates bazel's fundamental principle that
# each output file must have exactly one producer action. Hence the error:
cc_shared_library(
    name = "litert_tflite_runtime_c_api_so",
    additional_linker_inputs = export_lrt_tflite_runtime_script(),
    shared_lib_name = "libLiteRt.so",
    tags = ["manual"],  # To prevent build together with `litert_runtime_c_api_so`.
    user_link_flags = export_lrt_tflite_runtime_linkopt() + litert_android_linkopts(),
    deps = [":litert_tflite_runtime_c_api_so_shim"],
)

SELECT_LITERT_RUNTIME_C_API_SO = select({
    ":resolve_symbols_in_exec": [],
    "@platforms//os:ios": [":litert_runtime_c_api_dylib"],
    ":litert_macos_wheel_use_so": [":litert_runtime_c_api_so"],
    ":litert_macos_default": [":litert_runtime_c_api_dylib"],
    "@platforms//os:windows": [],
    "//conditions:default": [":litert_runtime_c_api_so"],
})

# Allows referring to the above as a target with the name :libLiteRt.so.
filegroup(
    name = "libLiteRt.so",
    srcs = SELECT_LITERT_RUNTIME_C_API_SO,
    visibility = ["//litert:litert_public"],
)

# Config for LiteRT client executables to export LiteRt* symbols and let accelerator shared libs
# resolve the symbols, not from libLiteRt.so.
config_setting(
    name = "resolve_symbols_in_exec",
    define_values = {"resolve_symbols_in_exec": "true"},
)

config_setting(
    name = "litert_macos_wheel_use_so",
    constraint_values = ["@platforms//os:macos"],
    define_values = {"litert_macos_wheel_use_so": "true"},
)

config_setting(
    name = "litert_macos_default",
    constraint_values = ["@platforms//os:macos"],
    define_values = {"litert_macos_wheel_use_so": "false"},
)

filegroup(
    name = "c_api_headers",
    srcs = glob(["litert_*.h"]),
    visibility = ["//litert/cc:__pkg__"],
)

# This is cc_library target based on `libLiteRt.so`.
cc_library(
    name = "litert_runtime_c_api_shared_lib",
    srcs = SELECT_LITERT_RUNTIME_C_API_SO,
    hdrs = [
        ":c_api_headers",
        "//litert/c/internal:internal_headers",
        "//litert/c/options:options_headers",
        "//tflite:builtin_ops.h",  # Needed for litert/c/litert_op_code.h
        "//tflite/c:tensorflowlite_c_api_hdrs_filegroup",  # Needed for litert/c/litert_model.h
        "//tflite/core/c:headers_filegroup",
    ],
    linkopts = select({
        "@platforms//os:ios": ["-Wl,-rpath,@loader_path"],
        "@platforms//os:macos": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    deps = [
        # only depend on headers
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//tflite/core/api",
        "//tflite/profiling:memory_info",
        "//tflite/profiling:profile_buffer",
        "//litert/build_common:build_config",
    ] + gles_headers() + select({
        ":resolve_symbols_in_exec": [],
        "@platforms//os:windows": [":litert_runtime_c_api_import"],
        "//conditions:default": [],
    }),
)

# copybara:uncomment_begin(google-only)
# # Check that litert runtime doesn't depend on MLIR.
# check_dependencies(
#     of = [":litert_runtime_c_api_shared_lib"],
#     dont_match_regexp = "^//third_party/llvm/llvm-project/mlir",
# )
# copybara:uncomment_end

exports_files(srcs = glob(["litert_*.h"]))

# copybara:uncomment_begin(google-only)
# objc_library(
#     name = "litert_tensor_buffer_test_library",
#     testonly = True,
#     srcs = ["litert_tensor_buffer_test.mm"],
#     deps = [
#         ":litert_any",
#         ":litert_common",
#         ":litert_environment",
#         ":litert_environment_options",
#         ":litert_model",
#         ":litert_tensor_buffer",
#         ":litert_tensor_buffer_types",
#         "//third_party/apple_frameworks:XCTest",
#         "//litert/cc:litert_any",
#         "//litert/cc:litert_layout",
#         "//litert/runtime/accelerators/gpu:ml_drift_metal_accelerator",
#     ],
# )
#
# macos_unit_test(
#     name = "litert_tensor_buffer_test_macos",
#     minimum_os_version = LITERT_MIN_MACOS_VERSION,
#     tags = ["notap"],
#     deps = [":litert_tensor_buffer_test_library"],
# )
#
# ios_unit_test(
#     name = "litert_tensor_buffer_test_ios",
#     minimum_os_version = LITERT_MIN_IOS_VERSION,
#     runner = "@build_bazel_rules_apple//apple/testing/default_runner:ios_default_runner",
#     deps = [":litert_tensor_buffer_test_library"],
# )
# copybara:uncomment_end(google-only)
