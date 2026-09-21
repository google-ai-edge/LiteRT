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
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "mock_plugin_common",
    hdrs = ["mock_plugin_common.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/c/internal:litert_abi_header",
        "//litert/c/internal:litert_compiler_context_header",
        "//litert/cc:litert_macros",
        "//litert/compiler/cc:litert_model",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/c:litert_compiler_plugin_api",
        "//litert/vendors/examples:example_transformations",
    ],
)

cc_library(
    name = "mock_dispatch_common",
    hdrs = ["mock_dispatch_common.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c/internal:litert_runtime_context_header",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

# ------------------------------------------------------------------------------
# Mock Compiler Plugins
# ------------------------------------------------------------------------------

litert_dynamic_lib(
    name = "older_supported_plugin",
    srcs = ["older_supported_plugin.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "older_supported_plugin_so",
    so_name = "libLiteRtCompilerPlugin_OlderSupported.so",
    visibility = ["//litert:litert_public"],
    deps = [
        ":mock_plugin_common",
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_compiler_plugin_api",
    ],
)

litert_dynamic_lib(
    name = "older_unsupported_plugin",
    srcs = ["older_unsupported_plugin.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "older_unsupported_plugin_so",
    so_name = "libLiteRtCompilerPlugin_OlderUnsupported.so",
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_compiler_plugin_api",
    ],
)

litert_dynamic_lib(
    name = "newer_supported_plugin",
    srcs = ["newer_supported_plugin.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "newer_supported_plugin_so",
    so_name = "libLiteRtCompilerPlugin_NewerSupported.so",
    visibility = ["//litert:litert_public"],
    deps = [
        ":mock_plugin_common",
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_compiler_plugin_api",
    ],
)

litert_dynamic_lib(
    name = "newer_unsupported_plugin",
    srcs = ["newer_unsupported_plugin.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "newer_unsupported_plugin_so",
    so_name = "libLiteRtCompilerPlugin_NewerUnsupported.so",
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_compiler_plugin_api",
    ],
)

litert_dynamic_lib(
    name = "same_supported_plugin",
    srcs = ["same_supported_plugin.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "same_supported_plugin_so",
    so_name = "libLiteRtCompilerPlugin_SameSupported.so",
    visibility = ["//litert:litert_public"],
    deps = [
        ":mock_plugin_common",
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_compiler_plugin_api",
    ],
)

litert_dynamic_lib(
    name = "null_interface_plugin",
    srcs = ["null_interface_plugin.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "null_interface_plugin_so",
    so_name = "libLiteRtCompilerPlugin_NullInterface.so",
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_compiler_plugin_api",
    ],
)

litert_dynamic_lib(
    name = "incompatible_plugin",
    srcs = ["incompatible_plugin.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "incompatible_plugin_so",
    so_name = "libLiteRtCompilerPlugin_Incompatible.so",
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_compiler_plugin_api",
    ],
)

litert_dynamic_lib(
    name = "corrupted_header_plugin",
    srcs = ["corrupted_header_plugin.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "corrupted_header_plugin_so",
    so_name = "libLiteRtCompilerPlugin_CorruptedHeader.so",
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_compiler_plugin_api",
    ],
)

filegroup(
    name = "compiler_plugin_compatibility_so_data",
    srcs = [
        ":corrupted_header_plugin_so",
        ":incompatible_plugin_so",
        ":newer_supported_plugin_so",
        ":newer_unsupported_plugin_so",
        ":null_interface_plugin_so",
        ":older_supported_plugin_so",
        ":older_unsupported_plugin_so",
        ":same_supported_plugin_so",
    ],
    visibility = ["//litert:litert_public"],
)

# ------------------------------------------------------------------------------
# Mock Dispatch Libraries
# ------------------------------------------------------------------------------

litert_dynamic_lib(
    name = "older_supported_dispatch",
    srcs = ["older_supported_dispatch.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "older_supported_dispatch_so",
    so_name = "libLiteRtDispatch_OlderSupported.so",
    visibility = ["//litert:litert_public"],
    deps = [
        ":mock_dispatch_common",
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)

litert_dynamic_lib(
    name = "older_unsupported_dispatch",
    srcs = ["older_unsupported_dispatch.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "older_unsupported_dispatch_so",
    so_name = "libLiteRtDispatch_OlderUnsupported.so",
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)

litert_dynamic_lib(
    name = "newer_supported_dispatch",
    srcs = ["newer_supported_dispatch.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "newer_supported_dispatch_so",
    so_name = "libLiteRtDispatch_NewerSupported.so",
    visibility = ["//litert:litert_public"],
    deps = [
        ":mock_dispatch_common",
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)

litert_dynamic_lib(
    name = "newer_unsupported_dispatch",
    srcs = ["newer_unsupported_dispatch.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "newer_unsupported_dispatch_so",
    so_name = "libLiteRtDispatch_NewerUnsupported.so",
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)

litert_dynamic_lib(
    name = "same_supported_dispatch",
    srcs = ["same_supported_dispatch.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "same_supported_dispatch_so",
    so_name = "libLiteRtDispatch_SameSupported.so",
    visibility = ["//litert:litert_public"],
    deps = [
        ":mock_dispatch_common",
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)

litert_dynamic_lib(
    name = "null_interface_dispatch",
    srcs = ["null_interface_dispatch.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "null_interface_dispatch_so",
    so_name = "libLiteRtDispatch_NullInterface.so",
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)

litert_dynamic_lib(
    name = "incompatible_dispatch",
    srcs = ["incompatible_dispatch.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "incompatible_dispatch_so",
    so_name = "libLiteRtDispatch_Incompatible.so",
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)

litert_dynamic_lib(
    name = "corrupted_header_dispatch",
    srcs = ["corrupted_header_dispatch.cc"],
    export_litert_only = True,
    linkopts = select({
        "//litert:windows": [],
        "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
    }),
    linkstatic = 1,
    shared_lib_name = "corrupted_header_dispatch_so",
    so_name = "libLiteRtDispatch_CorruptedHeader.so",
    visibility = ["//litert:litert_public"],
    deps = [
        "//litert/c:litert_common",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)

filegroup(
    name = "dispatch_compatibility_so_data",
    srcs = [
        ":corrupted_header_dispatch_so",
        ":incompatible_dispatch_so",
        ":newer_supported_dispatch_so",
        ":newer_unsupported_dispatch_so",
        ":null_interface_dispatch_so",
        ":older_supported_dispatch_so",
        ":older_unsupported_dispatch_so",
        ":same_supported_dispatch_so",
    ],
    visibility = ["//litert:litert_public"],
)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

cc_test(
    name = "compiler_plugin_compatibility_test",
    srcs = ["compiler_plugin_compatibility_test.cc"],
    data = [
        ":compiler_plugin_compatibility_so_data",
        "//litert/test:mlir_test_data",
    ],
    target_compatible_with = select({
        "//litert:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/core:filesystem",
        "//litert/test:common",
        "//litert/test:load_test_model",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dispatch_compatibility_test",
    srcs = ["dispatch_compatibility_test.cc"],
    data = [
        ":dispatch_compatibility_so_data",
    ],
    target_compatible_with = select({
        "//litert:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_options",
        "//litert/c/internal:litert_runtime_context_testonly",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_macros",
        "//litert/core:filesystem",
        "//litert/runtime/dispatch",
        "//litert/runtime/dispatch:dispatch_test_helper",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/test:simple_buffer",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
