#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

cc_library(
    name = "quantization",
    srcs = ["quantization.cc"],
    hdrs = ["quantization.h"],
)

cc_library(
    name = "conv2d",
    srcs = ["conv2d.cc"],
    hdrs = ["conv2d.h"],
    deps = [":quantization", "@tosa_for_spirv_codegen"],
)

cc_library(
    name = "conv2d_adapter",
    srcs = ["conv2d_adapter.cc"],
    hdrs = ["conv2d_adapter.h"],
    deps = [
        ":conv2d",
        "//litert/c:litert_op_options",
        "//litert/cc/internal:litert_extended_model",
    ],
)

cc_library(
    name = "operation",
    srcs = ["operation.cc"],
    hdrs = ["operation.h"],
    deps = [
        ":conv2d", ":conv2d_adapter", ":quantization",
        "//litert/c:litert_op_options",
        "//litert/cc/internal:litert_extended_model",
        "@tosa_for_spirv_codegen",
    ],
)

cc_library(
    name = "partition",
    srcs = ["partition.cc"],
    hdrs = ["partition.h"],
    deps = [
        ":conv2d", ":operation",
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        "//litert/cc/internal:litert_extended_model",
        "@tosa_for_spirv_codegen",
    ],
)

cc_binary(
    name = "conv2d_fixture",
    testonly = True,
    srcs = ["conv2d_fixture.cc"],
    deps = [
        "//litert/vendors/arm_vulkan_ml/dispatch:vulkan_invocation",
        ":conv2d_adapter",
        ":conv2d",
        "//litert/cc/internal:litert_extended_model",
        "//tflite:framework",
        "//tflite/kernels:reference_ops",
        "//tflite/schema:schema_fbs",
        "@tosa_for_spirv_codegen",
    ],
)

cc_test(
    name = "quantization_test",
    srcs = ["quantization_test.cc"],
    deps = [":quantization", "@com_google_googletest//:gtest_main"],
)

cc_test(
    name = "conv2d_test",
    srcs = ["conv2d_test.cc"],
    deps = [":conv2d", "@tosa_for_spirv_codegen", "@com_google_googletest//:gtest_main"],
)

litert_dynamic_lib(
    name = "compiler_plugin",
    srcs = ["compiler_plugin.cc"],
    hdrs = ["//litert/vendors/c:litert_compiler_plugin.h"],
    copts = [
        "-Os",
        "-fno-exceptions",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    export_litert_only = True,
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/../../..",
        "-Wl,--no-undefined",
    ],
    linkstatic = 1,
    shared_lib_name = "compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_ArmVulkanML.so",
    ungrte = True,
    visibility = [
        "//ci/tools/python/wheel:__subpackages__",
        "//litert:__subpackages__",
    ],
    deps = [
        "//litert/c:litert_builder_header",
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/c/internal:litert_compiler_context",
        "//litert/c/internal:litert_logging",
        "//litert/c/options:litert_arm_vulkan_ml_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_context_wrapper",
        "//litert/cc/internal:litert_extended_model",
        "//litert/cc/internal:litert_handle",
        "//litert/cc/internal:litert_opaque_options_wrapper",
        "//litert/cc/internal:litert_options_wrapper",
        "//litert/cc/options:litert_arm_vulkan_ml_options",
        "//litert/compiler/cc:litert_model",
        ":operation",
        ":partition",
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        "//litert/vendors/c:litert_compiler_plugin_api",
        # copybara:comment_begin(oss-only)
        # copybara:comment_end
    ],
)

sh_test(
    name = "arm_vulkan_ml_compiler_plugin_so_symbol_test",
    srcs = ["//litert/test:litert_compiler_plugin_symbol_test.sh"],
    args = [
        "$(rootpath :compiler_plugin_so)",
    ],
    data = [
        ":compiler_plugin_so",
    ],
)

cc_test(
    name = "compiler_plugin_test",
    srcs = ["compiler_plugin_test.cc"],
    data = ["//litert/test:tflite_test_data"],
    linkstatic = True,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        "noheapcheck",
    ],
    deps = [
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        ":compiler_plugin",  # buildcleaner: keep
        # copybara:uncomment_begin(no OSS for gunit_main_no_heapcheck)
        # "@com_google_googletest//:gtest_main_no_heapcheck",
        # copybara:uncomment_end_and_comment_begin
        "@com_google_googletest//:gtest_main",
        # copybara:comment_end
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/c/internal:litert_compiler_context_testonly",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_arm_vulkan_ml_options",
        "//litert/core/model",
        "//litert/test:load_test_model",
        "//litert/test:matchers",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/cc:litert_compiler_plugin",
    ],
)

cc_test(
    name = "quantization_reference_test",
    srcs = ["quantization_reference_test.cc"],
    deps = [
        ":quantization",
        "//tflite/kernels/internal:reference_base",
        "//tflite/kernels/internal:quantization_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "operation_test",
    srcs = ["operation_test.cc"],
    deps = [
        ":operation",
        ":partition",
        "//litert/cc/internal:litert_extended_model",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)
