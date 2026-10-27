#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")
load("//litert/build_common:special_rule.bzl", "litert_android_linkopts")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "vulkan_invocation",
    srcs = ["vulkan_invocation.cc"],
    hdrs = ["vulkan_invocation.h"],
    linkopts = ["-ldl"],
    visibility = ["//litert:__subpackages__"],
    deps = [
        ":tensor_capabilities",
        "@com_google_absl//absl/synchronization",
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        "@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers",
    ],
)

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = ["dispatch_api.cc"],
    copts = [
        "-Os",
        "-fno-exceptions",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    export_litert_only = True,
    linkopts = litert_android_linkopts() + [
        "-Wl,-rpath,$$ORIGIN",
        "-Wl,-soname=libLiteRtDispatch_ArmVulkanML.so",
    ],
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_ArmVulkanML.so",
    visibility = ["//litert:litert_public"],
    deps = [
        "@com_google_absl//absl/synchronization",
        ":vulkan_invocation",
        "//litert/c/internal:litert_runtime_context",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_requirements",
        "//litert/c/internal:litert_scheduling_info",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)

cc_library(
    name = "tensor_capabilities",
    srcs = ["tensor_capabilities.cc"],
    copts = ["-fno-exceptions"],
    hdrs = ["tensor_capabilities.h"],
    deps = [
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        "@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers",
    ],
)

cc_test(
    name = "tensor_capabilities_test",
    srcs = ["tensor_capabilities_test.cc"],
    deps = [
        ":tensor_capabilities",
        "@com_google_googletest//:gtest_main",
    ],
)
