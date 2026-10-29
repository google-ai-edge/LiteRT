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
    srcs = ["vulkan_invocation.cc", "external_buffer.cc"],
    hdrs = ["vulkan_invocation.h", "device_memory_allocator.h", "external_buffer.h"],
    linkopts = ["-ldl"] + select({"//litert:android": ["-landroid"], "//conditions:default": []}),
    visibility = ["//litert:__subpackages__"],
    deps = [
        ":tensor_capabilities",
        ":vulkan_validation",
        "@com_google_absl//absl/synchronization",
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        "@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers",
    ],
)

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = ["dispatch_api.cc", "buffer_events.h"],
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
    visibility = ["//litert:__subpackages__"],
    srcs = ["tensor_capabilities.cc", "device_capabilities.cc"],
    copts = ["-fno-exceptions"],
    hdrs = ["tensor_capabilities.h", "device_capabilities.h"],
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

cc_library(
    name = "vulkan_validation",
    srcs = ["vulkan_validation.cc"],
    hdrs = ["vulkan_validation.h"],
    copts = ["-fno-exceptions"],
    visibility = ["//litert:__subpackages__"],
    deps = ["@com_google_absl//absl/synchronization", "@com_github_nlohmann_json//:json", "@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers"],
)

cc_test(
    name = "vulkan_validation_test",
    srcs = ["vulkan_validation_test.cc"],
    deps = [":vulkan_validation", "@com_google_googletest//:gtest_main"],
)

cc_test(
    name = "vulkan_invocation_test",
    data = ["//litert/test:tflite_test_data"],
    srcs = ["vulkan_invocation_test.cc"],
    copts = ["-fno-exceptions"],
    linkstatic = True,
    deps = [
        ":dispatch_api_test_source",
        ":vulkan_invocation",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_requirements",
        "//litert/c/internal:litert_runtime_context_testonly",
        "//litert/vendors/arm_vulkan_ml/compiler:compiler_plugin",
        "//litert/test:load_test_model",
        "//litert/cc/internal:litert_extended_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_arm_vulkan_ml_options",
        "//litert/core/model",
        "//litert/vendors/cc:litert_compiler_plugin",
        "//tflite/schema:schema_fbs",
        "@arm_vulkan_ml_dep_spirv_headers//:spirv_cpp_headers",
        "@com_google_googletest//:gtest_main",
    ],
)

# The resolver harness constructs an invocation with fake Vulkan entry points,
# then exercises the production dispatch interface and real LiteRT buffers.
cc_library(
    name = "dispatch_api_test_source",
    testonly = True,
    textual_hdrs = ["dispatch_api.cc"],
    hdrs = ["buffer_events.h"],
    deps = [
        ":vulkan_invocation",
        "@com_google_absl//absl/synchronization",
        "//litert/c/internal:litert_runtime_context",
        "//litert/c/internal:litert_scheduling_info",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_requirements",
        "//litert/vendors/c:litert_dispatch_c_api",
    ],
)

cc_test(
    name = "buffer_events_test",
    linkstatic = True,
    size = "small",
    srcs = ["buffer_events_test.cc", "buffer_events.h"],
    copts = ["-fno-exceptions"],
    deps = ["//litert/c/internal:litert_runtime_context", "@com_google_googletest//:gtest_main"],
)
