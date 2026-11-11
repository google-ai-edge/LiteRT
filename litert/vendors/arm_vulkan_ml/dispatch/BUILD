#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")
load("//litert/build_common:special_rule.bzl", "litert_android_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "vulkan_invocation",
    srcs = [
        "external_buffer.cc",
        "vulkan_invocation.cc",
    ],
    hdrs = [
        "device_memory_allocator.h",
        "external_buffer.h",
        "vulkan_invocation.h",
    ],
    linkopts = ["-ldl"] + select({
        "//litert:android": ["-landroid"],
        "//conditions:default": [],
    }),
    visibility = ["//litert:__subpackages__"],
    deps = [
        ":tensor_capabilities",
        ":vulkan_validation",
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        "@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = [
        "buffer_events.h",
        "dispatch_api.cc",
    ],
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
        ":vulkan_invocation",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_requirements",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_runtime_context",
        "//litert/c/internal:litert_scheduling_info",
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "tensor_capabilities",
    srcs = [
        "device_capabilities.cc",
        "tensor_capabilities.cc",
    ],
    hdrs = [
        "device_capabilities.h",
        "tensor_capabilities.h",
    ],
    copts = ["-fno-exceptions"],
    visibility = ["//litert:__subpackages__"],
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
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        "@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "vulkan_validation",
    srcs = ["vulkan_validation.cc"],
    hdrs = ["vulkan_validation.h"],
    copts = ["-fno-exceptions"],
    visibility = ["//litert:__subpackages__"],
    deps = [
        "@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "vulkan_validation_test",
    srcs = ["vulkan_validation_test.cc"],
    deps = [
        ":vulkan_validation",
        "@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "vulkan_invocation_test",
    srcs = ["vulkan_invocation_test.cc"],
    copts = ["-fno-exceptions"],
    data = ["//litert/test:tflite_test_data"],
    linkstatic = True,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        "noheapcheck",
    ],
    deps = [
        ":dispatch_api_test_source",
        ":vulkan_invocation",
        # copybara:uncomment_begin(no OSS for gunit_main_no_heapcheck)
        # "@com_google_googletest//:gtest_main_no_heapcheck",
        # copybara:uncomment_end_and_comment_begin
        "@com_google_googletest//:gtest_main",
        # copybara:comment_end
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@flatbuffers",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_requirements",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_compiler_context_header",
        "//litert/c/internal:litert_runtime_context_testonly",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/cc/internal:litert_extended_model",
        "//litert/cc/options:litert_arm_vulkan_ml_options",
        "//litert/core/model",
        "//litert/test:load_test_model",
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        "//litert/vendors/arm_vulkan_ml/compiler:compiler_plugin",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@arm_vulkan_ml_dep_spirv_headers//:spirv_cpp11_headers",
        "//tflite/schema:schema_fbs",
        "@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers",
    ],
)

# The resolver harness constructs an invocation with fake Vulkan entry points,
# then exercises the production dispatch interface and real LiteRT buffers.
cc_library(
    name = "dispatch_api_test_source",
    testonly = True,
    hdrs = ["buffer_events.h"],
    textual_hdrs = ["dispatch_api.cc"],
    deps = [
        ":vulkan_invocation",
        "//litert/c:litert_common",
        "//litert/c:litert_model_types",
        "//litert/c:litert_tensor_buffer_requirements",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/c/internal:litert_runtime_context",
        "//litert/c/internal:litert_scheduling_info",
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        "//litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "buffer_events_test",
    size = "small",
    srcs = [
        "buffer_events.h",
        "buffer_events_test.cc",
    ],
    copts = ["-fno-exceptions"],
    linkstatic = True,
    deps = [
        "//litert/c:litert_common",
        "//litert/c/internal:litert_runtime_context",
        "@com_google_googletest//:gtest_main",
    ],
)
