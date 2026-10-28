# Copyright 2026 The LiteRT Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
cc_library(
    name = "devtool",
    linkstatic = True,
    srcs = ["devtool/tool.cc", "devtool/trace.cc"],
    hdrs = ["devtool/tool.h"],
    copts = ["-fno-exceptions"],
    linkopts = ["-ldl"],
    deps = [
        "//litert/tools:imagenet_eval_data",
        "//litert/tools:imagenet_eval_runner_lib",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_extended_model",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/vendors/arm_vulkan_ml/compiler:operation",
        "//litert/vendors/arm_vulkan_ml/compiler:partition",
        "//litert/vendors/arm_vulkan_ml/dispatch:vulkan_invocation",
        "//litert/vendors/arm_vulkan_ml/dispatch:tensor_capabilities",
        "@arm_vulkan_ml_tool_spirv_tools//:spirv_tools",
        "@com_google_absl//absl/strings",
        "//tflite:framework", "//tflite/kernels:reference_ops", "//tflite/schema:schema_fbs",
    ],
)
cc_binary(
    name = "litert_arm_vulkan_ml_tool",
    srcs = ["devtool/main.cc"],
    copts = ["-fno-exceptions"],
    deps = [":devtool"],
)

cc_test(
    name = "devtool_test",
    size = "small",
    linkstatic = True,
    srcs = ["devtool/tool_test.cc"],
    deps = [
        ":devtool",
        "//litert/vendors/arm_vulkan_ml/dispatch:tensor_capabilities",
        "@com_google_googletest//:gtest_main",
    ],
)
