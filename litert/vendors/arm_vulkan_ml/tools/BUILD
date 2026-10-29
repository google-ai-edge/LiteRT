# Copyright 2026 The LiteRT Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
cc_library(
    name = "devtool",
    linkstatic = True,
    srcs = ["devtool/tool.cc", "devtool/trace.cc", "devtool/tosa_query.cc", "devtool/coverage.cc", "devtool/vgf_export.cc"],
    hdrs = ["devtool/tool.h", ":coverage_manifest"],
    copts = ["-fno-exceptions"],
    linkopts = ["-ldl"],
    deps = [
        "@ai_ml_sdk_vgf_library//:vgf_encoder",
        "//litert/tools:imagenet_eval_data",
        "//litert/tools:imagenet_eval_runner_lib",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_extended_model",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/vendors/arm_vulkan_ml/compiler:operation",
        "//litert/vendors/arm_vulkan_ml/compiler:partition",
        "//litert/vendors/arm_vulkan_ml/dispatch:vulkan_invocation",
        "//litert/vendors/arm_vulkan_ml/dispatch:vulkan_validation",
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

cc_test(
    name = "tosa_query_test",
    size = "small",
    srcs = ["devtool/tosa_query_test.cc"],
    copts = ["-fno-exceptions"],
    linkstatic = True,
    deps = [":devtool", "@com_google_googletest//:gtest_main"],
)

genrule(
    name = "coverage_manifest",
    srcs = ["tosa_capabilities/implementation_coverage.json"],
    outs = ["coverage_manifest.inc"],
    cmd = "printf 'constexpr char kCoverageManifest[] = R\"coverage(' > $@; cat $(SRCS) >> $@; printf ')coverage\";\\n' >> $@",
)

cc_test(
    name = "vgf_export_test",
    size = "small",
    linkstatic = True,
    srcs = ["devtool/vgf_export_test.cc"],
    copts = ["-fno-exceptions"],
    deps = [":devtool", "@ai_ml_sdk_vgf_library//:vgf", "@com_google_googletest//:gtest_main"],
)
