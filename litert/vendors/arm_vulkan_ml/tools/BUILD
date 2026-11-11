# Copyright 2026 The LiteRT Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# copybara:uncomment_begin(google-only)
# package(
#     default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
# )
# copybara:uncomment_end

cc_library(
    name = "devtool",
    srcs = [
        "devtool/coverage.cc",
        "devtool/quantized_reference.cc",
        "devtool/softmax_reference.cc",
        "devtool/tool.cc",
        "devtool/tosa_query.cc",
        "devtool/trace.cc",
        "devtool/vgf_export.cc",
    ],
    hdrs = [
        "devtool/tool.h",
        ":coverage_manifest",
    ],
    copts = ["-fno-exceptions"],
    linkopts = ["-ldl"],
    linkstatic = True,
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_environment_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc/internal:litert_extended_model",
        "//litert/cc/options:litert_arm_vulkan_ml_options",
        "//litert/compiler/plugin:compiler_plugin",
        "//litert/core/model",
        "//litert/core/model:model_serialize",
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        "//litert/vendors/arm_vulkan_ml/compiler:graph_ir",
        "//litert/vendors/arm_vulkan_ml/compiler:op_registry",
        "//litert/vendors/arm_vulkan_ml/compiler:operation",
        "//litert/vendors/arm_vulkan_ml/compiler:partition",
        "//litert/vendors/arm_vulkan_ml/compiler:softmax",
        "//litert/vendors/arm_vulkan_ml/dispatch:tensor_capabilities",
        "//litert/vendors/arm_vulkan_ml/dispatch:vulkan_invocation",
        "//litert/vendors/arm_vulkan_ml/dispatch:vulkan_validation",
        "//tflite:framework",
        "//tflite/kernels:reference_ops",
        "//tflite/schema:schema_fbs",
        "@XNNPACK",
        "@ai_ml_sdk_vgf_library//:encoder",
        "@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers",
        "@arm_vulkan_ml_tool_spirv_tools//:spirv_tools",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
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
    srcs = ["devtool/tool_test.cc"],
    linkstatic = True,
    deps = [
        ":devtool",
        "//litert/c:litert_op_code",
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        "//litert/vendors/arm_vulkan_ml/compiler:op_registry",
        "//litert/vendors/arm_vulkan_ml/compiler:softmax",
        "//litert/vendors/arm_vulkan_ml/dispatch:tensor_capabilities",
        "//tflite:framework",
        "//tflite/schema:schema_fbs",
        "@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_test(
    name = "tosa_query_test",
    size = "small",
    srcs = ["devtool/tosa_query_test.cc"],
    copts = ["-fno-exceptions"],
    linkstatic = True,
    deps = [
        ":devtool",
        "@com_google_googletest//:gtest_main",
    ],
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
    srcs = ["devtool/vgf_export_test.cc"],
    copts = ["-fno-exceptions"],
    linkstatic = True,
    deps = [
        ":devtool",
        "//litert/vendors/arm_vulkan_ml/common:arm_vulkan_ml_jit_executable",
        "@ai_ml_sdk_vgf_library//:decoder",
        "@arm_vulkan_ml_dep_vulkan_headers//:vulkan_headers",
        "@com_google_googletest//:gtest_main",
    ],
)
