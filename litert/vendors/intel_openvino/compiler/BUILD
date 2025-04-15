#Copyright (C) 2025 Intel Corporation
#SPDX-License-Identifier: Apache-2.0

load("//litert/build_common:litert_build_defs.bzl", "litert_lib", "litert_test", "litert_dynamic_lib")

package(
    default_visibility = ["//litert:__subpackages__"],
)

litert_dynamic_lib(
    name = "openvino_compiler_plugin",
    srcs = ["openvino_compiler_plugin.cc"],
    hdrs = [
        "//litert/vendors/c:litert_compiler_plugin.h",
    ],
    export_litert_only = True,
    shared_lib_name = "intel_openvino_compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_IntelOpenvino.so",
    ungrte = True,
    visibility = ["//litert:__subpackages__"],
    deps = [
        ":graph_iterator",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "@intel_openvino//:openvino",
    ],
)

litert_lib(
    name = "graph_iterator",
    srcs = ["graph_iterator.cc"],
    hdrs = ["graph_iterator.h"],
    tags = [
        "nobuilder",
    ],
    deps = [
        ":decoder",
        "//litert/c:litert_logging",
        "//litert/c:litert_model",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_model",
        "@intel_openvino//:openvino",
    ],
)

litert_lib(
    name = "decoder",
    srcs = ["decoder.cc"],
    hdrs = ["decoder.h"],
    deps = [
        "//litert/c:litert_logging",
        "//litert/c:litert_options",
        "//litert/cc:litert_model",
        "//litert/tools:dump",
        "@intel_openvino//:openvino",
    ],
)

litert_test(
    name = "decoder_test",
    srcs = [
        "decoder_test.cc",
    ],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":decoder",
        "//litert/test:common",
        "//litert/test:matchers_oss",
        "//litert/test:test_models",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@intel_openvino//:openvino",
    ],
)

litert_test(
    name = "openvino_compiler_plugin_test",
    srcs = [
        "openvino_compiler_plugin_test.cc",
    ],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        "no_oss",
    ],
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        ":openvino_compiler_plugin",  # buildcleaner: keep
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model_predicates",
        "//litert/cc:litert_model",
        "//litert/test:common",
        "//litert/test:matchers_oss",
        "//litert/test:test_models",
        "//litert/vendors/cc:litert_compiler_plugin",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
    ],
)
