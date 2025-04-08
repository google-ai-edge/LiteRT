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
        "openvino_compiler_plugin.h",
    ],
    export_litert_only = True,
    linkstatic = 1,
    shared_lib_name = "intel_openvino_compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_IntelOpenvino.so",
    ungrte = True,
    visibility = ["//litert:__subpackages__"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
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
#    linkstatic = True,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        # Don't build/test in OS until qnn is available.
        "nobuilder",
        "no_oss",
        # Sanitizer runtime doesn't work with anything that loads libQnnHtp.so.
        "nosan",
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
