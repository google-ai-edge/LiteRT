load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_copts", "tflite_self_contained_libs_test_suite")
load("//tflite/core:special_rules.bzl", "builtin_ops_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "builtin_op_kernels.h",
        "register.h",
    ],
    visibility = [
        "//research/drishti/benchmarking/async:__subpackages__",
        "//tflite:__subpackages__",
    ],
)

cc_test(
    name = "builtin_ops_test",
    srcs = ["register_test.cc"],
    deps = [
        ":builtin_ops",
        "//tflite:mutable_op_resolver",
        "//tflite/c:common",
        "//tflite/converter/tools/versioning",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "builtin_ops",
    srcs = ["register.cc"],
    hdrs = [
        "builtin_op_kernels.h",
        "register.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tflite:__subpackages__",
    ] + builtin_ops_visibility_allowlist(),
    deps = [
        "//tflite:mutable_op_resolver",
        "//tflite:tflite_with_xnnpack_optional",
        "//tflite/core:cc_api_stable",
        "//tflite/core/c:common",
        "//tflite/kernels:builtin_op_kernels",
        "//tflite/schema:schema_fbs",
    ],
)

# For internal usage by shared libraries only.
cc_library(
    name = "builtin_ops_all_linked",
    srcs = ["register.cc"],
    hdrs = [
        "builtin_op_kernels.h",
        "register.h",
    ],
    copts = tflite_copts(),
    # Limit visibility to TFLite only.
    visibility = [
        "//tflite:__subpackages__",
    ],
    deps = [
        "//tflite:framework_stable",
        "//tflite:tflite_with_xnnpack_optional",
        "//tflite/core:cc_api_stable",
        "//tflite/core/c:common",
        "//tflite/kernels:builtin_op_kernels",
        "//tflite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_test_suite")
