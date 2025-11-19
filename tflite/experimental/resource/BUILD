load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "cache_buffer",
    srcs = ["cache_buffer.cc"],
    hdrs = [
        "cache_buffer.h",
        "//tflite/core/c:common.h",
    ],
    deps = [
        ":resource",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "cache_buffer_test",
    srcs = ["cache_buffer_test.cc"],
    deps = [
        ":cache_buffer",
        "//tflite/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "resource",
    srcs = [
        "initialization_status.cc",
        "resource_variable.cc",
        "static_hashtable.cc",
    ],
    hdrs = [
        "initialization_status.h",
        "lookup_interfaces.h",
        "lookup_util.h",
        "resource_base.h",
        "resource_variable.h",
        "static_hashtable.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tflite:string_util",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels/internal:tensor",
        "//tflite/kernels/internal:types",
    ],
)

cc_test(
    name = "resource_variable_test",
    srcs = [
        "resource_variable_test.cc",
    ],
    deps = [
        ":resource",
        "//tflite:util",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
