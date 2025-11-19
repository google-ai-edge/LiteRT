load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "sparsity_format_converter",
    srcs = ["sparsity_format_converter.cc"],
    hdrs = ["sparsity_format_converter.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:common",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "sparsity_format_converter_test",
    srcs = ["sparsity_format_converter_test.cc"],
    data = ["//tflite:testdata/sparse_tensor.bin"],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":sparsity_format_converter",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core:framework",
        "@com_google_googletest//:gtest_main",
    ],
)
