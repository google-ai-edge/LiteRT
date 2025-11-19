load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        ":input_generator",
        "//tflite/core/c:common",
        "//tflite/testing:split",
        "//tflite/testing:tflite_driver",
    ] + select({
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework_internal",
            "@org_tensorflow//tensorflow/core:lib",
        ],
        "@org_tensorflow//tensorflow:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib",
        ],
    }),
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    data = [
        ":testdata/test_input.csv",
        "//tflite:testdata/add.bin",
    ],
    deps = [
        ":util",
        "//tflite/testing:tflite_driver",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_binary(
    name = "tflite_kernel_runner",
    srcs = ["tflite_kernel_runner.cc"],
    deps = [
        ":util",
    ],
)

tf_cc_binary(
    name = "generate_diff_report",
    srcs = ["generate_diff_report.cc"],
    deps = [
        ":diff_analyzer",
        "@org_tensorflow//tensorflow/core:framework_internal",
    ],
)

cc_library(
    name = "input_generator",
    srcs = ["input_generator.cc"],
    hdrs = ["input_generator.h"],
    deps = [
        "//tflite:framework",
        "//tflite:string",
        "//tflite/core:framework",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/testing:join",
        "//tflite/testing:split",
    ],
)

cc_test(
    name = "input_generator_test",
    size = "small",
    srcs = ["input_generator_test.cc"],
    data = [
        ":testdata/test_input.csv",
        "//tflite:testdata/multi_add.bin",
    ],
    deps = [
        ":input_generator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "diff_analyzer",
    srcs = ["diff_analyzer.cc"],
    hdrs = ["diff_analyzer.h"],
    deps = [
        "//tflite:string",
        "//tflite/core/c:c_api_types",
        "//tflite/core/c:common",
        "//tflite/testing:split",
    ],
)

tf_cc_test(
    name = "diff_analyzer_test",
    size = "small",
    srcs = ["diff_analyzer_test.cc"],
    data = [
        ":testdata/test_input.csv",
    ],
    deps = [
        ":diff_analyzer",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
