load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

# copybara:uncomment package(default_applicable_licenses = ["@org_tensorflow//tensorflow:license"])

cc_library(
    name = "rematerializer",
    srcs = ["rematerializer.cc"],
    hdrs = ["rematerializer.h"],
    deps = [
    ],
)

tf_cc_test(
    name = "rematerializer_test",
    size = "small",
    srcs = ["rematerializer_test.cc"],
    deps = [
        ":rematerializer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "metadata_util",
    srcs = ["metadata_util.cc"],
    hdrs = ["metadata_util.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tflite/converter:__pkg__",
        "//tflite/core:__pkg__",
        "//tflite/delegates:__pkg__",
    ],
    deps = [
        "//tflite/converter:control_edges",
    ],
)

tf_cc_test(
    name = "metadata_util_test",
    size = "small",
    srcs = ["metadata_util_test.cc"],
    deps = [
        ":metadata_util",
        "@com_google_googletest//:gtest_main",
    ],
)
