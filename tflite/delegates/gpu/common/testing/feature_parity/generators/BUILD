load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_testonly = 1,
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "all_generators",
    hdrs = [
        "add.h",
    ],
    deps = [
        ":add",
        "//tflite/delegates/gpu/common/testing/feature_parity:utils",
    ],
)

cc_library(
    name = "add",
    srcs = ["add.cc"],
    hdrs = ["add.h"],
    deps = [
        "//tflite:schema_fbs_version",
        "//tflite/delegates/gpu/common/testing/feature_parity:utils",
        "//tflite/schema:schema_fbs",
        "@flatbuffers",
    ],
)
