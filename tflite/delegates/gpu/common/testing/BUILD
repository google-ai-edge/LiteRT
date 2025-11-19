load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "interpreter_utils",
    testonly = 1,
    srcs = ["interpreter_utils.cc"],
    hdrs = ["interpreter_utils.h"],
    deps = [
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/api",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:tensor",
        "//tflite/schema:schema_fbs",
    ],
)
