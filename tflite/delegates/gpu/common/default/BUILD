load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "custom_parsers",
    srcs = ["custom_parsers.cc"],
    hdrs = ["//tflite/delegates/gpu/common:custom_parsers.h"],
    deps = [
        "//tflite/delegates/gpu/common:operation_parser",
        "//tflite/delegates/gpu/common:unimplemented_operation_parser",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "custom_transformations",
    srcs = ["custom_transformations.cc"],
    hdrs = ["//tflite/delegates/gpu/common:custom_transformations.h"],
    deps = [
        "//tflite/delegates/gpu/common:model_transformer",
    ],
)
