load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//tflite/delegates/gpu/gl:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "shared_buffer",
    hdrs = ["shared_buffer.h"],
    deps = [
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/gl:gl_buffer",
        "//tflite/delegates/gpu/gl:object",
        "//tflite/delegates/gpu/gl:portable",
    ],
)
