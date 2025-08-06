load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//tflite:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "signature_def",
    hdrs = ["signature_def.h"],
    compatible_with = get_compatible_with_portable(),
)
