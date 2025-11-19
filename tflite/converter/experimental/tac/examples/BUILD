load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "example_hardware",
    srcs = ["example_hardware.cc"],
    hdrs = ["example_hardware.h"],
    deps = [
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter/experimental/tac:device_transform_patterns",
        "//tflite/converter/experimental/tac/hardwares:simple_hardware",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "example-hardware-translate",
    deps = [
        ":example_hardware",
        "//tflite/converter/experimental/tac:tac-translate-lib",
        "//tflite/converter/experimental/tac/hardwares:all-target-hardwares",
    ],
)
