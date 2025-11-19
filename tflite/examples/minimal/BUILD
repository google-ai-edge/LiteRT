# Description:
#   TensorFlow Lite minimal example.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("//tflite:build_def.bzl", "tflite_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_binary(
    name = "minimal",
    srcs = [
        "minimal.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tflite:framework",
        "//tflite/core:cc_api_stable",
        "//tflite/kernels:builtin_ops",
    ],
)
