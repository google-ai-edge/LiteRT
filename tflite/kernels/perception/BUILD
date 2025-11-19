load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "pybind_extension")
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
    name = "perception_ops",
    srcs = [
        "dense_image_warp.cc",
        "max_pool_with_argmax.cc",
        "max_unpooling_2d.cc",
        "perception_ops.cc",
    ],
    hdrs = [
        "perception_ops.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels:padding",
        "//tflite/kernels/internal:common",
        "//tflite/kernels/internal:compatibility",
        "//tflite/kernels/internal:tensor",
        "//tflite/kernels/internal:types",
        "@flatbuffers",
    ],
)

cc_test(
    name = "perception_ops_test",
    size = "small",
    srcs = [
        "dense_image_warp_test.cc",
        "max_pool_with_argmax_test.cc",
        "max_unpooling_2d_test.cc",
    ],
    deps = [
        ":perception_ops",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/c:common",
        "//tflite/kernels:test_main",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

pybind_extension(
    name = "pywrap_perception_ops",
    srcs = [
        "perception_ops_wrapper.cc",
    ],
    hdrs = ["perception_ops.h"],
    additional_exported_symbols = ["PerceptionOpsRegisterer"],
    enable_stub_generation = True,
    link_in_framework = True,
    pytype_srcs = [
        "pywrap_perception_ops.pyi",
    ],
    deps = [
        ":perception_ops",
        "//tflite:mutable_op_resolver",
        "//tflite/c:common",
        "@pybind11",
    ],
)
