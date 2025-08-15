load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    # By default, these targets should only be used within the quantization library.
    default_visibility = [
        "//tflite/converter:__subpackages__",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization:__subpackages__",
        "@org_tensorflow//tensorflow/compiler/mlir/stablehlo:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "test_base",
    testonly = 1,
    srcs = [],
    hdrs = ["test_base.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter/quantization/ir:QuantOps",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common:func",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo/cc:context",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/core:test",
        "@stablehlo//:stablehlo_ops",
    ],
)
