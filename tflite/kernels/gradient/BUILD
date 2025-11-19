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
    name = "bcast_grad_args_op",
    srcs = [
        "bcast_grad_args.cc",
    ],
    hdrs = [
        "bcast_grad_args.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tflite:framework",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:tensor",
        "//tflite/kernels/internal:types",
    ],
)

cc_library(
    name = "gradient_ops",
    srcs = [
        "gradient_ops.cc",
    ],
    hdrs = [
        "gradient_ops.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":bcast_grad_args_op",
        "//tflite:framework",
    ],
)

cc_test(
    name = "bcast_grad_args_op_test",
    size = "small",
    srcs = [
        "bcast_grad_args_test.cc",
    ],
    deps = [
        ":bcast_grad_args_op",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/kernels:test_main",
        "//tflite/kernels:test_util",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

pybind_extension(
    name = "pywrap_gradient_ops",
    srcs = [
        "gradient_ops_wrapper.cc",
    ],
    hdrs = ["gradient_ops.h"],
    additional_exported_symbols = ["GradientOpsRegisterer"],
    enable_stub_generation = True,
    link_in_framework = True,
    pytype_srcs = [
        "pywrap_gradient_ops.pyi",
    ],
    deps = [
        ":gradient_ops",
        "//tflite:mutable_op_resolver",
        "@pybind11",
    ],
)
