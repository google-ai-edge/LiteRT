load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(":tests.bzl", "hexagon_op_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "hexagon_delegate_op_model",
    testonly = 1,
    hdrs = ["hexagon_delegate_op_model.h"],
    deps = [
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/delegates/hexagon:hexagon_delegate",
        "//tflite/kernels:test_util",
        "//tflite/kernels/internal:reference_base",
        "//tflite/kernels/internal:tensor",
        "//tflite/kernels/internal:types",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

hexagon_op_tests(
    srcs = [
        "activations_test.cc",
        "arg_min_max_test.cc",
        "arithmetic_test.cc",
        "concat_test.cc",
        "conv_test.cc",
        "l2_norm_test.cc",
        "matmul_test.cc",
        "min_max_builder_test.cc",
        "mirror_pad_test.cc",
        "mul_test.cc",
        "neg_test.cc",
        "pack_test.cc",
        "pad_test.cc",
        "pool_test.cc",
        "quantize_test.cc",
        "reduce_test.cc",
        "reshape_test.cc",
        "resize_test.cc",
        "rsqrt_test.cc",
        "slice_test.cc",
        "softmax_test.cc",
        "space_to_depth_test.cc",
        "split_test.cc",
        "squared_difference_test.cc",
        "strided_slice_test.cc",
        "transpose_conv_test.cc",
        "transpose_test.cc",
    ],
    deps = [
        ":hexagon_delegate_op_model",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels:reshape_test_common",
        "//tflite/kernels/internal:test_util",
        "//tflite/kernels/internal:types",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "batch_seq_config_test",
    srcs = [
        "batch_seq_config_test.cc",
    ],
    tags = [
        "no_oss",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tflite:framework",
        "//tflite:string",
        "//tflite/c:common",
        "//tflite/core/api:op_resolver",
        "//tflite/core/c:common",
        "//tflite/delegates/hexagon:hexagon_delegate",
        "//tflite/kernels:builtin_ops",
        "//tflite/kernels:kernel_util",
        "//tflite/testing:util",
        "//tflite/tools:logging",
        "//tflite/tools/benchmark:benchmark_utils",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/core:framework_internal",
    ],
)
