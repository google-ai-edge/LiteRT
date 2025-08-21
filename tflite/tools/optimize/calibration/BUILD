load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tflite:build_def.bzl", "tflite_copts")
load("//tflite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "builtin_logging_op",
    srcs = ["builtin_logging_ops/lstm.cc"],
    hdrs = ["builtin_logging_ops/lstm.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_logger",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/api",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels:lstm_shared",
        "//tflite/kernels:op_macros",
        "//tflite/kernels/internal:reference",
        "//tflite/kernels/internal:tensor_utils",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "custom_logging_op",
    srcs = ["custom_logging_ops/lstm.cc"],
    hdrs = ["custom_logging_ops/lstm.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_logger",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "//tflite/core/api",
        "//tflite/core/c:common",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels:lstm_shared",
        "//tflite/kernels:op_macros",
        "//tflite/kernels/internal:reference",
        "//tflite/kernels/internal:tensor_utils",
    ],
)

cc_library(
    name = "calibrator_lib",
    srcs = ["calibrator.cc"],
    hdrs = ["calibrator.h"],
    copts = tflite_copts(),
    deps = [
        ":builtin_logging_op",
        ":calibration_common",
        ":calibration_logger",
        ":calibration_reader",
        ":custom_logging_op",
        ":logging_op",
        ":logging_op_resolver",
        "//tflite:framework",
        "//tflite:minimal_logging",
        "//tflite:string",
        "//tflite/converter:allocation",
        "//tflite/converter/schema:schema_utils",
        "//tflite/core:framework",
        "//tflite/core/api",
        "//tflite/core/c:common",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "calibrator_test",
    srcs = ["calibrator_test.cc"],
    args = [
        "--test_model_file=$(location //tflite:testdata/multi_add.bin)",
    ],
    data = [
        "//tflite:testdata/call_once_mul.bin",
        "//tflite:testdata/custom_lstm.bin",
        "//tflite:testdata/lstm.bin",
        "//tflite:testdata/multi_add.bin",
        "//tflite:testdata/multi_subgraphs_while.bin",
        "//tflite:testdata/unidirectional_sequence_lstm.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":calibration_reader",
        ":calibrator_lib",
        "//tflite:framework",
        "//tflite:string",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:framework",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/core:framework_internal",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "logging_op_resolver",
    srcs = ["logging_op_resolver.cc"],
    hdrs = ["logging_op_resolver.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_common",
        "//tflite:framework",
        "//tflite:util",
        "//tflite/c:common",
        "//tflite/core/api",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "logging_op_resolver_test",
    srcs = ["logging_op_resolver_test.cc"],
    deps = [
        ":calibration_common",
        ":logging_op_resolver",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/kernels:builtin_ops",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "calibration_reader",
    srcs = ["calibration_reader.cc"],
    hdrs = ["calibration_reader.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_logger",
        "//tflite:framework",
        "//tflite/c:common",
        "//tflite/core:framework",
        "//tflite/core/c:c_api_types",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "calibration_logger",
    srcs = ["calibration_logger.cc"],
    hdrs = ["calibration_logger.h"],
    copts = tflite_copts(),
    deps = [
        "//tflite:framework",
        "//tflite:minimal_logging",
        "//tflite/c:c_api_types",
        "//tflite/core/api",
        "//tflite/core/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "calibration_common",
    hdrs = ["calibration_common.h"],
    copts = tflite_copts(),
    deps = [
        "//tflite:framework",
    ],
)

cc_library(
    name = "logging_op",
    hdrs = ["logging_op.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_logger",
        "//tflite/core/c:common",
    ],
)

tflite_portable_test_suite()
