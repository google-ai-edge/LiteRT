load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tflite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "reduced_precision_support",
    srcs = [],
    hdrs = [
        "reduced_precision_support.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir/lite/tools/optimize:reduced_precision_metadata",
    ],
)

tf_cc_test(
    name = "reduced_precision_support_test",
    srcs = ["reduced_precision_support_test.cc"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":reduced_precision_support",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/schema:schema_fbs",
        "//tflite/schema:schema_utils",
        "//tflite/testing:util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/core/platform:platform_port",
    ],
)

cc_library(
    name = "modify_model_interface",
    srcs = ["modify_model_interface.cc"],
    hdrs = ["modify_model_interface.h"],
    deps = [
        ":model_utils",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/kernels/internal:compatibility",
        "//tflite/schema:schema_fbs",
        "//tflite/schema:schema_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "modify_model_interface_test",
    srcs = ["modify_model_interface_test.cc"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":modify_model_interface",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/schema:schema_fbs",
        "//tflite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_binary(
    name = "modify_model_interface_main",
    srcs = ["modify_model_interface_main.cc"],
    deps = [
        ":modify_model_interface",
        ":quantize_model",
    ],
)

cc_library(
    name = "quantization_wrapper_utils",
    srcs = ["quantization_wrapper_utils.cc"],
    hdrs = ["quantization_wrapper_utils.h"],
    deps = [
        ":operator_property",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/api",
        "//tflite/schema:schema_fbs",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "quantization_wrapper_utils_test",
    srcs = ["quantization_wrapper_utils_test.cc"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantization_wrapper_utils",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/schema:schema_fbs",
        "//tflite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "quantization_wrapper_utils_custom_test",
    srcs = [
        "quantization_wrapper_utils.cc",
        "quantization_wrapper_utils.h",
        "quantization_wrapper_utils_custom_test.cc",
    ],
    defines = [
        "TFLITE_CUSTOM_LSTM",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":operator_property",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/core/api",
        "//tflite/schema:schema_fbs",
        "//tflite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantization_wrapper",
    srcs = ["quantization_wrapper.cc"],
    hdrs = ["quantization_wrapper.h"],
    deps = [
        ":quantization_wrapper_utils",
        "//tflite:framework",
        "//tflite/core/api",
        "//tflite/schema:schema_fbs",
        "//tflite/tools/optimize:quantize_model",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantization_utils",
    srcs = ["quantization_utils.cc"],
    hdrs = ["quantization_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":model_utils",
        "//tflite:framework",
        "//tflite:minimal_logging",
        "//tflite/c:c_api_types",
        "//tflite/core/api",
        "//tflite/core/c:common",
        "//tflite/kernels/internal:cppmath",
        "//tflite/kernels/internal:quantization_util",
        "//tflite/kernels/internal:tensor_utils",
        "//tflite/kernels/internal:types",
        "//tflite/schema:schema_fbs",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "model_utils",
    srcs = ["model_utils.cc"],
    hdrs = ["model_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":operator_property",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/kernels/internal:tensor_utils",
        "//tflite/kernels/internal:types",
        "//tflite/schema:schema_conversion_utils",
        "//tflite/schema:schema_fbs",
        "//tflite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "model_utils_test",
    srcs = ["model_utils_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":model_utils",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

alias(
    name = "operator_property",
    actual = "@org_tensorflow//tensorflow/compiler/mlir/lite/tools/optimize:operator_property",
)

tf_cc_test(
    name = "quantization_utils_test",
    srcs = ["quantization_utils_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantization_utils",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/schema:schema_fbs",
        "//tflite/schema:schema_utils",
        "//tflite/testing:util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:test_util",
        "@org_tensorflow//tensorflow/core:framework_internal",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "quantize_model",
    srcs = ["quantize_model.cc"],
    hdrs = ["quantize_model.h"],
    deps = [
        ":model_utils",
        ":operator_property",
        ":quantization_utils",
        "//tflite:framework",
        "//tflite:util",
        "//tflite/core:framework",
        "//tflite/core/api",
        "//tflite/kernels/internal:cppmath",
        "//tflite/schema:schema_fbs",
        "//tflite/schema:schema_utils",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "quantize_model_test",
    srcs = ["quantize_model_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/add_with_const_input.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/argmax.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/broadcast_to.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/concat.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/fc.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/fc_qat.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/gather_nd.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/lstm_calibrated.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/lstm_calibrated2.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/lstm_quantized.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/lstm_quantized2.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/maximum.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/minimum.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/mixed.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/mixed16x8.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/multi_input_add_reshape.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/pack.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/resource_vars_calibrated.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_avg_pool_min_minus_5_max_plus_5.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_no_bias.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_minus_127_max_plus_127.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_softmax_min_minus_5_max_plus_5.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/split.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/svdf_calibrated.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/svdf_quantized.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/transpose.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/unidirectional_sequence_lstm_calibrated.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/unidirectional_sequence_lstm_quantized.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/unpack.bin",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:testdata/where.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantize_model",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/schema:schema_fbs",
        "//tflite/schema:schema_utils",
        "//tflite/testing:util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/quantization/lite:test_util",
        "@org_tensorflow//tensorflow/core:framework_internal",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

tflite_portable_test_suite()
