load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")
load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        ":friends",
        "@org_tensorflow//tensorflow:__pkg__",
    ],
    licenses = ["notice"],
)

exports_files(glob([
    "testdata/*.bin",
]))

package_group(
    name = "friends",
    packages = [
        "//tflite/...",
        "//tflite/converter/...",
    ],
)

cc_library(
    name = "quantize_model",
    srcs = [
        "quantize_model.cc",
    ],
    hdrs = [
        "quantize_model.h",
    ],
    deps = [
        "//tflite/converter:common",
        "//tflite/converter:flatbuffer_translate_lib",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter:tf_tfl_passes",
        "//tflite/converter/debug",
        "//tflite/converter/debug:debug_options_proto_cc",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:error_util",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "quantize_weights",
    srcs = [
        "quantize_weights.cc",
    ],
    hdrs = [
        "quantize_weights.h",
    ],
    deps = [
        "//tflite/converter:common",
        "//tflite/converter:flatbuffer_translate_lib",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter:tf_tfl_passes",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:error_util",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "tfl_to_std",
    srcs = [
        "tfl_to_std.cc",
    ],
    hdrs = [
        "tfl_to_std.h",
        "//tflite/converter/quantization/common/quantization_lib:quantization_utils.h",
    ],
    deps = [
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter/quantization/common/quantization_lib",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "//tflite/converter/quantization/ir:QuantOps",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

# Binary to apply quantization on the annotated files.
tf_cc_binary(
    name = "tfl_quantizer",
    srcs = [
        "tfl_quantizer.cc",
    ],
    deps = [
        ":quantize_model",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_test(
    name = "quantize_model_test",
    srcs = ["quantize_model_test.cc"],
    args = [
        "--test_model_file=$(location //tflite/converter/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "//tflite/converter/quantization/lite:testdata/add_with_const_input.bin",
        "//tflite/converter/quantization/lite:testdata/argmax.bin",
        "//tflite/converter/quantization/lite:testdata/broadcast_to.bin",
        "//tflite/converter/quantization/lite:testdata/concat.bin",
        "//tflite/converter/quantization/lite:testdata/fc.bin",
        "//tflite/converter/quantization/lite:testdata/fc_qat.bin",
        "//tflite/converter/quantization/lite:testdata/gather_nd.bin",
        "//tflite/converter/quantization/lite:testdata/lstm_calibrated.bin",
        "//tflite/converter/quantization/lite:testdata/lstm_calibrated2.bin",
        "//tflite/converter/quantization/lite:testdata/lstm_quantized.bin",
        "//tflite/converter/quantization/lite:testdata/lstm_quantized2.bin",
        "//tflite/converter/quantization/lite:testdata/maximum.bin",
        "//tflite/converter/quantization/lite:testdata/minimum.bin",
        "//tflite/converter/quantization/lite:testdata/mixed.bin",
        "//tflite/converter/quantization/lite:testdata/mixed16x8.bin",
        "//tflite/converter/quantization/lite:testdata/multi_input_add_reshape.bin",
        "//tflite/converter/quantization/lite:testdata/pack.bin",
        "//tflite/converter/quantization/lite:testdata/single_avg_pool_min_minus_5_max_plus_5.bin",
        "//tflite/converter/quantization/lite:testdata/single_conv_no_bias.bin",
        "//tflite/converter/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin",
        "//tflite/converter/quantization/lite:testdata/single_conv_weights_min_minus_127_max_plus_127.bin",
        "//tflite/converter/quantization/lite:testdata/single_softmax_min_minus_5_max_plus_5.bin",
        "//tflite/converter/quantization/lite:testdata/split.bin",
        "//tflite/converter/quantization/lite:testdata/svdf_calibrated.bin",
        "//tflite/converter/quantization/lite:testdata/svdf_quantized.bin",
        "//tflite/converter/quantization/lite:testdata/transpose.bin",
        "//tflite/converter/quantization/lite:testdata/unidirectional_sequence_lstm_calibrated.bin",
        "//tflite/converter/quantization/lite:testdata/unidirectional_sequence_lstm_quantized.bin",
        "//tflite/converter/quantization/lite:testdata/unpack.bin",
        "//tflite/converter/quantization/lite:testdata/where.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantize_model",
        ":test_util",
        "//tflite/converter/core:absl_error_model_builder",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/schema:schema_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@org_tensorflow//tensorflow/core:framework_internal",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "quantize_weights_test",
    srcs = ["quantize_weights_test.cc"],
    args = [
        "--test_model_file=$(location //tflite/converter/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "//tflite/converter/quantization/lite:testdata/custom_op.bin",
        "//tflite/converter/quantization/lite:testdata/quantized_with_gather.bin",
        "//tflite/converter/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin",
        "//tflite/converter/quantization/lite:testdata/weight_shared_between_convs.bin",
    ],
    tags = [
        # TODO(b/327796566): re-enable after the bug is fixed
        "manual",
        "noasan",
        "noguitar",
        "notap",
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantize_weights",
        ":test_util",
        "//tflite/converter/core:absl_error_model_builder",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/schema:schema_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@local_xla//xla/tsl/platform:logging",
        "@org_tensorflow//tensorflow/core:framework_internal",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//tflite/converter/core/api:error_reporter",
        "@com_google_googletest//:gtest",
    ],
)
