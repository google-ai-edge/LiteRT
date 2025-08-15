load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    # By default, these targets should only be used within the quantization library.
    # copybara:uncomment_begin(google-only)
    # default_visibility = [
    # "//platforms/darwinn/compiler:__subpackages__",
    # "@org_tensorflow//tensorflow:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    default_visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    licenses = ["notice"],
)

cc_library(
    name = "tfl_quantization_driver",
    srcs = [
        "tfl_quantization_driver.cc",
    ],
    hdrs = [
        "tfl_quantization_driver.h",
    ],
    deps = [
        ":quantization_config",
        ":quantization_lib",
        "//tflite/converter:tensorflow_lite_ops",
        "//tflite/converter/quantization/ir:QuantOps",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "quantization_lib",
    srcs = [
        "quantization_driver.cc",
        "quantization_interface.cc.inc",
        "quantization_utils.cc",
    ],
    hdrs = [
        "quantization_driver.h",
        "quantization_interface.h.inc",
        "quantization_traits.h",
        "quantization_utils.h",
    ],
    deps = [
        ":quantization_config",
        ":quantization_interfaces_inc_gen",
        "//tflite/converter/quantization/ir:QuantOps",
        "//tflite/converter/quantization/lite/toco_legacy:portable_tensor_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "@org_tensorflow//tensorflow/compiler/mlir/tools/optimize:quantization_utils",
        "@org_tensorflow//tensorflow/core:lib_proto_parsing",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "quantization_driver_test",
    srcs = ["quantization_driver_test.cc"],
    deps = [
        ":quantization_lib",
        "//tflite/converter/quantization/common:test_base",
        "//tflite/converter/quantization/ir:QuantOps",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common:attrs_and_constraints",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/common:func",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
    ],
)

td_library(
    name = "quantization_td_files",
    srcs = [
        "quantization.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tflite/converter/quantization/ir:QuantizationOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "quantization_interfaces_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "quantization_interface.h.inc": ["-gen-op-interface-decls"],
        "quantization_interface.cc.inc": ["-gen-op-interface-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "quantization.td",
    deps = [
        ":quantization_td_files",
    ],
)

cc_library(
    name = "quantization_config",
    srcs = [
        "quantization_config.cc",
    ],
    hdrs = [
        "quantization_config.h",
    ],
    deps = [
        "//tflite/converter/tools/optimize:reduced_precision_metadata",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

exports_files([
    "quantization_traits.h",
    "quantization_config.h",
    "quantization_utils.h",
])
