load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = ["//tflite/converter/..."],
)

td_library(
    name = "ptq_td_files",
    srcs = [
        "fallback_to_flex_patterns.td",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    deps = [
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "ptq_fallback_to_flex_inc_gen",
    compatible_with = get_compatible_with_portable(),
    includes = [
        "/external/org_tensorflow",
        "/third_party/tensorflow",
    ],
    tbl_outs = {"fallback_to_flex_patterns.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "fallback_to_flex_patterns.td",
    deps = [":ptq_td_files"],
)

cc_library(
    name = "tf_quantization_passes",
    srcs = [
        "fallback_to_flex_ops.cc",
        "fallback_to_flex_patterns.inc",
        "tf_to_quant.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":ptq_fallback_to_flex_inc_gen",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter/quantization/common/quantization_lib",
        "//tflite/converter/quantization/ir:QuantOps",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:translate_utils",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:statusor",
    ],
    alwayslink = 1,
)
