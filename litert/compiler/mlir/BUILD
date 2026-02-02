load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "pybind_ext_core_headers",
    srcs = [
        "converter_api_core.h",
        "model_utils_core.h",
    ],
)

cc_library(
    name = "status_scoped_diagnostic_handler",
    srcs = ["status_scoped_diagnostic_handler.cc"],
    hdrs = [
        "status_scoped_diagnostic_handler.h",
    ],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "converter_api_core",
    srcs = ["converter_api_core.cc"],
    hdrs = ["converter_api_core.h"],
    deps = [
        ":status_scoped_diagnostic_handler",
        "//tflite/converter:common",
        "//tflite/converter:converter_flags_proto_cc",
        "//tflite/converter:flatbuffer_export",
        "//tflite/converter:model_flags_proto_cc",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter:tf_to_tfl_flatbuffer",
        "//tflite/converter:types_proto_cc",
        "//tflite/converter/metrics:converter_error_data_proto_cc",
        "//tflite/converter/python:tf_tfl_flatbuffer_helpers",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "//tflite/converter/stablehlo:prepare_hlo",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:RegisterAllDialects",  # buildcleaner: keep
        "@llvm-project//mlir:RegisterAllExtensions",  # buildcleaner: keep
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_passes_optimization",
        "@stablehlo//:vhlo_ops",
    ],
)

cc_library(
    name = "model_utils_core",
    srcs = ["model_utils_core.cc"],
    hdrs = ["model_utils_core.h"],
    deps = [
        "//tflite/converter:flatbuffer_export",
        "//tflite/converter:flatbuffer_import",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter:tensorflow_lite_ops",
        "//tflite/converter:tensorflow_lite_optimize",
        "//tflite/converter:tensorflow_lite_quantize",
        "//tflite/converter/quantization/ir:QuantOps",
        "//tflite/converter/stablehlo:prepare_hlo",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:FileCheckLib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_passes_optimization",
        "@stablehlo//:vhlo_ops",
    ],
)
