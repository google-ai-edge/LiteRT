load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "pybind_ext_headers",
    srcs = [
        "model_utils_core.h",
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
