load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:py_library.bzl", "py_library")
load("//litert/build_common:symlink_files.bzl", "symlink_inputs")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
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
        "//tflite/converter/quantization/ir:QuantOps",
        "//tflite/converter/stablehlo:prepare_hlo",
        "//third_party/stablehlo:register",
        "//third_party/stablehlo:stablehlo_ops",
        "//third_party/stablehlo:vhlo_ops",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Transforms",
    ],
)

symlink_inputs(
    name = "_mlir_libs",
    rule = py_library,
    symlinked_inputs = {"srcs": {
        ".": [
            "@llvm-project//mlir/python:MlirLibsPyFiles",
        ],
    }},
    deps = [
        ":_mlir",
    ],
)

cc_library(
    name = "litert_mlir_capi__shared_library",
    srcs = select({
        "//conditions:default": [":litert_mlir.so"],
    }),
)

cc_binary(
    name = "litert_mlir.so",
    # linkopts = [
    #     "-Wl,-rpath='$$ORIGIN/../../..'",
    # ],
    linkshared = 1,
    deps = [
        ":model_utils_core",
        "@llvm-project//mlir:CAPIArithObjects",
        "@llvm-project//mlir:CAPIIRObjects",
        "@llvm-project//mlir:CAPITransformsObjects",
        "@llvm-project//mlir:MLIRBindingsPythonCAPIObjects",
        "@nanobind",
    ],
)

pybind_extension(
    name = "_mlir",
    srcs = [
        "@llvm-project//mlir:MLIRBindingsPythonSourceFiles",
        "@llvm-project//mlir:lib/Bindings/Python/MainModule.cpp",
    ],
    hdrs = [
        "@llvm-project//mlir:MLIRBindingsPythonCoreHeaders",
    ],
    common_lib_packages = ["litert/python"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    wrap_py_init = False,
    # linkopts = [
    #     "-Wl,-rpath='$$ORIGIN/../../..'",
    # ],
    # features = ["-use_header_modules"],
    # includes = [
    #     "@llvm-project//mlir:lib/Bindings/Python",
    # ],
    # link_in_framework = True,
    deps = [
        ":litert_mlir_capi__shared_library",
        "@llvm-project//mlir:MLIRBindingsPythonCoreNoCAPI",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
    ],
)

pybind_extension(
    name = "model_utils_ext",
    srcs = [
        "model_utils_ext.cc",
    ],
    hdrs = [
        "model_utils_core.h",
        "@llvm-project//mlir:MLIRBindingsPythonCoreHeaders",
    ],
    common_lib_packages = ["litert/python"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    wrap_py_init = False,
    # linkopts = [
    #     "-Wl,-rpath='$$ORIGIN/../../..'",
    # ],
    # features = ["-use_header_modules"],
    # includes = [
    #     "@llvm-project//mlir:lib/Bindings/Python",
    # ],
    # link_in_framework = True,
    deps = [
        ":litert_mlir_capi__shared_library",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:CAPIIRHeaders",
        "@llvm-project//mlir:MLIRBindingsPythonCoreNoCAPI",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "@local_xla//third_party/python_runtime:headers",
        "@nanobind",
    ],
)
