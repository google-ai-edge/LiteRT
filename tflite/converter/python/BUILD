load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_library")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "tf_python_pybind_extension")
load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@xla//xla/tsl/platform:build_config_root.bzl", "if_pywrap")

licenses(["notice"])

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        ":friends",
        "@org_tensorflow//tensorflow:__pkg__",
    ],
)

package_group(
    name = "friends",
    packages = [
        "//litert/...",
        "//tflite/python/...",
        "//tflite/python/converter/...",
        "//tflite/toco/...",
    ],
)

cc_library(
    name = "tf_tfl_flatbuffer_helpers",
    srcs = ["tf_tfl_flatbuffer_helpers.cc"],
    hdrs = ["tf_tfl_flatbuffer_helpers.h"],
    deps = [
        "//tflite/converter:common",
        "//tflite/converter:converter_flags_proto_cc",
        "//tflite/converter:model_flags_proto_cc",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter:tf_to_tfl_flatbuffer",
        "//tflite/converter:types_proto_cc",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "//tflite/converter/tools/optimize:reduced_precision_metadata",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@org_tensorflow//tensorflow/core:core_cpu_base",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "graphdef_to_tfl_flatbuffer",
    srcs = ["graphdef_to_tfl_flatbuffer.cc"],
    hdrs = [
        "graphdef_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":tf_tfl_flatbuffer_helpers",
        "//tflite/converter:common",
        "//tflite/converter:converter_flags_proto_cc",
        "//tflite/converter:model_flags_proto_cc",
        "//tflite/converter:types_proto_cc",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:IR",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow/translate/tools:parsers",
        "@org_tensorflow//tensorflow/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "@org_tensorflow//tensorflow/core:core_cpu_base",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "saved_model_to_tfl_flatbuffer",
    srcs = ["saved_model_to_tfl_flatbuffer.cc"],
    hdrs = [
        "saved_model_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":tf_tfl_flatbuffer_helpers",
        "//tflite/converter:common",
        "//tflite/converter:converter_flags_proto_cc",
        "//tflite/converter:model_flags_proto_cc",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter:tf_to_tfl_flatbuffer",
        "//tflite/converter:types_proto_cc",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "jax_to_tfl_flatbuffer",
    srcs = ["jax_to_tfl_flatbuffer.cc"],
    hdrs = [
        "jax_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":tf_tfl_flatbuffer_helpers",
        "//tflite/converter:common",
        "//tflite/converter:converter_flags_proto_cc",
        "//tflite/converter:model_flags_proto_cc",
        "//tflite/converter:tensorflow_lite",
        "//tflite/converter:types_proto_cc",
        "//tflite/converter/quantization/common/quantization_lib:quantization_config",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/utility",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@xla//xla/hlo/parser:hlo_parser",
        "@xla//xla/hlo/translate:stablehlo",
        "@xla//xla/service:hlo_proto_cc",
    ],
)

# Smaller version of flatbuffer_translate which only converts flatbuffer to MLIR.
cc_library(
    name = "flatbuffer_to_mlir",
    srcs = [
        "flatbuffer_to_mlir.cc",
    ],
    hdrs = [
        "flatbuffer_to_mlir.h",
    ],
    deps = [
        "//tflite/converter:flatbuffer_import",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)

py_strict_library(
    name = "wrap_converter",
    srcs = [
        "wrap_converter.py",
    ],
    deps = [
        ":_pywrap_converter_api",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib_py",
        "@org_tensorflow//tensorflow/python:pywrap_tensorflow",
    ] + if_pywrap(
        if_true = [] + select({
            "//conditions:default": [],
            # This reqirement is for LiteRT only as tensorflow/compiler/mlirt/lite is
            # duplicated in LiteRT/tflite/converter.
            "@org_tensorflow//tensorflow:disable_tf_lite_py": [
                "//tflite/python:pywrap_tflite",
            ],
        }),
    ),
)

config_setting(
    name = "tflite_convert_with_select_tf_ops",
    define_values = {"tflite_convert_with_select_tf_ops": "true"},
    visibility = ["//visibility:private"],
)

filegroup(
    name = "converter_python_api_hdrs",
    srcs = [
        "converter_python_api.h",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "converter_python_api",
    srcs = ["converter_python_api.cc"],
    hdrs = ["converter_python_api.h"],
    features = ["-parse_headers"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":flatbuffer_to_mlir",
        ":graphdef_to_tfl_flatbuffer",
        ":jax_to_tfl_flatbuffer",
        ":saved_model_to_tfl_flatbuffer",
        "//tflite/converter:converter_flags_proto_cc",
        "//tflite/converter:model_flags_proto_cc",
        "//tflite/converter:types_proto_cc",
        "//tflite/converter/core:absl_error_model_builder",
        "//tflite/converter/debug:debug_options_proto_cc",
        "//tflite/converter/metrics:error_collector",
        "//tflite/converter/python/interpreter_wrapper:python_error_reporter",
        "//tflite/converter/python/interpreter_wrapper:python_utils",
        "//tflite/converter/quantization/lite:quantize_model",
        "//tflite/converter/schema:schema_fbs",
        "//tflite/converter/sparsity:sparsify_model",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//:protobuf_headers",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/c:kernels",
        "@org_tensorflow//tensorflow/c:tf_status_headers",
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@xla//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
        "@xla//xla/tsl/platform:status",
    ] + select({
        # This is required when running `tflite_convert` from `bazel`.
        # It requires to link with TensorFlow Ops to get the op definitions.
        ":tflite_convert_with_select_tf_ops": [
            "@org_tensorflow//tensorflow/core:ops",
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
)

tf_python_pybind_extension(
    name = "_pywrap_converter_api",
    srcs = [
        "converter_python_api_wrapper.cc",
    ],
    hdrs = [":converter_python_api_hdrs"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_converter_api.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "@org_tensorflow//tensorflow/python/lib/core:pybind11_lib",
        "@pybind11",
        "@xla//third_party/python_runtime:headers",
    ] + if_pywrap([":converter_python_api"]),
)
