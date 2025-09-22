load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_library")
load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_features_nolayering_check_if_ios",
    "tf_gen_op_wrapper_py",
    "tf_opts_nortti_if_android",
)
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension", "tf_custom_op_py_strict_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tflite:build_def.bzl", "tf_to_tflite", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["@org_tensorflow//tensorflow:internal"],
    licenses = ["notice"],
)

exports_files(glob([
    "*.pb",
    "*.pbtxt",
]))

tf_to_tflite(
    name = "permute_float",
    src = "permute.pbtxt",
    out = "permute_float.tflite",
    options = [
        "--input_arrays=input",
        "--output_arrays=output",
    ],
)

tf_to_tflite(
    name = "permute_uint8",
    src = "permute.pbtxt",
    out = "permute_uint8.tflite",
    options = [
        "--input_arrays=input",
        "--output_arrays=output",
        "--inference_type=QUANTIZED_UINT8",
        "--std_dev_values=1",
        "--mean_values=0",
        "--default_ranges_min=0",
        "--default_ranges_max=255",
    ],
)

tf_to_tflite(
    name = "gather_string",
    src = "gather.pbtxt",
    out = "gather_string.tflite",
    options = [
        "--input_arrays=input,indices",
        "--output_arrays=output",
    ],
)

tf_to_tflite(
    name = "gather_string_0d",
    src = "gather_0d.pbtxt",
    out = "gather_string_0d.tflite",
    options = [
        "--input_arrays=input,indices",
        "--output_arrays=output",
    ],
)

filegroup(
    name = "interpreter_test_data",
    srcs = [
        "pc_conv.bin",
        "two_signatures.tflite",
        ":gather_string",
        ":gather_string_0d",
        ":permute_float",
        ":permute_uint8",
    ],
    visibility = ["@org_tensorflow//tensorflow:__subpackages__"],
)

cc_library(
    name = "test_delegate",
    testonly = 1,
    srcs = ["test_delegate.cc"],
    visibility = ["//visibility:private"],
    deps = [
        "//tflite/core/c:common",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "test_delegate.so",
    testonly = 1,
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":test_delegate",
    ],
)

cc_library(
    name = "double_op_and_kernels",
    testonly = 1,
    srcs = ["double_op.cc"],
    copts = tflite_copts() + tf_opts_nortti_if_android() + select({
        "@org_tensorflow//tensorflow:android": ["-Wno-private-header"],
        "//conditions:default": [],
    }),
    features = tf_features_nolayering_check_if_ios(),
    deps = select({
        "@org_tensorflow//tensorflow:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "@org_tensorflow//tensorflow:ios": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/core:lib",
        ],
    }),
    alwayslink = 1,
)

tf_custom_op_library(
    name = "_double_op.so",
    testonly = 1,
    srcs = ["double_op.cc"],
)

tf_gen_op_wrapper_py(
    name = "gen_double_op_wrapper",
    testonly = 1,
    out = "double_op_wrapper.py",
    extra_py_deps = [
        "@org_tensorflow//tensorflow/python:pywrap_tfe",
        "@org_tensorflow//tensorflow/python/util:dispatch",
        "@org_tensorflow//tensorflow/python/util:deprecation",
        "@org_tensorflow//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = [":double_op_and_kernels"],
)

tf_custom_op_py_strict_library(
    name = "double_op",
    testonly = 1,
    srcs = ["double_op.py"],
    dso = [":_double_op.so"],
    kernels = [":double_op_and_kernels"],
    deps = [
        ":gen_double_op_wrapper",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:load_library",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
    ],
)

cc_library(
    name = "test_registerer",
    srcs = ["test_registerer.cc"],
    hdrs = ["test_registerer.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tflite:framework",
        "//tflite/kernels:builtin_ops",
        "//tflite/kernels:kernel_util",
        "//tflite/kernels/internal:tensor",
    ],
    alwayslink = 1,
)

pybind_extension(
    name = "_pywrap_test_registerer",
    srcs = [
        "test_registerer_wrapper.cc",
    ],
    hdrs = ["test_registerer.h"],
    additional_exported_symbols = ["TF_TestRegisterer"],
    enable_stub_generation = True,
    link_in_framework = True,
    pytype_srcs = [
        "_pywrap_test_registerer.pyi",
    ],
    deps = [
        ":test_registerer",
        "//tflite:framework",
        "//tflite/kernels:builtin_ops",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)
