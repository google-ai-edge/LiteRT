load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "calibration_wrapper_lib",
    srcs = ["calibration_wrapper.cc"],
    hdrs = ["calibration_wrapper.h"],
    deps = [
        "//tflite:framework",
        "//tflite:shared_library",
        "//tflite:string_util",
        "//tflite/converter:offset_buffer",
        "//tflite/converter/schema:schema_fbs_with_mutable",
        "//tflite/core:framework",
        "//tflite/core/c:common",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/python/interpreter_wrapper:numpy",
        "//tflite/python/interpreter_wrapper:python_error_reporter",
        "//tflite/python/interpreter_wrapper:python_utils",
        "//tflite/tools/optimize:quantization_wrapper_utils",
        "//tflite/tools/optimize:quantize_model",
        "//tflite/tools/optimize/calibration:calibration_reader",
        "//tflite/tools/optimize/calibration:calibrator_lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

pybind_extension(
    name = "_pywrap_tensorflow_lite_calibration_wrapper",
    srcs = [
        "calibration_wrapper_pybind11.cc",
    ],
    hdrs = ["calibration_wrapper.h"],
    additional_stubgen_deps = [
        "@org_tensorflow//third_party/py/numpy:numpy",
    ],
    common_lib_packages = [
        "litert/python",
        "tflite/python",
    ],
    enable_stub_generation = True,
    link_in_framework = True,
    pytype_srcs = [
        "_pywrap_tensorflow_lite_calibration_wrapper.pyi",
    ],
    wrap_py_init = True,
    deps = [
        ":calibration_wrapper_lib",
        "//tflite:framework",
        "//tflite/core:framework_stable",
        "@local_xla//third_party/python_runtime:headers",
        "@org_tensorflow//tensorflow/python/lib/core:pybind11_lib",
        "@pybind11",
    ],
)

py_strict_library(
    name = "calibrator",
    srcs = [
        "calibrator.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tensorflow_lite_calibration_wrapper",  # buildcleaner: keep
        "//tflite/python:convert_phase",
        "//tflite/python:interpreter",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/util:lazy_loader",
        "@org_tensorflow//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "calibrator_test",
    srcs = ["calibrator_test.py"],
    data = [
        ":test_data",
        "//tflite:testdata/multi_add.bin",
    ],
    tags = ["no_oss"],
    deps = [
        ":calibrator",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py_no_contrib",
        "//tflite/python:lite",
        "//tflite/python:schema_py",
        "//tflite/tools:flatbuffer_utils",
        "@org_tensorflow//tensorflow/python/framework:dtypes",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
    ],
)
