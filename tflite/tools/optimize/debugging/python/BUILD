# QuantizationDebugger for TFLite accuracy tooling.
load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "debugger",
    srcs = ["debugger.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/python:convert",
        "//tflite/python:interpreter",
        "//tflite/python/metrics",
        "@org_tensorflow//tensorflow/python/util:tf_export",
        "@org_tensorflow//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "debugger_test",
    srcs = [
        "debugger_test.py",
    ],
    deps = [
        ":debugger",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tflite/python:convert",
        "//tflite/python:lite",
        "//tflite/python/metrics",
        "@org_tensorflow//tensorflow/python/framework:test_lib",
        "@org_tensorflow//tensorflow/python/platform:client_testlib",
        "@org_tensorflow//tensorflow/python/trackable:autotrackable",
    ],
)
