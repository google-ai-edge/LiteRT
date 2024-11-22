load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "authoring",
    srcs = [
        "authoring.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/python:convert",
        "//tflite/python:lite",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "@org_tensorflow//tensorflow/python/util:tf_export",
    ],
)

py_strict_test(
    name = "authoring_test",
    srcs = ["authoring_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":authoring",
        #internal proto upb dep
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
