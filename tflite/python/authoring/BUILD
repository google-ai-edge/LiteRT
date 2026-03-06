load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("@xla//third_party/rules_python/python:py_test.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "authoring",
    srcs = [
        "authoring.py",
    ],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tflite/converter/metrics:converter_error_data_proto_py",
        "//tflite/python:convert",
        "//tflite/python:lite",
        "@org_tensorflow//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "authoring_test",
    srcs = ["authoring_test.py"],
    strict_deps = True,
    deps = [
        ":authoring",
        #internal proto upb dep
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tflite/python:lite",
    ],
)
