load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_test")
load("@org_tensorflow//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

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

py_strict_test(
    name = "authoring_test",
    srcs = ["authoring_test.py"],
    deps = [
        ":authoring",
        #internal proto upb dep
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tflite/python:lite",
    ],
)
