load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_binary")
load("@org_tensorflow//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "mnist_tflite",
    srcs = ["mnist_tflite.py"],
    deps = [
        ":dataset",
        "//tflite/python:lite",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//third_party/py/numpy",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    strict_deps = True,
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//third_party/py/numpy",
    ],
)
