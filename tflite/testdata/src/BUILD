load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "intermediate_tensor_output",
    srcs = ["intermediate_tensor_output.py"],
    python_version = "PY3",
    deps = [
        "//tflite/python:lite",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
