load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "stack_trace_example",
    srcs = ["stack_trace_example.py"],
    deps = [
        "//tflite/python:lite",
        "@absl_py//absl:app",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
