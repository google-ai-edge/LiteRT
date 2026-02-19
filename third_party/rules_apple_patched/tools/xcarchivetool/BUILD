load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(
    name = "make_xcarchive",
    srcs = ["make_xcarchive.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = ["//tools/wrapper_common:execute"],
)
