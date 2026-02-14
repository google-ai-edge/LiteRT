load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

licenses(["notice"])

py_binary(
    name = "plisttool",
    srcs = ["plisttool.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [":plisttool_lib"],
)

py_library(
    name = "plisttool_lib",
    srcs = [
        "__init__.py",
        "plisttool.py",
    ],
    srcs_version = "PY3",
)

py_test(
    name = "plisttool_unittest",
    srcs = ["plisttool_unittest.py"],
    python_version = "PY3",
    tags = ["requires-darwin"],
    deps = [":plisttool_lib"],
)

# Consumed by bazel tests.
filegroup(
    name = "for_bazel_tests",
    testonly = True,
    srcs = glob(["**"]),
    visibility = [
        "//tools:__pkg__",
    ],
)
