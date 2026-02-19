load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

licenses(["notice"])

py_binary(
    name = "versiontool",
    srcs = [
        "__init__.py",
        "versiontool.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [":versiontool_lib"],
)

py_library(
    name = "versiontool_lib",
    srcs = ["versiontool.py"],
    srcs_version = "PY3",
)

py_test(
    name = "versiontool_unittest",
    srcs = ["versiontool_unittest.py"],
    python_version = "PY3",
    deps = [":versiontool_lib"],
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
