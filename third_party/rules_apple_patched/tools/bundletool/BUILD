load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

licenses(["notice"])

py_binary(
    name = "bundletool",
    srcs = ["bundletool.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [":bundletool_lib"],
)

py_library(
    name = "bundletool_lib",
    srcs = [
        "__init__.py",
        "bundletool.py",
    ],
    srcs_version = "PY3",
)

py_binary(
    name = "bundletool_experimental",
    srcs = ["bundletool_experimental.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//apple/internal:__pkg__",
    ],
)

py_test(
    name = "bundletool_test",
    srcs = ["bundletool_test.py"],
    python_version = "PY3",
    deps = [":bundletool_lib"],
)

filegroup(
    name = "process_and_sign_template",
    srcs = ["process_and_sign.sh.template"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
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
