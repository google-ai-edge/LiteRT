load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("//tools:binary_env.bzl", "binary_env")

licenses(["notice"])

py_binary(
    name = "xctoolrunner",
    srcs = ["xctoolrunner.py"],
    env = binary_env.common_python_utf8_env(),
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [":xctoolrunner_lib"],
)

py_library(
    name = "xctoolrunner_lib",
    srcs = [
        "__init__.py",
        "xctoolrunner.py",
    ],
    srcs_version = "PY3",
    deps = ["//tools/wrapper_common:execute"],
)

py_test(
    name = "xctoolrunner_test",
    srcs = ["xctoolrunner_test.py"],
    python_version = "PY3",
    tags = ["requires-darwin"],
    deps = [
        ":xctoolrunner_lib",
        "//tools/wrapper_common:execute",
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
