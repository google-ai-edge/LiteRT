load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_test.bzl", "py_test")

licenses(["notice"])

py_binary(
    name = "clangrttool",
    srcs = [
        "__init__.py",
        "clangrttool.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//apple/internal:__pkg__",
    ],
)

py_test(
    name = "clangrttool_unittest",
    srcs = ["clangrttool_unittest.py"],
    python_version = "PY3",
    deps = [":clangrttool"],
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
