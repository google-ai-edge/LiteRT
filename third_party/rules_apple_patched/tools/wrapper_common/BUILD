load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

licenses(["notice"])

py_library(
    name = "execute",
    srcs = [
        "__init__.py",
        "execute.py",
    ],
    srcs_version = "PY3",
    visibility = [
        "//tools/bitcode_strip:__pkg__",
        "//tools/codesigningtool:__pkg__",
        "//tools/imported_dynamic_framework_processor:__pkg__",
        "//tools/swift_stdlib_tool:__pkg__",
        "//tools/xcarchivetool:__pkg__",
        "//tools/xctoolrunner:__pkg__",
        "//tools/xctrunnertool:__pkg__",
    ],
)

py_test(
    name = "execute_test",
    srcs = ["execute_test.py"],
    python_version = "PY3",
    tags = ["requires-darwin"],
    deps = [":execute"],
)

py_library(
    name = "lipo",
    srcs = ["lipo.py"],
    srcs_version = "PY3",
    visibility = [
        "//tools:__subpackages__",
    ],
    deps = [":execute"],
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
