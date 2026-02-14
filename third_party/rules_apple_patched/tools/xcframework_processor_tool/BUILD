load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

licenses(["notice"])

py_binary(
    name = "xcframework_processor_tool",
    srcs = ["xcframework_processor_tool.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [":xcframework_processor_tool_lib"],
)

py_library(
    name = "xcframework_processor_tool_lib",
    srcs = ["xcframework_processor_tool.py"],
    srcs_version = "PY3",
    visibility = [
        "//tools:__subpackages__",
    ],
)

py_test(
    name = "xcframework_processor_tool_test",
    srcs = ["xcframework_processor_tool_test.py"],
    python_version = "PY3",
    tags = ["requires-darwin"],
    deps = [":xcframework_processor_tool_lib"],
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
