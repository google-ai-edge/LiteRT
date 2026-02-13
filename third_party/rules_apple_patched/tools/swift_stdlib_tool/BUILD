load("@rules_python//python:py_binary.bzl", "py_binary")

licenses(["notice"])

py_binary(
    name = "swift_stdlib_tool",
    srcs = ["swift_stdlib_tool.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//tools/bitcode_strip",
        "//tools/wrapper_common:execute",
        "//tools/wrapper_common:lipo",
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
