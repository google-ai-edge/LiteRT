load("@rules_python//python:py_binary.bzl", "py_binary")

licenses(["notice"])

py_binary(
    name = "provisioning_profile_tool",
    srcs = ["provisioning_profile_tool.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
