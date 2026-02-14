load("@rules_python//python:py_binary.bzl", "py_binary")

licenses(["notice"])

py_binary(
    name = "alticonstool",
    srcs = ["alticonstool.py"],
    srcs_version = "PY3",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
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
