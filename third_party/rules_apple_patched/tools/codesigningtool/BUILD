load("@build_bazel_apple_support//rules:toolchain_substitution.bzl", "toolchain_substitution")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")

licenses(["notice"])

py_binary(
    name = "codesigningtool",
    srcs = ["codesigningtool.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [":codesigningtool_lib"],
)

py_library(
    name = "codesigningtool_lib",
    srcs = ["codesigningtool.py"],
    srcs_version = "PY3",
    visibility = [
        "//tools:__subpackages__",
    ],
    deps = ["//tools/wrapper_common:execute"],
)

toolchain_substitution(
    name = "disable_signing_resource_rules",
    src = "disable_signing_resource_rules.plist",
    var_name = "RESOURCE_RULES",
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
