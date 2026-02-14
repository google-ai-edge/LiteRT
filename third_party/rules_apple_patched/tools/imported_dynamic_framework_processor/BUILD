load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

licenses(["notice"])

py_binary(
    name = "imported_dynamic_framework_processor",
    srcs = ["imported_dynamic_framework_processor.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//tools/imported_dynamic_framework_processor:imported_dynamic_framework_processor_lib",
    ],
)

py_library(
    name = "imported_dynamic_framework_processor_lib",
    srcs = ["imported_dynamic_framework_processor.py"],
    srcs_version = "PY3",
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//tools/bitcode_strip",
        "//tools/codesigningtool:codesigningtool_lib",
        "//tools/wrapper_common:execute",
        "//tools/wrapper_common:lipo",
    ],
)

py_test(
    name = "imported_dynamic_framework_processor_test",
    srcs = ["imported_dynamic_framework_processor_test.py"],
    python_version = "PY3",
    tags = ["requires-darwin"],
    deps = [":imported_dynamic_framework_processor_lib"],
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
