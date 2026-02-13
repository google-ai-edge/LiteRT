load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

licenses(["notice"])

py_binary(
    name = "dossier_codesigningtool",
    srcs = ["dossier_codesigningtool.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        ":dossier_codesigningtool_lib",
    ],
)

py_library(
    name = "dossier_codesigningtool_lib",
    srcs = ["dossier_codesigningtool.py"],
    srcs_version = "PY3",
    visibility = [
        "//tools:__subpackages__",
    ],
    deps = [
        ":dossier_codesigning_reader_lib",
    ],
)

py_test(
    name = "dossier_codesigningtool_lib_test",
    srcs = ["dossier_codesigningtool_lib_test.py"],
    python_version = "PY3",
    deps = [
        ":dossier_codesigningtool_lib",
    ],
)

py_binary(
    name = "dossier_codesigning_reader",
    srcs = ["dossier_codesigning_reader.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [":dossier_codesigning_reader_lib"],
)

py_library(
    name = "dossier_codesigning_reader_lib",
    srcs = ["dossier_codesigning_reader.py"],
    srcs_version = "PY3",
)

filegroup(
    name = "watchos_dossier_zip",
    testonly = True,
    srcs = ["//test/starlark_tests/targets_under_test/watchos:app_companion"],
    output_group = "combined_dossier_zip",
)

py_test(
    name = "dossier_codesigning_reader_test",
    srcs = ["dossier_codesigning_reader_test.py"],
    data = [
        ":watchos_dossier_zip",
        "//test/starlark_tests/targets_under_test/ios:app",
        "//test/starlark_tests/targets_under_test/watchos:app_companion",
    ],
    python_version = "PY3",
    tags = ["requires-darwin"],
    deps = [
        ":dossier_codesigning_reader_lib",
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
