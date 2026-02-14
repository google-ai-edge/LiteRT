load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# Partials support.

# This file exists to create a package under which to load the processor
# partials support bzl files.

licenses(["notice"])

bzl_library(
    name = "resources_support",
    srcs = ["resources_support.bzl"],
    visibility = [
        "//apple/internal:__subpackages__",
    ],
    deps = [
        "//apple:utils",
        "//apple/internal:intermediates",
        "//apple/internal:processor",
        "//apple/internal:resource_actions",
        "@bazel_skylib//lib:paths",
    ],
)

# Consumed by bazel tests.
filegroup(
    name = "for_bazel_tests",
    testonly = True,
    srcs = glob(["**"]),
    visibility = [
        "//apple:__subpackages__",
    ],
)
