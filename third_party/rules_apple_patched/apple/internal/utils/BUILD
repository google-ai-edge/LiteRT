load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# Utils.

# This file exists to create a package under which to load the utility
# bzl files.

licenses(["notice"])

package(
    default_visibility = ["//apple:__subpackages__"],
)

bzl_library(
    name = "bundle_paths",
    srcs = ["bundle_paths.bzl"],
    deps = ["@bazel_skylib//lib:paths"],
)

bzl_library(
    name = "clang_rt_dylibs",
    srcs = ["clang_rt_dylibs.bzl"],
    deps = ["@bazel_tools//tools/cpp:toolchain_utils.bzl"],
)

bzl_library(
    name = "main_thread_checker_dylibs",
    srcs = ["main_thread_checker_dylibs.bzl"],
    deps = ["@bazel_tools//tools/cpp:toolchain_utils.bzl"],
)

bzl_library(
    name = "defines",
    srcs = ["defines.bzl"],
)

bzl_library(
    name = "files",
    srcs = ["files.bzl"],
    deps = ["@bazel_skylib//lib:paths"],
)

bzl_library(
    name = "xctoolrunner",
    srcs = ["xctoolrunner.bzl"],
)

# Consumed by bazel tests.
filegroup(
    name = "for_bazel_tests",
    testonly = True,
    srcs = glob(["**"]),
)
