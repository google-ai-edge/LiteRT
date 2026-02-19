load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# Resource actions.

# This file exists to create a package under which to load the resource actions
# bzl files.

licenses(["notice"])

bzl_library(
    name = "actool",
    srcs = ["actool.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple:utils",
        "//apple/internal:apple_product_type",
        "//apple/internal:bundling_support",
        "//apple/internal/utils:xctoolrunner",
        "@bazel_skylib//lib:paths",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "app_intents",
    srcs = ["app_intents.bzl"],
    visibility = [
        "//apple/internal/partials:__pkg__",
    ],
    deps = [
        "//apple/internal:intermediates",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "datamodel",
    srcs = ["datamodel.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal/utils:xctoolrunner",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "ibtool",
    srcs = ["ibtool.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal/utils:xctoolrunner",
        "@bazel_skylib//lib:paths",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "intent",
    srcs = ["intent.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "@bazel_skylib//lib:versions",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "metals",
    srcs = ["metals.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "@bazel_skylib//lib:paths",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "mlmodel",
    srcs = ["mlmodel.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal/utils:xctoolrunner",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "plist",
    srcs = ["plist.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple:providers",
        "//apple/internal:intermediates",
        "//apple/internal:platform_support",
        "@bazel_skylib//lib:paths",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "png",
    srcs = ["png.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "texture_atlas",
    srcs = ["texture_atlas.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "xcstrings",
    srcs = ["xcstrings.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal/utils:xctoolrunner",
        "@build_bazel_apple_support//lib:apple_support",
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
