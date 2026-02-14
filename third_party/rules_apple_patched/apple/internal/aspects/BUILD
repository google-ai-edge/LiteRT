load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# Aspects.

# This file exists to create a package under which to load the aspect
# bzl files.

licenses(["notice"])

bzl_library(
    name = "app_intents_aspect",
    srcs = ["app_intents_aspect.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple/internal:cc_info_support",
        "//apple/internal/providers:app_intents_info",
        "@bazel_skylib//lib:collections",
        "@build_bazel_apple_support//lib:apple_support",
        "@build_bazel_rules_swift//swift",
        "@build_bazel_rules_swift//swift:module_name",
    ],
)

bzl_library(
    name = "framework_provider_aspect",
    srcs = ["framework_provider_aspect.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple:providers",
        "//apple/internal/providers:embeddable_info",
    ],
)

bzl_library(
    name = "resource_aspect",
    srcs = ["resource_aspect.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
        "//apple/internal/testing:__pkg__",
    ],
    deps = [
        "//apple:providers",
        "//apple/internal:apple_toolchains",
        "//apple/internal:platform_support",
        "//apple/internal:providers",
        "//apple/internal:resources",
        "//apple/internal:swift_support",
        "//apple/internal/providers:apple_debug_info",
        "//apple/internal/providers:framework_import_bundle_info",
        "@bazel_skylib//lib:dicts",
        "@bazel_skylib//lib:partial",
        "@build_bazel_apple_support//lib:apple_support",
        "@build_bazel_rules_swift//swift",
    ],
)

bzl_library(
    name = "resource_aspect_hint",
    srcs = ["resource_aspect_hint.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
        "//apple/internal/testing:__pkg__",
    ],
    deps = [
        "@build_bazel_rules_swift//swift",
    ],
)

bzl_library(
    name = "swift_dynamic_framework_aspect",
    srcs = ["swift_dynamic_framework_aspect.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "@build_bazel_rules_swift//swift",
    ],
)

bzl_library(
    name = "swift_usage_aspect",
    srcs = ["swift_usage_aspect.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "@build_bazel_rules_swift//swift",
    ],
)

bzl_library(
    name = "docc_archive_aspect",
    srcs = ["docc_archive_aspect.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple:providers",
        "@build_bazel_rules_swift//swift",
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
