load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# Resource rules.

# This file exists to create a package under which to load the testing
# rules bzl files.

licenses(["notice"])

bzl_library(
    name = "apple_bundle_import",
    srcs = ["apple_bundle_import.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple/internal:providers",
        "//apple/internal:resources",
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "apple_intent_library",
    srcs = ["apple_intent_library.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple/internal:platform_support",
        "//apple/internal:resource_actions",
        "//apple/internal:swift_support",
        "@bazel_skylib//lib:dicts",
        "@bazel_skylib//lib:paths",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "apple_metal_library",
    srcs = ["apple_metal_library.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple/internal:apple_toolchains",
        "//apple/internal:features_support",
        "//apple/internal:platform_support",
        "//apple/internal:resources",
        "@bazel_skylib//lib:dicts",
        "@bazel_skylib//lib:paths",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "apple_core_ml_library",
    srcs = ["apple_core_ml_library.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple/internal:apple_toolchains",
        "//apple/internal:platform_support",
        "//apple/internal:resource_actions",
        "//apple/internal:rule_attrs",
        "//apple/internal:swift_support",
        "@bazel_skylib//lib:dicts",
        "@bazel_skylib//lib:paths",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "apple_core_data_model",
    srcs = ["apple_core_data_model.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple:utils",
        "//apple/internal:apple_toolchains",
        "//apple/internal:platform_support",
        "//apple/internal:resource_actions",
        "//apple/internal:rule_attrs",
        "@bazel_skylib//lib:dicts",
        "@bazel_skylib//lib:paths",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "apple_resource_bundle",
    srcs = ["apple_resource_bundle.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple/internal:providers",
    ],
)

bzl_library(
    name = "apple_precompiled_resource_bundle",
    srcs = ["apple_precompiled_resource_bundle.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple/internal:apple_product_type",
        "//apple/internal:apple_toolchains",
        "//apple/internal:codesigning_support",
        "//apple/internal:features_support",
        "//apple/internal:partials",
        "//apple/internal:platform_support",
        "//apple/internal:processor",
        "//apple/internal:providers",
        "//apple/internal:resources",
        "//apple/internal:rule_attrs",
        "//apple/internal:rule_support",
        "@bazel_skylib//lib:dicts",
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "apple_resource_group",
    srcs = ["apple_resource_group.bzl"],
    visibility = [
        "//apple:__subpackages__",
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
