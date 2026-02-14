load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# Testing.

# This file exists to create a package under which to load the testing
# rules bzl files.

licenses(["notice"])

bzl_library(
    name = "apple_test_assembler",
    srcs = ["apple_test_assembler.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
)

bzl_library(
    name = "apple_test_bundle_support",
    srcs = ["apple_test_bundle_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple:providers",
        "//apple/internal:apple_product_type",
        "//apple/internal:apple_toolchains",
        "//apple/internal:bundling_support",
        "//apple/internal:experimental",
        "//apple/internal:features_support",
        "//apple/internal:linking_support",
        "//apple/internal:outputs",
        "//apple/internal:partials",
        "//apple/internal:platform_support",
        "//apple/internal:processor",
        "//apple/internal:providers",
        "//apple/internal:resources",
        "//apple/internal:rule_support",
        "//apple/internal:swift_support",
        "//apple/internal/utils:clang_rt_dylibs",
        "@bazel_skylib//lib:types",
        "@build_bazel_rules_swift//swift",
    ],
)

bzl_library(
    name = "apple_test_rule_support",
    srcs = ["apple_test_rule_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple:providers",
        "@bazel_skylib//lib:dicts",
    ],
)

bzl_library(
    name = "build_test_rules",
    srcs = ["build_test_rules.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple/internal:providers",
        "//apple/internal:transition_support",
    ],
)

bzl_library(
    name = "ios_rules",
    srcs = ["ios_rules.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_test_bundle_support",
        ":apple_test_rule_support",
        "//apple:providers",
        "//apple/internal:apple_product_type",
        "//apple/internal:bundling_support",
        "//apple/internal:providers",
        "//apple/internal:rule_attrs",
        "//apple/internal:rule_factory",
        "//apple/internal:transition_support",
        "//apple/internal/aspects:framework_provider_aspect",
        "//apple/internal/aspects:resource_aspect",
    ],
)

bzl_library(
    name = "macos_rules",
    srcs = ["macos_rules.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_test_bundle_support",
        ":apple_test_rule_support",
        "//apple:providers",
        "//apple/internal:apple_product_type",
        "//apple/internal:bundling_support",
        "//apple/internal:providers",
        "//apple/internal:rule_attrs",
        "//apple/internal:rule_factory",
        "//apple/internal:transition_support",
        "//apple/internal/aspects:framework_provider_aspect",
        "//apple/internal/aspects:resource_aspect",
    ],
)

bzl_library(
    name = "tvos_rules",
    srcs = ["tvos_rules.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_test_bundle_support",
        ":apple_test_rule_support",
        "//apple:providers",
        "//apple/internal:apple_product_type",
        "//apple/internal:bundling_support",
        "//apple/internal:providers",
        "//apple/internal:rule_attrs",
        "//apple/internal:rule_factory",
        "//apple/internal:transition_support",
        "//apple/internal/aspects:framework_provider_aspect",
        "//apple/internal/aspects:resource_aspect",
    ],
)

bzl_library(
    name = "visionos_rules",
    srcs = ["visionos_rules.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_test_bundle_support",
        ":apple_test_rule_support",
        "//apple:providers",
        "//apple/internal:apple_product_type",
        "//apple/internal:bundling_support",
        "//apple/internal:providers",
        "//apple/internal:rule_attrs",
        "//apple/internal:rule_factory",
        "//apple/internal:transition_support",
        "//apple/internal/aspects:framework_provider_aspect",
        "//apple/internal/aspects:resource_aspect",
    ],
)

bzl_library(
    name = "watchos_rules",
    srcs = ["watchos_rules.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_test_bundle_support",
        ":apple_test_rule_support",
        "//apple:providers",
        "//apple/internal:apple_product_type",
        "//apple/internal:bundling_support",
        "//apple/internal:providers",
        "//apple/internal:rule_attrs",
        "//apple/internal:rule_factory",
        "//apple/internal:transition_support",
        "//apple/internal/aspects:framework_provider_aspect",
        "//apple/internal/aspects:resource_aspect",
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
