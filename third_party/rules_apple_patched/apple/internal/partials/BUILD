load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# Partials.

# This file exists to create a package under which to load the processor
# partials bzl files.

licenses(["notice"])

bzl_library(
    name = "app_assets_validation",
    srcs = ["app_assets_validation.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:apple_product_type",
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "app_intents_metadata_bundle",
    srcs = ["app_intents_metadata_bundle.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:linking_support",
        "//apple/internal:processor",
        "//apple/internal/providers:app_intents_info",
        "//apple/internal/resource_actions:app_intents",
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "apple_bundle_info",
    srcs = ["apple_bundle_info.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:outputs",
        "//apple/internal:providers",
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "apple_symbols_file",
    srcs = ["apple_symbols_file.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:intermediates",
        "//apple/internal:processor",
        "@bazel_skylib//lib:partial",
        "@bazel_skylib//lib:shell",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "binary",
    srcs = ["binary.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:outputs",
        "//apple/internal:processor",
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "cc_info_dylibs",
    srcs = ["cc_info_dylibs.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:intermediates",
        "//apple/internal:processor",
        "@bazel_skylib//lib:partial",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "clang_rt_dylibs",
    srcs = ["clang_rt_dylibs.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:intermediates",
        "//apple/internal:processor",
        "//apple/internal/utils:clang_rt_dylibs",
        "@bazel_skylib//lib:partial",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "main_thread_checker_dylibs",
    srcs = ["main_thread_checker_dylibs.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:intermediates",
        "//apple/internal:processor",
        "//apple/internal/utils:main_thread_checker_dylibs",
        "@bazel_skylib//lib:partial",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "codesigning_dossier",
    srcs = ["codesigning_dossier.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:codesigning_support",
        "//apple/internal:experimental",
        "//apple/internal:intermediates",
        "//apple/internal:outputs",
        "//apple/internal:processor",
        "//apple/internal:providers",
        "@bazel_skylib//lib:dicts",
        "@bazel_skylib//lib:new_sets",
        "@bazel_skylib//lib:partial",
        "@bazel_skylib//lib:paths",
    ],
)

bzl_library(
    name = "debug_symbols",
    srcs = ["debug_symbols.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple:providers",
        "//apple/internal:intermediates",
        "//apple/internal:providers",
        "//apple/internal:resource_actions",
        "//apple/internal/providers:apple_debug_info",
        "//apple/internal/utils:defines",
        "@bazel_skylib//lib:partial",
        "@bazel_skylib//lib:shell",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "embedded_bundles",
    srcs = ["embedded_bundles.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:experimental",
        "//apple/internal:processor",
        "//apple/internal/providers:embeddable_info",
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "extension_safe_validation",
    srcs = ["extension_safe_validation.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "framework_header_modulemap",
    srcs = ["framework_header_modulemap.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:intermediates",
        "//apple/internal:processor",
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "framework_headers",
    srcs = ["framework_headers.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:processor",
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "framework_import",
    srcs = ["framework_import.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple:providers",
        "//apple/internal:codesigning_support",
        "//apple/internal:intermediates",
        "//apple/internal:processor",
        "//apple/internal/utils:bundle_paths",
        "@bazel_skylib//lib:partial",
        "@bazel_skylib//lib:paths",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "framework_provider",
    srcs = ["framework_provider.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:framework_import_support",
        "//apple/internal:providers",
        "@bazel_skylib//lib:partial",
        "@bazel_skylib//lib:paths",
    ],
)

bzl_library(
    name = "macos_additional_contents",
    srcs = ["macos_additional_contents.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple:providers",
        "//apple/internal:processor",
        "//apple/internal/utils:bundle_paths",
        "@bazel_skylib//lib:partial",
        "@bazel_skylib//lib:paths",
    ],
)

bzl_library(
    name = "messages_stub",
    srcs = ["messages_stub.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:intermediates",
        "//apple/internal:processor",
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "provisioning_profile",
    srcs = ["provisioning_profile.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:intermediates",
        "//apple/internal:processor",
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "resources",
    srcs = ["resources.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple:providers",
        "//apple/internal:intermediates",
        "//apple/internal:outputs",
        "//apple/internal:processor",
        "//apple/internal:resource_actions",
        "//apple/internal:resources",
        "//apple/internal/partials/support:resources_support",
        "//apple/internal/utils:bundle_paths",
        "@bazel_skylib//lib:new_sets",
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "settings_bundle",
    srcs = ["settings_bundle.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple:providers",
        "//apple/internal:processor",
        "//apple/internal:resources",
        "//apple/internal/utils:bundle_paths",
        "@bazel_skylib//lib:partial",
    ],
)

bzl_library(
    name = "swift_dylibs",
    srcs = ["swift_dylibs.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:intermediates",
        "//apple/internal:processor",
        "//apple/internal/utils:defines",
        "@bazel_skylib//lib:partial",
        "@bazel_skylib//lib:paths",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "swift_framework",
    srcs = ["swift_framework.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:processor",
        "//apple/internal:swift_info_support",
        "@bazel_skylib//lib:partial",
        "@bazel_skylib//lib:paths",
    ],
)

bzl_library(
    name = "swift_dynamic_framework",
    srcs = ["swift_dynamic_framework.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:intermediates",
        "//apple/internal:processor",
        "@bazel_skylib//lib:partial",
        "@bazel_skylib//lib:paths",
    ],
)

bzl_library(
    name = "watchos_stub",
    srcs = ["watchos_stub.bzl"],
    visibility = [
        "//apple/internal:__pkg__",
    ],
    deps = [
        "//apple/internal:intermediates",
        "//apple/internal:processor",
        "@bazel_skylib//lib:partial",
    ],
)

# Consumed by bazel tests.
filegroup(
    name = "for_bazel_tests",
    testonly = True,
    srcs = glob(["**"]) + [
        "//apple/internal/partials/support:for_bazel_tests",
    ],
    visibility = [
        "//apple:__subpackages__",
    ],
)
