# Build settings used throughout rules_apple build rules.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "@bazel_skylib//rules:common_settings.bzl",
    "bool_flag",
    "bool_setting",
    "string_flag",
)
load(
    "//apple/build_settings:build_settings.bzl",
    "build_flags",
    "build_settings",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Flags are configurable from the command line; pure "*_setting"s are not. Rely on this distinction
# to keep Starlark custom build settings de-facto "private" vs public while keeping them accessible
# to the toolchains.
[
    bool_flag(
        name = build_flag_name,
        build_setting_default = build_flag_info.default,
    )
    for build_flag_name, build_flag_info in build_flags.items()
    if type(build_flag_info.default) == "bool"
]

[
    bool_setting(
        name = build_setting_name,
        build_setting_default = build_setting_info.default,
    )
    for build_setting_name, build_setting_info in build_settings.items()
    if type(build_setting_info.default) == "bool"
]

[
    string_flag(
        name = build_flag_name,
        build_setting_default = build_flag_info.default,
    )
    for build_flag_name, build_flag_info in build_flags.items()
    if type(build_flag_info.default) == "string"
]

bzl_library(
    name = "build_settings",
    srcs = ["build_settings.bzl"],
    visibility = [
        "//:__subpackages__",
    ],
    deps = [
        "@bazel_skylib//lib:dicts",
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
