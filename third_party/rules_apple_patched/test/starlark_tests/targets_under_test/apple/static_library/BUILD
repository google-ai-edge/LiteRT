load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//apple:apple_static_library.bzl",
    "apple_static_library",
)
load(
    "//test/starlark_tests:common.bzl",
    "common",
)

licenses(["notice"])

package(
    default_testonly = True,
    default_visibility = ["//test/starlark_tests:__subpackages__"],
)

objc_library(
    name = "main_lib",
    srcs = ["@bazel_tools//tools/objc:objc_dummy.mm"],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

apple_static_library(
    name = "example_library_oldest_supported_ios",
    minimum_os_version = common.min_os_ios.oldest_supported,
    platform_type = "ios",
    tags = common.fixture_tags,
    deps = [":main_lib"],
)

apple_static_library(
    name = "example_library_arm_sim_support",
    minimum_os_version = common.min_os_ios.arm_sim_support,
    platform_type = "ios",
    tags = common.fixture_tags,
    deps = [":main_lib"],
)

apple_static_library(
    name = "example_watch_library_arm_sim_support",
    minimum_os_version = common.min_os_watchos.arm_sim_support,
    platform_type = "watchos",
    tags = common.fixture_tags,
    deps = [":main_lib"],
)

apple_static_library(
    name = "example_library_with_avoid_deps",
    avoid_deps = ["//test/starlark_tests/targets_under_test/apple:StaticFmwkLowerLib"],
    minimum_os_version = common.min_os_ios.baseline,
    platform_type = "ios",
    tags = common.fixture_tags,
    deps = ["//test/starlark_tests/targets_under_test/apple:StaticFmwkUpperLib"],
)

apple_static_library(
    name = "example_library_with_cc_avoid_deps",
    avoid_deps = ["//test/starlark_tests/targets_under_test/apple:StaticFmwkCcLowerLib"],
    minimum_os_version = common.min_os_ios.baseline,
    platform_type = "ios",
    tags = common.fixture_tags,
    deps = ["//test/starlark_tests/targets_under_test/apple:StaticFmwkCcUpperLib"],
)

apple_static_library(
    name = "example_vision_library",
    minimum_os_version = common.min_os_visionos.baseline,
    platform_type = "visionos",
    tags = common.fixture_tags,
    deps = [":main_lib"],
)
