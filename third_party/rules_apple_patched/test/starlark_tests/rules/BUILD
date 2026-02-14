load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

bzl_library(
    name = "test_rules",
    srcs = glob(
        include = ["*.bzl"],
    ),
    visibility = [
        "//test:__subpackages__",
    ],
    deps = [
        "//apple:providers",
        "//apple/build_settings",
        "//apple/internal:apple_product_type",
        "//apple/internal:intermediates",
        "//apple/internal:providers",
        "@bazel_skylib//lib:dicts",
        "@bazel_skylib//lib:new_sets",
        "@bazel_skylib//lib:paths",
        "@bazel_skylib//lib:unittest",
        "@build_bazel_apple_support//lib:apple_support",
        "@build_bazel_rules_swift//swift",
    ],
)

sh_binary(
    name = "generate_xcframework_xcodebuild_tool",
    srcs = ["generate_xcframework_xcodebuild_tool.sh"],
    tags = ["requires-darwin"],
)
