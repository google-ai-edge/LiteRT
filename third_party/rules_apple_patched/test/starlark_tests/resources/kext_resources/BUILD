load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//test/starlark_tests:common.bzl",
    "common",
)

licenses(["notice"])

package(
    default_visibility = ["//test/starlark_tests:__subpackages__"],
)

# Exports all files in this package as targets to make it easier to depend on them.
# Because these are for Starlark tests, we can scope the visibility to just this
# package.
exports_files(
    glob(["**"]),
    visibility = ["//test/starlark_tests:__subpackages__"],
)

cc_library(
    name = "kext_lib",
    srcs = [
        "KextBuilder.cc",
        "KextBuilder.h",
        "kext-builder_info.cc",
    ],
    copts = [
        "-mkernel",
        "-fapple-kext",
        "-isystem__BAZEL_XCODE_SDKROOT__/System/Library/Frameworks/Kernel.framework/Headers",
    ],
    tags = common.fixture_tags,
    alwayslink = True,
)
