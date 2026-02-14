load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//apple:dtrace.bzl",
    "dtrace_compile",
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

dtrace_compile(
    name = "dtrace",
    srcs = [
        "folder1/probes.d",
        "folder2/probes.d",
    ],
    tags = common.fixture_tags,
)

objc_library(
    name = "dtrace_lib",
    srcs = ["dtrace_test.m"],
    tags = common.fixture_tags,
    deps = [":dtrace"],
)
