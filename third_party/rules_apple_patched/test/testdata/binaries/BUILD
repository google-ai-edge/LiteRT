load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//apple:apple_binary.bzl",
    "apple_binary",
)

# Public only because these are used by the integration tests from generated
# workspaces. Please no not depend on them as they can change at any time.
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

apple_binary(
    name = "empty_tvos_dylib",
    binary_type = "dylib",
    minimum_os_version = "11.0",
    platform_type = "tvos",
    deps = [":dummy_lib"],
)

cc_library(
    name = "dummy_lib",
    srcs = ["@bazel_tools//tools/objc:dummy.c"],
)
