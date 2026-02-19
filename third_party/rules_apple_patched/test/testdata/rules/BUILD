load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# Public only because these are used by the integration tests from generated
# workspaces. Please no not depend on them as they can change at any time.
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "dummy_runner",
    srcs = [
        "BUILD",
        "dummy_test_runner.bzl",
        "dummy_test_runner.template",
    ],
)

exports_files([
    "dummy_test_runner.template",
])

bzl_library(
    name = "dummy_test_runner",
    srcs = ["dummy_test_runner.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        "//apple:providers",
    ],
)
