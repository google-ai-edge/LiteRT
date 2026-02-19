load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_python//python:py_binary.bzl", "py_binary")
load(
    "//apple/testing/default_runner:ios_test_runner.bzl",
    "ios_test_runner",
)
load(
    "//apple/testing/default_runner:ios_xctestrun_runner.bzl",
    "ios_xctestrun_runner",
)
load(
    "//apple/testing/default_runner:macos_test_runner.bzl",
    "macos_test_runner",
)
load(
    "//apple/testing/default_runner:tvos_test_runner.bzl",
    "tvos_test_runner",
)
load(
    "//apple/testing/default_runner:visionos_test_runner.bzl",
    "visionos_test_runner",
)
load(
    "//apple/testing/default_runner:watchos_test_runner.bzl",
    "watchos_test_runner",
)

licenses(["notice"])

bzl_library(
    name = "ios_test_runner",
    srcs = ["ios_test_runner.bzl"],
    visibility = ["//apple:__pkg__"],
    deps = [
        "//apple:providers",
    ],
)

bzl_library(
    name = "ios_xctestrun_runner",
    srcs = ["ios_xctestrun_runner.bzl"],
    visibility = ["//apple:__pkg__"],
    deps = [
        "//apple:providers",
    ],
)

bzl_library(
    name = "macos_test_runner",
    srcs = ["macos_test_runner.bzl"],
    deps = [
        "//apple:providers",
        "@build_bazel_apple_support//lib:xcode_support",
    ],
)

bzl_library(
    name = "tvos_test_runner",
    srcs = ["tvos_test_runner.bzl"],
    deps = [
        "//apple:providers",
    ],
)

bzl_library(
    name = "visionos_test_runner",
    srcs = ["visionos_test_runner.bzl"],
    deps = [
        "//apple:providers",
    ],
)

bzl_library(
    name = "watchos_test_runner",
    srcs = ["watchos_test_runner.bzl"],
    deps = [
        "//apple:providers",
    ],
)

exports_files([
    "ios_test_runner.template.sh",
    "ios_xctestrun_runner.template.sh",
    "ios_xctestrun_runner.template.xctestrun",
    "macos_test_runner.template.sh",
    "macos_test_runner.template.xctestrun",
    "tvos_test_runner.template.sh",
    "visionos_test_runner.template.sh",
    "watchos_test_runner.template.sh",
    "xctrunner_entitlements.template.plist",
])

py_binary(
    name = "simulator_creator",
    srcs = ["simulator_creator.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)

ios_test_runner(
    name = "ios_default_runner",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)

ios_xctestrun_runner(
    name = "ios_xctestrun_random_runner",
    random = True,
    visibility = ["//visibility:public"],
)

ios_xctestrun_runner(
    name = "ios_xctestrun_ordered_runner",
    random = False,
    visibility = ["//visibility:public"],
)

macos_test_runner(
    name = "macos_default_runner",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)

tvos_test_runner(
    name = "tvos_default_runner",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)

visionos_test_runner(
    name = "visionos_default_runner",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)

watchos_test_runner(
    name = "watchos_default_runner",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)

# Consumed by bazel tests.
filegroup(
    name = "for_bazel_tests",
    testonly = True,
    srcs = glob(["**"]),
    visibility = [
        "//:__subpackages__",
    ],
)
