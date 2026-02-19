load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//apple/internal/aspects:resource_aspect_hint.bzl", "apple_resource_hint", "apple_resource_hint_action")
load(":cc_toolchain_forwarder.bzl", "cc_toolchain_forwarder")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Create a set of public config_settings that can be used in select()s.
[
    alias(
        name = arch,
        actual = "@build_bazel_apple_support//configs:{}".format(arch),
    )
    for arch in [
        "darwin_x86_64",
        "darwin_arm64",
        "darwin_arm64e",
        "ios_x86_64",
        "ios_arm64",
        "ios_arm64e",
        "ios_sim_arm64",
        "tvos_x86_64",
        "tvos_arm64",
        "tvos_sim_arm64",
        "watchos_arm64",
        "watchos_armv7k",
        "watchos_arm64_32",
        "watchos_x86_64",
        "watchos_device_arm64",
        "watchos_device_arm64e",
    ]
]

bzl_library(
    name = "apple",
    srcs = ["apple.bzl"],
    deps = [
        "//apple:apple_static_library",
        "//apple/internal:apple_framework_import",
        "//apple/internal:apple_universal_binary",
        "//apple/internal:apple_xcframework_import",
        "//apple/internal:local_provisioning_profiles",
        "//apple/internal:xcframework_rules",
        "@bazel_features//:features",
        "@rules_cc//cc:find_cc_toolchain_bzl",
        "@rules_cc//cc/common",
    ],
)

bzl_library(
    name = "apple_binary",
    srcs = ["apple_binary.bzl"],
    deps = [
        "//apple/internal:linking_support",
        "//apple/internal:rule_attrs",
        "//apple/internal:rule_factory",
        "//apple/internal:transition_support",
    ],
)

bzl_library(
    name = "apple_static_library",
    srcs = ["apple_static_library.bzl"],
    deps = [
        ":providers",
        "//apple/internal:linking_support",
        "//apple/internal:providers",
        "//apple/internal:rule_attrs",
        "//apple/internal:rule_factory",
        "//apple/internal:transition_support",
    ],
)

bzl_library(
    name = "aspects",
    srcs = ["aspects.bzl"],
    deps = [
        "//apple/internal/aspects:framework_provider_aspect",
    ],
)

bzl_library(
    name = "capabilities",
    srcs = ["capabilities.bzl"],
    deps = [
        "//apple/internal:capabilities_rules",
    ],
)

bzl_library(
    name = "cc_toolchain_forwarder",
    srcs = ["cc_toolchain_forwarder.bzl"],
    deps = [
        ":providers",
        "//apple/internal:providers",
        "@bazel_tools//tools/cpp:toolchain_utils.bzl",
    ],
)

bzl_library(
    name = "common",
    srcs = ["common.bzl"],
)

bzl_library(
    name = "dtrace",
    srcs = ["dtrace.bzl"],
    deps = [
        "//apple/internal/utils:bundle_paths",
        "@bazel_skylib//lib:dicts",
        "@bazel_skylib//lib:paths",
        "@build_bazel_apple_support//lib:apple_support",
        "@rules_cc//cc/common",
    ],
)

bzl_library(
    name = "ios",
    srcs = ["ios.bzl"],
    deps = [
        "//apple/internal:apple_product_type",
        "//apple/internal:ios_rules",
        "//apple/internal/testing:apple_test_assembler",
        "//apple/internal/testing:build_test_rules",
        "//apple/internal/testing:ios_rules",
        "@rules_cc//cc:find_cc_toolchain_bzl",
        "@rules_cc//cc/common",
    ],
)

bzl_library(
    name = "ios.doc",
    srcs = ["ios.doc.bzl"],
    deps = [
        ":ios",
        "//apple/testing/default_runner:ios_test_runner",
        "//apple/testing/default_runner:ios_xctestrun_runner",
    ],
)

bzl_library(
    name = "macos",
    srcs = ["macos.bzl"],
    deps = [
        "//apple/internal:capabilities_rules",
        "//apple/internal:macos_binary_support",
        "//apple/internal:macos_rules",
        "//apple/internal/testing:apple_test_assembler",
        "//apple/internal/testing:build_test_rules",
        "//apple/internal/testing:macos_rules",
        "@rules_cc//cc:find_cc_toolchain_bzl",
        "@rules_cc//cc/common",
    ],
)

bzl_library(
    name = "macos.doc",
    srcs = ["macos.doc.bzl"],
    deps = ["macos"],
)

bzl_library(
    name = "providers",
    srcs = ["providers.bzl"],
    deps = [
        "//apple/internal:providers",
    ],
)

bzl_library(
    name = "repositories",
    srcs = ["repositories.bzl"],
)

bzl_library(
    name = "resources",
    srcs = ["resources.bzl"],
    deps = [
        "//apple/internal/resource_rules:apple_bundle_import",
        "//apple/internal/resource_rules:apple_core_data_model",
        "//apple/internal/resource_rules:apple_core_ml_library",
        "//apple/internal/resource_rules:apple_intent_library",
        "//apple/internal/resource_rules:apple_metal_library",
        "//apple/internal/resource_rules:apple_precompiled_resource_bundle",
        "//apple/internal/resource_rules:apple_resource_bundle",
        "//apple/internal/resource_rules:apple_resource_group",
        "@rules_cc//cc:action_names.bzl",
        "@rules_cc//cc:core_rules",
        "@rules_cc//cc:find_cc_toolchain_bzl",
        "@rules_cc//cc/common",
    ],
)

bzl_library(
    name = "tvos",
    srcs = ["tvos.bzl"],
    deps = [
        "//apple/internal:tvos_rules",
        "//apple/internal/testing:apple_test_assembler",
        "//apple/internal/testing:build_test_rules",
        "//apple/internal/testing:tvos_rules",
        "@rules_cc//cc:find_cc_toolchain_bzl",
        "@rules_cc//cc/common",
    ],
)

bzl_library(
    name = "tvos.doc",
    srcs = ["tvos.doc.bzl"],
    deps = ["tvos"],
)

bzl_library(
    name = "utils",
    srcs = ["utils.bzl"],
)

bzl_library(
    name = "versioning",
    srcs = ["versioning.bzl"],
    deps = [
        "//apple/internal:apple_toolchains",
        "//apple/internal:providers",
        "@bazel_skylib//lib:dicts",
    ],
)

bzl_library(
    name = "visionos",
    srcs = ["visionos.bzl"],
    deps = [
        "//apple/internal:visionos_rules",
        "//apple/internal/testing:apple_test_assembler",
        "//apple/internal/testing:build_test_rules",
        "//apple/internal/testing:visionos_rules",
        "@rules_cc//cc:find_cc_toolchain_bzl",
        "@rules_cc//cc/common",
    ],
)

bzl_library(
    name = "visionos.doc",
    srcs = ["visionos.doc.bzl"],
    deps = ["visionos"],
)

bzl_library(
    name = "watchos",
    srcs = ["watchos.bzl"],
    deps = [
        "//apple/internal:watchos_rules",
        "//apple/internal/testing:apple_test_assembler",
        "//apple/internal/testing:build_test_rules",
        "//apple/internal/testing:watchos_rules",
        "@rules_cc//cc:find_cc_toolchain_bzl",
        "@rules_cc//cc/common",
    ],
)

bzl_library(
    name = "watchos.doc",
    srcs = ["watchos.doc.bzl"],
    deps = ["watchos"],
)

bzl_library(
    name = "xcarchive",
    srcs = ["xcarchive.bzl"],
    deps = ["//apple/internal:xcarchive"],
)

bzl_library(
    name = "xctrunner",
    srcs = ["xctrunner.bzl"],
    deps = ["//apple/internal:xctrunner"],
)

bzl_library(
    name = "docc",
    srcs = ["docc.bzl"],
    deps = ["//apple/internal:docc"],
)

bzl_library(
    name = "header_map",
    srcs = ["header_map.bzl"],
    deps = ["//apple/internal:header_map"],
)

cc_toolchain_forwarder(
    name = "default_cc_toolchain_forwarder",
)

# Consumed by bazel tests.
filegroup(
    name = "for_bazel_tests",
    testonly = True,
    srcs = glob(["**"]) + [
        "//apple/build_settings:for_bazel_tests",
        "//apple/internal:for_bazel_tests",
        "//apple/testing:for_bazel_tests",
    ],
    visibility = ["//:__pkg__"],
)

# An aspect hint that enables runfile inclusion as AppleResources
# for cc_libraries. Runfiles keep their folder structure.
apple_resource_hint(
    name = "use_resources",
    action = apple_resource_hint_action.resources,
)

apple_resource_hint(
    name = "use_runfiles",
    action = apple_resource_hint_action.runfiles,
)

apple_resource_hint(
    name = "suppress_resources",
    action = apple_resource_hint_action.suppress,
)
