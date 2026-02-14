# Internal implementation details of the Apple rules.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "//apple/build_settings:build_settings.bzl",
    "build_settings_labels",
)
load(
    "//apple/internal:apple_toolchains.bzl",
    "apple_mac_tools_toolchain",
    "apple_xplat_tools_toolchain",
)
load(
    ":environment_plist.bzl",
    "environment_plist",
)

licenses(["notice"])

apple_mac_tools_toolchain(
    name = "mac_tools_toolchain",
    alticonstool = "//tools/alticonstool",
    bundletool_experimental = "//tools/bundletool:bundletool_experimental",
    clangrttool = "//tools/clangrttool",
    codesigningtool = "//tools/codesigningtool",
    dossier_codesigningtool = "//tools/dossier_codesigningtool",
    dsym_info_plist_template = "//apple/internal/templates:dsym_info_plist_template",
    environment_plist_tool = "//tools/environment_plist",
    imported_dynamic_framework_processor = "//tools/imported_dynamic_framework_processor",
    main_thread_checker_tool = "//tools/main_thread_checker_tool",
    plisttool = "//tools/plisttool",
    process_and_sign_template = "//tools/bundletool:process_and_sign_template",
    provisioning_profile_tool = "//tools/provisioning_profile_tool",
    swift_stdlib_tool = "//tools/swift_stdlib_tool",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
    xcframework_processor_tool = "//tools/xcframework_processor_tool",
    xctoolrunner = "//tools/xctoolrunner",
)

apple_xplat_tools_toolchain(
    name = "xplat_tools_toolchain",
    build_settings = build_settings_labels.all_labels,
    bundletool = "//tools/bundletool",
    versiontool = "//tools/versiontool",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "apple_framework_import",
    srcs = ["apple_framework_import.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":cc_toolchain_info_support",
        ":framework_import_support",
        ":resources",
        ":rule_attrs",
        "//apple:providers",
        "//apple:utils",
        "//apple/internal/aspects:swift_usage_aspect",
        "@bazel_skylib//lib:collections",
        "@bazel_skylib//lib:dicts",
        "@bazel_skylib//lib:partial",
        "@bazel_skylib//lib:paths",
        "@bazel_skylib//lib:sets",
        "@bazel_tools//tools/cpp:toolchain_utils.bzl",
        "@build_bazel_rules_swift//swift",
    ],
)

bzl_library(
    name = "apple_universal_binary",
    srcs = ["apple_universal_binary.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":linking_support",
        ":providers",
        ":rule_attrs",
        ":rule_factory",
        ":transition_support",
    ],
)

bzl_library(
    name = "apple_product_type",
    srcs = ["apple_product_type.bzl"],
    visibility = [
        "//apple:__subpackages__",
        "//test/starlark_tests:__subpackages__",
    ],
)

bzl_library(
    name = "apple_toolchains",
    srcs = ["apple_toolchains.bzl"],
    visibility = ["//apple:__subpackages__"],
    deps = [
        "@bazel_skylib//rules:common_settings",
    ],
)

bzl_library(
    name = "apple_xcframework_import",
    srcs = ["apple_xcframework_import.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_toolchains",
        ":cc_toolchain_info_support",
        ":framework_import_support",
        ":intermediates",
        ":rule_attrs",
        "//apple:providers",
        "//apple/internal/aspects:swift_usage_aspect",
        "@bazel_skylib//lib:dicts",
        "@bazel_skylib//lib:paths",
        "@bazel_tools//tools/cpp:toolchain_utils.bzl",
        "@build_bazel_apple_support//lib:apple_support",
        "@build_bazel_rules_swift//swift",
    ],
)

bzl_library(
    name = "bundle_package_type",
    srcs = ["bundle_package_type.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
)

bzl_library(
    name = "bundling_support",
    srcs = ["bundling_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple:providers",
    ],
)

bzl_library(
    name = "capabilities_rules",
    srcs = ["capabilities_rules.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":bundling_support",
        ":providers",
        "//apple:providers",
    ],
)

bzl_library(
    name = "cc_info_support",
    srcs = ["cc_info_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
)

bzl_library(
    name = "cc_toolchain_info_support",
    srcs = ["cc_toolchain_info_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
)

bzl_library(
    name = "codesigning_support",
    srcs = ["codesigning_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":intermediates",
        ":rule_support",
        "//apple/internal/utils:defines",
        "@bazel_skylib//lib:paths",
        "@bazel_skylib//lib:shell",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "entitlements_support",
    srcs = ["entitlements_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_product_type",
        ":bundling_support",
        ":resource_actions",
        "//apple:common",
        "//apple/internal/utils:defines",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "environment_plist",
    srcs = ["environment_plist.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_toolchains",
        ":platform_support",
        ":rule_attrs",
        "@bazel_skylib//lib:dicts",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "experimental",
    srcs = ["experimental.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple/internal/utils:defines",
    ],
)

bzl_library(
    name = "features_support",
    srcs = ["features_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "@bazel_skylib//lib:new_sets",
    ],
)

bzl_library(
    name = "framework_import_support",
    srcs = ["framework_import_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple:providers",
        "//apple:utils",
        "//apple/internal:providers",
        "//apple/internal/utils:files",
        "@bazel_skylib//lib:paths",
        "@build_bazel_rules_swift//swift",
    ],
)

bzl_library(
    name = "intermediates",
    srcs = ["intermediates.bzl"],
    visibility = [
        "//apple:__subpackages__",
        "//test:__subpackages__",
    ],
    deps = [
        "@bazel_skylib//lib:paths",
    ],
)

bzl_library(
    name = "ios_rules",
    srcs = ["ios_rules.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_product_type",
        ":apple_toolchains",
        ":bundling_support",
        ":cc_info_support",
        ":entitlements_support",
        ":features_support",
        ":framework_import_support",
        ":linking_support",
        ":outputs",
        ":partials",
        ":platform_support",
        ":processor",
        ":providers",
        ":resources",
        ":rule_attrs",
        ":rule_factory",
        ":rule_support",
        ":run_support",
        ":stub_support",
        ":swift_support",
        ":transition_support",
        "//apple:providers",
        "//apple/internal/aspects:framework_provider_aspect",
        "//apple/internal/aspects:resource_aspect",
        "//apple/internal/aspects:resource_aspect_hint",
        "//apple/internal/utils:clang_rt_dylibs",
        "//apple/internal/utils:main_thread_checker_dylibs",
        "@bazel_skylib//lib:collections",
        "@bazel_tools//tools/cpp:toolchain_utils.bzl",
        "@build_bazel_rules_swift//swift",
    ],
)

bzl_library(
    name = "linking_support",
    srcs = [
        "linking_support.bzl",
        "multi_arch_binary_support.bzl",
    ],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_toolchains",
        ":cc_toolchain_info_support",
        ":entitlements_support",
        ":providers",
        ":rule_support",
        "@bazel_skylib//lib:collections",
        "@build_bazel_apple_support//lib:lipo",
    ],
)

bzl_library(
    name = "macos_binary_support",
    srcs = ["macos_binary_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_product_type",
        ":apple_toolchains",
        ":bundling_support",
        ":intermediates",
        ":linking_support",
        ":platform_support",
        ":resource_actions",
        ":rule_attrs",
        ":rule_support",
        "//apple:providers",
        "@bazel_skylib//lib:dicts",
    ],
)

bzl_library(
    name = "macos_rules",
    srcs = ["macos_rules.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_product_type",
        ":apple_toolchains",
        ":bundling_support",
        ":codesigning_support",
        ":entitlements_support",
        ":features_support",
        ":framework_import_support",
        ":linking_support",
        ":outputs",
        ":partials",
        ":platform_support",
        ":processor",
        ":providers",
        ":resources",
        ":rule_attrs",
        ":rule_factory",
        ":rule_support",
        ":run_support",
        ":swift_support",
        ":transition_support",
        "//apple:providers",
        "//apple/internal/aspects:framework_provider_aspect",
        "//apple/internal/aspects:resource_aspect",
        "//apple/internal/aspects:resource_aspect_hint",
        "//apple/internal/utils:clang_rt_dylibs",
        "//apple/internal/utils:main_thread_checker_dylibs",
    ],
)

bzl_library(
    name = "outputs",
    srcs = ["outputs.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":experimental",
        ":intermediates",
        "@bazel_skylib//lib:paths",
    ],
)

bzl_library(
    name = "partials",
    srcs = ["partials.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple/internal:providers",
        "//apple/internal/partials:app_assets_validation",
        "//apple/internal/partials:app_intents_metadata_bundle",
        "//apple/internal/partials:apple_bundle_info",
        "//apple/internal/partials:apple_symbols_file",
        "//apple/internal/partials:binary",
        "//apple/internal/partials:cc_info_dylibs",
        "//apple/internal/partials:clang_rt_dylibs",
        "//apple/internal/partials:codesigning_dossier",
        "//apple/internal/partials:debug_symbols",
        "//apple/internal/partials:embedded_bundles",
        "//apple/internal/partials:extension_safe_validation",
        "//apple/internal/partials:framework_header_modulemap",
        "//apple/internal/partials:framework_headers",
        "//apple/internal/partials:framework_import",
        "//apple/internal/partials:framework_provider",
        "//apple/internal/partials:macos_additional_contents",
        "//apple/internal/partials:main_thread_checker_dylibs",
        "//apple/internal/partials:messages_stub",
        "//apple/internal/partials:provisioning_profile",
        "//apple/internal/partials:resources",
        "//apple/internal/partials:settings_bundle",
        "//apple/internal/partials:swift_dylibs",
        "//apple/internal/partials:swift_dynamic_framework",
        "//apple/internal/partials:swift_framework",
        "//apple/internal/partials:watchos_stub",
    ],
)

bzl_library(
    name = "platform_support",
    srcs = ["platform_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
)

bzl_library(
    name = "processor",
    srcs = ["processor.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":codesigning_support",
        ":experimental",
        ":intermediates",
        ":outputs",
        "//apple/internal/utils:bundle_paths",
        "//apple/internal/utils:defines",
        "@bazel_skylib//lib:partial",
        "@bazel_skylib//lib:paths",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "providers",
    srcs = ["providers.bzl"],
    visibility = [
        "//apple:__subpackages__",
        "//test:__subpackages__",
    ],
)

bzl_library(
    name = "resource_actions",
    srcs = ["resource_actions.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple/internal/resource_actions:actool",
        "//apple/internal/resource_actions:datamodel",
        "//apple/internal/resource_actions:ibtool",
        "//apple/internal/resource_actions:intent",
        "//apple/internal/resource_actions:metals",
        "//apple/internal/resource_actions:mlmodel",
        "//apple/internal/resource_actions:plist",
        "//apple/internal/resource_actions:png",
        "//apple/internal/resource_actions:texture_atlas",
        "//apple/internal/resource_actions:xcstrings",
    ],
)

bzl_library(
    name = "resources",
    srcs = ["resources.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple:providers",
        "//apple/internal:providers",
        "//apple/internal/partials/support:resources_support",
        "//apple/internal/utils:bundle_paths",
        "@bazel_skylib//lib:partial",
        "@bazel_skylib//lib:paths",
        "@bazel_skylib//lib:types",
    ],
)

bzl_library(
    name = "rule_attrs",
    srcs = ["rule_attrs.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_toolchains",
        ":bundling_support",
        "//apple:common",
        "//apple:providers",
        "//apple/internal/aspects:app_intents_aspect",
        "//apple/internal/aspects:framework_provider_aspect",
        "//apple/internal/aspects:resource_aspect",
        "//apple/internal/aspects:resource_aspect_hint",
        "//apple/internal/aspects:swift_usage_aspect",
        "//apple/internal/testing:apple_test_bundle_support",
        "@bazel_skylib//lib:dicts",
        "@build_bazel_apple_support//lib:apple_support",
        "@build_bazel_rules_swift//swift",
    ],
)

bzl_library(
    name = "rule_factory",
    srcs = ["rule_factory.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_product_type",
        ":rule_attrs",
        ":rule_support",
        ":transition_support",
        "//apple:common",
        "//apple:providers",
        "//apple/internal/aspects:framework_provider_aspect",
        "//apple/internal/aspects:resource_aspect",
        "//apple/internal/aspects:resource_aspect_hint",
        "//apple/internal/aspects:swift_dynamic_framework_aspect",
        "//apple/internal/aspects:swift_usage_aspect",
        "//apple/internal/testing:apple_test_bundle_support",
        "//apple/internal/testing:apple_test_rule_support",
        "@bazel_skylib//lib:dicts",
        "@bazel_tools//tools/cpp:toolchain_utils.bzl",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "rule_support",
    srcs = ["rule_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_product_type",
        ":bundle_package_type",
    ],
)

bzl_library(
    name = "run_support",
    srcs = ["run_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":outputs",
    ],
)

bzl_library(
    name = "stub_support",
    srcs = ["stub_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":intermediates",
        "@build_bazel_apple_support//lib:apple_support",
        "@build_bazel_apple_support//lib:lipo",
    ],
)

bzl_library(
    name = "swift_info_support",
    srcs = ["swift_info_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":intermediates",
        "@bazel_skylib//lib:sets",
        "@build_bazel_rules_swift//swift",
    ],
)

bzl_library(
    name = "swift_support",
    srcs = ["swift_support.bzl"],
    visibility = ["//apple:__subpackages__"],
    deps = [
        "//apple/internal/aspects:swift_usage_aspect",
    ],
)

bzl_library(
    name = "transition_support",
    srcs = ["transition_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple/build_settings",
        "@bazel_skylib//lib:dicts",
        "@build_bazel_apple_support//configs:platforms",
    ],
)

bzl_library(
    name = "tvos_rules",
    srcs = ["tvos_rules.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_product_type",
        ":apple_toolchains",
        ":bundling_support",
        ":cc_info_support",
        ":entitlements_support",
        ":features_support",
        ":framework_import_support",
        ":linking_support",
        ":outputs",
        ":partials",
        ":platform_support",
        ":processor",
        ":providers",
        ":resources",
        ":rule_attrs",
        ":rule_factory",
        ":rule_support",
        ":run_support",
        ":swift_support",
        ":transition_support",
        "//apple:providers",
        "//apple/internal/aspects:framework_provider_aspect",
        "//apple/internal/aspects:resource_aspect",
        "//apple/internal/aspects:resource_aspect_hint",
        "//apple/internal/utils:clang_rt_dylibs",
        "//apple/internal/utils:main_thread_checker_dylibs",
        "@bazel_tools//tools/cpp:toolchain_utils.bzl",
        "@build_bazel_rules_swift//swift",
    ],
)

bzl_library(
    name = "visionos_rules",
    srcs = ["visionos_rules.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_product_type",
        ":apple_toolchains",
        ":bundling_support",
        ":entitlements_support",
        ":features_support",
        ":framework_import_support",
        ":linking_support",
        ":outputs",
        ":partials",
        ":platform_support",
        ":processor",
        ":providers",
        ":resources",
        ":rule_attrs",
        ":rule_factory",
        ":rule_support",
        ":run_support",
        ":stub_support",
        ":swift_support",
        "//apple:providers",
        "//apple/internal/aspects:framework_provider_aspect",
        "//apple/internal/aspects:resource_aspect",
        "//apple/internal/aspects:resource_aspect_hint",
        "//apple/internal/utils:clang_rt_dylibs",
        "//apple/internal/utils:main_thread_checker_dylibs",
        "@bazel_skylib//lib:sets",
        "@bazel_tools//tools/cpp:toolchain_utils.bzl",
    ],
)

bzl_library(
    name = "watchos_rules",
    srcs = ["watchos_rules.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_product_type",
        ":apple_toolchains",
        ":bundling_support",
        ":entitlements_support",
        ":features_support",
        ":framework_import_support",
        ":linking_support",
        ":outputs",
        ":partials",
        ":platform_support",
        ":processor",
        ":providers",
        ":resources",
        ":rule_attrs",
        ":rule_factory",
        ":rule_support",
        ":stub_support",
        ":swift_support",
        "//apple:providers",
        "//apple/internal/aspects:framework_provider_aspect",
        "//apple/internal/aspects:resource_aspect",
        "//apple/internal/aspects:resource_aspect_hint",
        "//apple/internal/utils:clang_rt_dylibs",
        "//apple/internal/utils:main_thread_checker_dylibs",
        "@bazel_skylib//lib:sets",
        "@bazel_tools//tools/cpp:toolchain_utils.bzl",
    ],
)

bzl_library(
    name = "xcframework_rules",
    srcs = ["xcframework_rules.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":apple_product_type",
        ":apple_toolchains",
        ":cc_info_support",
        ":experimental",
        ":intermediates",
        ":linking_support",
        ":outputs",
        ":partials",
        ":platform_support",
        ":processor",
        ":providers",
        ":resources",
        ":rule_attrs",
        ":rule_factory",
        ":rule_support",
        ":swift_support",
        ":transition_support",
        "//apple:providers",
        "//apple/internal/aspects:resource_aspect",
        "//apple/internal/aspects:resource_aspect_hint",
        "//apple/internal/aspects:swift_usage_aspect",
        "//apple/internal/utils:files",
        "@bazel_skylib//lib:partial",
        "@bazel_skylib//lib:paths",
        "@build_bazel_apple_support//lib:apple_support",
        "@build_bazel_rules_swift//swift",
    ],
)

bzl_library(
    name = "local_provisioning_profiles",
    srcs = ["local_provisioning_profiles.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple:providers",
    ],
)

bzl_library(
    name = "xcarchive",
    srcs = ["xcarchive.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple:providers",
        "//apple/internal/providers:apple_debug_info",
    ],
)

bzl_library(
    name = "xctrunner",
    srcs = ["xctrunner.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple:providers",
        "//apple/internal/providers:apple_debug_info",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "docc",
    srcs = ["docc.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "//apple:providers",
        "//apple/internal/aspects:docc_archive_aspect",
        "@bazel_skylib//lib:dicts",
        "@build_bazel_apple_support//lib:apple_support",
    ],
)

bzl_library(
    name = "header_map",
    srcs = ["header_map.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        "@build_bazel_rules_swift//swift",
        "@rules_cc//cc/common",
    ],
)

bzl_library(
    name = "header_map_support",
    srcs = ["header_map_support.bzl"],
    visibility = [
        "//apple:__subpackages__",
    ],
    deps = [
        ":header_map",
    ],
)

# Consumed by bazel tests.
filegroup(
    name = "for_bazel_tests",
    testonly = True,
    srcs = glob(["**"]) + [
        "//apple/internal/aspects:for_bazel_tests",
        "//apple/internal/partials:for_bazel_tests",
        "//apple/internal/providers:for_bazel_tests",
        "//apple/internal/resource_actions:for_bazel_tests",
        "//apple/internal/resource_rules:for_bazel_tests",
        "//apple/internal/templates:for_bazel_tests",
        "//apple/internal/testing:for_bazel_tests",
        "//apple/internal/utils:for_bazel_tests",
    ],
    visibility = [
        "//apple:__subpackages__",
    ],
)

environment_plist(
    name = "environment_plist_ios",
    platform_type = "ios",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)

environment_plist(
    name = "environment_plist_macos",
    platform_type = "macos",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)

environment_plist(
    name = "environment_plist_visionos",
    platform_type = "visionos",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)

environment_plist(
    name = "environment_plist_watchos",
    platform_type = "watchos",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)

environment_plist(
    name = "environment_plist_tvos",
    platform_type = "tvos",
    # Used by the rule implementations, so it needs to be public; but
    # should be considered an implementation detail of the rules and
    # not used by other things.
    visibility = ["//visibility:public"],
)
