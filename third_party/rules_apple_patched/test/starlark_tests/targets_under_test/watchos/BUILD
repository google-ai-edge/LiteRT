load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//apple:apple.bzl",
    "apple_dynamic_framework_import",
    "apple_dynamic_xcframework_import",
    "apple_static_framework_import",
    "apple_static_xcframework_import",
)
load(
    "//apple:ios.bzl",
    "ios_application",
)
load(
    "//apple:watchos.bzl",
    "watchos_application",
    "watchos_dynamic_framework",
    "watchos_extension",
    "watchos_framework",
    "watchos_static_framework",
    "watchos_ui_test",
    "watchos_unit_test",
)
load(
    "//test/starlark_tests:common.bzl",
    "common",
)
load(
    "//test/starlark_tests/rules:generate_framework.bzl",
    "generate_import_framework",
)
load(
    "//test/starlark_tests/rules:generate_xcframework.bzl",
    "generate_dynamic_xcframework",
    "generate_static_xcframework",
)

licenses(["notice"])

package(
    default_testonly = True,
    default_visibility = ["//test/starlark_tests:__subpackages__"],
)

watchos_application(
    name = "app",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    extension = ":ext",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
)

watchos_application(
    name = "app_with_ext_with_invalid_watchos_version",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    extension = ":ext",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.requires_single_target_app,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
)

watchos_application(
    name = "single_target_app",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.single_target_app,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:localization",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

watchos_application(
    name = "app_with_icon_bundle",
    app_icons = ["//test/starlark_tests/resources:icon_bundle_assets"],
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_watchos.icon_bundle_required,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

watchos_application(
    name = "app_with_icon_bundle_only_for_low_minimum_os_version",
    app_icons = ["//test/starlark_tests/resources:icon_bundle_assets"],
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_watchos.single_target_app,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

watchos_application(
    name = "app_with_icon_bundle_and_xcassets_app_icons",
    app_icons = [
        "//test/starlark_tests/resources:WatchAppIcon.xcassets",
        "//test/starlark_tests/resources:icon_bundle_assets",
    ],
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_watchos.single_target_app,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

watchos_extension(
    name = "generic_ext",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_watchos.single_target_app,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "ios_app_with_watchos_app_with_generic_ext",
    bundle_id = "com.google",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    watch_application = ":single_target_app_with_generic_ext",
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

watchos_application(
    name = "single_target_app_with_extension",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    extensions = [":watchos_app_extension"],
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.single_target_app,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:localization",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

watchos_application(
    name = "app_with_ext_with_imported_fmwk",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    extension = ":ext_with_imported_fmwk",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    tags = common.fixture_tags,
)

watchos_application(
    name = "single_target_app_with_imported_fmwk",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_watchos.single_target_app,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":dynamic_fmwk_depending_lib",
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

watchos_application(
    name = "app_arm64_support",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.arm64_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

watchos_extension(
    name = "ext",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = [
        "//test/starlark_tests/resources:complications_watchos",
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:localization",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_ext_main_lib",
    ],
)

watchos_extension(
    name = "ext_arm64_support",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.arm64_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_ext_main_lib",
    ],
)

watchos_extension(
    name = "ext_with_fmwk",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_ext_main_lib",
    ],
)

watchos_extension(
    name = "ext_with_two_fmwk_provisioned",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    frameworks = [
        ":fmwk_with_provisioning",
        ":second_fmwk_with_provisioning",
    ],
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_ext_main_lib",
    ],
)

watchos_extension(
    name = "ext_with_fmwk_with_fmwk_provisioned",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    frameworks = [":fmwk_with_fmwk_with_provisioning"],
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_ext_main_lib",
    ],
)

watchos_extension(
    name = "ext_with_fmwk_with_fmwk",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    frameworks = [":fmwk_with_fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_ext_main_lib",
    ],
)

watchos_extension(
    name = "ext_with_imported_fmwk",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":dynamic_fmwk_depending_lib",
        "//test/starlark_tests/resources:watchkit_ext_main_lib",
    ],
)

watchos_extension(
    name = "ext_with_runtime_framework_using_import_static_lib_dep",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_imported_static_framework"],
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_watchos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

watchos_extension(
    name = "ext_multiple_infoplists",
    bundle_id = "com.google.example.ext",
    infoplists = [
        "//test/starlark_tests/resources:Another.plist",
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    tags = common.fixture_tags,
)

ios_application(
    name = "app_companion",
    bundle_id = "com.google",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    visibility = [
        "//test/starlark_tests:__subpackages__",
        "//tools/dossier_codesigningtool:__pkg__",
    ],
    watch_application = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_companion_arm64_support",
    bundle_id = "com.google",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    visibility = [
        "//test/starlark_tests:__subpackages__",
        "//tools/dossier_codesigningtool:__pkg__",
    ],
    watch_application = ":app_arm64_support",
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------

watchos_framework(
    name = "fmwk",
    bundle_id = "com.google.example.framework",
    extension_safe = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

watchos_framework(
    name = "fmwk_dead_stripped",
    bundle_id = "com.google.example.framework",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionShared.exp",
    ],
    extension_safe = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-dead_strip"],
    minimum_os_version = common.min_os_watchos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

watchos_framework(
    name = "fmwk_with_provisioning",
    bundle_id = "com.google.example.framework",
    extension_safe = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

watchos_framework(
    name = "second_fmwk_with_provisioning",
    bundle_id = "com.google.example.frameworktoo",
    extension_safe = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

watchos_framework(
    name = "fmwk_with_fmwk",
    bundle_id = "com.google.example.frameworkception",
    extension_safe = True,
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

watchos_framework(
    name = "fmwk_with_fmwk_with_provisioning",
    bundle_id = "com.google.example.frameworkception",
    extension_safe = True,
    frameworks = [":fmwk_with_provisioning"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

watchos_framework(
    name = "fmwk_with_imported_static_framework",
    bundle_id = "com.google.example.framework",
    extension_safe = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":static_fmwk_depending_lib",
    ],
)

# ---------------------------------------------------------------------------------------

watchos_static_framework(
    name = "static_fmwk",
    hdrs = [
        "//test/starlark_tests/resources:shared.h",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

watchos_static_framework(
    name = "swift_static_fmwk",
    bundle_name = "swift_static_fmwk",
    minimum_os_version = common.min_os_watchos.baseline,
    tags = common.fixture_tags,
    deps = [":swift_fmwk_lib"],
)

swift_library(
    name = "swift_fmwk_lib",
    srcs = ["//test/testdata/sources:main.swift"],
    module_name = "swift_static_fmwk",
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------

objc_library(
    name = "dynamic_fmwk_depending_lib",
    tags = common.fixture_tags,
    deps = [":imported_watchos_dynamic_fmwk"],
)

apple_dynamic_framework_import(
    name = "imported_watchos_dynamic_fmwk",
    framework_imports = [":generated_watchos_dynamic_fmwk"],
    tags = common.fixture_tags,
)

generate_import_framework(
    name = "generated_watchos_dynamic_fmwk",
    archs = ["x86_64"],
    libtype = "dynamic",
    minimum_os_version = common.min_os_watchos.baseline,
    sdk = "watchsimulator",
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------

objc_library(
    name = "static_fmwk_depending_lib",
    srcs = ["@bazel_tools//tools/objc:dummy.c"],
    deps = [":imported_watchos_static_fmwk"],
)

apple_static_framework_import(
    name = "imported_watchos_static_fmwk",
    framework_imports = [":generated_watchos_static_fmwk"],
)

generate_import_framework(
    name = "generated_watchos_static_fmwk",
    archs = ["x86_64"],
    libtype = "static",
    minimum_os_version = common.min_os_watchos.baseline,
    sdk = "watchsimulator",
)

# ---------------------------------------------------------------------------------------

ios_application(
    name = "ios_watchos_with_watchos_extension",
    bundle_id = "com.google",
    bundle_name = "companion",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.arm_sim_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    watch_application = ":watchos_app_with_extension",
    deps = [
        ":swift_lib",
    ],
)

ios_application(
    name = "ios_watchos_with_watchos_extension_within_extensions",
    bundle_id = "com.google",
    bundle_name = "companion",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.arm_sim_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    watch_application = ":watchos_app_with_extension_within_extensions",
    deps = [
        ":swift_lib",
    ],
)

ios_application(
    name = "ios_watchos_with_watchos_extension_and_symbols_in_bundle",
    bundle_id = "com.google",
    bundle_name = "companion",
    families = ["iphone"],
    include_symbols_in_bundle = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.arm_sim_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    watch_application = ":watchos_app_with_extension",
    deps = [
        ":swift_lib",
    ],
)

watchos_extension(
    name = "watchos_app_extension",
    application_extension = True,
    bundle_id = "com.google.example.watchosext.nestedwatchosext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

watchos_extension(
    name = "watchos_ext_with_extension",
    bundle_id = "com.google.example.watchosext",
    bundle_name = "ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    extensions = [":watchos_app_extension"],
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

watchos_application(
    name = "watchos_app_with_extension",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    bundle_name = "app",
    extension = ":watchos_ext_with_extension",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
)

watchos_application(
    name = "watchos_app_with_extension_within_extensions",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    bundle_name = "app",
    extensions = [":watchos_ext_with_extension"],
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
)

watchos_application(
    name = "single_target_app_too_low_minos",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = "6.0",
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:localization",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

watchos_application(
    name = "single_target_app_with_watch2_ext",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    extension = ":watchos_app_extension",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.single_target_app,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:localization",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------

ios_application(
    name = "ios_with_swift_watchos_no_swift",
    bundle_id = "com.google",
    bundle_name = "companion",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    watch_application = ":watchos_app_no_swift",
    deps = [
        ":swift_lib",
    ],
)

ios_application(
    name = "ios_no_swift_watchos_with_swift",
    bundle_id = "com.google",
    bundle_name = "companion",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    watch_application = ":watchos_app_swift",
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "ios_with_swift_watchos_with_swift",
    bundle_id = "com.google",
    bundle_name = "companion",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    watch_application = ":watchos_app_swift",
    deps = [
        ":swift_lib",
    ],
)

ios_application(
    name = "ios_with_swift_watchos_with_swift_stable_abi",
    bundle_id = "com.google",
    bundle_name = "companion",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.stable_swift_abi,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    watch_application = ":watchos_app_swift_stable_abi",
    deps = [
        ":swift_lib",
    ],
)

ios_application(
    name = "ios_with_swift_single_target_watchos_with_swift_stable_abi",
    bundle_id = "com.google",
    bundle_name = "companion",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.stable_swift_abi,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    watch_application = ":watchos_single_target_app_swift_stable_abi",
    deps = [
        ":swift_lib",
    ],
)

watchos_application(
    name = "watchos_app_no_swift",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    bundle_name = "app",
    extension = ":watchos_ext_no_swift",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
)

watchos_extension(
    name = "watchos_ext_no_swift",
    bundle_id = "com.google.example.ext",
    bundle_name = "ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

watchos_application(
    name = "watchos_app_swift",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    bundle_name = "app",
    extension = ":watchos_ext_swift",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
)

watchos_extension(
    name = "watchos_ext_swift",
    bundle_id = "com.google.example.ext",
    bundle_name = "ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_ext_main_lib",
    ],
)

watchos_application(
    name = "watchos_app_swift_stable_abi",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    bundle_name = "app",
    extension = ":watchos_ext_swift_stable_abi",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = "6.0",
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
)

watchos_application(
    name = "watchos_single_target_app_swift_stable_abi",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    bundle_name = "app",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.single_target_app,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

watchos_extension(
    name = "watchos_ext_swift_stable_abi",
    bundle_id = "com.google.example.ext",
    bundle_name = "ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.stable_swift_abi,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_ext_main_lib",
    ],
)

swift_library(
    name = "swift_lib",
    srcs = ["//test/testdata/sources:main.swift"],
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------

swift_library(
    name = "basic_framework_lib",
    srcs = [
        "//test/starlark_tests/resources:BasicFramework.swift",
        "//test/starlark_tests/resources:main.swift",
    ],
    features = ["swift.no_generated_module_map"],
    generates_header = True,
    module_name = "BasicFramework",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
)

watchos_dynamic_framework(
    name = "basic_framework",
    bundle_id = "com.google.example.framework",
    bundle_name = "BasicFramework",
    infoplists = [
        "//test/starlark_tests/resources:Info-watchos.plist",
    ],
    minimum_os_version = "6.0",
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_lib",
    ],
)

swift_library(
    name = "basic_framework_with_direct_dependency_lib",
    srcs = [
        "//test/starlark_tests/resources:DirectDependencyTest.swift",
    ],
    features = ["swift.no_generated_module_map"],
    generates_header = True,
    module_name = "DirectDependencyTest",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    deps = [
        ":basic_framework_lib",
    ],
)

watchos_dynamic_framework(
    name = "basic_framework_with_direct_dependency",
    bundle_id = "com.google.example.framework",
    bundle_name = "DirectDependencyTest",
    infoplists = [
        "//test/starlark_tests/resources:Info-watchos.plist",
    ],
    minimum_os_version = "6.0",
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_direct_dependency_lib",
    ],
)

swift_library(
    name = "basic_framework_with_transitive_dependency_lib",
    srcs = [
        "//test/starlark_tests/resources:TransitiveDependencyTest.swift",
    ],
    features = ["swift.no_generated_module_map"],
    generates_header = True,
    module_name = "TransitiveDependencyTest",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    deps = [
        ":basic_framework_lib",
        ":basic_framework_with_direct_dependency_lib",
    ],
)

watchos_dynamic_framework(
    name = "basic_framework_with_transitive_dependency",
    bundle_id = "com.google.example.framework",
    bundle_name = "TransitiveDependencyTest",
    infoplists = [
        "//test/starlark_tests/resources:Info-watchos.plist",
    ],
    minimum_os_version = "6.0",
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_transitive_dependency_lib",
    ],
)

watchos_dynamic_framework(
    name = "dynamic_fmwk_with_multiple_dependencies",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = "6.0",
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_lib",
        ":basic_framework_with_direct_dependency_lib",
    ],
)

watchos_dynamic_framework(
    name = "dynamic_fmwk_with_objc_library",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = "6.0",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_lib_with_resources",
    ],
)

watchos_ui_test(
    name = "ui_test",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.test_runner_support,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

watchos_ui_test(
    name = "single_target_ui_test",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.test_runner_support,
    tags = common.fixture_tags,
    test_host = ":single_target_app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

watchos_unit_test(
    name = "unit_test",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.test_runner_support,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

# ---------------------------------------------------------------------------------------

watchos_extension(
    name = "ext_with_objc_library_dep_with_watchos_framework_data",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    tags = common.fixture_tags,
    deps = [":objc_lib_with_data_fmwk"],
)

watchos_extension(
    name = "ext_with_objc_lib_dep_with_inner_lib_with_data_fmwk",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    # frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    tags = common.fixture_tags,
    deps = [":objc_lib_with_inner_lib_with_data_fmwk"],
)

watchos_extension(
    name = "ext_with_load_and_runtime_framework_dep",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    tags = common.fixture_tags,
    deps = [":objc_lib_with_data_fmwk"],
)

objc_library(
    name = "objc_lib_with_inner_lib_with_data_fmwk",
    srcs = ["//test/starlark_tests/resources:main.m"],
    data = [":fmwk_with_fmwk"],
    deps = [":objc_lib_with_data_fmwk"],
)

objc_library(
    name = "objc_lib_with_data_fmwk",
    srcs = ["//test/starlark_tests/resources:main.m"],
    data = [":fmwk_with_provisioning"],
)

# -----------------------------------------------------------------------------------------
# Targets for Apple dynamic XCFramework import tests.

watchos_application(
    name = "app_with_imported_xcframework",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.arm64_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":dynamic_xcframework_depending_lib",
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

objc_library(
    name = "dynamic_xcframework_depending_lib",
    tags = common.fixture_tags,
    deps = [
        ":watchos_imported_dynamic_xcframework",
    ],
)

apple_dynamic_xcframework_import(
    name = "watchos_imported_dynamic_xcframework",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    xcframework_imports = [":generated_dynamic_watchos_xcframework"],
)

generate_dynamic_xcframework(
    name = "generated_dynamic_watchos_xcframework",
    srcs = ["//test/starlark_tests/resources/frameworks:SharedClass.m"],
    hdrs = ["//test/starlark_tests/resources/frameworks:SharedClass.h"],
    minimum_os_versions = {
        "watchos": common.min_os_watchos.arm64_support,
        "watchos_simulator": common.min_os_watchos.arm_sim_support,
    },
    platforms = {
        "watchos": [
            "arm64",
            "arm64_32",
            "arm64e",
            "armv7k",
        ],
        "watchos_simulator": [
            "x86_64",
            "arm64",
        ],
    },
    tags = common.fixture_tags,
)

# -----------------------------------------------------------------------------------------
# Targets for Apple static XCFramework import tests.

watchos_application(
    name = "app_with_imported_static_xcframework",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.arm64_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":static_xcframework_depending_lib",
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

objc_library(
    name = "static_xcframework_depending_lib",
    tags = common.fixture_tags,
    deps = [":watchos_imported_static_xcframework"],
)

apple_static_xcframework_import(
    name = "watchos_imported_static_xcframework",
    features = ["-swift.layering_check"],
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    xcframework_imports = [":generated_static_watchos_xcframework"],
)

generate_static_xcframework(
    name = "generated_static_watchos_xcframework",
    srcs = ["//test/starlark_tests/resources/frameworks:SharedClass.m"],
    hdrs = ["//test/starlark_tests/resources/frameworks:SharedClass.h"],
    minimum_os_versions = {
        "watchos": common.min_os_watchos.arm64_support,
        "watchos_simulator": common.min_os_watchos.arm_sim_support,
    },
    platforms = {
        "watchos": [
            "armv7k",
            "arm64_32",
        ],
        "watchos_simulator": [
            "x86_64",
            "arm64",
        ],
    },
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------
# Targets for AppIntents

watchos_application(
    name = "app_with_app_intents",
    app_intents = [
        "//test/starlark_tests/resources:app_intent",
    ],
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.app_intents_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for base_bundle_id and bundle_id flows with and without shared capabilities.

watchos_application(
    name = "app_with_ext_with_capability_set_derived_bundle_id",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    bundle_name = "app",
    extension = ":ext_with_capability_set_derived_bundle_id",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
)

watchos_extension(
    name = "ext_with_capability_set_derived_bundle_id",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosExtensionCapabilityDerivedInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:localization",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_ext_main_lib",
    ],
)

watchos_application(
    name = "single_target_app_with_capability_set_derived_bundle_id",
    app_icons = ["//test/starlark_tests/resources:WatchAppIcon.xcassets"],
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.single_target_app,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:watchkit_single_target_app_main_lib",
    ],
)

watchos_ui_test(
    name = "ui_test_with_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_id_suffix = "ui-test-with-base-bundle-id-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.test_runner_support,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

watchos_unit_test(
    name = "unit_test_with_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_id_suffix = "unit-test-with-base-bundle-id-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_watchos.test_runner_support,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

watchos_ui_test(
    name = "single_target_ui_test_with_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_id_suffix = "single-target-ui-test-with-base-bundle-id-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:WatchosAppInfo.plist",
    ],
    minimum_os_version = common.min_os_watchos.test_runner_support,
    tags = common.fixture_tags,
    test_host = ":single_target_app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)
