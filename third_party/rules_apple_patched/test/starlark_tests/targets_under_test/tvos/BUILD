load("@bazel_skylib//rules:write_file.bzl", "write_file")
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
    "//apple:tvos.bzl",
    "tvos_application",
    "tvos_dynamic_framework",
    "tvos_extension",
    "tvos_framework",
    "tvos_static_framework",
    "tvos_ui_test",
    "tvos_unit_test",
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

tvos_application(
    name = "app",
    app_icons = ["//test/starlark_tests/resources:TVBrandAssets.xcassets"],
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    launch_storyboard = "//test/starlark_tests/resources:launch_screen_tvos.storyboard",
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:localization",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_uikit_appdelegate",
    ],
)

tvos_application(
    name = "app_multiple_infoplists",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Another.plist",
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------

tvos_application(
    name = "app_with_ext",
    bundle_id = "com.google.example",
    extensions = [":ext"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_extension(
    name = "ext",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_application(
    name = "app_with_swift_ext",
    bundle_id = "com.google.example",
    extensions = [":swift_ext"],
    infoplists = ["//test/starlark_tests/resources:Info.plist"],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = ["//test/starlark_tests/resources:objc_main_lib"],
)

tvos_application(
    name = "app_with_extensionkit_ext",
    bundle_id = "com.google.example",
    extensions = [":extensionkit_ext"],
    infoplists = ["//test/starlark_tests/resources:Info.plist"],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = ["//test/starlark_tests/resources:objc_main_lib"],
)

tvos_extension(
    name = "swift_ext",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = ["//test/starlark_tests/resources:Info.plist"],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

tvos_extension(
    name = "extensionkit_ext",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    extensionkit_extension = True,
    infoplists = ["//test/starlark_tests/resources:ExtensionKitInfo.plist"],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_extension(
    name = "ext_with_legacy_entry_point",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = ["//test/starlark_tests/resources:Info.plist"],
    legacy_entry_point = True,
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_application(
    name = "app_with_ext_with_fmwk_not_extension_safe",
    bundle_id = "com.google.example",
    extensions = [":ext_with_fmwk_not_extension_safe"],
    frameworks = [":fmwk_not_extension_safe"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_extension(
    name = "ext_with_fmwk_not_extension_safe",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    frameworks = [":fmwk_not_extension_safe"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_application(
    name = "app_with_framework_depending_ext",
    bundle_id = "com.google.example",
    extensions = [":ext_with_framework"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_application(
    name = "app_with_framework_and_framework_depending_ext",
    bundle_id = "com.google.example",
    extensions = [":ext_with_framework"],
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_extension(
    name = "ext_with_framework",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_application(
    name = "app_with_ext_min_os_nplus1",
    bundle_id = "com.google.example",
    extensions = [":ext_min_os_nplus1"],
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_tvos.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_extension(
    name = "ext_min_os_nplus1",
    bundle_id = "com.google.example.ext",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.nplus1,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_tvos.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_application(
    name = "app_with_ext_and_fmwk_with_common_structured_resources",
    bundle_id = "com.google.example",
    extensions = [":ext_with_framework_with_structured_resources"],
    frameworks = [":fmwk_with_structured_resources"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib_with_structured_images",
    ],
)

tvos_extension(
    name = "ext_with_framework_with_structured_resources",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    frameworks = [":fmwk_with_structured_resources"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib_with_structured_images",
    ],
)

# ---------------------------------------------------------------------------------------

tvos_application(
    name = "app_with_fmwk",
    bundle_id = "com.google.example",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_application(
    name = "app_structured_resources_and_fmwk_with_structured_resources",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_structured_resources"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib_with_structured_images",
    ],
)

tvos_application(
    name = "app_with_fmwk_with_transitive_structured_resources",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_transitive_structured_resources"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib_with_transitive_structured_images",
    ],
)

tvos_application(
    name = "app_with_fmwk_with_resource_bundles",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_resource_bundles"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_application(
    name = "app_with_resource_bundles_and_fmwk_with_resource_bundles",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_resource_bundles"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib_with_transitive_resource_bundles",
    ],
)

tvos_application(
    name = "app_with_resource_bundles_and_fmwk_with_resource_bundles_nplus1",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_resource_bundles_nplus1"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib_with_transitive_resource_bundles",
    ],
)

tvos_application(
    name = "app_with_two_fmwk_provisioned",
    bundle_id = "com.google.example",
    frameworks = [
        ":fmwk_with_provisioning",
        ":second_fmwk_with_provisioning",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_application(
    name = "app_with_fmwk_with_fmwk_provisioned",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_fmwk_with_provisioning"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_application(
    name = "app_with_fmwk_with_fmwk",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
        "//test/starlark_tests/resources:Another.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_application(
    name = "app_with_imported_fmwk",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":dynamic_fmwk_depending_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_application(
    name = "app_with_runtime_framework_using_import_static_lib_dep",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_imported_static_framework"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_application(
    name = "app_with_runtime_framework_using_import_framework_dep",
    bundle_id = "com.google.example",
    frameworks = [":fmwk_with_imported_dynamic_framework"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_framework(
    name = "fmwk",
    hdrs = [
        "//test/starlark_tests/resources:shared.h",
    ],
    bundle_id = "com.google.example.framework",
    extension_safe = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
        "//test/starlark_tests/resources:Another.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib_with_transitive_resources",
    ],
)

tvos_framework(
    name = "fmwk_with_no_version",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info-noversion.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

tvos_framework(
    name = "fmwk_with_structured_resources",
    bundle_id = "com.google.example.framework",
    extension_safe = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-application_extension"],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_common_lib_with_structured_images",
    ],
)

tvos_framework(
    name = "fmwk_with_transitive_structured_resources",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib_with_transitive_structured_resources",
    ],
)

tvos_framework(
    name = "fmwk_with_resource_bundles",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    resources = [],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib_with_resource_bundles",
    ],
)

tvos_framework(
    name = "fmwk_with_resource_bundles_nplus1",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.nplus1,
    resources = [],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib_with_resource_bundles",
    ],
)

tvos_framework(
    name = "fmwk_dead_stripped",
    bundle_id = "com.google.example.framework",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionShared.exp",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-dead_strip"],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

tvos_framework(
    name = "fmwk_with_provisioning",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

tvos_framework(
    name = "second_fmwk_with_provisioning",
    bundle_id = "com.google.example.frameworktoo",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

tvos_framework(
    name = "fmwk_with_fmwk",
    bundle_id = "com.google.example.frameworkception",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
        "//test/starlark_tests/resources:Another.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:another_objc_shared_lib_with_transitive_resources",
    ],
)

tvos_framework(
    name = "fmwk_with_fmwk_with_provisioning",
    bundle_id = "com.google.example.frameworkception",
    frameworks = [":fmwk_with_provisioning"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

tvos_framework(
    name = "fmwk_with_imported_static_framework",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":static_fmwk_depending_lib",
    ],
)

tvos_framework(
    name = "fmwk_not_extension_safe",
    bundle_id = "com.google.example.framework",
    extension_safe = False,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

tvos_framework(
    name = "fmwk_with_imported_dynamic_framework",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":dynamic_fmwk_depending_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for Apple dynamic framework import tests.

objc_library(
    name = "dynamic_fmwk_depending_lib",
    tags = common.fixture_tags,
    deps = [":imported_tvos_dynamic_fmwk"],
)

apple_dynamic_framework_import(
    name = "imported_tvos_dynamic_fmwk",
    framework_imports = [":generated_tvos_dynamic_fmwk"],
    tags = common.fixture_tags,
)

generate_import_framework(
    name = "generated_tvos_dynamic_fmwk",
    archs = ["x86_64"],
    include_resource_bundle = True,
    libtype = "dynamic",
    minimum_os_version = common.min_os_tvos.baseline,
    sdk = "appletvsimulator",
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------
# Targets for Apple static framework import tests.

objc_library(
    name = "static_fmwk_depending_lib",
    tags = common.fixture_tags,
    deps = [":imported_tvos_static_fmwk"],
)

apple_static_framework_import(
    name = "imported_tvos_static_fmwk",
    framework_imports = [":generated_tvos_static_fmwk"],
    tags = common.fixture_tags,
)

generate_import_framework(
    name = "generated_tvos_static_fmwk",
    archs = ["x86_64"],
    libtype = "static",
    minimum_os_version = common.min_os_tvos.baseline,
    sdk = "appletvsimulator",
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------
# Targets for Apple dynamic XCFramework import tests.

tvos_application(
    name = "app_with_imported_dynamic_xcframework",
    bundle_id = "com.google.example",
    infoplists = ["//test/starlark_tests/resources:Info.plist"],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":dynamic_xcframework_depending_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "dynamic_xcframework_depending_lib",
    tags = common.fixture_tags,
    deps = [
        ":tvos_imported_dynamic_xcframework",
    ],
)

apple_dynamic_xcframework_import(
    name = "tvos_imported_dynamic_xcframework",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    xcframework_imports = [":generated_dynamic_tvos_xcframework"],
)

generate_dynamic_xcframework(
    name = "generated_dynamic_tvos_xcframework",
    srcs = ["//test/starlark_tests/resources/frameworks:SharedClass.m"],
    hdrs = ["//test/starlark_tests/resources/frameworks:SharedClass.h"],
    minimum_os_versions = {
        "tvos": common.min_os_tvos.baseline,
        "tvos_simulator": common.min_os_tvos.arm_sim_support,
    },
    platforms = {
        "tvos": [
            "arm64",
        ],
        "tvos_simulator": [
            "arm64",
            "x86_64",
        ],
    },
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------
# Targets for Apple static XCFramework import tests.

tvos_application(
    name = "app_with_imported_static_xcframework",
    bundle_id = "com.google.example",
    infoplists = ["//test/starlark_tests/resources:Info.plist"],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":static_xcframework_depending_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "static_xcframework_depending_lib",
    tags = common.fixture_tags,
    deps = [":tvos_imported_static_xcframework"],
)

apple_static_xcframework_import(
    name = "tvos_imported_static_xcframework",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    xcframework_imports = [":generated_static_tvos_xcframework"],
)

generate_static_xcframework(
    name = "generated_static_tvos_xcframework",
    srcs = ["//test/starlark_tests/resources/frameworks:SharedClass.m"],
    hdrs = ["//test/starlark_tests/resources/frameworks:SharedClass.h"],
    minimum_os_versions = {
        "tvos": common.min_os_tvos.baseline,
        "tvos_simulator": common.min_os_tvos.arm_sim_support,
    },
    platforms = {
        "tvos": [
            "arm64",
        ],
        "tvos_simulator": [
            "arm64",
            "x86_64",
        ],
    },
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------

tvos_static_framework(
    name = "static_fmwk",
    hdrs = [
        "//test/starlark_tests/resources:shared.h",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

tvos_static_framework(
    name = "swift_static_fmwk",
    bundle_name = "swift_static_fmwk",
    minimum_os_version = common.min_os_tvos.baseline,
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

tvos_ui_test(
    name = "ui_test",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

tvos_ui_test(
    name = "ui_test_custom_bundle_id",
    bundle_id = "my.test.bundle.id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

tvos_ui_test(
    name = "ui_test_invalid_bundle_id",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

tvos_ui_test(
    name = "ui_test_dummy_runner",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    runner = "//test/starlark_tests/targets_under_test/apple:dummy_test_runner",
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

# ---------------------------------------------------------------------------------------

tvos_unit_test(
    name = "unit_test",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

tvos_unit_test(
    name = "unit_test_no_host",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.arm_sim_support,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

tvos_unit_test(
    name = "unit_test_custom_bundle_id",
    bundle_id = "my.test.bundle.id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

tvos_unit_test(
    name = "unit_test_invalid_bundle_id",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

tvos_unit_test(
    name = "unit_test_with_fmwk",
    frameworks = [
        ":fmwk",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

tvos_unit_test(
    name = "unit_test_with_fmwk_from_objc_library_data",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_tvos_framework",
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

tvos_unit_test(
    name = "unit_test_with_imported_fmwk",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":dynamic_fmwk_depending_lib",
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

tvos_unit_test(
    name = "unit_test_with_host_importing_same_fmwk",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    test_host = ":app_with_imported_fmwk",
    deps = [
        ":dynamic_fmwk_depending_lib",
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

tvos_unit_test(
    name = "unit_test_dummy_runner",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    runner = "//test/starlark_tests/targets_under_test/apple:dummy_test_runner",
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

tvos_unit_test(
    name = "unit_test_with_swift_deps",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.arm_sim_support,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_swift_depending_test_lib",
    ],
)

# ---------------------------------------------------------------------------------------

tvos_application(
    name = "app_with_direct_swift_dep",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [":swift_lib"],
)

tvos_application(
    name = "app_with_indirect_swift_dep",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [":objc_to_swift_lib"],
)

tvos_application(
    name = "app_with_direct_swift_dep_stable_abi",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.stable_swift_abi,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [":swift_lib"],
)

swift_library(
    name = "swift_lib",
    srcs = ["//test/testdata/sources:main.swift"],
    tags = common.fixture_tags,
)

objc_library(
    name = "objc_to_swift_lib",
    tags = common.fixture_tags,
    deps = [":swift_lib"],
)

# ---------------------------------------------------------------------------------------
# Targets for the app/test resource deduping test.

objc_library(
    name = "shared_lib",
    data = [
        "//test/starlark_tests/resources:basic_bundle",
    ],
    tags = common.fixture_tags,
)

objc_library(
    name = "app_lib",
    data = [
        "//test/starlark_tests/resources:empty.strings",
    ],
    tags = common.fixture_tags,
    deps = [":shared_lib"],
)

objc_library(
    name = "test_lib",
    data = [
        "//test/starlark_tests/resources:nonlocalized.strings",
    ],
    tags = common.fixture_tags,
    deps = [":app_lib"],
)

objc_library(
    name = "main_lib",
    tags = common.fixture_tags,
    deps = [
        ":app_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_framework(
    name = "dedupe_test_fmwk",
    bundle_id = "com.fmwk",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":shared_lib",
    ],
)

tvos_application(
    name = "dedupe_test_app",
    bundle_id = "com.app",
    frameworks = [
        ":dedupe_test_fmwk",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_tvos.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":main_lib",
    ],
)

tvos_unit_test(
    name = "dedupe_test_test",
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    test_host = ":dedupe_test_app",
    deps = [
        ":test_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for Apple Frameworks propagated from objc_library/swift_library data attribute tests.

swift_library(
    name = "basic_framework_lib",
    srcs = [
        "//test/starlark_tests/resources:BasicFramework.swift",
    ],
    features = ["swift.no_generated_module_map"],
    generates_header = True,
    module_name = "BasicFramework",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
)

tvos_dynamic_framework(
    name = "basic_framework",
    bundle_id = "com.google.example.framework",
    bundle_name = "BasicFramework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
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

tvos_dynamic_framework(
    name = "basic_framework_with_direct_dependency",
    bundle_id = "com.google.example.framework",
    bundle_name = "DirectDependencyTest",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
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

tvos_dynamic_framework(
    name = "basic_framework_with_transitive_dependency",
    bundle_id = "com.google.example.framework",
    bundle_name = "TransitiveDependencyTest",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_transitive_dependency_lib",
    ],
)

tvos_dynamic_framework(
    name = "dynamic_fmwk_with_multiple_dependencies",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_lib",
        ":basic_framework_with_direct_dependency_lib",
    ],
)

tvos_dynamic_framework(
    name = "dynamic_fmwk_with_objc_library",
    bundle_id = "com.google.example.framework",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_lib_with_resources",
    ],
)

# ---------------------------------------------------------------------------------------

tvos_application(
    name = "app_with_fmwks_from_objc_swift_libraries_using_data",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_tvos_framework",
        ":swift_lib_with_tvos_framework",
    ],
)

tvos_application(
    name = "app_with_fmwks_from_transitive_objc_swift_libraries_using_data",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_inner_lib_with_tvos_framework",
        ":swift_lib_with_tvos_framework",
    ],
)

tvos_application(
    name = "app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
    bundle_id = "com.google.example",
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_tvos_framework",
        ":swift_lib_with_tvos_framework",
    ],
)

objc_library(
    name = "objc_lib_with_inner_lib_with_tvos_framework",
    srcs = [":objc_lib_with_inner_lib_with_tvos_framework_file"],
    tags = common.fixture_tags,
    deps = [":objc_lib_with_tvos_framework"],
)

write_file(
    name = "objc_lib_with_inner_lib_with_tvos_framework_file",
    out = "objc_lib_with_inner_lib_with_tvos_framework_file.m",
    content = ["""
int objc_lib_with_inner_lib_with_tvos_framework_function() {
  return 0;
}
"""],
    tags = common.fixture_tags,
)

objc_library(
    name = "objc_lib_with_tvos_framework",
    srcs = [":objc_lib_with_tvos_framework_file"],
    data = [":fmwk_with_structured_resources"],
    tags = common.fixture_tags,
)

write_file(
    name = "objc_lib_with_tvos_framework_file",
    out = "objc_lib_with_tvos_framework_file.m",
    content = ["""
int main() {
  return 0;
}
"""],
    tags = common.fixture_tags,
)

swift_library(
    name = "swift_lib_with_tvos_framework",
    srcs = ["//test/starlark_tests/resources:DummySwiftFile"],
    data = [":fmwk_with_resource_bundles"],
    tags = common.fixture_tags,
)

tvos_application(
    name = "app_with_resources_and_fmwks_with_resources_from_objc_swift_libraries_using_data",
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_tvos_framework",
        ":swift_lib_with_tvos_framework",
        "//test/starlark_tests/resources:another_basic_bundle_lib",
        "//test/starlark_tests/resources:objc_common_lib_with_structured_resources",
    ],
)

tvos_extension(
    name = "ext_with_fmwks_from_objc_swift_libraries_using_data",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_tvos_framework",
        ":swift_lib_with_tvos_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_framework(
    name = "fmwk_with_fmwks_from_objc_swift_libraries_using_data",
    hdrs = [
        "//test/starlark_tests/resources:shared.h",
    ],
    bundle_id = "com.google.example.framework",
    extension_safe = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_tvos_framework",
        ":swift_lib_with_tvos_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------

tvos_extension(
    name = "ext_for_bundle_loader",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_common_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_unit_test(
    name = "unit_test_with_bundle_loader",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    test_host = ":ext_for_bundle_loader",
    deps = [
        "//test/starlark_tests/resources:objc_common_test_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for AppIntents

tvos_application(
    name = "app_with_app_intents",
    app_intents = [
        "//test/starlark_tests/resources:app_intent",
    ],
    bundle_id = "com.google.example",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.app_intents_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_uikit_appdelegate",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for base_bundle_id and bundle_id flows with and without shared capabilities.

tvos_application(
    name = "app_with_capability_set_derived_bundle_id",
    bundle_name = "app-with-capability-set-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_extension(
    name = "ext_with_capability_set_derived_bundle_id",
    bundle_name = "ext-with-capability-set-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

tvos_framework(
    name = "fmwk_with_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_name = "fmwk-with-base-bundle-id-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

tvos_ui_test(
    name = "ui_test_with_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_id_suffix = "ui-test-with-base-bundle-id-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

tvos_unit_test(
    name = "unit_test_with_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_id_suffix = "unit-test-with-base-bundle-id-derived-bundle-id",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_tvos.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)
