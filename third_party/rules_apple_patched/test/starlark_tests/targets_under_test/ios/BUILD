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
    "//apple:docc.bzl",
    "docc_archive",
)
load(
    "//apple:ios.bzl",
    "ios_app_clip",
    "ios_application",
    "ios_dynamic_framework",
    "ios_extension",
    "ios_framework",
    "ios_imessage_application",
    "ios_imessage_extension",
    "ios_static_framework",
    "ios_sticker_pack_extension",
    "ios_ui_test",
    "ios_unit_test",
)
load(
    "//apple:resources.bzl",
    "apple_resource_group",
)
load(
    "//apple:xcarchive.bzl",
    "xcarchive",
)
load(
    "//apple:xctrunner.bzl",
    "xctrunner",
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
    "//test/starlark_tests/rules:generate_framework_dsym.bzl",
    "generate_import_framework_dsym",
)
load(
    "//test/starlark_tests/rules:generate_xcframework.bzl",
    "generate_dynamic_xcframework",
    "generate_static_framework_xcframework",
    "generate_static_xcframework",
)
load("//test/testdata/rules:substitution.bzl", "substitution")

licenses(["notice"])

package(
    default_testonly = True,
    default_visibility = ["//test/starlark_tests:__subpackages__"],
)

# This is a list of iOS targets to be used for Starlark unit tests. These may not build correctly
# since most of them do not have any source dependencies, so they are all tagged as manual and
# notap to avoid them being built/tested when testing the //test/...
# package.

ios_application(
    name = "app minimal has several spaces",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_minimal_bundle_name_with_spaces",
    bundle_id = "com.google.example",
    bundle_name = "app minimal bundle name has several spaces",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_minimal",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_minimal_with_space_and_post_processor",
    bundle_id = "com.google.example",
    bundle_name = "app minimal",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

xcarchive(
    name = "app_minimal.xcarchive",
    bundle = ":app_minimal",
    tags = common.fixture_tags,
)

ios_application(
    name = "app_minimal_with_deployment_version",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_deployment_os_version = "14.0",
    minimum_os_version = "11.0",
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_icon_bundle",
    app_icons = ["//test/starlark_tests/resources:icon_bundle_assets"],
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.icon_bundle_required,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_icon_bundle_only_for_low_minimum_os_version",
    app_icons = ["//test/starlark_tests/resources:icon_bundle_assets"],
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_icon_bundle_and_xcassets_app_icons",
    app_icons = [
        "//test/starlark_tests/resources:dark_and_tinted_app_icons_ios",
        "//test/starlark_tests/resources:icon_bundle_assets",
    ],
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_xcassets_for_ios_26",
    app_icons = [
        "//test/starlark_tests/resources:dark_and_tinted_app_icons_ios",
    ],
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.icon_bundle_required,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

# TODO: b/425967223 - Add .icon test cases for multiple app icons with a primary app icon selected.

ios_application(
    name = "app",
    app_icons = ["//test/testdata/resources:app_icons_ios"],
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    launch_storyboard = "//test/starlark_tests/resources:launch_screen_ios.storyboard",
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:localization",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    settings_bundle = "//test/starlark_tests/resources:settings_bundle_ios",
    tags = common.fixture_tags,
    visibility = [
        "//test/starlark_tests:__subpackages__",
        "//tools/dossier_codesigningtool:__pkg__",
    ],
    deps = [
        "//test/starlark_tests/resources:basic_bundle_lib",
        "//test/starlark_tests/resources:bundle_library_ios_lib",
        "//test/starlark_tests/resources:empty_strings_file_lib",
        "//test/starlark_tests/resources:ios_localized_assets_lib",
        "//test/starlark_tests/resources:ios_non_localized_assets_lib",
        "//test/starlark_tests/resources:nested_bundle_lib",
        "//test/starlark_tests/resources:objc_linkopt_lib",
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:sticker_pack_ios_lib",
    ],
)

ios_application(
    name = "app_precompiled_resource_bundle",
    app_icons = ["//test/testdata/resources:app_icons_ios"],
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    launch_storyboard = "//test/starlark_tests/resources:launch_screen_ios.storyboard",
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:localization",
        "//test/starlark_tests/resources:precompiled_resource_bundle",
    ],
    settings_bundle = "//test/starlark_tests/resources:settings_bundle_ios",
    tags = common.fixture_tags,
    visibility = [
        "//test/starlark_tests:__subpackages__",
        "//tools/dossier_codesigningtool:__pkg__",
    ],
    deps = [
        "//test/starlark_tests/resources:basic_bundle_lib",
        "//test/starlark_tests/resources:empty_strings_file_lib",
        "//test/starlark_tests/resources:ios_localized_assets_lib",
        "//test/starlark_tests/resources:ios_non_localized_assets_lib",
        "//test/starlark_tests/resources:nested_bundle_lib",
        "//test/starlark_tests/resources:objc_linkopt_lib",
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:precompiled_bundle_library_ios_lib",
        "//test/starlark_tests/resources:sticker_pack_ios_lib",
    ],
)

ios_application(
    name = "app_multiple_infoplists",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Another.plist",
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_launch_storyboard_as_xib",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    launch_storyboard = "//test/starlark_tests/resources:launch_screen_ios.xib",
    minimum_os_version = common.min_os_ios.oldest_supported,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_custom_executable_name",
    bundle_id = "com.google.example",
    bundle_name = "custom_bundle_name",
    executable_name = "app.exe",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_structured_resources_in_resources_folder",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:structured_resources_in_resources_lib",
    ],
)

ios_application(
    name = "app_with_processed_resources_in_structured_resources",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:processed_resources_in_structured_resources_lib",
    ],
)

ios_application(
    name = "app_with_precompiled_resource_bundle_processed_resources_in_structured_resources",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:precompiled_resource_bundle_processed_resources_in_structured_resources_lib",
    ],
)

ios_application(
    name = "app_with_xcstrings",
    app_icons = ["//test/testdata/resources:app_icons_ios"],
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = ["//test/starlark_tests/resources:Info.plist"],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = ["//test/starlark_tests/resources:bundle_library_xcstrings"],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:empty_xcstrings_lib",
        "//test/starlark_tests/resources:ios_localized_xcstrings_lib",
        "//test/starlark_tests/resources:ios_non_localized_xcstrings_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_launch_images",
    app_icons = ["//test/testdata/resources:app_icons_ios"],
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    launch_images = ["//test/starlark_tests/resources:launch_images_ios"],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_library_scoped_localized_assets",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:ios_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_duplicated_library_scoped_localized_assets",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:apple_localized_assets_lib",
        "//test/starlark_tests/resources:ios_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_top_level_and_library_scoped_localized_assets",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    strings = [
        "//test/starlark_tests/resources:localized_strings",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:ios_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_launch_images_from_library",
    app_icons = ["//test/testdata/resources:app_icons_ios"],
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:launch_images_ios_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_duplicated_library_scoped_launch_images",
    app_icons = ["//test/testdata/resources:app_icons_ios"],
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:launch_images_ios_lib",
        "//test/starlark_tests/resources:launch_images_ios_lib_duplicated_target",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_top_level_and_library_scoped_launch_images",
    app_icons = ["//test/testdata/resources:app_icons_ios"],
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    launch_images = ["//test/starlark_tests/resources:launch_images_ios"],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:launch_images_ios_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_alternate_icons",
    alternate_icons = ["//test/testdata/resources:alternate_icons_ios"],
    app_icons = ["//test/testdata/resources:app_icons_ios"],
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    launch_images = ["//test/starlark_tests/resources:launch_images_ios"],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:launch_images_ios_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "iphone_only_app_with_alternate_icons",
    alternate_icons = ["//test/testdata/resources:alternate_icons_ios"],
    app_icons = ["//test/testdata/resources:app_icons_ios"],
    bundle_id = "com.google.example",
    families = [
        "iphone",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    launch_images = ["//test/starlark_tests/resources:launch_images_ios"],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:launch_images_ios_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_alternate_app_icons",
    app_icons = ["//test/testdata/resources:app_icons_with_alts_ios"],
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    primary_app_icon = "app_icon",
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_alternate_app_icons_with_isolated_icon_bundles",
    app_icons = [
        "//test/starlark_tests/resources:app_icons_with_alts_ios",
        "//test/starlark_tests/resources:icon_bundle_assets",
        "//test/starlark_tests/resources:icon_bundle_assets_alternative",
        "//test/starlark_tests/resources:icon_bundle_assets_mismatched_with_xcassets_alternative",
    ],
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    primary_app_icon = "app_icon",
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_alternate_app_icons_with_full_icon_bundle_coverage",
    app_icons = [
        "//test/starlark_tests/resources:app_icons_with_alts_ios",
        "//test/starlark_tests/resources:icon_bundle_assets",
        "//test/starlark_tests/resources:icon_bundle_assets_alternative",
    ],
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    primary_app_icon = "app_icon",
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_alternate_app_icons_without_primary",
    app_icons = ["//test/testdata/resources:app_icons_with_alts_ios"],
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_dark_and_tinted_app_icons",
    app_icons = ["//test/starlark_tests/resources:dark_and_tinted_app_icons_ios"],
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------

ios_application(
    name = "app_with_swift_library_scoped_resources",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_lib_with_resources",
    ],
)

ios_application(
    name = "app_with_transitive_swift_library_scoped_resources",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_lib_with_transitive_resources",
    ],
)

ios_application(
    name = "app_with_transitive_swift_libraries_with_storyboards",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_lib_with_storyboard_a",
    ],
)

ios_application(
    name = "app_with_transitive_swift_libraries_with_asset_catalogs",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_lib_with_asset_catalog_a",
    ],
)

ios_application(
    name = "app_with_multiple_storyboards_in_bundle_root_from_multiple_swift_libraries",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_lib_with_storyboard_copy",
    ],
)

# ---------------------------------------------------------------------------------------

ios_application(
    name = "app_with_same_resource_names_as_framework",
    bundle_id = "com.google.example",
    families = ["iphone"],
    frameworks = [
        ":fmwk_with_duplicate_resource_names",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":app_structured_resources_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_framework(
    name = "fmwk_with_duplicate_resource_names",
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":fmwk_structured_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

objc_library(
    name = "fmwk_structured_resources_lib",
    data = ["//test/starlark_tests/resources:structured_resources"],
    tags = common.fixture_tags,
)

objc_library(
    name = "app_structured_resources_lib",
    data = ["//test/starlark_tests/resources:structured_resources_duplicate"],
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------

ios_application(
    name = "app_with_inner_and_outer_fmwk",
    bundle_id = "com.google.example",
    families = ["iphone"],
    frameworks = [
        ":fmwk_with_imported_fmwk",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_framework(
    name = "fmwk_with_imported_fmwk",
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":iOSImportedDynamicFramework",
        "//test/starlark_tests/resources:objc_shared_lib_with_resources",
    ],
)

ios_application(
    name = "app_with_imported_dynamic_fmwk_with_bitcode",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":iOSImportedDynamicFrameworkWithBitcode",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_imported_dynamic_fmwk_with_dsym",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    include_symbols_in_bundle = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":iOSImportedDynamicFrameworkWithDebugInfo",
        ":iOSImportedDynamicFrameworkWithDsym",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_swift_dep",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

docc_archive(
    name = "app_with_swift_dep.doccarchive",
    dep = ":app_with_swift_dep",
    fallback_bundle_identifier = "com.google.example",
    fallback_bundle_version = "1.0.0",
    fallback_display_name = "app_with_swift_dep",
    tags = common.fixture_tags,
)

ios_static_framework(
    name = "swift_static_framework_with_swiftinterface",
    bundle_name = "swift_framework_lib",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_framework_lib_with_swiftinterface",
    ],
)

ios_application(
    name = "app_with_inner_and_outer_static_fmwk",
    bundle_id = "com.google.example",
    families = ["iphone"],
    frameworks = [
        ":fmwk_with_imported_static_fmwk",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_framework(
    name = "fmwk_with_imported_static_fmwk",
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":iOSImportedStaticFramework",
        "//test/starlark_tests/resources:objc_shared_lib_with_resources",
    ],
)

ios_application(
    name = "app_with_imported_static_fmwk_and_ext",
    bundle_id = "com.google.example",
    extensions = [":ext"],
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":iOSImportedStaticFramework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

apple_static_framework_import(
    name = "iOSImportedStaticFramework",
    features = ["-swift.layering_check"],
    framework_imports = [":iOSStaticFramework"],
    tags = common.fixture_tags,
)

generate_import_framework(
    name = "iOSStaticFramework",
    archs = ["x86_64"],
    include_resource_bundle = True,
    libtype = "static",
    minimum_os_version = "11.0",
    sdk = "iphonesimulator",
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------

ios_application(
    name = "app_min_os_baseline",
    bundle_id = "com.google.example",
    extensions = [":ext_min_os_nplus1"],
    families = ["iphone"],
    frameworks = [":fmwk_min_os_baseline"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":fmwk_lib",
        ":main_lib_with_fmwk_lib_dep",
    ],
)

ios_extension(
    name = "ext_min_os_nplus1",
    bundle_id = "com.google.example.ext",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":fmwk_min_os_baseline"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.nplus1,
    tags = common.fixture_tags,
    deps = [
        ":fmwk_lib",
        ":main_lib_with_fmwk_lib_dep",
    ],
)

ios_framework(
    name = "fmwk_min_os_baseline",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    extension_safe = True,
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":fmwk_lib",
    ],
)

objc_library(
    name = "fmwk_lib",
    srcs = [
        "//test/starlark_tests/resources:shared.h",
        "//test/starlark_tests/resources:shared.m",
    ],
    tags = common.fixture_tags,
)

objc_library(
    name = "main_lib_with_fmwk_lib_dep",
    srcs = [":main_lib_with_fmwk_lib_dep_file"],
    tags = common.fixture_tags,
    deps = [":fmwk_lib"],
)

write_file(
    name = "main_lib_with_fmwk_lib_dep_file",
    out = "main_lib_with_fmwk_lib_dep_file.m",
    content = ["""
int main() {
  return 0;
}
"""],
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------

ios_application(
    name = "app_with_app_clip",
    app_clips = [":app_clip"],
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.appclip_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_app_clip(
    name = "app_clip",
    bundle_id = "com.google.example.clip",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.appclip_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_app_clip_with_swift_support",
    app_clips = [":app_clip_with_swift_support"],
    bundle_id = "com.google.example",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.appclip_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_concurrency_main_lib",
    ],
)

ios_app_clip(
    name = "app_clip_with_swift_support",
    bundle_id = "com.google.example.clip",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.appclip_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_concurrency_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------

ios_application(
    name = "app_with_ext_with_fmwk_provisioned",
    bundle_id = "com.google.example",
    extensions = [":ext_with_fmwk_provisioned"],
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------

ios_application(
    name = "app_with_framework_and_resources",
    bundle_id = "com.google.example",
    families = ["iphone"],
    frameworks = [
        ":fmwk_with_resources",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:objc_shared_lib_with_resources",
    ],
)

ios_framework(
    name = "fmwk_with_resources",
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-application_extension"],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib_with_resources",
    ],
)

ios_application(
    name = "app_with_framework_and_shared_resources",
    bundle_id = "com.google.example",
    families = ["iphone"],
    frameworks = [
        ":fmwk_with_resources",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":lib_with_structured_resources",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_multiple_refs_to_same_structured_resources",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":app_structured_resources_lib",
        ":fmwk_structured_resources_lib",
        ":lib_with_structured_resources",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_multiple_resource_bundles_with_shared_resources",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:bundle_library_apple_lib",
        "//test/starlark_tests/resources:bundle_library_ios_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_multiple_precompiled_resource_bundles_with_shared_resources",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:precompiled_bundle_library_apple_lib",
        "//test/starlark_tests/resources:precompiled_bundle_library_ios_lib",
    ],
)

ios_application(
    name = "app_with_resource_bundle_with_structured_resource_group",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = ["//test/starlark_tests/resources:resource_bundle_with_structured_resource_group"],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_precompiled_resource_bundle_with_structured_resource_group",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = ["//test/starlark_tests/resources:precompiled_resource_bundle_with_structured_resource_group"],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_resource_bundle_with_bundle_id",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = ["//test/starlark_tests/resources:resource_bundle_with_bundle_id"],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_precompiled_resource_bundle_with_bundle_id",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = ["//test/starlark_tests/resources:precompiled_resource_bundle_with_bundle_id"],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_precompiled_resource_bundle_with_bundle_id_no_infoplist",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = ["//test/starlark_tests/resources:precompiled_resource_bundle_with_bundle_id_no_infoplist"],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_transitive_precompiled_resource_bundle_with_bundle_id",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib_with_transitive_precompiled_resource_bundles",
    ],
)

ios_application(
    name = "app_with_resource_group_with_resource_bundle",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = ["//test/starlark_tests/resources:resources_with_resource_bundle"],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "lib_with_structured_resources",
    data = [
        "//test/starlark_tests/resources:structured_resources",
    ],
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------

ios_application(
    name = "app_with_nplus1_min_os_and_baseline_fmwk",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":fmwk_min_os_baseline_with_bundle"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.nplus1,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:basic_bundle_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_baseline_min_os_and_nplus1_fmwk",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":fmwk_min_os_nplus1"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:basic_bundle_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_fmwk_with_bundle_resources",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":fmwk_min_os_nplus1"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_framework(
    name = "fmwk_min_os_nplus1",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.nplus1,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:basic_bundle_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "fmwk_min_os_baseline_with_bundle",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:basic_bundle_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

# ---------------------------------------------------------------------------------------

ios_application(
    name = "app_with_ext",
    bundle_id = "com.google.example",
    extensions = [":ext"],
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

xcarchive(
    name = "app_with_ext.xcarchive",
    bundle = ":app_with_ext",
    tags = common.fixture_tags,
)

ios_application(
    name = "app_with_ext_space_in_path",
    bundle_id = "com.google.example",
    extensions = [":ext_with_space"],
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

xcarchive(
    name = "app_with_ext_space_in_path.xcarchive",
    bundle = ":app_with_ext_space_in_path",
    tags = common.fixture_tags,
)

ios_application(
    name = "app_with_swift_ext",
    bundle_id = "com.google.example",
    extensions = [":swift_ext"],
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_extensionkit_ext",
    bundle_id = "com.google.example",
    extensions = [":extensionkit_ext"],
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_ext_and_fmwk_provisioned",
    bundle_id = "com.google.example",
    extensions = [":ext_with_fmwk_provisioned"],
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [
        ":fmwk_with_provisioning",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_ext_and_fmwk_and_symbols_in_bundle",
    bundle_id = "com.google.example",
    extensions = [":ext_with_fmwk_provisioned"],
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [
        ":fmwk_with_provisioning",
    ],
    include_symbols_in_bundle = True,
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":iOSImportedDynamicFrameworkWithDebugInfo",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_extension(
    name = "ext",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_extension(
    name = "ext_multiple_infoplists",
    bundle_id = "com.google.example.ext",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
        "//test/starlark_tests/resources:Another.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_extension(
    name = "ext_with_fmwk_provisioned",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [
        ":fmwk_with_provisioning",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_extension(
    name = "swift_ext",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

ios_extension(
    name = "swift_ios_app_clip_ext",
    bundle_id = "com.google.example.clip.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

ios_extension(
    name = "extensionkit_ext",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    extensionkit_extension = True,
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:ExtensionKitInfo.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_extension(
    name = "ext_with_fmwk_not_extension_safe",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = ["iphone"],
    frameworks = [":fmwk_not_extension_safe"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_extension(
    name = "ext_with_app_intents",
    app_intents = [
        "//test/starlark_tests/resources:app_intent",
    ],
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.app_intents_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:app_intent",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_extension(
    name = "ext_with_space",
    bundle_id = "com.google.example.ext",
    bundle_name = "ext with space",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------

substitution(
    name = "resource_rules",
    src = "//test/starlark_tests/resources:ResourceRules.plist",
    tags = common.fixture_tags,
    var_name = "RULES",
)

ios_application(
    name = "app_with_codesignopts",
    bundle_id = "com.google.example",
    codesign_inputs = [":resource_rules"],
    codesignopts = [
        "--digest-algorithm=sha1",
        "--digest-algorithm=sha384",
        "--resource-rules=$(RULES)",
    ],
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    toolchains = [":resource_rules"],
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_fmwk",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:ios_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_bundle_only_fmwks",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":bundle_only_fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":bundle_only_imported_fmwk",
        "//test/starlark_tests/resources:ios_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/targets_under_test/apple:ios_imported_dynamic_xcframework_bundle_only",
    ],
)

ios_application(
    name = "app_with_two_fmwk_provisioned",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [
        ":fmwk_with_provisioning",
        ":second_fmwk_with_provisioning",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:ios_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_fmwk_with_fmwk_provisioned",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":fmwk_with_fmwk_with_provisioning"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:ios_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_fmwk_importing_fmwk",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":fmwk_with_provisioning_importing_fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:ios_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_fmwk_with_fmwk",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":fmwk_with_fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:ios_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_fmwk_with_multiple_objc_library_and_swift_library_deps",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":fmwk_with_multiple_objc_library_and_swift_library_deps"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_ext_with_fmwk_not_extension_safe",
    bundle_id = "com.google.example",
    extensions = [":ext_with_fmwk_not_extension_safe"],
    families = ["iphone"],
    frameworks = [":fmwk_not_extension_safe"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_framework(
    name = "fmwk_no_version",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info-noversion.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "fmwk_with_multiple_objc_library_and_swift_library_deps",
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
        "//test/starlark_tests/resources:swift_shared_lib",
    ],
)

ios_framework(
    name = "fmwk",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "bundle_only_fmwk",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    bundle_only = True,
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "fmwk_dead_stripped",
    bundle_id = "com.google.example.framework",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionShared.exp",
    ],
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-dead_strip"],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "fmwk_dead_stripped_two_exported_symbol_lists",
    bundle_id = "com.google.example.framework",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionShared.exp",
        "//test/starlark_tests/resources:ExportDontCallMeShared.exp",
    ],
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-dead_strip"],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "fmwk_stripped",
    bundle_id = "com.google.example.framework",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionShared.exp",
    ],
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-x"],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "fmwk_stripped_two_exported_symbol_lists",
    bundle_id = "com.google.example.framework",
    exported_symbols_lists = [
        "//test/starlark_tests/resources:ExportAnotherFunctionShared.exp",
        "//test/starlark_tests/resources:ExportDontCallMeShared.exp",
    ],
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-x"],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "fmwk_with_provisioning",
    bundle_id = "com.google.example.framework",
    extension_safe = True,
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "fmwk_with_provisioning_importing_fmwk",
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":dynamic_fmwk_depending_lib",
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "second_fmwk_with_provisioning",
    bundle_id = "com.google.example.frameworktoo",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "fmwk_with_fmwk",
    bundle_id = "com.google.example.frameworkception",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
)

ios_framework(
    name = "fmwk_with_fmwk_with_provisioning",
    bundle_id = "com.google.example.frameworkception",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":fmwk_with_provisioning"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "fmwk_multiple_infoplists",
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
        "//test/starlark_tests/resources:Another.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "fmwk_with_dot.dynamic_framework",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "fmwk_not_extension_safe",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    extension_safe = False,
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for Apple dynamic framework import tests.

# Objective-C app with imported Objective-C dynamic framework.
ios_application(
    name = "app_with_imported_fmwk",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    ipa_post_processor = "//test/starlark_tests/targets_under_test/apple:ipa_post_processor_verify_codesigning",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":dynamic_fmwk_depending_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "dynamic_fmwk_depending_lib",
    srcs = [":dynamic_fmwk_depending_lib_file"],
    tags = common.fixture_tags,
    deps = [":iOSImportedDynamicFramework"],
)

write_file(
    name = "dynamic_fmwk_depending_lib_file",
    out = "DynamicFmwkDependingLib.m",
    content = ["""
#import <iOSDynamicFramework/iOSDynamicFramework.h>

int dynamic_fmwk_depending_lib_function() {
  SharedClass *sharedClass = [[SharedClass alloc] init];
  [sharedClass doSomethingShared];
  return 0;
}
"""],
    tags = common.fixture_tags,
)

# Swift app with imported Objective-C dynamic framework.
ios_application(
    name = "swift_app_with_imported_dynamic_fmwk",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":dynamic_fmwk_depending_swift_lib",
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

swift_library(
    name = "dynamic_fmwk_depending_swift_lib",
    srcs = [":dynamic_fmwk_depending_swift_lib_file"],
    tags = common.fixture_tags,
    deps = [":iOSImportedDynamicFramework"],
)

write_file(
    name = "dynamic_fmwk_depending_swift_lib_file",
    out = "DynamicFmwkDependingSwiftLib.swift",
    content = ["""
import iOSDynamicFramework

func dynamic_fmwk_depending_swift_lib_function() {
  let sharedClass = SharedClass()
  sharedClass.doSomethingShared()
}
"""],
    tags = common.fixture_tags,
)

ios_application(
    name = "app_with_imported_dynamic_fmwk_and_ext",
    bundle_id = "com.google.example",
    extensions = [":ext"],
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":iOSImportedDynamicFramework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

apple_dynamic_framework_import(
    name = "iOSImportedDynamicFramework",
    features = ["-swift.layering_check"],
    framework_imports = [":iOSDynamicFramework"],
    tags = common.fixture_tags,
)

generate_import_framework(
    name = "iOSDynamicFramework",
    archs = ["x86_64"],
    include_resource_bundle = True,
    libtype = "dynamic",
    minimum_os_version = "11.0",
    sdk = "iphonesimulator",
    tags = common.fixture_tags,
)

apple_dynamic_framework_import(
    name = "iOSImportedDynamicFrameworkWithBitcode",
    framework_imports = [":iOSDynamicFrameworkWithBitcode"],
)

generate_import_framework(
    name = "iOSDynamicFrameworkWithBitcode",
    archs = ["x86_64"],
    embed_bitcode = True,
    libtype = "dynamic",
    minimum_os_version = "11.0",
    sdk = "iphonesimulator",
)

apple_dynamic_framework_import(
    name = "iOSImportedDynamicFrameworkWithDebugInfo",
    framework_imports = [":iOSDynamicFrameworkWithDebugInfo"],
)

generate_import_framework(
    name = "iOSDynamicFrameworkWithDebugInfo",
    archs = ["x86_64"],
    embed_debug_info = True,
    libtype = "dynamic",
    minimum_os_version = "11.0",
    sdk = "iphonesimulator",
)

apple_dynamic_framework_import(
    name = "iOSImportedDynamicFrameworkWithDsym",
    dsym_imports = [":iOSDynamicFrameworkDsym"],
    framework_imports = [":iOSDynamicFramework"],
)

generate_import_framework_dsym(
    name = "iOSDynamicFrameworkDsym",
    framework_imports = [":iOSDynamicFramework"],
)

# --------------------------------------------------------------------------------------------------
# Targets for Apple static framework import tests.

# Objective-C app with imported Objective-C static framework.
ios_application(
    name = "app_with_imported_static_fmwk",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":static_fmwk_depending_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "static_fmwk_depending_lib",
    srcs = [":static_fmwk_depending_lib_file"],
    tags = common.fixture_tags,
    deps = [":iOSImportedStaticFramework"],
)

write_file(
    name = "static_fmwk_depending_lib_file",
    out = "StaticFmwkDependingLib.m",
    content = ["""
#import <iOSStaticFramework/iOSStaticFramework.h>

int static_fmwk_depending_lib_function() {
  SharedClass *sharedClass = [[SharedClass alloc] init];
  [sharedClass doSomethingShared];
  return 0;
}
"""],
    tags = common.fixture_tags,
)

# Objective-C app with imported Swift static framework.
ios_application(
    name = "app_with_imported_swift_static_fmwk",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":swift_static_fmwk_depending_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "swift_static_fmwk_depending_lib",
    srcs = [":swift_static_fmwk_depending_lib_file"],
    tags = common.fixture_tags,
    deps = [":iOSImportedSwiftStaticFramework"],
)

write_file(
    name = "swift_static_fmwk_depending_lib_file",
    out = "SwiftStaticFmwkDependingLib.m",
    content = ["""
#import <iOSSwiftStaticFramework/iOSSwiftStaticFramework.h>

int swift_static_fmwk_depending_lib_function() {
  SharedClass *sharedClass = [[SharedClass alloc] init];
  [sharedClass doSomethingShared];
  return 0;
}
"""],
    tags = common.fixture_tags,
)

# Objective-C app with imported Swift static framework without Swift module interface files.
ios_application(
    name = "app_with_imported_swift_static_fmwk_without_module_interface_files",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":swift_static_without_module_interfaces_fmwk_depending_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "swift_static_without_module_interfaces_fmwk_depending_lib",
    srcs = [":swift_static_without_module_interfaces_fmwk_depending_lib_src"],
    tags = common.fixture_tags,
    deps = [":iOSImportedSwiftStaticFrameworkWithoutModuleInterface"],
)

write_file(
    name = "swift_static_without_module_interfaces_fmwk_depending_lib_src",
    out = "SwiftStaticFmwkWithoutInterfaceFilesDependingLib.m",
    content = ["""
#import <iOSSwiftStaticFrameworkWithoutModuleInterfaces/iOSSwiftStaticFrameworkWithoutModuleInterfaces.h>

int swift_static_without_module_interfaces_fmwk_depending_lib_function() {
  SharedClass *sharedClass = [[SharedClass alloc] init];
  [sharedClass doSomethingShared];
  return 0;
}
"""],
    tags = common.fixture_tags,
)

# Swift app with imported Swift static framework.
ios_application(
    name = "swift_app_with_imported_swift_static_fmwk",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":swift_static_fmwk_depending_swift_lib",
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

swift_library(
    name = "swift_static_fmwk_depending_swift_lib",
    srcs = [":swift_static_fmwk_depending_swift_lib_file"],
    module_name = "swift_static_fmwk_depending_swift_lib",
    tags = common.fixture_tags,
    deps = [":iOSImportedSwiftStaticFramework"],
)

write_file(
    name = "swift_static_fmwk_depending_swift_lib_file",
    out = "SwiftStaticFmwkDependingSwiftLib.swift",
    content = ["""
import iOSSwiftStaticFramework

func swift_static_fmwk_depending_swift_lib_function() {
  let sharedClass = SharedClass()
  sharedClass.doSomethingShared()
}
"""],
    tags = common.fixture_tags,
)

apple_static_framework_import(
    name = "iOSImportedSwiftStaticFramework",
    features = ["-swift.layering_check"],
    framework_imports = [":iOSSwiftStaticFramework"],
    tags = common.fixture_tags,
)

apple_static_framework_import(
    name = "iOSImportedSwiftStaticFrameworkWithoutModuleInterface",
    features = ["-swift.layering_check"],
    framework_imports = [":iOSSwiftStaticFrameworkWithoutModuleInterfaces"],
    has_swift = True,
    tags = common.fixture_tags,
)

generate_import_framework(
    name = "iOSSwiftStaticFramework",
    archs = ["x86_64"],
    libtype = "static",
    minimum_os_version = "11.0",
    sdk = "iphonesimulator",
    swift_library = "//test/starlark_tests/resources/frameworks:swift_library",
    tags = common.fixture_tags,
)

generate_import_framework(
    name = "iOSSwiftStaticFrameworkWithoutModuleInterfaces",
    archs = ["x86_64"],
    include_module_interface_files = False,
    libtype = "static",
    minimum_os_version = "11.0",
    sdk = "iphonesimulator",
    swift_library = "//test/starlark_tests/resources/frameworks:swift_library",
    tags = common.fixture_tags,
)

# -----------------------------------------------------------------------------------------
# Targets for Apple dynamic XCFramework import tests.

# Objective-C app with imported Objective-C XCFramework.
ios_application(
    name = "app_with_imported_xcframework",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":dynamic_xcframework_depending_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "dynamic_xcframework_depending_lib",
    srcs = [":dynamic_xcframework_depending_lib_file"],
    tags = common.fixture_tags,
    deps = [":ios_imported_dynamic_xcframework"],
)

write_file(
    name = "dynamic_xcframework_depending_lib_file",
    out = "lib_with_imported_objc_xcframework.m",
    content = ["""
#import <generated_dynamic_xcframework_with_headers/generated_dynamic_xcframework_with_headers.h>

int dynamic_xcframework_depending_lib_function() {
  SharedClass *sharedClass = [[SharedClass alloc] init];
  [sharedClass doSomethingShared];
  return 0;
}
"""],
    tags = common.fixture_tags,
)

# Objective-C app with implementation deps imported Objective-C XCFramework.
ios_application(
    name = "app_with_implementation_deps_imported_xcframework",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":dynamic_xcframework_implemetation_deps_depending_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "dynamic_xcframework_implemetation_deps_depending_lib",
    srcs = [":dynamic_xcframework_depending_lib_file"],
    implementation_deps = [":ios_imported_dynamic_xcframework"],
    tags = common.fixture_tags,
)

# Swift app with imported Objective-C XCFramework.
ios_application(
    name = "swift_app_with_imported_objc_xcframework",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":dynamic_objc_xcframework_depending_swift_lib",
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

swift_library(
    name = "dynamic_objc_xcframework_depending_swift_lib",
    srcs = [":swift_with_framework_src"],
    tags = common.fixture_tags,
    deps = [":ios_imported_dynamic_xcframework"],
)

write_file(
    name = "swift_with_framework_src",
    out = "SwiftWithObjcFramework.swift",
    content = ["""
import generated_dynamic_xcframework_with_headers

func swift_with_framework_function() {
  let sc = generated_dynamic_xcframework_with_headers.SharedClass()
  sc.doSomethingShared()
}
"""],
    tags = common.fixture_tags,
)

apple_dynamic_xcframework_import(
    name = "ios_imported_dynamic_xcframework",
    features = ["-swift.layering_check"],
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    xcframework_imports = [":generated_dynamic_xcframework_with_headers"],
)

generate_dynamic_xcframework(
    name = "generated_dynamic_xcframework_with_headers",
    srcs = ["//test/starlark_tests/resources/frameworks:SharedClass.m"],
    hdrs = ["//test/starlark_tests/resources/frameworks:SharedClass.h"],
    minimum_os_versions = {
        "ios_simulator": common.min_os_ios.baseline,
        "ios": common.min_os_ios.baseline,
    },
    platforms = {
        "ios_simulator": [
            "x86_64",
            "arm64",
        ],
        "ios": [
            "arm64",
            "arm64e",
        ],
    },
    tags = common.fixture_tags,
)

# Objective-C app with imported Swift XCFramework.
ios_application(
    name = "app_with_imported_swift_xcframework",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":dynamic_swift_xcframework_depending_objc_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "dynamic_swift_xcframework_depending_objc_lib",
    srcs = [":objc_with_framework_src"],
    tags = common.fixture_tags,
    deps = [":ios_imported_swift_dynamic_xcframework"],
)

write_file(
    name = "objc_with_framework_src",
    out = "ObjcWithFramework.m",
    content = ["""
#import <Swift3PFmwkWithGenHeader/Swift3PFmwkWithGenHeader.h>

int objc_with_framework_function() {
  SharedClass *sharedClass = [[SharedClass alloc] init];
  [sharedClass doSomethingShared];
  return 0;
}
"""],
    tags = common.fixture_tags,
)

# Swift app with imported Swift XCFramework.
ios_application(
    name = "swift_app_with_imported_swift_xcframework",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":dynamic_swift_xcframework_depending_swift_lib",
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

swift_library(
    name = "dynamic_swift_xcframework_depending_swift_lib",
    srcs = [":swift_with_dynamic_swift_framework_src"],
    tags = common.fixture_tags,
    deps = [":ios_imported_swift_dynamic_xcframework"],
)

write_file(
    name = "swift_with_dynamic_swift_framework_src",
    out = "SwiftWithDynamicSwiftFramework.swift",
    content = ["""
import Swift3PFmwkWithGenHeader

func swift_with_dynamic_swift_framework_function() {
  let sc = Swift3PFmwkWithGenHeader.SharedClass()
  sc.doSomethingShared()
}
"""],
    tags = common.fixture_tags,
)

apple_dynamic_xcframework_import(
    name = "ios_imported_swift_dynamic_xcframework",
    features = ["-swift.layering_check"],
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    xcframework_imports = [":ios_swift_dynamic_xcframework"],
)

genrule(
    name = "ios_swift_dynamic_xcframework",
    srcs = ["//test/starlark_tests/targets_under_test/apple:ios_swift_3p_xcframework_with_generated_header"],
    outs = [
        "Swift3PFmwkWithGenHeader.xcframework/Info.plist",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64/Swift3PFmwkWithGenHeader.framework/Info.plist",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64/Swift3PFmwkWithGenHeader.framework/Swift3PFmwkWithGenHeader",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64/Swift3PFmwkWithGenHeader.framework/Headers/Swift3PFmwkWithGenHeader.h",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64/Swift3PFmwkWithGenHeader.framework/Modules/Swift3PFmwkWithGenHeader.swiftmodule/arm64.swiftdoc",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64/Swift3PFmwkWithGenHeader.framework/Modules/Swift3PFmwkWithGenHeader.swiftmodule/arm64.swiftinterface",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64/Swift3PFmwkWithGenHeader.framework/Modules/module.modulemap",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64_x86_64-simulator/Swift3PFmwkWithGenHeader.framework/Swift3PFmwkWithGenHeader",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64_x86_64-simulator/Swift3PFmwkWithGenHeader.framework/Headers/Swift3PFmwkWithGenHeader.h",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64_x86_64-simulator/Swift3PFmwkWithGenHeader.framework/Info.plist",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64_x86_64-simulator/Swift3PFmwkWithGenHeader.framework/Modules/Swift3PFmwkWithGenHeader.swiftmodule/arm64.swiftdoc",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64_x86_64-simulator/Swift3PFmwkWithGenHeader.framework/Modules/Swift3PFmwkWithGenHeader.swiftmodule/arm64.swiftinterface",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64_x86_64-simulator/Swift3PFmwkWithGenHeader.framework/Modules/Swift3PFmwkWithGenHeader.swiftmodule/x86_64.swiftdoc",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64_x86_64-simulator/Swift3PFmwkWithGenHeader.framework/Modules/Swift3PFmwkWithGenHeader.swiftmodule/x86_64.swiftinterface",
        "Swift3PFmwkWithGenHeader.xcframework/ios-arm64_x86_64-simulator/Swift3PFmwkWithGenHeader.framework/Modules/module.modulemap",
    ],
    cmd = "unzip -qq $(execpath //test/starlark_tests/targets_under_test/apple:ios_swift_3p_xcframework_with_generated_header) -d $(RULEDIR)",
    tags = common.fixture_tags,
)

# Swift app with imported Objective-C static framework.
ios_application(
    name = "swift_app_with_imported_static_fmwk",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":static_fmwk_depending_swift_lib",
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

swift_library(
    name = "static_fmwk_depending_swift_lib",
    srcs = [":static_fmwk_depending_swift_lib_file"],
    tags = common.fixture_tags,
    deps = [":iOSImportedStaticFramework"],
)

write_file(
    name = "static_fmwk_depending_swift_lib_file",
    out = "StaticFmwkDependingSwiftLib.swift",
    content = ["""
import iOSStaticFramework

func static_fmwk_depending_swift_lib_cuntion() {
  let sharedClass = SharedClass()
  sharedClass.doSomethingShared()
}
"""],
    tags = common.fixture_tags,
)

# Swift app with imported Swift static framework with data attribute
ios_application(
    name = "swift_app_with_imported_static_fmwk_with_data",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":iOSImportedStaticFrameworkWithData",
        ":iOSImportedStaticFrameworkWithDataAndBundle",
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

apple_static_framework_import(
    name = "iOSImportedStaticFrameworkWithData",
    data = ["//test/starlark_tests/resources:sample.png"],
    features = ["-swift.layering_check"],
    framework_imports = [":iOSSwiftStaticFramework"],
    tags = common.fixture_tags,
)

apple_static_framework_import(
    name = "iOSImportedStaticFrameworkWithDataAndBundle",
    data = ["//test/starlark_tests/resources:localized_xibs_ios"],
    features = ["-swift.layering_check"],
    framework_imports = [":iOSStaticFramework"],
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------
# Targets for Apple static XCFramework import tests.

# Objective-C app with imported Objective-C XCFramework
ios_application(
    name = "app_with_imported_xcframework_with_static_library",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":static_xcframework_depending_objc_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "static_xcframework_depending_objc_lib",
    srcs = [":objc_with_static_xcframework_src"],
    tags = common.fixture_tags,
    deps = [":ios_imported_static_xcframework"],
)

write_file(
    name = "objc_with_static_xcframework_src",
    out = "ObjcWithXCFrameworkStaticLibrary.m",
    content = ["""
#import <generated_static_xcframework_with_headers/generated_static_xcframework_with_headers.h>

int objc_with_static_xcframework_function() {
  SharedClass *sharedClass = [[SharedClass alloc] init];
  [sharedClass doSomethingShared];
  return 0;
}
"""],
    tags = common.fixture_tags,
)

# Objective-C app with imported Swift XCFramework
ios_application(
    name = "app_with_imported_swift_xcframework_with_static_library",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":swift_static_xcframework_depending_objc_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "swift_static_xcframework_depending_objc_lib",
    srcs = [":objc_with_swift_static_xcframework_src"],
    tags = common.fixture_tags,
    deps = [":ios_imported_swift_static_xcframework"],
)

write_file(
    name = "objc_with_swift_static_xcframework_src",
    out = "ObjcWithSwiftXCFrameworkStaticLibrary.m",
    content = ["""
#import <generated_swift_static_xcframework/generated_swift_static_xcframework.h>

int objc_with_swift_static_xcframework_function() {
  SharedClass *sharedClass = [[SharedClass alloc] init];
  [sharedClass doSomethingShared];
  return 0;
}
"""],
    tags = common.fixture_tags,
)

# Objective-C app with imported Swift XCFramework with no .swiftmodule files and has_swift enabled
ios_application(
    name = "app_with_imported_swift_xcframework_with_static_library_without_swiftmodule",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":swift_static_xcframework_without_swiftmodule_depending_objc_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "swift_static_xcframework_without_swiftmodule_depending_objc_lib",
    srcs = [":objc_with_swift_static_xcframework_without_swiftmodule_src"],
    tags = common.fixture_tags,
    deps = [":ios_imported_swift_static_xcframework_without_swiftmodule_and_has_swift_enabled"],
)

write_file(
    name = "objc_with_swift_static_xcframework_without_swiftmodule_src",
    out = "ObjcWithSwiftXCFrameworkStaticLibraryWithoutSwiftmodule.m",
    content = ["""
#import <generated_swift_static_xcframework_without_swiftmodule/generated_swift_static_xcframework_without_swiftmodule.h>

int objc_with_swift_static_xcframework_without_swiftmodule_function() {
  SharedClass *sharedClass = [[SharedClass alloc] init];
  [sharedClass doSomethingShared];
  return 0;
}
"""],
    tags = common.fixture_tags,
)

apple_static_xcframework_import(
    name = "ios_imported_swift_static_xcframework_without_swiftmodule_and_has_swift_enabled",
    has_swift = True,
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    xcframework_imports = [":generated_swift_static_xcframework_without_swiftmodule"],
)

generate_static_xcframework(
    name = "generated_swift_static_xcframework_without_swiftmodule",
    include_module_interface_files = False,
    platforms = {"ios_simulator": ["x86_64"]},
    swift_library = "//test/starlark_tests/resources/frameworks:swift_lib_for_static_xcframework",
    tags = common.fixture_tags,
)

# Swift app with imported Objective-C XCFramework
ios_application(
    name = "swift_app_with_imported_xcframework_with_static_library",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = ["//test/starlark_tests/resources:Info.plist"],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":static_xcframework_depending_swift_lib",
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

swift_library(
    name = "static_xcframework_depending_swift_lib",
    srcs = [":static_xcframework_depending_swift_lib_src"],
    tags = common.fixture_tags,
    deps = [":ios_imported_static_xcframework"],
)

write_file(
    name = "static_xcframework_depending_swift_lib_src",
    out = "SwiftWithStaticFramework.swift",
    content = ["""
import generated_static_xcframework_with_headers

func static_xcframework_depending_swift_lib_function_function() {
  let sc = generated_static_xcframework_with_headers.SharedClass()
  sc.doSomethingShared()
}
"""],
    tags = common.fixture_tags,
)

apple_static_xcframework_import(
    name = "ios_imported_static_xcframework",
    features = ["-swift.layering_check"],
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    xcframework_imports = [":generated_static_xcframework_with_headers"],
)

generate_static_xcframework(
    name = "generated_static_xcframework_with_headers",
    srcs = ["//test/starlark_tests/resources/frameworks:SharedClass.m"],
    hdrs = ["//test/starlark_tests/resources/frameworks:SharedClass.h"],
    minimum_os_versions = {
        "ios_simulator": common.min_os_ios.baseline,
        "ios": common.min_os_ios.baseline,
    },
    platforms = {
        "ios_simulator": [
            "x86_64",
            "arm64",
        ],
        "ios": [
            "arm64",
            "arm64e",
        ],
    },
    tags = common.fixture_tags,
)

# Swift app with imported Swift XCFramework
ios_application(
    name = "swift_app_with_imported_swift_xcframework_with_static_library",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = ["//test/starlark_tests/resources:Info.plist"],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":static_swift_xcframework_depending_swift_lib",
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

swift_library(
    name = "swift_multi_level_static_xcframework",
    srcs = [":swift_multi_level_static_xcframework_src"],
    tags = common.fixture_tags,
    deps = [
        ":ios_imported_static_xcframework_depends_on_base",
    ],
)

generate_static_framework_xcframework(
    name = "generated_static_xcframework_base",
    srcs = ["//test/starlark_tests/resources/frameworks:SharedClass.m"],
    hdrs = ["//test/starlark_tests/resources/frameworks:SharedClass.h"],
    minimum_os_versions = {"ios_simulator": "11.0"},
    platforms = {"ios_simulator": ["x86_64"]},
    tags = common.fixture_tags,
)

genrule(
    name = "generated_static_xcframework_depends_on_base_header",
    outs = ["generated_static_xcframework_depends_on_base_header.h"],
    cmd = "echo '@import generated_static_xcframework_base;' > $(OUTS)",
)

generate_static_framework_xcframework(
    name = "generated_static_xcframework_depends_on_base",
    srcs = ["@bazel_tools//tools/objc:objc_dummy.mm"],
    hdrs = ["generated_static_xcframework_depends_on_base_header"],
    minimum_os_versions = {"ios_simulator": "11.0"},
    platforms = {"ios_simulator": ["x86_64"]},
    tags = common.fixture_tags,
)

apple_static_xcframework_import(
    name = "ios_imported_static_xcframework_base",
    tags = common.fixture_tags,
    xcframework_imports = [":generated_static_xcframework_base"],
)

apple_static_xcframework_import(
    name = "ios_imported_static_xcframework_depends_on_base",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    xcframework_imports = [":generated_static_xcframework_depends_on_base"],
    deps = [":ios_imported_static_xcframework_base"],
)

swift_library(
    name = "static_swift_xcframework_depending_swift_lib",
    srcs = [":static_swift_xcframework_depending_swift_lib_src"],
    tags = common.fixture_tags,
    deps = [
        "ios_imported_static_xcframework_depends_on_base",
        "ios_imported_swift_static_xcframework",
    ],
)

write_file(
    name = "static_swift_xcframework_depending_swift_lib_src",
    out = "SwiftWithStaticSwiftFramework.swift",
    content = ["""
import generated_swift_static_xcframework

func static_swift_xcframework_depending_swift_lib_function() {
  let sc = generated_swift_static_xcframework.SharedClass()
  sc.doSomethingShared()
}
"""],
    tags = common.fixture_tags,
)

genrule(
    name = "swift_multi_level_static_xcframework_src",
    outs = ["SwiftMultiLevelStaticXCFramework.swift"],
    cmd = "echo 'import generated_static_xcframework_depends_on_base' > $(OUTS)",
)

ios_application(
    name = "app_with_swift_multi_level_static_xcframework",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = ["//test/starlark_tests/resources:Info.plist"],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":swift_multi_level_static_xcframework",
        "//test/starlark_tests/resources:swift_main_lib",
    ],
)

apple_static_xcframework_import(
    name = "ios_imported_swift_static_xcframework",
    features = [
        "-swift.layering_check",
        "apple._import_framework_via_swiftinterface",
    ],
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    xcframework_imports = [":generated_swift_static_xcframework"],
)

generate_static_xcframework(
    name = "generated_swift_static_xcframework",
    platforms = {"ios_simulator": ["x86_64"]},
    swift_library = "//test/starlark_tests/resources/frameworks:swift_lib_for_static_xcframework",
    tags = common.fixture_tags,
)

generate_static_xcframework(
    name = "generated_static_xcframework_with_headers_and_no_modulemap",
    srcs = ["//test/starlark_tests/resources/frameworks:SharedClass.m"],
    hdrs = ["//test/starlark_tests/resources/frameworks:SharedClass.h"],
    generate_modulemap = False,
    minimum_os_versions = {
        "ios_simulator": common.min_os_ios.baseline,
        "ios": common.min_os_ios.baseline,
    },
    platforms = {
        "ios_simulator": [
            "x86_64",
            "arm64",
        ],
        "ios": ["arm64"],
    },
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------

ios_imessage_application(
    name = "imessage_app",
    bundle_id = "com.google.example",
    extension = ":imessage_ext",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
)

ios_imessage_extension(
    name = "imessage_ext",
    app_icons = ["//test/testdata/resources:message_ext_app_icon_ios"],
    bundle_id = "com.google.example.MessagesExtension",
    bundle_name = "imessage-ext",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:MessagesExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_imessage_extension(
    name = "imessage_ext_with_wrong_appicons",
    app_icons = ["//test/testdata/resources:app_icons_ios"],
    bundle_id = "com.google.example.MessagesExtension",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:MessagesExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_application(
    name = "app_with_imessage_ext_with_imported_dynamic_xcfmwk",
    bundle_id = "com.google.example",
    extensions = [":imessage_ext_imported_dynamic_xcfmwk"],
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_imessage_extension(
    name = "imessage_ext_imported_dynamic_xcfmwk",
    app_icons = ["//test/testdata/resources:message_ext_app_icon_ios"],
    bundle_id = "com.google.example.MessagesExtension",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = ["//test/starlark_tests/resources:MessagesExtensionInfo.plist"],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = ["//test/starlark_tests/resources:swift_lib_importing_imported_dynamic_xcfw"],
)

ios_sticker_pack_extension(
    name = "sticker_ext",
    bundle_id = "com.google.example.stickerext",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:StickerPackInfo.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    sticker_assets = [],
    tags = common.fixture_tags,
)

ios_sticker_pack_extension(
    name = "sticker_ext_with_sticker_assets",
    bundle_id = "com.google.example.stickerext",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:StickerPackInfo.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    sticker_assets = ["//test/starlark_tests/resources:sticker_pack_ios"],
    tags = common.fixture_tags,
)

ios_sticker_pack_extension(
    name = "sticker_ext_with_wrong_appicons",
    bundle_id = "com.google.example.stickerext",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:StickerPackInfo.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    sticker_assets = ["//test/testdata/resources:app_icons_ios"],
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------

ios_ui_test(
    name = "ui_test",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_ui_test(
    name = "ui_test_custom_bundle_id",
    bundle_id = "my.test.bundle.id",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_ui_test(
    name = "ui_test_invalid_bundle_id",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_ui_test(
    name = "ui_test_multiple_infoplists",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
        "//test/starlark_tests/resources:Another.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_ui_test(
    name = "ui_test_with_fmwk",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [
        ":fmwk",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_ui_test(
    name = "ui_test_dummy_runner",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    runner = "//test/starlark_tests/targets_under_test/apple:dummy_test_runner",
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

xctrunner(
    name = "ui_test_xctrunner_app",
    testonly = True,
    tests = [":ui_test"],
)

xctrunner(
    name = "ui_test_xctrunner_app_multiple_targets",
    testonly = True,
    tests = [
        ":ui_test",
        ":ui_test_with_fmwk",
    ],
)

# ---------------------------------------------------------------------------------------

ios_unit_test(
    name = "unit_test",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_unit_test(
    name = "unit_test_no_bundle_loader",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    tags = common.fixture_tags,
    test_host = ":app",
    test_host_is_bundle_loader = False,
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_unit_test(
    name = "unit_test_custom_bundle_id",
    bundle_id = "my.test.bundle.id",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_unit_test(
    name = "unit_test_custom_bundle_name",
    bundle_name = "UnitTestCustomBundleName",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_unit_test(
    name = "unit_test_invalid_bundle_id",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_unit_test(
    name = "unit_test_multiple_infoplists",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
        "//test/starlark_tests/resources:Another.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_unit_test(
    name = "unit_test_no_host",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.arm_sim_support,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_unit_test(
    name = "unit_test_with_fmwk",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [
        ":fmwk",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_unit_test(
    name = "unit_test_with_imported_fmwk",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":dynamic_fmwk_depending_lib",
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_unit_test(
    name = "unit_test_with_fmwk_from_objc_library_data",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_ios_framework",
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_unit_test(
    name = "unit_test_with_host_importing_same_fmwk",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    test_host = ":app_with_imported_fmwk",
    deps = [
        ":dynamic_fmwk_depending_lib",
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_unit_test(
    name = "unit_test_dummy_runner",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    runner = "//test/starlark_tests/targets_under_test/apple:dummy_test_runner",
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_unit_test(
    name = "unit_test_with_swift_deps",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.arm_sim_support,
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_swift_depending_test_lib",
    ],
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

ios_framework(
    name = "dedupe_test_fmwk",
    bundle_id = "com.fmwk",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":shared_lib",
    ],
)

ios_application(
    name = "dedupe_test_app",
    bundle_id = "com.app",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [
        ":dedupe_test_fmwk",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":main_lib",
    ],
)

ios_unit_test(
    name = "dedupe_test_test",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    test_host = ":dedupe_test_app",
    deps = [
        ":test_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for iOS dynamic framework tests.

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

ios_dynamic_framework(
    name = "basic_framework",
    bundle_id = "com.google.example.framework",
    bundle_name = "BasicFramework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
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

ios_dynamic_framework(
    name = "basic_framework_with_direct_dependency",
    bundle_id = "com.google.example.framework",
    bundle_name = "DirectDependencyTest",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
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

ios_dynamic_framework(
    name = "basic_framework_with_transitive_dependency",
    bundle_id = "com.google.example.framework",
    bundle_name = "TransitiveDependencyTest",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_transitive_dependency_lib",
    ],
)

swift_library(
    name = "basic_framework_with_dynamic_framework_dependency_lib",
    srcs = [
        "//test/starlark_tests/resources:DirectDependencyTest.swift",
    ],
    features = ["swift.no_generated_module_map"],
    module_name = "DirectDependencyWithDynamicFrameworkTest",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    deps = [
        ":basic_framework",
    ],
)

ios_dynamic_framework(
    name = "basic_framework_with_dynamic_framework_dependency",
    bundle_id = "com.google.example.framework",
    bundle_name = "DirectDependencyWithDynamicFrameworkTest",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_dynamic_framework_dependency_lib",
    ],
)

swift_library(
    name = "basic_framework_with_transitive_dependency_with_dynamic_frameworks_lib",
    srcs = [
        "//test/starlark_tests/resources:TransitiveDependencyTest.swift",
    ],
    features = ["swift.no_generated_module_map"],
    module_name = "TransitiveDependencyWithDynamicFrameworksTest",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    deps = [
        ":basic_framework",
        ":basic_framework_with_direct_dependency",
    ],
)

ios_dynamic_framework(
    name = "basic_framework_with_transitive_dependency_with_dynamic_frameworks",
    bundle_id = "com.google.example.framework",
    bundle_name = "TransitiveDependencyWithDynamicFrameworksTest",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_transitive_dependency_with_dynamic_frameworks_lib",
    ],
)

ios_dynamic_framework(
    name = "dynamic_framework_with_resources",
    bundle_id = "com.google.example.framework",
    bundle_name = "swift_lib_with_resources",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    linkopts = ["-application_extension"],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_lib_with_structured_resources",
    ],
)

ios_application(
    name = "app_with_dynamic_framework_and_resources",
    bundle_id = "com.google.example",
    families = ["iphone"],
    frameworks = [
        ":dynamic_framework_with_resources",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:swift_lib_with_resources",
    ],
)

ios_dynamic_framework(
    name = "swift_common_lib_framework",
    bundle_id = "com.google.example.framework",
    bundle_name = "swift_common_lib",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = ["//test/starlark_tests/resources:swift_common_lib"],
)

ios_dynamic_framework(
    name = "swift_shared_lib_framework",
    bundle_id = "com.google.example.framework",
    bundle_name = "swift_shared_lib",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_shared_lib",
    ],
)

ios_dynamic_framework(
    name = "swift_transitive_lib_framework",
    bundle_id = "com.google.example.framework",
    bundle_name = "swift_transitive_lib",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":swift_shared_lib_framework"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = ["//test/starlark_tests/resources:swift_transitive_lib"],
)

ios_application(
    name = "app_with_dynamic_framework_with_dynamic_framework",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    frameworks = [":swift_transitive_lib_framework"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:ios_non_localized_assets_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_dynamic_framework(
    name = "dynamic_fmwk_with_multiple_dependencies",
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_lib",
        ":basic_framework_with_direct_dependency_lib",
    ],
)

ios_dynamic_framework(
    name = "dynamic_fmwk_with_objc_library",
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_shared_lib_with_resources",
    ],
)

ios_static_framework(
    name = "objc_static_framework",
    hdrs = [
        "//test/starlark_tests/resources:common.h",
    ],
    bundle_name = "objc_static_framework",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_common_lib",
    ],
)

ios_static_framework(
    name = "objc_static_framework_without_modulemap",
    bundle_name = "objc_static_framework_without_modulemap",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_common_lib",
    ],
)

ios_static_framework(
    name = "objc_static_framework_with_sdk_dylibs_dep",
    bundle_name = "objc_static_framework_with_sdk_dylibs_dep",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_lib_with_sdk_dylibs",
    ],
)

ios_static_framework(
    name = "objc_static_framework_with_sdk_fmwks_dep",
    bundle_name = "objc_static_framework_with_sdk_fmwks_dep",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_lib_with_sdk_fmwks",
    ],
)

swift_library(
    name = "basic_framework_with_dynamic_framework_import_lib",
    srcs = [
        "//test/starlark_tests/resources:BasicFramework.swift",
    ],
    features = ["swift.no_generated_module_map"],
    module_name = "DynamicFrameworkImportTest",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    deps = [
        ":iOSImportedDynamicFramework",
    ],
)

ios_dynamic_framework(
    name = "basic_framework_with_dynamic_framework_import",
    bundle_id = "com.google.example.framework",
    bundle_name = "DynamicFrameworkImportTest",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_dynamic_framework_import_lib",
    ],
)

swift_library(
    name = "basic_framework_with_static_framework_import_lib",
    srcs = [
        "//test/starlark_tests/resources:BasicFramework.swift",
    ],
    features = ["swift.no_generated_module_map"],
    module_name = "StaticFrameworkImportTest",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
    deps = [
        ":iOSImportedStaticFramework",
    ],
)

ios_dynamic_framework(
    name = "basic_framework_with_static_framework_import",
    bundle_id = "com.google.example.framework",
    bundle_name = "StaticFrameworkImportTest",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_with_static_framework_import_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for iOS static framework tests.

ios_static_framework(
    name = "swift_ios_static_framework",
    bundle_name = "SwiftFmwk",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [":SwiftFmwk"],
)

ios_static_framework(
    name = "static_fmwk_with_swift_and_avoid_deps",
    avoid_deps = [":SwiftFmwkLowerLib"],
    bundle_name = "SwiftFmwkUpperLib",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [":SwiftFmwkUpperLib"],
)

ios_static_framework(
    name = "static_framework_with_generated_header",
    bundle_name = "SwiftStaticFmwkWithGenHeader",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [":SwiftStaticFmwkWithGenHeader"],
)

ios_static_framework(
    name = "static_framework_from_objc_library",
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [":objc_library_with_sdk_dylib_and_framework"],
)

ios_static_framework(
    name = "static_framework_with_header_and_exclude_resources",
    hdrs = [
        "//test/starlark_tests/resources:shared.h",
    ],
    avoid_deps = [":framework_dependent_lib_with_resource"],
    exclude_resources = True,
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [":static_framework_lib_with_resources"],
)

ios_static_framework(
    name = "static_framework_with_no_header_and_exclude_resources",
    avoid_deps = [":framework_dependent_lib_with_resource"],
    exclude_resources = True,
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [":static_framework_lib_with_resources"],
)

ios_static_framework(
    name = "static_framework_with_transitive_resources",
    hdrs = [
        "//test/starlark_tests/resources:shared.h",
    ],
    avoid_deps = [":framework_dependent_lib_with_resource"],
    exclude_resources = False,
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [":static_framework_lib_with_resources"],
)

write_file(
    name = "dummy_swift",
    out = "Dummy.swift",
    content = ["public struct Dummy {}"],
    tags = common.fixture_tags,
)

swift_library(
    name = "SwiftFmwk",
    srcs = [":dummy_swift"],
    module_name = "SwiftFmwk",
    tags = common.fixture_tags,
)

swift_library(
    name = "SwiftFmwkUpperLib",
    srcs = [":dummy_swift"],
    module_name = "SwiftFmwkUpperLib",
    tags = common.fixture_tags,
    deps = [":SwiftFmwkLowerLib"],
)

swift_library(
    name = "SwiftFmwkLowerLib",
    srcs = [":dummy_swift"],
    module_name = "SwiftFmwkLowerLib",
    tags = common.fixture_tags,
    deps = [":SwiftFmwkLowestLib"],
)

swift_library(
    name = "SwiftFmwkLowestLib",
    srcs = [":dummy_swift"],
    module_name = "SwiftFmwkLowestLib",
    tags = common.fixture_tags,
)

swift_library(
    name = "SwiftStaticFmwkWithGenHeader",
    srcs = [":dummy_swift"],
    generates_header = True,
    module_name = "SwiftStaticFmwkWithGenHeader",
    tags = common.fixture_tags,
)

apple_dynamic_framework_import(
    name = "bundle_only_imported_fmwk",
    bundle_only = True,
    framework_imports = [":generated_ios_dynamic_fmwk"],
    tags = common.fixture_tags,
)

generate_import_framework(
    name = "generated_ios_dynamic_fmwk",
    archs = ["x86_64"],
    libtype = "dynamic",
    minimum_os_version = common.min_os_ios.baseline,
    sdk = "iphonesimulator",
    tags = common.fixture_tags,
)

objc_library(
    name = "objc_library_with_sdk_dylib_and_framework",
    linkopts = ["-lc++"],
    sdk_dylibs = ["sqlite3"],
    sdk_frameworks = ["UIKit"],
    tags = common.fixture_tags,
    deps = [
        ":fmwk_lib",
    ],
)

objc_library(
    name = "static_framework_lib_with_resources",
    srcs = [
        "//test/starlark_tests/resources:shared.h",
        "//test/starlark_tests/resources:shared.m",
    ],
    data = [
        "//test/testdata/resources:assets",
        "//test/testdata/resources:nonlocalized.strings",
        "//test/testdata/resources:storyboard_ios.storyboard",
        "//test/testdata/resources:unversioned_datamodel",
        "//test/testdata/resources:versioned_datamodel",
        "//test/testdata/resources:view_ios.xib",
    ],
    tags = common.fixture_tags,
    deps = [":framework_dependent_lib_with_resource"],
    alwayslink = True,
)

objc_library(
    name = "framework_dependent_lib_with_resource",
    data = [
        "//test/testdata/resources:view_ios.xib",
    ],
    tags = common.fixture_tags,
)

# ---------------------------------------------------------------------------------------
# Targets for Apple Frameworks propagated from objc_library/swift_library data attribute tests.

ios_application(
    name = "app_with_fmwks_from_objc_swift_libraries_using_data",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_ios_framework",
        ":swift_lib_with_ios_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_fmwks_from_transitive_objc_swift_libraries_using_data",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_inner_lib_with_ios_framework",
        ":swift_lib_with_ios_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_fmwks_from_frameworks_and_objc_swift_libraries_using_data",
    bundle_id = "com.google.example",
    families = ["iphone"],
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_ios_framework",
        ":swift_lib_with_ios_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

objc_library(
    name = "objc_lib_with_inner_lib_with_ios_framework",
    srcs = [":objc_lib_with_inner_lib_with_ios_framework_file"],
    tags = common.fixture_tags,
    deps = [":objc_lib_with_ios_framework"],
)

write_file(
    name = "objc_lib_with_inner_lib_with_ios_framework_file",
    out = "objc_lib_with_inner_lib_with_ios_framework_file.m",
    content = ["""
int objc_lib_with_inner_lib_with_ios_framework_function() {
  return 0;
}
"""],
    tags = common.fixture_tags,
)

objc_library(
    name = "objc_lib_with_ios_framework",
    srcs = [
        "//test/starlark_tests/resources:shared.h",
        "//test/starlark_tests/resources:shared.m",
    ],
    data = [
        ":fmwk_with_resources",
        ":resource_group_with_framework",
    ],
    tags = common.fixture_tags,
)

apple_resource_group(
    name = "resource_group_with_framework",
    resources = [":fmwk_no_version"],
    tags = common.fixture_tags,
)

swift_library(
    name = "swift_lib_with_ios_framework",
    srcs = ["//test/starlark_tests/resources:DummySwiftFile"],
    data = [":fmwk_min_os_baseline_with_bundle"],
    tags = common.fixture_tags,
)

ios_application(
    name = "app_with_resources_and_fmwks_with_resources_from_objc_swift_libraries_using_data",
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_ios_framework",
        ":swift_lib_with_ios_framework",
        "//test/starlark_tests/resources:another_basic_bundle_lib",
        "//test/starlark_tests/resources:objc_lib_with_resources",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_fmwk_and_ext_with_objc_lib_with_nested_ios_framework",
    bundle_id = "com.google.example",
    extensions = [":ext_with_objc_lib_with_nested_ios_framework"],
    families = ["iphone"],
    frameworks = [":fmwk"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_nested_ios_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_extension(
    name = "ext_with_objc_lib_with_nested_ios_framework",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_nested_ios_framework",
    ],
)

objc_library(
    name = "objc_lib_with_nested_ios_framework",
    srcs = [
        "//test/starlark_tests/resources:shared.h",
        "//test/starlark_tests/resources:shared.m",
    ],
    data = [":fmwk_with_fmwk"],
    tags = common.fixture_tags,
)

ios_extension(
    name = "ext_with_fmwks_from_objc_swift_libraries_using_data",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_ios_framework",
        ":swift_lib_with_ios_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_framework(
    name = "fmwk_with_fmwks_from_objc_swift_libraries_using_data",
    hdrs = ["//test/starlark_tests/resources:common.h"],
    bundle_id = "com.google.example.framework",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_ios_framework",
        ":swift_lib_with_ios_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_app_clip(
    name = "app_clip_with_fmwks_from_objc_swift_libraries_using_data",
    bundle_id = "com.google.example.clip",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.appclip_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_ios_framework",
        ":swift_lib_with_ios_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_app_clip(
    name = "app_clip_with_ext",
    bundle_id = "com.google.example.clip",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    extensions = [":swift_ios_app_clip_ext"],
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.appclip_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        ":objc_lib_with_ios_framework",
        ":swift_lib_with_ios_framework",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for importing XCFramework tests.

ios_application(
    name = "app_with_imported_dynamic_xcfmwk",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:swift_lib_importing_imported_dynamic_xcfw",
    ],
)

ios_application(
    name = "app_with_imported_dynamic_xcfmwk_with_lib_ids",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:swift_lib_importing_imported_dynamic_xcfw",
    ],
)

ios_application(
    name = "app_with_ext_with_imported_dynamic_xcfmwk",
    bundle_id = "com.google.example",
    extensions = [":ext_with_imported_dynamic_xcfmwk"],
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_extension(
    name = "ext_with_imported_dynamic_xcfmwk",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_common_lib",
        "//test/starlark_tests/resources:swift_lib_importing_imported_dynamic_xcfw",
    ],
)

ios_application(
    name = "app_with_imported_static_xcfmwk",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_lib_importing_imported_static_xcfw",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_imported_static_xcfmwk_with_lib_ids",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_lib_importing_imported_static_xcfw",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_imported_xcfmwk_bundling_static_fmwks",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:swift_lib_importing_imported_xcfw_bundling_static_fmwks",
    ],
)

ios_application(
    name = "app_with_imported_xcfmwk_bundling_static_fmwks_with_resources",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework_with_resources_import",
    ],
)

ios_application(
    name = "app_with_imported_xcfmwk_bundling_resources_and_data",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/targets_under_test/apple:ios_static_xcframework_with_resources_and_data",
    ],
)

ios_application(
    name = "app_with_imported_static_xcfmwk_with_module_map",
    bundle_id = "com.google.example",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
        "//test/starlark_tests/resources:swift_lib_importing_imported_static_xcfw",
    ],
)

# ---------------------------------------------------------------------------------------

ios_extension(
    name = "ext_for_bundle_loader",
    bundle_id = "com.google.example.ext",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_common_lib",
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_unit_test(
    name = "unit_test_with_bundle_loader",
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    test_host = ":ext_for_bundle_loader",
    deps = [
        "//test/starlark_tests/resources:objc_common_test_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for AppIntents

ios_application(
    name = "app_with_app_intents",
    app_intents = [
        "//test/starlark_tests/resources:app_intent",
    ],
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.app_intents_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:app_intent",
        "//test/starlark_tests/resources:swift_uikit_appdelegate",
    ],
)

ios_application(
    name = "app_with_widget_configuration_intent",
    app_intents = [
        "//test/starlark_tests/resources:widget_configuration_intent",
    ],
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.widget_configuration_intents_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:swift_uikit_appdelegate",
        "//test/starlark_tests/resources:widget_configuration_intent",
    ],
)

ios_application(
    name = "app_with_app_intent_and_widget_configuration_intent",
    app_intents = [
        "//test/starlark_tests/resources:app_intent",
        "//test/starlark_tests/resources:widget_configuration_intent",
    ],
    bundle_id = "com.google.example",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.widget_configuration_intents_support,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:app_intent",
        "//test/starlark_tests/resources:swift_uikit_appdelegate",
        "//test/starlark_tests/resources:widget_configuration_intent",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for base_bundle_id and bundle_id flows with and without shared capabilities.

ios_application(
    name = "app_with_no_bundle_id",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_empty_bundle_id",
    bundle_id = "",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_just_dot_bundle_id",
    bundle_id = ".",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_leading_dot_bundle_id",
    bundle_id = ".my.bundle.id",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_trailing_dot_bundle_id",
    bundle_id = "my.bundle.id.",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_double_dot_bundle_id",
    bundle_id = "my..bundle.id",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_invalid_character_bundle_id",
    bundle_id = "my#bundle",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_capability_set_derived_bundle_id",
    bundle_name = "app-with-capability-set-derived-bundle-id",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_custom_bundle_id_suffix_derived_bundle_id",
    bundle_id_suffix = "bundle-id-suffix",
    bundle_name = "app-with-capability-set-derived-bundle-id",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_ambiguous_shared_capabilities_bundle_id",
    bundle_id = "com.bazel.app.example.bundle-id",
    bundle_name = "app-with-ambiguous-shared-capabilities-bundle-id",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_absent_shared_capabilities_bundle_id",
    bundle_name = "app-with-absent-shared-capabilities-bundle-id",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:empty_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_conflicting_shared_capabilities_bundle_id",
    bundle_name = "app-with-conflicting-shared-capabilities-bundle-id",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:alt_capability_set",
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_extension(
    name = "ext_with_capability_set_derived_bundle_id",
    bundle_name = "ext-with-capability-set-derived-bundle-id",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_framework(
    name = "fmwk_with_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_name = "fmwk-with-base-bundle-id-derived-bundle-id",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_framework(
    name = "fmwk_with_ambiguous_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_id = "com.bazel.app.example.bundle-id",
    bundle_name = "fmwk-with-base-bundle-id-derived-bundle-id",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:framework_resources_lib",
        "//test/starlark_tests/resources:objc_shared_lib",
    ],
)

ios_imessage_application(
    name = "imessage_app_with_capability_set_derived_bundle_id",
    bundle_name = "imessage-app-with-capability-set-derived-bundle-id",
    extension = ":imessage_ext_with_capability_set_derived_bundle_id",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
)

ios_imessage_extension(
    name = "imessage_ext_with_capability_set_derived_bundle_id",
    bundle_name = "MessagesExtension",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:MessagesExtensionInfo.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_sticker_pack_extension(
    name = "sticker_ext_with_capability_set_derived_bundle_id",
    bundle_name = "sticker-ext-with-capability-set-derived-bundle-id",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:StickerPackInfo.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    shared_capabilities = [
        "//test/starlark_tests/targets_under_test/apple/capabilities:base_capability_set",
    ],
    sticker_assets = [],
    tags = common.fixture_tags,
)

ios_ui_test(
    name = "ui_test_with_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_id_suffix = "ui-test-with-base-bundle-id-derived-bundle-id",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_unit_test(
    name = "unit_test_with_base_bundle_id_derived_bundle_id",
    base_bundle_id = "//test/starlark_tests/targets_under_test/apple/capabilities:base_bundle_id",
    bundle_id_suffix = "unit-test-with-base-bundle-id-derived-bundle-id",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    resources = [
        "//test/starlark_tests/resources:example_filegroup",
        "//test/starlark_tests/resources:resource_bundle",
    ],
    tags = common.fixture_tags,
    test_host = ":app",
    deps = [
        "//test/starlark_tests/resources:objc_test_lib",
    ],
)

ios_application(
    name = "app_with_duplicate_bundle",
    bundle_id = "com.google.example",
    bundle_name = "app_with_duplicate_bundle",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

ios_application(
    name = "app_with_duplicate_bundle_beta",
    bundle_id = "com.google.example",
    bundle_name = "app_with_duplicate_bundle",
    families = ["iphone"],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)

# ---------------------------------------------------------------------------------------
# Targets for testing DocC support.

swift_library(
    name = "basic_framework_lib_with_docc_bundle",
    srcs = ["//test/starlark_tests/resources:BasicFramework.swift"],
    data = ["//test/starlark_tests/resources:basic_docc_bundle_files"],
    module_name = "BasicFrameworkWithDocCBundle",
    tags = common.fixture_tags,
    visibility = ["//visibility:public"],
)

objc_library(
    name = "basic_objc_lib_with_data_and_docc_bundle_dependency",
    srcs = [
        "//test/starlark_tests/resources:shared.h",
        "//test/starlark_tests/resources:shared.m",
    ],
    data = ["//test/starlark_tests/resources:additional.txt"],
    deps = [":basic_framework_lib_with_docc_bundle"],
)

docc_archive(
    name = "basic_objc_lib_with_data_and_docc_bundle_dependency.doccarchive",
    dep = ":basic_objc_lib_with_data_and_docc_bundle_dependency",
    fallback_bundle_identifier = "com.google.example.objc.lib",
    fallback_bundle_version = "1.0",
    fallback_display_name = "BasicLib",
)

ios_dynamic_framework(
    name = "basic_framework_with_docc_bundle",
    bundle_id = "com.google.example.framework",
    bundle_name = "BasicFrameworkWithDocCBundle",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    tags = common.fixture_tags,
    deps = [
        ":basic_framework_lib_with_docc_bundle",
    ],
)

docc_archive(
    name = "basic_framework_with_docc_bundle.doccarchive",
    dep = ":basic_framework_with_docc_bundle",
    fallback_bundle_identifier = "com.google.example.framework",
    fallback_bundle_version = "1.0",
    fallback_display_name = "BasicFramework",
)

docc_archive(
    name = "basic_framework_with_docc_bundle_custom_base_path.doccarchive",
    dep = ":basic_framework_with_docc_bundle",
    fallback_bundle_identifier = "com.google.example.framework",
    fallback_bundle_version = "1.0",
    fallback_display_name = "BasicFramework",
    hosting_base_path = "custom/base/path",
)

docc_archive(
    name = "basic_framework_with_transitive_dependency.doccarchive",
    dep = ":basic_framework_with_transitive_dependency",
    fallback_bundle_identifier = "com.google.example.framework",
    fallback_bundle_version = "1.0",
    fallback_display_name = "BasicFramework",
)

# ---------------------------------------------------------------------------------------
# Target for extension resource bundling.

ios_extension(
    name = "ext_with_resources",
    bundle_id = "com.google.example.ext_with_resources",
    entitlements = "//test/starlark_tests/resources:entitlements.plist",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//test/starlark_tests/resources:Info.plist",
    ],
    minimum_os_version = common.min_os_ios.baseline,
    provisioning_profile = "//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    resources = [
        "//test/starlark_tests/resources:additional.txt",
    ],
    tags = common.fixture_tags,
    deps = [
        "//test/starlark_tests/resources:objc_main_lib",
    ],
)
